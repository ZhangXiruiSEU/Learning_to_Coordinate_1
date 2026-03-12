import argparse
import json
import os
import time
import sys
from pathlib import Path


def _early_set_backend():
    backend = None
    import sys

    for i, arg in enumerate(sys.argv):
        if arg.startswith("--backend="):
            backend = arg.split("=", 1)[1]
            break
        if arg == "--backend" and i + 1 < len(sys.argv):
            backend = sys.argv[i + 1]
            break
    if backend in ("cpu", "cuda"):
        os.environ["JAX_PLATFORMS"] = backend


_early_set_backend()

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = ROOT / "archive"
for _path in (ROOT, ARCHIVE_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import BFGS, GradientDescent, LBFGS, NonlinearCG

from diagnose_subp2_step import build_original_step_nlp, summarize_ipopt_stats
from scan_subp2_snapshot import (
    build_context,
    build_original_step_reference,
    build_parser as build_snapshot_parser,
    build_step_problem,
    capture_subp2_snapshot,
    pack_original_para2,
    primal_feas_metrics,
)


def build_parser():
    p = argparse.ArgumentParser(
        description="Benchmark the postdoc barrier/continuation reformulation on one real SubP2 step.",
        parents=[build_snapshot_parser()],
        conflict_handler="resolve",
    )
    p.add_argument("--target-admm-iter", type=int, default=1)
    p.add_argument("--target-step", type=int, default=0)
    p.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--mu-schedule", type=str, default="1e-1,3e-2,1e-2,3e-3")
    p.add_argument("--lbfgs-maxiter", type=int, default=60)
    p.add_argument("--lbfgs-tol", type=float, default=1e-5)
    p.add_argument(
        "--optimizer",
        choices=["lbfgs", "bfgs", "nonlinear_cg", "gd"],
        default="lbfgs",
    )
    p.add_argument("--history-size", type=int, default=10)
    p.add_argument("--maxls", type=int, default=20)
    p.add_argument("--linesearch", type=str, default="zoom")
    p.add_argument("--eq-penalty-scale", type=float, default=100.0)
    p.add_argument("--eq-rho", type=float, default=100.0)
    p.add_argument("--eq-rho-schedule", type=str, default="")
    p.add_argument(
        "--eq-rho-coupled-to-mu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, use eq_penalty_scale / mu as before; otherwise use a decoupled ALM rho schedule.",
    )
    p.add_argument("--ineq-penalty-scale", type=float, default=100.0)
    p.add_argument("--barrier-scale", type=float, default=1.0)
    p.add_argument("--barrier-eps", type=float, default=1e-6)
    p.add_argument("--barrier-min-slack", type=float, default=1e-8)
    p.add_argument("--eq-lam-update", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eq-lam-update-scale", type=float, default=0.25)
    p.add_argument("--eq-lam-clip", type=float, default=1e3)
    p.add_argument("--prox-weight", type=float, default=0.0)
    p.add_argument("--use-scales", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--project-unit-equalities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Project quaternion/cable-direction unit-norm equalities instead of penalizing them.",
    )
    p.add_argument(
        "--wrench-only-equality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only keep wrench-consensus equalities in the augmented equality term.",
    )
    p.add_argument(
        "--project-wrench-equality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Project load control to satisfy wrench-consensus equality exactly.",
    )
    p.add_argument(
        "--eliminate-ul",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optimize in reduced space by eliminating load control and reconstructing it via wrench projection.",
    )
    p.add_argument("--projection-eps", type=float, default=1e-9)
    p.add_argument("--bench-repeats", type=int, default=3)
    p.add_argument("--json-out", default="")
    return p


def parse_mu_schedule(text):
    return jnp.array([float(x.strip()) for x in text.split(",") if x.strip()], dtype=jnp.float32)


def parse_optional_schedule(text, fallback_value, length):
    if not text.strip():
        return jnp.full((length,), float(fallback_value), dtype=jnp.float32)
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != length:
        raise ValueError(f"schedule length must be {length}, got {len(vals)}")
    return jnp.array(vals, dtype=jnp.float32)


def solve_postdoc_variant(planner, dims, x_init, params_t, x_orig, args):
    nxl, nul, nxi, nui, nq = [int(v) for v in dims[:5]]
    quat_slice = slice(6, 10)

    if args.eliminate_ul and not args.project_wrench_equality:
        raise ValueError("--eliminate-ul requires --project-wrench-equality")

    def _project_full_x(x):
        xl = x[:nxl]
        ul = x[nxl:nxl + nul]
        rem = x[nxl + nul:].reshape(nq, nxi + nui)
        xc = rem[:, :nxi]
        uc = rem[:, nxi:]

        if args.project_unit_equalities:
            ql = xl[quat_slice]
            ql = ql / jnp.maximum(jnp.linalg.norm(ql), args.projection_eps)
            xl = xl.at[quat_slice].set(ql)

            di = xc[:, 0:3]
            di_norm = jnp.maximum(jnp.linalg.norm(di, axis=1, keepdims=True), args.projection_eps)
            xc = xc.at[:, 0:3].set(di / di_norm)

        if args.project_wrench_equality and float(params_t["is_terminal"]) < 0.5:
            ql = xl[quat_slice]
            Rl = planner._q_2_rotation_jax(ql)
            di_vecs = xc[:, 0:3]
            ti_mags = xc[:, 12]
            fi_inertial = di_vecs * ti_mags[:, None]
            fi_body = (Rl.T @ fi_inertial.T).T
            wrench_generated = params_t["Pt"] @ fi_body.flatten()
            ul_force = Rl @ wrench_generated[:3]
            ul = ul.at[0:3].set(ul_force)
            ul = ul.at[3:6].set(wrench_generated[3:6])

        return jnp.concatenate([xl, ul, jnp.concatenate([xc, uc], axis=1).reshape(-1)])

    def _reduce_x(x):
        return jnp.concatenate([x[:nxl], x[nxl + nul :]])

    def _expand_z(z):
        if not args.eliminate_ul:
            return _project_full_x(z)
        xl = z[:nxl]
        rem = z[nxl:]
        ul_seed = jnp.zeros((nul,), dtype=z.dtype)
        x = jnp.concatenate([xl, ul_seed, rem])
        return _project_full_x(x)

    def _select_equalities(c_all):
        if args.project_wrench_equality:
            return c_all[0:0]
        if args.wrench_only_equality:
            return c_all[1 + nq :]
        return c_all

    f_step = lambda z: planner.ipoptax_objective(_expand_z(z), params_t, dims)

    if args.use_scales:
        c_step = lambda z: _select_equalities(
            params_t["eq_scale"] * planner.ipoptax_equality(_expand_z(z), params_t, dims)
        )
        g_step = lambda z: params_t["ineq_scale"] * planner.ipoptax_inequality(_expand_z(z), params_t, dims)
    else:
        c_step = lambda z: _select_equalities(
            planner.ipoptax_equality(_expand_z(z), params_t, dims)
        )
        g_step = lambda z: planner.ipoptax_inequality(_expand_z(z), params_t, dims)

    x_ref_full = _project_full_x(x_init)
    x_ref = _reduce_x(x_ref_full) if args.eliminate_ul else x_ref_full

    mu_schedule = parse_mu_schedule(args.mu_schedule)
    eq_rho_schedule = parse_optional_schedule(args.eq_rho_schedule, args.eq_rho, int(mu_schedule.shape[0]))

    def barrier_loss(z, p, mu, rho_eq, z_ref_local, lam_eq):
        f = f_step(z)
        c = c_step(z)
        g = g_step(z)
        slack = jnp.maximum(-g + args.barrier_eps, args.barrier_min_slack)
        barrier = -args.barrier_scale * mu * jnp.sum(jnp.log(slack))
        eq_lin = jnp.dot(lam_eq, c)
        if args.eq_rho_coupled_to_mu:
            rho_eq = args.eq_penalty_scale / mu
        eq_pen = 0.5 * rho_eq * jnp.sum(c**2)
        vio = jnp.maximum(g - args.barrier_eps, 0.0)
        ineq_pen = 0.5 * (args.ineq_penalty_scale / mu) * jnp.sum(vio**2)
        prox = 0.5 * args.prox_weight * jnp.sum((z - z_ref_local) ** 2)
        return f + barrier + eq_lin + eq_pen + ineq_pen + prox

    solver_kwargs = dict(
        fun=barrier_loss,
        maxiter=args.lbfgs_maxiter,
        tol=args.lbfgs_tol,
        linesearch=args.linesearch,
        maxls=args.maxls,
        jit=True,
        verbose=False,
    )
    if args.optimizer == "lbfgs":
        solver = LBFGS(history_size=args.history_size, **solver_kwargs)
    elif args.optimizer == "bfgs":
        solver = BFGS(**solver_kwargs)
    elif args.optimizer == "nonlinear_cg":
        solver = NonlinearCG(**solver_kwargs)
    else:
        solver = GradientDescent(**solver_kwargs)

    def solve_once():
        z = x_ref
        lam_eq = jnp.zeros_like(c_step(x_ref))
        last_iter = 0
        last_err = jnp.inf
        for stage_idx, mu in enumerate(mu_schedule):
            rho_eq = eq_rho_schedule[stage_idx]
            out = solver.run(z, None, mu, rho_eq, x_ref, lam_eq)
            z = out.params
            last_iter = int(np.array(out.state.iter_num))
            last_err = float(np.array(out.state.error))
            if args.eq_lam_update:
                c_val = c_step(z)
                rho_eq = float(args.eq_penalty_scale / float(mu)) if args.eq_rho_coupled_to_mu else float(rho_eq)
                lam_eq = jnp.clip(
                    lam_eq + args.eq_lam_update_scale * rho_eq * c_val,
                    -args.eq_lam_clip,
                    args.eq_lam_clip,
                )
        return z, last_iter, last_err

    runs_ms = []
    x_sol = None
    last_iter = None
    last_err = None
    for rep in range(max(args.bench_repeats, 1)):
        t0 = time.perf_counter()
        z_i, iter_i, err_i = solve_once()
        x_i = _expand_z(z_i)
        jax.block_until_ready(x_i)
        runs_ms.append((time.perf_counter() - t0) * 1000.0)
        if rep == 0:
            x_sol = np.array(x_i, dtype=float)
            last_iter = iter_i
            last_err = err_i

    x_sol = np.array(_project_full_x(jnp.asarray(x_sol)), dtype=float)
    eq_inf, ineq_vio, feas = primal_feas_metrics(planner, x_sol, params_t, dims)
    diff = x_sol - x_orig
    return {
        "runs_ms": runs_ms,
        "warm_ms_mean": float(np.mean(runs_ms[1:])) if len(runs_ms) > 1 else float(runs_ms[0]),
        "warm_ms_min": float(np.min(runs_ms[1:])) if len(runs_ms) > 1 else float(runs_ms[0]),
        "last_inner_iter": int(last_iter),
        "last_opt_err": float(last_err),
        "eq_inf": float(eq_inf),
        "ineq_vio": float(ineq_vio),
        "feas": float(feas),
        "orig_rmse": float(np.sqrt(np.mean(diff**2))),
        "orig_max_abs": float(np.max(np.abs(diff))),
    }


def main():
    args = build_parser().parse_args()
    if args.backend == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif args.backend == "cuda":
        os.environ["JAX_PLATFORMS"] = "cuda"

    setup_t0 = time.perf_counter()
    ctx = build_context(args)
    build_context_ms = (time.perf_counter() - setup_t0) * 1000.0

    snap_t0 = time.perf_counter()
    para2 = capture_subp2_snapshot(ctx, args.target_admm_iter, args.max_iter_admm)
    capture_snapshot_ms = (time.perf_counter() - snap_t0) * 1000.0

    planner = ctx["planner"]
    orig_planner = ctx["orig_planner"]

    prob_t0 = time.perf_counter()
    dims, x_init, params_t = build_step_problem(planner, para2, args.target_step)
    build_step_problem_ms = (time.perf_counter() - prob_t0) * 1000.0

    orig_para2 = pack_original_para2(para2, orig_planner)
    orig_step_nlp = build_original_step_nlp(orig_planner, orig_para2, args.target_step)
    ipopt_t0 = time.perf_counter()
    sol_orig = orig_step_nlp["solver"](
        x0=orig_step_nlp["x0"],
        lbx=orig_step_nlp["lbx"],
        ubx=orig_step_nlp["ubx"],
        p=orig_step_nlp["p"],
        lbg=orig_step_nlp["lbg"],
        ubg=orig_step_nlp["ubg"],
    )
    ipopt_ms = (time.perf_counter() - ipopt_t0) * 1000.0
    orig_stats = summarize_ipopt_stats(orig_step_nlp["solver"].stats())
    x_orig = build_original_step_reference(orig_planner.ADMM_SubP2(orig_para2), planner, args.target_step)
    ipopt_eq_inf, ipopt_ineq_vio, ipopt_feas = primal_feas_metrics(planner, x_orig, params_t, dims)

    postdoc_res = solve_postdoc_variant(planner, dims, x_init, params_t, x_orig, args)

    out = {
        "meta": {
            "task_idx": int(args.task_idx),
            "horizon": int(args.horizon),
            "max_iter_admm": int(args.max_iter_admm),
            "target_admm_iter": int(args.target_admm_iter),
            "target_step": int(args.target_step),
            "backend": args.backend,
            "decision_dim": int(np.array(x_init).shape[0]),
            "eq_dim": int(np.array(planner.ipoptax_equality(x_init, params_t, dims)).shape[0]),
            "ineq_dim": int(np.array(planner.ipoptax_inequality(x_init, params_t, dims)).shape[0]),
        },
        "setup_ms": {
            "build_context_ms": build_context_ms,
            "capture_snapshot_ms": capture_snapshot_ms,
            "build_step_problem_ms": build_step_problem_ms,
            "total_pre_solve_ms": build_context_ms + capture_snapshot_ms + build_step_problem_ms,
        },
        "ipopt": {
            "wall_ms": ipopt_ms,
            "iter_count": orig_stats.get("iter_count"),
            "return_status": orig_stats.get("return_status"),
            "eq_inf": float(ipopt_eq_inf),
            "ineq_vio": float(ipopt_ineq_vio),
            "feas": float(ipopt_feas),
        },
        "postdoc_barrier": postdoc_res,
    }

    print(json.dumps(out, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
