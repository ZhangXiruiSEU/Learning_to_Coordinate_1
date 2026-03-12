import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = ROOT / "archive"
for _path in (ROOT, ARCHIVE_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


def _early_set_backend():
    backend = None
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

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import BFGS, GradientDescent, LBFGS, NonlinearCG

from scan_subp2_snapshot import (
    build_context,
    build_original_step_reference,
    build_parser as build_snapshot_parser,
    capture_subp2_snapshot,
    pack_original_para2,
)


def build_parser():
    p = argparse.ArgumentParser(
        description="Batched JAX hybrid barrier reformulation prototype for SubP2 snapshot."
    )
    p.add_argument("--task-idx", type=int, default=0)
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--max-iter-admm", type=int, default=3)
    p.add_argument("--target-admm-iter", type=int, default=1)
    p.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--mu-schedule", type=str, default="1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4")
    p.add_argument("--optimizer", choices=["bfgs", "lbfgs", "gd", "nonlinear_cg"], default="bfgs")
    p.add_argument("--optimizer-schedule", type=str, default="")
    p.add_argument("--maxiter", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--stepsize", type=float, default=1e-2)
    p.add_argument("--history-size", type=int, default=10)
    p.add_argument("--linesearch", type=str, default="zoom")
    p.add_argument("--maxls", type=int, default=20)
    p.add_argument("--eq-penalty-scale", type=float, default=100.0)
    p.add_argument("--ineq-penalty-scale", type=float, default=100.0)
    p.add_argument("--barrier-scale", type=float, default=1.0)
    p.add_argument("--barrier-eps", type=float, default=1e-6)
    p.add_argument("--barrier-min-slack", type=float, default=1e-8)
    p.add_argument("--eq-lam-update", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eq-lam-update-scale", type=float, default=0.25)
    p.add_argument("--eq-lam-clip", type=float, default=1e3)
    p.add_argument("--prox-weight", type=float, default=0.0)
    p.add_argument("--use-scales", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--project-unit-equalities", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--project-wrench-equality", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wrench-only-equality", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--projection-eps", type=float, default=1e-9)
    p.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--early-stop-eq", type=float, default=1e-6)
    p.add_argument("--early-stop-ineq", type=float, default=1e-8)
    p.add_argument("--early-stop-step", type=float, default=2e-3)
    p.add_argument("--bench-repeats", type=int, default=1)
    return p


def parse_args():
    args = build_parser().parse_args()
    defaults = build_snapshot_parser().parse_args([])
    for key, value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def parse_mu_schedule(text):
    return jnp.array([float(x.strip()) for x in text.split(",") if x.strip()], dtype=jnp.float32)


def main():
    args = parse_args()
    print("[BARRIER-HYBRID] Building context", flush=True)
    ctx = build_context(args)
    planner = ctx["planner"]
    orig_planner = ctx["orig_planner"]
    para2 = capture_subp2_snapshot(ctx, args.target_admm_iter, args.max_iter_admm)
    w_init_batch, params_batch = planner._prepare_subp2_batch(para2)

    N = planner.N
    nxl, nul, nxi, nui, nq = planner.nxl, planner.nul, planner.nxi, planner.nui, int(planner.nq)
    dims = (nxl, nul, nxi, nui, nq, int(planner.num_dis))
    quat_slice = slice(6, 10)

    x_batch0 = jnp.array(w_init_batch[:N])
    params_batch = {k: jnp.array(v[:N]) for k, v in params_batch.items()}

    orig_para2 = pack_original_para2(para2, orig_planner)
    orig_sol = orig_planner.ADMM_SubP2(orig_para2)
    orig_ref_batch = np.stack(
        [build_original_step_reference(orig_sol, orig_planner, k) for k in range(N)],
        axis=0,
    )

    def project_full_x(x, p):
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

        if args.project_wrench_equality:
            ql = xl[quat_slice]
            Rl = planner._q_2_rotation_jax(ql)
            di_vecs = xc[:, 0:3]
            ti_mags = xc[:, 12]
            fi_inertial = di_vecs * ti_mags[:, None]
            fi_body = (Rl.T @ fi_inertial.T).T
            wrench_generated = p["Pt"] @ fi_body.flatten()
            ul_force = Rl @ wrench_generated[:3]
            ul_proj = ul.at[0:3].set(ul_force)
            ul_proj = ul_proj.at[3:6].set(wrench_generated[3:6])
            ul = jnp.where((p["is_terminal"] < 0.5), ul_proj, ul)

        return jnp.concatenate([xl, ul, jnp.concatenate([xc, uc], axis=1).reshape(-1)])

    v_project_full_x = jax.jit(jax.vmap(project_full_x, in_axes=(0, 0)))

    def select_equalities(c_all):
        if args.project_wrench_equality:
            return c_all[0:0]
        if args.wrench_only_equality:
            return c_all[1 + nq :]
        return c_all

    def f_step(x, p):
        return planner.ipoptax_objective(project_full_x(x, p), p, dims)

    def c_step(x, p):
        x_proj = project_full_x(x, p)
        c = planner.ipoptax_equality(x_proj, p, dims)
        c = select_equalities(c)
        return p["eq_scale"] * c if args.use_scales else c

    def g_step(x, p):
        x_proj = project_full_x(x, p)
        g = planner.ipoptax_inequality(x_proj, p, dims)
        return p["ineq_scale"] * g if args.use_scales else g

    x_ref_batch = v_project_full_x(x_batch0, params_batch)

    def barrier_loss(x, p, mu, x_ref, lam_eq):
        f = f_step(x, p)
        c = c_step(x, p)
        g = g_step(x, p)
        slack = jnp.maximum(-g + args.barrier_eps, args.barrier_min_slack)
        barrier = -args.barrier_scale * mu * jnp.sum(jnp.log(slack))
        eq_lin = jnp.dot(lam_eq, c)
        eq_pen = 0.5 * (args.eq_penalty_scale / mu) * jnp.sum(c**2)
        vio = jnp.maximum(g - args.barrier_eps, 0.0)
        ineq_pen = 0.5 * (args.ineq_penalty_scale / mu) * jnp.sum(vio**2)
        prox = 0.5 * args.prox_weight * jnp.sum((x - x_ref) ** 2)
        return f + barrier + eq_lin + eq_pen + ineq_pen + prox

    mu_schedule = parse_mu_schedule(args.mu_schedule)

    optimizer_schedule = [x.strip().lower() for x in args.optimizer_schedule.split(",") if x.strip()]
    if optimizer_schedule and len(optimizer_schedule) != int(mu_schedule.shape[0]):
        raise ValueError("--optimizer-schedule length must match mu-schedule length")

    def build_solver(opt_name):
        solver_kwargs = dict(
            fun=barrier_loss,
            maxiter=args.maxiter,
            tol=args.tol,
            jit=True,
            verbose=False,
        )
        if opt_name == "bfgs":
            return BFGS(
                linesearch=args.linesearch,
                maxls=args.maxls,
                **solver_kwargs,
            )
        if opt_name == "lbfgs":
            return LBFGS(
                history_size=args.history_size,
                linesearch=args.linesearch,
                maxls=args.maxls,
                **solver_kwargs,
            )
        if opt_name == "nonlinear_cg":
            return NonlinearCG(
                linesearch=args.linesearch,
                maxls=args.maxls,
                **solver_kwargs,
            )
        return GradientDescent(
            stepsize=args.stepsize,
            acceleration=False,
            **solver_kwargs,
        )

    stage_solvers = []
    stage_batched_solves = []
    for i in range(int(mu_schedule.shape[0])):
        opt_name = optimizer_schedule[i] if optimizer_schedule else args.optimizer
        solver = build_solver(opt_name)

        def solve_one(x0, p, mu, x_ref, lam_eq, _solver=solver):
            out = _solver.run(x0, p, mu, x_ref, lam_eq)
            x_proj = project_full_x(out.params, p)
            value = getattr(out.state, "value", jnp.nan)
            return x_proj, out.state.iter_num, out.state.error, value

        stage_solvers.append(opt_name)
        stage_batched_solves.append(jax.jit(jax.vmap(solve_one, in_axes=(0, 0, None, 0, 0))))

    def solve_one(x0, p, mu, x_ref, lam_eq):
        out = stage_batched_solves[0].__wrapped__(x0, p, mu, x_ref, lam_eq)  # unreachable placeholder
        x_proj = project_full_x(out.params, p)
        value = getattr(out.state, "value", jnp.nan)
        return x_proj, out.state.iter_num, out.state.error, value

    def summarize(tag, x_batch_np, step_norms=None, iter_nums=None, errors=None):
        x_batch_proj = np.array(v_project_full_x(jnp.array(x_batch_np), params_batch))
        c_raw = np.array(
            jax.vmap(lambda x, p: planner.ipoptax_equality(x, p, dims))(jnp.array(x_batch_proj), params_batch)
        )
        g_raw = np.array(
            jax.vmap(lambda x, p: planner.ipoptax_inequality(x, p, dims))(jnp.array(x_batch_proj), params_batch)
        )
        eq_inf = np.max(np.abs(c_raw), axis=1)
        ineq_vio = np.maximum(0.0, np.max(g_raw, axis=1))
        diff = x_batch_proj - orig_ref_batch
        rmse = np.sqrt(np.mean(diff**2, axis=1))
        max_abs = np.max(np.abs(diff), axis=1)
        extra = ""
        if step_norms is not None:
            extra += f" step_norm_mean={float(np.mean(step_norms)):.3e}"
        if iter_nums is not None:
            extra += f" iter_mean={float(np.mean(iter_nums)):.1f}"
        if errors is not None:
            extra += f" opt_err_max={float(np.max(errors)):.3e}"
        print(
            f"[{tag}] eq_max={float(np.max(eq_inf)):.3e} "
            f"ineq_max={float(np.max(ineq_vio)):.3e} "
            f"rmse_mean={float(np.mean(rmse)):.3e} "
            f"rmse_max={float(np.max(rmse)):.3e} "
            f"max_abs={float(np.max(max_abs)):.3e}{extra}",
            flush=True,
        )
        return float(np.max(eq_inf)), float(np.max(ineq_vio)), float(np.mean(rmse))

    summarize("XINIT", np.array(x_batch0))

    def run_once(tag_prefix):
        x_batch = x_ref_batch
        c0_batch = jax.vmap(c_step)(x_batch, params_batch)
        lam_eq_batch = jnp.zeros_like(c0_batch)
        t0 = time.perf_counter()
        for i, mu in enumerate(mu_schedule):
            prev_x = x_batch
            x_batch, iter_nums, errors, values = stage_batched_solves[i](
                x_batch, params_batch, mu, x_ref_batch, lam_eq_batch
            )
            step_norms = np.linalg.norm(np.array(x_batch - prev_x), axis=1)
            eq_max, ineq_max, _ = summarize(
                f"{tag_prefix} mu={float(mu):.2e} opt={stage_solvers[i]}",
                np.array(x_batch),
                step_norms=step_norms,
                iter_nums=np.array(iter_nums),
                errors=np.array(errors),
            )
            if args.eq_lam_update:
                c_batch = jax.vmap(c_step)(x_batch, params_batch)
                rho_eq = args.eq_penalty_scale / float(mu)
                lam_eq_batch = jnp.clip(
                    lam_eq_batch + args.eq_lam_update_scale * rho_eq * c_batch,
                    -args.eq_lam_clip,
                    args.eq_lam_clip,
                )
            if (
                args.early_stop
                and eq_max <= args.early_stop_eq
                and ineq_max <= args.early_stop_ineq
                and float(np.mean(step_norms)) <= args.early_stop_step
            ):
                print(f"[{tag_prefix}] early-stop at stage={i+1} mu={float(mu):.2e}", flush=True)
                break
        total_ms = (time.perf_counter() - t0) * 1000.0
        summarize(f"{tag_prefix} FINAL", np.array(x_batch))
        print(f"[{tag_prefix}] final_ms={total_ms:.3f}", flush=True)
        return total_ms

    total_ms_list = []
    for rep in range(args.bench_repeats):
        tag = "BARRIER" if args.bench_repeats == 1 else f"BARRIER run{rep+1}"
        total_ms_list.append(run_once(tag))

    if args.bench_repeats > 1:
        arr = np.array(total_ms_list, dtype=float)
        print(
            f"[BARRIER] repeats={args.bench_repeats} "
            f"mean_ms={float(np.mean(arr)):.3f} "
            f"best_ms={float(np.min(arr)):.3f} "
            f"last_ms={float(arr[-1]):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
