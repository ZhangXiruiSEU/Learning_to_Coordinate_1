import argparse
import json
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

import jax
import jax.numpy as jnp
import numpy as np

from diagnose_subp2_step import build_original_step_nlp, summarize_ipopt_stats
from ipoptax.solver import LinearSystemFormulation, solve as ipoptax_solve
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
        description="Authoritative real SubP2 single-step benchmark with separated setup/solve timing.",
        parents=[build_snapshot_parser()],
        conflict_handler="resolve",
    )
    p.add_argument("--target-admm-iter", type=int, default=1)
    p.add_argument("--target-step", type=int, default=0)
    p.add_argument("--solver-formulation", default="SYMMETRIC_INDIRECT_2x2")
    p.add_argument("--max-solver-iters", type=int, default=120)
    p.add_argument("--max-kkt-violation", type=float, default=1e-6)
    p.add_argument("--armijo-factor", type=float, default=1e-4)
    p.add_argument("--feasibility-armijo-factor", type=float, default=1e-4)
    p.add_argument("--tau-min", type=float, default=0.99)
    p.add_argument("--mu-min", type=float, default=1e-8)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--gamma-y", type=float, default=1e-4)
    p.add_argument("--gamma-z", type=float, default=1e-2)
    p.add_argument("--line-search-factor", type=float, default=0.2)
    p.add_argument("--line-search-min-step-size", type=float, default=1e-10)
    p.add_argument("--s0-cap", type=float, default=0.5)
    p.add_argument("--z0-init", type=float, default=1e-2)
    p.add_argument("--bench-repeats", type=int, default=3)
    p.add_argument(
        "--jax-variant",
        choices=["baseline", "soft_stop", "plateau_stop", "all"],
        default="all",
    )
    p.add_argument("--json-out", default="")
    return p


def ensure_backend(backend):
    if backend == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif backend == "cuda":
        os.environ["JAX_PLATFORMS"] = "cuda"


def formulation_from_name(name):
    return getattr(LinearSystemFormulation, name)


def _variant_extra(name):
    if name == "soft_stop":
        return dict(
            soft_stop_enabled=True,
            soft_mu_tol=1e-8,
            soft_comp_tol=1e-8,
            soft_dual_tol=5e-5,
            soft_eq_tol=1e-4,
            soft_ineq_tol=5e-3,
        )
    if name == "plateau_stop":
        return dict(
            plateau_stop_enabled=True,
            plateau_min_iterations=12,
            plateau_patience=8,
            plateau_phi_tol=1e-7,
            plateau_ineq_tol=1e-7,
        )
    return {}


def run_jax_variant(planner, dims, x_init, params_t, x_orig, args, variant):
    c_raw = lambda x: planner.ipoptax_equality(x, params_t, dims)
    g_raw = lambda x: planner.ipoptax_inequality(x, params_t, dims)
    f = lambda x: planner.ipoptax_objective(x, params_t, dims)
    c_scale = params_t["eq_scale"]
    g_scale = params_t["ineq_scale"]
    c = lambda x: c_scale * c_raw(x)
    g = lambda x: g_scale * g_raw(x)

    g0 = g(x_init)
    ws_s = jnp.minimum(jnp.maximum(-g0 + 1e-3, 1e-3), args.s0_cap)
    ws_y = jnp.zeros_like(c(x_init))
    ws_z = jnp.full_like(ws_s, args.z0_init)

    kwargs = dict(
        f=f,
        c=c,
        g=g,
        ws_x=x_init,
        ws_s=ws_s,
        ws_y=ws_y,
        ws_z=ws_z,
        max_iterations=args.max_solver_iters,
        max_kkt_violation=args.max_kkt_violation,
        lin_sys_formulation=formulation_from_name(args.solver_formulation),
        tau_min=args.tau_min,
        mu_min=args.mu_min,
        min_delta=args.min_delta,
        gamma_y=args.gamma_y,
        gamma_z=args.gamma_z,
        armijo_factor=args.armijo_factor,
        feasibility_armijo_factor=args.feasibility_armijo_factor,
        line_search_factor=args.line_search_factor,
        line_search_min_step_size=args.line_search_min_step_size,
        positivity_floor=1e-12,
        print_logs=False,
        trace_length=0,
    )
    kwargs.update(_variant_extra(variant))

    runs_ms = []
    res = None
    for rep in range(max(args.bench_repeats, 1)):
        t0 = time.perf_counter()
        res_i = ipoptax_solve(**kwargs)
        jax.block_until_ready(res_i["x"])
        runs_ms.append((time.perf_counter() - t0) * 1000.0)
        if rep == 0:
            res = res_i

    x_sol = np.array(res["x"], dtype=float)
    eq_inf, ineq_vio, feas = primal_feas_metrics(planner, x_sol, params_t, dims)
    diff = x_sol - x_orig
    return {
        "runs_ms": runs_ms,
        "warm_ms_mean": float(np.mean(runs_ms[1:])) if len(runs_ms) > 1 else float(runs_ms[0]),
        "warm_ms_min": float(np.min(runs_ms[1:])) if len(runs_ms) > 1 else float(runs_ms[0]),
        "iterations": int(np.array(res["iteration"])),
        "converged": bool(np.array(res["converged"])),
        "eq_inf": float(eq_inf),
        "ineq_vio": float(ineq_vio),
        "feas": float(feas),
        "orig_rmse": float(np.sqrt(np.mean(diff**2))),
        "orig_max_abs": float(np.max(np.abs(diff))),
    }


def main():
    args = build_parser().parse_args()
    ensure_backend(args.backend)

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

    variants = ["baseline", "soft_stop", "plateau_stop"] if args.jax_variant == "all" else [args.jax_variant]
    jax_results = {name: run_jax_variant(planner, dims, x_init, params_t, x_orig, args, name) for name in variants}

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
        "jax_ipoptax": jax_results,
    }

    print(json.dumps(out, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
