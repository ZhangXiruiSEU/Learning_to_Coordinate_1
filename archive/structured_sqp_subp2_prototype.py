import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import OSQP

from ipoptax.linalg_helpers import project_psd_cone
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
    p = argparse.ArgumentParser(description="Prototype structured SQP for one SubP2 snapshot step.")
    p.add_argument("--task-idx", type=int, default=0)
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--max-iter-admm", type=int, default=3)
    p.add_argument("--target-admm-iter", type=int, default=1)
    p.add_argument("--target-step", type=int, default=0)
    p.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--sqp-iters", type=int, default=2)
    p.add_argument("--active-margin", type=float, default=5e-3)
    p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--step-norm-cap", type=float, default=1.0)
    p.add_argument("--line-search-factor", type=float, default=0.5)
    p.add_argument("--line-search-min-alpha", type=float, default=1e-3)
    p.add_argument("--merit-rho", type=float, default=10.0)
    p.add_argument("--qp-active-iters", type=int, default=20)
    p.add_argument("--qp-violation-tol", type=float, default=1e-8)
    p.add_argument("--objective-weight", type=float, default=1.0)
    p.add_argument("--prox-weight", type=float, default=0.0)
    return p


def ensure_backend(backend):
    import os

    if backend == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif backend == "cuda":
        os.environ["JAX_PLATFORMS"] = "cuda"


def parse_args():
    args = build_parser().parse_args()
    defaults = build_snapshot_parser().parse_args([])
    for key, value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def merit_fn(f, c, g, rho, x):
    cval = c(x)
    gval = g(x)
    return f(x) + rho * (jnp.linalg.norm(cval) + jnp.linalg.norm(jnp.maximum(gval, 0.0)))


def solve_sqp_step(
    f,
    c,
    g,
    x,
    reg,
    step_norm_cap,
    objective_weight,
    prox_weight,
    x_ref,
):
    grad_f = jax.grad(f)(x)
    hess_f = jax.hessian(f)(x)
    H = project_psd_cone(hess_f, delta=reg, use_lapack=False, iterate=True)

    cval = c(x)
    gval = g(x)
    Aeq = np.array(jax.jacfwd(c)(x))
    Aineq = np.array(jax.jacfwd(g)(x))

    x_np = np.array(x)
    x_ref_np = np.array(x_ref)
    Hn = (
        objective_weight * np.array(H)
        + (1.0 - objective_weight) * reg * np.eye(np.array(H).shape[0])
        + prox_weight * np.eye(np.array(H).shape[0])
    )
    gn = objective_weight * np.array(grad_f) + prox_weight * (x_np - x_ref_np)
    beq = -np.array(cval)
    bineq = -np.array(gval)
    solver = OSQP(
        eq_qp_solve="lu",
        maxiter=200,
        tol=1e-5,
        termination_check_frequency=1,
        check_primal_dual_infeasability=False,
        sigma=1e-4,
    )
    sol, _ = solver.run(
        init_params=None,
        params_obj=(jnp.array(Hn), jnp.array(gn)),
        params_eq=(jnp.array(Aeq), jnp.array(beq)),
        params_ineq=(jnp.array(Aineq), jnp.array(bineq)),
    )
    d = np.array(sol.primal)

    d_norm = np.linalg.norm(d)
    if d_norm > step_norm_cap:
        d = d * (step_norm_cap / max(d_norm, 1e-12))

    active = np.array(gval) >= -1e-8
    return jnp.array(d), int(np.sum(active))


def main():
    args = parse_args()
    ensure_backend(args.backend)

    print("[SQP] Building context", flush=True)
    ctx = build_context(args)
    para2 = capture_subp2_snapshot(ctx, args.target_admm_iter, args.max_iter_admm)
    planner = ctx["planner"]
    dims, x_init, params_t = build_step_problem(planner, para2, args.target_step)

    orig_para2 = pack_original_para2(para2, ctx["orig_planner"])
    orig_sol = ctx["orig_planner"].ADMM_SubP2(orig_para2)
    orig_ref = build_original_step_reference(orig_sol, ctx["orig_planner"], args.target_step)

    f = lambda x: planner.ipoptax_objective(x, params_t, dims)
    c = lambda x: planner.ipoptax_equality(x, params_t, dims)
    g = lambda x: planner.ipoptax_inequality(x, params_t, dims)

    x = jnp.array(x_init)
    x_ref = jnp.array(x_init)
    merit0 = float(merit_fn(f, c, g, args.merit_rho, x))
    print(f"[SQP] init merit={merit0:.6e}", flush=True)

    t0 = time.perf_counter()
    for it in range(args.sqp_iters):
        d, n_active = solve_sqp_step(
            f=f,
            c=c,
            g=g,
            x=x,
            reg=args.reg,
            step_norm_cap=args.step_norm_cap,
            objective_weight=args.objective_weight,
            prox_weight=args.prox_weight,
            x_ref=x_ref,
        )
        alpha = 1.0
        merit_x = float(merit_fn(f, c, g, args.merit_rho, x))
        accepted = False
        while alpha >= args.line_search_min_alpha:
            x_trial = x + alpha * d
            merit_trial = float(merit_fn(f, c, g, args.merit_rho, x_trial))
            if merit_trial <= merit_x:
                x = x_trial
                accepted = True
                break
            alpha *= args.line_search_factor

        eq_inf, ineq_vio, feas = primal_feas_metrics(planner, np.array(x), params_t, dims)
        diff = np.array(x) - orig_ref
        rmse = float(np.sqrt(np.mean(diff**2)))
        max_abs = float(np.max(np.abs(diff)))
        print(
            f"[SQP] iter={it+1} accepted={int(accepted)} alpha={alpha:.3e} "
            f"active={n_active} eq_inf={eq_inf:.3e} ineq_vio={ineq_vio:.3e} "
            f"rmse={rmse:.3e} max_abs={max_abs:.3e}",
            flush=True,
        )

    total_ms = (time.perf_counter() - t0) * 1000.0
    eq_inf, ineq_vio, feas = primal_feas_metrics(planner, np.array(x), params_t, dims)
    diff = np.array(x) - orig_ref
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    print(
        f"[SQP] final_ms={total_ms:.3f} eq_inf={eq_inf:.3e} ineq_vio={ineq_vio:.3e} "
        f"rmse={rmse:.3e} max_abs={max_abs:.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
