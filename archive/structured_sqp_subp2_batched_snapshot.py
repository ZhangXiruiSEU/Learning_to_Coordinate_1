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
from ipoptax.solver import LinearSystemFormulation, solve as ipoptax_solve
from scan_subp2_snapshot import (
    build_context,
    build_parser as build_snapshot_parser,
    build_original_step_reference,
    capture_subp2_snapshot,
    pack_original_para2,
)


def build_parser():
    p = argparse.ArgumentParser(description="Batched structured SQP prototype for SubP2 snapshot.")
    p.add_argument("--task-idx", type=int, default=0)
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--max-iter-admm", type=int, default=3)
    p.add_argument("--target-admm-iter", type=int, default=1)
    p.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--sqp-iters", type=int, default=4)
    p.add_argument("--objective-weight", type=float, default=1.0)
    p.add_argument("--prox-weight", type=float, default=0.0)
    p.add_argument("--prox-center", choices=["init", "current"], default="init")
    p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--step-norm-cap", type=float, default=1.0)
    p.add_argument("--trust-adapt", action="store_true")
    p.add_argument("--trust-shrink", type=float, default=0.5)
    p.add_argument("--trust-grow", type=float, default=1.5)
    p.add_argument("--trust-ratio-bad", type=float, default=0.1)
    p.add_argument("--trust-ratio-good", type=float, default=0.75)
    p.add_argument("--trust-step-min", type=float, default=0.05)
    p.add_argument("--trust-step-max", type=float, default=1.0)
    p.add_argument("--trust-prox-min", type=float, default=0.0)
    p.add_argument("--trust-prox-max", type=float, default=1.0)
    p.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--early-stop-eq", type=float, default=2.5e-4)
    p.add_argument("--early-stop-ineq", type=float, default=1e-6)
    p.add_argument("--early-stop-step", type=float, default=1.1e-2)
    p.add_argument("--merit-rho", type=float, default=10.0)
    p.add_argument("--osqp-maxiter", type=int, default=200)
    p.add_argument("--osqp-tol", type=float, default=1e-5)
    p.add_argument("--osqp-warm-start", action="store_true")
    p.add_argument("--osqp-rho-start", type=float, default=0.1)
    p.add_argument("--osqp-sigma", type=float, default=1e-4)
    p.add_argument("--osqp-momentum", type=float, default=1.6)
    p.add_argument("--use-lagrangian-hessian", action="store_true")
    p.add_argument("--qp-var-scale", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--qp-row-scale", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--active-set-k", type=int, default=0)
    p.add_argument("--reduce-eq", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--compare-ipoptax", action="store_true")
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


def main():
    args = parse_args()
    ensure_backend(args.backend)

    print("[BSQP] Building context", flush=True)
    ctx = build_context(args)
    planner = ctx["planner"]
    orig_planner = ctx["orig_planner"]
    para2 = capture_subp2_snapshot(ctx, args.target_admm_iter, args.max_iter_admm)
    w_init_batch, params_batch = planner._prepare_subp2_batch(para2)
    N = planner.N
    dims = (planner.nxl, planner.nul, planner.nxi, planner.nui, int(planner.nq), int(planner.num_dis))

    x_batch0 = jnp.array(w_init_batch[:N])
    params_batch = {k: jnp.array(v[:N]) for k, v in params_batch.items()}
    x_ref_batch0 = x_batch0

    orig_para2 = pack_original_para2(para2, orig_planner)
    orig_sol = orig_planner.ADMM_SubP2(orig_para2)
    orig_ref_batch = np.stack(
        [build_original_step_reference(orig_sol, orig_planner, k) for k in range(N)],
        axis=0,
    )

    solver = OSQP(
        eq_qp_solve="lu",
        maxiter=args.osqp_maxiter,
        tol=args.osqp_tol,
        termination_check_frequency=1,
        check_primal_dual_infeasability=False,
        sigma=args.osqp_sigma,
        rho_start=args.osqp_rho_start,
        momentum=args.osqp_momentum,
    )
    alpha_candidates = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125], dtype=x_batch0.dtype)

    def f_step(x, p):
        return planner.ipoptax_objective(x, p, dims)

    def c_step(x, p):
        return planner.ipoptax_equality(x, p, dims)

    def g_step(x, p):
        return planner.ipoptax_inequality(x, p, dims)

    def merit_step(x, p):
        cval = c_step(x, p)
        gval = g_step(x, p)
        return f_step(x, p) + args.merit_rho * (
            jnp.linalg.norm(cval) + jnp.linalg.norm(jnp.maximum(gval, 0.0))
        )

    def summarize_batch(tag, x_batch_np):
        c_batch = jax.vmap(c_step)(jnp.array(x_batch_np), params_batch)
        g_batch = jax.vmap(g_step)(jnp.array(x_batch_np), params_batch)
        eq_inf = np.max(np.abs(np.array(c_batch)), axis=1)
        ineq_vio = np.maximum(0.0, np.max(np.array(g_batch), axis=1))
        diff = np.array(x_batch_np) - orig_ref_batch
        rmse = np.sqrt(np.mean(diff**2, axis=1))
        max_abs = np.max(np.abs(diff), axis=1)
        print(
            f"[{tag}] eq_max={float(np.max(eq_inf)):.3e} "
            f"ineq_max={float(np.max(ineq_vio)):.3e} "
            f"rmse_mean={float(np.mean(rmse)):.3e} "
            f"rmse_max={float(np.max(rmse)):.3e} "
            f"max_abs={float(np.max(max_abs)):.3e}",
            flush=True,
        )

    summarize_batch("XINIT", np.array(x_batch0))

    if args.compare_ipoptax:
        eps_s = 1e-3

        def solve_ipoptax_one(x_init, p_t):
            c0 = c_step(x_init, p_t)
            g0 = g_step(x_init, p_t)
            ws_s0 = jnp.minimum(jnp.maximum(-g0 + eps_s, eps_s), 0.5)
            ws_z0 = jnp.full_like(ws_s0, 1e-3)
            ws_y0 = jnp.zeros_like(c0)
            res = ipoptax_solve(
                f=lambda x: f_step(x, p_t),
                c=lambda x: p_t["eq_scale"] * planner.ipoptax_equality(x, p_t, dims),
                g=lambda x: p_t["ineq_scale"] * planner.ipoptax_inequality(x, p_t, dims),
                ws_x=x_init,
                ws_s=ws_s0,
                ws_y=ws_y0,
                ws_z=ws_z0,
                max_iterations=120,
                max_kkt_violation=1e-3,
                lin_sys_formulation=LinearSystemFormulation.STABLE_DIRECT_4x4,
                tau_min=0.99,
                mu_min=1e-8,
                min_delta=1e-4,
                gamma_y=1e-4,
                gamma_z=1e-2,
                line_search_factor=0.2,
                line_search_min_step_size=1e-10,
                positivity_floor=1e-12,
                soft_stop_enabled=False,
                plateau_stop_enabled=False,
                print_logs=False,
                trace_length=0,
            )
            return res["x"]

        batched_ipoptax = jax.jit(jax.vmap(solve_ipoptax_one, in_axes=(0, 0)))
        t_ipm = time.perf_counter()
        x_ipm = np.array(batched_ipoptax(x_batch0, params_batch))
        ipm_ms = (time.perf_counter() - t_ipm) * 1000.0
        print(f"[IPOPTAX] solve_ms={ipm_ms:.3f}", flush=True)
        summarize_batch("IPOPTAX", x_ipm)

    def build_qp_data(x, p, x_ref, lam_eq, lam_ineq, prox_weight):
        grad_f = jax.grad(lambda xx: f_step(xx, p))(x)
        cval = c_step(x, p)
        gval = g_step(x, p)
        Aeq = jax.jacfwd(lambda xx: c_step(xx, p))(x)
        Aineq = jax.jacfwd(lambda xx: g_step(xx, p))(x)
        if args.use_lagrangian_hessian:
            lag = lambda xx: f_step(xx, p) + jnp.dot(lam_eq, c_step(xx, p)) + jnp.dot(lam_ineq, g_step(xx, p))
            hess_base = jax.hessian(lag)(x)
        else:
            hess_base = jax.hessian(lambda xx: f_step(xx, p))(x)
        H = project_psd_cone(hess_base, delta=args.reg, use_lapack=False, iterate=True)
        Hn = (
            args.objective_weight * H
            + (1.0 - args.objective_weight) * args.reg * jnp.eye(H.shape[0], dtype=H.dtype)
            + prox_weight * jnp.eye(H.shape[0], dtype=H.dtype)
        )
        gn = args.objective_weight * grad_f + prox_weight * (x - x_ref)
        beq = -cval
        bineq = -gval

        if args.active_set_k > 0:
            k = min(args.active_set_k, int(gval.shape[0]))
            top_idx = jnp.argsort(gval)[-k:]
            Aineq = Aineq[top_idx, :]
            bineq = bineq[top_idx]

        if args.qp_var_scale:
            hdiag = jnp.maximum(jnp.abs(jnp.diag(Hn)), args.reg)
            var_scale = jnp.sqrt(hdiag)
        else:
            var_scale = jnp.ones(Hn.shape[0], dtype=Hn.dtype)

        d_inv = 1.0 / jnp.maximum(var_scale, 1e-8)
        Hs = (d_inv[:, None] * Hn) * d_inv[None, :]
        gs = d_inv * gn
        Aeqs = Aeq * d_inv[None, :]
        Aineqs = Aineq * d_inv[None, :]

        if args.qp_row_scale:
            eq_row = 1.0 / jnp.maximum(jnp.linalg.norm(Aeqs, axis=1), 1.0)
            ineq_row = 1.0 / jnp.maximum(jnp.linalg.norm(Aineqs, axis=1), 1.0)
            Aeqs = eq_row[:, None] * Aeqs
            beq = eq_row * beq
            Aineqs = ineq_row[:, None] * Aineqs
            bineq = ineq_row * bineq

        return Hs, gs, Aeqs, beq, Aineqs, bineq, var_scale

    def reduce_eq_qp(Hn, gn, Aeq, beq, Aineq, bineq):
        n = Hn.shape[0]
        meq = Aeq.shape[0]
        # Full row rank is assumed for this SubP2 block.
        U, S, Vh = jnp.linalg.svd(Aeq, full_matrices=True)
        V = Vh.T
        Z = V[:, meq:]
        y = jnp.linalg.solve(Aeq @ Aeq.T + 1e-9 * jnp.eye(meq, dtype=Aeq.dtype), beq)
        d0 = Aeq.T @ y
        Hred = Z.T @ Hn @ Z
        gred = Z.T @ (Hn @ d0 + gn)
        Gred = Aineq @ Z
        hred = bineq - Aineq @ d0
        return d0, Z, Hred, gred, Gred, hred

    def init_osqp_params_one(x, p, x_ref, prox_weight):
        c0 = c_step(x, p)
        g0 = g_step(x, p)
        lam_eq0 = jnp.zeros_like(c0)
        if args.active_set_k > 0:
            lam_ineq0 = jnp.zeros((min(args.active_set_k, int(g0.shape[0])),), dtype=g0.dtype)
        else:
            lam_ineq0 = jnp.zeros_like(g0)
        Hn, gn, Aeq, beq, Aineq, bineq, _ = build_qp_data(x, p, x_ref, lam_eq0, lam_ineq0, prox_weight)
        if args.reduce_eq:
            d0, Z, Hred, gred, Gred, hred = reduce_eq_qp(Hn, gn, Aeq, beq, Aineq, bineq)
            del d0, Z
            return solver.init_params(
                init_x=jnp.zeros((Hred.shape[0],), dtype=x.dtype),
                params_obj=(Hred, gred),
                params_eq=(jnp.zeros((0, Hred.shape[0]), dtype=x.dtype), jnp.zeros((0,), dtype=x.dtype)),
                params_ineq=(Gred, hred),
            )
        return solver.init_params(
            init_x=jnp.zeros_like(x),
            params_obj=(Hn, gn),
            params_eq=(Aeq, beq),
            params_ineq=(Aineq, bineq),
        )

    def sqp_direction_one_cold(x, p, x_ref, prox_weight, step_norm_cap):
        c0 = c_step(x, p)
        g0 = g_step(x, p)
        lam_eq0 = jnp.zeros_like(c0)
        lam_ineq0 = jnp.zeros_like(g0)
        Hn, gn, Aeq, beq, Aineq, bineq, var_scale = build_qp_data(x, p, x_ref, lam_eq0, lam_ineq0, prox_weight)
        if args.reduce_eq:
            d0, Z, Hred, gred, Gred, hred = reduce_eq_qp(Hn, gn, Aeq, beq, Aineq, bineq)
            init_params_red = solver.init_params(
                init_x=jnp.zeros((Hred.shape[0],), dtype=x.dtype),
                params_obj=(Hred, gred),
                params_eq=(jnp.zeros((0, Hred.shape[0]), dtype=x.dtype), jnp.zeros((0,), dtype=x.dtype)),
                params_ineq=(Gred, hred),
            )
            step = solver.run(
                init_params=init_params_red,
                params_obj=(Hred, gred),
                params_eq=(jnp.zeros((0, Hred.shape[0]), dtype=x.dtype), jnp.zeros((0,), dtype=x.dtype)),
                params_ineq=(Gred, hred),
            )
            sol = step.params
            d = d0 + Z @ sol.primal
            scaled_sol = sol
        else:
            step = solver.run(
                init_params=None,
                params_obj=(Hn, gn),
                params_eq=(Aeq, beq),
                params_ineq=(Aineq, bineq),
            )
            sol = step.params
            d = sol.primal / jnp.maximum(var_scale, 1e-8)
            scaled_sol = sol
        d_norm = jnp.linalg.norm(d)
        scale = jnp.minimum(1.0, step_norm_cap / jnp.maximum(d_norm, 1e-12))
        pred_red = -(jnp.dot(gn, d) + 0.5 * jnp.dot(d, Hn @ d))
        if not args.reduce_eq:
            scaled_sol = sol._replace(primal=sol.primal * scale)
        return d * scale, scaled_sol, step.state.iter_num, pred_red

    def sqp_direction_one_warm(x, p, x_ref, init_params, prox_weight, step_norm_cap):
        lam_eq = init_params.dual_eq
        lam_ineq = init_params.dual_ineq
        Hn, gn, Aeq, beq, Aineq, bineq, var_scale = build_qp_data(x, p, x_ref, lam_eq, lam_ineq, prox_weight)
        if args.reduce_eq:
            d0, Z, Hred, gred, Gred, hred = reduce_eq_qp(Hn, gn, Aeq, beq, Aineq, bineq)
            step = solver.run(
                init_params=init_params,
                params_obj=(Hred, gred),
                params_eq=(jnp.zeros((0, Hred.shape[0]), dtype=x.dtype), jnp.zeros((0,), dtype=x.dtype)),
                params_ineq=(Gred, hred),
            )
            sol = step.params
            d = d0 + Z @ sol.primal
            scaled_sol = sol
        else:
            step = solver.run(
                init_params=init_params,
                params_obj=(Hn, gn),
                params_eq=(Aeq, beq),
                params_ineq=(Aineq, bineq),
            )
            sol = step.params
            d = sol.primal / jnp.maximum(var_scale, 1e-8)
            scaled_sol = sol
        d_norm = jnp.linalg.norm(d)
        scale = jnp.minimum(1.0, step_norm_cap / jnp.maximum(d_norm, 1e-12))
        pred_red = -(jnp.dot(gn, d) + 0.5 * jnp.dot(d, Hn @ d))
        if not args.reduce_eq:
            scaled_sol = sol._replace(primal=sol.primal * scale)
        return d * scale, scaled_sol, step.state.iter_num, pred_red

    batched_init_params = jax.jit(jax.vmap(init_osqp_params_one, in_axes=(0, 0, 0, None)))
    batched_direction_cold = jax.jit(jax.vmap(sqp_direction_one_cold, in_axes=(0, 0, 0, None, None)))
    batched_direction_warm = jax.jit(jax.vmap(sqp_direction_one_warm, in_axes=(0, 0, 0, 0, None, None)))

    def choose_alpha_one(x, p, d):
        m0 = merit_step(x, p)
        trial_x = x[None, :] + alpha_candidates[:, None] * d[None, :]
        trial_m = jax.vmap(lambda xx: merit_step(xx, p))(trial_x)
        ok = trial_m <= m0
        first_idx = jnp.argmax(ok)
        any_ok = jnp.any(ok)
        alpha = jnp.where(any_ok, alpha_candidates[first_idx], 0.0)
        return alpha

    batched_alpha = jax.jit(jax.vmap(choose_alpha_one, in_axes=(0, 0, 0)))

    x_batch = x_batch0
    step_norm_cap = float(args.step_norm_cap)
    prox_weight = float(args.prox_weight)
    x_ref_batch = x_batch if args.prox_center == "current" else x_ref_batch0
    osqp_params_batch = batched_init_params(x_batch, params_batch, x_ref_batch, prox_weight)
    t0 = time.perf_counter()
    for it in range(args.sqp_iters):
        x_ref_batch = x_batch if args.prox_center == "current" else x_ref_batch0
        merit_before = np.array(jax.vmap(merit_step)(x_batch, params_batch))
        if args.osqp_warm_start:
            d_batch, sol_batch, osqp_iters, pred_red = batched_direction_warm(
                x_batch, params_batch, x_ref_batch, osqp_params_batch, prox_weight, step_norm_cap
            )
            osqp_params_batch = sol_batch
        else:
            d_batch, sol_batch, osqp_iters, pred_red = batched_direction_cold(
                x_batch, params_batch, x_ref_batch, prox_weight, step_norm_cap
            )
        alpha_batch = batched_alpha(x_batch, params_batch, d_batch)
        x_batch = x_batch + alpha_batch[:, None] * d_batch
        merit_after = np.array(jax.vmap(merit_step)(x_batch, params_batch))
        pred_red_np = np.maximum(np.array(pred_red), 1e-12)
        act_red_np = np.maximum(merit_before - merit_after, 0.0)
        ratio_np = act_red_np / pred_red_np
        ratio_mean = float(np.mean(ratio_np))
        step_norm_mean = float(np.mean(np.linalg.norm(np.array(d_batch), axis=1)))
        if args.trust_adapt:
            if ratio_mean < args.trust_ratio_bad:
                step_norm_cap = max(args.trust_step_min, step_norm_cap * args.trust_shrink)
                prox_weight = min(args.trust_prox_max, max(args.trust_prox_min, prox_weight * 2.0 + 1e-4))
            elif ratio_mean > args.trust_ratio_good and float(np.mean(np.array(alpha_batch))) > 0.9:
                step_norm_cap = min(args.trust_step_max, step_norm_cap * args.trust_grow)
                prox_weight = max(args.trust_prox_min, prox_weight * 0.5)

        c_batch = jax.vmap(c_step)(x_batch, params_batch)
        g_batch = jax.vmap(g_step)(x_batch, params_batch)
        eq_inf = np.max(np.abs(np.array(c_batch)), axis=1)
        ineq_vio = np.maximum(0.0, np.max(np.array(g_batch), axis=1))
        diff = np.array(x_batch) - orig_ref_batch
        rmse = np.sqrt(np.mean(diff**2, axis=1))
        print(
            f"[BSQP] iter={it+1} "
            f"osqp_iter_mean={float(np.mean(np.array(osqp_iters))):.1f} "
            f"pred_ratio_mean={ratio_mean:.3e} "
            f"step_norm_mean={step_norm_mean:.3e} "
            f"step_cap={step_norm_cap:.3e} "
            f"prox={prox_weight:.3e} "
            f"eq_max={float(np.max(eq_inf)):.3e} "
            f"ineq_max={float(np.max(ineq_vio)):.3e} "
            f"rmse_mean={float(np.mean(rmse)):.3e} "
            f"rmse_max={float(np.max(rmse)):.3e}",
            flush=True,
        )
        if (
            args.early_stop
            and float(np.max(eq_inf)) <= args.early_stop_eq
            and float(np.max(ineq_vio)) <= args.early_stop_ineq
            and step_norm_mean <= args.early_stop_step
        ):
            print(f"[BSQP] early-stop at iter={it+1}", flush=True)
            break

    total_ms = (time.perf_counter() - t0) * 1000.0
    c_batch = jax.vmap(c_step)(x_batch, params_batch)
    g_batch = jax.vmap(g_step)(x_batch, params_batch)
    eq_inf = np.max(np.abs(np.array(c_batch)), axis=1)
    ineq_vio = np.maximum(0.0, np.max(np.array(g_batch), axis=1))
    diff = np.array(x_batch) - orig_ref_batch
    rmse = np.sqrt(np.mean(diff**2, axis=1))
    print(
        f"[BSQP] final_ms={total_ms:.3f} "
        f"eq_max={float(np.max(eq_inf)):.3e} "
        f"ineq_max={float(np.max(ineq_vio)):.3e} "
        f"rmse_mean={float(np.mean(rmse)):.3e} "
        f"rmse_max={float(np.max(rmse)):.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
