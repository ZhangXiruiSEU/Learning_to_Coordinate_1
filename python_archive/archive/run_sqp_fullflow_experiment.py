import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import OSQP

import JustWorkingOnIt as JAX_Planner
import verify_jax_vs_original
from ipoptax.linalg_helpers import project_psd_cone


def build_parser():
    p = argparse.ArgumentParser(description="Run fullflow verify with SQP-based SubP2 monkeypatch.")
    p.add_argument("--task-idx", type=int, default=0)
    p.add_argument("--show-plots", action="store_true")
    p.add_argument("--sqp-iters", type=int, default=4)
    p.add_argument("--objective-weight", type=float, default=1.0)
    p.add_argument("--prox-weight", type=float, default=0.0)
    p.add_argument("--step-norm-cap", type=float, default=1.0)
    p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--merit-rho", type=float, default=10.0)
    p.add_argument("--osqp-maxiter", type=int, default=200)
    p.add_argument("--osqp-tol", type=float, default=1e-5)
    p.add_argument("--osqp-rho-start", type=float, default=0.1)
    p.add_argument("--osqp-sigma", type=float, default=1e-4)
    p.add_argument("--osqp-momentum", type=float, default=1.6)
    p.add_argument("--use-lagrangian-hessian", action="store_true")
    p.add_argument("--qp-var-scale", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--qp-row-scale", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--early-stop-eq", type=float, default=2.5e-4)
    p.add_argument("--early-stop-ineq", type=float, default=1e-6)
    p.add_argument("--early-stop-step", type=float, default=1e-2)
    return p


def install_sqp_subp2_patch(args):
    def patched_jax_admm_subp2(self, Para2_dict):
        MPC_Planner = self
        N = self.N
        dims = (self.nxl, self.nul, self.nxi, self.nui, int(self.nq), int(self.num_dis))

        w_init_batch, params_batch = self._prepare_subp2_batch(Para2_dict)
        x_batch = jnp.array(w_init_batch)
        params_batch = {k: jnp.array(v) for k, v in params_batch.items()}
        x_ref_batch = x_batch

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
        alpha_candidates = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125], dtype=x_batch.dtype)

        def f_step(x, p):
            return MPC_Planner.ipoptax_objective(x, p, dims)

        def c_step(x, p):
            return MPC_Planner.ipoptax_equality(x, p, dims)

        def g_step(x, p):
            return MPC_Planner.ipoptax_inequality(x, p, dims)

        def merit_step(x, p):
            cval = c_step(x, p)
            gval = g_step(x, p)
            return f_step(x, p) + args.merit_rho * (
                jnp.linalg.norm(cval) + jnp.linalg.norm(jnp.maximum(gval, 0.0))
            )

        def build_qp_data(x, p, x_ref, lam_eq, lam_ineq):
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
                + args.prox_weight * jnp.eye(H.shape[0], dtype=H.dtype)
            )
            gn = args.objective_weight * grad_f + args.prox_weight * (x - x_ref)
            beq = -cval
            bineq = -gval
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

        def init_osqp_params_one(x, p, x_ref):
            c0 = c_step(x, p)
            g0 = g_step(x, p)
            lam_eq0 = jnp.zeros_like(c0)
            lam_ineq0 = jnp.zeros_like(g0)
            Hn, gn, Aeq, beq, Aineq, bineq, _ = build_qp_data(x, p, x_ref, lam_eq0, lam_ineq0)
            return solver.init_params(
                init_x=jnp.zeros_like(x),
                params_obj=(Hn, gn),
                params_eq=(Aeq, beq),
                params_ineq=(Aineq, bineq),
            )

        def sqp_direction_one_warm(x, p, x_ref, init_params):
            lam_eq = init_params.dual_eq
            lam_ineq = init_params.dual_ineq
            Hn, gn, Aeq, beq, Aineq, bineq, var_scale = build_qp_data(x, p, x_ref, lam_eq, lam_ineq)
            step = solver.run(
                init_params=init_params,
                params_obj=(Hn, gn),
                params_eq=(Aeq, beq),
                params_ineq=(Aineq, bineq),
            )
            sol = step.params
            d = sol.primal / jnp.maximum(var_scale, 1e-8)
            d_norm = jnp.linalg.norm(d)
            scale = jnp.minimum(1.0, args.step_norm_cap / jnp.maximum(d_norm, 1e-12))
            pred_red = -(jnp.dot(gn, d) + 0.5 * jnp.dot(d, Hn @ d))
            scaled_sol = sol._replace(primal=sol.primal * scale)
            return d * scale, scaled_sol, step.state.iter_num, pred_red

        batched_init_params = jax.jit(jax.vmap(init_osqp_params_one, in_axes=(0, 0, 0)))
        batched_direction_warm = jax.jit(jax.vmap(sqp_direction_one_warm, in_axes=(0, 0, 0, 0)))

        def choose_alpha_one(x, p, d):
            m0 = merit_step(x, p)
            trial_x = x[None, :] + alpha_candidates[:, None] * d[None, :]
            trial_m = jax.vmap(lambda xx: merit_step(xx, p))(trial_x)
            ok = trial_m <= m0
            first_idx = jnp.argmax(ok)
            any_ok = jnp.any(ok)
            return jnp.where(any_ok, alpha_candidates[first_idx], 0.0)

        batched_alpha = jax.jit(jax.vmap(choose_alpha_one, in_axes=(0, 0, 0)))

        osqp_params_batch = batched_init_params(x_batch, params_batch, x_ref_batch)
        diag_iters = []
        diag_eq_inf = []
        diag_ineq_vio = []
        mode_codes = np.full((N + 1,), 1, dtype=int)  # main
        for it in range(args.sqp_iters):
            d_batch, sol_batch, osqp_iters, pred_red = batched_direction_warm(
                x_batch, params_batch, x_ref_batch, osqp_params_batch
            )
            osqp_params_batch = sol_batch
            alpha_batch = batched_alpha(x_batch, params_batch, d_batch)
            x_batch = x_batch + alpha_batch[:, None] * d_batch

            c_batch = jax.vmap(c_step)(x_batch, params_batch)
            g_batch = jax.vmap(g_step)(x_batch, params_batch)
            eq_inf = np.max(np.abs(np.array(c_batch)), axis=1)
            ineq_vio = np.maximum(0.0, np.max(np.array(g_batch), axis=1))
            step_norm_mean = float(np.mean(np.linalg.norm(np.array(d_batch), axis=1)))
            diag_iters = np.array(osqp_iters, dtype=int)
            diag_eq_inf = eq_inf
            diag_ineq_vio = ineq_vio
            if (
                args.early_stop
                and float(np.max(eq_inf)) <= args.early_stop_eq
                and float(np.max(ineq_vio)) <= args.early_stop_ineq
                and step_norm_mean <= args.early_stop_step
            ):
                break

        result = self._unpack_subp2_results(x_batch)
        result["diag"] = {
            "mode": np.array(["main"] * (N + 1), dtype=object),
            "mode_code": mode_codes,
            "converged": np.ones((N + 1,), dtype=bool),
            "iterations": np.array(diag_iters, dtype=int),
            "eq_inf": np.array(diag_eq_inf, dtype=float),
            "ineq_vio": np.array(diag_ineq_vio, dtype=float),
            "tail_mu": np.full((N + 1,), np.nan, dtype=float),
            "tail_ineq": np.full((N + 1,), np.nan, dtype=float),
            "tail_comp": np.full((N + 1,), np.nan, dtype=float),
            "tail_dual": np.full((N + 1,), np.nan, dtype=float),
        }
        return result

    JAX_Planner.MPC_Planner.jax_ADMM_SubP2 = patched_jax_admm_subp2


def main():
    args = build_parser().parse_args()
    install_sqp_subp2_patch(args)
    t0 = time.perf_counter()
    summary = verify_jax_vs_original.verify_jax_planner(task_idx=args.task_idx, show_plots=args.show_plots)
    total_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[SQP-FULLFLOW] total_ms={total_ms:.3f}")
    print(summary)


if __name__ == "__main__":
    main()
