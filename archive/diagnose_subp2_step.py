import argparse
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp

from ipoptax.solver import LinearSystemFormulation, solve as ipoptax_solve
from scan_subp2_snapshot import (
    build_parser as build_snapshot_parser,
    build_context,
    build_original_step_reference,
    build_step_problem,
    capture_subp2_snapshot,
    pack_original_para2,
    primal_feas_metrics,
)


def build_parser():
    p = argparse.ArgumentParser(description="Diagnose one SubP2 step: IPOPT stats vs JAX trace.")
    p.add_argument("--task-idx", type=int, default=0)
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--max-iter-admm", type=int, default=2)
    p.add_argument("--target-admm-iter", type=int, default=0)
    p.add_argument("--target-step", type=int, default=0)
    p.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="auto")
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
    p.add_argument("--z0-init", type=float, default=1e-3)
    p.add_argument("--soft-stop-enabled", action="store_true")
    p.add_argument("--soft-mu-tol", type=float, default=1e-8)
    p.add_argument("--soft-comp-tol", type=float, default=1e-8)
    p.add_argument("--soft-dual-tol", type=float, default=5e-5)
    p.add_argument("--soft-eq-tol", type=float, default=1e-4)
    p.add_argument("--soft-ineq-tol", type=float, default=5e-3)
    p.add_argument("--plateau-stop-enabled", action="store_true")
    p.add_argument("--plateau-min-iterations", type=int, default=6)
    p.add_argument("--plateau-patience", type=int, default=4)
    p.add_argument("--plateau-phi-tol", type=float, default=5e-7)
    p.add_argument("--plateau-ineq-tol", type=float, default=5e-7)
    p.add_argument("--csv-out", default="")
    p.add_argument("--bench-jax-repeats", type=int, default=2)
    return p


def ensure_backend(backend):
    if backend == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif backend == "cuda":
        os.environ["JAX_PLATFORMS"] = "cuda"


def formulation_from_name(name):
    return getattr(LinearSystemFormulation, name)


def build_original_step_nlp(orig_planner, orig_para2, target_step):
    nxl, nul = orig_planner.nxl, orig_planner.nul
    nxi, nui = orig_planner.nxi, orig_planner.nui
    nq, N = int(orig_planner.nq), orig_planner.N
    n_start_pl = 3 * nxl * (N + 1) + 3 * nul * N + 3 * nxi * nq * (N + 1) + 2 * nui * nq * N + nui * nq
    para_l = orig_para2[n_start_pl : n_start_pl + orig_planner.npl]
    para_i = orig_para2[n_start_pl + orig_planner.npl : n_start_pl + orig_planner.npl + orig_planner.npi]
    a = orig_para2[-1]

    if target_step < N:
        w0 = []
        xl_ref = orig_para2[target_step * nxl : (target_step + 1) * nxl]
        ul_ref = orig_para2[2 * nxl * (N + 1) + target_step * nul : 2 * nxl * (N + 1) + (target_step + 1) * nul]
        w0.extend(xl_ref.tolist())
        w0.extend(ul_ref.tolist())

        n_start_xl = nxl * (N + 1) + nul * N + nxi * nq * (N + 1) + nui * nq
        n_start_scxL = n_start_xl + nxl * (N + 1)
        n_start_ul = n_start_scxL + nxl * (N + 1)
        n_start_scuL = n_start_ul + nul * N
        n_start_xc = n_start_scuL + nul * N
        n_start_scxC = n_start_xc + nxi * nq * (N + 1)
        n_start_uc = n_start_scxC + nxi * nq * (N + 1)
        n_start_scuC = n_start_uc + nui * nq * N

        xl_k = orig_para2[n_start_xl + target_step * nxl : n_start_xl + (target_step + 1) * nxl]
        scxL_k = orig_para2[n_start_scxL + target_step * nxl : n_start_scxL + (target_step + 1) * nxl]
        ul_k = orig_para2[n_start_ul + target_step * nul : n_start_ul + (target_step + 1) * nul]
        scuL_k = orig_para2[n_start_scuL + target_step * nul : n_start_scuL + (target_step + 1) * nul]
        xc_k = orig_para2[n_start_xc + target_step * nxi * nq : n_start_xc + (target_step + 1) * nxi * nq]
        scxC_k = orig_para2[n_start_scxC + target_step * nxi * nq : n_start_scxC + (target_step + 1) * nxi * nq]
        uc_k = orig_para2[n_start_uc + target_step * nui * nq : n_start_uc + (target_step + 1) * nui * nq]
        scuC_k = orig_para2[n_start_scuC + target_step * nui * nq : n_start_scuC + (target_step + 1) * nui * nq]
        xq_ref_k = orig_para2[
            nxl * (N + 1) + nul * N + target_step * nxi * nq :
            nxl * (N + 1) + nul * N + (target_step + 1) * nxi * nq
        ]
        ui_ref_block = orig_para2[
            nxl * (N + 1) + nul * N + nxi * nq * (N + 1) :
            nxl * (N + 1) + nul * N + nxi * nq * (N + 1) + nui * nq
        ]
        for i in range(nq):
            w0.extend(xq_ref_k[i * nxi : (i + 1) * nxi].tolist())
            w0.extend(ui_ref_block[i * nui : (i + 1) * nui].tolist())

        p = np.concatenate([xl_k, scxL_k, ul_k, scuL_k, xc_k, scxC_k, uc_k, scuC_k, para_l, para_i, [a]])
        return {
            "x0": np.array(w0, dtype=float),
            "p": p,
            "lbx": orig_planner.lbw2,
            "ubx": orig_planner.ubw2,
            "lbg": orig_planner.lbg2,
            "ubg": orig_planner.ubg2,
            "solver": orig_planner.solver2,
            "kind": "path",
        }

    w0 = []
    xl_ref = orig_para2[N * nxl : (N + 1) * nxl]
    w0.extend(xl_ref.tolist())
    xq_ref_N = orig_para2[
        nxl * (N + 1) + nul * N + nxi * nq * N :
        nxl * (N + 1) + nul * N + nxi * nq * (N + 1)
    ]
    for i in range(nq):
        w0.extend(xq_ref_N[i * nxi : (i + 1) * nxi].tolist())

    n_start_xl = nxl * (N + 1) + nul * N + nxi * nq * (N + 1) + nui * nq
    n_start_scxL = n_start_xl + nxl * (N + 1)
    n_start_xc = n_start_scxL + nxl * (N + 1) + 2 * nul * N
    n_start_scxC = n_start_xc + nxi * nq * (N + 1)
    xl_N = orig_para2[n_start_xl + N * nxl : n_start_xl + (N + 1) * nxl]
    scxL_N = orig_para2[n_start_scxL + N * nxl : n_start_scxL + (N + 1) * nxl]
    xc_N = orig_para2[n_start_xc + N * nxi * nq : n_start_xc + (N + 1) * nxi * nq]
    scxC_N = orig_para2[n_start_scxC + N * nxi * nq : n_start_scxC + (N + 1) * nxi * nq]
    p = np.concatenate([xl_N, scxL_N, xc_N, scxC_N, para_l, para_i, [a]])
    return {
        "x0": np.array(w0, dtype=float),
        "p": p,
        "lbx": orig_planner.lbw2N,
        "ubx": orig_planner.ubw2N,
        "lbg": orig_planner.lbg2N,
        "ubg": orig_planner.ubg2N,
        "solver": orig_planner.solver2N,
        "kind": "terminal",
    }


def summarize_ipopt_stats(stats):
    out = {
        "success": stats.get("success"),
        "return_status": stats.get("return_status"),
        "iter_count": stats.get("iter_count"),
        "t_wall_total": stats.get("t_wall_total"),
    }
    return out


def main():
    args = build_parser().parse_args()
    snapshot_defaults = build_snapshot_parser().parse_args([])
    for key, value in vars(snapshot_defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    ensure_backend(args.backend)
    t0 = time.perf_counter()

    def log(msg):
        print(f"[LOG +{time.perf_counter() - t0:7.2f}s] {msg}", flush=True)

    log("Building planner/task context")
    ctx = build_context(args)
    log(f"Capturing snapshot at admm_iter={args.target_admm_iter}")
    para2 = capture_subp2_snapshot(ctx, args.target_admm_iter, args.max_iter_admm)
    planner = ctx["planner"]
    orig_planner = ctx["orig_planner"]

    dims, x_init, params_t = build_step_problem(planner, para2, args.target_step)
    orig_para2 = pack_original_para2(para2, orig_planner)
    orig_step_nlp = build_original_step_nlp(orig_planner, orig_para2, args.target_step)
    log(f"Running original IPOPT step solver ({orig_step_nlp['kind']})")
    t_orig = time.perf_counter()
    sol_orig = orig_step_nlp["solver"](
        x0=orig_step_nlp["x0"],
        lbx=orig_step_nlp["lbx"],
        ubx=orig_step_nlp["ubx"],
        p=orig_step_nlp["p"],
        lbg=orig_step_nlp["lbg"],
        ubg=orig_step_nlp["ubg"],
    )
    t_orig_ms = (time.perf_counter() - t_orig) * 1000.0
    orig_stats = summarize_ipopt_stats(orig_step_nlp["solver"].stats())
    orig_step_ref = build_original_step_reference(
        orig_planner.ADMM_SubP2(orig_para2),
        planner,
        args.target_step,
    )

    log("Running JAX step solver with trace collection")
    c_raw = lambda x: planner.ipoptax_equality(x, params_t, dims)
    g_raw = lambda x: planner.ipoptax_inequality(x, params_t, dims)
    f = lambda x: planner.ipoptax_objective(x, params_t, dims)
    c_scale = params_t["eq_scale"]
    g_scale = params_t["ineq_scale"]
    c = lambda x: c_scale * c_raw(x)
    g = lambda x: g_scale * g_raw(x)

    eps_s = 1e-3
    c0 = c(x_init)
    g0 = g(x_init)
    ws_s0 = jnp.minimum(jnp.maximum(-g0 + eps_s, eps_s), args.s0_cap)
    ws_z0 = jnp.full_like(ws_s0, args.z0_init)
    ws_y0 = jnp.zeros_like(c0)
    solve_kwargs = dict(
        f=f,
        c=c,
        g=g,
        ws_x=x_init,
        ws_s=ws_s0,
        ws_y=ws_y0,
        ws_z=ws_z0,
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
        soft_stop_enabled=args.soft_stop_enabled,
        soft_mu_tol=args.soft_mu_tol,
        soft_comp_tol=args.soft_comp_tol,
        soft_dual_tol=args.soft_dual_tol,
        soft_eq_tol=args.soft_eq_tol,
        soft_ineq_tol=args.soft_ineq_tol,
        plateau_stop_enabled=args.plateau_stop_enabled,
        plateau_min_iterations=args.plateau_min_iterations,
        plateau_patience=args.plateau_patience,
        plateau_phi_tol=args.plateau_phi_tol,
        plateau_ineq_tol=args.plateau_ineq_tol,
        print_logs=False,
    )

    jax_trace_runs_ms = []
    res = None
    for i_run in range(max(args.bench_jax_repeats, 1)):
        t_jax = time.perf_counter()
        res_i = ipoptax_solve(
            **solve_kwargs,
            trace_length=args.max_solver_iters,
        )
        jax.block_until_ready(res_i["x"])
        dt_ms = (time.perf_counter() - t_jax) * 1000.0
        jax_trace_runs_ms.append(dt_ms)
        if i_run == 0:
            res = res_i
    assert res is not None

    jax_notrace_runs_ms = []
    for i_run in range(max(args.bench_jax_repeats, 1)):
        t_jax = time.perf_counter()
        res_no_trace = ipoptax_solve(
            **solve_kwargs,
            trace_length=0,
        )
        jax.block_until_ready(res_no_trace["x"])
        dt_ms = (time.perf_counter() - t_jax) * 1000.0
        jax_notrace_runs_ms.append(dt_ms)
    x_sol = np.array(res["x"])
    eq_inf, ineq_vio, feas = primal_feas_metrics(planner, x_sol, params_t, dims)
    diff = x_sol - orig_step_ref
    trace_rows = []
    n_trace = min(int(np.array(res["iteration"])), len(np.array(res["trace_alpha"])))
    for i in range(n_trace):
        trace_rows.append(
            {
                "iter": i,
                "alpha": float(np.array(res["trace_alpha"])[i]),
                "mu": float(np.array(res["trace_mu"])[i]),
                "phi": float(np.array(res["trace_phi"])[i]),
                "merit": float(np.array(res["trace_merit"])[i]),
                "eq_norm": float(np.array(res["trace_eq"])[i]),
                "ineq_norm": float(np.array(res["trace_ineq"])[i]),
                "comp_inf": float(np.array(res["trace_comp"])[i]),
                "dual_inf": float(np.array(res["trace_dual"])[i]),
                "accept_code": int(round(float(np.array(res["trace_accept"])[i]))),
                "linsys": float(np.array(res["trace_linsys"])[i]),
            }
        )

    print("")
    print("IPOPT step stats:")
    for k, v in orig_stats.items():
        print(f"  {k}: {v}")
    print(f"  wall_ms_measured: {t_orig_ms:.3f}")
    print("")
    print("JAX final stats:")
    print(f"  wall_ms_trace_runs: {jax_trace_runs_ms}")
    print(f"  wall_ms_notrace_runs: {jax_notrace_runs_ms}")
    print(f"  converged: {bool(np.array(res['converged']))}")
    print(f"  iterations: {int(np.array(res['iteration']))}")
    print(f"  eq_inf: {eq_inf:.6e}")
    print(f"  ineq_vio: {ineq_vio:.6e}")
    print(f"  feas: {feas:.6e}")
    print(f"  orig_rmse: {float(np.sqrt(np.mean(diff**2))):.6e}")
    print(f"  orig_max_abs: {float(np.max(np.abs(diff))):.6e}")
    print("")
    print("JAX trace:")
    for row in trace_rows[: min(len(trace_rows), 20)]:
        print(
            f"  it={row['iter']:02d} alpha={row['alpha']:.3e} mu={row['mu']:.3e} "
            f"phi={row['phi']:.3e} eq={row['eq_norm']:.3e} ineq={row['ineq_norm']:.3e} "
            f"comp={row['comp_inf']:.3e} dual={row['dual_inf']:.3e} "
            f"accept={row['accept_code']} linsys={row['linsys']:.3e}"
        )

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as fcsv:
            writer = csv.DictWriter(
                fcsv,
                fieldnames=[
                    "iter",
                    "alpha",
                    "mu",
                    "phi",
                    "merit",
                    "eq_norm",
                    "ineq_norm",
                    "comp_inf",
                    "dual_inf",
                    "accept_code",
                    "linsys",
                ],
            )
            writer.writeheader()
            for row in trace_rows:
                writer.writerow(row)
        print(f"\nWrote JAX trace CSV to {args.csv_out}")


if __name__ == "__main__":
    main()
