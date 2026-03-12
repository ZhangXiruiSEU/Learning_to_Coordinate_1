"""
Scan JAX SubP2 main hyper-parameters using full ADMM forward error.

Ranking targets the true end goal:
1. final trajectory/control deviation from the original planner
2. SubP2 feasibility diagnostics accumulated over the full forward run
"""

import argparse
import csv
import itertools
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import Dynamics_load_cable_autotuning_2nd_COM_Dyn as Original_Dyn
import Kinodynamic_Planning_ADMM_quaternion_DDP_autotuning_2nd_COM_Dyn as Original_Planner
import Neural_network


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--task-idx", type=int, default=0)
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--admm-iters", type=int, default=2)
    p.add_argument("--initial-model", type=int, default=4)
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--out-csv", type=str, default="")
    p.add_argument("--tau-min-list", type=str, default="0.99")
    p.add_argument("--mu-min-list", type=str, default="1e-8")
    p.add_argument("--min-delta-list", type=str, default="1e-4,1e-3")
    p.add_argument("--gamma-y-list", type=str, default="1e-4,1e-2")
    p.add_argument("--gamma-z-list", type=str, default="1e-2")
    p.add_argument("--line-search-factor-list", type=str, default="0.2")
    p.add_argument("--line-search-min-step-list", type=str, default="1e-10")
    p.add_argument("--s0-cap-list", type=str, default="0.5")
    p.add_argument("--z0-init-list", type=str, default="1e-3")
    return p


def parse_float_list(raw):
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"empty float list: {raw!r}")
    return vals


def iter_grid(args):
    names_and_vals = [
        ("tau_min", parse_float_list(args.tau_min_list)),
        ("mu_min", parse_float_list(args.mu_min_list)),
        ("min_delta", parse_float_list(args.min_delta_list)),
        ("gamma_y", parse_float_list(args.gamma_y_list)),
        ("gamma_z", parse_float_list(args.gamma_z_list)),
        ("line_search_factor", parse_float_list(args.line_search_factor_list)),
        ("line_search_min_step_size", parse_float_list(args.line_search_min_step_list)),
        ("s0_cap", parse_float_list(args.s0_cap_list)),
        ("z0_init", parse_float_list(args.z0_init_list)),
    ]
    keys = [k for k, _ in names_and_vals]
    vals = [v for _, v in names_and_vals]
    for idx, combo in enumerate(itertools.product(*vals)):
        if args.max_cases > 0 and idx >= args.max_cases:
            break
        yield dict(zip(keys, combo))


def convert_nn_output(col, dim):
    row = np.zeros((1, dim))
    for i in range(dim):
        row[0, i] = col[i, 0]
    return row


def build_fullflow_context(args):
    import JustWorkingOnIt as JAX_Planner

    m1, m2 = 0.45, 0.25
    mtot = m1 + m2
    nq = 4
    mq = 0.25
    fqmax = 0.75 * 9.81
    cl0 = 1.0
    rq, rl, ro = 0.15, 0.25, 0.65
    dt = 0.04
    pob1, pob2 = np.array([[1.7, 1.15]]).T, np.array([[0.3, 3.05]]).T
    max_radius = 0.15
    k_const = 10.0

    sysm_para = np.array([
        m1, m2,
        1 / 4 * m1 * rl**2, 1 / 4 * m1 * rl**2, 1 / 2 * m1 * rl**2,
        rl, nq, rq, mq, fqmax,
        cl0, ro,
    ])

    planner = JAX_Planner.MPC_Planner(sysm_para, dt, args.horizon)
    planner.verbose = False
    planner.pob1 = np.array([pob1[0, 0], pob1[1, 0], 0.0])
    planner.pob2 = np.array([pob2[0, 0], pob2[1, 0], 0.0])

    sysm = Original_Dyn.multilift_model(sysm_para, dt)

    rp_task = np.load("trained_data_meta_COM_Dyn/rp_task.npy")[args.task_idx]
    rg_task = (m2 / mtot) * rp_task
    if rp_task.ndim == 1:
        rp_task = rp_task.reshape(3, 1)
    if rg_task.ndim == 1:
        rg_task = rg_task.reshape(3, 1)

    planner.rg = rg_task
    planner.allocation_martrix(rg_task)

    sysm.Rotational_Inertia(rp_task)
    sysm.model()
    orig = Original_Planner.MPC_Planner(sysm_para, dt, args.horizon)
    orig.Rotational_Inertia(rp_task)
    orig.allocation_martrix(rg_task)
    orig.SetStateVariables(sysm.xl, sysm.xi)
    orig.SetCtrlVariables(sysm.ul, sysm.ui)
    orig.SetDyns(sysm.model_l, sysm.model_i)
    orig.SetWeightPara()
    orig.SetPayloadCostDyn(args.admm_iters)
    orig.SetCableCostDyn(args.admm_iters)
    orig.SetConstriants(pob1, pob2)
    orig.SetADMMSubP2_SoftCost_k()
    orig.SetADMMSubP2_SoftCost_N()
    orig.ADMM_SubP2_Init()
    orig.ADMM_SubP2_N_Init()
    orig.Load_derivatives_DDP_ADMM()
    orig.Cable_derivatives_DDP_ADMM()
    orig.system_derivatives_SubP2_ADMM_k()
    orig.system_derivatives_SubP2_ADMM_N()
    orig.system_derivatives_SubP3_ADMM()

    path_l = f"trained_data_multiagent_meta_COM_Dyn/trained_nn_l_{args.initial_model}_3_n.pt"
    path_i = f"trained_data_multiagent_meta_COM_Dyn/trained_nn_i_{args.initial_model}_3_n.pt"
    nn_l = torch.load(path_l, map_location=torch.device("cpu"), weights_only=False)
    nn_i = torch.load(path_i, map_location=torch.device("cpu"), weights_only=False)

    dummy_x = np.zeros((13, 1))
    dummy_u = np.zeros((6, 1))
    dummy_para = np.zeros((1, planner.n_Pauto))
    gs = JAX_Planner.Gradient_Solver(
        sysm_para, args.horizon,
        dummy_x, dummy_u, dummy_x, dummy_u,
        np.zeros((planner.nxi, 1)), np.zeros((planner.nui, 1)),
        np.zeros((planner.nxi, 1)), np.zeros((planner.nui, 1)),
        dummy_para, np.zeros((1, planner.npl)), np.zeros((1, planner.npi))
    )

    nn_input = np.reshape(rg_task[0:2] / max_radius * k_const, (2, 1))
    with torch.no_grad():
        out_l = nn_l(torch.FloatTensor(nn_input)).numpy()
        out_i = nn_i(torch.FloatTensor(nn_input)).numpy()
    p_weight1 = gs.Set_Parameters_nn_l(convert_nn_output(out_l, planner.npl))
    p_weight2 = gs.Set_Parameters_nn_i(convert_nn_output(out_i, planner.npi))

    coeffx = np.zeros((2, 8))
    coeffy = np.zeros((2, 8))
    coeffz = np.zeros((2, 8))
    for k in range(2):
        coeffx[k, :] = np.load(f"Reference_traj_4/coeffx{k+1}.npy")
        coeffy[k, :] = np.load(f"Reference_traj_4/coeffy{k+1}.npy")
        coeffz[k, :] = np.load(f"Reference_traj_4/coeffz{k+1}.npy")

    ref_xl_mat = np.zeros((args.horizon + 1, planner.nxl))
    ref_ul_mat = np.zeros((args.horizon, planner.nul))
    t = 0.0
    for k in range(args.horizon):
        ref_x_k, ref_u_k = sysm.minisnap_load_circle(coeffx, coeffy, coeffz, t, rg_task)
        ref_xl_mat[k, :] = ref_x_k.flatten()
        ref_ul_mat[k, :] = ref_u_k.flatten()
        t += dt
    ref_xl_mat[args.horizon, :] = ref_xl_mat[args.horizon - 1, :]
    ref_xl = ref_xl_mat.reshape(-1)
    ref_ul = ref_ul_mat.reshape(-1)

    try:
        xl_init_file = np.load(f"trained_data_meta_COM_Dyn/xl_init_{args.admm_iters}.npy")
        xl_init_task = np.zeros(planner.nxl)
        xl_init_task[0] = float(xl_init_file[0]) + float(rg_task[0, 0])
        xl_init_task[1] = float(xl_init_file[1]) + float(rg_task[1, 0])
        xl_init_task[2:] = xl_init_file[2:]
    except FileNotFoundError:
        xl_init_task = ref_xl_mat[0, :].copy()

    ref_uq = np.zeros(nq * planner.nui)
    xq_init_list = []
    ref_xq_list = []
    for _ in range(nq):
        xi_0 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.81 * mtot / nq, 0], dtype=float)
        xq_init_list.append(xi_0)
        ref_xq_list.append(np.tile(xi_0, args.horizon + 1))

    return {
        "planner": planner,
        "orig_planner": orig,
        "ref_xl": ref_xl,
        "ref_ul": ref_ul,
        "ref_xq": ref_xq_list,
        "ref_uq": ref_uq,
        "xl_fb": xl_init_task,
        "xq_fb": np.concatenate(xq_init_list),
        "paral": p_weight1,
        "parac": p_weight2,
    }


@contextmanager
def subp2_env(cfg):
    mapping = {
        "JAX_SUBP2_MAIN_TAU_MIN": cfg["tau_min"],
        "JAX_SUBP2_MAIN_MU_MIN": cfg["mu_min"],
        "JAX_SUBP2_MAIN_MIN_DELTA": cfg["min_delta"],
        "JAX_SUBP2_MAIN_GAMMA_Y": cfg["gamma_y"],
        "JAX_SUBP2_MAIN_GAMMA_Z": cfg["gamma_z"],
        "JAX_SUBP2_MAIN_LINE_SEARCH_FACTOR": cfg["line_search_factor"],
        "JAX_SUBP2_MAIN_LINE_SEARCH_MIN_STEP": cfg["line_search_min_step_size"],
        "JAX_SUBP2_S0_CAP": cfg["s0_cap"],
        "JAX_SUBP2_Z0_INIT": cfg["z0_init"],
    }
    old = {k: os.environ.get(k) for k in mapping}
    try:
        for k, v in mapping.items():
            os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def diff_stats(a, b):
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.max(np.abs(diff))), float(np.sqrt(np.mean(diff**2)))


def aggregate_subp2_diag(diag_history):
    modes = []
    conv = []
    iters = []
    eqs = []
    ineqs = []
    for diag in diag_history:
        if not diag:
            continue
        modes.extend(list(diag.get("mode", [])))
        conv.extend(list(np.asarray(diag.get("converged", []), dtype=bool)))
        iters.extend(list(np.asarray(diag.get("iterations", []), dtype=int)))
        eqs.extend(list(np.asarray(diag.get("eq_inf", []), dtype=float)))
        ineqs.extend(list(np.asarray(diag.get("ineq_vio", []), dtype=float)))
    return {
        "subp2_steps": len(modes),
        "subp2_nonconv": int(sum(not x for x in conv)),
        "subp2_max_iters": int(max(iters) if iters else 0),
        "subp2_max_eq_inf": float(max(eqs) if eqs else 0.0),
        "subp2_max_ineq_vio": float(max(ineqs) if ineqs else 0.0),
        "subp2_fallback_count": int(sum(m == "fallback" for m in modes)),
        "subp2_bestfeas_count": int(sum("bestfeas" in str(m) or str(m) == "best-feas" for m in modes)),
    }


def run_original_once(ctx, admm_iters):
    t0 = time.perf_counter()
    sol, *_ = ctx["orig_planner"].ADMM_forward_MPC(
        Ref_xl=ctx["ref_xl"],
        Ref_ul=ctx["ref_ul"],
        ref_xq=ctx["ref_xq"],
        ref_uq=ctx["ref_uq"],
        xl_fb=ctx["xl_fb"],
        xq_fb=ctx["xq_fb"],
        paral=ctx["paral"],
        paraC=ctx["parac"],
        max_iter_ADMM=admm_iters,
    )
    return sol, time.perf_counter() - t0


def run_jax_case(ctx, admm_iters, cfg):
    t0 = time.perf_counter()
    with subp2_env(cfg):
        sol = ctx["planner"].jax_ADMM_forward_MPC(
            Ref_xl=ctx["ref_xl"],
            Ref_ul=ctx["ref_ul"],
            ref_xq=ctx["ref_xq"],
            ref_uq=ctx["ref_uq"],
            xl_fb=ctx["xl_fb"],
            xq_fb=ctx["xq_fb"],
            paral=ctx["paral"],
            parac=ctx["parac"],
            max_iter_ADMM=admm_iters,
        )
    return sol, time.perf_counter() - t0


def score_row(row):
    return (
        row["load_control_rmse"],
        row["cable_state_rmse"],
        row["load_state_rmse"],
        row["subp2_max_ineq_vio"],
        row["subp2_max_eq_inf"],
        row["wall_time_s"],
    )


def print_top(rows):
    print("")
    print("Top cases:")
    for i, row in enumerate(rows[:10], start=1):
        print(
            f"{i:02d} | ul_rmse={row['load_control_rmse']:.3e} | "
            f"xc_rmse={row['cable_state_rmse']:.3e} | xl_rmse={row['load_state_rmse']:.3e} | "
            f"max_ineq={row['subp2_max_ineq_vio']:.3e} | max_eq={row['subp2_max_eq_inf']:.3e} | "
            f"bestfeas={row['subp2_bestfeas_count']} | fallback={row['subp2_fallback_count']} | "
            f"tau={row['tau_min']:.2g} mu={row['mu_min']:.1e} d={row['min_delta']:.1e} "
            f"gy={row['gamma_y']:.1e} gz={row['gamma_z']:.1e} "
            f"ls={row['line_search_factor']:.2g} min_ls={row['line_search_min_step_size']:.1e} "
            f"s0={row['s0_cap']:.2g} z0={row['z0_init']:.1e}"
        )


def write_csv(path, rows):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = build_parser().parse_args()
    t0 = time.perf_counter()

    def log(msg):
        print(f"[LOG +{time.perf_counter()-t0:7.2f}s] {msg}", flush=True)

    import JustWorkingOnIt  # noqa: F401

    log("Building fullflow context")
    ctx = build_fullflow_context(args)
    log("Running original planner once")
    orig_sol, orig_wall = run_original_once(ctx, args.admm_iters)
    log(f"Original forward finished in {orig_wall:.2f}s")

    rows = []
    cases = list(iter_grid(args))
    total = len(cases)
    log(f"Scanning {total} fullflow cases")
    for idx, cfg in enumerate(cases, start=1):
        case_t0 = time.perf_counter()
        jax_sol, wall = run_jax_case(ctx, args.admm_iters, cfg)
        xl_max, xl_rmse = diff_stats(jax_sol["load_trajectory"], orig_sol["xl_traj"])
        ul_max, ul_rmse = diff_stats(jax_sol["control_inputs"], orig_sol["ul_traj"])
        xc_max, xc_rmse = diff_stats(jax_sol["cable_trajectories"], orig_sol["xc_traj"])
        row = {
            **cfg,
            "wall_time_s": float(wall),
            "load_state_max_abs": xl_max,
            "load_state_rmse": xl_rmse,
            "load_control_max_abs": ul_max,
            "load_control_rmse": ul_rmse,
            "cable_state_max_abs": xc_max,
            "cable_state_rmse": xc_rmse,
        }
        row.update(aggregate_subp2_diag(jax_sol.get("subp2_diag_history", [])))
        rows.append(row)

        if idx == 1 or idx == total or (args.progress_every > 0 and idx % args.progress_every == 0):
            elapsed = time.perf_counter() - t0
            best = min(rows, key=score_row)
            avg = elapsed / idx
            eta = avg * (total - idx)
            log(
                f"Scanned {idx}/{total} | case_time={time.perf_counter()-case_t0:.2f}s | "
                f"last_ul_rmse={row['load_control_rmse']:.3e} last_xc_rmse={row['cable_state_rmse']:.3e} | "
                f"best_ul_rmse={best['load_control_rmse']:.3e} best_xc_rmse={best['cable_state_rmse']:.3e} | "
                f"eta={eta:.1f}s"
            )

    rows.sort(key=score_row)
    print_top(rows)
    out_csv = args.out_csv or f"subp2_fullflow_scan_task{args.task_idx}_h{args.horizon}_a{args.admm_iters}.csv"
    write_csv(out_csv, rows)
    log(f"Wrote CSV: {out_csv}")


if __name__ == "__main__":
    main()
