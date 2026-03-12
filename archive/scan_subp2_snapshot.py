"""
Scan ipoptax SubP2 hyper-parameters on a fixed single-step snapshot.

Workflow:
1. Rebuild the same planner/task/reference context as the JAX forward path.
2. Advance ADMM until a chosen iteration and capture the SubP2 input snapshot.
3. Extract one time-step NLP from that snapshot.
4. Run a Cartesian grid over selected ipoptax parameters and rank results by feasibility.
"""

import argparse
import csv
import itertools
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import Dynamics_load_cable_autotuning_2nd_COM_Dyn
import Kinodynamic_Planning_ADMM_quaternion_DDP_autotuning_2nd_COM_Dyn as Original_Planner
import Neural_network  # required for torch.load class resolution
from ipoptax.solver import LinearSystemFormulation, solve as ipoptax_solve
from run_jax_forward_with_trained_nn import (
    build_initial_cable_state,
    build_references,
    convert_nn_output,
    ensure_jax_backend,
    select_stage1_reference,
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--initial-model", type=int, default=4)
    parser.add_argument("--max-iter-admm", type=int, default=3)
    parser.add_argument("--initial-model-stage1", type=int, default=4)
    parser.add_argument("--max-iter-admm-stage1", type=int, default=3)
    parser.add_argument("--weight-mode-stage1", type=str, default="n", choices=["n", "f"])
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "metal", "cpu", "cuda"])
    parser.add_argument("--execution-mode", type=str, default="serial", choices=["serial", "batched"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-admm-iter", type=int, default=0, help="0-based ADMM iter to snapshot before SubP2")
    parser.add_argument("--target-step", type=int, default=0, help="0-based time step inside the SubP2 batch")
    parser.add_argument("--max-cases", type=int, default=0, help="truncate the grid to the first N cases; 0 means all")
    parser.add_argument("--solver-formulation", type=str, default="stable_direct_4x4")
    parser.add_argument("--max-solver-iters", type=int, default=120)
    parser.add_argument("--max-kkt-violation", type=float, default=1e-3)
    parser.add_argument("--tau-min-list", type=str, default="0.99,0.95")
    parser.add_argument("--mu-min-list", type=str, default="1e-8,1e-6")
    parser.add_argument("--min-delta-list", type=str, default="1e-4,1e-3")
    parser.add_argument("--gamma-y-list", type=str, default="1e-4,1e-2")
    parser.add_argument("--gamma-z-list", type=str, default="1e-4,1e-2")
    parser.add_argument("--line-search-factor-list", type=str, default="0.2,0.1")
    parser.add_argument("--line-search-min-step-list", type=str, default="1e-10,1e-12")
    parser.add_argument("--s0-cap-list", type=str, default="0.5,1.0")
    parser.add_argument("--z0-init-list", type=str, default="1e-3,1e-2")
    parser.add_argument("--armijo-factor", type=float, default=1e-4)
    parser.add_argument("--feasibility-armijo-factor", type=float, default=1e-4)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--out-csv", type=str, default="")
    return parser


def parse_float_list(raw):
    vals = []
    for part in raw.split(","):
        item = part.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise ValueError(f"empty list: {raw!r}")
    return vals


def formulation_from_name(name):
    key = name.strip().lower()
    mapping = {
        "autodiff": LinearSystemFormulation.AUTODIFF,
        "stable_direct_4x4": LinearSystemFormulation.STABLE_DIRECT_4x4,
        "symmetric_direct_4x4": LinearSystemFormulation.SYMMETRIC_DIRECT_4x4,
        "symmetric_indirect_3x3": LinearSystemFormulation.SYMMETRIC_INDIRECT_3x3,
        "symmetric_indirect_2x2": LinearSystemFormulation.SYMMETRIC_INDIRECT_2x2,
    }
    if key not in mapping:
        raise ValueError(f"unsupported formulation: {name}")
    return mapping[key]


def build_context(args):
    import JustWorkingOnIt as JAX_Planner

    m1, m2 = 0.45, 0.25
    mtot = m1 + m2
    nq = 4
    mq = 0.25
    fqmax = 0.75 * 9.81
    cl0, rq, rl, ro = 1.0, 0.15, 0.25, 0.65
    dt = 0.04
    pob1, pob2 = np.array([[1.7, 1.15]]).T, np.array([[0.3, 3.05]]).T
    max_radius, k_const = 0.15, 10.0

    sysm_para = np.array(
        [
            m1,
            m2,
            1 / 4 * m1 * rl**2,
            1 / 4 * m1 * rl**2,
            1 / 2 * m1 * rl**2,
            rl,
            nq,
            rq,
            mq,
            fqmax,
            cl0,
            ro,
        ]
    )

    sysm = Dynamics_load_cable_autotuning_2nd_COM_Dyn.multilift_model(sysm_para, dt)
    rp0 = np.array([[0.05, 0.05, 0.0]]).T
    sysm.Rotational_Inertia(rp0)
    sysm.model()

    planner = JAX_Planner.MPC_Planner(sysm_para, dt, args.horizon)
    planner.verbose = False
    planner.pob1 = np.array([pob1[0, 0], pob1[1, 0], 0.0], dtype=np.float32)
    planner.pob2 = np.array([pob2[0, 0], pob2[1, 0], 0.0], dtype=np.float32)

    rp_task_all = np.load("trained_data_meta_COM_Dyn/rp_task.npy")
    rp_task = rp_task_all[args.task_idx]
    rg_task = m2 / mtot * rp_task
    if rp_task.ndim == 1:
        rp_task = rp_task.reshape(3, 1)
    if rg_task.ndim == 1:
        rg_task = rg_task.reshape(3, 1)

    sysm.Rotational_Inertia(rp_task)
    sysm.model()
    planner.Rotational_Inertia(rp_task)
    planner.Jl_inv = np.linalg.inv(np.array(planner.Jl))
    planner.rg = rg_task
    planner.allocation_martrix(rg_task)

    orig_planner = Original_Planner.MPC_Planner(sysm_para, dt, args.horizon)
    orig_planner.Rotational_Inertia(rp_task)
    orig_planner.allocation_martrix(rg_task)
    orig_planner.SetStateVariables(sysm.xl, sysm.xi)
    orig_planner.SetCtrlVariables(sysm.ul, sysm.ui)
    orig_planner.SetDyns(sysm.model_l, sysm.model_i)
    orig_planner.SetWeightPara()
    orig_planner.SetPayloadCostDyn(args.max_iter_admm)
    orig_planner.SetCableCostDyn(args.max_iter_admm)
    orig_planner.SetConstriants(pob1, pob2)
    orig_planner.SetADMMSubP2_SoftCost_k()
    orig_planner.SetADMMSubP2_SoftCost_N()
    orig_planner.ADMM_SubP2_Init()
    orig_planner.ADMM_SubP2_N_Init()

    di_path, ti_path = select_stage1_reference(args)
    di_ref = np.load(di_path)
    ti_ref = np.load(ti_path)
    di_ref = di_ref[:, :, : args.horizon + 1]
    ti_ref = ti_ref[:, : args.horizon + 1]

    path_l = f"trained_data_multiagent_meta_COM_Dyn/trained_nn_l_{args.initial_model}_{args.max_iter_admm}_n.pt"
    path_i = f"trained_data_multiagent_meta_COM_Dyn/trained_nn_i_{args.initial_model}_{args.max_iter_admm}_n.pt"
    nn_l = torch.load(path_l, map_location=torch.device("cpu"), weights_only=False)
    nn_i = torch.load(path_i, map_location=torch.device("cpu"), weights_only=False)

    dummy_xl = np.zeros((planner.nxl, 1))
    dummy_ul = np.zeros((planner.nul, 1))
    dummy_xi = np.zeros((planner.nxi, 1))
    dummy_ui = np.zeros((planner.nui, 1))
    dummy_pauto = np.zeros((1, planner.n_Pauto))
    dummy_w1 = np.zeros((1, planner.npl))
    dummy_w2 = np.zeros((1, planner.npi))
    grad_solver = JAX_Planner.Gradient_Solver(
        sysm_para,
        args.horizon,
        dummy_xl,
        dummy_ul,
        dummy_xl,
        dummy_ul,
        dummy_xi,
        dummy_ui,
        dummy_xi,
        dummy_ui,
        dummy_pauto,
        dummy_w1,
        dummy_w2,
    )

    nn_input = np.reshape(rg_task[0:2] / max_radius * k_const, (2, 1))
    nn_input_t = torch.FloatTensor(nn_input)
    with torch.no_grad():
        nn_l_out = convert_nn_output(nn_l(nn_input_t).numpy(), planner.npl)
        nn_i_out = convert_nn_output(nn_i(nn_input_t).numpy(), planner.npi)
    p_weight1 = grad_solver.Set_Parameters_nn_l(nn_l_out)
    p_weight2 = grad_solver.Set_Parameters_nn_i(nn_i_out)

    coeffx = np.zeros((2, 8))
    coeffy = np.zeros((2, 8))
    coeffz = np.zeros((2, 8))
    for k in range(2):
        coeffx[k, :] = np.load(f"Reference_traj_4/coeffx{k+1}.npy")
        coeffy[k, :] = np.load(f"Reference_traj_4/coeffy{k+1}.npy")
        coeffz[k, :] = np.load(f"Reference_traj_4/coeffz{k+1}.npy")

    ref_xl, ref_ul, ref_xc = build_references(
        sysm,
        coeffx,
        coeffy,
        coeffz,
        rg_task,
        di_ref,
        ti_ref,
        args.horizon,
        sysm.nxl,
        sysm.nul,
        sysm.nxi,
        nq,
    )
    ref_uq = np.zeros(int(nq) * sysm.nui)
    xq_init = build_initial_cable_state(di_ref, ti_ref, sysm.nxi, nq)

    xl_init = np.load(f"trained_data_meta_COM_Dyn/xl_init_{args.max_iter_admm}.npy")
    xl_init_task = np.zeros(sysm.nxl)
    xl_init_task[0] = float(xl_init[0]) + float(rg_task[0, 0])
    xl_init_task[1] = float(xl_init[1]) + float(rg_task[1, 0])
    xl_init_task[2:sysm.nxl] = xl_init[2:sysm.nxl]

    return {
        "planner": planner,
        "orig_planner": orig_planner,
        "ref_xl": ref_xl,
        "ref_ul": ref_ul,
        "ref_xc": ref_xc,
        "ref_uq": ref_uq,
        "xl_init_task": xl_init_task,
        "xq_init": xq_init,
        "p_weight1": p_weight1,
        "p_weight2": p_weight2,
    }


def capture_subp2_snapshot(ctx, target_admm_iter, total_admm_iters):
    planner = ctx["planner"]
    ref_xl = ctx["ref_xl"]
    ref_ul = ctx["ref_ul"]
    ref_xc = ctx["ref_xc"]
    ref_uq = ctx["ref_uq"]
    xl_init_task = ctx["xl_init_task"]
    xq_init = ctx["xq_init"]
    p_weight1 = ctx["p_weight1"]
    p_weight2 = ctx["p_weight2"]

    planner.max_iter_ADMM = max(total_admm_iters, 1)
    scxl_traj, scul_traj, scxc_traj, scuc_traj = planner._initialize_trajectories(ref_xl, ref_ul, ref_xc, ref_uq)
    y_xl = np.zeros_like(scxl_traj)
    y_ul = np.zeros_like(scul_traj)
    y_xc = np.zeros_like(scxc_traj)
    y_uc = np.zeros_like(scuc_traj)

    for i_admm in range(target_admm_iter + 1):
        para_l = planner._pack_paraL(xl_init_task, ref_xl, ref_ul, p_weight1, scxl_traj, scul_traj, y_xl, y_ul, i_admm)
        para_c = planner._pack_paraC(xq_init, ref_xc, ref_uq, p_weight2, scxc_traj, scuc_traj, y_xc, y_uc, i_admm)
        sol1_load, _ = planner.jax_MPC_Load_DDP_Planning_SubP1(para_l)
        sol1_cable, _ = planner.jax_MPC_Cable_DDP_Planning_SubP1(para_c)

        xl_opt = sol1_load["xl_traj"][0]
        ul_opt = sol1_load["ul_traj"][0]
        xc_opt = np.array(sol1_cable["xc_traj"])
        uc_opt = np.array(sol1_cable["uc_traj"])

        para2 = planner._pack_para2(
            xl_opt,
            ul_opt,
            xc_opt,
            uc_opt,
            ref_xl,
            ref_ul,
            ref_xc,
            ref_uq,
            y_xl,
            y_ul,
            y_xc,
            y_uc,
            p_weight1,
            p_weight2,
            i_admm,
        )
        para2["para_l"] = np.array(p_weight1)
        para2["para_i"] = np.array(p_weight2)
        para2["i_admm"] = float(i_admm)

        if i_admm == target_admm_iter:
            return para2

        sol2 = planner.jax_ADMM_SubP2(para2)
        scxl_cons = sol2["scxl_traj"]
        scul_cons = sol2["scul_traj"]
        scxc_cons = np.array(sol2["scxc_traj"])
        scuc_cons = np.array(sol2["scuc_traj"])

        sol3 = planner.jax_ADMM_SubP3(
            xl_opt,
            scxl_cons,
            y_xl,
            ul_opt,
            scul_cons,
            y_ul,
            xc_opt,
            scxc_cons,
            y_xc,
            uc_opt,
            scuc_cons,
            y_uc,
            p_weight1[-4],
            p_weight1[-3],
            p_weight1[-2],
            p_weight1[-1],
            p_weight2[-4],
            p_weight2[-3],
            p_weight2[-2],
            p_weight2[-1],
            planner.max_iter_ADMM,
            i_admm,
        )
        y_xl = sol3["scxL_traj_new"]
        y_ul = sol3["scuL_traj_new"]
        y_xc = sol3["scxC_traj_new"]
        y_uc = sol3["scuC_traj_new"]
        scxl_traj = scxl_cons
        scul_traj = scul_cons
        scxc_traj = scxc_cons
        scuc_traj = scuc_cons

    raise RuntimeError("failed to capture SubP2 snapshot")


def build_step_problem(planner, para2, target_step):
    dims = (planner.nxl, planner.nul, planner.nxi, planner.nui, int(planner.nq), int(planner.num_dis))
    w_init_batch, params_batch = planner._prepare_subp2_batch(para2)
    if target_step < 0 or target_step >= w_init_batch.shape[0]:
        raise ValueError(f"target_step out of range: {target_step}")

    x_init = w_init_batch[target_step]
    params_t = {k: v[target_step] for k, v in params_batch.items()}
    return dims, x_init, params_t


def pack_original_para2(para2, planner):
    nxl, nul, nxi, nui, nq, N = planner.nxl, planner.nul, planner.nxi, planner.nui, int(planner.nq), planner.N
    xl_ref = np.array(para2["xl_ref"]).reshape(N + 1, nxl)
    ul_ref = np.array(para2["ul_ref"]).reshape(N, nul)
    xc_ref = np.array(para2["xc_ref"]).reshape(nq, N + 1, nxi).transpose(1, 0, 2)
    uc_ref_single = np.array(para2["uc_ref_single"]).reshape(nq, nui)
    xl_ideal = np.array(para2["xl_ideal"]).reshape(N + 1, nxl)
    y_xl = np.array(para2["y_xl"]).reshape(N + 1, nxl)
    ul_ideal = np.array(para2["ul_ideal"]).reshape(N, nul)
    y_ul = np.array(para2["y_ul"]).reshape(N, nul)
    xc_ideal = np.array(para2["xc_ideal"]).reshape(nq, N + 1, nxi).transpose(1, 0, 2)
    y_xc = np.array(para2["y_xc"]).reshape(nq, N + 1, nxi).transpose(1, 0, 2)
    uc_ideal = np.array(para2["uc_ideal"]).reshape(nq, N, nui).transpose(1, 0, 2)
    y_uc = np.array(para2["y_uc"]).reshape(nq, N, nui).transpose(1, 0, 2)
    blocks = [
        xl_ref.reshape(-1),
        ul_ref.reshape(-1),
        xc_ref.reshape(-1),
        uc_ref_single.reshape(-1),
        xl_ideal.reshape(-1),
        y_xl.reshape(-1),
        ul_ideal.reshape(-1),
        y_ul.reshape(-1),
        xc_ideal.reshape(-1),
        y_xc.reshape(-1),
        uc_ideal.reshape(-1),
        y_uc.reshape(-1),
        np.array(para2["para_l"]).reshape(-1),
        np.array(para2["para_i"]).reshape(-1),
        np.array([para2["i_admm"]], dtype=float),
    ]
    return np.concatenate(blocks)


def build_original_step_reference(orig_sol, planner, target_step):
    nxl, nul, nxi, nui, nq, N = planner.nxl, planner.nul, planner.nxi, planner.nui, int(planner.nq), planner.N
    xl = np.array(orig_sol["scxl_traj"][target_step], dtype=float)
    if target_step < N:
        ul = np.array(orig_sol["scul_traj"][target_step], dtype=float)
        cable_parts = []
        for i in range(nq):
            xi = np.array(orig_sol["scxc_traj"][i][target_step], dtype=float)
            ui = np.array(orig_sol["scuc_traj"][i][target_step], dtype=float)
            cable_parts.append(np.concatenate([xi, ui]))
    else:
        ul = np.zeros((nul,), dtype=float)
        cable_parts = []
        for i in range(nq):
            xi = np.array(orig_sol["scxc_traj"][i][target_step], dtype=float)
            ui = np.zeros((nui,), dtype=float)
            cable_parts.append(np.concatenate([xi, ui]))
    return np.concatenate([xl, ul, np.concatenate(cable_parts)])


def primal_feas_metrics(planner, x_val, params_t, dims):
    c_raw = planner.ipoptax_equality(x_val, params_t, dims)
    g_raw = planner.ipoptax_inequality(x_val, params_t, dims)
    eq_inf = float(np.max(np.abs(np.array(c_raw))))
    ineq_vio = float(max(0.0, np.max(np.array(g_raw))))
    return eq_inf, ineq_vio, max(eq_inf, ineq_vio)


def solve_one_case(planner, dims, x_init, params_t, formulation, args, cfg, orig_step_ref=None):
    import jax.numpy as jnp

    eps_s = 1e-3
    f = lambda x: planner.ipoptax_objective(x, params_t, dims)
    c_raw = lambda x: planner.ipoptax_equality(x, params_t, dims)
    g_raw = lambda x: planner.ipoptax_inequality(x, params_t, dims)
    c_scale = params_t["eq_scale"]
    g_scale = params_t["ineq_scale"]
    c = lambda x: c_scale * c_raw(x)
    g = lambda x: g_scale * g_raw(x)

    c0 = c(x_init)
    g0 = g(x_init)
    ws_s0 = jnp.minimum(jnp.maximum(-g0 + eps_s, eps_s), cfg["s0_cap"])
    ws_z0 = jnp.full_like(ws_s0, cfg["z0_init"])
    ws_y0 = jnp.zeros_like(c0)

    res = ipoptax_solve(
        f=f,
        c=c,
        g=g,
        ws_x=x_init,
        ws_s=ws_s0,
        ws_y=ws_y0,
        ws_z=ws_z0,
        max_iterations=args.max_solver_iters,
        max_kkt_violation=args.max_kkt_violation,
        lin_sys_formulation=formulation,
        tau_min=cfg["tau_min"],
        mu_min=cfg["mu_min"],
        min_delta=cfg["min_delta"],
        gamma_y=cfg["gamma_y"],
        gamma_z=cfg["gamma_z"],
        armijo_factor=args.armijo_factor,
        feasibility_armijo_factor=args.feasibility_armijo_factor,
        line_search_factor=cfg["line_search_factor"],
        line_search_min_step_size=cfg["line_search_min_step_size"],
        print_logs=False,
    )

    x_sol = np.array(res["x"])
    finite = bool(np.all(np.isfinite(x_sol)))
    conv = bool(np.array(res["converged"]))
    iterations = int(np.array(res["iteration"]))
    eq_inf, ineq_vio, feas = primal_feas_metrics(planner, x_sol if finite else np.array(x_init), params_t, dims)
    row = {
        **cfg,
        "converged": conv,
        "iterations": iterations,
        "finite": finite,
        "eq_inf": eq_inf,
        "ineq_vio": ineq_vio,
        "feas": feas,
    }
    if orig_step_ref is not None and finite:
        diff = x_sol - orig_step_ref
        row["orig_max_abs"] = float(np.max(np.abs(diff)))
        row["orig_rmse"] = float(np.sqrt(np.mean(diff**2)))
    elif orig_step_ref is not None:
        row["orig_max_abs"] = float("inf")
        row["orig_rmse"] = float("inf")
    return row


def solve_case_batch(planner, dims, x_init, params_t, formulation, args, cfg_list, orig_step_ref=None):
    import jax
    import jax.numpy as jnp

    if not cfg_list:
        return []

    eps_s = 1e-3
    f = lambda x: planner.ipoptax_objective(x, params_t, dims)
    c_raw = lambda x: planner.ipoptax_equality(x, params_t, dims)
    g_raw = lambda x: planner.ipoptax_inequality(x, params_t, dims)
    c_scale = params_t["eq_scale"]
    g_scale = params_t["ineq_scale"]
    c = lambda x: c_scale * c_raw(x)
    g = lambda x: g_scale * g_raw(x)

    x_init_j = jnp.array(x_init)
    c0 = c(x_init_j)
    g0 = g(x_init_j)

    cfg_keys = [
        "tau_min",
        "mu_min",
        "min_delta",
        "gamma_y",
        "gamma_z",
        "line_search_factor",
        "line_search_min_step_size",
        "s0_cap",
        "z0_init",
    ]
    cfg_batch = {k: jnp.array([cfg[k] for cfg in cfg_list]) for k in cfg_keys}

    def solve_cfg(tau_min, mu_min, min_delta, gamma_y, gamma_z, line_search_factor, line_search_min_step_size, s0_cap, z0_init):
        ws_s0 = jnp.minimum(jnp.maximum(-g0 + eps_s, eps_s), s0_cap)
        ws_z0 = jnp.full_like(ws_s0, z0_init)
        ws_y0 = jnp.zeros_like(c0)
        return ipoptax_solve(
            f=f,
            c=c,
            g=g,
            ws_x=x_init_j,
            ws_s=ws_s0,
            ws_y=ws_y0,
            ws_z=ws_z0,
            max_iterations=args.max_solver_iters,
            max_kkt_violation=args.max_kkt_violation,
            lin_sys_formulation=formulation,
            tau_min=tau_min,
            mu_min=mu_min,
            min_delta=min_delta,
            gamma_y=gamma_y,
            gamma_z=gamma_z,
            armijo_factor=args.armijo_factor,
            feasibility_armijo_factor=args.feasibility_armijo_factor,
            line_search_factor=line_search_factor,
            line_search_min_step_size=line_search_min_step_size,
            print_logs=False,
        )

    batched_solver = jax.jit(
        jax.vmap(
            solve_cfg,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0),
        )
    )

    res = batched_solver(
        cfg_batch["tau_min"],
        cfg_batch["mu_min"],
        cfg_batch["min_delta"],
        cfg_batch["gamma_y"],
        cfg_batch["gamma_z"],
        cfg_batch["line_search_factor"],
        cfg_batch["line_search_min_step_size"],
        cfg_batch["s0_cap"],
        cfg_batch["z0_init"],
    )

    rows = []
    xs = np.array(res["x"])
    convs = np.array(res["converged"]).astype(bool)
    iterations = np.array(res["iteration"]).astype(int)
    for i, cfg in enumerate(cfg_list):
        x_sol = xs[i]
        finite = bool(np.all(np.isfinite(x_sol)))
        conv = bool(convs[i])
        iters = int(iterations[i])
        eq_inf, ineq_vio, feas = primal_feas_metrics(planner, x_sol if finite else np.array(x_init), params_t, dims)
        row = {
            **cfg,
            "converged": conv,
            "iterations": iters,
            "finite": finite,
            "eq_inf": eq_inf,
            "ineq_vio": ineq_vio,
            "feas": feas,
        }
        if orig_step_ref is not None and finite:
            diff = x_sol - orig_step_ref
            row["orig_max_abs"] = float(np.max(np.abs(diff)))
            row["orig_rmse"] = float(np.sqrt(np.mean(diff**2)))
        elif orig_step_ref is not None:
            row["orig_max_abs"] = float("inf")
            row["orig_rmse"] = float("inf")
        rows.append(row)
    return rows


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


def print_top(results, init_feas):
    print("")
    print(f"Initial feasibility: {init_feas:.6e}")
    print("Top cases:")
    for rank, row in enumerate(results[:10], start=1):
        print(
            f"{rank:02d} | rmse={row.get('orig_rmse', float('nan')):.3e} | "
            f"max_abs={row.get('orig_max_abs', float('nan')):.3e} | "
            f"feas={row['feas']:.3e} | eq={row['eq_inf']:.3e} | "
            f"ineq={row['ineq_vio']:.3e} | conv={int(row['converged'])} | "
            f"finite={int(row['finite'])} | it={row['iterations']} | "
            f"tau={row['tau_min']:.2g} mu={row['mu_min']:.1e} d={row['min_delta']:.1e} "
            f"gy={row['gamma_y']:.1e} gz={row['gamma_z']:.1e} "
            f"ls={row['line_search_factor']:.2g} min_ls={row['line_search_min_step_size']:.1e} "
            f"s0={row['s0_cap']:.2g} z0={row['z0_init']:.1e}"
        )


def format_case_brief(row):
    return (
        f"tau={row['tau_min']:.2g}, mu={row['mu_min']:.1e}, d={row['min_delta']:.1e}, "
        f"gy={row['gamma_y']:.1e}, gz={row['gamma_z']:.1e}, "
        f"ls={row['line_search_factor']:.2g}, min_ls={row['line_search_min_step_size']:.1e}, "
        f"s0={row['s0_cap']:.2g}, z0={row['z0_init']:.1e}"
    )


def write_csv(path, rows):
    fieldnames = [
        "feas",
        "eq_inf",
        "ineq_vio",
        "orig_max_abs",
        "orig_rmse",
        "converged",
        "finite",
        "iterations",
        "tau_min",
        "mu_min",
        "min_delta",
        "gamma_y",
        "gamma_z",
        "line_search_factor",
        "line_search_min_step_size",
        "s0_cap",
        "z0_init",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = build_parser().parse_args()
    t0 = time.perf_counter()

    def log(msg):
        dt = time.perf_counter() - t0
        print(f"[LOG +{dt:7.2f}s] {msg}", flush=True)

    ensure_jax_backend(args)
    formulation = formulation_from_name(args.solver_formulation)

    log("Building planner/task context")
    ctx = build_context(args)

    if args.target_admm_iter < 0 or args.target_admm_iter >= args.max_iter_admm:
        raise ValueError(
            f"target-admm-iter must be in [0, {args.max_iter_admm - 1}], got {args.target_admm_iter}"
        )

    log(f"Capturing SubP2 snapshot at admm_iter={args.target_admm_iter}")
    para2 = capture_subp2_snapshot(ctx, args.target_admm_iter, args.max_iter_admm)
    log("Running original IPOPT SubP2 on the same snapshot")
    orig_para2 = pack_original_para2(para2, ctx["orig_planner"])
    orig_sol = ctx["orig_planner"].ADMM_SubP2(orig_para2)

    planner = ctx["planner"]
    dims, x_init, params_t = build_step_problem(planner, para2, args.target_step)
    orig_step_ref = build_original_step_reference(orig_sol, planner, args.target_step)
    _, _, init_feas = primal_feas_metrics(planner, np.array(x_init), params_t, dims)
    log(
        "Snapshot ready: "
        f"task={args.task_idx}, horizon={args.horizon}, admm_iter={args.target_admm_iter}, "
        f"step={args.target_step}, init_feas={init_feas:.3e}"
    )

    results = []
    cases = list(iter_grid(args))
    total_cases = len(cases)
    log(f"Running {total_cases} grid cases with execution_mode={args.execution_mode}")
    if total_cases:
        log(f"First case template: {format_case_brief(cases[0])}")
    scan_t0 = time.perf_counter()
    if args.execution_mode == "serial":
        for idx, cfg in enumerate(cases, start=1):
            case_t0 = time.perf_counter()
            row = solve_one_case(planner, dims, x_init, params_t, formulation, args, cfg, orig_step_ref=orig_step_ref)
            results.append(row)
            should_log = (
                idx == 1
                or idx == total_cases
                or (args.progress_every > 0 and idx % args.progress_every == 0)
            )
            if should_log:
                elapsed = time.perf_counter() - scan_t0
                avg = elapsed / idx
                remain = avg * (total_cases - idx)
                best = min(
                    results,
                    key=lambda r: (
                        r.get("orig_rmse", float("inf")),
                        r["feas"],
                        0 if r["converged"] else 1,
                        0 if r["finite"] else 1,
                        r["iterations"],
                    ),
                )
                log(
                    f"Scanned {idx}/{total_cases} | "
                    f"last_case_time={time.perf_counter() - case_t0:.2f}s | "
                    f"last_feas={row['feas']:.3e} last_rmse={row.get('orig_rmse', float('nan')):.3e} "
                    f"conv={int(row['converged'])} it={row['iterations']} | "
                    f"best_feas={best['feas']:.3e} best_rmse={best.get('orig_rmse', float('nan')):.3e} "
                    f"conv={int(best['converged'])} it={best['iterations']} | "
                    f"eta={remain:.1f}s"
                )
                log(f"Last case: {format_case_brief(row)}")
                log(f"Best case so far: {format_case_brief(best)}")
    else:
        if args.batch_size <= 0:
            raise ValueError(f"batch-size must be positive, got {args.batch_size}")
        num_chunks = (total_cases + args.batch_size - 1) // args.batch_size
        for chunk_idx in range(num_chunks):
            start = chunk_idx * args.batch_size
            end = min(total_cases, start + args.batch_size)
            chunk_cases = cases[start:end]
            chunk_t0 = time.perf_counter()
            chunk_rows = solve_case_batch(
                planner,
                dims,
                x_init,
                params_t,
                formulation,
                args,
                chunk_cases,
                orig_step_ref=orig_step_ref,
            )
            results.extend(chunk_rows)
            idx = end
            elapsed = time.perf_counter() - scan_t0
            avg = elapsed / idx
            remain = avg * (total_cases - idx)
            best = min(
                results,
                key=lambda r: (
                    r.get("orig_rmse", float("inf")),
                    r["feas"],
                    0 if r["converged"] else 1,
                    0 if r["finite"] else 1,
                    r["iterations"],
                ),
            )
            last = chunk_rows[-1]
            log(
                f"Chunk {chunk_idx + 1}/{num_chunks} | "
                f"cases {start + 1}-{end}/{total_cases} | "
                f"chunk_time={time.perf_counter() - chunk_t0:.2f}s | "
                f"last_feas={last['feas']:.3e} last_rmse={last.get('orig_rmse', float('nan')):.3e} | "
                f"best_feas={best['feas']:.3e} best_rmse={best.get('orig_rmse', float('nan')):.3e} | "
                f"eta={remain:.1f}s"
            )
            log(f"Last case: {format_case_brief(last)}")
            log(f"Best case so far: {format_case_brief(best)}")

    results.sort(
        key=lambda r: (
            r.get("orig_rmse", float("inf")),
            r["feas"],
            0 if r["converged"] else 1,
            0 if r["finite"] else 1,
            r["iterations"],
        )
    )
    print_top(results, init_feas)

    out_csv = args.out_csv
    if not out_csv:
        out_csv = (
            f"subp2_scan_task{args.task_idx}_h{args.horizon}_a{args.target_admm_iter}"
            f"_k{args.target_step}.csv"
        )
    write_csv(out_csv, results)
    log(f"Wrote CSV: {out_csv}")


if __name__ == "__main__":
    main()
