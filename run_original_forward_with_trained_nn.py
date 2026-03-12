"""
Run one full original (CasADi) forward pass with trained neural-network weights.

This script does NOT train models. It:
1) loads trained NN checkpoints,
2) builds original planner/dynamics,
3) runs one ADMM forward optimization for one task,
4) saves result plots.
"""

import argparse
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial.transform import Rotation as Rot
import torch

import Dynamics_load_cable_autotuning_2nd_COM_Dyn
import Kinodynamic_Planning_ADMM_quaternion_DDP_autotuning_2nd_COM_Dyn
import Neural_network  # required for torch.load class resolution


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--initial-model", type=int, default=4, help="stage-2 model index")
    parser.add_argument("--max-iter-admm", type=int, default=3, help="stage-2 ADMM truncation")
    parser.add_argument("--initial-model-stage1", type=int, default=4, help="stage-1 model index")
    parser.add_argument("--max-iter-admm-stage1", type=int, default=3, help="stage-1 ADMM truncation")
    parser.add_argument("--weight-mode-stage1", type=str, default="n", choices=["n", "f"])
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="Planning_plots_multiagent_meta_COM_Dyn/forward_once_original")
    return parser


def convert_nn_output(out_column, d_out):
    out_row = np.zeros((1, d_out))
    for i in range(d_out):
        out_row[0, i] = out_column[i, 0]
    return out_row


def build_references(sysm, coeffx, coeffy, coeffz, rg_task, di_ref, ti_ref, horizon, nxl, nul, nxi, nqi):
    ref_xl = np.zeros(nxl * (horizon + 1))
    ref_ul = np.zeros(nul * horizon)
    ref_xc = [np.zeros((horizon + 1) * nxi) for _ in range(nqi)]

    t = 0.0
    for k in range(horizon):
        xk, uk = sysm.minisnap_load_circle(coeffx, coeffy, coeffz, t, rg_task)
        ref_xl[k * nxl:(k + 1) * nxl] = xk
        ref_ul[k * nul:(k + 1) * nul] = uk
        t += sysm.dt
        for i in range(nqi):
            di_k = np.reshape(di_ref[i][:, k], (3, 1))
            wi_k = np.zeros((3, 1))
            ai_k = np.zeros((3, 1))
            ji_k = np.zeros((3, 1))
            ti_k = np.reshape(ti_ref[i, k], (1, 1))
            dti_k = np.zeros((1, 1))
            xi_k = np.reshape(np.vstack((di_k, wi_k, ai_k, ji_k, ti_k, dti_k)), nxi)
            ref_xc[i][k * nxi:(k + 1) * nxi] = xi_k

    xN, _uN = sysm.minisnap_load_circle(coeffx, coeffy, coeffz, t, rg_task)
    ref_xl[horizon * nxl:(horizon + 1) * nxl] = xN
    for i in range(nqi):
        di_N = np.reshape(di_ref[i][:, -1], (3, 1))
        wi_N = np.zeros((3, 1))
        ai_N = np.zeros((3, 1))
        ji_N = np.zeros((3, 1))
        ti_N = np.reshape(ti_ref[i, -1], (1, 1))
        dti_N = np.zeros((1, 1))
        xi_N = np.reshape(np.vstack((di_N, wi_N, ai_N, ji_N, ti_N, dti_N)), nxi)
        ref_xc[i][horizon * nxi:(horizon + 1) * nxi] = xi_N

    return ref_xl, ref_ul, ref_xc


def build_initial_cable_state(di_ref, ti_ref, nxi, nqi):
    xq_init = np.zeros(nqi * nxi)
    for i in range(nqi):
        di_init = np.reshape(di_ref[i][:, 0], (3, 1))
        wi_init = np.zeros((3, 1))
        ai_init = np.zeros((3, 1))
        ji_init = np.zeros((3, 1))
        ti_init = np.array([[ti_ref[i, 0]]])
        dti_init = np.zeros((1, 1))
        xi_init = np.reshape(np.vstack((di_init, wi_init, ai_init, ji_init, ti_init, dti_init)), nxi)
        xq_init[i * nxi:(i + 1) * nxi] = xi_init
    return xq_init


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Physical constants (same as original main)
    m1, m2 = 0.45, 0.25
    mtot = m1 + m2
    nq = 4
    mq = 0.25
    fqmax = 0.75 * 9.81
    cl0, rq, rl, ro = 1.0, 0.15, 0.25, 0.65
    dt = 0.04
    pob1, pob2 = np.array([[1.7, 1.15]]).T, np.array([[0.3, 3.05]]).T

    sysm_para = np.array([
        m1, m2,
        1 / 4 * m1 * rl**2, 1 / 4 * m1 * rl**2, 1 / 2 * m1 * rl**2,
        rl, nq, rq, mq, fqmax,
        cl0, ro
    ])

    # Dynamics and planner
    sysm = Dynamics_load_cable_autotuning_2nd_COM_Dyn.multilift_model(sysm_para, dt)
    rp0 = np.array([[0.05, 0.05, 0.0]]).T
    sysm.Rotational_Inertia(rp0)
    sysm.model()

    planner = Kinodynamic_Planning_ADMM_quaternion_DDP_autotuning_2nd_COM_Dyn.MPC_Planner(sysm_para, dt, args.horizon)
    planner.Rotational_Inertia(rp0)
    planner.allocation_martrix(m2 / mtot * rp0)
    planner.SetStateVariables(sysm.xl, sysm.xi)
    planner.SetCtrlVariables(sysm.ul, sysm.ui)
    planner.SetDyns(sysm.model_l, sysm.model_i)
    planner.SetWeightPara()
    planner.SetPayloadCostDyn(args.max_iter_admm)
    planner.SetCableCostDyn(args.max_iter_admm)
    planner.SetConstriants(pob1, pob2)
    planner.SetADMMSubP2_SoftCost_k()
    planner.SetADMMSubP2_SoftCost_N()
    planner.ADMM_SubP2_Init()
    planner.ADMM_SubP2_N_Init()
    planner.Load_derivatives_DDP_ADMM()
    planner.Cable_derivatives_DDP_ADMM()
    planner.system_derivatives_SubP2_ADMM_k()
    planner.system_derivatives_SubP2_ADMM_N()
    planner.system_derivatives_SubP3_ADMM()

    nxl, nul, nxi, nui = sysm.nxl, sysm.nul, sysm.nxi, sysm.nui
    max_radius, k_const = 0.15, 10.0

    # Load task/meta data
    rp_task_all = np.load("trained_data_meta_COM_Dyn/rp_task.npy")
    if args.task_idx < 0 or args.task_idx >= len(rp_task_all):
        raise ValueError(f"task_idx out of range: {args.task_idx}")
    rp_task = rp_task_all[args.task_idx]
    rg_task = m2 / mtot * rp_task
    if rp_task.ndim == 1:
        rp_task = rp_task.reshape(3, 1)
    if rg_task.ndim == 1:
        rg_task = rg_task.reshape(3, 1)

    # Update task-specific model terms
    sysm.Rotational_Inertia(rp_task)
    sysm.model()
    planner.Rotational_Inertia(rp_task)
    planner.allocation_martrix(rg_task)
    planner.SetDyns(sysm.model_l, sysm.model_i)
    planner.SetConstriants(pob1, pob2)
    planner.SetADMMSubP2_SoftCost_k()
    planner.SetADMMSubP2_SoftCost_N()
    planner.ADMM_SubP2_Init()
    planner.ADMM_SubP2_N_Init()
    planner.Load_derivatives_DDP_ADMM()
    planner.system_derivatives_SubP2_ADMM_k()
    planner.system_derivatives_SubP2_ADMM_N()

    # Stage-1 reference cable direction/tension
    loss_stage1_path = (
        f"trained_data_meta_COM_Dyn/loss_train_{args.initial_model_stage1}_{args.max_iter_admm_stage1}_{args.weight_mode_stage1}.npy"
    )
    i_train_pref = None
    if os.path.exists(loss_stage1_path):
        i_train_pref = len(np.load(loss_stage1_path)) - 1

    patt_any = (
        f"Planning_plots_meta_COM_Dyn/cable_direction_*_{args.task_idx}_*_{args.max_iter_admm_stage1}_{args.weight_mode_stage1}.npy"
    )
    cand = []
    for p in glob.glob(patt_any):
        m = re.match(
            r"cable_direction_(\d+)_(\d+)_(\d+)_(\d+)_([a-zA-Z])\.npy$",
            os.path.basename(p)
        )
        if not m:
            continue
        i_train_k, task_k, model1_k, admm1_k, wm_k = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
        if task_k != args.task_idx or admm1_k != args.max_iter_admm_stage1 or wm_k != args.weight_mode_stage1:
            continue
        cand.append((p, i_train_k, model1_k))

    if not cand:
        raise FileNotFoundError(
            "No stage-1 cable_direction references found in Planning_plots_meta_COM_Dyn for this task/ADMM/mode."
        )

    # Prefer user-requested stage1 model; if unavailable, fallback to any model.
    cand_model = [c for c in cand if c[2] == args.initial_model_stage1]
    if not cand_model:
        cand_model = cand
        print(
            f"[WARN] No stage-1 references for initial_model_stage1={args.initial_model_stage1}; "
            "fallback to available model(s)."
        )

    # Prefer training index inferred from loss file; else pick latest available.
    chosen = None
    if i_train_pref is not None:
        hit = [c for c in cand_model if c[1] == i_train_pref]
        if hit:
            chosen = hit[0]
    if chosen is None:
        chosen = sorted(cand_model, key=lambda x: x[1], reverse=True)[0]

    di_path, i_train_1, model1_used = chosen
    ti_path = di_path.replace("cable_direction_", "tension_magnitude_")
    if not os.path.exists(ti_path):
        raise FileNotFoundError(f"Matched DI path but TI path missing: {ti_path}")
    print(f"Using stage-1 references: i_train_1={i_train_1}, model1={model1_used}")

    di_ref = np.load(di_path)
    ti_ref = np.load(ti_path)
    if di_ref.shape[2] < args.horizon + 1 or ti_ref.shape[1] < args.horizon + 1:
        raise ValueError("Reference DI/TI horizon is shorter than requested --horizon")
    if di_ref.shape[2] != args.horizon + 1:
        di_ref = di_ref[:, :, :args.horizon + 1]
        ti_ref = ti_ref[:, :args.horizon + 1]

    # Load trained NN
    path_l = f"trained_data_multiagent_meta_COM_Dyn/trained_nn_l_{args.initial_model}_{args.max_iter_admm}_n.pt"
    path_i = f"trained_data_multiagent_meta_COM_Dyn/trained_nn_i_{args.initial_model}_{args.max_iter_admm}_n.pt"
    if not os.path.exists(path_l) or not os.path.exists(path_i):
        raise FileNotFoundError(f"Trained NN not found: {path_l} or {path_i}")
    nn_l = torch.load(path_l, weights_only=False)
    nn_i = torch.load(path_i, weights_only=False)

    # Convert NN output to hyper-parameters using Gradient_Solver helpers
    grad_solver = Kinodynamic_Planning_ADMM_quaternion_DDP_autotuning_2nd_COM_Dyn.Gradient_Solver(
        sysm_para, args.horizon,
        planner.xl, planner.ul, planner.scxl, planner.scul,
        planner.xi, planner.ui, planner.scxi, planner.scui,
        planner.P_auto, planner.para_l, planner.para_i
    )

    nn_input = np.reshape(rg_task[0:2] / max_radius * k_const, (2, 1))
    nn_input_t = torch.FloatTensor(nn_input)
    with torch.no_grad():
        nn_l_out = convert_nn_output(nn_l(nn_input_t).numpy(), planner.npl)
        nn_i_out = convert_nn_output(nn_i(nn_input_t).numpy(), planner.npi)
    p_weight1 = grad_solver.Set_Parameters_nn_l(nn_l_out)
    p_weight2 = grad_solver.Set_Parameters_nn_i(nn_i_out)

    # Build references and initial states
    coeffx = np.zeros((2, 8))
    coeffy = np.zeros((2, 8))
    coeffz = np.zeros((2, 8))
    for k in range(2):
        coeffx[k, :] = np.load(f"Reference_traj_4/coeffx{k+1}.npy")
        coeffy[k, :] = np.load(f"Reference_traj_4/coeffy{k+1}.npy")
        coeffz[k, :] = np.load(f"Reference_traj_4/coeffz{k+1}.npy")

    ref_xl, ref_ul, ref_xc = build_references(
        sysm, coeffx, coeffy, coeffz, rg_task, di_ref, ti_ref, args.horizon, nxl, nul, nxi, nq
    )
    ref_uq = np.zeros(int(nq) * nui)
    xq_init = build_initial_cable_state(di_ref, ti_ref, nxi, nq)

    xl_init = np.load(f"trained_data_meta_COM_Dyn/xl_init_{args.max_iter_admm}.npy")
    xl_init_task = np.zeros(nxl)
    xl_init_task[0] = float(xl_init[0]) + float(rg_task[0, 0])
    xl_init_task[1] = float(xl_init[1]) + float(rg_task[1, 0])
    xl_init_task[2:nxl] = xl_init[2:nxl]

    # Full original forward pass
    print(f"Running original ADMM forward once: task={args.task_idx}, model={args.initial_model}, admm={args.max_iter_admm}")
    opt_sol, *_ = planner.ADMM_forward_MPC(
        ref_xl, ref_ul, ref_xc, ref_uq, xl_init_task, xq_init, p_weight1, p_weight2, args.max_iter_admm
    )

    # Unpack result
    xl_traj = opt_sol["xl_traj"]           # (N+1, 13)
    scxl_traj = opt_sol["scxl_traj"]       # (N+1, 13)
    xc_traj = opt_sol["xc_traj"]           # list[nq] each (N+1, 14)
    scxc_traj = opt_sol["scxc_traj"]       # list[nq] each (N+1, 14)

    time = np.arange(args.horizon) * dt

    # Rebuild reference matrix for plotting
    ref_xl_mat = np.zeros((nxl, args.horizon))
    t = 0.0
    for k in range(args.horizon):
        xk, _uk = sysm.minisnap_load_circle(coeffx, coeffy, coeffz, t, rg_task)
        ref_xl_mat[:, k:k+1] = np.reshape(xk, (nxl, 1))
        t += dt

    # 1) Load XY trajectory
    fig, ax = plt.subplots(figsize=(6, 6), dpi=240)
    ax.add_patch(Circle((pob1[0, 0], pob1[1, 0]), ro, color="red", alpha=0.35))
    ax.add_patch(Circle((pob2[0, 0], pob2[1, 0]), ro, color="red", alpha=0.35))
    ax.plot(ref_xl_mat[0, :], ref_xl_mat[1, :], "--", linewidth=1.0, label="Ref")
    ax.plot(xl_traj[:args.horizon, 0], xl_traj[:args.horizon, 1], linewidth=1.1, label="Planned")
    ax.plot(scxl_traj[:args.horizon, 0], scxl_traj[:args.horizon, 1], linewidth=1.0, label="Safe-copy")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(args.out_dir, f"load_xy_task{args.task_idx}_m{args.initial_model}_a{args.max_iter_admm}.png"), dpi=300)
    plt.close(fig)

    # 2) Quadrotor XY trajectories
    fig, ax = plt.subplots(figsize=(6, 6), dpi=240)
    ax.add_patch(Circle((pob1[0, 0], pob1[1, 0]), ro, color="red", alpha=0.35))
    ax.add_patch(Circle((pob2[0, 0], pob2[1, 0]), ro, color="red", alpha=0.35))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i in range(nq):
        pi = np.zeros((3, args.horizon))
        scpi = np.zeros((3, args.horizon))
        ri = np.reshape(planner.ra[:, i], (3, 1))
        for k in range(args.horizon):
            pl_k = np.reshape(xl_traj[k, 0:3], (3, 1))
            ql_k = np.reshape(xl_traj[k, 6:10], (4, 1))
            rl_k = sysm.q_2_rotation(ql_k)
            di_k = np.reshape(xc_traj[i][k, 0:3], (3, 1))
            scdi_k = np.reshape(scxc_traj[i][k, 0:3], (3, 1))
            ai_k = pl_k + rl_k @ ri
            pi[:, k:k+1] = ai_k + cl0 * di_k
            scpi[:, k:k+1] = ai_k + cl0 * scdi_k
        ax.plot(pi[0, :], pi[1, :], color=colors[i], linewidth=1.1, label=f"Q{i+1}")
        ax.plot(scpi[0, :], scpi[1, :], color=colors[i], linewidth=0.9, linestyle="--", label=f"Q{i+1}-sc")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(os.path.join(args.out_dir, f"quad_xy_task{args.task_idx}_m{args.initial_model}_a{args.max_iter_admm}.png"), dpi=300)
    plt.close(fig)

    # 3) Cable tension
    fig, ax = plt.subplots(figsize=(7, 4), dpi=240)
    for i in range(nq):
        ax.plot(time, xc_traj[i][:args.horizon, 12], linewidth=1.2, label=f"Cable {i+1}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Tension [N]")
    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(args.out_dir, f"tension_task{args.task_idx}_m{args.initial_model}_a{args.max_iter_admm}.png"), dpi=300)
    plt.close(fig)

    # 4) Height + Euler
    fig, ax = plt.subplots(figsize=(7, 4), dpi=240)
    ax.plot(time, xl_traj[:args.horizon, 2], linewidth=1.2, label="actual z")
    ax.plot(time, ref_xl_mat[2, :], "--", linewidth=1.0, label="ref z")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Height [m]")
    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(args.out_dir, f"height_task{args.task_idx}_m{args.initial_model}_a{args.max_iter_admm}.png"), dpi=300)
    plt.close(fig)

    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
