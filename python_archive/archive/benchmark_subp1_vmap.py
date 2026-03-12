import json
import os
import sys
import time as TM
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import Dynamics_load_cable_autotuning_2nd_COM_Dyn as Original_Dyn
import Kinodynamic_Planning_ADMM_quaternion_DDP_autotuning_2nd_COM_Dyn as Original_Planner
import JustWorkingOnIt as JAX_Planner
import Neural_network


def _convert_nn_col_to_row(nn_outcolumn):
    out = np.zeros((1, nn_outcolumn.shape[0]))
    for i in range(nn_outcolumn.shape[0]):
        out[0, i] = nn_outcolumn[i, 0]
    return out


def _build_setup(task_idx: int, horizon: int, max_iter_admm: int):
    m1, m2 = 0.45, 0.25
    mtot = m1 + m2
    nq = 4
    mq = 0.25
    fqmax = 0.75 * 9.81
    cl0 = 1.0
    rq, rl, ro = 0.15, 0.25, 0.65
    dt = 0.04
    pob1, pob2 = np.array([[1.7, 1.15]]).T, np.array([[0.3, 3.05]]).T

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

    MPC_jax = JAX_Planner.MPC_Planner(sysm_para, dt, horizon)
    MPC_jax.verbose = False
    MPC_jax.max_iter_ADMM = max_iter_admm
    MPC_jax.pob1 = np.array([pob1[0, 0], pob1[1, 0], 0.0])
    MPC_jax.pob2 = np.array([pob2[0, 0], pob2[1, 0], 0.0])

    sysm = Original_Dyn.multilift_model(sysm_para, dt)

    rp_task = np.load("trained_data_meta_COM_Dyn/rp_task.npy")
    rp_task_val = rp_task[task_idx]
    rg_task_val = (m2 / mtot) * rp_task_val
    if rp_task_val.ndim == 1:
        rp_task_val = rp_task_val.reshape(3, 1)
    if rg_task_val.ndim == 1:
        rg_task_val = rg_task_val.reshape(3, 1)

    MPC_jax.rg = rg_task_val
    MPC_jax.allocation_martrix(rg_task_val)

    sysm.Rotational_Inertia(rp_task_val)
    sysm.model()
    MPC_orig = Original_Planner.MPC_Planner(sysm_para, dt, horizon)
    MPC_orig.Rotational_Inertia(rp_task_val)
    MPC_orig.allocation_martrix(rg_task_val)
    MPC_orig.SetStateVariables(sysm.xl, sysm.xi)
    MPC_orig.SetCtrlVariables(sysm.ul, sysm.ui)
    MPC_orig.SetDyns(sysm.model_l, sysm.model_i)
    MPC_orig.SetWeightPara()
    MPC_orig.SetPayloadCostDyn(max_iter_admm)
    MPC_orig.SetCableCostDyn(max_iter_admm)
    MPC_orig.SetConstriants(pob1, pob2)
    MPC_orig.SetADMMSubP2_SoftCost_k()
    MPC_orig.SetADMMSubP2_SoftCost_N()
    MPC_orig.ADMM_SubP2_Init()
    MPC_orig.ADMM_SubP2_N_Init()
    MPC_orig.Load_derivatives_DDP_ADMM()
    MPC_orig.Cable_derivatives_DDP_ADMM()
    MPC_orig.system_derivatives_SubP2_ADMM_k()
    MPC_orig.system_derivatives_SubP2_ADMM_N()
    MPC_orig.system_derivatives_SubP3_ADMM()

    max_radius = 0.15
    k_const = 10.0
    initial_model = 4
    model_name_suffix = f"{initial_model}_3_n"
    path_l = f"trained_data_multiagent_meta_COM_Dyn/trained_nn_l_{model_name_suffix}.pt"
    path_i = f"trained_data_multiagent_meta_COM_Dyn/trained_nn_i_{model_name_suffix}.pt"
    if not os.path.exists(path_l):
        path_l = "trained_data_multiagent_meta_COM_Dyn/initial_NN_l.pt"
        path_i = "trained_data_multiagent_meta_COM_Dyn/initial_NN_i.pt"

    nn_l = torch.load(path_l, map_location=torch.device("cpu"), weights_only=False)
    nn_i = torch.load(path_i, map_location=torch.device("cpu"), weights_only=False)
    nn_input_np = np.reshape(rg_task_val[0:2] / max_radius * k_const, (2, 1))
    nn_input_torch = torch.FloatTensor(nn_input_np)
    with torch.no_grad():
        out_l = nn_l(nn_input_torch).numpy()
        out_i = nn_i(nn_input_torch).numpy()

    tunable_l = _convert_nn_col_to_row(out_l)
    tunable_i = _convert_nn_col_to_row(out_i)

    dummy_x = np.zeros((13, 1))
    dummy_u = np.zeros((6, 1))
    dummy_para = np.zeros((1, MPC_jax.n_Pauto))
    GS = JAX_Planner.Gradient_Solver(
        sysm_para,
        horizon,
        dummy_x,
        dummy_u,
        dummy_x,
        dummy_u,
        np.zeros((MPC_jax.nxi, 1)),
        np.zeros((MPC_jax.nui, 1)),
        np.zeros((MPC_jax.nxi, 1)),
        np.zeros((MPC_jax.nui, 1)),
        dummy_para,
        np.zeros((1, MPC_jax.npl)),
        np.zeros((1, MPC_jax.npi)),
    )
    P_weight1 = GS.Set_Parameters_nn_l(tunable_l)
    P_weight2 = GS.Set_Parameters_nn_i(tunable_i)

    coeffx = np.zeros((2, 8))
    coeffy = np.zeros((2, 8))
    coeffz = np.zeros((2, 8))
    for k in range(2):
        coeffx[k, :] = np.load(f"Reference_traj_4/coeffx{k+1}.npy")
        coeffy[k, :] = np.load(f"Reference_traj_4/coeffy{k+1}.npy")
        coeffz[k, :] = np.load(f"Reference_traj_4/coeffz{k+1}.npy")

    Ref_xl_mat = np.zeros((horizon + 1, MPC_jax.nxl))
    Ref_ul_mat = np.zeros((horizon, MPC_jax.nul))
    time_t = 0.0
    for k in range(horizon):
        ref_x_k, ref_u_k = sysm.minisnap_load_circle(coeffx, coeffy, coeffz, time_t, rg_task_val)
        Ref_xl_mat[k, :] = ref_x_k.flatten()
        Ref_ul_mat[k, :] = ref_u_k.flatten()
        time_t += dt
    Ref_xl_mat[horizon, :] = Ref_xl_mat[horizon - 1, :]
    Ref_xl_flat = Ref_xl_mat.reshape(-1)
    Ref_ul_flat = Ref_ul_mat.reshape(-1)

    try:
        xl_init_file = np.load(f"trained_data_meta_COM_Dyn/xl_init_{max_iter_admm}.npy")
        xl_init_task = np.zeros(MPC_jax.nxl)
        xl_init_task[0] = float(xl_init_file[0]) + float(rg_task_val[0, 0])
        xl_init_task[1] = float(xl_init_file[1]) + float(rg_task_val[1, 0])
        xl_init_task[2:] = xl_init_file[2:]
    except FileNotFoundError:
        xl_init_task = Ref_xl_mat[0, :].copy()

    ref_uq_single = np.zeros(nq * MPC_jax.nui)
    xq_init_list = []
    ref_xq_list = []
    for _ in range(nq):
        xi_0 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.81 * mtot / nq, 0])
        xq_init_list.append(xi_0)
        ref_xq_list.append(np.tile(xi_0, horizon + 1))
    xq_fb = np.concatenate(xq_init_list)

    scxl_traj, scul_traj, scxc_traj, scuc_traj = MPC_jax._initialize_trajectories(
        Ref_xl_flat, Ref_ul_flat, ref_xq_list, ref_uq_single
    )
    y_xl = np.zeros_like(np.array(scxl_traj))
    y_ul = np.zeros_like(np.array(scul_traj))
    y_xc = np.zeros_like(np.array(scxc_traj))
    y_uc = np.zeros_like(np.array(scuc_traj))

    ParaL = MPC_jax._pack_paraL(
        xl_init_task,
        Ref_xl_flat,
        Ref_ul_flat,
        P_weight1,
        np.array(scxl_traj),
        np.array(scul_traj),
        y_xl,
        y_ul,
        0,
    )
    ParaC = MPC_jax._pack_paraC(
        xq_fb,
        ref_xq_list,
        ref_uq_single,
        P_weight2,
        np.array(scxc_traj),
        np.array(scuc_traj),
        y_xc,
        y_uc,
        0,
    )

    return {
        "MPC_jax": MPC_jax,
        "MPC_orig": MPC_orig,
        "ParaL": ParaL,
        "ParaC": ParaC,
        "xl_fb": xl_init_task,
        "Ref_xl_flat": Ref_xl_flat,
        "Ref_ul_flat": Ref_ul_flat,
        "P_weight1": P_weight1,
        "scxl_traj": np.array(scxl_traj),
        "scul_traj": np.array(scul_traj),
        "y_xl": y_xl,
        "y_ul": y_ul,
    }


def _time_call(fn, repeats=2):
    times = []
    last = None
    for _ in range(repeats):
        t0 = TM.time()
        last = fn()
        times.append((TM.time() - t0) * 1000.0)
    return times, last


def main():
    task_idx = int(os.environ.get("BENCH_TASK_IDX", "0"))
    horizon = int(os.environ.get("BENCH_HORIZON", "20"))
    max_iter_admm = int(os.environ.get("BENCH_ADMM_ITERS", "3"))

    setup = _build_setup(task_idx, horizon, max_iter_admm)
    MPC_jax = setup["MPC_jax"]
    MPC_orig = setup["MPC_orig"]
    ParaL = setup["ParaL"]
    ParaC = setup["ParaC"]

    load_args = (
        setup["xl_fb"],
        setup["Ref_xl_flat"],
        setup["Ref_ul_flat"],
        setup["P_weight1"],
        setup["scxl_traj"].reshape(-1),
        setup["scul_traj"].reshape(-1),
        setup["y_xl"].reshape(-1),
        setup["y_ul"].reshape(-1),
        10,
        1e-2,
        0,
    )

    orig_load_times, _ = _time_call(lambda: MPC_orig.DDP_Load_ADMM_Subp1(*load_args), repeats=2)
    orig_cable_times, _ = _time_call(lambda: MPC_orig.MPC_Cable_DDP_Planning_SubP1(ParaC), repeats=2)

    jax_load_times, _ = _time_call(lambda: MPC_jax.jax_MPC_Load_DDP_Planning_SubP1(ParaL), repeats=2)

    prev_mode = os.environ.get("JAX_CABLE_SUBP1_MODE")
    try:
        os.environ["JAX_CABLE_SUBP1_MODE"] = "batched"
        jax_cable_batched_times, _ = _time_call(lambda: MPC_jax.jax_MPC_Cable_DDP_Planning_SubP1(ParaC), repeats=2)
        os.environ["JAX_CABLE_SUBP1_MODE"] = "legacy"
        jax_cable_legacy_times, _ = _time_call(lambda: MPC_jax.jax_MPC_Cable_DDP_Planning_SubP1(ParaC), repeats=2)
    finally:
        if prev_mode is None:
            os.environ.pop("JAX_CABLE_SUBP1_MODE", None)
        else:
            os.environ["JAX_CABLE_SUBP1_MODE"] = prev_mode

    summary = {
        "task_idx": task_idx,
        "horizon": horizon,
        "admm_iters": max_iter_admm,
        "original": {
            "load_ms": orig_load_times,
            "cable_ms": orig_cable_times,
            "subp1_total_ms": [orig_load_times[i] + orig_cable_times[i] for i in range(len(orig_load_times))],
        },
        "jax": {
            "load_ms": jax_load_times,
            "cable_batched_ms": jax_cable_batched_times,
            "cable_legacy_ms": jax_cable_legacy_times,
            "subp1_total_batched_ms": [jax_load_times[i] + jax_cable_batched_times[i] for i in range(len(jax_load_times))],
            "subp1_total_legacy_ms": [jax_load_times[i] + jax_cable_legacy_times[i] for i in range(len(jax_load_times))],
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
