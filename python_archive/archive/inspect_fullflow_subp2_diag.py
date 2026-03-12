import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Dynamics_load_cable_autotuning_2nd_COM_Dyn as Original_Dyn
import Neural_network  # noqa: F401


def main():
    import JustWorkingOnIt as JAX_Planner

    task_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    horizon = int(os.environ.get("VERIFY_HORIZON", "2"))
    max_iter_admm = int(os.environ.get("VERIFY_ADMM_ITERS", "2"))

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
    initial_model = 4

    sysm_para = np.array(
        [m1, m2, 1 / 4 * m1 * rl**2, 1 / 4 * m1 * rl**2, 1 / 2 * m1 * rl**2, rl, nq, rq, mq, fqmax, cl0, ro]
    )

    MPC_jax = JAX_Planner.MPC_Planner(sysm_para, dt, horizon)
    MPC_jax.verbose = False
    MPC_jax.pob1 = np.array([pob1[0, 0], pob1[1, 0], 0.0])
    MPC_jax.pob2 = np.array([pob2[0, 0], pob2[1, 0], 0.0])

    rp_task = np.load("trained_data_meta_COM_Dyn/rp_task.npy")
    rp_task_val = rp_task[task_idx]
    rg_task_val = (m2 / mtot) * rp_task_val
    if rp_task_val.ndim == 1:
        rp_task_val = rp_task_val.reshape(3, 1)
    if rg_task_val.ndim == 1:
        rg_task_val = rg_task_val.reshape(3, 1)
    MPC_jax.rg = rg_task_val
    MPC_jax.allocation_martrix(rg_task_val)

    sysm = Original_Dyn.multilift_model(sysm_para, dt)
    sysm.Rotational_Inertia(rp_task_val)
    sysm.model()

    model_name_suffix = f"{initial_model}_3_n"
    path_l = f"trained_data_multiagent_meta_COM_Dyn/trained_nn_l_{model_name_suffix}.pt"
    path_i = f"trained_data_multiagent_meta_COM_Dyn/trained_nn_i_{model_name_suffix}.pt"
    nn_l = torch.load(path_l, map_location=torch.device("cpu"), weights_only=False)
    nn_i = torch.load(path_i, map_location=torch.device("cpu"), weights_only=False)

    nn_input_np = np.reshape(rg_task_val[0:2] / max_radius * k_const, (2, 1))
    nn_input_torch = torch.FloatTensor(nn_input_np)
    with torch.no_grad():
        out_l = nn_l(nn_input_torch).numpy()
        out_i = nn_i(nn_input_torch).numpy()

    def convert_nn_l(nn_l_outcolumn):
        nn_l_row = np.zeros((1, MPC_jax.npl))
        for i in range(MPC_jax.npl):
            nn_l_row[0, i] = nn_l_outcolumn[i, 0]
        return nn_l_row

    def convert_nn_i(nn_i_outcolumn):
        nn_i_row = np.zeros((1, MPC_jax.npi))
        for i in range(MPC_jax.npi):
            nn_i_row[0, i] = nn_i_outcolumn[i, 0]
        return nn_i_row

    tunable_l = convert_nn_l(out_l)
    tunable_i = convert_nn_i(out_i)

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
    p_weight1 = GS.Set_Parameters_nn_l(tunable_l)
    p_weight2 = GS.Set_Parameters_nn_i(tunable_i)

    coeffx = np.zeros((2, 8))
    coeffy = np.zeros((2, 8))
    coeffz = np.zeros((2, 8))
    for k in range(2):
        coeffx[k, :] = np.load(f"Reference_traj_4/coeffx{k+1}.npy")
        coeffy[k, :] = np.load(f"Reference_traj_4/coeffy{k+1}.npy")
        coeffz[k, :] = np.load(f"Reference_traj_4/coeffz{k+1}.npy")

    ref_xl_mat = np.zeros((horizon + 1, MPC_jax.nxl))
    ref_ul_mat = np.zeros((horizon, MPC_jax.nul))
    time_t = 0.0
    for k in range(horizon):
        ref_x_k, ref_u_k = sysm.minisnap_load_circle(coeffx, coeffy, coeffz, time_t, rg_task_val)
        ref_xl_mat[k, :] = ref_x_k.flatten()
        ref_ul_mat[k, :] = ref_u_k.flatten()
        time_t += dt
    ref_xl_mat[horizon, :] = ref_xl_mat[horizon - 1, :]
    ref_xl_flat = ref_xl_mat.reshape(-1)
    ref_ul_flat = ref_ul_mat.reshape(-1)

    try:
        xl_init_file = np.load(f"trained_data_meta_COM_Dyn/xl_init_{max_iter_admm}.npy")
        xl_init_task = np.zeros(MPC_jax.nxl)
        xl_init_task[0] = float(xl_init_file[0]) + float(rg_task_val[0, 0])
        xl_init_task[1] = float(xl_init_file[1]) + float(rg_task_val[1, 0])
        xl_init_task[2:] = xl_init_file[2:]
    except FileNotFoundError:
        xl_init_task = ref_xl_mat[0, :].copy()

    ref_uq_single = np.zeros(nq * MPC_jax.nui)
    xq_init_list = []
    ref_xq_list = []
    for _ in range(nq):
        xi_0 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.81 * mtot / nq, 0])
        xq_init_list.append(xi_0)
        ref_xq_list.append(np.tile(xi_0, horizon + 1))

    results = MPC_jax.jax_ADMM_forward_MPC(
        Ref_xl=ref_xl_flat,
        Ref_ul=ref_ul_flat,
        ref_xq=ref_xq_list,
        ref_uq=ref_uq_single,
        xl_fb=xl_init_task,
        xq_fb=np.concatenate(xq_init_list),
        paral=p_weight1,
        parac=p_weight2,
        max_iter_ADMM=max_iter_admm,
    )

    hist = results.get("subp2_diag_history", [])
    for i, diag in enumerate(hist):
        print(f"ADMM iter {i}:")
        for key in ["mode", "converged", "iterations", "eq_inf", "ineq_vio", "tail_mu", "tail_ineq", "tail_comp", "tail_dual"]:
            if key in diag:
                print(f"  {key}: {np.array(diag[key])}")


if __name__ == "__main__":
    main()
