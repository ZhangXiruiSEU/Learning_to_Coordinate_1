"""
1. 加载原版训练好的神经网络 (.pt)
2. 针对指定 Task (0-9) 生成权重参数
3. 在相同权重下分别调用 原版/CasADi 与 JAX 的 forward MPC
4. 输出数值误差并绘制对比图
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import time as TM
from scipy.spatial.transform import Rotation as Rot
import torch
import os
import sys
import subprocess

# --- 导入 JAX 模块 ---
import Dynamics_load_cable_autotuning_2nd_COM_Dyn as Original_Dyn
import Kinodynamic_Planning_ADMM_quaternion_DDP_autotuning_2nd_COM_Dyn as Original_Planner
import Neural_network # 必须有这个文件来加载模型类 # Required for torch.load to recognize the model class structure


def _reexec_with_backend_fallback():
    """
    默认使用 CPU；可选尝试 Apple Metal（通过环境变量开启）。
    通过子进程执行，避免 METAL 后端初始化失败导致当前进程直接崩溃。
    """
    if os.environ.get("VERIFY_JAX_BACKEND_SELECTED") == "1":
        return

    script = os.path.abspath(__file__)
    argv = [sys.executable, script] + sys.argv[1:]

    # 默认 CPU
    env_cpu = os.environ.copy()
    env_cpu["VERIFY_JAX_BACKEND_SELECTED"] = "1"
    env_cpu["JAX_PLATFORMS"] = "cpu"
    if os.environ.get("VERIFY_JAX_TRY_METAL", "0") != "1":
        print("Using JAX backend: CPU")
        rc = subprocess.run(argv, env=env_cpu).returncode
        sys.exit(rc)

    # 可选先试 METAL，再回退 CPU
    env_metal = os.environ.copy()
    env_metal["VERIFY_JAX_BACKEND_SELECTED"] = "1"
    env_metal["JAX_PLATFORMS"] = "METAL"
    print("Trying JAX backend: METAL")
    rc = subprocess.run(argv, env=env_metal).returncode
    if rc == 0:
        sys.exit(0)

    print(f"METAL failed with code {rc}, falling back to CPU...")
    rc2 = subprocess.run(argv, env=env_cpu).returncode
    sys.exit(rc2)

def _resolve_task_idx(task_idx=None):
    if task_idx is not None:
        return int(task_idx)
    task_idx_str = os.environ.get("VERIFY_TASK_IDX")
    if task_idx_str is None:
        if len(sys.argv) > 1:
            task_idx_str = sys.argv[1]
        else:
            task_idx_str = "0"
    try:
        return int(task_idx_str)
    except (TypeError, ValueError):
        print("Invalid task index, defaulting to Task 0")
        return 0


def _safe_nanmax(values):
    arr = np.array(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmax(arr))


def verify_jax_planner(task_idx=None, show_plots=None):
    # 延迟导入：确保后端选择逻辑先执行
    import JustWorkingOnIt as JAX_Planner  # 新文件叫 JustWorkingOnIt.py

    # ================= 配置区域 (严格对齐原版) =================
    # 1. 基础物理参数
    m1, m2 = 0.45, 0.25
    mtot = m1 + m2
    nq = 4
    mq = 0.25
    fqmax = 0.75 * 9.81
    cl0 = 1.0
    rq, rl, ro = 0.15, 0.25, 0.65
    dt = 0.04
    horizon = int(os.environ.get("VERIFY_HORIZON", "100"))
    pob1, pob2 = np.array([[1.7, 1.15]]).T, np.array([[0.3, 3.05]]).T
    '''
   * `m1` (0.45 kg): 负载（Payload）本身的质量（比如一个篮子）。
   * `m2` (0.25 kg): 负载上额外加的重物质量（为了制造偏心效果）。
   * `mtot` (0.7 kg): 总吊重。
   * `nq` (4): 无人机的数量（4 架）。
   * `mq` (0.25 kg): 每架无人机的质量。
   * `fqmax`: 无人机的最大拉力（0.75倍重力加速度 * 9.81）。这意味着单架飞机的推力上限。
   * `cl0` (1.0 m): 缆绳的长度（1 米）。
   * `rq` (0.15 m): 无人机的半径（体型大小，用于避障）。
   * `rl` (0.25 m): 负载（篮子）的半径。
   * `ro` (0.65 m): 障碍物（那根柱子）的半径。
   * `dt` (0.04 s): 仿真步长（每一步代表现实中的 0.04 秒，即 25Hz）。
   * `horizon` (100): 预测视界（往后看 100 步，即 4 秒）
    '''
    # 2. 神经网络输入归一化参数 (必须一致!)
    max_radius = 0.15
    k_const = 10.0
    
    # 3. 验证参数
    max_iter_ADMM = int(os.environ.get("VERIFY_ADMM_ITERS", "3"))
    initial_model = 4  # 默认读取第4号模型 
    weight_mode = 'n'  # 神经网络模式
    
    sysm_para = np.array([m1, m2, 
                          1/4*m1*rl**2, 1/4*m1*rl**2, 1/2*m1*rl**2, 
                          rl, nq, rq, mq, fqmax,
                          cl0, ro])

    # ================= 初始化环境与规划器 =================
    print(f"Loading JAX Planner from JustWorkingOnIt.py...")
    # 初始化 JAX 规划器
    MPC_jax = JAX_Planner.MPC_Planner(sysm_para, dt, horizon)
    MPC_jax.verbose = os.environ.get("VERIFY_JAX_VERBOSE", "0") == "1"
    MPC_jax.pob1 = np.array([pob1[0, 0], pob1[1, 0], 0.0])
    MPC_jax.pob2 = np.array([pob2[0, 0], pob2[1, 0], 0.0])
    
    # 初始化原版动力学 (用于生成参考轨迹 Minisnap + 原版 forward)
    sysm = Original_Dyn.multilift_model(sysm_para, dt)
    
    # 定义辅助转换函数 (用于处理神经网络输出)
    # 这些数字来自原版 main 脚本
    D_outl = MPC_jax.npl 
    D_outi = MPC_jax.npi
    
    def convert_nn_l(nn_l_outcolumn):
        nn_l_row = np.zeros((1, D_outl))
        for i in range(D_outl):
            nn_l_row[0,i] = nn_l_outcolumn[i,0]
        return nn_l_row

    def convert_nn_i(nn_i_outcolumn):
        nn_i_row = np.zeros((1, D_outi))
        for i in range(D_outi):
            nn_i_row[0,i] = nn_i_outcolumn[i,0]
        return nn_i_row

    # ================= 用户交互 =================
    print("=============================================")
    task_idx = _resolve_task_idx(task_idx)
        
    print(f"Task {task_idx} selected. ADMM Iterations: {max_iter_ADMM}")
    print("=============================================")

    # ================= 加载任务与网络 =================
    # 1. 加载任务
    if not os.path.exists('trained_data_meta_COM_Dyn/rp_task.npy'):
        print("Error: rp_task.npy not found in trained_data_meta_COM_Dyn/")
        return

    rp_task = np.load('trained_data_meta_COM_Dyn/rp_task.npy')
    rp_task_val = rp_task[task_idx]
    rg_task_val = (m2 / mtot) * rp_task_val # [m]
    # rg_task_val 应该是 (3, 1) 的形状
    if rp_task_val.ndim == 1:
        rp_task_val = rp_task_val.reshape(3, 1)
    if rg_task_val.ndim == 1:
        rg_task_val = rg_task_val.reshape(3, 1)
        
    print(f"Load Eccentricity (rg): {rg_task_val.flatten()}")
    
    # 更新 JAX 规划器的 rg 参数
    MPC_jax.rg = rg_task_val
    MPC_jax.allocation_martrix(rg_task_val)

    # 初始化原版规划器（完整符号图）
    sysm.Rotational_Inertia(rp_task_val)
    sysm.model()
    MPC_orig = Original_Planner.MPC_Planner(sysm_para, dt, horizon)
    MPC_orig.Rotational_Inertia(rp_task_val)
    MPC_orig.allocation_martrix(rg_task_val)
    MPC_orig.SetStateVariables(sysm.xl, sysm.xi)
    MPC_orig.SetCtrlVariables(sysm.ul, sysm.ui)
    MPC_orig.SetDyns(sysm.model_l, sysm.model_i)
    MPC_orig.SetWeightPara()
    MPC_orig.SetPayloadCostDyn(max_iter_ADMM)
    MPC_orig.SetCableCostDyn(max_iter_ADMM)
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

    # 2. 加载神经网络
    # 注意：这里假设你使用的是 'trained_nn_l_4_3_n.pt' 这种格式，需根据实际情况调整文件名
    # 用训练好的模型，这里拼凑文件名
    model_name_suffix = f"{initial_model}_3_n" # 假设是用3次迭代训练出来的模型
    
    # 检查文件是否存在，如果不存在则回退到 initial_NN
    PATH_L = f"trained_data_multiagent_meta_COM_Dyn/trained_nn_l_{model_name_suffix}.pt"
    PATH_I = f"trained_data_multiagent_meta_COM_Dyn/trained_nn_i_{model_name_suffix}.pt"
    
    if not os.path.exists(PATH_L):
        print(f"Warning: Model {PATH_L} not found. Trying generic 'initial' model...")
        PATH_L = "trained_data_multiagent_meta_COM_Dyn/initial_NN_l.pt"
        PATH_I = "trained_data_multiagent_meta_COM_Dyn/initial_NN_i.pt"

    print(f"Loading NN from: {PATH_L}")
    # 兼容性加载，强制 weights_only=False
    nn_l = torch.load(PATH_L, map_location=torch.device('cpu'), weights_only=False)
    nn_i = torch.load(PATH_I, map_location=torch.device('cpu'), weights_only=False)

    # 3. 前向推理 (生成权重)
    # 输入归一化逻辑：(rg / max_radius) * k_const
    # rg_task_val[0:2] 取 x, y 分量
    nn_input_np = np.reshape(rg_task_val[0:2] / max_radius * k_const, (2, 1))
    nn_input_torch = torch.FloatTensor(nn_input_np)
    
    with torch.no_grad():
        out_l = nn_l(nn_input_torch).numpy()
        out_i = nn_i(nn_input_torch).numpy()
        
    tunable_l = convert_nn_l(out_l)
    tunable_i = convert_nn_i(out_i)
    
    # 实例化辅助求解器来做参数映射 (Weight Mapping)
    # 构造一些假的输入只是为了初始化 Gradient_Solver
    dummy_x = np.zeros((13,1))
    dummy_u = np.zeros((6,1))
    dummy_para = np.zeros((1, MPC_jax.n_Pauto))

    # ：参数映射函数 Set_Parameters_nn_l 被定义在 Gradient_Solver 类里。
    # 为了能调用这个函数，我们造一个“假”的 Gradient Solver 对象。
    # 我们给它塞了一堆全 0 的假数据（dummy_x），只为了能借用它里面的那个方法
    GS = JAX_Planner.Gradient_Solver(sysm_para, horizon, dummy_x, dummy_u, dummy_x, dummy_u, 
                                     np.zeros((MPC_jax.nxi,1)), np.zeros((MPC_jax.nui,1)),
                                     np.zeros((MPC_jax.nxi,1)), np.zeros((MPC_jax.nui,1)),
                                     dummy_para, np.zeros((1, MPC_jax.npl)), np.zeros((1, MPC_jax.npi)))
    
    P_weight1 = GS.Set_Parameters_nn_l(tunable_l)
    P_weight2 = GS.Set_Parameters_nn_i(tunable_i)
    
    print("Weights generated successfully.")
    # P_weight1 是一个 array, 打印部分参数验证
    print(f"rho_lx (px): {P_weight1[-4]:.4f}, rho_lu (pu): {P_weight1[-3]:.4f}")

    # ================= 生成参考轨迹 =================
    Coeffx = np.zeros((2,8)); Coeffy = np.zeros((2,8)); Coeffz = np.zeros((2,8))
    # 确保 Reference_traj_4 文件夹存在
    for k in range(2):
        Coeffx[k,:] = np.load(f'Reference_traj_4/coeffx{k+1}.npy')
        Coeffy[k,:] = np.load(f'Reference_traj_4/coeffy{k+1}.npy')
        Coeffz[k,:] = np.load(f'Reference_traj_4/coeffz{k+1}.npy')
        
    # 参考轨迹维度生成
    Ref_xl_mat = np.zeros((horizon+1, MPC_jax.nxl)) # (N+1, 13)
    Ref_ul_mat = np.zeros((horizon, MPC_jax.nul))   # (N, 6)
    time_t = 0
    for k in range(horizon):
        ref_x_k, ref_u_k = sysm.minisnap_load_circle(Coeffx, Coeffy, Coeffz, time_t, rg_task_val)
        Ref_xl_mat[k, :] = ref_x_k.flatten()
        Ref_ul_mat[k, :] = ref_u_k.flatten()
        time_t += dt
    # 补最后一个点的参考 (通常假设静止或同上)
    Ref_xl_mat[horizon, :] = Ref_xl_mat[horizon-1, :]
    Ref_xl_flat = Ref_xl_mat.reshape(-1)
    Ref_ul_flat = Ref_ul_mat.reshape(-1)

    # 初始化线缆参考
    # 构造初始状态
    # 尝试从文件加载初始状态，如果失败则使用参考轨迹起点
    try:
        xl_init_file = np.load(f'trained_data_meta_COM_Dyn/xl_init_{max_iter_ADMM}.npy')
        xl_init_task = np.zeros(MPC_jax.nxl)
        xl_init_task[0] = float(xl_init_file[0]) + float(rg_task_val[0, 0]) # x + bias
        xl_init_task[1] = float(xl_init_file[1]) + float(rg_task_val[1, 0]) # y + bias
        xl_init_task[2:] = xl_init_file[2:]
    except FileNotFoundError:
        print("Warning: xl_init file not found, using Ref_xl[0] as start.")
        xl_init_task = Ref_xl_mat[0, :].copy()

    # 构造 Cable 初值和参考
    # 原版 main 脚本里 ref_uq 是一个长向量 [nq*nui]，ref_xq 是 list
    ref_uq_single = np.zeros(nq * MPC_jax.nui) # 全0控制作为参考
    xq_init_list = [] # list of (nxi,)
    ref_xq_list = []  # list of (nxi*(N+1),)
    
    # 简单的垂直参考
    # 假设 cl0, mtot 等参数
    for i in range(nq):
        # 初始状态: 垂直向下，张力平衡重力
        xi_0 = np.array([0, 0, 1,   0, 0, 0,   0, 0, 0,   0, 0, 0,   9.81*mtot/nq, 0])
        xq_init_list.append(xi_0)
        # 参考轨迹铺满 (Flattened: shape (nxi*(N+1), ) )
        ref_xq_flat = np.tile(xi_0, horizon + 1)
        ref_xq_list.append(ref_xq_flat)

    # ================= 执行原版求解 =================
    print(">> Starting ORIGINAL (CasADi) ADMM Forward Calculation...")
    start_time = TM.time()
    opt_sol_orig, *_ = MPC_orig.ADMM_forward_MPC(
        Ref_xl=Ref_xl_flat,
        Ref_ul=Ref_ul_flat,
        ref_xq=ref_xq_list,
        ref_uq=ref_uq_single,
        xl_fb=xl_init_task,
        xq_fb=np.concatenate(xq_init_list),
        paral=P_weight1,
        paraC=P_weight2,
        max_iter_ADMM=max_iter_ADMM
    )
    end_time = TM.time()
    print(f">>> ORIGINAL Calculation Finished in {(end_time - start_time)*1000:.2f} ms")

    # ================= 执行 JAX 求解 =================
    print(">> Starting JAX ADMM Forward Calculation...")
    start_time = TM.time()
    results = MPC_jax.jax_ADMM_forward_MPC(
        Ref_xl=Ref_xl_flat,
        Ref_ul=Ref_ul_flat,
        ref_xq=ref_xq_list, # List of flattened arrays
        ref_uq=ref_uq_single,
        xl_fb=xl_init_task,
        xq_fb=np.concatenate(xq_init_list),
        paral=P_weight1,
        parac=P_weight2,
        max_iter_ADMM=max_iter_ADMM
    )
    
    end_time = TM.time()
    print(f">>> JAX Calculation Finished in {(end_time - start_time)*1000:.2f} ms")

    # ================= 数值对比 =================
    xl_orig = np.array(opt_sol_orig['xl_traj'])            # (N+1, 13)
    ul_orig = np.array(opt_sol_orig['ul_traj'])            # (N, 6)
    xc_orig = np.array(opt_sol_orig['xc_traj'])            # (nq, N+1, 14)

    xl_jax = np.array(results['load_trajectory'])          # (N+1, 13)
    ul_jax = np.array(results['control_inputs'])           # (N, 6)
    xc_jax = np.array(results['cable_trajectories'])       # (nq, N+1, 14)
    subp2_hist = results.get('subp2_diag_history', [])

    def report_diff(name, a, b):
        diff = a - b
        max_abs = float(np.max(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        print(f"[DIFF] {name}: max_abs={max_abs:.6e}, rmse={rmse:.6e}")
        return {"max_abs": max_abs, "rmse": rmse}

    diff_load_state = report_diff("load_state_xl", xl_jax, xl_orig)
    diff_load_control = report_diff("load_control_ul", ul_jax, ul_orig)
    diff_cable_state = report_diff("cable_state_xc", xc_jax, xc_orig)

    print("[SUBP2] summary:")
    subp2_summary = []
    for i_admm, diag in enumerate(subp2_hist):
        modes = list(diag.get("mode", []))
        tail_mu = np.array(diag.get("tail_mu", []), dtype=float)
        tail_ineq = np.array(diag.get("tail_ineq", []), dtype=float)
        tail_comp = np.array(diag.get("tail_comp", []), dtype=float)
        tail_dual = np.array(diag.get("tail_dual", []), dtype=float)
        counts = {}
        for mode in modes:
            counts[mode] = counts.get(mode, 0) + 1
        counts_str = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        item = {
            "admm_iter": i_admm,
            "mode_counts": counts,
            "iter_mean": float(diag.get("iter_mean", np.nan)),
            "iter_p50": float(diag.get("iter_p50", np.nan)),
            "iter_p90": float(diag.get("iter_p90", np.nan)),
            "iter_max": int(diag.get("iter_max", -1)),
            "max_eq": float(np.max(np.array(diag.get('eq_inf', [np.nan]), dtype=float))),
            "max_raw_ineq": float(np.max(np.array(diag.get('ineq_vio', [np.nan]), dtype=float))),
            "max_tail_mu": _safe_nanmax(tail_mu),
            "max_tail_ineq": _safe_nanmax(tail_ineq),
            "max_tail_comp": _safe_nanmax(tail_comp),
            "max_tail_dual": _safe_nanmax(tail_dual),
            "batch_total_ms": float(diag.get("batch_total_ms", np.nan)),
            "batch_build_ms_total": float(diag.get("batch_build_ms_total", np.nan)),
            "batch_solve_ms_total": float(diag.get("batch_solve_ms_total", np.nan)),
            "bridge_prepare_ms": float(diag.get("bridge_prepare_ms", np.nan)),
            "bridge_payload_ms": float(diag.get("bridge_payload_ms", np.nan)),
            "bridge_request_ms": float(diag.get("bridge_request_ms", np.nan)),
            "bridge_rebuild_ms": float(diag.get("bridge_rebuild_ms", np.nan)),
        }
        subp2_summary.append(item)
        iter_mean_str = f"{item['iter_mean']:.2f}" if np.isfinite(item["iter_mean"]) else "n/a"
        iter_p90_str = f"{item['iter_p90']:.1f}" if np.isfinite(item["iter_p90"]) else "n/a"
        iter_max_str = str(item["iter_max"]) if item["iter_max"] >= 0 else "n/a"
        print(
            f"[SUBP2] admm_iter={i_admm} | modes={counts_str or 'n/a'} | "
            f"iter_mean={iter_mean_str} | "
            f"iter_p90={iter_p90_str} | "
            f"iter_max={iter_max_str} | "
            f"max_eq={item['max_eq']:.3e} | "
            f"max_raw_ineq={item['max_raw_ineq']:.3e} | "
            f"batch_build_ms_total={item['batch_build_ms_total']:.3f} | "
            f"batch_solve_ms_total={item['batch_solve_ms_total']:.3f} | "
            f"batch_total_ms={item['batch_total_ms']:.3f} | "
            f"bridge_prepare_ms={item['bridge_prepare_ms']:.3f} | "
            f"bridge_payload_ms={item['bridge_payload_ms']:.3f} | "
            f"bridge_request_ms={item['bridge_request_ms']:.3f} | "
            f"bridge_rebuild_ms={item['bridge_rebuild_ms']:.3f} | "
            f"max_tail_mu={item['max_tail_mu']:.3e} | "
            f"max_tail_ineq={item['max_tail_ineq']:.3e} | "
            f"max_tail_comp={item['max_tail_comp']:.3e} | "
            f"max_tail_dual={item['max_tail_dual']:.3e}"
        )
    
    # ================= 绘图 (Result Evaluation) =================
    # 准备绘图数据
    Time = np.arange(horizon) * dt
    
    # 1. 绘制负载轨迹 (XY平面)
    plt.figure(figsize=(6,6), dpi=100)
    plt.title(f"Load Trajectory Compare (Task {task_idx})")
    # 障碍物
    obs1 = Circle((pob1[0, 0], pob1[1, 0]), ro, color='red', alpha=0.3)
    obs2 = Circle((pob2[0, 0], pob2[1, 0]), ro, color='red', alpha=0.3)
    plt.gca().add_patch(obs1)
    plt.gca().add_patch(obs2)
    
    # 轨迹
    plt.plot(Ref_xl_mat[:, 0], Ref_xl_mat[:, 1], 'g--', label='Reference')
    plt.plot(xl_orig[:, 0], xl_orig[:, 1], 'k.-', label='Original')
    plt.plot(xl_jax[:, 0], xl_jax[:, 1], 'b.-', label='JAX')
    
    # 起点
    plt.scatter(xl_init_task[0], xl_init_task[1], c='k', marker='x', label='Start')
    
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    save_path = f"verify_compare_load_task_{task_idx}.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # 2. 绘制张力曲线 (Tension)
    # xi = [di(3), wi(3), ai(3), ji(3), ti(1), vti(1)] -> Tension index = 12
    plt.figure(figsize=(10, 4))
    plt.title("Cable Tensions Compare (Original vs JAX)")
    for i in range(nq):
        plt.plot(xc_orig[i, :, 12], label=f'Orig C{i+1}', linewidth=1.2)
        plt.plot(xc_jax[i, :, 12], '--', label=f'JAX C{i+1}', linewidth=1.2)
    
    plt.xlabel("Step")
    plt.ylabel("Tension [N]")
    plt.legend()
    plt.grid(True)
    save_path2 = f"verify_compare_tension_task_{task_idx}.png"
    plt.savefig(save_path2)
    print(f"Plot saved to {save_path2}")
    if show_plots is None:
        show_plots = os.environ.get("VERIFY_SHOW_PLOTS", "1") == "1"
    if show_plots:
        plt.show()
    else:
        plt.close("all")
    return {
        "task_idx": task_idx,
        "horizon": horizon,
        "admm_iters": max_iter_ADMM,
        "jax_profile": results.get("profile"),
        "diff": {
            "load_state_xl": diff_load_state,
            "load_control_ul": diff_load_control,
            "cable_state_xc": diff_cable_state,
        },
        "subp2_summary": subp2_summary,
        "subp2_acceptance": {
            "soft_mu_tol": 1e-8,
            "soft_comp_tol": 1e-8,
            "soft_dual_tol": 5e-5,
            "soft_eq_tol": 1e-4,
            "soft_raw_ineq_tol": 2e-1,
            "soft_trace_ineq_tol": 5e-3,
        },
        "plots": [save_path, save_path2],
    }

if __name__ == "__main__":
    _reexec_with_backend_fallback()
    verify_jax_planner()
