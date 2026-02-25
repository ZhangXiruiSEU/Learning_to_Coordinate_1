
"""
验证脚本：JAX Planner 等价性测试
文件名: verify_jax_vs_original.py
功能: 
1. 加载原版训练好的神经网络 (.pt)
2. 针对指定 Task (0-9) 生成权重参数
3. 调用 JustWorkingOnIt.py 中的 jax_ADMM_forward_MPC 执行前向规划 (3次迭代)
4. 绘制轨迹图用于与原版对比
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import time as TM
from scipy.spatial.transform import Rotation as Rot
import torch
import os

# --- 导入 JAX 模块 ---
import JustWorkingOnIt as JAX_Planner  # 新文件叫 JustWorkingOnIt.py
import Dynamics_load_cable_autotuning_2nd_COM_Dyn as Original_Dyn
import Neural_network # 必须有这个文件来加载模型类

def verify_jax_planner():
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
    horizon = 100
    
    # 2. 神经网络输入归一化参数 (必须一致!)
    max_radius = 0.15
    k_const = 10.0
    
    # 3. 验证参数
    max_iter_ADMM = 3  # 根据要求：固定为 3 次
    initial_model = 4  # 默认读取第4号模型 
    weight_mode = 'n'  # 神经网络模式
    
    sysm_para = np.array([m1, m2, 
                          1/4*m1*rl**2, 1/4*m1*rl**2, 1/2*m1*rl**2, 
                          rl, nq, rq, mq, fqmax,
                          cl0, ro])

    # ================= 初始化环境与规划器 =================
    print(f"Loading JAX Planner from JustWorkingOnIt.py...")
    # 初始化 JAX 规划器
    MPC_load = JAX_Planner.MPC_Planner(sysm_para, dt, horizon)
    
    # 初始化原版动力学 (用于生成参考轨迹 Minisnap)
    sysm = Original_Dyn.multilift_model(sysm_para, dt)
    
    # 定义辅助转换函数 (用于处理神经网络输出)
    # 这些数字来自原版 main 脚本
    D_outl = MPC_load.npl 
    D_outi = MPC_load.npi
    
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
    task_idx_str = input("Please select a task index (0-9): ")
    try:
        task_idx = int(task_idx_str)
    except ValueError:
        print("Invalid input, defaulting to Task 0")
        task_idx = 0
        
    print(f"Task {task_idx} selected. ADMM Iterations: {max_iter_ADMM}")
    print("=============================================")

    # ================= 加载任务与网络 =================
    # 1. 加载任务
    if not os.path.exists('trained_data_meta_COM_Dyn/rp_task.npy'):
        print("Error: rp_task.npy not found in trained_data_meta_COM_Dyn/")
        return

    rp_task = np.load('trained_data_meta_COM_Dyn/rp_task.npy')
    rg_task_val = (m2 / mtot) * rp_task[task_idx] # [m]
    # rg_task_val 应该是 (3, 1) 的形状
    if rg_task_val.ndim == 1:
        rg_task_val = rg_task_val.reshape(3, 1)
        
    print(f"Load Eccentricity (rg): {rg_task_val.flatten()}")
    
    # 更新规划器的 rg 参数
    MPC_load.rg = rg_task_val # 注入 rg
    MPC_load.allocation_martrix(rg_task_val) # 重新计算 Pt

    # 2. 加载神经网络
    # 注意：这里假设你使用的是 'trained_nn_l_4_3_n.pt' 这种格式，需根据实际情况调整文件名
    # 博士后一般会指定用训练好的模型，这里拼凑文件名
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
    dummy_para = np.zeros((1, MPC_load.n_Pauto))
    
    # 这里有点 hack，直接从 MPC_load 里借用 Gradient_Solver 类
    # 确保 JustWorkingOnIt.py 里有 Gradient_Solver 类定义
    GS = JAX_Planner.Gradient_Solver(sysm_para, horizon, dummy_x, dummy_u, dummy_x, dummy_u, 
                                     np.zeros((8,1)), np.zeros((4,1)), np.zeros((8,1)), np.zeros((4,1)),
                                     dummy_para, np.zeros((1, MPC_load.npl)), np.zeros((1, MPC_load.npi)))
    
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
    Ref_xl = np.zeros((MPC_load.nxl, horizon+1)) # (13, N+1)
    Ref_ul = np.zeros((MPC_load.nul, horizon))   # (6, N)
    time_t = 0
    for k in range(horizon):
        ref_x_k, ref_u_k = sysm.minisnap_load_circle(Coeffx, Coeffy, Coeffz, time_t, rg_task_val)
        Ref_xl[:, k] = ref_x_k.flatten()
        Ref_ul[:, k] = ref_u_k.flatten()
        time_t += dt
    # 补最后一个点的参考 (通常假设静止或同上)
    Ref_xl[:, horizon] = Ref_xl[:, horizon-1]

    # 初始化线缆参考
    # 构造初始状态
    # 尝试从文件加载初始状态，如果失败则使用参考轨迹起点
    try:
        xl_init_file = np.load(f'trained_data_meta_COM_Dyn/xl_init_{max_iter_ADMM}.npy')
        xl_init_task = np.zeros(MPC_load.nxl)
        xl_init_task[0] = xl_init_file[0] + rg_task_val[0] # x + bias
        xl_init_task[1] = xl_init_file[1] + rg_task_val[1] # y + bias
        xl_init_task[2:] = xl_init_file[2:]
    except FileNotFoundError:
        print("Warning: xl_init file not found, using Ref_xl[0] as start.")
        xl_init_task = Ref_xl[:, 0].copy()

    # 构造 Cable 初值和参考
    # 原版 main 脚本里 ref_uq 是一个长向量 [nq*nui]，ref_xq 是 list
    ref_uq_single = np.zeros(nq * MPC_load.nui) # 全0控制作为参考
    xq_init_list = [] # list of (8,)
    ref_xq_list = []  # list of (8, N+1)
    
    # 简单的垂直参考
    # 假设 cl0, mtot 等参数
    for i in range(nq):
        # 初始状态: 垂直向下，张力平衡重力
        xi_0 = np.array([0, 0, 1, 0, 0, 0, 9.81*mtot/nq, 0]) 
        xq_init_list.append(xi_0)
        # 参考轨迹铺满 (Flattened: shape (8*(N+1), ) )
        # 这里需要注意 _pack_paraC 的期望输入
        ref_xq_flat = np.tile(xi_0[:, None], (1, horizon+1)).flatten('F') # 按列展平
        ref_xq_list.append(ref_xq_flat)

    # ================= 执行 JAX 求解 (核心步骤) =================
    print("
>>> Starting JAX ADMM Forward Calculation...")
    start_time = TM.time()
    
    # Ref_xl 和 Ref_ul 需要 flatten
    Ref_xl_flat = Ref_xl.flatten('F') # 按列展平 (Time major if N+1 is col) -> Check packing order
    # _pack_paraL: Ref_xl.flatten() -> 默认是 'C' (Row major). 
    # 原版 np.reshape 是 C-style. 
    # 让我们保持默认 flatten() 即可，只要 _pack_paraL 和 jax_MPC_Load 读取顺序一致。
    
    results = MPC_load.jax_ADMM_forward_MPC(
        Ref_xl=Ref_xl.flatten(), 
        Ref_ul=Ref_ul.flatten(),
        ref_xq=ref_xq_list, # List of flattened arrays
        ref_uq=ref_uq_single,
        xl_fb=xl_init_task,
        xq_fb=np.concatenate(xq_init_list),
        paral=P_weight1,
        paraC=P_weight2,
        max_iter_ADMM=max_iter_ADMM
    )
    
    end_time = TM.time()
    print(f">>> JAX Calculation Finished in {(end_time - start_time)*1000:.2f} ms")
    
    # ================= 绘图 (Result Evaluation) =================
    # 从 results 字典中提取轨迹
    # 结果结构: {'load_trajectory': (N+1, 13), 'cable_trajectories': (nq, N+1, 8), ...}
    xl_opt = results['load_trajectory'] # (N+1, 13)
    xc_opt = results['cable_trajectories'] # (nq, N+1, 8)
    
    # 准备绘图数据
    Time = np.arange(horizon) * dt
    
    # 1. 绘制负载轨迹 (XY平面)
    plt.figure(figsize=(6,6), dpi=100)
    plt.title(f"Load Trajectory (Task {task_idx}, JAX)")
    # 障碍物
    obs1 = Circle((MPC_load.pob1[0], MPC_load.pob1[1]), ro, color='red', alpha=0.3)
    obs2 = Circle((MPC_load.pob2[0], MPC_load.pob2[1]), ro, color='red', alpha=0.3)
    plt.gca().add_patch(obs1)
    plt.gca().add_patch(obs2)
    
    # 轨迹
    plt.plot(Ref_xl[0, :], Ref_xl[1, :], 'g--', label='Reference')
    plt.plot(xl_opt[:, 0], xl_opt[:, 1], 'b.-', label='JAX Solution')
    
    # 起点
    plt.scatter(xl_init_task[0], xl_init_task[1], c='k', marker='x', label='Start')
    
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    save_path = f"verify_jax_task_{task_idx}.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # 2. 绘制张力曲线 (Tension)
    # JAX 版本中，xi = [di(3), wi(3), ti(1), vti(1)] -> Tension index is 6
    plt.figure(figsize=(10, 4))
    plt.title("Cable Tensions (JAX)")
    for i in range(nq):
        # xc_opt shape: (nq, N+1, 8)
        tensions = xc_opt[i, :, 6] # 取第 6 列
        plt.plot(tensions, label=f'Cable {i+1}')
    
    plt.xlabel("Step")
    plt.ylabel("Tension [N]")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    verify_jax_planner()
