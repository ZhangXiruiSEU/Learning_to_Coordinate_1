from casadi import *
import numpy as np
from numpy import linalg as LA
import math
from scipy.spatial.transform import Rotation as Rot
from scipy import linalg as sLA
from scipy.linalg import null_space
import time as TM
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import ArpackNoConvergence
import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxopt import BFGS, OSQP
from ddp_vmap_jax import ilqr_ddp_batched, ILQRConfig
from ipoptax.solver import solve as ipoptax_solve, LinearSystemFormulation
from ipoptax.linalg_helpers import project_psd_cone
from functools import partial

"""
Reference
[1] Cao, K., Xu, X., Jin, W., Johansson, K.H. and Xie, L., 2025. 
    A differential dynamic programming framework for inverse reinforcement learning. IEEE Transactions on Robotics.
[2] Jin, W., Wang, Z., Yang, Z. and Mou, S., 2020. 
    Pontryagin differentiable programming: An end-to-end learning and control framework. 
    Advances in Neural Information Processing Systems, 33, pp.7979-7992.

"""

# -----------------------------------------------------------------------------
# 这个文件包含三层东西，文件的布局也对应这个顺序：
# 1. 当前稳定主线会直接走到的 Python/JAX 逻辑，在 1288 行之前
#    - JAX SubP1
#    - SubP2 batch 打包/解包
#    - JAX SubP3
#    - 以及给 persistent Julia SubP2 做诊断时复用的 objective/residual 定义
# 2. 纯 JAX SubP2 实验分支
#    - ipoptax 直接求解
#    - SQP / barrier 试验
# 3. 更老的 CasADi/梯度求解与训练代码
#
# 当前正式稳定主线是：
# Python/JAX SubP1 + Python 侧 batch 打包 + 常驻 Julia worker 解 SubP2 + JAX SubP3。
# 这条主线由 run_julia_subp2_fullflow_persistent.py 在外部拼起来，
# 并且会在运行时把 self.jax_ADMM_SubP2 替换成 Julia IPC 版本。
#

# 读懂当前正式主线的顺序建议：
# 1. MPC_Planner.__init__
# 2. _solve_load_subp1_forward_only / _solve_cable_subp1_forward_only
# 3. _initialize_trajectories / _get_static_params
# 4. _pack_para2
# 5. _prepare_subp2_batch / _unpack_subp2_results
# 6. jax_ADMM_forward_MPC
# 7. jax_ADMM_SubP3
# 8. ipoptax_objective / ipoptax_equality / ipoptax_inequality
#    这三个在当前稳定 Julia 路线里主要用于快照导出和结果核对，不是实际 SubP2 求解器
# _pack_paraL / _pack_paraC 仅旧 packaged SubP1 路径使用

# 下面三个词在这个文件里经常出现，也不是当前 persistent Julia 主线的默认路径：
# - packaged: 旧 SubP1 包装接口。
#   指把当前轮输入重新压成 ParaL/ParaC 这种 legacy 长向量，再交给兼容包装器拆包求解。
# - derivs: 轨迹求完以后额外导出的导数包。
#   典型内容是 Fx/Fu/Qxu/Quu_inv/K_FB，只在旧接口兼容、导数对齐或诊断时需要。
# - legacy: 更老的原版 CasADi/DDP/训练实现。
#   这些代码保留是为了历史对照和兼容。
# -----------------------------------------------------------------------------

# 以下仍然留在文件前部的是"当前稳定主线"会在 __init__ 里直接绑定的全局 JIT 缓存。
# 也就是 SubP1 forward-only / 初始化轨迹这条主线真正会默认走到的部分。
_GLOBAL_LOAD_SOLVER_JIT = None
_GLOBAL_CABLE_SOLVER_JIT = None
_GLOBAL_INIT_LOAD_TRAJ_JIT = None
_GLOBAL_INIT_CABLE_TRAJ_JIT = None


def _get_global_load_solver_jit():
    global _GLOBAL_LOAD_SOLVER_JIT
    if _GLOBAL_LOAD_SOLVER_JIT is None:
        _GLOBAL_LOAD_SOLVER_JIT = jax.jit(
            partial(
                ilqr_ddp_batched,
                dynamics_fn=MPC_Planner.jax_load_dynamics,
                cost_fn=MPC_Planner.jax_load_stage_cost,
                term_cost_fn=MPC_Planner.jax_load_terminal_cost,
            ),
            static_argnames=["cfg"],
        )
    return _GLOBAL_LOAD_SOLVER_JIT


def _get_global_cable_solver_jit():
    global _GLOBAL_CABLE_SOLVER_JIT
    if _GLOBAL_CABLE_SOLVER_JIT is None:
        _GLOBAL_CABLE_SOLVER_JIT = jax.jit(
            partial(
                ilqr_ddp_batched,
                dynamics_fn=MPC_Planner.jax_cable_dynamics_single,
                cost_fn=MPC_Planner.jax_cable_stage_cost,
                term_cost_fn=MPC_Planner.jax_cable_terminal_cost,
            ),
            static_argnames=["cfg"],
        )
    return _GLOBAL_CABLE_SOLVER_JIT


def _get_global_init_load_traj_jit():
    global _GLOBAL_INIT_LOAD_TRAJ_JIT
    if _GLOBAL_INIT_LOAD_TRAJ_JIT is None:
        _GLOBAL_INIT_LOAD_TRAJ_JIT = jax.jit(
            MPC_Planner._build_init_load_traj,
            static_argnames=("nxl",),
        )
    return _GLOBAL_INIT_LOAD_TRAJ_JIT


def _get_global_init_cable_traj_jit():
    global _GLOBAL_INIT_CABLE_TRAJ_JIT
    if _GLOBAL_INIT_CABLE_TRAJ_JIT is None:
        _GLOBAL_INIT_CABLE_TRAJ_JIT = jax.jit(
            MPC_Planner._build_init_cable_traj,
            static_argnames=("nxi", "horizon"),
        )
    return _GLOBAL_INIT_CABLE_TRAJ_JIT



class MPC_Planner:
    # 当前稳定主线必经入口：
    # - 建立系统常量、维度、分配矩阵、JAX jit 缓存
    # - 为后面的 SubP1 / SubP2 打包 / SubP3 提供公共静态数据

    def __init__(self, sysm_para, dt_ctrl, horizon):
        # Payload's parameters
        self.m1     = sysm_para[0] # the payload's mass [kg]
        self.m2     = sysm_para[1] # the added mass [kg]
        self.Jlcom  = np.diag(sysm_para[2:5]) # rotational inertia of m1 about its Geometric Center (GC)
        self.rl     = sysm_para[5] # the radius of load [m]
        self.ml     = self.m1 + self.m2 # the total mass [kg]
        # Quadrotor's parameters
        self.nq     = int(sysm_para[6]) # the number of quadrotors
        self.rq     = sysm_para[7] # the radius of quadrotor [m]
        self.mq     = sysm_para[8] # the quadrotor's mass [kg]
        self.fmax   = sysm_para[9] # the maximum quadrotor's thrust [N]
        # Cable and obstacle's parameters
        self.cl0    = sysm_para[10] # the cable length [m]
        self.ro     = sysm_para[11] # the radius of obstacle [m]
        # Unit direction vector free of coordinate
        self.ex     = jnp.array([[1, 0, 0]]).T
        self.ey     = jnp.array([[0, 1, 0]]).T
        self.ez     = jnp.array([[0, 0, 1]]).T
        # Gravitational acceleration
        self.g      = 9.81      
        self.dt     = dt_ctrl
        # MPC's horizon
        self.N      = int(horizon)
        # barrier parameter
        self.p_bar  = 1e-6
        # lower bound of the ADMM penalty parameter
        self.p_min  = 1e-3
        # hard constraints aligned with original CasADi/IPOPT setup
        self.t_min = 0.01
        self.t_max = 5.0
        self.ui_bound = 1e3

        # before are the original code except the "jnp",

        self.rp = np.array([[0.05, 0.05, 0.0]]).T # 参照 main 脚本的 rp0
        self.rg = (self.m2 / self.ml) * self.rp
        # below are code newly add for JAX implementation
        self._load_derivs_jit = _get_global_load_derivs_jit()
        self._cable_derivs_jit = _get_global_cable_derivs_jit()
        # 1. allocation matrix

        # 我们在 __init__ 里一次性把所有死的数据（如分配矩阵 Pt）算好，变成固定的常量。
        # 这样 JAX 编译时就知道这是一块“石头”，不需要再去追踪它的变化。

        self.alpha  = 2*np.pi/self.nq
        dis_two = 2 * self.rl * math.sin(self.alpha / 2)
        self.num_dis = max(1, int(self.cl0 / max(dis_two, 1e-6)))
        r0          = np.array([[self.rl,0,0]]).T - np.reshape(np.vstack((self.rg[0],self.rg[1],0)),(3,1))  # 1st cable attachment point in {Bl}
        self.ra     = r0
        S_r0        = self.skew_sym_numpy(r0)
        I3          = np.identity(3) # 3-by-3 identity matrix
        self.Pt      = np.vstack((I3,S_r0))
        for i in range(int(self.nq)-1):
            ri      = np.array([[self.rl*(math.cos((i+1)*self.alpha)),self.rl*(math.sin((i+1)*self.alpha)),0]]).T - np.reshape(np.vstack((self.rg[0],self.rg[1],0)),(3,1))
            S_ri    = self.skew_sym_numpy(ri)
            Pi      = np.vstack((I3,S_ri))
            self.Pt = np.append(self.Pt,Pi,axis=1) # the tension mapping matrix: 6-by-3nq with a rank of 6
            self.ra = np.append(self.ra,ri,axis=1) # a matrix that stores the attachment points
        
        self.Pt = jnp.array(self.Pt)
        self.ra = jnp.array(self.ra)


        # 障碍物位置初始化 (先设为0，后续由 Main 传入)
        self.pob1 = jnp.zeros(3)
        self.pob2 = jnp.zeros(3)


        # 2. subproblem 1
        self._load_solver_jit = _get_global_load_solver_jit()
        self._cable_solver_jit = _get_global_cable_solver_jit()
        self._init_load_traj_jit = _get_global_init_load_traj_jit()
        self._init_cable_traj_jit = _get_global_init_cable_traj_jit()
        # 3. subproblem 2
        def subp2_pure_func(Para2_dict):
            return self.jax_ADMM_SubP2(Para2_dict)
        
        self._subp2_jit = jax.jit(subp2_pure_func)

        # 4. 状态维度定义


        # 负载状态 (pl, vl, ql, wl)
        #    * 状态 `xl`：
        #        * pl (位置): 3维
        #        * vl (速度): 3维
        #        * ql (四元数): 4维
        #        * wl (角速度): 3维
        #        * 合计：$3 + 3 + 4 + 3 = 13$
        #    * 控制 `ul`：
        #        * Fl (合力): 3维
        #        * Ml (合力矩): 3维
        #        * 合计：$3 + 3 = 6$
        self.nxl = 13
        self.nul = 6

        # 缆绳状态 (di, wi, ai, ji, ti, vti)
        #     * 状态 `xi`：
        #        * di (缆绳方向): 3维
        #        * wi (缆绳角速度): 3维
        #        * ai (缆绳角加速度): 3维
        #        * ji (缆绳角加加速度): 3维
        #        * ti (张力大小): 1维
        #        * vti (张力变化率): 1维
        #        * 合计：$3 + 3 + 3 + 3 + 1 + 1 = 14$
        #    * 控制 `ui`：
        #        * si (角跃度导数/snap): 3维
        #        * ati (张力加速度): 1维
        #        * 合计：$3 + 1 = 4$
        self.nxi = 14
        self.nui = 4

        # 超参数维度 (来自 Gradient_Solver 的配置)
        self.npl = 2 * self.nxl + self.nul + 4 # 状态权重 + 控制权重 + 4个动态调度参数
        self.npi = 2 * self.nxi + self.nui + 4
        self.n_Pauto = self.npl + self.npi


        # 5. Rotational_Inertia
        ratio_m    = self.m1*self.m2/self.ml
        self.Jl    = self.Jlcom + ratio_m*(self.rp.T@self.rp*np.identity(3)-self.rp@self.rp.T)
        self.Jl  =  jnp.array(self.Jl) #变成jnp
        self.Jl_inv = jnp.linalg.inv(jnp.array(self.Jl))# 逆矩阵

    @staticmethod
    def _jax_load_continuous_dynamics(x, u, params):
        """连续时间负载动力学 x_dot = f(x, u)。"""
        vl = x[3:6]
        ql = x[6:10]
        wl = x[10:13]
        Fl = u[0:3]
        Ml = u[3:6]

        dpl = vl
        dvl = -9.81 * jnp.array([0.0, 0.0, 1.0], dtype=x.dtype) + (1.0 / params['ml']) * Fl

        w1, w2, w3 = wl[0], wl[1], wl[2]
        Omega = jnp.array([
            [0.0, -w1, -w2, -w3],
            [w1, 0.0, w3, -w2],
            [w2, -w3, 0.0, w1],
            [w3, w2, -w1, 0.0]
        ], dtype=x.dtype)
        dql = 0.5 * Omega @ ql

        dwl = params['Jl_inv'] @ (Ml - jnp.cross(wl, params['Jl'] @ wl))
        return jnp.concatenate([dpl, dvl, dql, dwl])

    @staticmethod
    def jax_load_dynamics(x, u, params):
        """离散时间负载动力学 x_{k+1} = x_k + dt * RK4(f)."""
        dt = params['dt']
        f = MPC_Planner._jax_load_continuous_dynamics
        k1 = f(x, u, params)
        k2 = f(x + 0.5 * dt * k1, u, params)
        k3 = f(x + 0.5 * dt * k2, u, params)
        k4 = f(x + dt * k3, u, params)
        return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    @staticmethod
    def _jax_cable_continuous_dynamics(x, u, params):
        """连续时间缆绳动力学 x_dot = f(x, u)。"""
        di = x[0:3]
        wi = x[3:6]
        ai = x[6:9]
        ji = x[9:12]
        vti = x[13]
        si = u[0:3]
        ati = u[3]

        di_dot = jnp.cross(wi, di)
        wi_dot = ai
        ai_dot = ji
        ji_dot = si
        ti_dot = vti
        ti_ddot = ati
        return jnp.concatenate([di_dot, wi_dot, ai_dot, ji_dot, jnp.array([ti_dot, ti_ddot], dtype=x.dtype)])

    @staticmethod
    def jax_cable_dynamics_single(x, u, params):
        """离散时间缆绳动力学 x_{k+1} = x_k + dt * RK4(f)."""
        dt = params['dt']
        f = MPC_Planner._jax_cable_continuous_dynamics
        k1 = f(x, u, params)
        k2 = f(x + 0.5 * dt * k1, u, params)
        k3 = f(x + 0.5 * dt * k2, u, params)
        k4 = f(x + dt * k3, u, params)
        return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    @staticmethod
    def _build_init_load_traj(ref_ul_seq, params, nxl):
        zero_xl = jnp.zeros((nxl,), dtype=ref_ul_seq.dtype)
        scxl_body = jax.vmap(lambda u: MPC_Planner.jax_load_dynamics(zero_xl, u, params))(ref_ul_seq)
        return jnp.concatenate(
            [scxl_body, jnp.zeros((1, nxl), dtype=ref_ul_seq.dtype)],
            axis=0,
        )

    @staticmethod
    def _build_init_cable_traj(ref_uq_mat, params, nxi, horizon):
        nq = ref_uq_mat.shape[0]
        nui = ref_uq_mat.shape[1]
        zero_xc = jnp.zeros((nq, nxi), dtype=ref_uq_mat.dtype)
        scxc_body = jax.vmap(
            lambda x0, ui: MPC_Planner.jax_cable_dynamics_single(x0, ui, params)
        )(zero_xc, ref_uq_mat)
        scxc_traj = jnp.concatenate(
            [
                jnp.broadcast_to(scxc_body[:, None, :], (nq, horizon, nxi)),
                jnp.zeros((nq, 1, nxi), dtype=ref_uq_mat.dtype),
            ],
            axis=1,
        )
        scuc_traj = jnp.broadcast_to(ref_uq_mat[:, None, :], (nq, horizon, nui))
        return scxc_traj, scuc_traj


    # ---------------------------------------------------------------------
    # 当前主线公共数学工具。
    # - open_loop_penalty_jax: SubP1 / 打包阶段都会直接用到的动态 ADMM 罚权重
    # - _q_2_rotation_jax: 当前 Julia 结果核对、SubP2 约束/残差诊断都会直接用到
    # 这两项属于当前稳定主线公共依赖，不应该埋在后面的 legacy 区里。
    # ---------------------------------------------------------------------
    @staticmethod
    def open_loop_penalty_jax(rho, gamma, a, ADMM_max):
        """
        [UseThis 版特有] 随迭代次数 a 变化的动态惩罚系数
        rho: 最终权重, gamma: 陡峭度, a: 当前迭代步, ADMM_max: 总步数
        """
        p_min = 1e-3
        return p_min + (rho - p_min) * 1.0 / (1.0 + jnp.exp(-gamma * (a - (ADMM_max - 1) / 2.0)))

    @staticmethod
    def _q_2_rotation_jax(q):
        """新增的 jax 版本的四元数转旋转矩阵 无归一化版本以保持导数平滑"""
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        return jnp.array([
            [2*(q0**2 + q1**2)-1, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 2*(q0**2 + q2**2)-1, 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 2*(q0**2 + q3**2)-1]
        ])

    @staticmethod
    def jax_load_stage_cost(x, u, params):
        """新增的 jax 版本的负载的单步运行代价 (Running Cost)"""
        # 1. 基础 Tracking 误差 (靠近参考轨迹)
        diff_x_ref = x - params['ref_x']
        diff_u_ref = u - params['ref_u']

        # 2. ADMM 一致性误差 (靠近 SubP2 算出的共识 scx, scu)
        # 这里 y 是对偶变量 (Lagrangian multiplier)，rho 是罚权重
        # 对应原版 resid_xl = x - scxl + scxL/p
        resid_x = x - params['scx'] + params['y_x'] / (params['rho_lx'] + 1e-6)
        resid_u = u - params['scu'] + params['y_u'] / (params['rho_lu'] + 1e-6)

        # 3. 汇总加权和
        # 注意：params['Q'] 和 params['R'] 应该是对角阵或权重向量
        cost = 0.5 * (jnp.sum(diff_x_ref**2 * params['Q_weight']) +
                    jnp.sum(diff_u_ref**2 * params['R_weight']))

        # ADMM 罚项
        cost += 0.5 * params['rho_lx'] * jnp.sum(resid_x**2)
        cost += 0.5 * params['rho_lu'] * jnp.sum(resid_u**2)

        return cost


    @staticmethod
    def jax_load_terminal_cost(x, params):
        """负载终点代价：统一使用 rho_lx"""
        # 1. 基础误差
        diff_x_ref = x - params['ref_x']

        # 2. 一致性残差 (修正：将 rho 改为 rho_lx)
        resid_x = x - params['scx'] + params['y_x'] / (params['rho_lx'] + 1e-6)

        # 3. 汇总
        cost = 0.5 * jnp.sum(diff_x_ref**2 * params['Q_terminal_weight'])
        cost += 0.5 * params['rho_lx'] * jnp.sum(resid_x**2)

        return cost    

    @staticmethod
    def jax_cable_stage_cost(x, u, params):
        """单根缆绳单步运行代价：区分状态和控制罚项"""
        # 1. 基础 Tracking 误差
        diff_x_ref = x - params['ref_x_i']
        diff_u_ref = u - params['ref_u_i']

        # 2. ADMM 一致性残差 (使用各自的 rho)
        # pix_dis 对应状态, piu_dis 对应控制
        resid_x = x - params['scx_i'] + params['y_x_i'] / (params['rho_ix'] + 1e-6)#  1e-6 as Numerical Guard
        resid_u = u - params['scu_i'] + params['y_u_i'] / (params['rho_iu'] + 1e-6)

        # 3. 汇总 Cost
        # 基础跟踪权重 (Qi, Ri)
        cost = 0.5 * (jnp.sum(diff_x_ref**2 * params['Qi_weight']) +
                    jnp.sum(diff_u_ref**2 * params['Ri_weight']))

        # 动态 ADMM 罚项
        cost += 0.5 * params['rho_ix'] * jnp.sum(resid_x**2)
        cost += 0.5 * params['rho_iu'] * jnp.sum(resid_u**2)

        return cost

    @staticmethod
    def jax_cable_terminal_cost(x, params):
        """单根缆绳终点代价"""
        diff_x_ref = x - params['ref_x_i']
        # 终点通常只看状态一致性
        resid_x = x - params['scx_i'] + params['y_x_i'] / (params['rho_ix'] + 1e-6)

        cost = 0.5 * jnp.sum(diff_x_ref**2 * params['Qi_terminal_weight'])
        cost += 0.5 * params['rho_ix'] * jnp.sum(resid_x**2)

        return cost

    # 当前稳定主线会直接走这里。
    # 这是负载侧 SubP1 的 fast path：只保留前向求解真正要用的轨迹结果，
    # 避免把旧 DDP 路径里那些后续根本不用的导数和中间量也一起搬出来。
    def _solve_load_subp1_forward_only(self, xl_fb, Ref_xl, Ref_ul, paral, scxl_traj, scul_traj, y_xl, y_ul, i_admm, return_host=False):
        """直接构造 batched tensors，只求 SubP1 轨迹，不提取未使用的导数。"""
        N, nx, nu = int(self.N), int(self.nxl), int(self.nul)
        weight_para = jnp.asarray(paral, dtype=jnp.float64).reshape(-1)
        rho_lx = self.open_loop_penalty_jax(weight_para[-4], weight_para[-2], i_admm, self.max_iter_ADMM)
        rho_lu = self.open_loop_penalty_jax(weight_para[-3], weight_para[-1], i_admm, self.max_iter_ADMM)

        ref_x_traj = jnp.asarray(Ref_xl, dtype=jnp.float64).reshape(N + 1, nx)
        ref_u_traj = jnp.asarray(Ref_ul, dtype=jnp.float64).reshape(N, nu)
        scx_traj = jnp.asarray(scxl_traj, dtype=jnp.float64).reshape(N + 1, nx)
        scu_traj = jnp.asarray(scul_traj, dtype=jnp.float64).reshape(N, nu)
        yx_traj = jnp.asarray(y_xl, dtype=jnp.float64).reshape(N + 1, nx)
        yu_traj = jnp.asarray(y_ul, dtype=jnp.float64).reshape(N, nu)

        x0_b = jnp.asarray(xl_fb, dtype=jnp.float64).reshape(1, nx)
        u_init_b = ref_u_traj[None, ...]
        params_b = {
            'static': {
                'ml': jnp.full((1,), float(self.ml), dtype=jnp.float64),
                'Jl': jnp.asarray(self.Jl, dtype=jnp.float64)[None, ...],
                'Jl_inv': jnp.asarray(self.Jl_inv, dtype=jnp.float64)[None, ...],
                'dt': jnp.full((1,), float(self.dt), dtype=jnp.float64),
                'rho_lx': jnp.asarray([rho_lx], dtype=jnp.float64),
                'rho_lu': jnp.asarray([rho_lu], dtype=jnp.float64),
                'Q_weight': jnp.asarray(weight_para[0:nx], dtype=jnp.float64)[None, ...],
                'R_weight': jnp.asarray(weight_para[2 * nx:2 * nx + nu], dtype=jnp.float64)[None, ...],
                'Q_terminal_weight': jnp.asarray(weight_para[nx:2 * nx], dtype=jnp.float64)[None, ...],
            },
            'stage': {
                'ref_x': ref_x_traj[:-1][None, ...],
                'ref_u': ref_u_traj[None, ...],
                'scx': scx_traj[:-1][None, ...],
                'scu': scu_traj[None, ...],
                'y_x': yx_traj[:-1][None, ...],
                'y_u': yu_traj[None, ...],
            },
            'terminal': {
                'ref_x': ref_x_traj[-1][None, ...],
                'scx': scx_traj[-1][None, ...],
                'y_x': yx_traj[-1][None, ...],
            },
        }

        cfg = ILQRConfig(max_iters=10, tol_g_norm=1e-2)
        results = self._load_solver_jit(x0_b, u_init_b, params_b, cfg=cfg)
        if return_host:
            xs_np = np.array(results.xs)
            us_np = np.array(results.us)
            return xs_np[0], us_np[0]
        return results.xs[0], results.us[0]

    # 当前稳定主线会直接走这里。
    # 这是多机缆绳侧 SubP1 的 fast path，和上面的负载版配套。
    # 当前 0.94s 那条主线就是靠这两个 forward-only 路径把 SubP1 压下来的。
    def _solve_cable_subp1_forward_only(self, xq_fb, ref_xq, ref_uq, paraC, scxc_traj, scuc_traj, y_xc, y_uc, i_admm, return_host=False):
        """直接构造 batched tensors，只求缆绳 SubP1 轨迹，不提取未使用的导数。"""
        B, N, nx, nu = int(self.nq), int(self.N), int(self.nxi), int(self.nui)
        weight_para = jnp.asarray(paraC, dtype=jnp.float64).reshape(-1)
        rho_ix = self.open_loop_penalty_jax(weight_para[-4], weight_para[-2], i_admm, self.max_iter_ADMM)
        rho_iu = self.open_loop_penalty_jax(weight_para[-3], weight_para[-1], i_admm, self.max_iter_ADMM)

        x0_batch = jnp.asarray(xq_fb, dtype=jnp.float64).reshape(B, nx)
        ref_x_batch = jnp.asarray(ref_xq, dtype=jnp.float64).reshape(B, N + 1, nx)
        ref_u_batch = jnp.asarray(ref_uq, dtype=jnp.float64).reshape(B, nu)
        scx_batch = jnp.asarray(scxc_traj, dtype=jnp.float64).reshape(B, N + 1, nx)
        scu_batch = jnp.asarray(scuc_traj, dtype=jnp.float64).reshape(B, N, nu)
        yx_batch = jnp.asarray(y_xc, dtype=jnp.float64).reshape(B, N + 1, nx)
        yu_batch = jnp.asarray(y_uc, dtype=jnp.float64).reshape(B, N, nu)
        u_init_batch = jnp.tile(ref_u_batch[:, None, :], (1, N, 1))

        qi_weight = jnp.asarray(weight_para[0:nx], dtype=jnp.float64)
        ri_weight = jnp.asarray(weight_para[2 * nx:2 * nx + nu], dtype=jnp.float64)
        qi_terminal_weight = jnp.asarray(weight_para[nx:2 * nx], dtype=jnp.float64)
        params_b = {
            'static': {
                'rho_ix': jnp.full((B,), rho_ix, dtype=jnp.float64),
                'rho_iu': jnp.full((B,), rho_iu, dtype=jnp.float64),
                'dt': jnp.full((B,), float(self.dt), dtype=jnp.float64),
                'Qi_weight': jnp.tile(qi_weight[None, :], (B, 1)),
                'Ri_weight': jnp.tile(ri_weight[None, :], (B, 1)),
                'Qi_terminal_weight': jnp.tile(qi_terminal_weight[None, :], (B, 1)),
            },
            'stage': {
                'ref_x_i': ref_x_batch[:, :-1, :],
                'ref_u_i': jnp.tile(ref_u_batch[:, None, :], (1, N, 1)),
                'scx_i': scx_batch[:, :-1, :],
                'scu_i': scu_batch,
                'y_x_i': yx_batch[:, :-1, :],
                'y_u_i': yu_batch,
            },
            'terminal': {
                'ref_x_i': ref_x_batch[:, -1, :],
                'scx_i': scx_batch[:, -1, :],
                'y_x_i': yx_batch[:, -1, :],
            },
        }

        cfg = ILQRConfig(
            max_iters=10,
            tol_g_norm=1e-2,
            reg_init=1e-6,
            reg_mult_inc=10.0,
            reg_mult_dec=1.0,
            line_search_alphas=(1.0, 0.5, 0.25, 0.125, 0.0625),
        )
        results = self._cable_solver_jit(x0_batch, u_init_batch, params_b, cfg=cfg)
        if return_host:
            return np.array(results.xs), np.array(results.us)
        return results.xs, results.us


    @staticmethod
    def _ipoptax_unpack(w, dims):
        """将决策向量 w 拆解为负载和缆绳的物理量"""
        nxl, nul, nxi, nui, nq, *_ = dims
        xl = w[0:nxl]
        ul = w[nxl:nxl+nul]
        # 缆绳部分 reshape
        rem = w[nxl+nul:].reshape(nq, nxi + nui)
        xc_mat = rem[:, 0:nxi]
        uc_mat = rem[:, nxi:]
        return xl, ul, xc_mat, uc_mat
    
    @staticmethod
    # 当前稳定 Julia 路线仍然会直接复用这个数学定义，
    # 但用途主要是：
    # - 导出 snapshot/reference 指标
    # - 核对 Julia 解的 objective / residual
    # 不是当前正式主线里的实际 SubP2 求解器。
    def ipoptax_objective(w, params_t, dims):
        xl, ul, xc_mat, uc_mat = MPC_Planner._ipoptax_unpack(w, dims)
        active_u = 1.0 - params_t['is_terminal']

        # 负载一致性项: 状态与控制分别使用 rho_lx / rho_lu
        res_xl = xl - params_t['xl_ideal'] + params_t['y_xl'] / (params_t['rho_lx'] + 1e-6)
        res_ul = ul - params_t['ul_ideal'] + params_t['y_ul'] / (params_t['rho_lu'] + 1e-6)
        cost = 0.5 * params_t['rho_lx'] * jnp.sum(res_xl**2)
        cost += active_u * 0.5 * params_t['rho_lu'] * jnp.sum(res_ul**2)

        # 缆绳一致性项: 状态与控制分别使用 rho_ix / rho_iu
        res_xc = xc_mat - params_t['xc_ideal'] + params_t['y_xc'] / (params_t['rho_ix'] + 1e-6)
        res_uc = uc_mat - params_t['uc_ideal'] + params_t['y_uc'] / (params_t['rho_iu'] + 1e-6)
        cost += 0.5 * params_t['rho_ix'] * jnp.sum(res_xc**2)
        cost += active_u * 0.5 * params_t['rho_iu'] * jnp.sum(res_uc**2)

        return cost

    @staticmethod
    # 同上：当前稳定 Julia 路线会复用它来算等式残差诊断，
    # 不是当前正式主线里的实际 SubP2 求解器入口。
    def ipoptax_equality(w, params_t, dims):
        xl, ul, xc_mat, uc_mat = MPC_Planner._ipoptax_unpack(w, dims)
        eqs = []
        active_u = 1.0 - params_t['is_terminal']

        # (1) 负载四元数归一化
        ql = xl[6:10]
        eqs.append(jnp.sum(ql**2) - 1.0)

        # (2) 每根缆绳的方向向量归一化
        di_vecs = xc_mat[:, 0:3] # (nq, 3)
        eqs.append(jnp.sum(di_vecs**2, axis=1) - 1.0)

        # (3) Wrench Consensus (力与力矩的一致性)
        # 负载挂点位置 (ra) + 绳子方向 * 张力 = 总合力/合力矩
        Rl = MPC_Planner._q_2_rotation_jax(ql)
        ti_mags = xc_mat[:, 12] # 张力在状态中 (xi = [di, wi, ai, ji, ti, vti])
        fi_inertial = di_vecs * ti_mags[:, None]
        # 将力转到机体系计算力矩
        fi_body = (Rl.T @ fi_inertial.T).T

        # 合力一致性 (Pt 矩阵派上用场了)
        # Pt @ [f1, f2... fn] = target_wrench (6维)
        wrench_generated = params_t['Pt'] @ fi_body.flatten()
        # 原版约束比较的是机体系总绳力矩与 [Rl^T*Fl, Ml]
        Fl_body = Rl.T @ ul[0:3]
        wrench_target = jnp.concatenate([Fl_body, ul[3:6]])
        eqs.append(active_u * (wrench_generated - wrench_target))

        return jnp.concatenate([jnp.atleast_1d(e).flatten() for e in eqs])

    @staticmethod
    # 同上：当前稳定 Julia 路线会复用它来算不等式残差诊断，
    # 不是当前正式主线里的实际 SubP2 求解器入口。
    def ipoptax_inequality(w, params_t, dims):
        xl, ul, xc_mat, uc_mat = MPC_Planner._ipoptax_unpack(w, dims)
        nq = int(dims[4])
        num_dis = max(1, int(dims[5])) if len(dims) > 5 else 1
        ineqs = []
        eps_margin = 1e-2
        active_u = 1.0 - params_t['is_terminal']

        # (1) 负载避障: po >= 1e-2  <=> 1e-2 - po <= 0
        pl = xl[0:3]
        Rl = MPC_Planner._q_2_rotation_jax(xl[6:10])
        di_vecs = xc_mat[:, 0:3]
        wi_vecs = xc_mat[:, 3:6]
        dwi_vecs = xc_mat[:, 6:9]
        ti_mags = xc_mat[:, 12]
        ra_mat = params_t['ra'].T

        safe_r_l = params_t['ro'] + 0.5 * params_t['rq']
        dist_l_obs1 = jnp.sum((pl[:2] - params_t['pob1'][:2])**2)
        dist_l_obs2 = jnp.sum((pl[:2] - params_t['pob2'][:2])**2)
        ineqs.append(safe_r_l**2 + eps_margin - dist_l_obs1)
        ineqs.append(safe_r_l**2 + eps_margin - dist_l_obs2)

        # (2) 无人机避障 (kc=num_dis 末端位置): go >= 1e-2
        pi_mat = pl[None, :] + (Rl @ params_t['ra']).T + params_t['cl0'] * di_vecs
        safe_r_q = params_t['ro'] + 2.0 * params_t['rq']
        dist_to_obs1 = jnp.sum((pi_mat[:, :2] - params_t['pob1'][:2])**2, axis=1)
        dist_to_obs2 = jnp.sum((pi_mat[:, :2] - params_t['pob2'][:2])**2, axis=1)
        ineqs.append(safe_r_q**2 + eps_margin - dist_to_obs1)
        ineqs.append(safe_r_q**2 + eps_margin - dist_to_obs2)

        # (3) 缆绳交叉与相互间隔: gij >= 1e-2, gio 区间约束
        pair_terms = []
        for kc in range(1, num_dis + 1):
            frac = float(kc) / float(num_dis)
            pib = ra_mat + frac * params_t['cl0'] * (Rl.T @ di_vecs.T).T

            min_pair_d2 = (frac * 4.0 * params_t['rq'])**2 + eps_margin
            for i in range(nq):
                for j in range(i + 1, nq):
                    dij_sq = jnp.sum((pib[i, :2] - pib[j, :2])**2)
                    pair_terms.append(min_pair_d2 - dij_sq)

            if kc == num_dis:
                ei = ra_mat / (jnp.linalg.norm(ra_mat, axis=1, keepdims=True) + 1e-9)
                ei_pib = jnp.sum(ei[:, :2] * pib[:, :2], axis=1)
                ineqs.append(ei_pib - (params_t['rl'] + params_t['cl0']))
                ineqs.append((-params_t['rl']) - ei_pib)
        if pair_terms:
            ineqs.append(jnp.stack(pair_terms))

        # (4) 张力限制 (ti_min <= ti <= ti_max)
        ineqs.append(params_t['t_min'] - ti_mags)
        ineqs.append(ti_mags - params_t['t_max'])

        # (5) 控制上下界（原版通过 lbx/ubx 实现；这里转为不等式）
        ineqs.append(active_u * (uc_mat - params_t['ui_bound']))
        ineqs.append(active_u * (-uc_mat - params_t['ui_bound']))

        # (6) 推力约束: 1e-2 <= ||fi||^2 <= fmax^2（terminal 步原版未显式加入）
        wl = xl[10:13]
        Fl = ul[0:3]
        Ml = ul[3:6]
        al = -params_t['g'] * jnp.array([0.0, 0.0, 1.0]) + Fl / params_t['ml']
        awl = params_t['Jl_inv'] @ (Ml - jnp.cross(wl, params_t['Jl'] @ wl))

        def _thrust_sq_i(ri, di_i, wi_i, dwi_i, ti_i):
            rot_term = Rl @ (jnp.cross(wl, jnp.cross(wl, ri)) + jnp.cross(awl, ri))
            cable_term = params_t['cl0'] * (
                jnp.cross(dwi_i, di_i) + jnp.cross(wi_i, jnp.cross(wi_i, di_i))
            )
            fi = params_t['mq'] * (
                al + rot_term + cable_term + params_t['g'] * jnp.array([0.0, 0.0, 1.0])
            ) + di_i * ti_i
            return jnp.sum(fi**2)

        thrust_sq = jax.vmap(_thrust_sq_i)(ra_mat, di_vecs, wi_vecs, dwi_vecs, ti_mags)
        ineqs.append(active_u * (thrust_sq - params_t['fmax']**2))
        ineqs.append(active_u * (eps_margin - thrust_sq))

        return jnp.concatenate([jnp.atleast_1d(i).flatten() for i in ineqs])

    # 当前稳定主线会直接复用这个解包函数。
    # 不管 SubP2 是纯 JAX 还是外部 Julia worker 解出来的，只要返回的是 batch 决策变量，
    # 最后都要靠这里还原成 scxl/scul/scxc/scuc 轨迹格式。
    def _unpack_subp2_results(self, w_opt_batch):
        """
        [JAX 血管函数] 将 SubP2 并行算出的 Tensor 拆解回 ADMM 轨迹格式
        w_opt_batch shape: (N+1, Total_Dim)
        """
        N, nq = self.N, int(self.nq)
        nxl, nul, nxi, nui = self.nxl, self.nul, self.nxi, self.nui

        # 1. 拆解负载 (Payload) 部分
        scxl_traj = w_opt_batch[:, 0:nxl] # (N+1, 13)
        scul_traj = w_opt_batch[:N, nxl:nxl+nul] # (N, 6) 控制量只取前 N 个

        # 2. 拆解缆绳 (Cables) 部分
        # 把剩下的变量 reshape 为 (N+1, nq, nxi + nui)
        cables_part = w_opt_batch[:, nxl+nul:].reshape(N + 1, nq, nxi + nui)

        # 分离状态和控制
        scxc_batch = cables_part[:, :, 0:nxi] # (N+1, nq, nxi)
        scuc_batch = cables_part[:N, :, nxi:] # (N, nq, 4)

        # 为了兼容原版代码中 list of arrays 的习惯，我们进行最后的转换
        scxc_traj_list = []
        scuc_traj_list = []
        for i in range(nq):
            # scxc_traj_list 里的每个元素是 (N+1, nxi)
            scxc_traj_list.append(np.array(scxc_batch[:, i, :]))
            # scuc_traj_list 里的每个元素是 (N, 4)
            scuc_traj_list.append(np.array(scuc_batch[:, i, :]))

        return {
            'scxl_traj': np.array(scxl_traj),
            'scul_traj': np.array(scul_traj),
            'scxc_traj': scxc_traj_list,
            'scuc_traj': scuc_traj_list
        }

    # 当前稳定主线会间接走这里。
    # 这是 SubP2 batch 语义的最低层构造器：给定 SubP1 结果、参考轨迹、对偶变量和 rho，
    # 统一整理成两份东西：
    # 1. w_init: 每个时间步一个扁平决策向量初值，形状 (N+1, total_dim)
    # 2. params_batch: 每个时间步对应的一整包参数，供 pure JAX / Julia 两条 SubP2 路线复用
    #
    # 这里的“规范形状”非常重要：
    # - 时间维永远放第一维，方便后续按 step 做 vmap / threaded solve
    # - 缆绳相关量统一整理成 (N+1, nq, ...)，避免不同调用方自己猜转置方向
    # - 控制量、控制对偶在 terminal 步都补 0，保持每个 step 的字段形状完全一致
    def _build_subp2_batch_from_components(
        self,
        xl_ideal,
        ul_ideal,
        xc_ideal,
        uc_ideal,
        Ref_xl,
        Ref_ul,
        ref_xq,
        ref_uq,
        y_xl,
        y_ul,
        y_xc,
        y_uc,
        rho_lx,
        rho_lu,
        rho_ix,
        rho_iu,
        admm_iter,
    ):
        N, nq = self.N, int(self.nq)
        nxl, nul, nxi, nui = self.nxl, self.nul, self.nxi, self.nui

        # --- 1. 把所有输入都标准化成明确的 dense tensor 形状 ---
        # 这里先保留“调用方最自然的输入布局”：
        # - 负载量通常已经是 (N+1, *) / (N, *)
        # - 缆绳量通常还是 (nq, N+1, *) / (nq, N, *)
        ref_xl = jnp.asarray(Ref_xl, dtype=jnp.float64).reshape(N + 1, nxl)
        ref_ul = jnp.asarray(Ref_ul, dtype=jnp.float64).reshape(N, nul)
        ref_xq_batch = jnp.asarray(ref_xq, dtype=jnp.float64).reshape(nq, N + 1, nxi)
        ref_uq_batch = jnp.asarray(ref_uq, dtype=jnp.float64).reshape(nq, nui)

        xl_ideal_arr = jnp.asarray(xl_ideal, dtype=jnp.float64).reshape(N + 1, nxl)
        ul_ideal_arr = jnp.asarray(ul_ideal, dtype=jnp.float64).reshape(N, nul)
        xc_ideal_arr = jnp.asarray(xc_ideal, dtype=jnp.float64).reshape(nq, N + 1, nxi)
        uc_ideal_arr = jnp.asarray(uc_ideal, dtype=jnp.float64).reshape(nq, N, nui)
        y_xl_arr = jnp.asarray(y_xl, dtype=jnp.float64).reshape(N + 1, nxl)
        y_ul_arr = jnp.asarray(y_ul, dtype=jnp.float64).reshape(N, nul)
        y_xc_arr = jnp.asarray(y_xc, dtype=jnp.float64).reshape(nq, N + 1, nxi)
        y_uc_arr = jnp.asarray(y_uc, dtype=jnp.float64).reshape(nq, N, nui)

        # --- 2. 组装按“时间步”为中心的 params_batch ---
        # 这是后续 SubP2 单步 evaluator / 单步求解器真正要读的参数表。
        # 每个字段都做成 (N+1, ...) 的形式，这样第 t 步只要索引 params_batch[field][t] 即可。
        params_batch = {
            # 标量型 ADMM/物理参数：直接沿时间维广播成 (N+1,)
            'rho_lx': jnp.full((N + 1,), rho_lx),
            'rho_lu': jnp.full((N + 1,), rho_lu),
            'rho_ix': jnp.full((N + 1,), rho_ix),
            'rho_iu': jnp.full((N + 1,), rho_iu),
            'admm_iter': jnp.full((N + 1,), admm_iter),
            'admm_total': jnp.full((N + 1,), float(self.max_iter_ADMM)),
            'rl': jnp.full((N + 1,), self.rl),
            'ro': jnp.full((N + 1,), self.ro),
            'rq': jnp.full((N + 1,), self.rq),
            'cl0': jnp.full((N + 1,), self.cl0),
            'p_bar': jnp.full((N + 1,), self.p_bar),
            'ml': jnp.full((N + 1,), self.ml),
            'mq': jnp.full((N + 1,), self.mq),
            'fmax': jnp.full((N + 1,), self.fmax),
            't_min': jnp.full((N + 1,), self.t_min),
            't_max': jnp.full((N + 1,), self.t_max),
            'ui_bound': jnp.full((N + 1,), self.ui_bound),
            'g': jnp.full((N + 1,), self.g),
            # terminal mask 让单步 evaluator 能区分最后一步哪些控制项应当失活。
            'is_terminal': jnp.concatenate([jnp.zeros((N,)), jnp.ones((1,))], axis=0),
            'eq_scale': jnp.full((N + 1,), 1e-1),
            'ineq_scale': jnp.full((N + 1,), 1e-2),
            # 矩阵型静态量：复制到每个时间步，避免后面每步再做广播/闭包捕获。
            'Pt': jnp.tile(self.Pt[None, ...], (N + 1, 1, 1)),
            'ra': jnp.tile(self.ra[None, ...], (N + 1, 1, 1)),
            'Jl': jnp.tile(self.Jl[None, ...], (N + 1, 1, 1)),
            'Jl_inv': jnp.tile(self.Jl_inv[None, ...], (N + 1, 1, 1)),
            'pob1': jnp.tile(self.pob1[None, ...], (N + 1, 1)),
            'pob2': jnp.tile(self.pob2[None, ...], (N + 1, 1)),
            # “ideal” 是当前轮 SubP1 给出的引导轨迹。
            # 对缆绳量要转成 (time, nq, dim)，这样每个 step 取出来就是该时刻所有无人机的数据。
            'xl_ideal': xl_ideal_arr,
            'ul_ideal': jnp.concatenate([ul_ideal_arr, jnp.zeros((1, nul), dtype=ul_ideal_arr.dtype)], axis=0),
            'xc_ideal': xc_ideal_arr.transpose(1, 0, 2),
            'uc_ideal': jnp.concatenate([uc_ideal_arr, jnp.zeros((nq, 1, nui), dtype=uc_ideal_arr.dtype)], axis=1).transpose(1, 0, 2),
            # “ref” 是 reference-based init / tracking 目标。
            # 当前 SubP2 仍然沿用“用参考轨迹做 x0”的习惯，而不是直接拿 ideal 当初值。
            'xl_ref': ref_xl,
            'ul_ref': jnp.concatenate([ref_ul, jnp.zeros((1, nul), dtype=ref_ul.dtype)], axis=0),
            'xc_ref': ref_xq_batch.transpose(1, 0, 2),
            'uc_ref': jnp.tile(ref_uq_batch[None, :, :], (N + 1, 1, 1)),
            # 对偶变量同样整理成按时间步读取的布局；terminal 控制对偶补 0。
            'y_xl': y_xl_arr,
            'y_ul': jnp.concatenate([y_ul_arr, jnp.zeros((1, nul), dtype=y_ul_arr.dtype)], axis=0),
            'y_xc': y_xc_arr.transpose(1, 0, 2),
            'y_uc': jnp.concatenate([y_uc_arr, jnp.zeros((nq, 1, nui), dtype=y_uc_arr.dtype)], axis=1).transpose(1, 0, 2),
        }

        # --- 3. 构造每个时间步的扁平初值 w_init ---
        # _ipoptax_unpack / Julia 单步模型都默认扁平变量顺序是：
        # [负载状态 xl, 负载控制 ul, (每架无人机的 [xi, ui] 交错拼接)]
        # 因此这里必须先把每架无人机的 (xc_ref, uc_ref) 按最后一维拼好，再整体 flatten。
        cables_init = jnp.concatenate([params_batch['xc_ref'], params_batch['uc_ref']], axis=2)
        w_init = jnp.concatenate(
            [params_batch['xl_ref'], params_batch['ul_ref'], cables_init.reshape(N + 1, -1)],
            axis=1,
        )
        return w_init, params_batch


    # 当前稳定主线会直接复用这个打包函数。
    # 纯 JAX SubP2 和 persistent Julia SubP2 共用同一份 batch 组织方式，
    # 区别只在于 batch 是交给 JAX solver 还是通过 IPC 送到 Julia worker。
    def _prepare_subp2_batch(self, Para2):
        """
        [JAX 血管函数] 为并行 ipoptax 准备全时域 Batch 数据
        """
        # 先看上游是否已经把本轮 SubP2 真正要吃的标准 batch 输入预构好了。
        # 注意：这不是“跨 ADMM 轮缓存旧结果”，因为每一轮的 ideal / y / rho 都会变。
        # 它只是“同一轮里提前构好的 (w_init, params_batch)”：
        # - _pack_para2 已经顺手把这轮 batch 拼好了
        # - 这里如果直接命中，就不用再从 Para2 各字段重组第二遍
        # 当前稳定主线（persistent Julia runner）通常都会命中这里。
        prebuilt = Para2.get('_prebuilt_subp2_batch', None)
        if prebuilt is not None:
            return prebuilt
        N, nq = self.N, int(self.nq)
        nxl, nul, nxi, nui = self.nxl, self.nul, self.nxi, self.nui

        # 走到这里说明调用方没有提供 _prebuilt_subp2_batch。
        # 下面这整段 fallback 重建逻辑主要是为了兼容旧路径/实验分支：
        # - pure JAX SubP2
        # - SQP / barrier 试验分支
        # - 手工构造 Para2 的诊断脚本
        # 对当前稳定主线来说，这一大段通常不是必经热路径。
        # --- 1. 基础物理量提取与广播 ---
        # 我们把这些静态常数复制 N+1 份，方便 vmap 同时读取
        params_batch = {
            'rho_lx': jnp.full((N+1,), Para2['rho_lx']),
            'rho_lu': jnp.full((N+1,), Para2['rho_lu']),
            'rho_ix': jnp.full((N+1,), Para2['rho_ix']),
            'rho_iu': jnp.full((N+1,), Para2['rho_iu']),
            'admm_iter': jnp.full((N+1,), Para2.get('admm_iter', 0.0)),
            'admm_total': jnp.full((N+1,), Para2.get('admm_total', float(self.max_iter_ADMM))),
            'rl':    jnp.full((N+1,), self.rl),
            'ro':    jnp.full((N+1,), self.ro),
            'rq':    jnp.full((N+1,), self.rq),
            'cl0':   jnp.full((N+1,), self.cl0),
            'p_bar': jnp.full((N+1,), self.p_bar),
            'ml':    jnp.full((N+1,), self.ml),
            'mq':    jnp.full((N+1,), self.mq),
            'fmax':  jnp.full((N+1,), self.fmax),
            't_min': jnp.full((N+1,), self.t_min),
            't_max': jnp.full((N+1,), self.t_max),
            'ui_bound': jnp.full((N+1,), self.ui_bound),
            'g':     jnp.full((N+1,), self.g),
            'is_terminal': jnp.concatenate([jnp.zeros((N,)), jnp.ones((1,))], axis=0),
            'eq_scale': jnp.full((N+1,), 1e-1),
            'ineq_scale': jnp.full((N+1,), 1e-2),

            # 广播 2D/3D 矩阵
            'Pt':    jnp.tile(self.Pt[None, ...], (N+1, 1, 1)),
            'ra':    jnp.tile(self.ra[None, ...], (N+1, 1, 1)),
            'Jl':    jnp.tile(self.Jl[None, ...], (N+1, 1, 1)),
            'Jl_inv': jnp.tile(self.Jl_inv[None, ...], (N+1, 1, 1)),
            'pob1':  jnp.tile(self.pob1[None, ...], (N+1, 1)),
            'pob2':  jnp.tile(self.pob2[None, ...], (N+1, 1)),

            # 提取 DDP 算出的理想值 (作为引导)
            'xl_ideal': jnp.array(Para2['xl_ideal']), # (N+1, 13)
            # 控制量补齐第 N+1 个点
            'ul_ideal': jnp.concatenate([jnp.array(Para2['ul_ideal']), jnp.zeros((1, nul))], axis=0), # (N+1, 6)
            # 缆绳部分：必须转置，确保时间维在第一位
            'xc_ideal': jnp.array(Para2['xc_ideal']).transpose(1, 0, 2), # (N+1, nq, nxi)
            'uc_ideal': jnp.concatenate([jnp.array(Para2['uc_ideal']), jnp.zeros((nq, 1, nui))], axis=1).transpose(1, 0, 2), # (N+1, nq, 4)

            # 对齐原版 IPOPT：SubP2 的 x0 使用参考轨迹
            'xl_ref': jnp.array(Para2['xl_ref']), # (N+1, nxl)
            'ul_ref': jnp.concatenate([jnp.array(Para2['ul_ref']), jnp.zeros((1, nul))], axis=0), # (N+1, nul)
            'xc_ref': jnp.array(Para2['xc_ref']).transpose(1, 0, 2), # (N+1, nq, nxi)
            'uc_ref': jnp.tile(jnp.array(Para2['uc_ref_single'])[None, :, :], (N + 1, 1, 1)), # (N+1, nq, nui)

            # 提取当前的对偶变量
            'y_xl': jnp.array(Para2['y_xl']), # (N+1, 13)
            'y_ul': jnp.concatenate([jnp.array(Para2['y_ul']), jnp.zeros((1, nul))], axis=0), # (N+1, 6)
            'y_xc': jnp.array(Para2['y_xc']).transpose(1, 0, 2), # (N+1, nq, nxi)
            'y_uc': jnp.concatenate([jnp.array(Para2['y_uc']), jnp.zeros((nq, 1, nui))], axis=1).transpose(1, 0, 2)  # (N+1, nq, 4)
        }

        # --- 2. 构造 w_init (每一行代表一个时刻的初始猜想) ---

        # (A) 负载状态与控制（reference-based init，与原版对齐）
        xl_init = params_batch['xl_ref']
        ul_init = params_batch['ul_ref']

        # (B)(C) 缆绳状态+控制需要按每架无人机交错拼接，才能匹配 _ipoptax_unpack:
        # rem = w[nxl+nul:].reshape(nq, nxi+nui) -> [xi_1, ui_1, xi_2, ui_2, ...]
        cables_init = jnp.concatenate(
            [params_batch['xc_ref'], params_batch['uc_ref']], axis=2
        )  # (N+1, nq, nxi+nui)
        cables_init_flat = cables_init.reshape(N + 1, -1)

        # (D) 终极拼接：[负载状态, 负载控制, 逐机交错后的缆绳变量]
        w_init = jnp.concatenate([xl_init, ul_init, cables_init_flat], axis=1)

        return w_init, params_batch


    # 当前 ADMM 前向主壳层。
    # 这个函数本身仍属于当前稳定主线的一部分；区别在于其中的 SubP2 调用位
    # self.jax_ADMM_SubP2(...) 会在 persistent Julia runner 里被替换成 Julia IPC 版本。
    #
    # 所以读当前稳定 fullflow 时，可以把这里理解成：
    # SubP1(JAX) -> SubP2(通常已被外部替换为 Julia) -> SubP3(JAX) 的总调度器。
    def jax_ADMM_forward_MPC(self, Ref_xl, Ref_ul, ref_xq, ref_uq, xl_fb, xq_fb, paral, parac, max_iter_ADMM):
        """
        [JAX 全并行版] ADMM 前向规划主流程
        """
        verbose = bool(getattr(self, "verbose", False))
        def finite_str(arr):
            a = np.asarray(arr)
            n_fin = int(np.isfinite(a).sum())
            return f"{n_fin}/{a.size}"

        self.max_iter_ADMM = max_iter_ADMM
        t_all = TM.time()
        profile = {
            "init_ms": 0.0,
            "total_ms": 0.0,
            "iters": [],
        }
        if verbose:
            print(f"[JAX-ADMM] start: horizon={self.N}, admm_iters={max_iter_ADMM}", flush=True)
            cable_mode = os.environ.get("JAX_CABLE_SUBP1_MODE", "batched").lower()
            print(f"[JAX-ADMM] cable_subp1_mode={cable_mode}", flush=True)

        # --- 1. 轨迹初始化 ---
        t_init = TM.time()
        scxl_traj, scul_traj, scxc_traj, scuc_traj = self._initialize_trajectories(Ref_xl, Ref_ul, ref_xq, ref_uq)
        y_xl, y_ul = jnp.zeros_like(scxl_traj), jnp.zeros_like(scul_traj)
        y_xc, y_uc = jnp.zeros_like(scxc_traj), jnp.zeros_like(scuc_traj)
        profile["init_ms"] = (TM.time() - t_init) * 1000.0
        if verbose:
            print(
                f"[JAX-ADMM] init done in {profile['init_ms']:.2f} ms | "
                f"scxl finite={finite_str(scxl_traj)} scxc finite={finite_str(scxc_traj)}",
                flush=True,
            )

        # --- 2. ADMM 迭代大循环 ---
        subp2_diag_history = []
        for i_admm in range(max_iter_ADMM):
            t_iter = TM.time()
            iter_profile = {
                "admm_iter": int(i_admm),
                "subp1_ms": 0.0,
                "subp2_ms": 0.0,
                "subp3_ms": 0.0,
                "iter_total_ms": 0.0,
            }
            if verbose:
                print(f"[JAX-ADMM] iter {i_admm+1}/{max_iter_ADMM} start", flush=True)

            # --- (A) 子问题 1：并行 DDP 规划 ---
            t_sp1 = TM.time()
            subp1_fast_mode = os.environ.get("JAX_SUBP1_FORWARD_ONLY_FAST_PATH", "direct").lower()
            cable_mode = os.environ.get("JAX_CABLE_SUBP1_MODE", "batched").lower()
            if subp1_fast_mode in ("1", "true", "direct") and cable_mode == "batched":
                xl_opt, ul_opt = self._solve_load_subp1_forward_only(
                    xl_fb, Ref_xl, Ref_ul, paral, scxl_traj, scul_traj, y_xl, y_ul, i_admm, return_host=True
                )
                xc_opt, uc_opt = self._solve_cable_subp1_forward_only(
                    xq_fb, ref_xq, ref_uq, parac, scxc_traj, scuc_traj, y_xc, y_uc, i_admm, return_host=True
                )
                xl_opt = jnp.asarray(xl_opt, dtype=jnp.float64)
                ul_opt = jnp.asarray(ul_opt, dtype=jnp.float64)
                xc_opt = jnp.asarray(xc_opt, dtype=jnp.float64)
                uc_opt = jnp.asarray(uc_opt, dtype=jnp.float64)
            elif subp1_fast_mode == "packaged" and cable_mode == "batched":
                # 非当前稳定主线：仍走旧 packaged SubP1 接口。
                # 这里先把输入压成 legacy ParaL/ParaC 长向量，再交给兼容包装器拆包求解。
                ParaL = self._pack_paraL(xl_fb, Ref_xl, Ref_ul, paral, scxl_traj, scul_traj, y_xl, y_ul, i_admm)
                ParaC = self._pack_paraC(xq_fb, ref_xq, ref_uq, parac, scxc_traj, scuc_traj, y_xc, y_uc, i_admm)

                sol1_load, _ = self.jax_MPC_Load_DDP_Planning_SubP1(ParaL, need_derivs=False)
                sol1_cable, _ = self.jax_MPC_Cable_DDP_Planning_SubP1(ParaC, need_derivs=False)

                xl_opt = sol1_load['xl_traj'][0]
                ul_opt = sol1_load['ul_traj'][0]
                xc_opt = jnp.array(sol1_cable['xc_traj']) # (nq, N+1, nxi)
                uc_opt = jnp.array(sol1_cable['uc_traj']) # (nq, N, 4)
            else:
                # 非当前稳定主线 fallback：保留旧 ParaL/ParaC 打包和 packaged SubP1 入口。
                ParaL = self._pack_paraL(xl_fb, Ref_xl, Ref_ul, paral, scxl_traj, scul_traj, y_xl, y_ul, i_admm)
                ParaC = self._pack_paraC(xq_fb, ref_xq, ref_uq, parac, scxc_traj, scuc_traj, y_xc, y_uc, i_admm)

                sol1_load, _ = self.jax_MPC_Load_DDP_Planning_SubP1(ParaL)
                sol1_cable, _ = self.jax_MPC_Cable_DDP_Planning_SubP1(ParaC)

                xl_opt = sol1_load['xl_traj'][0]
                ul_opt = sol1_load['ul_traj'][0]
                xc_opt = jnp.array(sol1_cable['xc_traj']) # (nq, N+1, nxi)
                uc_opt = jnp.array(sol1_cable['uc_traj']) # (nq, N, 4)
            iter_profile["subp1_ms"] = (TM.time() - t_sp1) * 1000.0
            if verbose:
                print(
                    f"[JAX-ADMM] iter {i_admm+1} subp1 done in {iter_profile['subp1_ms']:.2f} ms | "
                    f"xl={finite_str(xl_opt)} xc={finite_str(xc_opt)}",
                    flush=True,
                )

            # --- (B) 子问题 2：并行一致性投影 (ipoptax) ---
            t_sp2 = TM.time()
            Para2 = self._pack_para2(
                xl_opt, ul_opt, xc_opt, uc_opt,
                Ref_xl, Ref_ul, ref_xq, ref_uq,
                y_xl, y_ul, y_xc, y_uc, paral, parac, i_admm
            )
            sol2 = self.jax_ADMM_SubP2(Para2)
            subp2_diag_history.append(sol2.get('diag', {}))

            scxl_cons = sol2['scxl_traj']
            scul_cons = sol2['scul_traj']
            scxc_cons = jnp.array(sol2['scxc_traj'])
            scuc_cons = jnp.array(sol2['scuc_traj'])
            iter_profile["subp2_ms"] = (TM.time() - t_sp2) * 1000.0
            if verbose:
                print(
                    f"[JAX-ADMM] iter {i_admm+1} subp2 done in {iter_profile['subp2_ms']:.2f} ms | "
                    f"scxl={finite_str(scxl_cons)} scxc={finite_str(scxc_cons)}",
                    flush=True,
                )

            # --- (C) 子问题 3：对偶变量更新 ---
            # 调用我们之前改好的 JAX 版 SubP3
            t_sp3 = TM.time()
            sol3 = self.jax_ADMM_SubP3(
                xl_opt, scxl_cons, y_xl, ul_opt, scul_cons, y_ul,
                xc_opt, scxc_cons, y_xc, uc_opt, scuc_cons, y_uc,
                paral[-4], paral[-3], paral[-2], paral[-1],
                parac[-4], parac[-3], parac[-2], parac[-1],
                max_iter_ADMM, i_admm
            )

            # 更新对偶变量，存入下一轮
            y_xl, y_ul = sol3['scxL_traj_new'], sol3['scuL_traj_new']
            y_xc, y_uc = sol3['scxC_traj_new'], sol3['scuC_traj_new']

            # 同时更新 SubP2 用的共识变量
            scxl_traj, scul_traj = scxl_cons, scul_cons
            scxc_traj, scuc_traj = scxc_cons, scuc_cons
            iter_profile["subp3_ms"] = (TM.time() - t_sp3) * 1000.0
            iter_profile["iter_total_ms"] = (TM.time() - t_iter) * 1000.0
            profile["iters"].append(iter_profile)
            if verbose:
                print(
                    f"[JAX-ADMM] iter {i_admm+1} subp3 done in {iter_profile['subp3_ms']:.2f} ms | "
                    f"y_xl={finite_str(y_xl)} y_xc={finite_str(y_xc)}",
                    flush=True,
                )
                print(
                    f"[JAX-ADMM] iter {i_admm+1} total {iter_profile['iter_total_ms']:.2f} ms",
                    flush=True,
                )

            # --- (D) 收敛性检查 ---
            # ...

        profile["total_ms"] = (TM.time() - t_all) * 1000.0
        if verbose:
            print(f"[JAX-ADMM] finished in {profile['total_ms'] / 1000.0:.2f} s", flush=True)

        return {
            'load_trajectory': xl_opt,
            'cable_trajectories': xc_opt,
            'control_inputs': ul_opt,
            'subp2_diag_history': subp2_diag_history,
            'profile': profile,
        }

    # 当前稳定主线会直接走这里。
    # 这一步负责生成 ADMM 第 0 轮使用的共识初值，也是后面 benchmark 里 init_ms 的来源。
    def _initialize_trajectories(self, Ref_xl, Ref_ul, ref_xq, ref_uq):
        """
        [JAX 移植版] 轨迹初始化逻辑 (严格对应原版 L1624-L1650)
        """
        N, nq = self.N, self.nq
        params = self._get_static_params()

        # 原版 ADMM_forward_MPC 的 safe-copy 初始化并不是整条 rollout：
        # 每个时刻都从该切片当前值（初始全零）做一次单步动力学，terminal 保持为零。
        # 这里故意保留这个“坏初值”行为，先确保 forward 逻辑一致。
        ref_ul_seq = jnp.asarray(Ref_ul).reshape(N, self.nul)
        scxl_traj = self._init_load_traj_jit(ref_ul_seq, params, self.nxl)
        scul_traj = ref_ul_seq

        # 缆绳侧同理：每个时刻从零状态单独推进一步，terminal 保持为零。
        ref_uq_mat = jnp.asarray(ref_uq).reshape(nq, self.nui)
        scxc_traj, scuc_traj = self._init_cable_traj_jit(
            ref_uq_mat,
            params,
            self.nxi,
            N,
        )

        return scxl_traj, scul_traj, scxc_traj, scuc_traj

    def _get_static_params(self):
        """[JAX 血管函数] 返回 JAX 算子需要的静态物理字典"""
        cached = getattr(self, "_static_params_cache", None)
        if cached is None:
            cached = {
                'ml': float(self.ml),
                'dt': float(self.dt),
                'Jl': jnp.array(self.Jl, dtype=jnp.float64),
                'Jl_inv': jnp.array(self.Jl_inv, dtype=jnp.float64),
                'g': 9.81,
                'ez': jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)
            }
            self._static_params_cache = cached
        return cached

    # 当前稳定主线会直接走这里。
    # 这是新的 SubP2 输入构造器：不再沿用 ParaL/ParaC 那种 legacy 长向量接口，
    # 而是把 SubP1 输出、参考轨迹、对偶变量和 rho 组织成结构化字典。
    # persistent Julia runner 和纯 JAX SubP2 都复用这层语义。
    def _pack_para2(self, xl_opt, ul_opt, xc_opt, uc_opt,
                    Ref_xl, Ref_ul, ref_xq, ref_uq,
                    y_xl, y_ul, y_xc, y_uc, paral, paraC, i_admm):
        # 这一步是“上层 ADMM 语义 -> SubP2 输入语义”的适配层。
        # 它返回的 Para2 仍然是一个 Python dict，原因有两个：
        # 1. 纯 JAX SubP2 历史分支还能直接消费它；
        # 2. persistent Julia runner 可以优先取走里面已经预构好的 _prebuilt_subp2_batch，
        #    从而避免每轮再在 Python 里重复做一次 batch 组织。

        # --- 1. 先把 reference 轨迹整理成清晰的 numpy 形状 ---
        # 这些字段保存在 Para2 里，既方便诊断，也方便 _prepare_subp2_batch fallback 时重建。
        xl_ref_mat = np.array(Ref_xl).reshape(self.N + 1, self.nxl)
        ul_ref_mat = np.array(Ref_ul).reshape(self.N, self.nul)
        xc_ref = np.array([np.array(ref_xq[i]).reshape(self.N + 1, self.nxi) for i in range(self.nq)])
        uc_ref_single = np.array(ref_uq).reshape(self.nq, self.nui)

        # --- 2. 根据当前 ADMM 轮的超参数，先算出本轮真正使用的 rho ---
        # 这里不能直接把 paral/paraC 原样塞下去，因为 SubP2 关心的是“这一轮展开后的 rho 值”，
        # 不是生成 rho 的超参数本身。
        rho_lx = self.open_loop_penalty_jax(paral[-4], paral[-2], i_admm, self.max_iter_ADMM)
        rho_lu = self.open_loop_penalty_jax(paral[-3], paral[-1], i_admm, self.max_iter_ADMM)
        rho_ix = self.open_loop_penalty_jax(paraC[-4], paraC[-2], i_admm, self.max_iter_ADMM)
        rho_iu = self.open_loop_penalty_jax(paraC[-3], paraC[-1], i_admm, self.max_iter_ADMM)

        # --- 3. 直接预构一份“规范 batch” ---
        # 当前稳定 Julia 路线会优先复用这份结果；这样后续 runner 不用再靠 Para2 各字段重新拼一次。
        prebuilt_batch = self._build_subp2_batch_from_components(
            xl_opt, ul_opt, xc_opt, uc_opt,
            Ref_xl, Ref_ul, ref_xq, ref_uq,
            y_xl, y_ul, y_xc, y_uc,
            rho_lx, rho_lu, rho_ix, rho_iu,
            float(i_admm),
        )

        # --- 4. 返回结构化 Para2 字典 ---
        # 这里保留两层语义：
        # - 上层易读字段（ideal/ref/y/rho/...），方便日志、诊断和 fallback 重建
        # - _prebuilt_subp2_batch，方便当前稳定 Julia 主线直接取走
        return {
            # ideal: 当前轮 SubP1 给出的引导轨迹
            'xl_ideal': xl_opt,  # (N+1, nxl)
            'ul_ideal': ul_opt,  # (N, nul)
            'xc_ideal': xc_opt,  # (nq, N+1, nxi)
            'uc_ideal': uc_opt,  # (nq, N, nui)
            # ref: 参考轨迹 / 参考控制
            'xl_ref': xl_ref_mat,           # (N+1, nxl)
            'ul_ref': ul_ref_mat,           # (N, nul)
            'xc_ref': xc_ref,               # (nq, N+1, nxi)
            'uc_ref_single': uc_ref_single, # (nq, nui)
            # 当前 ADMM 对偶变量
            'y_xl': y_xl,
            'y_ul': y_ul,
            'y_xc': y_xc,
            'y_uc': y_uc,
            # 当前轮已经展开好的 rho
            'rho_lx': rho_lx,
            'rho_lu': rho_lu,
            'rho_ix': rho_ix,
            'rho_iu': rho_iu,
            # 运行时辅助信息
            'admm_iter': float(i_admm),
            'admm_total': float(self.max_iter_ADMM),
            'pob1': self.pob1,
            'pob2': self.pob2,
            # 当前主线优先复用的规范 batch：
            # prebuilt_batch = (w_init, params_batch)
            '_prebuilt_subp2_batch': prebuilt_batch,
        }


    @staticmethod
    def jax_subp3_update(xl_opt, scxl_cons, y_xl_old, rho_lx, ul_opt, scul_cons, y_ul_old, rho_lu):
        """
        [JAX 版] 负载对偶变量更新
        """
        # 对偶变量更新公式：y_new = y_old + rho * (primal - consensus)
        y_xl_new = y_xl_old + rho_lx * (xl_opt - scxl_cons)
        y_ul_new = y_ul_old + rho_lu * (ul_opt - scul_cons)
        return y_xl_new, y_ul_new

    @staticmethod
    def jax_subp3_update_cable(xc_opt, scxc_cons, y_xc_old, rho_ix, uc_opt, scuc_cons, y_uc_old, rho_iu):
        """
        [JAX 版] 缆绳对偶变量并行更新 (利用 JAX 的自动广播机制)
        """
        y_xc_new = y_xc_old + rho_ix * (xc_opt - scxc_cons)
        y_uc_new = y_uc_old + rho_iu * (uc_opt - scuc_cons)
        return y_xc_new, y_uc_new

    def jax_ADMM_SubP3(self, xl_traj, scxl_traj, scxL_traj, ul_traj, scul_traj, scuL_traj,
                  xc_traj, scxc_traj, scxC_traj, uc_traj, scuc_traj, scuC_traj,
                  px, pu, gammax, gammau, pix, piu, gammaix, gammaiu, ADMM_max, i_admm):
        """
        子问题 3 的类成员包装器
        """
        # 1. 计算当前步的动态惩罚系数
        rho_lx = self.open_loop_penalty_jax(px, gammax, i_admm, ADMM_max)
        rho_lu = self.open_loop_penalty_jax(pu, gammau, i_admm, ADMM_max)

        # 2. 调用 JAX 纯函数更新负载对偶变量
        y_xl_new, y_ul_new = self.jax_subp3_update(
            xl_traj, scxl_traj, scxL_traj, rho_lx,
            ul_traj, scul_traj, scuL_traj, rho_lu
        )

        # 3. 计算缆绳的动态惩罚 (对每一个无人机)
        # 假设所有无人机共享这套参数，JAX 会处理广播
        rho_ix = self.open_loop_penalty_jax(pix, gammaix, i_admm, ADMM_max)
        rho_iu = self.open_loop_penalty_jax(piu, gammaiu, i_admm, ADMM_max)

        # 4. 调用 JAX 纯函数更新缆绳对偶变量
        y_xc_new, y_uc_new = self.jax_subp3_update_cable(
            xc_traj, scxc_traj, scxC_traj, rho_ix,
            uc_traj, scuc_traj, scuC_traj, rho_iu
        )

        return {
            'scxL_traj_new': np.array(y_xl_new),
            'scuL_traj_new': np.array(y_ul_new),
            'scxC_traj_new': np.array(y_xc_new),
            'scuC_traj_new': np.array(y_uc_new)
        }
    # ---------------------------------------------------------------------
    # 以下是下沉后的"非当前稳定主线"方法区。
    # 先是旧 SubP1 packaged / derivs / legacy 分支，再往后才是 pure JAX SubP2 实验区。
    # 当前 persistent Julia 主线默认不直接走这里。
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # 非当前稳定主线：SubP1 的旧 packaged / derivs / legacy 分支。
    # 这些方法保留是为了兼容旧接口、导出导数、以及和 legacy DDP 做对照。
    # ---------------------------------------------------------------------
    @staticmethod
    # 非当前稳定主线默认热路径。
    # 只有在旧 packaged SubP1 路径、对齐/诊断实验，或显式 need_derivs=True 时，
    # 才需要沿已求出的轨迹额外提取 Fx/Fu/Qxu/Quu_inv/K_FB 这些导数量。
    def _static_load_derivs(xs, us, params_b):
        """[JAX 静态算子] 批量提取负载轨迹导数，仅供 need_derivs/旧路径使用"""
        def _calc_step(x, u, p):
            dyn = MPC_Planner.jax_load_dynamics
            cost = MPC_Planner.jax_load_stage_cost
            # 计算动力学 Jacobian
            fx = jax.jacfwd(dyn, 0)(x, u, p)
            fu = jax.jacfwd(dyn, 1)(x, u, p)
            # 计算代价函数 Hessian 和 Jacobian (用于反馈增益 K_fb)
            luu = jax.hessian(cost, 1)(x, u, p)
            lxu = jax.jacfwd(jax.grad(cost, 0), 1)(x, u, p)
            # 正则化求逆 (确保数值稳定性)
            Quu_inv = jnp.linalg.inv(luu + 1e-6 * jnp.eye(u.shape[0]))
            K_fb = - Quu_inv @ lxu.T
            return Quu_inv, lxu, K_fb, fx, fu

        # 对整条轨迹进行映射
        def _scan_time(xs_seq, us_seq, p_single):
            def _slice_stage(t):
                p_t = dict(p_single["static"])
                p_t.update(jax.tree_util.tree_map(lambda a: a[t], p_single["stage"]))
                return p_t
            return jax.vmap(
                lambda t, x, u: _calc_step(x, u, _slice_stage(t)),
                in_axes=(0, 0, 0),
            )(jnp.arange(us_seq.shape[0]), xs_seq[:-1], us_seq)

        return jax.vmap(_scan_time, in_axes=(0, 0, 0))(xs, us, params_b)

    @staticmethod
    # 非当前稳定主线默认热路径。
    # 作用和 _static_load_derivs 一样：在轨迹已经解出来之后，再补提一遍局部线性化/二阶信息。
    # 当前 persistent Julia 主线默认只要 xc/uc 轨迹，不需要这里的导数包。
    def _static_cable_derivs(xs, us, params_b):
        """[JAX 静态算子] 批量提取缆绳轨迹导数，仅供 need_derivs/旧路径使用"""
        def _calc_step(x, u, p):
            dyn = MPC_Planner.jax_cable_dynamics_single
            cost = MPC_Planner.jax_cable_stage_cost
            fx = jax.jacfwd(dyn, 0)(x, u, p)
            fu = jax.jacfwd(dyn, 1)(x, u, p)
            luu = jax.hessian(cost, 1)(x, u, p)
            lxu = jax.jacfwd(jax.grad(cost, 0), 1)(x, u, p)
            Quu_inv = jnp.linalg.inv(luu + 1e-6 * jnp.eye(u.shape[0]))
            K_fb = - Quu_inv @ lxu.T
            return Quu_inv, lxu, K_fb, fx, fu

        def _scan_time(xs_seq, us_seq, p_single):
            def _slice_stage(t):
                p_t = dict(p_single["static"])
                p_t.update(jax.tree_util.tree_map(lambda a: a[t], p_single["stage"]))
                return p_t
            return jax.vmap(
                lambda t, x, u: _calc_step(x, u, _slice_stage(t)),
                in_axes=(0, 0, 0),
            )(jnp.arange(us_seq.shape[0]), xs_seq[:-1], us_seq)

        return jax.vmap(_scan_time, in_axes=(0, 0, 0))(xs, us, params_b)

    def _ensure_legacy_cable_ddp_ready(self):
        if getattr(self, "_legacy_cable_ddp_ready", False):
            return

        import Dynamics_load_cable_autotuning_2nd_COM_Dyn as original_dyn

        rp_curr = np.array(self.rp, dtype=float)
        if getattr(self, "m2", 0.0) != 0.0:
            rp_from_rg = np.array(self.rg, dtype=float) * (self.ml / self.m2)
            if rp_from_rg.shape == rp_curr.shape:
                rp_curr = rp_from_rg

        sysm_para = np.array([
            self.m1, self.m2,
            self.Jlcom[0, 0], self.Jlcom[1, 1], self.Jlcom[2, 2],
            self.rl, self.nq, self.rq, self.mq, self.fmax,
            self.cl0, self.ro
        ], dtype=float)

        sysm = original_dyn.multilift_model(sysm_para, self.dt)
        sysm.Rotational_Inertia(rp_curr)
        sysm.model()

        self.Rotational_Inertia(rp_curr)
        self.SetStateVariables(sysm.xl, sysm.xi)
        self.SetCtrlVariables(sysm.ul, sysm.ui)
        self.SetDyns(sysm.model_l, sysm.model_i)
        self.SetWeightPara()
        self.SetPayloadCostDyn(int(self.max_iter_ADMM))
        self.SetCableCostDyn(int(self.max_iter_ADMM))
        self.Cable_derivatives_DDP_ADMM()
        self._legacy_cable_ddp_ready = True


    def _solve_cable_subp1_batched(self, ParaC, need_derivs=False):
        B, N, nx, nu = len(ParaC), int(self.N), self.nxi, self.nui
        x0_list, u_init_list, params_list = [], [], []

        # --- 1. 严格索引审计与数据打包 ---
        for i in range(B):
            parai = ParaC[i]
            idx = 0
            # 初值 [nxi]
            xi_fb = parai[idx : idx+nx]; idx += nx
            # 参考轨迹 (N+1)步 [nxi*(N+1)]
            ref_x = parai[idx : idx+nx*(N+1)]; idx += nx*(N+1)
            # 线缆参考控制 (单步) [4]
            ref_ui = parai[idx : idx+nu]; idx += nu
            # 共识状态 (N+1)步
            scxi = parai[idx : idx+nx*(N+1)]; idx += nx*(N+1)
            # 状态对偶变量 (N+1)步
            scxI = parai[idx : idx+nx*(N+1)]; idx += nx*(N+1)
            # 共识控制 N步
            scui = parai[idx : idx+nu*N]; idx += nu*N
            # 控制对偶变量 N步
            scuI = parai[idx : idx+nu*N]; idx += nu*N
            # 权重参数 (UseThis 变长版)
            weight_para = parai[idx : -1]
            i_admm = parai[-1]

            # 计算动态惩罚项 rho (状态与控制解耦)
            rho_ix = self.open_loop_penalty_jax(weight_para[-4], weight_para[-2], i_admm, self.max_iter_ADMM)
            rho_iu = self.open_loop_penalty_jax(weight_para[-3], weight_para[-1], i_admm, self.max_iter_ADMM)

            x0_list.append(jnp.array(xi_fb, dtype=jnp.float64))
            # 原版 cable DDP 用单步 Ref_ui 复制为整条初值控制。
            u_init_list.append(jnp.tile(jnp.array(ref_ui, dtype=jnp.float64), (N, 1)))

            ref_x_traj = jnp.array(ref_x, dtype=jnp.float64).reshape(N + 1, nx)
            scx_traj = jnp.array(scxi, dtype=jnp.float64).reshape(N + 1, nx)
            yx_traj = jnp.array(scxI, dtype=jnp.float64).reshape(N + 1, nx)
            ref_u_vec = jnp.array(ref_ui, dtype=jnp.float64)
            scu_traj = jnp.array(scui, dtype=jnp.float64).reshape(N, nu)
            yu_traj = jnp.array(scuI, dtype=jnp.float64).reshape(N, nu)

            params_list.append({
                'static': {
                    'rho_ix': rho_ix,
                    'rho_iu': rho_iu,
                    'dt': float(self.dt),
                    'Qi_weight': jnp.array(weight_para[0:nx], dtype=jnp.float64),
                    'Ri_weight': jnp.array(weight_para[2*nx:2*nx+nu], dtype=jnp.float64),
                    'Qi_terminal_weight': jnp.array(weight_para[nx:2*nx], dtype=jnp.float64),
                },
                'stage': {
                    'ref_x_i': ref_x_traj[:-1],
                    'ref_u_i': jnp.tile(ref_u_vec[None, :], (N, 1)),
                    'scx_i': scx_traj[:-1],
                    'scu_i': scu_traj,
                    'y_x_i': yx_traj[:-1],
                    'y_u_i': yu_traj,
                },
                'terminal': {
                    'ref_x_i': ref_x_traj[-1],
                    'scx_i': scx_traj[-1],
                    'y_x_i': yx_traj[-1],
                },
            })

        # --- 2. 批量并行求解 ---
        params_b = jax.tree_util.tree_map(lambda *args: jnp.stack(args).astype(jnp.float64), *params_list)
        x0_batch = jnp.stack(x0_list).astype(jnp.float64)
        u_batch  = jnp.stack(u_init_list).astype(jnp.float64)

        # 尽量贴近 legacy DDP 的数值路径：
        # reg_init=1e-6, reg_up=10, accepted 后不主动衰减 reg，
        # line-search 走 1, 0.5, 0.25, 0.125, 0.0625
        cfg = ILQRConfig(
            max_iters=10,
            tol_g_norm=1e-2,
            reg_init=1e-6,
            reg_mult_inc=10.0,
            reg_mult_dec=1.0,
            line_search_alphas=(1.0, 0.5, 0.25, 0.125, 0.0625),
        )
        results = self._cable_solver_jit(x0_batch, u_batch, params_b, cfg=cfg)

        # --- 3. 结果还原与接口适配 ---
        xs_np, us_np = np.array(results.xs), np.array(results.us)
        opt_solc = {"xc_traj": [xs_np[i] for i in range(B)], "uc_traj": [us_np[i] for i in range(B)]}

        if not need_derivs:
            return opt_solc, None

        # --- 4. 批量提取导数 (Jacobian/Hessian) ---
        # 使用 __init__ 中定义的 _cable_derivs_jit
        derivs = self._cable_derivs_jit(results.xs, results.us, params_b)
        Quu_inv_np, Qxu_np, K_fb_np, Fx_np, Fu_np = [np.array(d) for d in derivs]

        OPt_sol_c = []
        for i in range(B):
            OPt_sol_c.append({
                'xi_traj': xs_np[i], 'ui_traj': us_np[i],
                'K_FB': K_fb_np[i], 'Quu_inv': Quu_inv_np[i],
                'Qxu': Qxu_np[i], 'Fx': Fx_np[i], 'Fu': Fu_np[i]
            })

        return opt_solc, OPt_sol_c

    def jax_MPC_Cable_DDP_Planning_SubP1(self, ParaC, need_derivs=False):
        """
        [JAX] 缆绳子问题 1 求解器
        支持 `batched` / `legacy` / `compare` 三种模式。
        当前默认只返回轨迹；只有显式 need_derivs=True 才额外提取导数包。
        """
        mode = os.environ.get("JAX_CABLE_SUBP1_MODE", "batched").lower()
        if mode == "legacy":
            self._ensure_legacy_cable_ddp_ready()
            return self.MPC_Cable_DDP_Planning_SubP1(ParaC)

        batched_sol = self._solve_cable_subp1_batched(ParaC, need_derivs=need_derivs)
        if mode != "compare":
            return batched_sol

        self._ensure_legacy_cable_ddp_ready()
        legacy_sol = self.MPC_Cable_DDP_Planning_SubP1(ParaC)

        batched_opt, _ = batched_sol
        legacy_opt, _ = legacy_sol
        for i in range(int(self.nq)):
            x_b = np.array(batched_opt["xc_traj"][i])
            x_l = np.array(legacy_opt["xc_traj"][i])
            u_b = np.array(batched_opt["uc_traj"][i])
            u_l = np.array(legacy_opt["uc_traj"][i])
            x_rmse = float(np.sqrt(np.mean((x_b - x_l) ** 2)))
            u_rmse = float(np.sqrt(np.mean((u_b - u_l) ** 2)))
            x_max = float(np.max(np.abs(x_b - x_l)))
            u_max = float(np.max(np.abs(u_b - u_l)))
            print(
                f"[JAX-CableDDP][compare] cable={i} "
                f"x_max={x_max:.3e} x_rmse={x_rmse:.3e} "
                f"u_max={u_max:.3e} u_rmse={u_rmse:.3e}",
                flush=True,
            )
        return batched_sol

    # 非当前稳定主线：旧 packaged SubP1 的负载参数打包器。
    # 只在 jax_ADMM_forward_MPC 的 packaged/fallback 分支里使用，
    # 目的是把当前轮负载侧输入重新压回 legacy ParaL 长向量格式。
    def _pack_paraL(self, xl_fb, Ref_xl, Ref_ul, paral, scxl_traj, scul_traj, y_xl, y_ul, i_admm):
        # 负载只有一个，所以返回一个只包含一个元素的列表。
        # 顺序必须严格匹配旧 packaged SubP1 入口的拆包顺序：
        # 初值 -> 参考 x/u -> 共识 x/u -> 对偶 x/u -> 超参数 -> ADMM 迭代号。
        paral_arr = np.concatenate((
            xl_fb.flatten(),
            Ref_xl.flatten(),
            Ref_ul.flatten(),
            scxl_traj.flatten(),
            y_xl.flatten(),
            scul_traj.flatten(),
            y_ul.flatten(),
            paral.flatten(),
            [float(i_admm)]
        ))
        return [paral_arr]  # 返回列表以适配 B=1 的 batched packaged 接口

    # 非当前稳定主线：旧 packaged SubP1 的缆绳参数打包器。
    # 每根缆绳各自压成一个 legacy ParaC 长向量，供旧 batched 包装入口逐根拆包。
    def _pack_paraC(self, xq_fb, ref_xq, ref_uq, paraC, scxc_traj, scuc_traj, y_xc, y_uc, i_admm):
        ParaC_list = []
        for i in range(self.nq):
            # 这里的切片必须非常精准，否则旧 packaged 入口会把字段顺序读错。
            parai = np.concatenate((
                xq_fb[i*self.nxi : (i+1)*self.nxi],
                ref_xq[i].flatten(),
                ref_uq[i*self.nui : (i+1)*self.nui],
                scxc_traj[i].flatten(),
                y_xc[i].flatten(),
                scuc_traj[i].flatten(),
                y_uc[i].flatten(),
                paraC.flatten(),
                [float(i_admm)]
            ))
            ParaC_list.append(parai)
        return ParaC_list

    def jax_MPC_Load_DDP_Planning_SubP1(self, ParaL, need_derivs=False):
        """
        [JAX] 负载子问题 1 求解器
        ParaL: 来自 ADMM 主循环的负载参数包列表 (通常 B=1)
        当前默认只返回轨迹；只有显式 need_derivs=True 才额外提取导数包。
        """
        B, N, nx, nu = len(ParaL), int(self.N), 13, 6
        x0_list, u_guess_list, params_list = [], [], []

        # --- 1. 严格索引审计 (针对 COM_Dyn + UseThis 逻辑) ---
        for i in range(B):
            Parai = ParaL[i]
            idx = 0
            # (1) 初值 xl_fb [13]
            xl_fb = Parai[idx : idx+nx]; idx += nx
            # (2) 参考轨迹 Ref_xl [13*(N+1)], Ref_ul [6*N]
            Ref_xl_raw = Parai[idx : idx+nx*(N+1)]; idx += nx*(N+1)
            Ref_ul_raw = Parai[idx : idx+nu*N]; idx += nu*N
            # (3) 原版参数顺序是 scxl -> scxL -> scul -> scuL，不能串位。
            scxl_raw = Parai[idx : idx+nx*(N+1)]; idx += nx*(N+1)
            scxL_raw = Parai[idx : idx+nx*(N+1)]; idx += nx*(N+1)
            scul_raw = Parai[idx : idx+nu*N]; idx += nu*N
            scuL_raw = Parai[idx : idx+nu*N]; idx += nu*N
            # (5) 权重超参数 para_l [2*nx + nu + 4]
            weight_para = Parai[idx:-1]
            i_admm = Parai[-1]

            # 计算 UseThis 版动态惩罚 rho
            # weight_para[-4]: px, weight_para[-2]: gammax
            rho_lx = self.open_loop_penalty_jax(weight_para[-4], weight_para[-2], i_admm, self.max_iter_ADMM)
            rho_lu = self.open_loop_penalty_jax(weight_para[-3], weight_para[-1], i_admm, self.max_iter_ADMM)

            x0_list.append(jnp.array(xl_fb, dtype=jnp.float64))
            # 原版 load DDP 的初始控制轨迹来自 Ref_ul。
            u_guess_list.append(jnp.array(Ref_ul_raw, dtype=jnp.float64).reshape(N, nu))

            # 构造 params 字典 (必须与 jax_load_stage_cost 接口严格对应)
            ref_x_traj = jnp.array(Ref_xl_raw, dtype=jnp.float64).reshape(N + 1, nx)
            ref_u_traj = jnp.array(Ref_ul_raw, dtype=jnp.float64).reshape(N, nu)
            scx_traj = jnp.array(scxl_raw, dtype=jnp.float64).reshape(N + 1, nx)
            scu_traj = jnp.array(scul_raw, dtype=jnp.float64).reshape(N, nu)
            yx_traj = jnp.array(scxL_raw, dtype=jnp.float64).reshape(N + 1, nx)
            yu_traj = jnp.array(scuL_raw, dtype=jnp.float64).reshape(N, nu)

            params_list.append({
                'static': {
                    'ml': float(self.ml),
                    'Jl': jnp.array(self.Jl, dtype=jnp.float64),
                    'Jl_inv': jnp.array(self.Jl_inv, dtype=jnp.float64),
                    'dt': float(self.dt),
                    'rho_lx': rho_lx,
                    'rho_lu': rho_lu,
                    'Q_weight': jnp.array(weight_para[0:nx], dtype=jnp.float64),
                    'R_weight': jnp.array(weight_para[2*nx:2*nx+nu], dtype=jnp.float64),
                    'Q_terminal_weight': jnp.array(weight_para[nx:2*nx], dtype=jnp.float64),
                },
                'stage': {
                    'ref_x': ref_x_traj[:-1],
                    'ref_u': ref_u_traj,
                    'scx': scx_traj[:-1],
                    'scu': scu_traj,
                    'y_x': yx_traj[:-1],
                    'y_u': yu_traj,
                },
                'terminal': {
                    'ref_x': ref_x_traj[-1],
                    'scx': scx_traj[-1],
                    'y_x': yx_traj[-1],
                },
            })

        # --- 2. 打包并点火 (JAX 魔法) ---
        params_b = jax.tree_util.tree_map(lambda *args: jnp.stack(args).astype(jnp.float64), *params_list)
        x0_b = jnp.stack(x0_list).astype(jnp.float64)
        u_init_b = jnp.stack(u_guess_list).astype(jnp.float64)

        # 调用 JIT 编译好的 DDP 求解器
        cfg = ILQRConfig(max_iters=10, tol_g_norm=1e-2)
        results = self._load_solver_jit(x0_b, u_init_b, params_b, cfg=cfg)

        # --- 3. 结果适配并返回 (保持与原版 ADMM 接口兼容) ---
        xs_np = np.array(results.xs)
        us_np = np.array(results.us)

        opt_soll = {
            "xl_traj": [xs_np[i] for i in range(B)],
            "ul_traj": [us_np[i] for i in range(B)]
        }

        if not need_derivs:
            return opt_soll, None

        # --- 4. 提取导数 (为 ADMM 协同做准备) ---
        # 这一步通过 vmap 同时算全时域导数，比原版快得多
        derivs = self._load_derivs_jit(results.xs, results.us, params_b)
        Quu_inv_np, Qxu_np, K_fb_np, Fx_np, Fu_np = [np.array(d) for d in derivs]

        OPt_sol_l = []
        for i in range(B):
            OPt_sol_l.append({
                'xl_traj': xs_np[i],
                'ul_traj': us_np[i],
                'K_FB': K_fb_np[i],
                'Quu_inv': Quu_inv_np[i],
                'Qxu': Qxu_np[i],
                'Fx': Fx_np[i],
                'Fu': Fu_np[i]
            })

        return opt_soll, OPt_sol_l

    # ---------------------------------------------------------------------
    # 非当前稳定主线：纯 JAX SubP2 实验区。
    # 这些方法保留是为了研究/对照，但当前 persistent Julia 主线默认不直接走。
    # ---------------------------------------------------------------------

    # 这是"纯 JAX SubP2"总入口，不是当前稳定 persistent Julia 主线。
    # 现在正式 benchmark 里，这个方法通常会在外部 runner 中被 monkey-patch 掉，
    # 换成 "Python 打包 -> 常驻 Julia worker 求解 -> Python 解包" 的实现。
    #
    # 所以如果你只关心当前稳定主线：
    # - 这个函数主体可先跳过
    # - 但它依赖的 _prepare_subp2_batch / _unpack_subp2_results 仍然很重要
    def jax_ADMM_SubP2(self, Para2_dict):
        """
        [JAX 终极版] 子问题 2：利用 ipoptax 实现全时域并行硬约束优化
        """
        solver_mode = os.environ.get("JAX_SUBP2_SOLVER_MODE", "ipoptax").strip().lower()
        if solver_mode == "sqp":
            return self.jax_ADMM_SubP2_SQP(Para2_dict)
        if solver_mode == "barrier":
            return self.jax_ADMM_SubP2_BARRIER(Para2_dict)

        N, nq = self.N, int(self.nq)
        nxl, nul, nxi, nui = self.nxl, self.nul, self.nxi, self.nui
        dims = (nxl, nul, nxi, nui, nq, int(self.num_dis))
        verbose = bool(getattr(self, "verbose", False))
        eps_s = 1e-3

        def _env_int(name, default):
            try:
                return int(os.environ.get(name, default))
            except Exception:
                return int(default)

        def _env_float(name, default):
            try:
                return float(os.environ.get(name, default))
            except Exception:
                return float(default)

        damp_beta = _env_float("JAX_SUBP2_DAMP_BETA", 0.15)
        accept_ratio = _env_float("JAX_SUBP2_ACCEPT_FEAS_RATIO", 0.7)
        accept_abs = _env_float("JAX_SUBP2_ACCEPT_FEAS_ABS", 5e-3)
        retry_trigger_ratio = _env_float("JAX_SUBP2_RETRY_TRIGGER_RATIO", 0.98)
        filter_gamma_theta = _env_float("JAX_SUBP2_FILTER_GAMMA_THETA", 1e-2)
        filter_gamma_phi = _env_float("JAX_SUBP2_FILTER_GAMMA_PHI", 1e-5)
        restoration_theta_ratio = _env_float("JAX_SUBP2_RESTO_THETA_RATIO", 0.9)
        retry_on_nonconv = os.environ.get("JAX_SUBP2_RETRY_ON_NONCONV", "0") == "1"
        enable_retry2 = os.environ.get("JAX_SUBP2_ENABLE_RETRY2", "0") == "1"
        stage_log = os.environ.get("JAX_SUBP2_STAGE_LOG", "0") == "1"
        summary_log = os.environ.get("JAX_SUBP2_SUMMARY_LOG", "1") == "1"
        s0_cap = _env_float("JAX_SUBP2_S0_CAP", 0.5)
        z0_init = _env_float("JAX_SUBP2_Z0_INIT", 1e-3)

        cfg_main = {
            "name": "main",
            "eq_mult": 1.0,
            "ineq_mult": 1.0,
            "max_iterations": _env_int("JAX_SUBP2_MAIN_MAX_ITERS", 80),
            "max_kkt_violation": _env_float("JAX_SUBP2_MAIN_KKT", 1e-3),
            "lin_sys_formulation": LinearSystemFormulation.STABLE_DIRECT_4x4,
            "tau_min": _env_float("JAX_SUBP2_MAIN_TAU_MIN", 0.99),
            "mu_min": _env_float("JAX_SUBP2_MAIN_MU_MIN", 1e-8),
            "min_delta": _env_float("JAX_SUBP2_MAIN_MIN_DELTA", 1e-4),
            "gamma_y": _env_float("JAX_SUBP2_MAIN_GAMMA_Y", 1e-4),
            "gamma_z": _env_float("JAX_SUBP2_MAIN_GAMMA_Z", 1e-2),
            "armijo_factor": 1e-4,
            "line_search_factor": _env_float("JAX_SUBP2_MAIN_LINE_SEARCH_FACTOR", 0.2),
            "line_search_min_step_size": _env_float("JAX_SUBP2_MAIN_LINE_SEARCH_MIN_STEP", 1e-10),
        }
        cfg_retry1 = {
            "name": "retry1",
            "eq_mult": 0.5,
            "ineq_mult": 0.5,
            "max_iterations": _env_int("JAX_SUBP2_RETRY1_MAX_ITERS", 120),
            "max_kkt_violation": _env_float("JAX_SUBP2_RETRY1_KKT", 5e-3),
            "lin_sys_formulation": LinearSystemFormulation.STABLE_DIRECT_4x4,
            "tau_min": 0.95,
            "mu_min": 1e-6,
            "min_delta": 1e-3,
            "gamma_y": 1e-2,
            "gamma_z": 1e-2,
            "armijo_factor": 1e-3,
            "line_search_factor": 0.1,
            "line_search_min_step_size": 1e-12,
        }
        cfg_retry2 = {
            "name": "retry2",
            "eq_mult": 0.2,
            "ineq_mult": 0.2,
            "max_iterations": _env_int("JAX_SUBP2_RETRY2_MAX_ITERS", 320),
            "max_kkt_violation": _env_float("JAX_SUBP2_RETRY2_KKT", 1e-2),
            "lin_sys_formulation": LinearSystemFormulation.SYMMETRIC_DIRECT_4x4,
            "tau_min": 0.98,
            "mu_min": 1e-8,
            "min_delta": 1e-5,
            "gamma_y": 1e-4,
            "gamma_z": 1e-4,
            "armijo_factor": 1e-3,
            "line_search_factor": 0.2,
            "line_search_min_step_size": 1e-10,
        }

        # --- 1. 数据对齐与 Batch 构造 (Time-step Parallelism) ---
        # 这一步将 Para2_dict 中的 1D 数组还原为 (N+1, Dim)
        w_init_batch, params_batch = self._prepare_subp2_batch(Para2_dict)

        if summary_log and verbose:
            print(
                "[JAX-SubP2][config] "
                f"main(tau={cfg_main['tau_min']:.2g}, mu={cfg_main['mu_min']:.1e}, "
                f"delta={cfg_main['min_delta']:.1e}, gy={cfg_main['gamma_y']:.1e}, "
                f"gz={cfg_main['gamma_z']:.1e}, ls={cfg_main['line_search_factor']:.2g}, "
                f"min_ls={cfg_main['line_search_min_step_size']:.1e}) | "
                f"retry1(tau={cfg_retry1['tau_min']:.2g}, mu={cfg_retry1['mu_min']:.1e}, "
                f"delta={cfg_retry1['min_delta']:.1e}, gy={cfg_retry1['gamma_y']:.1e}, "
                f"gz={cfg_retry1['gamma_z']:.1e}) | "
                f"retry2={'on' if enable_retry2 else 'off'} | "
                f"s0_cap={s0_cap:.2g} | z0_init={z0_init:.1e} | "
                f"retry_on_nonconv={int(retry_on_nonconv)}",
                flush=True,
            )

        # --- 2. 定义单步求解闭包 (适配 ipoptax) ---
        def run_ipoptax_step_cfg(x_init, p_t, cfg):
            f = lambda x: MPC_Planner.ipoptax_objective(x, p_t, dims)
            c_raw = lambda x: MPC_Planner.ipoptax_equality(x, p_t, dims)
            g_raw = lambda x: MPC_Planner.ipoptax_inequality(x, p_t, dims)
            c_scale = p_t['eq_scale'] * cfg["eq_mult"]
            g_scale = p_t['ineq_scale'] * cfg["ineq_mult"]
            c = lambda x: c_scale * c_raw(x)
            g = lambda x: g_scale * g_raw(x)

            c0 = c(x_init)
            g0 = g(x_init)
            ws_s0 = jnp.minimum(jnp.maximum(-g0 + eps_s, eps_s), s0_cap)
            ws_z0 = jnp.full_like(ws_s0, z0_init)
            ws_y0 = jnp.zeros_like(c0)

            res = ipoptax_solve(
                f=f, c=c, g=g,
                ws_x=x_init,
                ws_s=ws_s0,
                ws_y=ws_y0,
                ws_z=ws_z0,
                max_iterations=cfg["max_iterations"],
                max_kkt_violation=cfg["max_kkt_violation"],
                lin_sys_formulation=cfg["lin_sys_formulation"],
                tau_min=cfg["tau_min"],
                mu_min=cfg["mu_min"],
                min_delta=cfg["min_delta"],
                gamma_y=cfg["gamma_y"],
                gamma_z=cfg["gamma_z"],
                armijo_factor=cfg["armijo_factor"],
                line_search_factor=cfg["line_search_factor"],
                line_search_min_step_size=cfg["line_search_min_step_size"],
                psd_use_lapack=solver_psd_use_lapack,
                psd_iterate=solver_psd_iterate,
                soft_stop_enabled=solver_soft_stop,
                soft_mu_tol=soft_mu_tol,
                soft_comp_tol=soft_comp_tol,
                soft_dual_tol=soft_dual_tol,
                soft_eq_tol=soft_eq_tol,
                soft_ineq_tol=soft_trace_ineq_tol,
                plateau_stop_enabled=solver_plateau_stop,
                plateau_min_iterations=plateau_min_iterations,
                plateau_patience=plateau_patience,
                plateau_phi_tol=plateau_phi_tol,
                plateau_ineq_tol=plateau_ineq_tol,
                print_logs=False,
                trace_length=cfg["max_iterations"],
            )
            return res

        def run_ipoptax_step(x_init, p_t):
            return run_ipoptax_step_cfg(x_init, p_t, cfg_main)

        def run_ipoptax_step_retry1(x_init, p_t):
            return run_ipoptax_step_cfg(x_init, p_t, cfg_retry1)

        def run_ipoptax_step_retry2(x_init, p_t):
            return run_ipoptax_step_cfg(x_init, p_t, cfg_retry2)

        def primal_feas_metrics(x_val, p_t):
            c_raw = MPC_Planner.ipoptax_equality(x_val, p_t, dims)
            g_raw = MPC_Planner.ipoptax_inequality(x_val, p_t, dims)
            eq_inf = float(np.max(np.abs(np.array(c_raw))))
            ineq_vio = float(max(0.0, np.max(np.array(g_raw))))
            return eq_inf, ineq_vio, max(eq_inf, ineq_vio)

        def primal_feas_metrics_jax(x_val, p_t):
            c_raw = MPC_Planner.ipoptax_equality(x_val, p_t, dims)
            g_raw = MPC_Planner.ipoptax_inequality(x_val, p_t, dims)
            eq_inf = jnp.max(jnp.abs(c_raw))
            ineq_vio = jnp.maximum(0.0, jnp.max(g_raw))
            return eq_inf, ineq_vio, jnp.maximum(eq_inf, ineq_vio)

        def filter_metrics(x_val, p_t):
            x_np = np.array(x_val)
            eq_inf, ineq_vio, feas = primal_feas_metrics(x_np, p_t)
            theta = float(np.linalg.norm(np.array(MPC_Planner.ipoptax_equality(x_np, p_t, dims))))
            theta += float(np.linalg.norm(np.maximum(0.0, np.array(MPC_Planner.ipoptax_inequality(x_np, p_t, dims)))))
            barr = float(np.array(MPC_Planner.ipoptax_objective(x_np, p_t, dims)))
            return {
                "eq_inf": eq_inf,
                "ineq_vio": ineq_vio,
                "feas": feas,
                "theta": theta,
                "barr": barr,
            }

        def filter_metrics_jax(x_val, p_t):
            c_raw = MPC_Planner.ipoptax_equality(x_val, p_t, dims)
            g_raw = MPC_Planner.ipoptax_inequality(x_val, p_t, dims)
            eq_inf = jnp.max(jnp.abs(c_raw))
            ineq_vio = jnp.maximum(0.0, jnp.max(g_raw))
            feas = jnp.maximum(eq_inf, ineq_vio)
            theta = jnp.linalg.norm(c_raw) + jnp.linalg.norm(jnp.maximum(0.0, g_raw))
            barr = MPC_Planner.ipoptax_objective(x_val, p_t, dims)
            return {
                "eq_inf": eq_inf,
                "ineq_vio": ineq_vio,
                "feas": feas,
                "theta": theta,
                "barr": barr,
            }

        def filter_accept(init_metrics, cand_metrics):
            theta_ok = cand_metrics["theta"] <= (1.0 - filter_gamma_theta) * init_metrics["theta"]
            barr_ok = cand_metrics["barr"] <= init_metrics["barr"] - filter_gamma_phi * init_metrics["theta"]
            feas_ok = cand_metrics["feas"] <= max(accept_abs, accept_ratio * init_metrics["feas"])
            return theta_ok or barr_ok or feas_ok

        def filter_accept_jax(init_metrics, cand_metrics):
            theta_ok = cand_metrics["theta"] <= (1.0 - filter_gamma_theta) * init_metrics["theta"]
            barr_ok = cand_metrics["barr"] <= init_metrics["barr"] - filter_gamma_phi * init_metrics["theta"]
            feas_ok = cand_metrics["feas"] <= jnp.maximum(accept_abs, accept_ratio * init_metrics["feas"])
            return jnp.logical_or(theta_ok, jnp.logical_or(barr_ok, feas_ok))

        soft_mu_tol = _env_float("JAX_SUBP2_SOFT_MU_TOL", max(float(cfg_main["mu_min"]) * 5.0, 1e-8))
        soft_comp_tol = _env_float("JAX_SUBP2_SOFT_COMP_TOL", 1e-8)
        soft_dual_tol = _env_float("JAX_SUBP2_SOFT_DUAL_TOL", 5e-5)
        soft_eq_tol = _env_float("JAX_SUBP2_SOFT_EQ_TOL", 1e-4)
        soft_raw_ineq_tol = _env_float("JAX_SUBP2_SOFT_RAW_INEQ_TOL", 2e-1)
        soft_trace_ineq_tol = _env_float("JAX_SUBP2_SOFT_TRACE_INEQ_TOL", 5e-3)
        solver_soft_stop = os.environ.get("JAX_SUBP2_SOLVER_SOFT_STOP", "1") == "1"
        solver_plateau_stop = os.environ.get("JAX_SUBP2_SOLVER_PLATEAU_STOP", "1") == "1"
        plateau_min_iterations = _env_int("JAX_SUBP2_PLATEAU_MIN_ITERS", 6)
        plateau_patience = _env_int("JAX_SUBP2_PLATEAU_PATIENCE", 4)
        plateau_phi_tol = _env_float("JAX_SUBP2_PLATEAU_PHI_TOL", 5e-7)
        plateau_ineq_tol = _env_float("JAX_SUBP2_PLATEAU_INEQ_TOL", 5e-7)
        solver_psd_use_lapack = os.environ.get("JAX_SUBP2_SOLVER_PSD_USE_LAPACK", "1") == "1"
        solver_psd_iterate = os.environ.get("JAX_SUBP2_SOLVER_PSD_ITERATE", "1") == "1"
        mode_name_to_code = {
            "fallback": 0,
            "main": 1,
            "main-soft": 2,
            "best-feas": 3,
            "retry1": 4,
            "retry1-soft": 5,
            "retry2": 6,
            "retry2-soft": 7,
            "main-bestfeas": 8,
            "retry1-bestfeas": 9,
            "retry2-bestfeas": 10,
        }
        mode_code_to_name = {v: k for k, v in mode_name_to_code.items()}

        def solver_tail_metrics(res):
            iters = int(np.array(res["iteration"]))
            if iters <= 0:
                return None
            trace_len = len(np.array(res["trace_mu"]))
            idx = min(iters, trace_len) - 1
            return {
                "mu": float(np.array(res["trace_mu"])[idx]),
                "phi": float(np.array(res["trace_phi"])[idx]),
                "ineq": float(np.array(res["trace_ineq"])[idx]),
                "comp": float(np.array(res["trace_comp"])[idx]),
                "dual": float(np.array(res["trace_dual"])[idx]),
            }

        def solver_tail_metrics_jax(batch_res):
            iters = jnp.asarray(batch_res["iteration"], dtype=jnp.int32)
            trace_mu = jnp.asarray(batch_res["trace_mu"])
            trace_phi = jnp.asarray(batch_res["trace_phi"])
            trace_ineq = jnp.asarray(batch_res["trace_ineq"])
            trace_comp = jnp.asarray(batch_res["trace_comp"])
            trace_dual = jnp.asarray(batch_res["trace_dual"])
            trace_len = trace_mu.shape[1]
            idx = jnp.clip(iters - 1, 0, trace_len - 1)
            gather_idx = idx[:, None]
            mu = jnp.take_along_axis(trace_mu, gather_idx, axis=1)[:, 0]
            phi = jnp.take_along_axis(trace_phi, gather_idx, axis=1)[:, 0]
            ineq = jnp.take_along_axis(trace_ineq, gather_idx, axis=1)[:, 0]
            comp = jnp.take_along_axis(trace_comp, gather_idx, axis=1)[:, 0]
            dual = jnp.take_along_axis(trace_dual, gather_idx, axis=1)[:, 0]
            valid = iters > 0
            nan_fill = jnp.full_like(mu, jnp.nan)
            return {
                "mu": jnp.where(valid, mu, nan_fill),
                "phi": jnp.where(valid, phi, nan_fill),
                "ineq": jnp.where(valid, ineq, nan_fill),
                "comp": jnp.where(valid, comp, nan_fill),
                "dual": jnp.where(valid, dual, nan_fill),
            }

        def soft_success(res, cand_metrics):
            tail = solver_tail_metrics(res)
            if tail is None:
                return False
            return (
                tail["mu"] <= soft_mu_tol
                and tail["comp"] <= soft_comp_tol
                and tail["dual"] <= soft_dual_tol
                and tail["ineq"] <= soft_trace_ineq_tol
                and cand_metrics["eq_inf"] <= soft_eq_tol
                and cand_metrics["ineq_vio"] <= soft_raw_ineq_tol
            )

        # --- 3. 执行求解 ---
        # 默认走 vmap 并行；verbose 模式下改为逐时刻求解并打印进度，便于定位“卡住”时刻
        if verbose:
            w_rows = []
            diag_modes = []
            diag_converged = []
            diag_iters = []
            diag_eq_inf = []
            diag_ineq_vio = []
            diag_mode_codes = []
            diag_tail_mu = []
            diag_tail_ineq = []
            diag_tail_comp = []
            diag_tail_dual = []
            for k in range(N + 1):
                t_k = TM.time()
                p_t = {kk: vv[k] for kk, vv in params_batch.items()}
                x_init_k = w_init_batch[k]
                init_metrics = filter_metrics(x_init_k, p_t)
                init_feas = init_metrics["feas"]

                res_main = run_ipoptax_step(x_init_k, p_t)
                x_main = res_main["x"]
                xm_np = np.array(x_main)
                conv_main = bool(np.array(res_main["converged"]))
                iter_main = int(np.array(res_main["iteration"]))
                finite_main = bool(np.all(np.isfinite(xm_np)))

                # 记录候选解并按可行性挑选
                x_k = x_init_k
                xk_np = np.array(x_k)
                best_feas = np.inf
                best_mode = "fallback"
                best_conv = False
                best_iter = 0
                eq_inf = 0.0
                ineq_vio = 0.0
                best_metrics = None

                if finite_main:
                    main_metrics = filter_metrics(x_main, p_t)
                    main_tail = solver_tail_metrics(res_main)
                    eq_m, ineq_m, feas_m = main_metrics["eq_inf"], main_metrics["ineq_vio"], main_metrics["feas"]
                    if stage_log:
                        print(
                            f"[JAX-SubP2][stage] k={k} main conv={int(conv_main)} it={iter_main} "
                            f"eq={eq_m:.2e} ineq={ineq_m:.2e} feas={feas_m:.2e}",
                            flush=True,
                        )
                    soft_main = soft_success(res_main, main_metrics)
                    if conv_main or soft_main:
                        x_k = x_main
                        xk_np = xm_np
                        best_mode = "main" if conv_main else "main-soft"
                        best_conv = True
                        best_iter = iter_main
                        eq_inf, ineq_vio, best_feas = eq_m, ineq_m, feas_m
                        best_metrics = main_metrics
                    else:
                        if feas_m < best_feas:
                            x_k = x_main
                            xk_np = xm_np
                            best_mode = "main-bestfeas"
                            best_iter = iter_main
                            eq_inf, ineq_vio, best_feas = eq_m, ineq_m, feas_m
                            best_metrics = main_metrics

                need_retry = (not finite_main) or (retry_on_nonconv and (best_feas >= retry_trigger_ratio * init_feas))
                if (not best_conv) and need_retry:
                    res_retry1 = run_ipoptax_step_retry1(x_init_k, p_t)
                    x_r1 = res_retry1["x"]
                    xr1_np = np.array(x_r1)
                    conv_r1 = bool(np.array(res_retry1["converged"]))
                    iter_r1 = int(np.array(res_retry1["iteration"]))
                    finite_r1 = bool(np.all(np.isfinite(xr1_np)))
                    if finite_r1:
                        r1_metrics = filter_metrics(x_r1, p_t)
                        eq_r1, ineq_r1, feas_r1 = r1_metrics["eq_inf"], r1_metrics["ineq_vio"], r1_metrics["feas"]
                        if stage_log:
                            print(
                                f"[JAX-SubP2][stage] k={k} retry1 conv={int(conv_r1)} it={iter_r1} "
                                f"eq={eq_r1:.2e} ineq={ineq_r1:.2e} feas={feas_r1:.2e}",
                                flush=True,
                            )
                        soft_r1 = soft_success(res_retry1, r1_metrics)
                        if conv_r1 or soft_r1:
                            x_k = x_r1
                            xk_np = xr1_np
                            best_mode = "retry1" if conv_r1 else "retry1-soft"
                            best_conv = True
                            best_iter = iter_r1
                            eq_inf, ineq_vio, best_feas = eq_r1, ineq_r1, feas_r1
                            best_metrics = r1_metrics
                        elif feas_r1 < best_feas:
                            x_k = x_r1
                            xk_np = xr1_np
                            best_mode = "retry1-bestfeas"
                            best_iter = iter_r1
                            eq_inf, ineq_vio, best_feas = eq_r1, ineq_r1, feas_r1
                            best_metrics = r1_metrics

                if (not best_conv) and need_retry and enable_retry2:
                    res_retry2 = run_ipoptax_step_retry2(x_init_k, p_t)
                    x_r2 = res_retry2["x"]
                    xr2_np = np.array(x_r2)
                    conv_r2 = bool(np.array(res_retry2["converged"]))
                    iter_r2 = int(np.array(res_retry2["iteration"]))
                    finite_r2 = bool(np.all(np.isfinite(xr2_np)))
                    if finite_r2:
                        r2_metrics = filter_metrics(x_r2, p_t)
                        eq_r2, ineq_r2, feas_r2 = r2_metrics["eq_inf"], r2_metrics["ineq_vio"], r2_metrics["feas"]
                        if stage_log:
                            print(
                                f"[JAX-SubP2][stage] k={k} retry2 conv={int(conv_r2)} it={iter_r2} "
                                f"eq={eq_r2:.2e} ineq={ineq_r2:.2e} feas={feas_r2:.2e}",
                                flush=True,
                            )
                        soft_r2 = soft_success(res_retry2, r2_metrics)
                        if conv_r2 or soft_r2:
                            x_k = x_r2
                            xk_np = xr2_np
                            best_mode = "retry2" if conv_r2 else "retry2-soft"
                            best_conv = True
                            best_iter = iter_r2
                            eq_inf, ineq_vio, best_feas = eq_r2, ineq_r2, feas_r2
                            best_metrics = r2_metrics
                        elif feas_r2 < best_feas:
                            x_k = x_r2
                            xk_np = xr2_np
                            best_mode = "retry2-bestfeas"
                            best_iter = iter_r2
                            eq_inf, ineq_vio, best_feas = eq_r2, ineq_r2, feas_r2
                            best_metrics = r2_metrics

                if (not best_conv) and (best_mode.endswith("bestfeas")):
                    filter_ok = (best_metrics is not None) and filter_accept(init_metrics, best_metrics)
                    resto_ok = (best_metrics is not None) and (best_metrics["theta"] <= restoration_theta_ratio * init_metrics["theta"])
                    if not (filter_ok or resto_ok):
                        x_k = x_init_k
                        xk_np = np.array(x_k)
                        best_mode = "fallback"
                        best_iter = 0
                if best_mode == "fallback":
                    eq_inf, ineq_vio, best_feas = primal_feas_metrics(x_k, p_t)

                n_fin = int(np.isfinite(xk_np).sum())
                print(
                    f"[JAX-SubP2] step {k+1}/{N+1} done in {(TM.time()-t_k)*1000:.2f} ms | "
                    f"finite={n_fin}/{xk_np.size} | mode={best_mode} | conv={int(best_conv)} | "
                    f"iters={best_iter} | eq_inf={eq_inf:.2e} | ineq_vio={ineq_vio:.2e}",
                    flush=True,
                )
                w_rows.append(x_k)
                diag_modes.append(best_mode)
                diag_mode_codes.append(mode_name_to_code.get(best_mode, 0))
                diag_converged.append(bool(best_conv))
                diag_iters.append(int(best_iter))
                diag_eq_inf.append(float(eq_inf))
                diag_ineq_vio.append(float(ineq_vio))
                diag_tail_mu.append(float(main_tail["mu"]) if finite_main and main_tail is not None else np.nan)
                diag_tail_ineq.append(float(main_tail["ineq"]) if finite_main and main_tail is not None else np.nan)
                diag_tail_comp.append(float(main_tail["comp"]) if finite_main and main_tail is not None else np.nan)
                diag_tail_dual.append(float(main_tail["dual"]) if finite_main and main_tail is not None else np.nan)
            w_opt_batch = jnp.stack(w_rows, axis=0)
        else:
            # 核心魔法：时间维度的向量化 (vmap)
            batch_solver = jax.vmap(run_ipoptax_step, in_axes=(0, 0))
            # w_opt_batch 形状为 (N+1, Total_Dim)
            batch_res = batch_solver(w_init_batch, params_batch)
            use_fast_path = os.environ.get("JAX_SUBP2_FAST_PATH", "1") == "1"
            if use_fast_path:
                w_main_batch = batch_res["x"]
                main_conv_mask = jnp.asarray(batch_res["converged"], dtype=bool)
                main_iters = jnp.asarray(batch_res["iteration"], dtype=jnp.int32)

                init_metrics = jax.vmap(filter_metrics_jax, in_axes=(0, 0))(w_init_batch, params_batch)
                main_metrics = jax.vmap(filter_metrics_jax, in_axes=(0, 0))(w_main_batch, params_batch)
                tail_metrics = solver_tail_metrics_jax(batch_res)

                retry1_x = w_init_batch
                retry1_finite = jnp.zeros((N + 1,), dtype=bool)
                retry1_conv = jnp.zeros((N + 1,), dtype=bool)
                retry1_iters = jnp.zeros((N + 1,), dtype=jnp.int32)
                retry1_metrics = {
                    "eq_inf": init_metrics["eq_inf"],
                    "ineq_vio": init_metrics["ineq_vio"],
                    "feas": jnp.full_like(init_metrics["feas"], jnp.inf),
                    "theta": jnp.full_like(init_metrics["theta"], jnp.inf),
                    "barr": jnp.full_like(init_metrics["barr"], jnp.inf),
                }
                retry1_tail = {
                    "mu": jnp.full_like(init_metrics["feas"], jnp.nan),
                    "ineq": jnp.full_like(init_metrics["feas"], jnp.nan),
                    "comp": jnp.full_like(init_metrics["feas"], jnp.nan),
                    "dual": jnp.full_like(init_metrics["feas"], jnp.nan),
                }
                if retry_on_nonconv:
                    batch_retry1 = jax.vmap(run_ipoptax_step_retry1, in_axes=(0, 0))
                    retry1_res = batch_retry1(w_init_batch, params_batch)
                    retry1_x = retry1_res["x"]
                    retry1_finite = jnp.all(jnp.isfinite(retry1_x), axis=1)
                    retry1_conv = jnp.asarray(retry1_res["converged"], dtype=bool)
                    retry1_iters = jnp.asarray(retry1_res["iteration"], dtype=jnp.int32)
                    retry1_metrics = jax.vmap(filter_metrics_jax, in_axes=(0, 0))(retry1_x, params_batch)
                    retry1_tail = solver_tail_metrics_jax(retry1_res)

                retry2_x = w_init_batch
                retry2_finite = jnp.zeros((N + 1,), dtype=bool)
                retry2_conv = jnp.zeros((N + 1,), dtype=bool)
                retry2_iters = jnp.zeros((N + 1,), dtype=jnp.int32)
                retry2_metrics = {
                    "eq_inf": init_metrics["eq_inf"],
                    "ineq_vio": init_metrics["ineq_vio"],
                    "feas": jnp.full_like(init_metrics["feas"], jnp.inf),
                    "theta": jnp.full_like(init_metrics["theta"], jnp.inf),
                    "barr": jnp.full_like(init_metrics["barr"], jnp.inf),
                }
                retry2_tail = {
                    "mu": jnp.full_like(init_metrics["feas"], jnp.nan),
                    "ineq": jnp.full_like(init_metrics["feas"], jnp.nan),
                    "comp": jnp.full_like(init_metrics["feas"], jnp.nan),
                    "dual": jnp.full_like(init_metrics["feas"], jnp.nan),
                }
                if retry_on_nonconv and enable_retry2:
                    batch_retry2 = jax.vmap(run_ipoptax_step_retry2, in_axes=(0, 0))
                    retry2_res = batch_retry2(w_init_batch, params_batch)
                    retry2_x = retry2_res["x"]
                    retry2_finite = jnp.all(jnp.isfinite(retry2_x), axis=1)
                    retry2_conv = jnp.asarray(retry2_res["converged"], dtype=bool)
                    retry2_iters = jnp.asarray(retry2_res["iteration"], dtype=jnp.int32)
                    retry2_metrics = jax.vmap(filter_metrics_jax, in_axes=(0, 0))(retry2_x, params_batch)
                    retry2_tail = solver_tail_metrics_jax(retry2_res)

                fast_res = _subp2_fast_select_kernel(
                    w_init_batch,
                    w_main_batch,
                    main_conv_mask,
                    main_iters,
                    init_metrics,
                    main_metrics,
                    tail_metrics,
                    retry1_x,
                    retry1_finite,
                    retry1_conv,
                    retry1_iters,
                    retry1_metrics,
                    retry1_tail,
                    retry2_x,
                    retry2_finite,
                    retry2_conv,
                    retry2_iters,
                    retry2_metrics,
                    retry2_tail,
                    retry_on_nonconv,
                    enable_retry2,
                    retry_trigger_ratio,
                    restoration_theta_ratio,
                    filter_gamma_theta,
                    filter_gamma_phi,
                    accept_abs,
                    accept_ratio,
                    soft_mu_tol,
                    soft_comp_tol,
                    soft_dual_tol,
                    soft_eq_tol,
                    soft_raw_ineq_tol,
                    soft_trace_ineq_tol,
                    mode_name_to_code["main"],
                    mode_name_to_code["main-soft"],
                    mode_name_to_code["best-feas"],
                    mode_name_to_code["fallback"],
                    mode_name_to_code["retry1"],
                    mode_name_to_code["retry1-soft"],
                    mode_name_to_code["retry1-bestfeas"],
                    mode_name_to_code["retry2"],
                    mode_name_to_code["retry2-soft"],
                    mode_name_to_code["retry2-bestfeas"],
                )

                w_opt_batch = fast_res["w_opt_batch"]
                diag_converged = np.array(fast_res["accepted_conv_mask"], dtype=bool)
                diag_iters = np.array(fast_res["diag_iters"], dtype=int)
                diag_eq_inf = np.array(fast_res["diag_eq_inf"], dtype=float)
                diag_ineq_vio = np.array(fast_res["diag_ineq_vio"], dtype=float)
                diag_tail_mu = np.array(fast_res["diag_tail_mu"], dtype=float)
                diag_tail_ineq = np.array(fast_res["diag_tail_ineq"], dtype=float)
                diag_tail_comp = np.array(fast_res["diag_tail_comp"], dtype=float)
                diag_tail_dual = np.array(fast_res["diag_tail_dual"], dtype=float)
                diag_mode_codes = np.array(fast_res["diag_mode_codes"], dtype=int)
                diag_modes = np.array(
                    [mode_code_to_name[int(code)] for code in diag_mode_codes.tolist()],
                    dtype=object,
                )
                if summary_log:
                    bad_rows = int(np.sum(~diag_converged))
                    print(f"[JAX-SubP2][batch-fast] rows needing non-main acceptance={bad_rows}/{N+1}", flush=True)
            else:
                w_opt_batch = batch_res["x"]
                diag_modes = np.array(["main"] * (N + 1), dtype=object)
                diag_mode_codes = np.full((N + 1,), mode_name_to_code["main"], dtype=int)
                diag_converged = np.array(np.array(batch_res["converged"]).astype(bool))
                diag_iters = np.array(np.array(batch_res["iteration"]).astype(int))
                diag_tail_mu = np.full((N + 1,), np.nan, dtype=float)
                diag_tail_ineq = np.full((N + 1,), np.nan, dtype=float)
                diag_tail_comp = np.full((N + 1,), np.nan, dtype=float)
                diag_tail_dual = np.full((N + 1,), np.nan, dtype=float)

                # 若并行结果存在非有限值或未收敛，逐时刻重试坏样本
                w_opt_np = np.array(w_opt_batch)
                w_init_np = np.array(w_init_batch)
                conv_np = np.array(batch_res["converged"]).astype(bool)
                bad_idx = np.where((~np.all(np.isfinite(w_opt_np), axis=1)) | (~conv_np))[0]
                if summary_log:
                    print(
                        f"[JAX-SubP2][batch] initial bad rows={bad_idx.size}/{N+1}",
                        flush=True,
                    )
                for k in bad_idx:
                    p_t = {kk: vv[k] for kk, vv in params_batch.items()}
                    x_init_k = w_init_batch[k]
                    init_metrics = filter_metrics(x_init_k, p_t)
                    init_feas = init_metrics["feas"]
                    x_best = w_opt_np[k]
                    best_feas = np.inf
                    best_metrics = None
                    finite_main = bool(np.all(np.isfinite(x_best)))
                    if finite_main:
                        best_metrics = filter_metrics(jnp.array(x_best), p_t)
                        best_feas = best_metrics["feas"]
                        res_main_k = {
                            "iteration": np.array(batch_res["iteration"])[k],
                            "trace_mu": np.array(batch_res["trace_mu"])[k],
                            "trace_phi": np.array(batch_res["trace_phi"])[k],
                            "trace_ineq": np.array(batch_res["trace_ineq"])[k],
                            "trace_comp": np.array(batch_res["trace_comp"])[k],
                            "trace_dual": np.array(batch_res["trace_dual"])[k],
                        }
                        main_tail = solver_tail_metrics(res_main_k)
                        if main_tail is not None:
                            diag_tail_mu[k] = main_tail["mu"]
                            diag_tail_ineq[k] = main_tail["ineq"]
                            diag_tail_comp[k] = main_tail["comp"]
                            diag_tail_dual[k] = main_tail["dual"]
                        if soft_success(res_main_k, best_metrics):
                            w_opt_np[k] = x_best
                            diag_modes[k] = "main-soft"
                            diag_mode_codes[k] = mode_name_to_code["main-soft"]
                            diag_converged[k] = True
                            diag_iters[k] = int(np.array(batch_res["iteration"])[k])
                            if summary_log:
                                print(
                                    f"[JAX-SubP2][batch] k={k} promoted main-soft | "
                                    f"init_feas={init_feas:.2e} | best_feas={best_feas:.2e}",
                                    flush=True,
                                )
                            continue
                        elif summary_log:
                            tail = solver_tail_metrics(res_main_k)
                            if tail is not None:
                                print(
                                    f"[JAX-SubP2][batch] k={k} main-tail | "
                                    f"mu={tail['mu']:.2e} phi={tail['phi']:.2e} "
                                    f"ineq_tr={tail['ineq']:.2e} comp={tail['comp']:.2e} dual={tail['dual']:.2e} | "
                                    f"raw_eq={best_metrics['eq_inf']:.2e} raw_ineq={best_metrics['ineq_vio']:.2e}",
                                    flush=True,
                                )

                    need_retry = (not finite_main) or (retry_on_nonconv and (best_feas >= retry_trigger_ratio * init_feas))
                    if need_retry:
                        res_retry1 = run_ipoptax_step_retry1(x_init_k, p_t)
                        x_r1 = np.array(res_retry1["x"])
                        finite_r1 = bool(np.all(np.isfinite(x_r1)))
                        conv_r1 = bool(np.array(res_retry1["converged"]))
                        if finite_r1:
                            r1_metrics = filter_metrics(jnp.array(x_r1), p_t)
                            feas_r1 = r1_metrics["feas"]
                            soft_r1 = soft_success(res_retry1, r1_metrics)
                            if conv_r1 or soft_r1 or (feas_r1 < best_feas):
                                x_best, best_metrics, best_feas = x_r1, r1_metrics, feas_r1
                            if conv_r1 or soft_r1:
                                w_opt_np[k] = x_best
                                diag_modes[k] = "retry1" if conv_r1 else "retry1-soft"
                                diag_mode_codes[k] = mode_name_to_code[diag_modes[k]]
                                diag_converged[k] = True
                                diag_iters[k] = int(np.array(res_retry1["iteration"]))
                                if summary_log:
                                    print(
                                        f"[JAX-SubP2][batch] k={k} recovered by {diag_modes[k]} | "
                                        f"init_feas={init_feas:.2e} | best_feas={best_feas:.2e}",
                                        flush=True,
                                    )
                                continue

                    if need_retry and enable_retry2:
                        res_retry2 = run_ipoptax_step_retry2(x_init_k, p_t)
                        x_r2 = np.array(res_retry2["x"])
                        finite_r2 = bool(np.all(np.isfinite(x_r2)))
                        conv_r2 = bool(np.array(res_retry2["converged"]))
                        if finite_r2:
                            r2_metrics = filter_metrics(jnp.array(x_r2), p_t)
                            feas_r2 = r2_metrics["feas"]
                            soft_r2 = soft_success(res_retry2, r2_metrics)
                            if conv_r2 or soft_r2 or (feas_r2 < best_feas):
                                x_best, best_metrics, best_feas = x_r2, r2_metrics, feas_r2
                            if conv_r2 or soft_r2:
                                w_opt_np[k] = x_best
                                diag_modes[k] = "retry2" if conv_r2 else "retry2-soft"
                                diag_mode_codes[k] = mode_name_to_code[diag_modes[k]]
                                diag_converged[k] = True
                                diag_iters[k] = int(np.array(res_retry2["iteration"]))
                                if summary_log:
                                    print(
                                        f"[JAX-SubP2][batch] k={k} recovered by {diag_modes[k]} | "
                                        f"init_feas={init_feas:.2e} | best_feas={best_feas:.2e}",
                                        flush=True,
                                    )
                                continue

                    improved = np.isfinite(best_feas) and (best_feas < (init_feas - 1e-6))
                    filter_ok = (best_metrics is not None) and filter_accept(init_metrics, best_metrics)
                    resto_ok = (best_metrics is not None) and (best_metrics["theta"] <= restoration_theta_ratio * init_metrics["theta"])
                    if filter_ok or (improved and resto_ok):
                        w_opt_np[k] = x_best
                        if diag_modes[k] not in ("retry1", "retry2"):
                            diag_modes[k] = "best-feas"
                            diag_mode_codes[k] = mode_name_to_code["best-feas"]
                        if summary_log:
                            print(
                                f"[JAX-SubP2][batch] k={k} accepted best-feas | "
                                f"init_feas={init_feas:.2e} | best_feas={best_feas:.2e}",
                                flush=True,
                            )
                    else:
                        w_opt_np[k] = w_init_np[k]
                        diag_modes[k] = "fallback"
                        diag_mode_codes[k] = mode_name_to_code["fallback"]
                        diag_converged[k] = False
                        diag_iters[k] = 0
                        if summary_log:
                            print(
                                f"[JAX-SubP2][batch] k={k} fallback to warm-start | "
                                f"init_feas={init_feas:.2e} | best_feas={best_feas:.2e}",
                                flush=True,
                            )
                w_opt_batch = jnp.array(w_opt_np)
                diag_eq_inf = []
                diag_ineq_vio = []
                for k in range(N + 1):
                    p_t = {kk: vv[k] for kk, vv in params_batch.items()}
                    eq_inf, ineq_vio, _ = primal_feas_metrics(w_opt_batch[k], p_t)
                    diag_eq_inf.append(float(eq_inf))
                    diag_ineq_vio.append(float(ineq_vio))

        # ipoptax 在个别时刻可能返回 NaN/Inf，避免污染后续 ADMM。
        # 对非有限结果回退到该时刻的 warm-start（DDP 理想值）。
        finite_rows = jnp.all(jnp.isfinite(w_opt_batch), axis=1, keepdims=True)
        w_opt_batch = jnp.where(finite_rows, w_opt_batch, w_init_batch)

        # --- 5. 结果拆解与还原 (NumPy 适配) ---
        result = self._unpack_subp2_results(w_opt_batch)
        result["diag"] = {
            "mode": [mode_code_to_name.get(int(code), "fallback") for code in np.array(diag_mode_codes).tolist()],
            "mode_code": np.array(diag_mode_codes, dtype=int),
            "converged": np.array(diag_converged, dtype=bool),
            "iterations": np.array(diag_iters, dtype=int),
            "eq_inf": np.array(diag_eq_inf, dtype=float),
            "ineq_vio": np.array(diag_ineq_vio, dtype=float),
            "tail_mu": np.array(diag_tail_mu, dtype=float),
            "tail_ineq": np.array(diag_tail_ineq, dtype=float),
            "tail_comp": np.array(diag_tail_comp, dtype=float),
            "tail_dual": np.array(diag_tail_dual, dtype=float),
        }
        return result

    # 可先跳过：纯 JAX SubP2 的 SQP 实验分支。
    # 这是研究/对照路径，不是当前稳定 persistent Julia 主线。
    def jax_ADMM_SubP2_SQP(self, Para2_dict):
        N, nq = self.N, int(self.nq)
        nxl, nul, nxi, nui = self.nxl, self.nul, self.nxi, self.nui
        dims = (nxl, nul, nxi, nui, nq, int(self.num_dis))

        def _env_int(name, default):
            try:
                return int(os.environ.get(name, default))
            except Exception:
                return int(default)

        def _env_float(name, default):
            try:
                return float(os.environ.get(name, default))
            except Exception:
                return float(default)

        sqp_iters = _env_int("JAX_SUBP2_SQP_ITERS", 4)
        objective_weight = _env_float("JAX_SUBP2_SQP_OBJECTIVE_WEIGHT", 1.0)
        prox_weight = _env_float("JAX_SUBP2_SQP_PROX_WEIGHT", 0.0)
        prox_center = os.environ.get("JAX_SUBP2_SQP_PROX_CENTER", "init").strip().lower()
        step_norm_cap = _env_float("JAX_SUBP2_SQP_STEP_NORM_CAP", 1.0)
        reg = _env_float("JAX_SUBP2_SQP_REG", 1e-4)
        merit_rho = _env_float("JAX_SUBP2_SQP_MERIT_RHO", 10.0)
        osqp_maxiter = _env_int("JAX_SUBP2_SQP_OSQP_MAXITER", 200)
        osqp_tol = _env_float("JAX_SUBP2_SQP_OSQP_TOL", 1e-5)
        osqp_rho_start = _env_float("JAX_SUBP2_SQP_OSQP_RHO_START", 0.1)
        osqp_sigma = _env_float("JAX_SUBP2_SQP_OSQP_SIGMA", 1e-4)
        osqp_momentum = _env_float("JAX_SUBP2_SQP_OSQP_MOMENTUM", 1.6)
        qp_var_scale = os.environ.get("JAX_SUBP2_SQP_QP_VAR_SCALE", "1") == "1"
        qp_row_scale = os.environ.get("JAX_SUBP2_SQP_QP_ROW_SCALE", "1") == "1"
        active_set_k = _env_int("JAX_SUBP2_SQP_ACTIVE_SET_K", 0)
        active_set_after_iter = _env_int("JAX_SUBP2_SQP_ACTIVE_SET_AFTER_ITER", 99)
        use_lagrangian_hessian = os.environ.get("JAX_SUBP2_SQP_USE_LAGRANGIAN_HESSIAN", "0") == "1"
        internal_fp32 = os.environ.get("JAX_SUBP2_SQP_INTERNAL_FP32", "0") == "1"
        early_stop = os.environ.get("JAX_SUBP2_SQP_EARLY_STOP", "1") == "1"
        early_stop_eq = _env_float("JAX_SUBP2_SQP_EARLY_STOP_EQ", 2.5e-4)
        early_stop_ineq = _env_float("JAX_SUBP2_SQP_EARLY_STOP_INEQ", 1e-6)
        early_stop_step = _env_float("JAX_SUBP2_SQP_EARLY_STOP_STEP", 1.1e-2)
        summary_log = os.environ.get("JAX_SUBP2_SUMMARY_LOG", "1") == "1"
        fixed_alpha = _env_float("JAX_SUBP2_SQP_FIXED_ALPHA", 1.0)

        w_init_batch, params_batch = self._prepare_subp2_batch(Para2_dict)
        solve_dtype = jnp.float32 if internal_fp32 else jnp.float64
        x_batch = jnp.array(w_init_batch, dtype=solve_dtype)
        params_batch = {k: jnp.array(v, dtype=solve_dtype) for k, v in params_batch.items()}
        x_ref_batch0 = x_batch

        runtime_full = self._get_subp2_sqp_runtime(
            dims=dims,
            objective_weight=objective_weight,
            prox_weight=prox_weight,
            step_norm_cap=step_norm_cap,
            reg=reg,
            merit_rho=merit_rho,
            osqp_maxiter=osqp_maxiter,
            osqp_tol=osqp_tol,
            osqp_rho_start=osqp_rho_start,
            osqp_sigma=osqp_sigma,
            osqp_momentum=osqp_momentum,
            qp_var_scale=qp_var_scale,
            qp_row_scale=qp_row_scale,
            active_set_k=0,
            fixed_alpha=fixed_alpha,
            use_lagrangian_hessian=use_lagrangian_hessian,
            solve_dtype=str(solve_dtype),
        )
        runtime_active = None
        if active_set_k > 0:
            runtime_active = self._get_subp2_sqp_runtime(
                dims=dims,
                objective_weight=objective_weight,
                prox_weight=prox_weight,
                step_norm_cap=step_norm_cap,
                reg=reg,
                merit_rho=merit_rho,
                osqp_maxiter=osqp_maxiter,
                osqp_tol=osqp_tol,
                osqp_rho_start=osqp_rho_start,
                osqp_sigma=osqp_sigma,
                osqp_momentum=osqp_momentum,
                qp_var_scale=qp_var_scale,
                qp_row_scale=qp_row_scale,
                active_set_k=active_set_k,
                fixed_alpha=fixed_alpha,
                use_lagrangian_hessian=use_lagrangian_hessian,
                solve_dtype=str(solve_dtype),
            )

        x_ref_batch = x_ref_batch0 if prox_center == "init" else x_batch
        runtime = runtime_full
        runtime_mode = "full"
        osqp_params_batch = runtime["batched_init_params"](x_batch, params_batch, x_ref_batch)
        diag_iters = np.zeros((N + 1,), dtype=int)
        diag_eq_inf_jax = jnp.full((N + 1,), jnp.nan, dtype=x_batch.dtype)
        diag_ineq_vio_jax = jnp.full((N + 1,), jnp.nan, dtype=x_batch.dtype)
        for it in range(sqp_iters):
            next_mode = "active" if (runtime_active is not None and (it + 1) > active_set_after_iter) else "full"
            runtime = runtime_active if next_mode == "active" else runtime_full
            x_ref_batch = x_ref_batch0 if prox_center == "init" else x_batch
            if next_mode != runtime_mode:
                osqp_params_batch = runtime["batched_init_params"](x_batch, params_batch, x_ref_batch)
                runtime_mode = next_mode
            d_batch, sol_batch, osqp_iters, _ = runtime["batched_direction"](x_batch, params_batch, x_ref_batch, osqp_params_batch)
            osqp_params_batch = sol_batch
            if fixed_alpha > 0.0:
                alpha_batch = jnp.full((N + 1,), fixed_alpha, dtype=x_batch.dtype)
            else:
                alpha_batch = runtime["batched_alpha"](x_batch, params_batch, d_batch)
            x_batch = x_batch + alpha_batch[:, None] * d_batch

            c_batch = runtime["batched_c"](x_batch, params_batch)
            g_batch = runtime["batched_g"](x_batch, params_batch)
            eq_inf_jax = jnp.max(jnp.abs(c_batch), axis=1)
            ineq_vio_jax = jnp.maximum(0.0, jnp.max(g_batch, axis=1))
            step_norm_mean_jax = jnp.mean(jnp.linalg.norm(d_batch, axis=1))
            diag_iters = np.array(osqp_iters, dtype=int)
            diag_eq_inf_jax = eq_inf_jax
            diag_ineq_vio_jax = ineq_vio_jax
            eq_inf_max = float(jnp.max(eq_inf_jax))
            ineq_vio_max = float(jnp.max(ineq_vio_jax))
            step_norm_mean = float(step_norm_mean_jax)
            if summary_log:
                print(
                    f"[JAX-SubP2][sqp] iter={it+1} "
                    f"osqp_iter_mean={float(np.mean(diag_iters)):.1f} "
                    f"step_norm_mean={step_norm_mean:.3e} "
                    f"eq_max={eq_inf_max:.3e} "
                    f"ineq_max={ineq_vio_max:.3e}",
                    flush=True,
                )
            if (
                early_stop
                and eq_inf_max <= early_stop_eq
                and ineq_vio_max <= early_stop_ineq
                and step_norm_mean <= early_stop_step
            ):
                if summary_log:
                    print(f"[JAX-SubP2][sqp] early-stop at iter={it+1}", flush=True)
                break

        diag_eq_inf = np.array(diag_eq_inf_jax, dtype=float)
        diag_ineq_vio = np.array(diag_ineq_vio_jax, dtype=float)

        result = self._unpack_subp2_results(x_batch.astype(jnp.float64))
        result["diag"] = {
            "mode": np.array(["main"] * (N + 1), dtype=object),
            "mode_code": np.full((N + 1,), 1, dtype=int),
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

    # 可先跳过：纯 JAX SubP2 的 barrier 实验分支。
    # 这也是研究/对照路径，不是当前稳定 persistent Julia 主线。
    def jax_ADMM_SubP2_BARRIER(self, Para2_dict):
        N, nq = self.N, int(self.nq)
        nxl, nul, nxi, nui = self.nxl, self.nul, self.nxi, self.nui
        dims = (nxl, nul, nxi, nui, nq, int(self.num_dis))

        def _env_int(name, default):
            try:
                return int(os.environ.get(name, default))
            except Exception:
                return int(default)

        def _env_float(name, default):
            try:
                return float(os.environ.get(name, default))
            except Exception:
                return float(default)

        def _env_bool(name, default):
            return os.environ.get(name, "1" if default else "0") == "1"

        mu_schedule_text = os.environ.get(
            "JAX_SUBP2_BARRIER_MU_SCHEDULE",
            "1e-3,3e-4,1e-4,3e-5",
        )
        mu_schedule = jnp.array(
            [float(x.strip()) for x in mu_schedule_text.split(",") if x.strip()],
            dtype=jnp.float64,
        )
        maxiter = _env_int("JAX_SUBP2_BARRIER_MAXITER", 60)
        maxiter_schedule_text = os.environ.get("JAX_SUBP2_BARRIER_MAXITER_SCHEDULE", "60,50,40,30").strip()
        if maxiter_schedule_text:
            maxiter_schedule = np.array(
                [int(x.strip()) for x in maxiter_schedule_text.split(",") if x.strip()],
                dtype=int,
            )
            if maxiter_schedule.size != int(mu_schedule.shape[0]):
                raise ValueError(
                    "JAX_SUBP2_BARRIER_MAXITER_SCHEDULE length must match JAX_SUBP2_BARRIER_MU_SCHEDULE length"
                )
        else:
            maxiter_schedule = np.full((int(mu_schedule.shape[0]),), maxiter, dtype=int)
        admm_maxiter_scale_text = os.environ.get("JAX_SUBP2_BARRIER_ADMM_MAXITER_SCALE_SCHEDULE", "").strip()
        if admm_maxiter_scale_text:
            admm_maxiter_scales = np.array(
                [float(x.strip()) for x in admm_maxiter_scale_text.split(",") if x.strip()],
                dtype=float,
            )
        else:
            admm_maxiter_scales = None
        tol = _env_float("JAX_SUBP2_BARRIER_TOL", 1e-4)
        tol_schedule_text = os.environ.get("JAX_SUBP2_BARRIER_TOL_SCHEDULE", "").strip()
        if tol_schedule_text:
            tol_schedule = np.array(
                [float(x.strip()) for x in tol_schedule_text.split(",") if x.strip()],
                dtype=float,
            )
            if tol_schedule.size != int(mu_schedule.shape[0]):
                raise ValueError(
                    "JAX_SUBP2_BARRIER_TOL_SCHEDULE length must match JAX_SUBP2_BARRIER_MU_SCHEDULE length"
                )
        else:
            tol_schedule = np.full((int(mu_schedule.shape[0]),), tol, dtype=float)
        maxls = _env_int("JAX_SUBP2_BARRIER_MAXLS", 20)
        linesearch = os.environ.get("JAX_SUBP2_BARRIER_LINESEARCH", "zoom").strip().lower()
        eq_penalty_scale = _env_float("JAX_SUBP2_BARRIER_EQ_PENALTY_SCALE", 100.0)
        eq_rho = _env_float("JAX_SUBP2_BARRIER_EQ_RHO", 100.0)
        eq_rho_schedule_text = os.environ.get("JAX_SUBP2_BARRIER_EQ_RHO_SCHEDULE", "").strip()
        eq_rho_coupled_to_mu = _env_bool("JAX_SUBP2_BARRIER_EQ_RHO_COUPLED_TO_MU", False)
        if eq_rho_schedule_text:
            eq_rho_schedule = np.array(
                [float(x.strip()) for x in eq_rho_schedule_text.split(",") if x.strip()],
                dtype=float,
            )
            if eq_rho_schedule.size != int(mu_schedule.shape[0]):
                raise ValueError(
                    "JAX_SUBP2_BARRIER_EQ_RHO_SCHEDULE length must match JAX_SUBP2_BARRIER_MU_SCHEDULE length"
                )
        else:
            eq_rho_schedule = np.full((int(mu_schedule.shape[0]),), eq_rho, dtype=float)
        ineq_penalty_scale = _env_float("JAX_SUBP2_BARRIER_INEQ_PENALTY_SCALE", 100.0)
        barrier_scale = _env_float("JAX_SUBP2_BARRIER_SCALE", 1.0)
        barrier_eps = _env_float("JAX_SUBP2_BARRIER_EPS", 1e-6)
        barrier_min_slack = _env_float("JAX_SUBP2_BARRIER_MIN_SLACK", 1e-8)
        eq_lam_update = _env_bool("JAX_SUBP2_BARRIER_EQ_LAM_UPDATE", True)
        eq_lam_update_scale = _env_float("JAX_SUBP2_BARRIER_EQ_LAM_UPDATE_SCALE", 0.25)
        eq_lam_clip = _env_float("JAX_SUBP2_BARRIER_EQ_LAM_CLIP", 1e3)
        prox_weight = _env_float("JAX_SUBP2_BARRIER_PROX_WEIGHT", 0.0)
        use_scales = _env_bool("JAX_SUBP2_BARRIER_USE_SCALES", True)
        project_unit_equalities = _env_bool("JAX_SUBP2_BARRIER_PROJECT_UNIT_EQUALITIES", True)
        wrench_only_equality = _env_bool("JAX_SUBP2_BARRIER_WRENCH_ONLY_EQUALITY", True)
        project_wrench_equality = _env_bool("JAX_SUBP2_BARRIER_PROJECT_WRENCH_EQUALITY", True)
        projection_eps = _env_float("JAX_SUBP2_BARRIER_PROJECTION_EPS", 1e-9)
        early_stop = _env_bool("JAX_SUBP2_BARRIER_EARLY_STOP", True)
        early_stop_eq = _env_float("JAX_SUBP2_BARRIER_EARLY_STOP_EQ", 1e-6)
        early_stop_ineq = _env_float("JAX_SUBP2_BARRIER_EARLY_STOP_INEQ", 1e-8)
        early_stop_step = _env_float("JAX_SUBP2_BARRIER_EARLY_STOP_STEP", 2e-3)
        admm_early_stop_step_scale_text = os.environ.get(
            "JAX_SUBP2_BARRIER_ADMM_EARLY_STOP_STEP_SCALE_SCHEDULE", ""
        ).strip()
        if admm_early_stop_step_scale_text:
            admm_early_stop_step_scales = np.array(
                [float(x.strip()) for x in admm_early_stop_step_scale_text.split(",") if x.strip()],
                dtype=float,
            )
        else:
            admm_early_stop_step_scales = None
        summary_log = os.environ.get("JAX_SUBP2_SUMMARY_LOG", "1") == "1"

        w_init_batch, params_batch = self._prepare_subp2_batch(Para2_dict)
        x_batch = jnp.array(w_init_batch, dtype=jnp.float64)
        params_batch = {k: jnp.array(v, dtype=jnp.float64) for k, v in params_batch.items()}
        admm_iter_idx = int(float(np.array(params_batch["admm_iter"][0])))
        if admm_maxiter_scales is not None and admm_maxiter_scales.size > 0:
            scale_idx = min(admm_iter_idx, admm_maxiter_scales.size - 1)
            maxiter_scale = float(admm_maxiter_scales[scale_idx])
            maxiter_schedule = np.maximum(1, np.rint(maxiter_schedule.astype(float) * maxiter_scale).astype(int))
        if admm_early_stop_step_scales is not None and admm_early_stop_step_scales.size > 0:
            scale_idx = min(admm_iter_idx, admm_early_stop_step_scales.size - 1)
            early_stop_step = float(early_stop_step) * float(admm_early_stop_step_scales[scale_idx])

        runtime_common = self._get_subp2_barrier_runtime_common(
            dims=dims,
            linesearch=linesearch,
            maxls=maxls,
            eq_penalty_scale=eq_penalty_scale,
            eq_rho_coupled_to_mu=eq_rho_coupled_to_mu,
            ineq_penalty_scale=ineq_penalty_scale,
            barrier_scale=barrier_scale,
            barrier_eps=barrier_eps,
            barrier_min_slack=barrier_min_slack,
            prox_weight=prox_weight,
            use_scales=use_scales,
            project_unit_equalities=project_unit_equalities,
            wrench_only_equality=wrench_only_equality,
            project_wrench_equality=project_wrench_equality,
            projection_eps=projection_eps,
        )

        x_ref_batch = runtime_common["v_project_full_x"](x_batch, params_batch)
        x_cur = x_ref_batch
        lam_eq = jnp.zeros_like(runtime_common["batched_c"](x_ref_batch, params_batch))
        diag_iters = np.zeros((N + 1,), dtype=int)
        diag_eq_inf_jax = jnp.full((N + 1,), jnp.nan, dtype=jnp.float64)
        diag_ineq_vio_jax = jnp.full((N + 1,), jnp.nan, dtype=jnp.float64)

        for it, mu in enumerate(mu_schedule):
            runtime = self._get_subp2_barrier_runtime_stage(
                runtime_common=runtime_common,
                maxiter=int(maxiter_schedule[it]),
                tol=float(tol_schedule[it]),
            )
            x_prev = x_cur
            rho_eq_stage = float(eq_rho_schedule[it])
            x_cur, iter_nums, errors, _ = runtime["batched_solve"](
                x_cur, params_batch, mu, rho_eq_stage, x_ref_batch, lam_eq
            )
            c_batch = runtime_common["batched_c"](x_cur, params_batch)
            c_raw_batch = runtime_common["batched_c_raw"](x_cur, params_batch)
            g_raw_batch = runtime_common["batched_g_raw"](x_cur, params_batch)
            eq_inf_jax = jnp.max(jnp.abs(c_raw_batch), axis=1)
            ineq_vio_jax = jnp.maximum(0.0, jnp.max(g_raw_batch, axis=1))
            step_norms = jnp.linalg.norm(x_cur - x_prev, axis=1)
            diag_iters = np.array(iter_nums, dtype=int)
            diag_eq_inf_jax = eq_inf_jax
            diag_ineq_vio_jax = ineq_vio_jax
            eq_max = float(jnp.max(eq_inf_jax))
            ineq_max = float(jnp.max(ineq_vio_jax))
            step_norm_mean = float(jnp.mean(step_norms))
            if summary_log:
                print(
                    f"[JAX-SubP2][barrier] stage={it+1} mu={float(mu):.2e} "
                    f"rho_eq={float(rho_eq_stage):.2e} "
                    f"maxiter={int(maxiter_schedule[it])} tol={float(tol_schedule[it]):.1e} "
                    f"iter_mean={float(np.mean(diag_iters)):.1f} "
                    f"step_norm_mean={step_norm_mean:.3e} "
                    f"eq_max={eq_max:.3e} ineq_max={ineq_max:.3e}",
                    flush=True,
                )
            if eq_lam_update and lam_eq.shape[1] > 0:
                rho_eq = (eq_penalty_scale / float(mu)) if eq_rho_coupled_to_mu else float(rho_eq_stage)
                lam_eq = jnp.clip(
                    lam_eq + eq_lam_update_scale * rho_eq * c_batch,
                    -eq_lam_clip,
                    eq_lam_clip,
                )
            if (
                early_stop
                and eq_max <= early_stop_eq
                and ineq_max <= early_stop_ineq
                and step_norm_mean <= early_stop_step
            ):
                if summary_log:
                    print(f"[JAX-SubP2][barrier] early-stop at stage={it+1}", flush=True)
                break

        diag_mode = np.array(["main"] * (N + 1), dtype=object)
        diag_mode_code = np.ones((N + 1,), dtype=int)
        diag_conv = np.ones((N + 1,), dtype=bool)
        diag_iters_full = np.array(diag_iters, dtype=int)
        diag_eq_full = np.array(diag_eq_inf_jax, dtype=float)
        diag_ineq_full = np.array(diag_ineq_vio_jax, dtype=float)

        result = self._unpack_subp2_results(x_cur)
        result["diag"] = {
            "mode": diag_mode,
            "mode_code": diag_mode_code,
            "converged": diag_conv,
            "iterations": diag_iters_full,
            "eq_inf": diag_eq_full,
            "ineq_vio": diag_ineq_full,
            "tail_mu": np.full((N + 1,), np.nan, dtype=float),
            "tail_ineq": np.full((N + 1,), np.nan, dtype=float),
            "tail_comp": np.full((N + 1,), np.nan, dtype=float),
            "tail_dual": np.full((N + 1,), np.nan, dtype=float),
        }
        return result

    def _get_subp2_barrier_runtime_common(
        self,
        *,
        dims,
        linesearch,
        maxls,
        eq_penalty_scale,
        eq_rho_coupled_to_mu,
        ineq_penalty_scale,
        barrier_scale,
        barrier_eps,
        barrier_min_slack,
        prox_weight,
        use_scales,
        project_unit_equalities,
        wrench_only_equality,
        project_wrench_equality,
        projection_eps,
    ):
        key = (
            dims,
            str(linesearch),
            int(maxls),
            float(eq_penalty_scale),
            bool(eq_rho_coupled_to_mu),
            float(ineq_penalty_scale),
            float(barrier_scale),
            float(barrier_eps),
            float(barrier_min_slack),
            float(prox_weight),
            bool(use_scales),
            bool(project_unit_equalities),
            bool(wrench_only_equality),
            bool(project_wrench_equality),
            float(projection_eps),
        )
        cache = _GLOBAL_SUBP2_BARRIER_RUNTIME_COMMON_CACHE
        if key in cache:
            return cache[key]

        nxl, nul, nxi, nui, nq = [int(v) for v in dims[:5]]
        quat_slice = slice(6, 10)

        def project_full_x(x, p):
            xl = x[:nxl]
            ul = x[nxl:nxl + nul]
            rem = x[nxl + nul :].reshape(nq, nxi + nui)
            xc = rem[:, :nxi]
            uc = rem[:, nxi:]

            if project_unit_equalities:
                ql = xl[quat_slice]
                ql = ql / jnp.maximum(jnp.linalg.norm(ql), projection_eps)
                xl = xl.at[quat_slice].set(ql)

                di = xc[:, 0:3]
                di_norm = jnp.maximum(jnp.linalg.norm(di, axis=1, keepdims=True), projection_eps)
                xc = xc.at[:, 0:3].set(di / di_norm)

            if project_wrench_equality:
                ql = xl[quat_slice]
                Rl = self._q_2_rotation_jax(ql)
                di_vecs = xc[:, 0:3]
                ti_mags = xc[:, 12]
                fi_inertial = di_vecs * ti_mags[:, None]
                fi_body = (Rl.T @ fi_inertial.T).T
                wrench_generated = p["Pt"] @ fi_body.flatten()
                ul_force = Rl @ wrench_generated[:3]
                ul_proj = ul.at[0:3].set(ul_force)
                ul_proj = ul_proj.at[3:6].set(wrench_generated[3:6])
                ul = jnp.where((p["is_terminal"] < 0.5), ul_proj, ul)

            return jnp.concatenate([xl, ul, jnp.concatenate([xc, uc], axis=1).reshape(-1)])

        def select_equalities(c_all):
            if project_wrench_equality:
                return c_all[0:0]
            if wrench_only_equality:
                return c_all[1 + nq :]
            return c_all

        def f_step_from_proj(x_proj, p):
            return self.ipoptax_objective(x_proj, p, dims)

        def c_step_from_proj(x_proj, p):
            c = self.ipoptax_equality(x_proj, p, dims)
            c = select_equalities(c)
            return p["eq_scale"] * c if use_scales else c

        def g_step_from_proj(x_proj, p):
            g = self.ipoptax_inequality(x_proj, p, dims)
            return p["ineq_scale"] * g if use_scales else g

        def f_step(x, p):
            return f_step_from_proj(project_full_x(x, p), p)

        def c_step(x, p):
            return c_step_from_proj(project_full_x(x, p), p)

        def g_step(x, p):
            return g_step_from_proj(project_full_x(x, p), p)

        def barrier_loss(x, p, mu, rho_eq, x_ref, lam_eq):
            x_proj = project_full_x(x, p)
            f = f_step_from_proj(x_proj, p)
            c = c_step_from_proj(x_proj, p)
            g = g_step_from_proj(x_proj, p)
            slack = jnp.maximum(-g + barrier_eps, barrier_min_slack)
            barrier = -barrier_scale * mu * jnp.sum(jnp.log(slack))
            eq_lin = jnp.dot(lam_eq, c)
            rho_eff = (eq_penalty_scale / mu) if eq_rho_coupled_to_mu else rho_eq
            eq_pen = 0.5 * rho_eff * jnp.sum(c**2)
            vio = jnp.maximum(g - barrier_eps, 0.0)
            ineq_pen = 0.5 * (ineq_penalty_scale / mu) * jnp.sum(vio**2)
            prox = 0.5 * prox_weight * jnp.sum((x - x_ref) ** 2)
            return f + barrier + eq_lin + eq_pen + ineq_pen + prox

        runtime = {
            "cache_key": key,
            "barrier_loss": barrier_loss,
            "linesearch": linesearch,
            "maxls": maxls,
            "project_one": jax.jit(project_full_x),
            "v_project_full_x": jax.jit(jax.vmap(project_full_x, in_axes=(0, 0))),
            "batched_c": jax.jit(jax.vmap(c_step, in_axes=(0, 0))),
            "batched_g": jax.jit(jax.vmap(g_step, in_axes=(0, 0))),
            "batched_c_raw": jax.jit(jax.vmap(lambda x, p: self.ipoptax_equality(project_full_x(x, p), p, dims), in_axes=(0, 0))),
            "batched_g_raw": jax.jit(jax.vmap(lambda x, p: self.ipoptax_inequality(project_full_x(x, p), p, dims), in_axes=(0, 0))),
        }
        cache[key] = runtime
        return runtime

    def _get_subp2_barrier_runtime_stage(self, *, runtime_common, maxiter, tol):
        key = (runtime_common["cache_key"], int(maxiter), float(tol))
        cache = _GLOBAL_SUBP2_BARRIER_RUNTIME_STAGE_CACHE
        if key in cache:
            return cache[key]

        solver = BFGS(
            fun=runtime_common["barrier_loss"],
            maxiter=maxiter,
            tol=tol,
            linesearch=runtime_common["linesearch"],
            maxls=runtime_common["maxls"],
            jit=True,
            verbose=False,
        )

        project_full_x = runtime_common["project_one"]

        def solve_one(x0, p, mu, rho_eq, x_ref, lam_eq):
            out = solver.run(x0, p, mu, rho_eq, x_ref, lam_eq)
            x_proj = project_full_x(out.params, p)
            return x_proj, out.state.iter_num, out.state.error, out.state.value

        stage_runtime = {
            "batched_solve": jax.jit(jax.vmap(solve_one, in_axes=(0, 0, None, None, 0, 0))),
        }
        cache[key] = stage_runtime
        return stage_runtime

    def _get_subp2_sqp_runtime(
        self,
        *,
        dims,
        objective_weight,
        prox_weight,
        step_norm_cap,
        reg,
        merit_rho,
        osqp_maxiter,
        osqp_tol,
        osqp_rho_start,
        osqp_sigma,
        osqp_momentum,
        qp_var_scale,
        qp_row_scale,
        active_set_k,
        fixed_alpha,
        use_lagrangian_hessian,
        solve_dtype,
    ):
        key = (
            dims,
            float(objective_weight),
            float(prox_weight),
            float(step_norm_cap),
            float(reg),
            float(merit_rho),
            int(osqp_maxiter),
            float(osqp_tol),
            float(osqp_rho_start),
            float(osqp_sigma),
            float(osqp_momentum),
            bool(qp_var_scale),
            bool(qp_row_scale),
            int(active_set_k),
            float(fixed_alpha),
            bool(use_lagrangian_hessian),
            str(solve_dtype),
        )
        cache = getattr(self, "_subp2_sqp_runtime_cache", None)
        if cache is None:
            cache = {}
            self._subp2_sqp_runtime_cache = cache
        if key in cache:
            return cache[key]

        solver = OSQP(
            eq_qp_solve="lu",
            maxiter=osqp_maxiter,
            tol=osqp_tol,
            termination_check_frequency=1,
            check_primal_dual_infeasability=False,
            sigma=osqp_sigma,
            rho_start=osqp_rho_start,
            momentum=osqp_momentum,
        )
        run_dtype = jnp.float32 if solve_dtype == str(jnp.float32) else jnp.float64
        alpha_candidates = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125], dtype=run_dtype)

        def f_step(x, p):
            return self.ipoptax_objective(x, p, dims)

        def c_step(x, p):
            return self.ipoptax_equality(x, p, dims)

        def g_step(x, p):
            return self.ipoptax_inequality(x, p, dims)

        def merit_step(x, p):
            cval = c_step(x, p)
            gval = g_step(x, p)
            return f_step(x, p) + merit_rho * (
                jnp.linalg.norm(cval) + jnp.linalg.norm(jnp.maximum(gval, 0.0))
            )

        def build_qp_data(x, p, x_ref, lam_eq, lam_ineq):
            grad_f = jax.grad(lambda xx: f_step(xx, p))(x)
            cval = c_step(x, p)
            gval = g_step(x, p)
            Aeq = jax.jacfwd(lambda xx: c_step(xx, p))(x)
            Aineq = jax.jacfwd(lambda xx: g_step(xx, p))(x)
            if use_lagrangian_hessian:
                lag = lambda xx: f_step(xx, p) + jnp.dot(lam_eq, c_step(xx, p)) + jnp.dot(lam_ineq, g_step(xx, p))
                hess_base = jax.hessian(lag)(x)
            else:
                hess_base = jax.hessian(lambda xx: f_step(xx, p))(x)
            H = project_psd_cone(hess_base, delta=reg, use_lapack=False, iterate=True)
            Hn = (
                objective_weight * H
                + (1.0 - objective_weight) * reg * jnp.eye(H.shape[0], dtype=H.dtype)
                + prox_weight * jnp.eye(H.shape[0], dtype=H.dtype)
            )
            gn = objective_weight * grad_f + prox_weight * (x - x_ref)
            beq = -cval
            bineq = -gval
            if active_set_k > 0:
                k = min(active_set_k, int(gval.shape[0]))
                top_idx = jnp.argsort(gval)[-k:]
                Aineq = Aineq[top_idx, :]
                bineq = bineq[top_idx]

            if qp_var_scale:
                hdiag = jnp.maximum(jnp.abs(jnp.diag(Hn)), reg)
                var_scale = jnp.sqrt(hdiag)
            else:
                var_scale = jnp.ones(Hn.shape[0], dtype=Hn.dtype)

            d_inv = 1.0 / jnp.maximum(var_scale, 1e-8)
            Hs = (d_inv[:, None] * Hn) * d_inv[None, :]
            gs = d_inv * gn
            Aeqs = Aeq * d_inv[None, :]
            Aineqs = Aineq * d_inv[None, :]
            if qp_row_scale:
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
            if active_set_k > 0:
                lam_ineq0 = jnp.zeros((min(active_set_k, int(g0.shape[0])),), dtype=g0.dtype)
            else:
                lam_ineq0 = jnp.zeros_like(g0)
            Hn, gn, Aeq, beq, Aineq, bineq, _ = build_qp_data(x, p, x_ref, lam_eq0, lam_ineq0)
            return solver.init_params(
                init_x=jnp.zeros_like(x),
                params_obj=(Hn, gn),
                params_eq=(Aeq, beq),
                params_ineq=(Aineq, bineq),
            )

        def sqp_direction_one(x, p, x_ref, init_params):
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
            scale = jnp.minimum(1.0, step_norm_cap / jnp.maximum(d_norm, 1e-12))
            pred_red = -(jnp.dot(gn, d) + 0.5 * jnp.dot(d, Hn @ d))
            scaled_sol = sol._replace(primal=sol.primal * scale)
            return d * scale, scaled_sol, step.state.iter_num, pred_red

        def choose_alpha_one(x, p, d):
            if fixed_alpha > 0.0:
                return jnp.array(fixed_alpha, dtype=x.dtype)
            m0 = merit_step(x, p)
            trial_x = x[None, :] + alpha_candidates[:, None] * d[None, :]
            trial_m = jax.vmap(lambda xx: merit_step(xx, p))(trial_x)
            ok = trial_m <= m0
            first_idx = jnp.argmax(ok)
            any_ok = jnp.any(ok)
            return jnp.where(any_ok, alpha_candidates[first_idx], 0.0)

        runtime = {
            "batched_init_params": jax.jit(jax.vmap(init_osqp_params_one, in_axes=(0, 0, 0))),
            "batched_direction": jax.jit(jax.vmap(sqp_direction_one, in_axes=(0, 0, 0, 0))),
            "batched_alpha": jax.jit(jax.vmap(choose_alpha_one, in_axes=(0, 0, 0))),
            "batched_c": jax.jit(jax.vmap(c_step, in_axes=(0, 0))),
            "batched_g": jax.jit(jax.vmap(g_step, in_axes=(0, 0))),
        }
        cache[key] = runtime
        return runtime




    # ---------------------------------------------------------------------
    # 以下开始大体进入"原版 CasADi / 旧 ADMM / 梯度训练"保留区。
    # 对当前稳定 persistent Julia benchmark 主线来说，这一大块基本都可先跳过。
    # 之所以还留在这个文件里，是因为历史实验、对照实现和训练代码还依赖它们。
    # ---------------------------------------------------------------------
    def Rotational_Inertia(self,rp):  # TODO: Deprecated Type 1
        # rp=(x,y,0), a column vector, is the coordinate of the point-mass added on the uniform circular plate in its body frame 
        ratio_m    = self.m1*self.m2/self.ml
        self.Jl    = self.Jlcom + ratio_m*(rp.T@rp*np.identity(3)-rp@rp.T)

    def allocation_martrix(self,rg):  # TODO: Deprecated Type 1
        self.alpha  = 2*np.pi/self.nq
        r0          = np.array([[self.rl,0,0]]).T - np.reshape(np.vstack((rg[0],rg[1],0)),(3,1))  # 1st cable attachment point in {Bl}
        self.ra     = r0
        S_r0        = self.skew_sym_numpy(r0)
        I3          = np.identity(3) # 3-by-3 identity matrix
        self.Pt      = np.vstack((I3,S_r0))
        for i in range(int(self.nq)-1):
            ri      = np.array([[self.rl*(math.cos((i+1)*self.alpha)),self.rl*(math.sin((i+1)*self.alpha)),0]]).T - np.reshape(np.vstack((rg[0],rg[1],0)),(3,1))
            S_ri    = self.skew_sym_numpy(ri)
            Pi      = np.vstack((I3,S_ri))
            self.Pt = np.append(self.Pt,Pi,axis=1) # the tension mapping matrix: 6-by-3nq with a rank of 6
            self.ra = np.append(self.ra,ri,axis=1) # a matrix that stores the attachment points
    
    def skew_sym_numpy(self, v): # TODO: Type2 
        v_cross = np.array([
            [0, -v[2, 0], v[1, 0]],
            [v[2, 0], 0, -v[0, 0]],
            [-v[1, 0], v[0, 0], 0]]
        )
        return v_cross
    
    def skew_sym(self, v):# TODO: Type2  # skew-symmetric operator
        v_cross = vertcat(
            horzcat(0, -v[2,0], v[1,0]),
            horzcat(v[2,0], 0, -v[0,0]),
            horzcat(-v[1,0], v[0,0], 0)
        )
        return v_cross
    
    @staticmethod
    def skew_sym_jax(v):
        """新增的 JAX 版本的向量叉乘的矩阵形式 """
        # v 是一个 3 维向量 [vx, vy, vz]
        return jnp.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0]
        ])

    def SetStateVariables(self, xl, xi): # TODO: Deprecated Type 1
        """Initialize symbolic state variables along with their bounds."""
        self.xl = xl
        self.xi = xi
        self.nxl = int(xl.numel())
        self.nxi = int(xi.numel())
        nq = int(self.nq)
        cable_stack_dim = self.nxi * nq

        # Safe copies and Lagrange multipliers for ADMM
        self.scxc = SX.sym('scxc', cable_stack_dim)
        self.xc = SX.sym('xc', cable_stack_dim)
        self.scxC = SX.sym('scxC', cable_stack_dim)
        self.scxl = SX.sym('scxl', self.nxl)
        self.scxi = SX.sym('scxi', self.nxi)
        self.scxL = SX.sym('scxL', self.nxl)  # Lagrangian multiplier of xl
        self.scxI = SX.sym('scxI', self.nxi)  # Lagrangian multiplier of xi

        # Box bounds for primal states
        self.xl_lb = [-1e19] * self.nxl
        self.xl_ub = [1e19] * self.nxl
        self.xi_lb = [-1e19] * self.nxi
        self.xi_ub = [1e19] * self.nxi

        # Physical tension limits
        self.t_min = 0.01
        self.t_max = 5  # the maximum tension force

        scxi_lb = [-1e19] * self.nxi
        scxi_ub = [1e19] * self.nxi
        if self.nxi >= 2:
            tension_idx = self.nxi - 2  # second last state stores the tension magnitude
            scxi_lb[tension_idx] = self.t_min
            scxi_ub[tension_idx] = self.t_max
        self.scxi_lb = scxi_lb
        self.scxi_ub = scxi_ub


    def SetCtrlVariables(self, ul, ui): # TODO: Deprecated Type 1
        self.ul    = ul
        self.ui    = ui
        self.nul   = ul.numel()
        self.nui   = ui.numel()
        self.scuc  = SX.sym('scuc',self.nui*int(self.nq))
        self.uc    = SX.sym('uc',self.nui*int(self.nq))
        self.scuC  = SX.sym('scuC',self.nui*int(self.nq))
        self.scul  = SX.sym('scul',self.nul)
        self.scui  = SX.sym('scui',self.nui)
        self.scuL  = SX.sym('scuL',self.nul) # Lagrangian multiplier of ul
        self.scuI  = SX.sym('scuI',self.nui) # Lagrangian multiplier of ui
        self.ul_lb = self.nul*[-1e19]
        self.ul_ub = self.nul*[1e19]
        self.ui_bound =1e3
        self.ui_lb = self.nui*[-self.ui_bound]
        self.ui_ub = self.nui*[self.ui_bound]

    def SetDyns(self, model_l, model_i): # TODO: Deprecated Type 1
        self.model_l = self.xl + self.dt*model_l # 4th-order Runge-Kutta discrete-time load dynamics model
        self.model_i = self.xi + self.dt*model_i # 4th-order Runge-Kutta discrete-time cable dynamics model
        self.model_l_fn = Function('mdynl',[self.xl, self.ul],[self.model_l],['xl0','ul0'],['mdynlf'])
        self.model_i_fn = Function('mdyni',[self.xi, self.ui],[self.model_i],['xi0','ui0'],['mdynif'])

    def SetWeightPara(self):# TODO: Deprecated 
        # self.nwsl    = self.nxl
        self.para_l  = SX.sym('paral',1,(2*self.nxl+self.nul+4)) # including the ADMM penalty parameter, px, pu, gammax, gammau
        self.npl     = self.para_l.numel()
        self.para_i  = SX.sym('parai',1,(2*self.nxi+self.nui+4)) # including the ADMM penalty parameter, pix, piu, gammaix, gammaiu
        self.npi     = self.para_i.numel()
        self.P_auto  = horzcat(self.para_l,self.para_i)
        self.n_Pauto = self.P_auto.numel()

    def Discount_rate(self,gamma,a,ADMM_max): # TODO Type 2
        dis = 1/(1+exp(-gamma*(a - int(ADMM_max/2))))
        return dis
    
    @staticmethod
    def Discount_rate_jax(gamma, a, ADMM_max):
        """新增的 jax 版本的基于 Sigmoid 的折扣率"""
        # gamma: 陡峭系数, a: 当前 ADMM 迭代步, ADMM_max: 最大迭代步
        return 1.0 / (1.0 + jnp.exp(-gamma * (a - ADMM_max / 2.0)))

    def open_loop_penalty(self,rho,gamma,a,ADMM_max,b=0.5): # TODO Type 2
        # rho_a = self.p_min + (rho - self.p_min) * 1/(1 + exp(-gamma*(a/(int(ADMM_max)-1)-b))) # iteration-dependent open-loop penalty policy
        rho_a = self.p_min + (rho - self.p_min) * 1/(1+exp(-gamma*(a - (ADMM_max-1)/2)))
        return rho_a
    
    def q_2_rotation(self, q):#TODO Type2 
         # from body frame to inertial frame
        # no normalization to avoid singularity in optimization
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3] # q0 denotes a scalar while q1, q2, and q3 represent rotational axes x, y, and z, respectively
        R = vertcat(
        horzcat( 2 * (q0 ** 2 + q1 ** 2) - 1, 2 * q1 * q2 - 2 * q0 * q3, 2 * q0 * q2 + 2 * q1 * q3),
        horzcat(2 * q0 * q3 + 2 * q1 * q2, 2 * (q0 ** 2 + q2 ** 2) - 1, 2 * q2 * q3 - 2 * q0 * q1),
        horzcat(2 * q1 * q3 - 2 * q0 * q2, 2 * q0 * q1 + 2 * q2 * q3, 2 * (q0 ** 2 + q3 ** 2) - 1)
        )
        return R
    
    def vee_map(self, v):
        vect = vertcat(v[2, 1], v[0, 2], v[1, 0])
        return vect
    
    @staticmethod
    def vee_map_jax(mat):
        """新增的jax 版本的，反对称矩阵转回向量 (skew_sym 的逆运算)"""
        # 输入 mat: [3, 3]
        return jnp.array([mat[2, 1], mat[0, 2], mat[1, 0]])

    def SetPayloadCostDyn(self,ADMM_max):# TODO: Deprecated 
        self.ref_xl   = SX.sym('refxl',self.nxl,1)
        self.ref_ul   = SX.sym('reful',self.nul,1) 
        track_error_l = self.xl - self.ref_xl
        ctrl_error_l  = self.ul - self.ref_ul
        self.a        = SX.sym('a',1) # the ADMM iteration index
        self.Ql_k     = diag(self.para_l[0,0:self.nxl])
        self.Ql_N     = diag(self.para_l[0,self.nxl:2*self.nxl])
        self.Rl_k     = diag(self.para_l[0,2*self.nxl:2*self.nxl+self.nul])
        self.px_dis   = self.open_loop_penalty(self.para_l[0,-4],self.para_l[0,-2],self.a,ADMM_max)
        self.pu_dis   = self.open_loop_penalty(self.para_l[0,-3],self.para_l[0,-1],self.a,ADMM_max)
        # path cost
        self.resid_xl = self.xl - self.scxl + self.scxL/self.px_dis
        self.resid_ul = self.ul - self.scul + self.scuL/self.pu_dis
        self.Jl_k     = 1/2 * (track_error_l.T@self.Ql_k@track_error_l + ctrl_error_l.T@self.Rl_k@ctrl_error_l) + self.px_dis/2*self.resid_xl.T@self.resid_xl + self.pu_dis/2*self.resid_ul.T@self.resid_ul
        self.Jl_kfn   = Function('Jl_k',[self.xl, self.ul, self.scxl, self.scxL, self.scul, self.scuL, self.ref_xl, self.ref_ul, self.para_l, self.a],[self.Jl_k],['xl0', 'ul0', 'scxl0', 'scxL0', 'scul0', 'scuL0', 'refxl0', 'reful0', 'paral0', 'a0'],['Jl_kf'])
        # terminal cost
        self.Jl_N     = 1/2 * track_error_l.T@self.Ql_N@track_error_l + self.px_dis/2*self.resid_xl.T@self.resid_xl
        self.Jl_Nfn   = Function('Jl_N',[self.xl, self.ref_xl, self.scxl, self.scxL, self.para_l, self.a],[self.Jl_N],['xl0', 'refxl0', 'scxl0', 'scxL0', 'paral0', 'a0'],['Jl_Nf'])
        # path cost of ADMM subproblem2
        self.Jl_P2_k  = self.px_dis/2*self.resid_xl.T@self.resid_xl + self.pu_dis/2*self.resid_ul.T@self.resid_ul 
        self.Jl_P2_k_fn = Function('Jl_P2_k',[self.xl, self.ul, self.scxl, self.scxL, self.scul, self.scuL, self.para_l, self.a],[self.Jl_P2_k],['xl0', 'ul0', 'scxl0', 'scxL0', 'scul0', 'scuL0', 'paral0', 'a0'],['Jl_P2_kf'])
        # terminal cost of ADMM subproblem2
        self.Jl_P2_N  = self.px_dis/2*self.resid_xl.T@self.resid_xl 
        self.Jl_P2_N_fn = Function('Jl_P2_N',[self.xl, self.scxl, self.scxL, self.para_l, self.a],[self.Jl_P2_N],['xl0', 'scxl0', 'scxL0', 'paral0', 'a0'],['Jl_P2_Nf'])


    def SetCableCostDyn(self,ADMM_max):# TODO: Deprecated 
        self.ref_xi   = SX.sym('refxi',self.nxi,1)
        self.ref_ui   = SX.sym('refui',self.nui,1)
        track_error_i = self.xi - self.ref_xi
        ctrl_error_i  = self.ui - self.ref_ui
        self.Qi_k     = diag(self.para_i[0,0:self.nxi])
        self.Qi_N     = diag(self.para_i[0,self.nxi:2*self.nxi])
        self.Ri_k     = diag(self.para_i[0,2*self.nxi:2*self.nxi+self.nui])
        self.pix_dis  = self.open_loop_penalty(self.para_i[0,-4],self.para_i[0,-2],self.a,ADMM_max)
        self.piu_dis  = self.open_loop_penalty(self.para_i[0,-3],self.para_i[0,-1],self.a,ADMM_max)
        # path cost
        self.resid_xi = self.xi - self.scxi + self.scxI/self.pix_dis
        self.resid_ui = self.ui - self.scui + self.scuI/self.piu_dis   
        self.Ji_k     = 1/2 * (track_error_i.T@self.Qi_k@track_error_i + ctrl_error_i.T@self.Ri_k@ctrl_error_i) + self.pix_dis/2*self.resid_xi.T@self.resid_xi + self.piu_dis/2*self.resid_ui.T@self.resid_ui 
        self.Ji_k_fn  = Function('Ji_k',[self.xi, self.ui, self.scxi, self.scxI, self.scui, self.scuI, self.ref_xi, self.ref_ui, self.para_i, self.a],[self.Ji_k],['xi0', 'ui0', 'scxi0', 'scxI0', 'scui0', 'scuI0', 'refxi0', 'refui0', 'parai0', 'a0'],['Ji_kf'])
        # terminal cost
        self.Ji_N     = 1/2 * track_error_i.T@self.Qi_N@track_error_i + self.pix_dis/2*self.resid_xi.T@self.resid_xi 
        self.Ji_N_fn  = Function('Ji_N',[self.xi, self.ref_xi, self.scxi, self.scxI, self.para_i, self.a],[self.Ji_N],['xi0', 'refxi0', 'scxi0', 'scxI0', 'parai0', 'a0'],['Ji_Nf'])
        # path cost of ADMM subproblem2
        self.Ji_P2_k  = self.pix_dis/2*self.resid_xi.T@self.resid_xi + self.piu_dis/2*self.resid_ui.T@self.resid_ui 
        self.Ji_P2_k_fn = Function('Ji_P2_k',[self.xi, self.scxi, self.scxI, self.ui, self.scui, self.scuI, self.para_i, self.a],[self.Ji_P2_k],['xi0', 'scxi0', 'scxI0', 'ui0', 'scui0', 'scuI0', 'parai0', 'a0'],['Ji_P2_kf'])
        # terminal cost of ADMM subproblem2
        self.Ji_P2_N  = self.pix_dis/2*self.resid_xi.T@self.resid_xi 
        self.Ji_P2_N_fn = Function('Ji_P2_N',[self.xi, self.scxi, self.scxI, self.para_i, self.a],[self.Ji_P2_N],['xi0', 'scxi0', 'scxI0', 'parai0', 'a0'],['Jl_P2_Nf'])

    def Load_derivatives_DDP_ADMM(self): # TODO: Type 5
        # alpha = 1
        self.Vxl      = SX.sym('Vxl',self.nxl)
        self.Vxlxl    = SX.sym('Vxlxl',self.nxl,self.nxl)
        # gradients of the system dynamics, the cost function, and the Q value function
        self.Fxl      = jacobian(self.model_l,self.xl)
        self.Fxl_fn   = Function('Fxl',[self.xl,self.ul],[self.Fxl],['xl0','ul0'],['Fxl_f'])
        self.Ful      = jacobian(self.model_l,self.ul)
        self.Ful_fn   = Function('Ful',[self.xl,self.ul],[self.Ful],['xl0','ul0'],['Ful_f'])
        self.lxl      = jacobian(self.Jl_k,self.xl)
        self.lxlN     = jacobian(self.Jl_N,self.xl)
        self.lxlN_fn  = Function('lxlN',[self.xl,self.ref_xl,self.scxl,self.scxL,self.para_l,self.a],[self.lxlN],['xl0', 'refxl0', 'scxl0', 'scxL0', 'paral0', 'a0'],['lxlN_f'])
        self.lul      = jacobian(self.Jl_k,self.ul)
        self.Qxl      = self.lxl.T + self.Fxl.T@self.Vxl
        self.Qxl_fn   = Function('Qxl',[self.xl,self.ul,self.Vxl,self.ref_xl,self.ref_ul,self.scxl,self.scxL,self.scul,self.scuL,self.para_l,self.a],[self.Qxl],['xl0','ul0','Vxl0','refxl0','reful0','scxl0','scxL0','scul0','scuL0','paral0','a0'],['Qxl_f'])
        self.Qul      = self.lul.T + self.Ful.T@self.Vxl
        self.Qul_fn   = Function('Qul',[self.xl,self.ul,self.Vxl,self.ref_xl,self.ref_ul,self.scxl,self.scxL,self.scul,self.scuL,self.para_l,self.a],[self.Qul],['xl0','ul0','Vxl0','refxl0','reful0','scxl0','scxL0','scul0','scuL0','paral0','a0'],['Qul_f'])
        # hessians of the system dynamics, the cost function, and the Q value function
        self.FxlVxl   = self.Fxl.T@self.Vxl
        self.dFxlVxldxl= jacobian(self.FxlVxl,self.xl) # the hessian of the system dynamics may cause heavy computational burden
        self.dFxlVxldul= jacobian(self.FxlVxl,self.ul)
        self.FulVxl   = self.Ful.T@self.Vxl
        self.dFulVxldul= jacobian(self.FulVxl,self.ul)
        self.lxlxl    = jacobian(self.lxl,self.xl)
        self.lxlxlN   = jacobian(self.lxlN,self.xl)
        self.lxlxlN_fn= Function('lxlxlN',[self.para_l,self.a],[self.lxlxlN],['paral0','a0'],['lxlxlN_f'])
        self.lxlul    = jacobian(self.lxl,self.ul)
        self.lulul    = jacobian(self.lul,self.ul)
        self.Qxlxl_bar    = self.lxlxl #+ alpha*self.dFxlVxldxl  # removing this model hessian can enhance the DDP stability for a larger time step!!!! The removal can also accelerate the DDP computation significantly!
        self.Qxlxl_bar_fn = Function('Qxlxl_bar',[self.xl,self.ul,self.Vxl,self.ref_xl,self.ref_ul,self.scxl,self.scxL,self.scul,self.scuL,self.para_l,self.a],[self.Qxlxl_bar],['xl0','ul0','Vxl0','refxl0','reful0','scxl0','scxL0','scul0','scuL0','paral0','a0'],['Qxlxl_bar_f'])
        self.Qxlxl_hat    = self.Fxl.T@self.Vxlxl@self.Fxl
        self.Qxlxl_hat_fn = Function('Qxlxl_hat',[self.xl,self.ul,self.Vxlxl],[self.Qxlxl_hat],['xl0','ul0','Vxlxl0'],['Qxlxl_hat_f'])
        self.Qxlul_bar    = self.lxlul #+ alpha*self.dFxlVxldul  # including the model hessian entails a very small time step size (e.g., 0.01s)
        self.Qxlul_bar_fn = Function('Qxlul_bar',[self.xl,self.ul,self.Vxl,self.ref_xl,self.ref_ul,self.scxl,self.scxL,self.scul,self.scuL,self.para_l,self.a],[self.Qxlul_bar],['xl0','ul0','Vxl0','refxl0','reful0','scxl0','scxL0','scul0','scuL0','paral0','a0'],['Qxlul_bar_f'])
        self.Qxlul_hat    = self.Fxl.T@self.Vxlxl@self.Ful
        self.Qxlul_hat_fn = Function('Qxlul_hat',[self.xl,self.ul,self.Vxlxl],[self.Qxlul_hat],['xl0','ul0','Vxlxl0'],['Qxlul_hat_f'])
        self.Qulul_bar    = self.lulul #+ alpha*self.dFulVxldul
        self.Qulul_bar_fn = Function('Qulul_bar',[self.xl,self.ul,self.Vxl,self.ref_xl,self.ref_ul,self.scxl,self.scxL,self.scul,self.scuL,self.para_l,self.a],[self.Qulul_bar],['xl0','ul0','Vxl0','refxl0','reful0','scxl0','scxL0','scul0','scuL0','paral0','a0'],['Qulul_bar_f'])
        self.Qulul_hat    = self.Ful.T@self.Vxlxl@self.Ful
        self.Qulul_hat_fn = Function('Qulul_hat',[self.xl,self.ul,self.Vxlxl],[self.Qulul_hat],['xl0','ul0','Vxlxl0'],['Qulul_hat_f'])
        # hessians w.r.t. the hyperparameters
        self.lxlp     = jacobian(self.lxl,self.P_auto)
        self.lxlp_fn  = Function('lxlp',[self.xl,self.ul,self.ref_xl,self.ref_ul,self.scxl,self.scxL,self.scul,self.scuL,self.para_l,self.a],[self.lxlp],['xl0','ul0','refxl0','reful0','scxl0','scxL0','scul0','scuL0','paral0','a0'],['lxlp_f'])
        self.lulp     = jacobian(self.lul,self.P_auto)
        self.lulp_fn  = Function('lulp',[self.xl,self.ul,self.ref_xl,self.ref_ul,self.scxl,self.scxL,self.scul,self.scuL,self.para_l,self.a],[self.lulp],['xl0','ul0','refxl0','reful0','scxl0','scxL0','scul0','scuL0','paral0','a0'],['lulp_f'])
        self.lxlNp    = jacobian(self.lxlN,self.P_auto)
        self.lxlNp_fn = Function('lxlNp',[self.xl,self.ref_xl,self.scxl,self.scxL,self.para_l,self.a],[self.lxlNp],['xl0', 'refxl0', 'scxl0', 'scxL0', 'paral0','a0'],['lxlNp_f'])


    
    def Cable_derivatives_DDP_ADMM(self): # TODO: Type 5
        self.Vxi      = SX.sym('Vxi',self.nxi)
        self.Vxixi    = SX.sym('Vxixi',self.nxi,self.nxi)
        # gradients of the system dynamics, the cost function, and the Q value function
        self.Fxi      = jacobian(self.model_i,self.xi)
        self.Fxi_fn   = Function('Fxi',[self.xi,self.ui],[self.Fxi],['xi0','ui0'],['Fxi_f'])
        self.Fui      = jacobian(self.model_i,self.ui)
        self.Fui_fn   = Function('Fui',[self.xi,self.ui],[self.Fui],['xi0','ui0'],['Fui_f'])
        self.lxi      = jacobian(self.Ji_k,self.xi)
        self.lxiN     = jacobian(self.Ji_N,self.xi)
        self.lxiN_fn  = Function('lxiN',[self.xi,self.ref_xi,self.scxi,self.scxI,self.para_i,self.a],[self.lxiN],['xi0', 'refxi0', 'scxi0', 'scxI0', 'parai0','a0'],['lxiN_f'])
        self.lui      = jacobian(self.Ji_k,self.ui)
        self.Qxi      = self.lxi.T + self.Fxi.T@self.Vxi
        self.Qxi_fn   = Function('Qxi',[self.xi,self.ui,self.Vxi,self.ref_xi,self.ref_ui,self.scxi,self.scxI,self.scui,self.scuI,self.para_i,self.a],[self.Qxi],['xi0','ui0','Vxi0','refxi0','refui0','scxi0','scxI0','scui0','scuI0','parai0','a0'],['Qxi_f'])
        self.Qui      = self.lui.T + self.Fui.T@self.Vxi
        self.Qui_fn   = Function('Qui',[self.xi,self.ui,self.Vxi,self.ref_xi,self.ref_ui,self.scxi,self.scxI,self.scui,self.scuI,self.para_i,self.a],[self.Qui],['xi0','ui0','Vxi0','refxi0','refui0','scxi0','scxI0','scui0','scuI0','parai0','a0'],['Qui_f'])
        # hessians of the system dynamics, the cost function, and the Q value function
        self.FxiVxi   = self.Fxi.T@self.Vxi
        self.dFxiVxidxi= jacobian(self.FxiVxi,self.xi) # the hessian of the system dynamics may cause heavy computational burden
        self.dFxiVxidui= jacobian(self.FxiVxi,self.ui)
        self.FuiVxi   = self.Fui.T@self.Vxi
        self.dFuiVxidui= jacobian(self.FuiVxi,self.ui)
        self.lxixi    = jacobian(self.lxi,self.xi) # already includes pi
        self.lxixiN   = jacobian(self.lxiN,self.xi)
        self.lxixiN_fn= Function('lxixiN',[self.para_i,self.a],[self.lxixiN],['parai0','a0'],['lxixiN_f'])
        self.lxiui    = jacobian(self.lxi,self.ui)
        self.luiui    = jacobian(self.lui,self.ui)
        self.Qxixi_bar    = self.lxixi #+ alpha*self.dFxiVxidxi  # removing this model hessian can enhance the DDP stability for a larger time step!!!! The removal can also accelerate the DDP computation significantly!
        self.Qxixi_bar_fn = Function('Qxixi_bar',[self.xi,self.ui,self.Vxi,self.ref_xi,self.ref_ui,self.scxi,self.scxI,self.scui,self.scuI,self.para_i,self.a],[self.Qxixi_bar],['xi0','ui0','Vxi0','refxi0','refui0','scxi0','scxI0','scui0','scuI0','parai0','a0'],['Qxixi_bar_f'])
        self.Qxixi_hat    = self.Fxi.T@self.Vxixi@self.Fxi
        self.Qxixi_hat_fn = Function('Qxixi_hat',[self.xi,self.ui,self.Vxixi],[self.Qxixi_hat],['xi0','ui0','Vxixi0'],['Qxixi_hat_f'])
        self.Qxiui_bar    = self.lxiui #+ alpha*self.dFxiVxidui  # including the model hessian entails a very small time step size (e.g., 0.01s)
        self.Qxiui_bar_fn = Function('Qxiui_bar',[self.xi,self.ui,self.Vxi,self.ref_xi,self.ref_ui,self.scxi,self.scxI,self.scui,self.scuI,self.para_i,self.a],[self.Qxiui_bar],['xi0','ui0','Vxi0','refxi0','refui0','scxi0','scxI0','scui0','scuI0','parai0','a0'],['Qxiui_bar_f'])
        self.Qxiui_hat    = self.Fxi.T@self.Vxixi@self.Fui
        self.Qxiui_hat_fn = Function('Qxiui_hat',[self.xi,self.ui,self.Vxixi],[self.Qxiui_hat],['xi0','ui0','Vxixi0'],['Qxiui_hat_f'])
        self.Quiui_bar    = self.luiui #+ alpha*self.dFuiVxidui
        self.Quiui_bar_fn = Function('Quiui_bar',[self.xi,self.ui,self.Vxi,self.ref_xi,self.ref_ui,self.scxi,self.scxI,self.scui,self.scuI,self.para_i,self.a],[self.Quiui_bar],['xi0','ui0','Vxi0','refxi0','refui0','scxi0','scxI0','scui0','scuI0','parai0','a0'],['Quiui_bar_f'])
        self.Quiui_hat    = self.Fui.T@self.Vxixi@self.Fui
        self.Quiui_hat_fn = Function('Quiui_hat',[self.xi,self.ui,self.Vxixi],[self.Quiui_hat],['xi0','ui0','Vxixi0'],['Quiui_hat_f'])
        # hessians w.r.t. the hyperparameters
        self.lxip     = jacobian(self.lxi,self.P_auto)
        self.lxip_fn  = Function('lxip',[self.xi,self.ui,self.ref_xi,self.ref_ui,self.scxi,self.scxI,self.scui,self.scuI,self.para_i,self.a],[self.lxip],['xi0','ui0','refxi0','refui0','scxi0','scxI0','scui0','scuI0','parai0','a0'],['lxip_f'])
        self.luip     = jacobian(self.lui,self.P_auto)
        self.luip_fn  = Function('luip',[self.xi,self.ui,self.ref_xi,self.ref_ui,self.scxi,self.scxI,self.scui,self.scuI,self.para_i,self.a],[self.luip],['xi0','ui0','refxi0','refui0','scxi0','scxI0','scui0','scuI0','parai0','a0'],['luip_f'])
        self.lxiNp    = jacobian(self.lxiN,self.P_auto)
        self.lxiNp_fn = Function('lxiNp',[self.xi,self.ref_xi,self.scxi,self.scxI,self.para_i,self.a],[self.lxiNp],['xi0', 'refxi0', 'scxi0', 'scxI0', 'parai0','a0'],['lxiNp_f'])


    
    
    def Get_AuxSys_DDP_Load(self,opt_sol,Ref_xl,Ref_ul,scxl,scul,scxL,scuL,weight1,i_admm):# TODO: Type 5
        xl_opt   = opt_sol['xl_traj']
        ul_opt   = opt_sol['ul_traj']
        LxlNp    = self.lxlNp_fn(xl0=xl_opt[-1,:],refxl0=Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl],scxl0=scxl[self.N*self.nxl:(self.N+1)*self.nxl],scxL0=scxL[self.N*self.nxl:(self.N+1)*self.nxl],paral0=weight1,a0=i_admm)['lxlNp_f'].full()
        LxlxlN   = self.lxlxlN_fn(paral0=weight1,a0=i_admm)['lxlxlN_f'].full()
        Lxlp     = self.N*[np.zeros((self.nxl,self.n_Pauto))]
        Lulp     = self.N*[np.zeros((self.nul,self.n_Pauto))]
        for k in range(self.N):
            Lxlp[k] = self.lxlp_fn(xl0=xl_opt[k,:],ul0=ul_opt[k,:],refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],
                                scxl0=scxl[k,:],scxL0=scxL[k,:],scul0=scul[k,:],scuL0=scuL[k,:],paral0=weight1,a0=i_admm)['lxlp_f'].full()
            Lulp[k] = self.lulp_fn(xl0=xl_opt[k,:],ul0=ul_opt[k,:],refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],
                                scxl0=scxl[k,:],scxL0=scxL[k,:],scul0=scul[k,:],scuL0=scuL[k,:],paral0=weight1,a0=i_admm)['lulp_f'].full()
        
        auxSysl = { "HxxN":LxlxlN,
                    "HxNp":LxlNp,
                    "Hxp":Lxlp,
                    "Hup":Lulp
                    }
        
        return auxSysl
    

    def Get_AuxSys_DDP_Cable(self,opt_sol,Ref_xi,Ref_ui,scxi,scui,scxI,scuI,weight2,i_admm):# TODO: Type 5
        xi_opt   = opt_sol['xi_traj']
        ui_opt   = opt_sol['ui_traj']
        LxiNp    = self.lxiNp_fn(xi0=xi_opt[-1,:],refxi0=Ref_xi[self.N*self.nxi:(self.N+1)*self.nxi],scxi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],scxI0=scxI[self.N*self.nxi:(self.N+1)*self.nxi],parai0=weight2,a0=i_admm)['lxiNp_f'].full()
        LxixiN   = self.lxixiN_fn(parai0=weight2,a0=i_admm)['lxixiN_f'].full()
        Lxip     = self.N*[np.zeros((self.nxi,self.n_Pauto))]
        Luip     = self.N*[np.zeros((self.nui,self.n_Pauto))]
        for k in range(self.N):
            Lxip[k] = self.lxip_fn(xi0=xi_opt[k,:],ui0=ui_opt[k,:],refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui[k*self.nui:(k+1)*self.nui],
                                scxi0=scxi[k,:],scxI0=scxI[k,:],scui0=scui[k,:],scuI0=scuI[k,:],parai0=weight2,a0=i_admm)['lxip_f'].full()
            Luip[k] = self.luip_fn(xi0=xi_opt[k,:],ui0=ui_opt[k,:],refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui[k*self.nui:(k+1)*self.nui],
                                scxi0=scxi[k,:],scxI0=scxI[k,:],scui0=scui[k,:],scuI0=scuI[k,:],parai0=weight2,a0=i_admm)['luip_f'].full()
        
        auxSysi = { "HxxN":LxixiN,
                    "HxNp":LxiNp,
                    "Hxp":Lxip,
                    "Hup":Luip
                    }
        
        return auxSysi
    

    def symmetry(self,A):
        return 0.5*(A + A.T)

    def chol_solve(self,L, B):
        # Solve (L L^T) X = B
        Y = LA.solve(L, B)
        return LA.solve(L.T, Y)

    def try_cholesky(self,A, jitter0=0.0, max_tries=5):
        """Try Cholesky with growing jitter on the diagonal."""
        jitter = jitter0
        for _ in range(max_tries):
            try:
                return LA.cholesky(A + jitter*np.eye(A.shape[0])), jitter
            except LA.LinAlgError:
                jitter = max(1e-12, 10*(jitter if jitter>0 else 1e-12))
        raise LA.LinAlgError("Cholesky failed even with jitter")

   
    def DDP_Load_ADMM_Subp1(self,xl_0,Ref_xl,Ref_ul,weight1,scxl,scul,scxL,scuL,max_iter,e_tol,i_admm):# TODO Type 4
        reg        = 1e-6 # Regularization term
        reg_max    = 1    # cap to avoid runaway
        reg_up     = 10.0 # how much to bump when ill-conditioned
        alpha_init = 1 # Initial alpha for line search
        alpha_min  = 1e-2  # Minimum allowable alpha
        alpha_factor = 0.5 # 
        max_line_search_steps = 5
        iteration = 1
        ratio = 10
        X_nominal = np.zeros((self.nxl,self.N+1))
        U_nominal = np.zeros((self.nul,self.N))
        X_nominal[:,0:1] = np.reshape(xl_0,(self.nxl,1))
        
        # Initial trajectory and initial cost 
        cost_prev = 0
        # if i_admm ==0:
        for k in range(self.N):
            u_k    = np.reshape(Ref_ul[k*self.nul:(k+1)*self.nul],(self.nul,1))
            # X_nominal[:,k:k+1] = np.reshape(Ref_xl[k*self.nxl:(k+1)*self.nxl],(self.nxl,1))
            X_nominal[:,k:k+1] = self.model_l_fn(xl0=X_nominal[:,k],ul0=u_k)['mdynlf'].full() # start from a bad state
            U_nominal[:,k:k+1]   = u_k
            cost_prev     += self.Jl_kfn(xl0=X_nominal[:,k],ul0=u_k,scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],
                                        scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],
                                        refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Jl_kf'].full()
        cost_prev += self.Jl_Nfn(xl0=X_nominal[:,-1],refxl0=Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl],scxl0=scxl[self.N*self.nxl:(self.N+1)*self.nxl],
                                 scxL0=scxL[self.N*self.nxl:(self.N+1)*self.nxl],paral0=weight1,a0=i_admm)['Jl_Nf'].full()
        # else:
        #     for k in range(self.N):
        #         X_nominal[:,k+1:k+2] = np.reshape(scxl[k*self.nxl:(k+1)*self.nxl],(self.nxl,1))
        #         U_nominal[:,k:k+1]   = np.reshape(scul[k*self.nul:(k+1)*self.nul],(self.nul,1))
        #         cost_prev     += self.Jl_kfn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],
        #                                 scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],
        #                                 refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Jl_kf'].full()
        #     cost_prev += self.Jl_Nfn(xl0=X_nominal[:,-1],refxl0=Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl],scxl0=scxl[self.N*self.nxl:(self.N+1)*self.nxl],
        #                          scxL0=scxL[self.N*self.nxl:(self.N+1)*self.nxl],paral0=weight1,a0=i_admm)['Jl_Nf'].full()    

        Qxx_bar     = self.N*[np.zeros((self.nxl,self.nxl))]
        Qxu_bar     = self.N*[np.zeros((self.nxl,self.nul))]
        Quu_bar     = self.N*[np.zeros((self.nul,self.nul))]
        Qxu         = self.N*[np.zeros((self.nxl,self.nul))]
        Quuinv      = self.N*[np.zeros((self.nul,self.nul))]
        Fx          = self.N*[np.zeros((self.nxl,self.nxl))]
        Fu          = self.N*[np.zeros((self.nxl,self.nul))]
        Vx          = (self.N+1)*[np.zeros((self.nxl,1))]
        Vxx         = (self.N+1)*[np.zeros((self.nxl,self.nxl))]
        K_fb        = self.N*[np.zeros((self.nul,self.nxl))] # feedback
        k_ff        = self.N*[np.zeros((self.nul,1))] # feedforward
        Qu_2        = 1000
        I_u         = np.identity(self.nul)
        while Qu_2>e_tol and iteration<=max_iter:
            Vx[self.N] = self.lxlN_fn(xl0=X_nominal[:,self.N],
                                      refxl0=Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl],
                                      scxl0=scxl[self.N*self.nxl:(self.N+1)*self.nxl],
                                      scxL0=scxL[self.N*self.nxl:(self.N+1)*self.nxl],
                                      paral0=weight1,
                                      a0=i_admm)['lxlN_f'].full()
            Vxx[self.N]= self.lxlxlN_fn(paral0=weight1,a0=i_admm)['lxlxlN_f'].full()
            # backward pass
            Qu_2    = 0
            chol_failed = False
            for k in reversed(range(self.N)): # N-1, N-2,...,0
                Qx_k  = self.Qxl_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],Vxl0=Vx[k+1],refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],
                                    scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Qxl_f'].full()
                Qu_k  = self.Qul_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],Vxl0=Vx[k+1],refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],
                                    scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Qul_f'].full()
                Qxx_bar_k = self.Qxlxl_bar_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],Vxl0=Vx[k+1],refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],
                                    scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Qxlxl_bar_f'].full()
                Qxx_hat_k = self.Qxlxl_hat_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],Vxlxl0=Vxx[k+1])['Qxlxl_hat_f'].full()
                Qxx_k     = Qxx_bar_k + Qxx_hat_k
                Qxu_bar_k = self.Qxlul_bar_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],Vxl0=Vx[k+1],refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],
                                    scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Qxlul_bar_f'].full()
                Qxu_hat_k = self.Qxlul_hat_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],Vxlxl0=Vxx[k+1])['Qxlul_hat_f'].full()
                Qxu_k     = Qxu_bar_k + Qxu_hat_k
                Quu_bar_k = self.Qulul_bar_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],Vxl0=Vx[k+1],refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],
                                    scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Qulul_bar_f'].full()
                Quu_hat_k = self.Qulul_hat_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],Vxlxl0=Vxx[k+1])['Qulul_hat_f'].full()
                Quu_k     = Quu_bar_k + Quu_hat_k 
                Quu_reg_k = Quu_k + reg*I_u
                try:
                    L, _jitter = self.try_cholesky(Quu_reg_k, jitter0=0.0)
                except LA.LinAlgError:
                    chol_failed = True
                    break
                Quu_inv      = self.chol_solve(L, I_u) # only for computing the gradients
                K_fb[k]      = self.chol_solve(L, -Qxu_k.T)
                k_ff[k]      = self.chol_solve(L, -Qu_k)
                Vx[k]        = Qx_k + Qxu_k @ k_ff[k]
                Vxx[k]       = self.symmetry(Qxx_k + Qxu_k @ K_fb[k])
                Fx[k]        = self.Fxl_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k])['Fxl_f'].full()
                Fu[k]        = self.Ful_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k])['Ful_f'].full()
                Qxx_bar[k]   = Qxx_bar_k
                Qxu_bar[k]   = Qxu_bar_k
                Quu_bar[k]   = Quu_bar_k
                Quuinv[k]    = Quu_inv
                Qxu[k]       = Qxu_k
                Qu_2         = max(Qu_2, (LA.norm(Qu_k)))
            # if backward failed, bump reg and retry (do NOT advance iteration)
            if chol_failed:
                reg = min(reg_max, reg * reg_up)
                # print(f'backward cholesky failed → increasing reg to {reg:.3e}')
                continue
            # forward pass with adaptive alpha (line search), adaptive alpha makes the DDP more stable!
            alpha = alpha_init
            accepted = False
            for _ in range(max_line_search_steps):
                X_new = np.zeros((self.nxl,self.N+1))
                U_new = np.zeros((self.nul,self.N))
                X_new[:,0:1] = np.reshape(xl_0,(self.nxl,1))
                cost_new = 0
                for k in range(self.N):
                    delta_x = np.reshape(X_new[:,k] - X_nominal[:,k],(self.nxl,1))
                    u_k     = np.reshape(U_nominal[:,k],(self.nul,1)) + K_fb[k]@delta_x + alpha*k_ff[k]
                    u_k     = np.reshape(u_k,(self.nul,1))
                    X_new[:,k+1:k+2]  = self.model_l_fn(xl0=np.reshape(X_new[:,k],(self.nxl,1)),ul0=u_k)['mdynlf'].full()
                    U_new[:,k:k+1]    = u_k
                    cost_new   += self.Jl_kfn(xl0=X_new[:,k],ul0=u_k,scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],
                                              scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],
                                              refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Jl_kf'].full()
                cost_new   += self.Jl_Nfn(xl0=X_new[:,-1],refxl0=Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl],scxl0=scxl[self.N*self.nxl:(self.N+1)*self.nxl],
                                          scxL0=scxL[self.N*self.nxl:(self.N+1)*self.nxl], paral0=weight1,a0=i_admm)['Jl_Nf'].full()
                # Check if the cost decreased
                if cost_new < cost_prev:
                    # update the trajectories
                    X_nominal = X_new
                    U_nominal = U_new
                    accepted  = True
                    break
                alpha = np.clip(alpha*alpha_factor,alpha_min,alpha_init)  # Reduce alpha if cost did not improve

            # if nothing accepted, nudge reg up to help next backward factorization
            if not accepted:
                reg = min(reg_max, reg * reg_up)

            ratio = np.abs(cost_new-cost_prev)/np.abs(cost_prev)
            print('iteration:',iteration,'ratio=',ratio,'Qu_2=',Qu_2)

            cost_prev = cost_new
            iteration += 1
        
        opt_sol={"xl_traj":X_nominal.T,
                 "ul_traj":U_nominal.T,
                 "Vxx":Vxx,
                 "Vx":Vx,
                 "K_FB":K_fb,
                 "Hxx":Qxx_bar,
                 "Qxu":Qxu,
                 "Hxu":Qxu_bar,
                 "Huu":Quu_bar,
                 "Quu_inv":Quuinv,
                 "Fx":Fx,
                 "Fu":Fu}
        return opt_sol
    

    def DDP_Cable_ADMM_Subp1(self,xi_0,Ref_xi,Ref_ui,weight2,scxi,scui,scxI,scuI,max_iter,e_tol,i_admm):# TODO Type 4
        reg          = 1e-6 # Regularization term
        reg_max      = 1    # cap to avoid runaway
        reg_up       = 10.0 # how much to bump when ill-conditioned
        alpha_init   = 1 # Initial alpha for line search
        alpha_min    = 1e-2  # Minimum allowable alpha
        alpha_factor = 0.5 # 
        max_line_search_steps = 5
        iteration = 1
        ratio = 10
        X_nominal = np.zeros((self.nxi,self.N+1))
        U_nominal = np.zeros((self.nui,self.N))
        X_nominal[:,0:1] = np.reshape(xi_0,(self.nxi,1))
        
        # Initial trajectory and initial cost 
        cost_prev = 0
        # if i_admm ==0:
        for k in range(self.N):
            u_k    = np.reshape(Ref_ui,(self.nui,1))
                # X_nominal[:,k:k+1] = np.reshape(Ref_xi[k*self.nxi:(k+1)*self.nxi],(self.nxi,1))
            X_nominal[:,k:k+1] = self.model_i_fn(xi0=X_nominal[:,k],ui0=u_k)['mdynif'].full() # start from a bad state
            U_nominal[:,k:k+1]   = u_k
            cost_prev     += self.Ji_k_fn(xi0=X_nominal[:,k],ui0=u_k,scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],
                                        scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],
                                        refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,parai0=weight2,a0=i_admm)['Ji_kf'].full()
        cost_prev += self.Ji_N_fn(xi0=X_nominal[:,-1],refxi0=Ref_xi[self.N*self.nxi:(self.N+1)*self.nxi],scxi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],
                                 scxI0=scxI[self.N*self.nxi:(self.N+1)*self.nxi],parai0=weight2,a0=i_admm)['Ji_Nf'].full()
        # else:
        #     for k in range(self.N):
        #         X_nominal[:,k:k+1] = np.reshape(scxi[k*self.nxi:(k+1)*self.nxi],(self.nxi,1))
        #         U_nominal[:,k:k+1]   = np.reshape(scui[k*self.nui:(k+1)*self.nui],(self.nui,1))
        #         cost_prev     += self.Ji_k_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],
        #                                 scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],
        #                                 refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,parai0=weight2,a0=i_admm)['Ji_kf'].full()
        #     cost_prev += self.Ji_N_fn(xi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],refxi0=Ref_xi[self.N*self.nxi:(self.N+1)*self.nxi],scxi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],
        #                          scxI0=scxI[self.N*self.nxi:(self.N+1)*self.nxi],parai0=weight2,a0=i_admm)['Ji_Nf'].full()

        Qxx_bar     = self.N*[np.zeros((self.nxi,self.nxi))]
        Qxu_bar     = self.N*[np.zeros((self.nxi,self.nui))]
        Quu_bar     = self.N*[np.zeros((self.nui,self.nui))]
        Qxu         = self.N*[np.zeros((self.nxi,self.nui))]
        Quuinv      = self.N*[np.zeros((self.nui,self.nui))]
        Fx          = self.N*[np.zeros((self.nxi,self.nxi))]
        Fu          = self.N*[np.zeros((self.nxi,self.nui))]
        Vx          = (self.N+1)*[np.zeros((self.nxi,1))]
        Vxx         = (self.N+1)*[np.zeros((self.nxi,self.nxi))]
        K_fb        = self.N*[np.zeros((self.nui,self.nxi))] # feedback
        k_ff        = self.N*[np.zeros((self.nui,1))] # feedforward
        Qu_2        = 1000
        I_u         = np.identity(self.nui)

        while Qu_2>e_tol and iteration<=max_iter:
            Vx[self.N] = self.lxiN_fn(xi0=X_nominal[:,self.N],
                                      refxi0=Ref_xi[self.N*self.nxi:(self.N+1)*self.nxi],
                                      scxi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],
                                      scxI0=scxI[self.N*self.nxi:(self.N+1)*self.nxi],
                                      parai0=weight2,
                                      a0=i_admm)['lxiN_f'].full()
            Vxx[self.N]= self.lxixiN_fn(parai0=weight2,a0=i_admm)['lxixiN_f'].full() 
            # backward pass
            Qu_2    = 0
            chol_failed = False
            for k in reversed(range(self.N)): # N-1, N-2,...,0
                Qx_k  = self.Qxi_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],Vxi0=Vx[k+1],refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,
                                    scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],parai0=weight2,a0=i_admm)['Qxi_f'].full()
                Qu_k  = self.Qui_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],Vxi0=Vx[k+1],refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,
                                    scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],parai0=weight2,a0=i_admm)['Qui_f'].full()
                Qxx_bar_k = self.Qxixi_bar_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],Vxi0=Vx[k+1],refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,
                                    scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],parai0=weight2,a0=i_admm)['Qxixi_bar_f'].full()
                Qxx_hat_k = self.Qxixi_hat_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],Vxixi0=Vxx[k+1])['Qxixi_hat_f'].full()
                Qxx_k     = Qxx_bar_k + Qxx_hat_k
                Qxu_bar_k = self.Qxiui_bar_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],Vxi0=Vx[k+1],refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,
                                    scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],parai0=weight2,a0=i_admm)['Qxiui_bar_f'].full()
                Qxu_hat_k = self.Qxiui_hat_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],Vxixi0=Vxx[k+1])['Qxiui_hat_f'].full()
                Qxu_k     = Qxu_bar_k + Qxu_hat_k
                Quu_bar_k = self.Quiui_bar_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],Vxi0=Vx[k+1],refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,
                                    scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],parai0=weight2,a0=i_admm)['Quiui_bar_f'].full()
                Quu_hat_k = self.Quiui_hat_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],Vxixi0=Vxx[k+1])['Quiui_hat_f'].full()
                Quu_k     = Quu_bar_k + Quu_hat_k
                Quu_reg_k = Quu_k + reg*I_u
                try:
                    L, _jitter = self.try_cholesky(Quu_reg_k, jitter0=0.0)
                except LA.LinAlgError:
                    chol_failed = True
                    break
                Quu_inv      = self.chol_solve(L, I_u) # only for computing the gradients
                K_fb[k]      = self.chol_solve(L, -Qxu_k.T)
                k_ff[k]      = self.chol_solve(L, -Qu_k)
                Vx[k]        = Qx_k + Qxu_k @ k_ff[k]
                Vxx[k]       = self.symmetry(Qxx_k + Qxu_k @ K_fb[k])
                Fx[k]    = self.Fxi_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k])['Fxi_f'].full()
                Fu[k]    = self.Fui_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k])['Fui_f'].full()
                Qxx_bar[k]   = Qxx_bar_k
                Qxu_bar[k]   = Qxu_bar_k
                Quu_bar[k]   = Quu_bar_k
                Quuinv[k]    = Quu_inv
                Qxu[k]       = Qxu_k
                Qu_2         = max(Qu_2, (LA.norm(Qu_k)))
            # if backward failed, bump reg and retry (do NOT advance iteration)
            if chol_failed:
                reg = min(reg_max, reg * reg_up)
                # print(f'backward cholesky failed → increasing reg to {reg:.3e}')
                continue
            # forward pass with adaptive alpha (line search), adaptive alpha makes the DDP more stable!
            alpha = alpha_init
            accepted = False
            for _ in range(max_line_search_steps):
                X_new = np.zeros((self.nxi,self.N+1))
                U_new = np.zeros((self.nui,self.N))
                X_new[:,0:1] = np.reshape(xi_0,(self.nxi,1))
                cost_new = 0
                for k in range(self.N):
                    delta_x = np.reshape(X_new[:,k] - X_nominal[:,k],(self.nxi,1))
                    u_k     = np.reshape(U_nominal[:,k],(self.nui,1)) + K_fb[k]@delta_x + alpha*k_ff[k]
                    u_k     = np.reshape(u_k,(self.nui,1))
                    X_new[:,k+1:k+2]  = self.model_i_fn(xi0=np.reshape(X_new[:,k],(self.nxi,1)),ui0=u_k)['mdynif'].full()
                    U_new[:,k:k+1]    = u_k
                    cost_new   += self.Ji_k_fn(xi0=X_new[:,k],ui0=u_k,scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],
                                              scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],
                                              refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,parai0=weight2,a0=i_admm)['Ji_kf'].full()
                cost_new   += self.Ji_N_fn(xi0=X_new[:,-1],refxi0=Ref_xi[self.N*self.nxi:(self.N+1)*self.nxi],scxi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],
                                          scxI0=scxI[self.N*self.nxi:(self.N+1)*self.nxi], parai0=weight2,a0=i_admm)['Ji_Nf'].full()
                # Check if the cost decreased
                if cost_new < cost_prev:
                    # update the trajectories
                    X_nominal = X_new
                    U_nominal = U_new
                    accepted  = True
                    break
                alpha = np.clip(alpha*alpha_factor,alpha_min,alpha_init)  # Reduce alpha if cost did not improve

            # if nothing accepted, nudge reg up to help next backward factorization
            if not accepted:
                reg = min(reg_max, reg * reg_up)

            ratio = np.abs(cost_new-cost_prev)/np.abs(cost_prev)
            print('iteration:',iteration,'ratio=',ratio,'Qu_2=',Qu_2)

            cost_prev = cost_new
            iteration += 1
        
        opt_sol={"xi_traj":X_nominal.T,
                 "ui_traj":U_nominal.T,
                 "Vxx":Vxx,
                 "Vx":Vx,
                 "K_FB":K_fb,
                 "Hxx":Qxx_bar,
                 "Qxu":Qxu,
                 "Hxu":Qxu_bar,
                 "Huu":Quu_bar,
                 "Quu_inv":Quuinv,
                 "Fx":Fx,
                 "Fu":Fu}
        return opt_sol


    def MPC_Cable_DDP_Planning_SubP1(self,ParaC):#TODO Type 4 原版多机函数， 已经由jax新版替代  # checked, correct, Apr.1 2025
        xc_traj      = [np.zeros((self.N+1,self.nxi)) for _ in range(int(self.nq))]
        uc_traj      = [np.zeros((self.N,self.nui)) for _ in range(int(self.nq))]
        OPt_sol_c    = []
        max_iter     = 10
        e_tol        = 1e-2
        for i in range(int(self.nq)):
            Parai    = ParaC[i]
            xi_fb    = Parai[0:self.nxi]
            Ref_xi   = Parai[self.nxi:self.nxi*(self.N+2)]
            Ref_ui   = Parai[self.nxi+self.nxi*(self.N+1):self.nxi+self.nxi*(self.N+1)+self.nui]
            # Solve the DDP
            n_scxi_start = self.nxi*(self.N+2)+self.nui
            scxi         = Parai[n_scxi_start:n_scxi_start+self.nxi*(self.N+1)]
            n_scxI_start = n_scxi_start + self.nxi*(self.N+1)
            scxI         = Parai[n_scxI_start:n_scxI_start+self.nxi*(self.N+1)]
            n_scui_start = n_scxI_start + self.nxi*(self.N+1)
            scui         = Parai[n_scui_start:n_scui_start+self.nui*self.N]
            n_scuI_start = n_scui_start + self.nui*self.N
            scuI         = Parai[n_scuI_start:n_scuI_start+self.nui*self.N]
            n_weig_start = n_scuI_start + self.nui*self.N
            weight2   = Parai[n_weig_start:n_weig_start+self.npi]
            i_admm    = Parai[-1]
            opt_sol_i = self.DDP_Cable_ADMM_Subp1(xi_fb,Ref_xi,Ref_ui,weight2,scxi,scui,scxI,scuI,max_iter,e_tol,i_admm)
            OPt_sol_c += [opt_sol_i]
            xc_traj[i] = opt_sol_i['xi_traj']
            uc_traj[i] = opt_sol_i['ui_traj']
        # output
        opt_solc = {"xc_traj":xc_traj,
                   "uc_traj":uc_traj
                   }
        
        return opt_solc, OPt_sol_c

    
    def SetConstriants(self, pob1, pob2):
        # dynamic coupling constraint at each step k
        pl_k     = self.scxl[0:3]
        vl_k     = self.scxl[3:6]
        ql_k     = self.scxl[6:10]
        wl_k     = self.scxl[10:self.nxl]
        Fl_k     = self.scul[0:3] #{I}
        Ml_k     = self.scul[3:6]
        tc_k     = SX.sym('tc_k',3*int(self.nq),1)
        Rl_k     = self.q_2_rotation(ql_k)
        ql_knorm = ql_k.T@ql_k
        self.ql_n     = 1/(2*self.p_bar)*(ql_knorm-1)**2
        self.ql_fn    = Function('norm_ql',[self.scxl],[ql_knorm],['scxl0'],['norm_qlf'])
        k           = 0
        self.fi     = [] # list that stores all the quadrotor thruster limit constraints
        self.Gi1    = [] # list that stores the obstacle-avoidance constraints of all the quadrotors for the 1st obstacle
        self.Gi2    = [] # list that stores the obstacle-avoidance constraints of all the quadrotors for the 2nd obstacle
        self.Gij    = [] # list that stores all the safe inter-robot inequality constraints
        self.Gio    = []
        self.Di     = []
        self.sumfi  = 0 # barrier functions of the quadrotor thrust limit
        self.gco    = 0 # barrier functions of the safe collision-avoidance constraints on quadrotors' planar positions
        self.G_lo   = 0
        self.gij    = 0 # barrier functions of the safe inter-robot constraints on quadrotors' planar positions
        self.Tcon   = 0 # barrier functions of the tension magnitude constraints
        self.Uicon  = 0 # barrier functions of the cable control inputs
        self.din    = 0 # barrier functions of the cable direction normalization 
        self.Ei_Pil = 0
        dis_two     = 2*self.rl*math.sin(self.alpha/2) # distance between two neighbour cable attachment points
        num_dis     = int(self.cl0/(dis_two)) # discretization number
        
        po1   = (pl_k[0:2]-pob1).T@(pl_k[0:2]-pob1) - ((self.ro)+self.rq/2)**2
        self.po1_fn= Function('pl1_admm',[self.scxl],[po1],['scxl0'],['po1f_admm'])
        self.G_lo += -self.p_bar * log(po1)
        po2   = (pl_k[0:2]-pob2).T@(pl_k[0:2]-pob2) - ((self.ro)+self.rq/2)**2
        self.po2_fn= Function('pl2_admm',[self.scxl],[po2],['scxl0'],['po2f_admm'])
        self.G_lo += -self.p_bar * log(po2)
        for kc in range(1,num_dis+1):
            for i in range(int(self.nq)):
                ri     = np.reshape(self.ra[:,i],(3,1))
                ei     = ri/norm_2(ri)
                xi_k   = self.scxc[i*self.nxi:(i+1)*self.nxi]
                ui_k   = self.scuc[i*self.nui:(i+1)*self.nui]
                di_k   = xi_k[0:3] # world frame
                pib_k  = ri + (kc/(num_dis))*self.cl0*Rl_k.T@di_k # body frame
                if kc == (num_dis):
                    wi_k   = xi_k[3:6]
                    dwi_k  = xi_k[6:9] # cable angular acceleration
                    ti_k   = xi_k[12]  # cable tension magnitude
                    self.Tcon += -self.p_bar * log(ti_k-self.t_min)
                    self.Tcon += -self.p_bar * log(self.t_max-ti_k)
                    for i_u in range(self.nui):
                        self.Uicon += -self.p_bar * log(self.ui_bound - ui_k[i_u])
                        self.Uicon += -self.p_bar * log(ui_k[i_u] + self.ui_bound)
                    pi_k = pl_k + Rl_k@ri + (kc/(num_dis))*self.cl0*di_k # ith quadrotor's position in {I}
                    diso1 = pi_k[0:2]-pob1
                    go1  = diso1.T@diso1 - ((self.rq + self.ro)+self.rq)**2 # safe constriant between the obstacle 1 and the ith quadrotor, which should be positive. 
                    go1_fn = Function('go1'+str(i),[self.scxl,self.scxc],[go1],['scxl0','scxc0'],['go1f'+str(i)])
                    self.gco += -self.p_bar * log(go1)
                    self.Gi1 += [go1_fn]
                    diso2 = pi_k[0:2]-pob2
                    go2  = diso2.T@diso2 - ((self.rq + self.ro)+self.rq)**2 # safe constriant between the obstacle 2 and the ith quadrotor, which should be positive
                    go2_fn = Function('go2'+str(i),[self.scxl,self.scxc],[go2],['scxl0','scxc0'],['go2f'+str(i)])
                    self.gco += -self.p_bar * log(go2)
                    self.Gi2 += [go2_fn]
                    dnorm = di_k.T@di_k
                    self.din += 1/(2*self.p_bar)*(dnorm-1)**2
                    d_fn  = Function('dn'+str(i),[self.scxc],[dnorm],['scxc0'],['dn'+str(i)])
                    self.Di += [d_fn]
                    # Thrust constraints
                    S_wl_k  = self.skew_sym(wl_k)
                    S_wi_k  = self.skew_sym(wi_k)
                    S_dwi_k = self.skew_sym(dwi_k)
                    al_k    = -self.g*self.ez + Fl_k/self.ml
                    awl_k   = LA.inv(self.Jl)@(Ml_k-S_wl_k@(self.Jl@wl_k))
                    S_awl_k = self.skew_sym(awl_k)
                    fi_k    = self.mq*(al_k+Rl_k@(S_wl_k@S_wl_k+S_awl_k)@ri+self.cl0*(S_dwi_k@di_k+S_wi_k@(S_wi_k@di_k))+self.g*self.ez) + di_k*ti_k
                    norm_fi = fi_k.T@fi_k
                    self.sumfi += -self.p_bar * log(self.fmax**2-norm_fi)
                    norm_fi_fn = Function('norm_f'+str(i),[self.scxl,self.scul,self.scxc,self.scuc],[norm_fi],['scxl0','scul0','scxc0','scuc0'],['norm_ff'+str(i)])
                    self.fi += [norm_fi_fn]
                    ti_kb = Rl_k.T@di_k*ti_k # cable tension vector in {B}
                    tc_k[i*3:(i+1)*3] = ti_kb
                    # cross cable safe constraintss
                    eiPil= ei[0:2].T@pib_k[0:2]
                    eiPil_fn = Function('gc'+str(i),[self.scxl,self.scxc],[eiPil],['scxl0','scxc0'],['gcf'+str(i)])
                    self.Ei_Pil += -self.p_bar * log(eiPil + self.rl)
                    self.Ei_Pil += -self.p_bar * log(self.cl0+self.rl - eiPil)
                    self.Gio    += [eiPil_fn ]

                for j in range(i+1,int(self.nq)): # safe inter-robot separation constraints
                    xj_k   = self.scxc[j*self.nxi:(j+1)*self.nxi]
                    dj_k   = xj_k[0:3]
                    rj     = np.reshape(self.ra[:,j],(3,1))
                    pjb_k  = rj + (kc/(num_dis))*self.cl0*Rl_k.T@dj_k # body frame
                    disij  = pib_k[0:2]-pjb_k[0:2]
                    gij    = disij.T@disij - (kc/(num_dis)*4*self.rq)**2 # 4rq in training
                    self.gij += -self.p_bar * log(gij)
                    gij_fn = Function('g'+str(k),[self.scxl,self.scxc],[gij],['scxl0','scxc0'],['gf'+str(k)])
                    self.Gij += [gij_fn]
                
                    k     += 1
            
        # control consensus constraint that maps tension forces to the load control wrench
        wrench   = vertcat(Rl_k.T@Fl_k,Ml_k) # body frame
        W_cons   = self.Pt@tc_k - wrench
        self.h_wcons  = 1/(2*self.p_bar)*W_cons.T@W_cons
        self.W_cons_fn = Function('W_cons',[self.scxl,self.scul,self.scxc],[W_cons],['scxl0','scul0','scxc0'],['W_consf'])


    def SetADMMSubP2_SoftCost_k(self):
        # at each step k
        self.J_2_soft_k    = self.Jl_P2_k  + self.gco + self.gij + self.Tcon + self.ql_n + self.din + self.sumfi + self.h_wcons + self.Uicon + self.G_lo + self.Ei_Pil
        for i in range(int(self.nq)):
            xi      = self.xc[i*self.nxi:(i+1)*self.nxi]   # cable primal state
            scxi    = self.scxc[i*self.nxi:(i+1)*self.nxi] # safe copy state of each cable
            scxI    = self.scxC[i*self.nxi:(i+1)*self.nxi] # Lagrangian multiplier
            ui      = self.uc[i*self.nui:(i+1)*self.nui]   # cable primal control
            scui    = self.scuc[i*self.nui:(i+1)*self.nui] # safe copy control of each cable
            scuI    = self.scuC[i*self.nui:(i+1)*self.nui] # Lagrangian multiplier
            resid_x = xi - scxi + scxI/self.pix_dis
            resid_u = ui - scui + scuI/self.piu_dis
            self.J_2_soft_k    += self.pix_dis/2*resid_x.T@resid_x + self.piu_dis/2*resid_u.T@resid_u
        self.J_2_soft_k_orig =   self.gco + self.gij + self.Tcon + self.ql_n + self.din + self.sumfi + self.h_wcons + self.Uicon + self.G_lo + self.Ei_Pil 
    

    def SetADMMSubP2_SoftCost_N(self):
        # at the terminal step N
        self.J_2_soft_N    = self.Jl_P2_N   + self.gco + self.gij + self.Tcon + self.ql_n + self.din + self.G_lo + self.Ei_Pil 
        for i in range(int(self.nq)):
            xi      = self.xc[i*self.nxi:(i+1)*self.nxi]   # cable primal state
            scxi    = self.scxc[i*self.nxi:(i+1)*self.nxi] # safe copy state of each cable
            scxI    = self.scxC[i*self.nxi:(i+1)*self.nxi] # Lagrangian multiplier
            resid_x = xi - scxi + scxI/self.pix_dis
            self.J_2_soft_N    += self.pix_dis/2*resid_x.T@resid_x 
        self.J_2_soft_N_orig =  self.gco + self.gij + self.Tcon + self.ql_n + self.din  + self.G_lo + self.Ei_Pil 


    

    def ADMM_SubP2_Init(self):
        # static optimization problem at step k 
        # start with an empty NLP
        w2        = [] # optimal trajectory list
        self.w02  = [] # initial guess list of optimal trajectory 
        self.lbw2 = [] # lower boundary list of optimal variables
        self.ubw2 = [] # upper boundary list of optimal variables
        g2        = [] # equality and inequality constraint list
        self.lbg2 = [] # lower boundary list of constraints
        self.ubg2 = [] # upper boundary list of constraints
        
        # hyperparameters + external signals
        Para2    = SX.sym('P2', (self.nxl # load primal state  
                                +self.nxl # load primal state's Lagrangian multiplier     
                                +self.nul # load primal control
                                +self.nul # load primal control's Lagrangian multuplier
                                +self.nxi*int(self.nq) # all the cable primal states
                                +self.nxi*int(self.nq) # all the cable primal states' Lagrangian multipliers
                                +self.nui*int(self.nq) # all the cable primal controls
                                +self.nui*int(self.nq) # all the cable primal controls' Lagrangian multipliers
                                +self.npl # load hyperparameters
                                +self.npi # cable shared hyperparameters     
                                +1 # the ADMM iteration index
                                )) 

        # formulate the NLP
        n_start_pl  = 2*(self.nxl+self.nul+self.nxi*int(self.nq)+self.nui*int(self.nq))
        para_l      = Para2[n_start_pl:n_start_pl+self.npl] 
        para_i      = Para2[n_start_pl+self.npl:n_start_pl+self.npl+self.npi]
        a           = Para2[-1]
        scxl_k      = SX.sym('scxl',self.nxl)
        w2         += [scxl_k]
        self.lbw2  += self.xl_lb
        self.ubw2  += self.xl_ub
        scul_k      = SX.sym('scul',self.nul)
        w2         += [scul_k]
        self.lbw2  += self.ul_lb
        self.ubw2  += self.ul_ub
        xl_k        = Para2[0:self.nxl]
        scxL_k      = Para2[self.nxl:2*self.nxl]
        ul_k        = Para2[2*self.nxl:2*self.nxl+self.nul]
        scuL_k      = Para2[2*self.nxl+self.nul:2*(self.nxl+self.nul)]
        # total cost at the step k that includes the load and all the cables
        J2          = self.Jl_P2_k_fn(xl0=xl_k,ul0=ul_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,paral0=para_l,a0=a)['Jl_P2_kf']
        scxc_k      = SX.sym('scxc',self.nxi*int(self.nq))
        scuc_k      = SX.sym('scuc',self.nui*int(self.nq))
        g2         += [self.ql_fn(scxl0=scxl_k)['norm_qlf']]
        self.lbg2  += [1]
        self.ubg2  += [1]
        g2         += [self.po1_fn(scxl0=scxl_k)['po1f_admm']]
        self.lbg2  += [1e-2]
        self.ubg2  += [1e4]
        g2         += [self.po2_fn(scxl0=scxl_k)['po2f_admm']]
        self.lbg2  += [1e-2]
        self.ubg2  += [1e4]
        for i in range(int(self.nq)):
            scxi_k  = SX.sym('scx'+str(i),self.nxi)
            w2     += [scxi_k]
            self.lbw2  += self.scxi_lb
            self.ubw2  += self.scxi_ub
            scxc_k[i*self.nxi:(i+1)*self.nxi] = scxi_k
            scui_k  = SX.sym('scu'+str(i),self.nui)
            w2     += [scui_k]
            self.lbw2  += self.ui_lb
            self.ubw2  += self.ui_ub
            scuc_k[i*self.nui:(i+1)*self.nui] = scui_k
            n_start_xi   = 2*(self.nxl+self.nul)
            xi_k    = Para2[n_start_xi+i*self.nxi:n_start_xi+(i+1)*self.nxi] # cable primal state
            n_start_scxI = n_start_xi + self.nxi*int(self.nq)
            scxI_k  = Para2[n_start_scxI+i*self.nxi:n_start_scxI+(i+1)*self.nxi] # cable primal state Lagrangian multiplier
            n_start_ui   = n_start_scxI + self.nxi*int(self.nq)
            ui_k    = Para2[n_start_ui+i*self.nui:n_start_ui+(i+1)*self.nui]
            n_start_scuI = n_start_ui + self.nui*int(self.nq)
            scuI_k  = Para2[n_start_scuI+i*self.nui:n_start_scuI+(i+1)*self.nui]
            J2     += self.Ji_P2_k_fn(xi0=xi_k,scxi0=scxi_k,scxI0=scxI_k,ui0=ui_k,scui0=scui_k,scuI0=scuI_k, parai0=para_i, a0=a)['Ji_P2_kf']
        
        for i in range(int(self.nq)):    
            # safe constriant between the obstacle 1 and the ith quadrotor
            goi1       = self.Gi1[i](scxl0=scxl_k,scxc0=scxc_k)['go1f'+str(i)]
            g2        += [goi1]
            self.lbg2 += [1e-2]
            self.ubg2 += [1e4] # add an upbound for numerical stability
            # safe constriant between the obstacle 2 and the ith quadrotor
            goi2       = self.Gi2[i](scxl0=scxl_k,scxc0=scxc_k)['go2f'+str(i)]
            g2        += [goi2]
            self.lbg2 += [1e-2]
            self.ubg2 += [1e4] # add an upbound for numerical stability
            # quadrotor's thrust limit
            gif        = self.fi[i](scxl0=scxl_k,scul0=scul_k,scxc0=scxc_k,scuc0=scuc_k)['norm_ff'+str(i)]
            g2        += [gif]
            self.lbg2 += [1e-2]
            self.ubg2 += [self.fmax**2] # tianchen's parameter
            # direction unit norm
            g2        += [self.Di[i](scxc0=scxc_k)['dn'+str(i)]]
            self.lbg2 += [1]
            self.ubg2 += [1] 
            # avoidance of cable-crossing constraints
            gio        = self.Gio[i](scxl0=scxl_k,scxc0=scxc_k)['gcf'+str(i)]
            g2        += [gio]
            self.lbg2 += [-self.rl]
            self.ubg2 += [self.rl+self.cl0] 
           
            
        
        for k in range(len(self.Gij)):
            gij        = self.Gij[k](scxl0=scxl_k,scxc0=scxc_k)['gf'+str(k)]
            g2        += [gij]
            self.lbg2 += [1e-2]
            self.ubg2 += [1e4]
      

        
        # control consensus constraint
        g_wc       = self.W_cons_fn(scxl0=scxl_k,scul0=scul_k,scxc0=scxc_k)['W_consf']
        g2        += [g_wc]
        self.lbg2 += self.nul*[0]
        self.ubg2 += self.nul*[0] 

        # create an NLP solver and solve it
        # optsi2 = {}
        # optsi2['ipopt.tol'] = 1e-8
        # optsi2['ipopt.print_level'] = 0
        # optsi2['print_time'] = 0
        # optsi2['ipopt.warm_start_init_point']='yes'
        # optsi2['ipopt.max_iter']=2e3
        # optsi2['ipopt.acceptable_tol']=1e-8
        # optsi2['ipopt.mu_strategy']='adaptive'
        # optsi2['ipopt.diverging_iterates_tol'] = 1e8


        optsi2 = {
        'ipopt.print_level': 0,
        'print_time': 0,
        # 'ipopt.mu_strategy': 'adaptive',
        'ipopt.tol': 1e-8,
        'ipopt.acceptable_tol': 1e-8,
        'ipopt.max_iter': 2e3,
        'ipopt.warm_start_init_point': 'yes'
        # 'ipopt.diverging_iterates_tol':1e5,
        # 'ipopt.acceptable_iter':20,
        # 'ipopt.bound_relax_factor':1e-8 # this value is very important for stability!
        # 'ipopt.nlp_scaling_method': 'gradient-based',
        # 'ipopt.nlp_scaling_max_gradient': 100.0
        }


        prob2 = {'f': J2, 
                'x': vertcat(*w2), 
                'p': Para2,
                'g': vertcat(*g2)}
        self.solver2 = nlpsol('solver2', 'ipopt', prob2, optsi2)  



    def ADMM_SubP2_N_Init(self):
        # static optimization problem at step N (terminal) 
        # start with an empty NLP
        w2N        = [] # optimal trajectory list
        self.w02N  = [] # initial guess list of optimal trajectory 
        self.lbw2N = [] # lower boundary list of optimal variables
        self.ubw2N = [] # upper boundary list of optimal variables
        g2N        = [] # equality and inequality constraint list
        self.lbg2N = [] # lower boundary list of constraints
        self.ubg2N = [] # upper boundary list of constraints
        
        # hyperparameters + external signals
        Para2    = SX.sym('P2N', (self.nxl # load primal state at step N
                                +self.nxl  # load primal state's Lagrangian multiplier     
                                +self.nxi*int(self.nq) # all the cable primal states
                                +self.nxi*int(self.nq) # all the cable primal states' Lagrangian multipliers
                                +self.npl  # load hyperparameters
                                +self.npi  # cable hyperparameters
                                +1         # ADMM iteration index
                                )) 

        # formulate the NLP
        n_start_pl = 2*(self.nxl+self.nxi*int(self.nq))
        para_l     = Para2[n_start_pl:n_start_pl+self.npl] # penalty parameter of the load
        para_i     = Para2[n_start_pl+self.npl:n_start_pl+self.npl+self.npi]
        a          = Para2[-1]
        scxl_k     = SX.sym('scxl',self.nxl)
        w2N      += [scxl_k]
        self.lbw2N  += self.xl_lb
        self.ubw2N  += self.xl_ub
        xl_k     = Para2[0:self.nxl]
        scxL_k   = Para2[self.nxl:2*self.nxl]
        # total cost at the step k that includes the load and all the cables
        J2       = self.Jl_P2_N_fn(xl0=xl_k,scxl0=scxl_k,scxL0=scxL_k,paral0=para_l,a0=a)['Jl_P2_Nf']
        scxc_k   = SX.sym('scxc',self.nxi*int(self.nq))
        g2N        += [self.ql_fn(scxl0=scxl_k)['norm_qlf']]
        self.lbg2N  += [1]
        self.ubg2N  += [1]
        g2N         += [self.po1_fn(scxl0=scxl_k)['po1f_admm']]
        self.lbg2N  += [1e-2]
        self.ubg2N  += [1e4]
        g2N         += [self.po2_fn(scxl0=scxl_k)['po2f_admm']]
        self.lbg2N  += [1e-2]
        self.ubg2N  += [1e4]
        for i in range(int(self.nq)):
            scxi_k  = SX.sym('scx'+str(i),self.nxi)
            w2N     += [scxi_k]
            self.lbw2N  += self.scxi_lb
            self.ubw2N  += self.scxi_ub
            scxc_k[i*self.nxi:(i+1)*self.nxi] = scxi_k
            xi_k    = Para2[2*self.nxl+i*self.nxi:2*self.nxl+(i+1)*self.nxi] # cable primal state
            scxI_k  = Para2[2*self.nxl+self.nxi*int(self.nq)+i*self.nxi:2*self.nxl+self.nxi*int(self.nq)+(i+1)*self.nxi] # cable primal state Lagrangian multiplier
            J2     += self.Ji_P2_N_fn(xi0=xi_k,scxi0=scxi_k,scxI0=scxI_k,parai0=para_i,a0=a)['Jl_P2_Nf']
        
        for i in range(int(self.nq)):
            # safe constriant between the obstacle 1 and the ith quadrotor
            goi1        = self.Gi1[i](scxl0=scxl_k,scxc0=scxc_k)['go1f'+str(i)]
            g2N        += [goi1]
            self.lbg2N += [1e-2]
            self.ubg2N += [1e4] # add an upbound for numerical stability
            # safe constriant between the obstacle 2 and the ith quadrotor
            goi2        = self.Gi2[i](scxl0=scxl_k,scxc0=scxc_k)['go2f'+str(i)]
            g2N        += [goi2]
            self.lbg2N += [1e-2]
            self.ubg2N += [1e4] # add an upbound for numerical stability
            # direction unit norm
            g2N        += [self.Di[i](scxc0=scxc_k)['dn'+str(i)]]
            self.lbg2N += [1]
            self.ubg2N += [1]
            # avoidance of cable-crossing constraints
            gio        = self.Gio[i](scxl0=scxl_k,scxc0=scxc_k)['gcf'+str(i)]
            g2N       += [gio]
            self.lbg2N += [-self.rl]
            self.ubg2N += [self.rl+self.cl0] 
        
        for k in range(len(self.Gij)):
            gij         = self.Gij[k](scxl0=scxl_k,scxc0=scxc_k)['gf'+str(k)]
            g2N        += [gij]
            self.lbg2N += [1e-2]
            self.ubg2N += [1e4]
        

        # create an NLP solver and solve it
        # optsi2N = {}
        # optsi2N['ipopt.tol'] = 1e-8
        # optsi2N['ipopt.print_level'] = 0
        # optsi2N['print_time'] = 0
        # optsi2N['ipopt.warm_start_init_point']='yes'
        # optsi2N['ipopt.max_iter']=2e3
        # optsi2N['ipopt.acceptable_tol']=1e-8
        # optsi2N['ipopt.mu_strategy']='adaptive'
        # optsi2['ipopt.bound_relax_factor']=1e-12
        # optsi2['ipopt.limited_memory_max_history'] = 20
        # optsi2['ipopt.nlp_scaling_method']='gradient-based'
        # optsi2['ipopt.limited_memory_initialization'] = 'scalar1'

        optsi2N = {
        'ipopt.print_level': 0,
        'print_time': 0,
        # 'ipopt.mu_strategy': 'adaptive',
        'ipopt.tol': 1e-8,
        'ipopt.acceptable_tol': 1e-8,# default value 1e-6
        'ipopt.max_iter': 2e3,
        'ipopt.warm_start_init_point': 'yes'
        # 'ipopt.diverging_iterates_tol':1e5, # default value 1e20
        # 'ipopt.acceptable_iter':20, # default value 15
        # 'ipopt.bound_relax_factor':1e-9 # default value 1e-8
        # 'ipopt.nlp_scaling_method': 'gradient-based',
        # 'ipopt.nlp_scaling_max_gradient': 100.0 # default value 1e2
        }

        prob2N = {'f': J2, 
                'x': vertcat(*w2N), 
                'p': Para2,
                'g': vertcat(*g2N)}
        self.solver2N = nlpsol('solver2N', 'ipopt', prob2N, optsi2N)  


    
    def ADMM_SubP2(self,Para2_cable):
        # Para2_cable = SX.sym('p2_cable',(self.nxl*(self.N+1) # load reference state for initialization
        #                                 +self.nul*self.N # load reference control for initialization 
        #                                 +self.nxi*self.nq*(self.N+1) # cables' reference states for initialization
        #                                 +self.nui*self.nq # cables' reference controls for initialization
        #---------------------------------------------------------------------------------------------------
        #                                 +self.nxl*(self.N+1) # load primal state trajectory
        #                                 +self.nxl*(self.N+1) # load primal state's Lagrangian multiplier trajectory
        #                                 +self.nul*self.N # load primal control trajectory
        #                                 +self.nul*self.N # load primal control's Lagrangian multiplier trajectory
        #                                 +self.nxi*self.nq*(self.N+1) # cables' primal state trajectories
        #                                 +self.nxi*self.nq*(self.N+1) # cables' primal state's Lagrangian multiplier trajectories
        #                                 +self.nui*self.nq*self.N # cables' primal control trajectories
        #                                 +self.nui*self.nq*self.N # cables' primal control's Lagrangian multiplier trajectories
        #                                 +self.npl # load hyperparameters
        #                                 +self.npi # cable hyperparameters
        #                                 +1 # ADMM iteration index
        #))
        scxl_traj    = np.zeros((self.N+1,self.nxl))
        scul_traj    = np.zeros((self.N,self.nul))
        scxc_traj    = [np.zeros((self.N+1,self.nxi)) for _ in range(int(self.nq))]
        scuc_traj    = [np.zeros((self.N,self.nui)) for _ in range(int(self.nq))]
        n_start_pl   = 3*self.nxl*(self.N+1)+3*self.nul*self.N+3*self.nxi*int(self.nq)*(self.N+1)+2*self.nui*int(self.nq)*self.N + self.nui*int(self.nq)
        para_l       = Para2_cable[n_start_pl:n_start_pl+self.npl] # load ADMM penalty parameter for load state
        para_i       = Para2_cable[n_start_pl+self.npl:n_start_pl+self.npl+self.npi]
        a            = Para2_cable[-1]
        for k in range(self.N):
            self.w02 = []
            xl_ref   = Para2_cable[k*self.nxl:(k+1)*self.nxl]
            ul_ref   = Para2_cable[2*self.nxl*(self.N+1)+k*self.nul:2*self.nxl*(self.N+1)+(k+1)*self.nul]
            scxl0    = []
            for j in range(self.nxl):
                scxl0 += [xl_ref[j]]
            self.w02 += scxl0
            scul0    = []
            for j in range(self.nul):
                scul0 += [ul_ref[j]]
            self.w02 += scul0
            n_start_xl   = self.nxl*(self.N+1)+self.nul*self.N+self.nxi*int(self.nq)*(self.N+1)+self.nui*int(self.nq)
            xl_k         = Para2_cable[n_start_xl+k*self.nxl:n_start_xl+(k+1)*self.nxl]
            n_start_scxL = n_start_xl + self.nxl*(self.N+1)
            scxL_k       = Para2_cable[n_start_scxL+k*self.nxl:n_start_scxL+(k+1)*self.nxl]
            n_start_ul   = n_start_scxL + self.nxl*(self.N+1)
            ul_k         = Para2_cable[n_start_ul+k*self.nul:n_start_ul+(k+1)*self.nul]
            n_start_scuL = n_start_ul + self.nul*self.N
            scuL_k       = Para2_cable[n_start_scuL+k*self.nul:n_start_scuL+(k+1)*self.nul]
            n_start_xc   = n_start_scuL + self.nul*self.N
            xc_k         = Para2_cable[n_start_xc+k*self.nxi*int(self.nq):n_start_xc+(k+1)*self.nxi*int(self.nq)]
            n_start_scxC = n_start_xc + self.nxi*int(self.nq)*(self.N+1)
            scxC_k       = Para2_cable[n_start_scxC+k*self.nxi*int(self.nq):n_start_scxC+(k+1)*self.nxi*int(self.nq)]
            n_start_uc   = n_start_scxC + self.nxi*int(self.nq)*(self.N+1)
            uc_k         = Para2_cable[n_start_uc+k*self.nui*int(self.nq):n_start_uc+(k+1)*self.nui*int(self.nq)]
            n_start_scuC = n_start_uc + self.nui*int(self.nq)*self.N
            scuC_k       = Para2_cable[n_start_scuC+k*self.nui*int(self.nq):n_start_scuC+(k+1)*self.nui*int(self.nq)]
            xq_ref_k     = Para2_cable[self.nxl*(self.N+1)+self.nul*self.N+k*self.nxi*int(self.nq):self.nxl*(self.N+1)+self.nul*self.N+(k+1)*self.nxi*int(self.nq)]
            
            for i in range(int(self.nq)):
                scxi0   = []
                xi_ref  = xq_ref_k[i*self.nxi:(i+1)*self.nxi]
                for j in range(self.nxi):
                    scxi0 +=[xi_ref[j]]
                self.w02 += scxi0
                scui0   = []
                ui_ref  = Para2_cable[self.nxl*(self.N+1)+self.nul*self.N+self.nxi*int(self.nq)*(self.N+1)+i*self.nui:self.nxl*(self.N+1)+self.nul*self.N+self.nxi*int(self.nq)*(self.N+1)+(i+1)*self.nui]
                for j in range(self.nui):
                    scui0 +=[ui_ref[j]]
                self.w02 += scui0
            para2   = np.concatenate((xl_k,scxL_k))
            para2   = np.concatenate((para2,ul_k))
            para2   = np.concatenate((para2,scuL_k))
            para2   = np.concatenate((para2,xc_k))
            para2   = np.concatenate((para2,scxC_k))
            para2   = np.concatenate((para2,uc_k))
            para2   = np.concatenate((para2,scuC_k))
            para2   = np.concatenate((para2,para_l))
            para2   = np.concatenate((para2,para_i))
            para2   = np.concatenate((para2,[a]))
            # Solve the NLP
            sol2 = self.solver2(x0=self.w02, 
                          lbx=self.lbw2, 
                          ubx=self.ubw2, 
                          p=para2,
                          lbg=self.lbg2, 
                          ubg=self.ubg2)
            w_opt2 = sol2['x'].full().flatten()
            # take the optimal control and state
            sol_traj = np.reshape(w_opt2, (-1, self.nxl + self.nul + (self.nxi+self.nui)*int(self.nq)))
            scxl_opt = sol_traj[:,0:self.nxl]
            scul_opt = sol_traj[:,self.nxl:self.nxl + self.nul]
            scc_opt  = sol_traj[:,self.nxl + self.nul:self.nxl + self.nul + (self.nxi+self.nui)*int(self.nq)]
            scxl_traj[k:k+1,:] = scxl_opt
            scul_traj[k:k+1,:] = scul_opt
            for i in range(int(self.nq)):
                scxc_traj[i][k:k+1,:]=scc_opt[:,i*(self.nxi+self.nui):i*(self.nxi+self.nui)+self.nxi]
                scuc_traj[i][k:k+1,:]=scc_opt[:,i*(self.nxi+self.nui)+self.nxi:(i+1)*(self.nxi+self.nui)]
        
        # terminal cost
        self.w02N = []
        xl_ref   = Para2_cable[self.N*self.nxl:(self.N+1)*self.nxl]
        scxl0N    = []
        for j in range(self.nxl):
            scxl0N += [xl_ref[j]]
        self.w02N += scxl0N
        xq_ref_N  = Para2_cable[self.nxl*(self.N+1)+self.nul*self.N+self.nxi*int(self.nq)*self.N:self.nxl*(self.N+1)+self.nul*self.N+self.nxi*int(self.nq)*(self.N+1)]
        for i in range(int(self.nq)):
            scxi0N   = []
            xi_ref  = xq_ref_N[i*self.nxi:(i+1)*self.nxi]
            for j in range(self.nxi):
                scxi0N +=[xi_ref[j]]
            self.w02N += scxi0N
        xl_N    = Para2_cable[n_start_xl+self.N*self.nxl:n_start_xl+(self.N+1)*self.nxl]
        scxL_N  = Para2_cable[n_start_scxL+self.N*self.nxl:n_start_scxL+(self.N+1)*self.nxl]
        xc_N    = Para2_cable[n_start_xc+self.N*self.nxi*int(self.nq):n_start_xc+(self.N+1)*self.nxi*int(self.nq)]
        scxC_N  = Para2_cable[n_start_scxC+self.N*self.nxi*int(self.nq):n_start_scxC+(self.N+1)*self.nxi*int(self.nq)]
        para2N  = np.concatenate((xl_N,scxL_N))
        para2N  = np.concatenate((para2N,xc_N))
        para2N  = np.concatenate((para2N,scxC_N))
        para2N  = np.concatenate((para2N,para_l))
        para2N  = np.concatenate((para2N,para_i))
        para2N  = np.concatenate((para2N,[a]))
        # Solve the NLP
        sol2N = self.solver2N(x0=self.w02N, 
                          lbx=self.lbw2N, 
                          ubx=self.ubw2N, 
                          p=para2N,
                          lbg=self.lbg2N, 
                          ubg=self.ubg2N)
        w_opt2N = sol2N['x'].full().flatten()
        sol_trajN = np.reshape(w_opt2N, (-1, self.nxl + self.nxi*int(self.nq)))
        scxl_optN = sol_trajN[:,0:self.nxl]
        scxc_optN = sol_trajN[:,self.nxl:self.nxl+ self.nxi*int(self.nq)]
        scxl_traj[self.N:self.N+1,:] = scxl_optN
        for i in range(int(self.nq)):
            scxc_traj[i][self.N:self.N+1,:]=scxc_optN[:,i*self.nxi:(i+1)*self.nxi]
        # output
        opt_sol2 = {"scxl_traj":scxl_traj,
                    "scul_traj":scul_traj,
                    "scxc_traj":scxc_traj,
                    "scuc_traj":scuc_traj
                    }
        
        return opt_sol2
    

    def system_derivatives_SubP2_ADMM_k(self):
        # gradients of the Lagrangian (augmented cost function with the soft constraints)
        self.Lscxl          = jacobian(self.J_2_soft_k,self.scxl)
        self.Lscul          = jacobian(self.J_2_soft_k,self.scul)
        self.Lscxc          = jacobian(self.J_2_soft_k,self.scxc)
        self.Lscuc          = jacobian(self.J_2_soft_k,self.scuc)
        # gradients of the original Lagrangian (augmented cost with the soft constraints but without the ADMM penalties)
        self.Lscxl_o        = jacobian(self.J_2_soft_k_orig,self.scxl)
        self.Lscul_o        = jacobian(self.J_2_soft_k_orig,self.scul)
        self.Lscxc_o        = jacobian(self.J_2_soft_k_orig,self.scxc)
        self.Lscuc_o        = jacobian(self.J_2_soft_k_orig,self.scuc)
        # hessians
        self.Lscxlscxl      = jacobian(self.Lscxl,self.scxl)
        self.Lscxlscxl_fn   = Function('Lscxlscxl',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscxlscxl],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscxlscxl_f'])
        self.Lscxlscul      = jacobian(self.Lscxl,self.scul)
        self.Lscxlscul_fn   = Function('Lscxlscul',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscxlscul],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscxlscul_f'])
        self.Lscxlscxc      = jacobian(self.Lscxl,self.scxc)
        self.Lscxlscxc_fn   = Function('Lscxlscxc',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscxlscxc],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscxlscxc_f'])
        self.Lscxlscuc      = jacobian(self.Lscxl,self.scuc)
        self.Lscxlscuc_fn   = Function('Lscxlscuc',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscxlscuc],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscxlscuc_f'])
        self.Lsculscul      = jacobian(self.Lscul,self.scul)
        self.Lsculscul_fn   = Function('Lsculscul',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lsculscul],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lsculscul_f'])
        self.Lsculscxc      = jacobian(self.Lscul,self.scxc)
        self.Lsculscxc_fn   = Function('Lsculscxc',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lsculscxc],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lsculscxc_f'])
        self.Lsculscuc      = jacobian(self.Lscul,self.scuc)
        self.Lsculscuc_fn   = Function('Lsculscuc',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lsculscuc],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lsculscuc_f'])
        self.Lscxcscxc      = jacobian(self.Lscxc,self.scxc)
        self.Lscxcscxc_fn   = Function('Lscxcscxc',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscxcscxc],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscxcscxc_f'])
        self.Lscxcscuc      = jacobian(self.Lscxc,self.scuc)
        self.Lscxcscuc_fn   = Function('Lscxcscuc',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscxcscuc],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscxcscuc_f'])
        self.Lscucscuc      = jacobian(self.Lscuc,self.scuc)
        self.Lscucscuc_fn   = Function('Lscucscuc',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscucscuc],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscucscuc_f'])
        # hessians of the original Lagrangian
        self.Lscxlscxl_o    = jacobian(self.Lscxl_o,self.scxl)
        self.Lscxlscxl_fno  = Function('Lscxlscxlo',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lscxlscxl_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lscxlscxlo_f'])
        self.Lscxlscul_o    = jacobian(self.Lscxl_o,self.scul)
        self.Lscxlscul_fno  = Function('Lscxlsculo',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lscxlscul_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lscxlsculo_f'])
        self.Lscxlscxc_o    = jacobian(self.Lscxl_o,self.scxc)
        self.Lscxlscxc_fno  = Function('Lscxlscxco',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lscxlscxc_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lscxlscxco_f'])
        self.Lscxlscuc_o    = jacobian(self.Lscxl_o,self.scuc)
        self.Lscxlscuc_fno  = Function('Lscxlscuco',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lscxlscuc_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lscxlscuco_f'])
        self.Lsculscul_o    = jacobian(self.Lscul_o,self.scul)
        self.Lsculscul_fno  = Function('Lsculsculo',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lsculscul_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lsculsculo_f'])
        self.Lsculscxc_o    = jacobian(self.Lscul_o,self.scxc)
        self.Lsculscxc_fno  = Function('Lsculscxco',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lsculscxc_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lsculscxco_f'])
        self.Lsculscuc_o    = jacobian(self.Lscul_o,self.scuc)
        self.Lsculscuc_fno  = Function('Lsculscuco',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lsculscuc_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lsculscuco_f'])
        self.Lscxcscxc_o    = jacobian(self.Lscxc_o,self.scxc)
        self.Lscxcscxc_fno  = Function('Lscxcscxco',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lscxcscxc_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lscxcscxco_f'])
        self.Lscxcscuc_o    = jacobian(self.Lscxc_o,self.scuc)
        self.Lscxcscuc_fno  = Function('Lscxcscuco',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lscxcscuc_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lscxcscuco_f'])
        self.Lscucscuc_o    = jacobian(self.Lscuc_o,self.scuc)
        self.Lscucscuc_fno  = Function('Lscucscuco',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lscucscuc_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lscucscuco_f'])

        # hessians w.r.t. the hyperparameters
        self.Lscxlp         = jacobian(self.Lscxl,self.P_auto)
        self.Lscxlp_fn      = Function('Lscxlp',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscxlp],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscxlp_f'])
        self.Lsculp         = jacobian(self.Lscul,self.P_auto)
        self.Lsculp_fn      = Function('Lsculp',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lsculp],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lsculp_f'])
        self.Lscxcp         = jacobian(self.Lscxc,self.P_auto)
        self.Lscxcp_fn      = Function('Lscxcp',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscxcp],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscxcp_f'])
        self.Lscucp         = jacobian(self.Lscuc,self.P_auto)
        self.Lscucp_fn      = Function('Lscucp',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscucp],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscucp_f'])

    def system_derivatives_SubP2_ADMM_N(self):
        # gradients of the Lagrangian (augmented cost function with the soft constraints)
        self.Lscxl_N        = jacobian(self.J_2_soft_N,self.scxl)
        self.Lscxc_N        = jacobian(self.J_2_soft_N,self.scxc)
        # gradients of the original Lagrangian (augmented cost with the soft constraints but without the ADMM penalties)
        self.Lscxl_N_o      = jacobian(self.J_2_soft_N_orig,self.scxl)
        self.Lscxc_N_o      = jacobian(self.J_2_soft_N_orig,self.scxc)
        # hessians
        self.Lscxlscxl_N    = jacobian(self.Lscxl_N,self.scxl)
        self.Lscxlscxl_N_fn = Function('LscxlscxlN',[self.xl,self.xc,self.scxl,self.scxL,self.scxc,self.scxC,self.P_auto,self.a],[self.Lscxlscxl_N],
                                       ['xl0','xc0','scxl0','scxL0','scxc0','scxC0','pauto0','a0'],['LscxlscxlN_f'])
        self.Lscxlscxc_N    = jacobian(self.Lscxl_N,self.scxc)
        self.Lscxlscxc_N_fn = Function('LscxlscxcN',[self.xl,self.xc,self.scxl,self.scxL,self.scxc,self.scxC,self.P_auto,self.a],[self.Lscxlscxc_N],
                                       ['xl0','xc0','scxl0','scxL0','scxc0','scxC0','pauto0','a0'],['LscxlscxcN_f'])
        self.Lscxcscxc_N    = jacobian(self.Lscxc_N,self.scxc)
        self.Lscxcscxc_N_fn = Function('LscxcscxcN',[self.xl,self.xc,self.scxl,self.scxL,self.scxc,self.scxC,self.P_auto,self.a],[self.Lscxcscxc_N],
                                       ['xl0','xc0','scxl0','scxL0','scxc0','scxC0','pauto0','a0'],['LscxcscxcN_f'])
        # hessians of the original Lagrangian
        self.Lscxlscxl_No    = jacobian(self.Lscxl_N_o,self.scxl)
        self.Lscxlscxl_N_fno = Function('LscxlscxlNo',[self.xl,self.xc,self.scxl,self.scxL,self.scxc,self.scxC],[self.Lscxlscxl_No],
                                       ['xl0','xc0','scxl0','scxL0','scxc0','scxC0'],['LscxlscxlNo_f'])
        self.Lscxlscxc_No    = jacobian(self.Lscxl_N_o,self.scxc)
        self.Lscxlscxc_N_fno = Function('LscxlscxcNo',[self.xl,self.xc,self.scxl,self.scxL,self.scxc,self.scxC],[self.Lscxlscxc_No],
                                       ['xl0','xc0','scxl0','scxL0','scxc0','scxC0'],['LscxlscxcNo_f'])
        self.Lscxcscxc_No    = jacobian(self.Lscxc_N_o,self.scxc)
        self.Lscxcscxc_N_fno = Function('LscxcscxcNo',[self.xl,self.xc,self.scxl,self.scxL,self.scxc,self.scxC],[self.Lscxcscxc_No],
                                       ['xl0','xc0','scxl0','scxL0','scxc0','scxC0'],['LscxcscxcNo_f'])
        # hessians w.r.t. the hyperparameters
        self.Lscxlp_N       = jacobian(self.Lscxl_N,self.P_auto)
        self.Lscxlp_N_fn    = Function('LscxlpN',[self.xl,self.xc,self.scxl,self.scxL,self.scxc,self.scxC,self.P_auto,self.a],[self.Lscxlp_N],
                                       ['xl0','xc0','scxl0','scxL0','scxc0','scxC0','pauto0','a0'],['LscxlpN_f'])
        self.Lscxcp_N       = jacobian(self.Lscxc_N,self.P_auto)
        self.Lscxcp_N_fn    = Function('LscxcpN',[self.xl,self.xc,self.scxl,self.scxL,self.scxc,self.scxC,self.P_auto,self.a],[self.Lscxcp_N],
                                       ['xl0','xc0','scxl0','scxL0','scxc0','scxC0','pauto0','a0'],['LscxcpN_f'])



    def Get_AuxSys_SubP2(self,opt_sol1_l,opt_sol1_c,opt_sol2,scxL,scuL,scxC_list,scuC_list,Pauto,i_admm):
        xl      = opt_sol1_l['xl_traj']
        ul      = opt_sol1_l['ul_traj']
        xc_list      = opt_sol1_c['xc_traj'] # list that contains all the cables' states
        uc_list      = opt_sol1_c['uc_traj'] # list that contains all the cables' controls
        scxl    = opt_sol2['scxl_traj']
        scul    = opt_sol2['scul_traj']
        scxc_list    = opt_sol2['scxc_traj'] # list that contains all the cables' safe states
        scuc_list    = opt_sol2['scuc_traj'] # list that contains all the cables' safe controls
        Lscxlscxl    = (self.N+1)*[np.zeros((self.nxl,self.nxl))]
        Lscxlscul    = self.N*[np.zeros((self.nxl,self.nul))]
        Lscxlscxc    = (self.N+1)*[np.zeros((self.nxl,self.nxi*int(self.nq)))]
        Lscxlscuc    = self.N*[np.zeros((self.nxl,self.nui*int(self.nq)))]
        Lsculscul    = self.N*[np.zeros((self.nul,self.nul))]
        Lsculscxc    = self.N*[np.zeros((self.nul,self.nxi*int(self.nq)))]
        Lsculscuc    = self.N*[np.zeros((self.nul,self.nui*int(self.nq)))]
        Lscxcscxc    = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.nxi*int(self.nq)))]
        Lscxcscuc    = self.N*[np.zeros((self.nxi*int(self.nq),self.nui*int(self.nq)))]
        Lscucscuc    = self.N*[np.zeros((self.nui*int(self.nq),self.nui*int(self.nq)))]
        Lscxlp       = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        Lsculp       = self.N*[np.zeros((self.nul,self.n_Pauto))]
        Lscxcp       = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.n_Pauto))]
        Lscucp       = self.N*[np.zeros((self.nui*int(self.nq),self.n_Pauto))]
        # hessians of the original Lagrangian for computing the minimal eigenvalue
        Lscxlscxl_o  = (self.N+1)*[np.zeros((self.nxl,self.nxl))]
        Lscxlscul_o  = self.N*[np.zeros((self.nxl,self.nul))]
        Lscxlscxc_o  = (self.N+1)*[np.zeros((self.nxl,self.nxi*int(self.nq)))]
        Lscxlscuc_o  = self.N*[np.zeros((self.nxl,self.nui*int(self.nq)))]
        Lsculscul_o  = self.N*[np.zeros((self.nul,self.nul))]
        Lsculscxc_o  = self.N*[np.zeros((self.nul,self.nxi*int(self.nq)))]
        Lsculscuc_o  = self.N*[np.zeros((self.nul,self.nui*int(self.nq)))]
        Lscxcscxc_o  = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.nxi*int(self.nq)))]
        Lscxcscuc_o  = self.N*[np.zeros((self.nxi*int(self.nq),self.nui*int(self.nq)))]
        Lscucscuc_o  = self.N*[np.zeros((self.nui*int(self.nq),self.nui*int(self.nq)))]
        for k in range(self.N):
            xl_k     = xl[k,:]
            ul_k     = ul[k,:]
            xc_k     = np.concatenate([xc_list[i][k,:] for i in range(int(self.nq))])
            uc_k     = np.concatenate([uc_list[i][k,:] for i in range(int(self.nq))])
            scxl_k   = scxl[k,:]
            scxL_k   = scxL[k,:]
            scul_k   = scul[k,:]
            scuL_k   = scuL[k,:]
            scxc_k   = np.concatenate([scxc_list[i][k,:] for i in range(int(self.nq))])
            scxC_k   = np.concatenate([scxC_list[i][k,:] for i in range(int(self.nq))])
            scuc_k   = np.concatenate([scuc_list[i][k,:] for i in range(int(self.nq))])
            scuC_k   = np.concatenate([scuC_list[i][k,:] for i in range(int(self.nq))])
            Lscxlscxl[k] = self.Lscxlscxl_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscxlscxl_f'].full()
            Lscxlscul[k] = self.Lscxlscul_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscxlscul_f'].full()
            Lscxlscxc[k] = self.Lscxlscxc_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscxlscxc_f'].full()
            Lscxlscuc[k] = self.Lscxlscuc_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscxlscuc_f'].full()
            Lsculscul[k] = self.Lsculscul_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lsculscul_f'].full()
            Lsculscxc[k] = self.Lsculscxc_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lsculscxc_f'].full()
            Lsculscuc[k] = self.Lsculscuc_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lsculscuc_f'].full()
            Lscxcscxc[k] = self.Lscxcscxc_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscxcscxc_f'].full()
            Lscxcscuc[k] = self.Lscxcscuc_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscxcscuc_f'].full()
            Lscucscuc[k] = self.Lscucscuc_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscucscuc_f'].full()
            Lscxlp[k]    = self.Lscxlp_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscxlp_f'].full()
            Lsculp[k]    = self.Lsculp_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lsculp_f'].full()
            Lscxcp[k]    = self.Lscxcp_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscxcp_f'].full()
            Lscucp[k]    = self.Lscucp_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscucp_f'].full()
            # hessians of the original Lagrangian
            Lscxlscxl_o[k] = self.Lscxlscxl_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lscxlscxlo_f'].full()
            Lscxlscul_o[k] = self.Lscxlscul_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lscxlsculo_f'].full()
            Lscxlscxc_o[k] = self.Lscxlscxc_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lscxlscxco_f'].full()
            Lscxlscuc_o[k] = self.Lscxlscuc_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lscxlscuco_f'].full()
            Lsculscul_o[k] = self.Lsculscul_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lsculsculo_f'].full()
            Lsculscxc_o[k] = self.Lsculscxc_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lsculscxco_f'].full()
            Lsculscuc_o[k] = self.Lsculscuc_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lsculscuco_f'].full()
            Lscxcscxc_o[k] = self.Lscxcscxc_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lscxcscxco_f'].full()
            Lscxcscuc_o[k] = self.Lscxcscuc_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lscxcscuco_f'].full()
            Lscucscuc_o[k] = self.Lscucscuc_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lscucscuco_f'].full()

        # ternimal hessians
        xl_N     = xl[-1,:]
        xc_N     = np.concatenate([xc_list[i][-1,:] for i in range(int(self.nq))])
        scxl_N   = scxl[-1,:]
        scxL_N   = scxL[-1,:]
        scxc_N   = np.concatenate([scxc_list[i][-1,:] for i in range(int(self.nq))])
        scxC_N   = np.concatenate([scxC_list[i][-1,:] for i in range(int(self.nq))])
        Lscxlscxl[self.N] = self.Lscxlscxl_N_fn(xl0=xl_N,xc0=xc_N,scxl0=scxl_N,scxL0=scxL_N,scxc0=scxc_N,scxC0=scxC_N,pauto0=Pauto,a0=i_admm)['LscxlscxlN_f'].full()
        Lscxlscxc[self.N] = self.Lscxlscxc_N_fn(xl0=xl_N,xc0=xc_N,scxl0=scxl_N,scxL0=scxL_N,scxc0=scxc_N,scxC0=scxC_N,pauto0=Pauto,a0=i_admm)['LscxlscxcN_f'].full()
        Lscxcscxc[self.N] = self.Lscxcscxc_N_fn(xl0=xl_N,xc0=xc_N,scxl0=scxl_N,scxL0=scxL_N,scxc0=scxc_N,scxC0=scxC_N,pauto0=Pauto,a0=i_admm)['LscxcscxcN_f'].full()
        Lscxlp[self.N]    = self.Lscxlp_N_fn(xl0=xl_N,xc0=xc_N,scxl0=scxl_N,scxL0=scxL_N,scxc0=scxc_N,scxC0=scxC_N,pauto0=Pauto,a0=i_admm)['LscxlpN_f'].full()
        Lscxcp[self.N]    = self.Lscxcp_N_fn(xl0=xl_N,xc0=xc_N,scxl0=scxl_N,scxL0=scxL_N,scxc0=scxc_N,scxC0=scxC_N,pauto0=Pauto,a0=i_admm)['LscxcpN_f'].full()
        # hessians of the original Lagrangian
        Lscxlscxl_o[self.N] = self.Lscxlscxl_N_fno(xl0=xl_N,xc0=xc_N,scxl0=scxl_N,scxL0=scxL_N,scxc0=scxc_N,scxC0=scxC_N)['LscxlscxlNo_f'].full()
        Lscxlscxc_o[self.N] = self.Lscxlscxc_N_fno(xl0=xl_N,xc0=xc_N,scxl0=scxl_N,scxL0=scxL_N,scxc0=scxc_N,scxC0=scxC_N)['LscxlscxcNo_f'].full()
        Lscxcscxc_o[self.N] = self.Lscxcscxc_N_fno(xl0=xl_N,xc0=xc_N,scxl0=scxl_N,scxL0=scxL_N,scxc0=scxc_N,scxC0=scxC_N)['LscxcscxcNo_f'].full()

        auxsys2 = {
            "Lscxlscxl":Lscxlscxl,
            "Lscxlscul":Lscxlscul,
            "Lscxlscxc":Lscxlscxc,
            "Lscxlscuc":Lscxlscuc,
            "Lsculscul":Lsculscul,
            "Lsculscxc":Lsculscxc,
            "Lsculscuc":Lsculscuc,
            "Lscxcscxc":Lscxcscxc,
            "Lscxcscuc":Lscxcscuc,
            "Lscucscuc":Lscucscuc,
            "Lscxlp":Lscxlp,
            "Lsculp":Lsculp,
            "Lscxcp":Lscxcp,
            "Lscucp":Lscucp,
            "Lscxlscxl_o":Lscxlscxl_o,
            "Lscxlscul_o":Lscxlscul_o,
            "Lscxlscxc_o":Lscxlscxc_o,
            "Lscxlscuc_o":Lscxlscuc_o,
            "Lsculscul_o":Lsculscul_o,
            "Lsculscxc_o":Lsculscxc_o,
            "Lsculscuc_o":Lsculscuc_o,
            "Lscxcscxc_o":Lscxcscxc_o,
            "Lscxcscuc_o":Lscxcscuc_o,
            "Lscucscuc_o":Lscucscuc_o
        }

        return auxsys2

    
    def ADMM_SubP3(self,xl_traj,scxl_traj,scxL_traj,ul_traj,scul_traj,scuL_traj,xc_traj,scxc_traj,scxC_traj,uc_traj,scuc_traj,scuC_traj,px,pu,gammax,gammau,pix,piu,gammaix,gammaiu,ADMM_max,i_admm):
        scxL_traj_new = np.zeros((self.N+1,self.nxl))
        scuL_traj_new = np.zeros((self.N,self.nul))
        scxC_traj_new = [np.zeros((self.N+1,self.nxi)) for _ in range(int(self.nq))]
        scuC_traj_new = [np.zeros((self.N,self.nui)) for _ in range(int(self.nq))]
        px_dis        = self.open_loop_penalty(px,gammax,i_admm,ADMM_max)
        pu_dis        = self.open_loop_penalty(pu,gammau,i_admm,ADMM_max)
        pix_dis       = self.open_loop_penalty(pix,gammaix,i_admm,ADMM_max)
        piu_dis       = self.open_loop_penalty(piu,gammaiu,i_admm,ADMM_max)
        for k in range(self.N):
            scxL_new  = scxL_traj[k,:] + px_dis*(xl_traj[k,:] - scxl_traj[k,:])
            scuL_new  = scuL_traj[k,:] + pu_dis*(ul_traj[k,:] - scul_traj[k,:])
            scxL_traj_new[k:k+1,:] = scxL_new
            scuL_traj_new[k:k+1,:] = scuL_new
            for i in range(int(self.nq)):
                scxI_new = scxC_traj[i][k,:] + pix_dis*(xc_traj[i][k,:] - scxc_traj[i][k,:])
                scxC_traj_new[i][k:k+1,:] = scxI_new
                scuI_new = scuC_traj[i][k,:] + piu_dis*(uc_traj[i][k,:] - scuc_traj[i][k,:])
                scuC_traj_new[i][k:k+1,:] = scuI_new
        #----terminal-----#
        scxL_new  = scxL_traj[self.N,:] + px_dis*(xl_traj[self.N,:] - scxl_traj[self.N,:])
        scxL_traj_new[self.N:self.N+1,:] = scxL_new
        for i in range(int(self.nq)):
            scxI_new = scxC_traj[i][self.N,:] + pix_dis*(xc_traj[i][self.N,:] - scxc_traj[i][self.N,:])
            scxC_traj_new[i][self.N:self.N+1,:] = scxI_new

        opt_sol3 = {"scxL_traj_new":scxL_traj_new,
                    "scuL_traj_new":scuL_traj_new,
                    "scxC_traj_new":scxC_traj_new,
                    "scuC_traj_new":scuC_traj_new
                    }
        
        return opt_sol3
    
    def system_derivatives_SubP3_ADMM(self):
        scxL_update = self.px_dis*(self.xl - self.scxl)
        scuL_update = self.pu_dis*(self.ul - self.scul)
        scxC_update = self.pix_dis*(self.xc - self.scxc)
        scuC_update = self.piu_dis*(self.uc - self.scuc)
        self.dscxL_updatedp    = jacobian(scxL_update,self.P_auto)
        self.dscxL_updatedp_fn = Function('dscxL_updatedp',[self.xl,self.scxl,self.para_l,self.a],[self.dscxL_updatedp],['xl0','scxl0','paral0','a0'],['dscxL_updatedp_f'])
        self.dscuL_updatedp    = jacobian(scuL_update,self.P_auto)
        self.dscuL_updatedp_fn = Function('dscuL_updatedp',[self.ul,self.scul,self.para_l,self.a],[self.dscuL_updatedp],['ul0','scul0','paral0','a0'],['dscuL_updatedp_f'])
        self.dscxC_updatedp    = jacobian(scxC_update,self.P_auto)
        self.dscxC_updatedp_fn = Function('dscxC_updatedp',[self.xc,self.scxc,self.para_i,self.a],[self.dscxC_updatedp],['xc0','scxc0','parai0','a0'],['dscxC_updatedp_f'])
        self.dscuC_updatedp    = jacobian(scuC_update,self.P_auto)
        self.dscuC_updatedp_fn = Function('dscuC_updatedp',[self.uc,self.scuc,self.para_i,self.a],[self.dscuC_updatedp],['uc0','scuc0','parai0','a0'],['dscuC_updatedp_f'])


    def Get_AuxSys_SubP3(self,opt_sol1_l,opt_sol1_c,opt_sol2,weight1,weight2,i_admm):
        xl      = opt_sol1_l['xl_traj']
        ul      = opt_sol1_l['ul_traj']
        xc_list = opt_sol1_c['xc_traj']
        uc_list = opt_sol1_c['uc_traj']
        scxl    = opt_sol2['scxl_traj']
        scul    = opt_sol2['scul_traj']
        scxc_list    = opt_sol2['scxc_traj']
        scuc_list    = opt_sol2['scuc_traj']
        dscxL_updatedp = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        dscuL_updatedp = self.N*[np.zeros((self.nul,self.n_Pauto))]
        dscxC_updatedp = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.n_Pauto))]
        dscuC_updatedp = self.N*[np.zeros((self.nui*int(self.nq),self.n_Pauto))]
        for k in range(self.N):
            xl_k     = xl[k,:]
            ul_k     = ul[k,:]
            xc_k     = np.concatenate([xc_list[i][k,:] for i in range(int(self.nq))])
            uc_k     = np.concatenate([uc_list[i][k,:] for i in range(int(self.nq))])
            scxl_k   = scxl[k,:]
            scul_k   = scul[k,:]
            scxc_k   = np.concatenate([scxc_list[i][k,:] for i in range(int(self.nq))])
            scuc_k   = np.concatenate([scuc_list[i][k,:] for i in range(int(self.nq))])
            dscxL_updatedp[k] = self.dscxL_updatedp_fn(xl0=xl_k,scxl0=scxl_k,paral0=weight1,a0=i_admm)['dscxL_updatedp_f'].full()
            dscuL_updatedp[k] = self.dscuL_updatedp_fn(ul0=ul_k,scul0=scul_k,paral0=weight1,a0=i_admm)['dscuL_updatedp_f'].full()
            dscxC_updatedp[k] = self.dscxC_updatedp_fn(xc0=xc_k,scxc0=scxc_k,parai0=weight2,a0=i_admm)['dscxC_updatedp_f'].full()
            dscuC_updatedp[k] = self.dscuC_updatedp_fn(uc0=uc_k,scuc0=scuc_k,parai0=weight2,a0=i_admm)['dscuC_updatedp_f'].full()
        xl_N     = xl[-1,:]
        scxl_N   = scxl[-1,:]
        xc_N     = np.concatenate([xc_list[i][-1,:] for i in range(int(self.nq))])
        scxc_N   = np.concatenate([scxc_list[i][-1,:] for i in range(int(self.nq))])
        dscxL_updatedp[self.N]= self.dscxL_updatedp_fn(xl0=xl_N,scxl0=scxl_N,paral0=weight1,a0=i_admm)['dscxL_updatedp_f'].full()
        dscxC_updatedp[self.N]= self.dscxC_updatedp_fn(xc0=xc_N,scxc0=scxc_N,parai0=weight2,a0=i_admm)['dscxC_updatedp_f'].full()

        auxSys3 = {
            "dscxL_updatedp":dscxL_updatedp,
            "dscuL_updatedp":dscuL_updatedp,
            "dscxC_updatedp":dscxC_updatedp,
            "dscuC_updatedp":dscuC_updatedp
        }
        return auxSys3


    def ADMM_forward_MPC(self,Ref_xl,Ref_ul,ref_xq,ref_uq,xl_fb,xq_fb,paral,paraC,max_iter_ADMM):
        # initial guess of the safe copy variable trajectories
        scxl_traj_tp = np.zeros(((self.N+1)*self.nxl))
        scul_traj_tp = Ref_ul
        for k in range(self.N):
            # scxl_traj_tp[k*self.nxl:(k+1)*self.nxl] = Ref_xl[k*self.nxl:(k+1)*self.nxl]
            scxl_traj_tp[(k)*self.nxl:(k+1)*self.nxl] = np.reshape(self.model_l_fn(xl0=scxl_traj_tp[k*self.nxl:(k+1)*self.nxl],ul0=Ref_ul[k*self.nul:(k+1)*self.nul])['mdynlf'].full(),self.nxl) # start from a bad initial state
        # scxl_traj_tp[self.N*self.nxl:(self.N+1)*self.nxl] = Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl]
        scxc_traj_tp = [np.zeros((self.N+1)*self.nxi) for _ in range(int(self.nq))]
        scuc_traj_tp = [np.zeros(self.N*self.nui)  for _ in range(int(self.nq))]
        for i in range(int(self.nq)):
            for k in range(self.N):
                # scxc_traj_tp[i][k*self.nxi:(k+1)*self.nxi] = ref_xq[i][k*self.nxi:(k+1)*self.nxi]
                scxc_traj_tp[i][(k)*self.nxi:(k+1)*self.nxi] = np.reshape(self.model_i_fn(xi0=scxc_traj_tp[i][k*self.nxi:(k+1)*self.nxi],ui0=ref_uq[i*self.nui:(i+1)*self.nui])['mdynif'].full(),self.nxi)   #ref_xq[i][k*self.nxi:(k+1)*self.nxi]
                scuc_traj_tp[i][k*self.nui:(k+1)*self.nui] = ref_uq[i*self.nui:(i+1)*self.nui]
            # scxc_traj_tp[i][self.N*self.nxi:(self.N+1)*self.nxi] = ref_xq[i][self.N*self.nxi:(self.N+1)*self.nxi]
        # initial guess of the Lagrangian multiplier trajectories
        scxL_traj_tp = np.zeros(((self.N+1)*self.nxl)) # 1D array
        scuL_traj_tp = np.zeros((self.N*self.nul)) # 1D array
        scxC_traj_tp = [np.zeros(((self.N+1)*self.nxi)) for _ in range(int(self.nq))] # list of 1D array
        scuC_traj_tp = [np.zeros(self.N*self.nui)  for _ in range(int(self.nq))]
        scxL_traj    = np.zeros((self.N+1,self.nxl))
        scuL_traj    = np.zeros((self.N,self.nul))
        scxC_traj    = [np.zeros((self.N+1,self.nxi)) for _ in range(int(self.nq))]
        scuC_traj    = [np.zeros((self.N,self.nui))  for _ in range(int(self.nq))]
        max_iter     = 10 # 5 for training
        e_tol        = 1e-2 # 1e-2 for training
        Opt_Sol1_l = []
        Opt_Sol1_cddp = []
        Opt_Sol1_c = []
        Opt_Sol2   = []
        Opt_Sol3   = []
        self.max_iter_ADMM = max_iter_ADMM
        # initial guess for Subproblem2 IPOPT
        scxl0   = Ref_xl
        scul0   = Ref_ul
        scxc0   = np.zeros((self.N+1)*int(self.nq)*self.nxi)
        scuc0   = np.zeros(self.N*int(self.nq)*self.nxi)
        for k in range(self.N):
            ref_xc_k    = np.zeros(int(self.nq)*self.nxi)
            for i in range(int(self.nq)):
                ref_xc_k[i*self.nxi:(i+1)*self.nxi]    = ref_xq[i][k*self.nxi:(k+1)*self.nxi]
            scxc0[k*int(self.nq)*self.nxi:(k+1)*int(self.nq)*self.nxi]    = ref_xc_k
            scuc0[k*int(self.nq)*self.nui:(k+1)*int(self.nq)*self.nui]    = ref_uq
        ref_xc_N      = np.zeros(int(self.nq)*self.nxi)
        for i in range(int(self.nq)):
            ref_xc_N[i*self.nxi:(i+1)*self.nxi]    = ref_xq[i][self.N*self.nxi:(self.N+1)*self.nxi]
        scxc0[self.N*int(self.nq)*self.nxi:(self.N+1)*int(self.nq)*self.nxi]    = ref_xc_N
        
        for i_admm in range(self.max_iter_ADMM):
            # solve Subproblem 1-load (dynamic)
            start_time = TM.time()
            # opt_sol = self.MPC_Load_Planning_SubP1(Paral)
            opt_sol_l = self.DDP_Load_ADMM_Subp1(xl_fb,Ref_xl,Ref_ul,paral,scxl_traj_tp,scul_traj_tp,scxL_traj_tp,scuL_traj_tp,max_iter,e_tol,i_admm)
            mpctime = (TM.time() - start_time)*1000
            print('ADMM_iteration=',i_admm+1,"subprblem1_load:--- %s ms ---" % format(mpctime,'.2f'))
            xl_traj = opt_sol_l['xl_traj']
            ul_traj = opt_sol_l['ul_traj']
            Kfbl_traj  = opt_sol_l['K_FB']
            xl_traj_tp = np.reshape(xl_traj,(self.N+1)*self.nxl)
            ul_traj_tp = np.reshape(ul_traj,self.N*self.nul)
            # solve Subproblem 1-cable (dynamic, across n cables)
            ParaC   = []
            for i in range(int(self.nq)):
                xi_fb  = xq_fb[i*self.nxi:(i+1)*self.nxi]
                ref_xi = ref_xq[i]
                ref_ui = ref_uq[i*self.nui:(i+1)*self.nui]
                scxi_traj = scxc_traj_tp[i]
                scxI_traj = scxC_traj_tp[i]
                scui_traj = scuc_traj_tp[i]
                scuI_traj = scuC_traj_tp[i]
                parai     = np.concatenate((xi_fb,ref_xi))
                parai     = np.concatenate((parai,ref_ui))
                parai     = np.concatenate((parai,scxi_traj))
                parai     = np.concatenate((parai,scxI_traj))
                parai     = np.concatenate((parai,scui_traj))
                parai     = np.concatenate((parai,scuI_traj))
                parai     = np.concatenate((parai,paraC))
                parai     = np.concatenate((parai,[i_admm]))
                ParaC  += [parai]
            start_time = TM.time()
            opt_solc, OPt_sol_c = self.MPC_Cable_DDP_Planning_SubP1(ParaC)
            mpctime = (TM.time() - start_time)*1000
            px_dis      = self.open_loop_penalty(paral[-4],paral[-2],i_admm,max_iter_ADMM)
            pu_dis      = self.open_loop_penalty(paral[-3],paral[-1],i_admm,max_iter_ADMM)
            pix_dis     = self.open_loop_penalty(paraC[-4],paraC[-2],i_admm,max_iter_ADMM)
            piu_dis     = self.open_loop_penalty(paraC[-3],paraC[-1],i_admm,max_iter_ADMM)
            print('ADMM_iteration=',i_admm+1,"subproblem1_cables:--- %s ms ---" % format(mpctime,'.2f'),'current_plx=',px_dis,'current_plu=',pu_dis,'current_pix=',pix_dis,'current_piu=',piu_dis)
            xc_traj  = opt_solc['xc_traj']
            uc_traj  = opt_solc['uc_traj']
            # solve Subproblem 2 (static, N independent steps, each step is a centralized problem)
            xc_traj_tp2 = np.zeros((self.N+1)*int(self.nq)*self.nxi)
            uc_traj_tp2 = np.zeros(self.N*int(self.nq)*self.nui)
            for k in range(self.N):
                xc_traj_k   = np.zeros(int(self.nq)*self.nxi)
                uc_traj_k   = np.zeros(int(self.nq)*self.nui)
                for i in range(int(self.nq)):
                    xc_traj_k[i*self.nxi:(i+1)*self.nxi] = xc_traj[i][k,:]
                    uc_traj_k[i*self.nui:(i+1)*self.nui] = uc_traj[i][k,:]
                xc_traj_tp2[k*int(self.nq)*self.nxi:(k+1)*int(self.nq)*self.nxi] = xc_traj_k
                uc_traj_tp2[k*int(self.nq)*self.nui:(k+1)*int(self.nq)*self.nui] = uc_traj_k
            xc_traj_N   = np.zeros(int(self.nq)*self.nxi)
            for i in range(int(self.nq)):
                xc_traj_N[i*self.nxi:(i+1)*self.nxi] = xc_traj[i][self.N,:]
            xc_traj_tp2[self.N*int(self.nq)*self.nxi:(self.N+1)*int(self.nq)*self.nxi] = xc_traj_N
            scxC_traj_tp2 = np.zeros((self.N+1)*int(self.nq)*self.nxi)
            ref_xc_tp2    = np.zeros((self.N+1)*int(self.nq)*self.nxi)
            scuC_traj_tp2 = np.zeros(self.N*int(self.nq)*self.nui)
            for k in range(self.N):
                scxC_traj_k = np.zeros(int(self.nq)*self.nxi)
                ref_xc_k    = np.zeros(int(self.nq)*self.nxi)
                scuC_traj_k = np.zeros(int(self.nq)*self.nui)
                for i in range(int(self.nq)):
                    scxC_traj_k[i*self.nxi:(i+1)*self.nxi] = scxC_traj_tp[i][k*self.nxi:(k+1)*self.nxi]
                    ref_xc_k[i*self.nxi:(i+1)*self.nxi]    = ref_xq[i][k*self.nxi:(k+1)*self.nxi]
                    scuC_traj_k[i*self.nui:(i+1)*self.nui] = scuC_traj_tp[i][k*self.nui:(k+1)*self.nui]
                scxC_traj_tp2[k*int(self.nq)*self.nxi:(k+1)*int(self.nq)*self.nxi] = scxC_traj_k
                ref_xc_tp2[k*int(self.nq)*self.nxi:(k+1)*int(self.nq)*self.nxi]    = ref_xc_k
                scuC_traj_tp2[k*int(self.nq)*self.nui:(k+1)*int(self.nq)*self.nui] = scuC_traj_k
            scxC_traj_N   = np.zeros(int(self.nq)*self.nxi)
            ref_xc_N      = np.zeros(int(self.nq)*self.nxi)
            for i in range(int(self.nq)):
                scxC_traj_N[i*self.nxi:(i+1)*self.nxi] = scxC_traj_tp[i][self.N*self.nxi:(self.N+1)*self.nxi]
                ref_xc_N[i*self.nxi:(i+1)*self.nxi]    = ref_xq[i][self.N*self.nxi:(self.N+1)*self.nxi]
            scxC_traj_tp2[self.N*int(self.nq)*self.nxi:(self.N+1)*int(self.nq)*self.nxi] = scxC_traj_N
            ref_xc_tp2[self.N*int(self.nq)*self.nxi:(self.N+1)*int(self.nq)*self.nxi]    = ref_xc_N
            Para2_cable = np.concatenate((scxl0,scul0))
            Para2_cable = np.concatenate((Para2_cable,scxc0))
            Para2_cable = np.concatenate((Para2_cable,ref_uq))
            Para2_cable = np.concatenate((Para2_cable,xl_traj_tp))
            Para2_cable = np.concatenate((Para2_cable,scxL_traj_tp))
            Para2_cable = np.concatenate((Para2_cable,ul_traj_tp))
            Para2_cable = np.concatenate((Para2_cable,scuL_traj_tp))
            Para2_cable = np.concatenate((Para2_cable,xc_traj_tp2))
            Para2_cable = np.concatenate((Para2_cable,scxC_traj_tp2))
            Para2_cable = np.concatenate((Para2_cable,uc_traj_tp2))
            Para2_cable = np.concatenate((Para2_cable,scuC_traj_tp2))
            Para2_cable = np.concatenate((Para2_cable,paral))
            Para2_cable = np.concatenate((Para2_cable,paraC))
            Para2_cable = np.concatenate((Para2_cable,[i_admm]))
            start_time  = TM.time()
            opt_sol2    = self.ADMM_SubP2(Para2_cable)
            mpctime = (TM.time() - start_time)*1000
            print("subproblem2:--- %s ms ---" % format(mpctime,'.2f'))
            scxl_traj   = opt_sol2['scxl_traj']
            scul_traj   = opt_sol2['scul_traj']
            scxc_traj   = opt_sol2['scxc_traj']
            scuc_traj   = opt_sol2['scuc_traj']
            # solve Subproblem 3
            opt_sol3    = self.ADMM_SubP3(xl_traj,scxl_traj,scxL_traj,ul_traj,scul_traj,scuL_traj,xc_traj,scxc_traj,scxC_traj,uc_traj,scuc_traj,scuC_traj,paral[-4],paral[-3],paral[-2],paral[-1],paraC[-4],paraC[-3],paraC[-2],paraC[-1],max_iter_ADMM,i_admm)
            scxL_traj   = opt_sol3['scxL_traj_new']
            scuL_traj   = opt_sol3['scuL_traj_new']
            scxC_traj   = opt_sol3['scxC_traj_new']
            scuC_traj   = opt_sol3['scuC_traj_new']
            # update trajectories
            scxl_traj_tp_new  = np.reshape(scxl_traj,(self.N+1)*self.nxl) # for Subproblem 1-load
            scul_traj_tp_new  = np.reshape(scul_traj,self.N*self.nul) # for Subproblem 1-load
            scxL_traj_tp  = np.reshape(scxL_traj,(self.N+1)*self.nxl) # for Subproblem 1-load and 2
            scuL_traj_tp  = np.reshape(scuL_traj,self.N*self.nul) # for Subproblem 1-load and 2
            xc_traj_tp    = [np.zeros((self.N+1)*self.nxi)  for _ in range(int(self.nq))] # for Subproblem 2-cable and 3
            uc_traj_tp    = [np.zeros(self.N*self.nui)  for _ in range(int(self.nq))]     # for Subproblem 2-cable and 3
            scxc_traj_tp_new  = [np.zeros((self.N+1)*self.nxi)  for _ in range(int(self.nq))] # for Subproblem 1-cable
            scxC_traj_tp  = [np.zeros((self.N+1)*self.nxi)  for _ in range(int(self.nq))] # for Subproblem 1-cable and 2
            scuc_traj_tp_new  = [np.zeros(self.N*self.nui)  for _ in range(int(self.nq))] # for Subproblem 1-cable
            scuC_traj_tp  = [np.zeros(self.N*self.nui)  for _ in range(int(self.nq))] # for Subproblem 1-cable and 2
            r_xc = [] # primal residual of cables' states
            r_uc = [] # primal residual of cables' controls
            s_xc = [] # dual residual of cables' states
            s_uc = [] # dual residual of cables' contorls
            for i in range(int(self.nq)):
                for k in range(self.N):
                    xc_traj_tp[i][k*self.nxi:(k+1)*self.nxi]       = xc_traj[i][k,:]
                    uc_traj_tp[i][k*self.nui:(k+1)*self.nui]       = uc_traj[i][k,:]
                    scxc_traj_tp_new[i][k*self.nxi:(k+1)*self.nxi] = scxc_traj[i][k,:]
                    scxC_traj_tp[i][k*self.nxi:(k+1)*self.nxi]     = scxC_traj[i][k,:]
                    scuc_traj_tp_new[i][k*self.nui:(k+1)*self.nui] = scuc_traj[i][k,:]
                    scuC_traj_tp[i][k*self.nui:(k+1)*self.nui]     = scuC_traj[i][k,:]
                xc_traj_tp[i][self.N*self.nxi:(self.N+1)*self.nxi]       = xc_traj[i][self.N,:]
                scxc_traj_tp_new[i][self.N*self.nxi:(self.N+1)*self.nxi] = scxc_traj[i][self.N,:]
                scxC_traj_tp[i][self.N*self.nxi:(self.N+1)*self.nxi]     = scxC_traj[i][self.N,:]
                r_xc += [LA.norm(xc_traj_tp[i]-scxc_traj_tp_new[i])]
                s_xc += [parai[-4]*LA.norm(scxc_traj_tp_new[i]-scxc_traj_tp[i])]
                r_uc += [LA.norm(uc_traj_tp[i]-scuc_traj_tp_new[i])]
                s_uc += [parai[-3]*LA.norm(scuc_traj_tp_new[i]-scuc_traj_tp[i])]
            # update the initial guess
            scxl0 = scxl_traj_tp_new 
            scul0 = scul_traj_tp_new
            scxc0 = np.zeros((self.N+1)*int(self.nq)*self.nxi)
            scuc0 = np.zeros((self.N)*int(self.nq)*self.nui)
            for k in range(self.N):
                scxc_k = np.zeros(int(self.nq)*self.nxi)
                scuc_k = np.zeros(int(self.nq)*self.nui)
                for i in range(int(self.nq)):
                    scxc_k[i*self.nxi:(i+1)*self.nxi]=scxc_traj[i][k,:]
                    scuc_k[i*self.nui:(i+1)*self.nui]=scuc_traj[i][k,:]
                scxc0[k*self.nxi*int(self.nq):(k+1)*self.nxi*int(self.nq)]=scxc_k
                scuc0[k*self.nui*int(self.nq):(k+1)*self.nui*int(self.nq)]=scuc_k

            # residuals
            
            # print('ADMM_iteration=',i_ADMM,'p=',paral[-1],'r_xl=',r_xl,'r_ul=',r_ul,'s_xl=',s_xl,'s_ul=',s_ul)
            # for i in range(int(self.nq)):
            #     print('ADMM_iteration=',i_ADMM,'r_xc_'+str(i+1)+'=',r_xc[i],'s_xc_'+str(i+1)+'=',s_xc[i])
            #     print('ADMM_iteration=',i_ADMM,'r_uc_'+str(i+1)+'=',r_uc[i],'s_uc_'+str(i+1)+'=',s_uc[i])
            # update
            scxl_traj_tp = scxl_traj_tp_new
            scul_traj_tp = scul_traj_tp_new
            scxc_traj_tp = scxc_traj_tp_new
            scuc_traj_tp = scuc_traj_tp_new

            Opt_Sol1_l += [opt_sol_l]
            Opt_Sol1_cddp += [OPt_sol_c]
            Opt_Sol1_c += [opt_solc]
            Opt_Sol2   += [opt_sol2]
            Opt_Sol3   += [opt_sol3]
        
        opt_sol = {"xl_traj":xl_traj,
                   "ul_traj":ul_traj,
                   "Kfbl_traj":Kfbl_traj,
                   "scxl_traj":scxl_traj,
                   "scul_traj":scul_traj,
                   "xc_traj":xc_traj,
                   "uc_traj":uc_traj,
                   "scxc_traj":scxc_traj,
                   "scuc_traj":scuc_traj
                    }
        
        return opt_sol, Opt_Sol1_l, Opt_Sol1_cddp, Opt_Sol1_c, Opt_Sol2, Opt_Sol3
    

            
    def DDP_Load_Gradient(self,opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, px,pu, gammax,gammau,ADMM_max, i_admm):
        Quuinv, Qxu, K_fb, F, G  = opt_sol['Quu_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSysl['HxNp'], auxSysl['Hxp'], auxSysl['Hup']
        px_dis     = self.open_loop_penalty(px,gammax,i_admm,ADMM_max)
        pu_dis     = self.open_loop_penalty(pu,gammau,i_admm,ADMM_max)
        S          = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        S[self.N]  = HxNp + scxL_grad[self.N] - px_dis*scxl_grad[self.N] # reduced to HxNp only in the single-agent problem
        v_FF       = self.N*[np.zeros((self.nul,self.n_Pauto))]
        xl_grad    = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))] 
        ul_grad    = self.N*[np.zeros((self.nul,self.n_Pauto))]
        #-------Backward recursion-------#         
        for k in reversed(range(self.N)): # N-1, N-2, ....., 0
            Hxp_k    = Hxp[k] + scxL_grad[k] - px_dis*scxl_grad[k]
            Hup_k    = Hup[k] + scuL_grad[k] - pu_dis*scul_grad[k]
            v_FF[k]  = -Quuinv[k]@(Hup_k + G[k].T@S[k+1])
            S[k]     = Hxp_k + F[k].T@S[k+1] + Qxu[k]@v_FF[k] # s[0] not used
        #-------Foreward recursion-------#
        for k in range(self.N):
            ul_grad[k]  = K_fb[k]@xl_grad[k]+v_FF[k]
            xl_grad[k+1]= F[k]@xl_grad[k]+G[k]@ul_grad[k]

        grad_outl ={"xl_grad":xl_grad,
                   "ul_grad":ul_grad
                }
        
        return grad_outl
    
    
    def Cao_Load_Gradient_s(self,opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, px,pu, gammax,gammau,ADMM_max,i_admm):
        Quuinv, Qxu, K_fb, F, G  = opt_sol['Quu_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSysl['HxNp'], auxSysl['Hxp'], auxSysl['Hup']
        px_dis     = self.open_loop_penalty(px,gammax,i_admm,ADMM_max)
        pu_dis     = self.open_loop_penalty(pu,gammau,i_admm,ADMM_max)
        S           = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))] # Vxp
        Vpp         = (self.N+1)*[np.zeros((self.n_Pauto,self.n_Pauto))]
        S[self.N]   = HxNp + scxL_grad[self.N] - px_dis*scxl_grad[self.N]
        Vpp[self.N] = np.zeros((self.n_Pauto,self.n_Pauto))
        v_FF        = self.N*[np.zeros((self.nul,self.n_Pauto))]
        xl_grad     = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))] 
        p_grad      = (self.N+1)*[np.identity(self.n_Pauto)]
        ul_grad     = self.N*[np.zeros((self.nul,self.n_Pauto))]
        
        #-------Backward recursion-------#         
        for k in reversed(range(self.N)): # N-1, N-2,...,0
            Hpp_k    = np.zeros((self.n_Pauto,self.n_Pauto))
            Hxp_k    = Hxp[k] + scxL_grad[k] - px_dis*scxl_grad[k]
            Hup_k    = Hup[k] + scuL_grad[k] - pu_dis*scul_grad[k]
            v_FF[k]  = -Quuinv[k]@(Hup_k + G[k].T@S[k+1])
            Vpp[k]   = Hpp_k + Vpp[k+1] + (Hup_k + G[k].T@S[k+1]).T@v_FF[k] # the augmented Riccati recursion, which is redundant
            S[k]     = Hxp_k + F[k].T@S[k+1] + Qxu[k]@v_FF[k] # s[0] not used
        #-------Foreward recursion-------#
        for k in range(self.N):
            ul_grad[k]  = K_fb[k]@xl_grad[k]+v_FF[k]@p_grad[k] # expanding the augmented control law gives this form, which is exactly the same as ours
            xl_grad[k+1]= F[k]@xl_grad[k]+G[k]@ul_grad[k]
            p_grad[k+1] = p_grad[k] # the augmented dynamics, which is redundant
        
        grad_out_cao ={"xl_grad":xl_grad,
                   "ul_grad":ul_grad,
                   "p_grad":p_grad
                }
        
        return grad_out_cao
    
    def Cao_Load_Gradient(self,opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, px,pu,gammax,gammau,ADMM_max,i_admm):
        # solve the augmented optimal problem using one-step DDP recursion
        Hxx, Hxu, Huu, F, G  = opt_sol['Hxx'], opt_sol['Hxu'], opt_sol['Huu'], opt_sol['Fx'], opt_sol['Fu']
        HxxN, HxNp, Hxp, Hup = auxSysl['HxxN'], auxSysl['HxNp'], auxSysl['Hxp'], auxSysl['Hup']
        # Vyy      = (self.N+1)*[np.zeros((self.n_Pauto+self.nxl,self.n_Pauto+self.nxl))] # a large matrix, leading to significant computation cost
        # we decompose Vyy into four smaller blocks
        px_dis     = self.open_loop_penalty(px,gammax,i_admm,ADMM_max)
        pu_dis     = self.open_loop_penalty(pu,gammau,i_admm,ADMM_max)
        Vpp         = (self.N+1)*[np.zeros((self.n_Pauto,self.n_Pauto))]
        Vpx         = (self.N+1)*[np.zeros((self.n_Pauto,self.nxl))]
        Vxp         = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        Vxx         = (self.N+1)*[np.zeros((self.nxl,self.nxl))]
        # Kfb_y    = self.N*[np.zeros((self.nul,self.n_Pauto+self.n_xl))] # augmented feedback gain
        Kfb_p       = self.N*[np.zeros((self.nul,self.n_Pauto))] # this matches exactly the feedforward gain!
        Kfb_x       = self.N*[np.zeros((self.nul,self.nxl))]    # this is the feedback gain
        p_grad      = (self.N+1)*[np.identity(self.n_Pauto)]
        xl_grad     = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))] 
        ul_grad     = self.N*[np.zeros((self.nul,self.n_Pauto))]
        # Vyy[self.N] = vertcat(
        #                         horzcat(np.zeros((self.n_Pauto,self.n_Pauto)),HxNp.T),
        #                         horzcat(HxNp,self.lxxN_fn(P1l0=weight1)['lxxNf'].full())
        #                     )
        Vpp[self.N] = np.zeros((self.n_Pauto,self.n_Pauto))
        Vpx[self.N] = (HxNp + scxL_grad[self.N] - px_dis*scxl_grad[self.N]).T
        Vxp[self.N] = HxNp + scxL_grad[self.N] - px_dis*scxl_grad[self.N]
        Vxx[self.N] = HxxN
        
        for k in reversed(range(self.N)):
            # Hyy_k   = vertcat(
            #     horzcat(np.zeros((self.n_Pauto,self.n_Pauto)),(Hxp[k]+ scxL_grad[k] - p1*scxl_grad[k]).T),
            #     horzcat((Hxp[k]+ scxL_grad[k] - p1*scxl_grad[k]),Hxx[k])
            # )
            # F_bar   = vertcat(
            #     horzcat(np.identity(self.n_Pauto),np.zeros((self.n_Pauto,self.n_xl))),
            #     horzcat(np.zeros((self.n_Pauto,self.n_xl)).T,F[k])
            # )
            # G_bar   = vertcat(np.zeros((self.n_Pauto,self.n_Wl)),G[k])
            # Huy_k   = horzcat((Hup[k]+ scWL_grad[k] - p1*scWl_grad[k]),Hxu[k].T)
            # Qyy_k   = Hyy_k + F_bar.T@Vyy[k+1]@F_bar
            # Quy_k   = Huy_k + G_bar.T@Vyy[k+1]@F_bar
            # Quu_k   = Huu[k] + G_bar.T@Vyy[k+1]@G_bar
            # Kfb_y[k]=-LA.inv(Quu_k)@Quy_k
            # Vyy[k]  = Qyy_k + Quy_k.T@Kfb_y[k]
            # Hpp_k    = np.zeros((self.n_Pauto,self.n_Pauto))
            Hpx_k    = (Hxp[k]+ scxL_grad[k] - px_dis*scxl_grad[k]).T
            # Hxp_k    = Hxp[k]+ scxL_grad[k] - dis_rn*p1*scxl_grad[k]
            Hxx_k    = Hxx[k]
            Hup_k    = Hup[k]+ scuL_grad[k] - pu_dis*scul_grad[k]
            Quu_k    = Huu[k]+G[k].T@Vxx[k+1]@G[k] 
            invQuu_k = LA.inv(Quu_k)
            Kfb_p[k] = -invQuu_k@(Hup_k+G[k].T@Vxp[k+1]) 
            Kfb_x[k] = -invQuu_k@(Hxu[k].T+G[k].T@Vxx[k+1]@F[k])
            Vpp[k]   = Vpp[k+1] + (Hup_k.T+Vpx[k+1]@G[k])@Kfb_p[k]
            Vpx[k]   = Hpx_k + Vpx[k+1]@F[k] + Kfb_p[k].T@(Hxu[k]+F[k].T@Vxx[k+1]@G[k]).T
            # Vxp[k]   = Hxp_k + F[k].T@Vxp[k+1] + (Hxu[k]+F[k].T@Vxx[k+1]@G[k])@Kfb_p[k]
            Vxp[k]   = Vpx[k].T
            Vxx[k]   = Hxx_k + F[k].T@Vxx[k+1]@F[k] + (Hxu[k]+F[k].T@Vxx[k+1]@G[k])@Kfb_x[k]

        for k in range(self.N):
            ul_grad[k]   = Kfb_p[k]@p_grad[k] + Kfb_x[k]@xl_grad[k]
            xl_grad[k+1] = F[k]@xl_grad[k]+G[k]@ul_grad[k]
            p_grad[k+1]  = p_grad[k]
        grad_out_cao ={"xl_grad":xl_grad,
                   "ul_grad":ul_grad,
                   "p_grad":p_grad
                }
        
        return grad_out_cao
    

    def PDP_Load_Gradient(self,opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, px,pu, gammax,gammau,ADMM_max,i_admm):
        Hxx, Hxu, Huu, F, G  = opt_sol['Hxx'], opt_sol['Hxu'], opt_sol['Huu'], opt_sol['Fx'], opt_sol['Fu']
        HxxN, HxNp, Hxp, Hup = auxSysl['HxxN'], auxSysl['HxNp'], auxSysl['Hxp'], auxSysl['Hup']
        px_dis      = self.open_loop_penalty(px,gammax,i_admm,ADMM_max)
        pu_dis      = self.open_loop_penalty(pu,gammau,i_admm,ADMM_max)
        P           = (self.N+1)*[np.zeros((self.nxl,self.nxl))]
        S           = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        A           = self.N*[np.zeros((self.nxl,self.nxl))]
        R           = self.N*[np.zeros((self.nxl,self.nxl))]
        M_p         = self.N*[np.zeros((self.nxl,self.n_Pauto))]
        invHuu      = self.N*[np.zeros((self.nul,self.nul))]
        PinvIRP     = self.N*[np.zeros((self.nxl,self.nxl))]
        P[self.N]   = HxxN
        S[self.N]   = HxNp  + scxL_grad[self.N] - px_dis*scxl_grad[self.N]
        xl_grad     = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))] 
        ul_grad     = self.N*[np.zeros((self.nul,self.n_Pauto))]
        I           = np.identity(self.nxl)
        Iu          = np.identity(self.nul)
        for k in reversed(range(self.N)):# N-1, N-2,...,0
            P_next      = P[k+1]
            S_next      = S[k+1]
            invHuu[k]   = LA.inv(Huu[k])
            GinvHuu     = G[k]@invHuu[k]
            HxuinvHuu   = Hxu[k]@invHuu[k]
            A[k]        = F[k]-GinvHuu@Hxu[k].T
            R[k]        = GinvHuu@G[k].T
            M_p[k]      = -GinvHuu@(Hup[k] + scuL_grad[k] - pu_dis*scul_grad[k])
            Q_k         = Hxx[k]-HxuinvHuu@Hxu[k].T
            N_p_k       = Hxp[k]+ scxL_grad[k] - px_dis*scxl_grad[k] - HxuinvHuu@(Hup[k] + scuL_grad[k] - pu_dis*scul_grad[k])
            PinvIRP[k]  = P_next@LA.inv(I+R[k]@P_next)
            P_curr      = Q_k + A[k].T@PinvIRP[k]@A[k]
            S_curr      = A[k].T@PinvIRP[k]@(M_p[k] - R[k]@S_next) + A[k].T@S_next + N_p_k
            P[k]        = P_curr
            S[k]        = S_curr
        
        for k in range(self.N):
            ul_grad[k]  = -invHuu[k]@((Hxu[k].T+G[k].T@PinvIRP[k]@A[k])@xl_grad[k] + G[k].T@PinvIRP[k]@(M_p[k]- R[k]@ S[k+1]) + G[k].T@S[k+1] + (Hup[k] + scuL_grad[k] - pu_dis*scul_grad[k]))
            xl_grad[k+1] = F[k]@xl_grad[k] + G[k]@ul_grad[k]

        grad_out ={"xl_grad":xl_grad,
                   "ul_grad":ul_grad
                }
        
        return grad_out
    

    
    def DDP_Cable_Gradient(self,opt_sol,auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, pix,piu, gammaix,gammaiu,ADMM_max, i_admm):
        Quuinv, Qxu, K_fb, F, G  = opt_sol['Quu_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSysi['HxNp'], auxSysi['Hxp'], auxSysi['Hup']
        pix_dis     = self.open_loop_penalty(pix,gammaix,i_admm,ADMM_max)
        piu_dis     = self.open_loop_penalty(piu,gammaiu,i_admm,ADMM_max)
        S           = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))]
        S[self.N]   = HxNp + scxI_grad[self.N] - pix_dis*scxi_grad[self.N] # reduced to HxNp only in the single-agent problem
        v_FF        = self.N*[np.zeros((self.nui,self.n_Pauto))]
        xi_grad     = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))] 
        ui_grad     = self.N*[np.zeros((self.nui,self.n_Pauto))]
        #-------Backward recursion-------#         
        for k in reversed(range(self.N)): 
            Hxp_k    = Hxp[k] + scxI_grad[k] - pix_dis*scxi_grad[k]
            Hup_k    = Hup[k] + scuI_grad[k] - piu_dis*scui_grad[k]
            v_FF[k]  = -Quuinv[k]@(Hup_k + G[k].T@S[k+1])
            S[k]     = Hxp_k + F[k].T@S[k+1] + Qxu[k]@v_FF[k] # s[0] not used
        #-------Foreward recursion-------#
        for k in range(self.N):
            ui_grad[k]  = K_fb[k]@xi_grad[k]+v_FF[k]
            xi_grad[k+1]= F[k]@xi_grad[k]+G[k]@ui_grad[k]

        grad_outi ={"xi_grad":xi_grad,
                   "ui_grad":ui_grad
                }
        
        return grad_outi
    
    def Cao_Cable_Gradient_s(self,opt_sol,auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, pix,piu, gammaix,gammaiu,ADMM_max,i_admm):
        Quuinv, Qxu, K_fb, F, G  = opt_sol['Quu_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSysi['HxNp'], auxSysi['Hxp'], auxSysi['Hup']
        pix_dis     = self.open_loop_penalty(pix,gammaix,i_admm,ADMM_max)
        piu_dis     = self.open_loop_penalty(piu,gammaiu,i_admm,ADMM_max)
        S           = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))] # Vxp
        Vpp         = (self.N+1)*[np.zeros((self.n_Pauto,self.n_Pauto))]
        S[self.N]   = HxNp + scxI_grad[self.N] - pix_dis*scxi_grad[self.N]
        Vpp[self.N] = np.zeros((self.n_Pauto,self.n_Pauto))
        v_FF        = self.N*[np.zeros((self.nui,self.n_Pauto))]
        xi_grad     = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))] 
        p_grad      = (self.N+1)*[np.identity(self.n_Pauto)]
        ui_grad     = self.N*[np.zeros((self.nui,self.n_Pauto))]
        
        #-------Backward recursion-------#         
        for k in reversed(range(self.N)): # N-1, N-2,...,0
            Hpp_k    = np.zeros((self.n_Pauto,self.n_Pauto))
            Hxp_k    = Hxp[k] + scxI_grad[k] - pix_dis*scxi_grad[k]
            Hup_k    = Hup[k] + scuI_grad[k] - piu_dis*scui_grad[k]
            v_FF[k]  = -Quuinv[k]@(Hup_k + G[k].T@S[k+1])
            Vpp[k]   = Hpp_k + Vpp[k+1] + (Hup_k + G[k].T@S[k+1]).T@v_FF[k] # the augmented Riccati recursion, which is redundant
            S[k]     = Hxp_k + F[k].T@S[k+1] + Qxu[k]@v_FF[k] # s[0] not used
        #-------Foreward recursion-------#
        for k in range(self.N):
            ui_grad[k]  = K_fb[k]@xi_grad[k]+v_FF[k]@p_grad[k] # expanding the augmented control law gives this form, which is exactly the same as ours
            xi_grad[k+1]= F[k]@xi_grad[k]+G[k]@ui_grad[k]
            p_grad[k+1] = p_grad[k] # the augmented dynamics, which is redundant
        
        grad_out_cao ={"xi_grad":xi_grad,
                   "ui_grad":ui_grad,
                   "p_grad":p_grad
                }
        
        return grad_out_cao

    def Cao_Cable_Gradient(self,opt_sol,auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, pix,piu,gammaix,gammaiu,ADMM_max,i_admm):
        # solve the augmented optimal problem using one-step DDP recursion
        Hxx, Hxu, Huu, F, G  = opt_sol['Hxx'], opt_sol['Hxu'], opt_sol['Huu'], opt_sol['Fx'], opt_sol['Fu']
        HxxN, HxNp, Hxp, Hup = auxSysi['HxxN'], auxSysi['HxNp'], auxSysi['Hxp'], auxSysi['Hup']
        # Vyy      = (self.N+1)*[np.zeros((self.n_Pauto+self.nxl,self.n_Pauto+self.nxl))] # a large matrix, leading to significant computation cost
        # we decompose Vyy into four smaller blocks
        pix_dis     = self.open_loop_penalty(pix,gammaix,i_admm,ADMM_max)
        piu_dis     = self.open_loop_penalty(piu,gammaiu,i_admm,ADMM_max)
        Vpp         = (self.N+1)*[np.zeros((self.n_Pauto,self.n_Pauto))]
        Vpx         = (self.N+1)*[np.zeros((self.n_Pauto,self.nxi))]
        Vxp         = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))]
        Vxx         = (self.N+1)*[np.zeros((self.nxi,self.nxi))]
        # Kfb_y    = self.N*[np.zeros((self.nul,self.n_Pauto+self.n_xl))] # augmented feedback gain
        Kfb_p       = self.N*[np.zeros((self.nui,self.n_Pauto))] # this matches exactly the feedforward gain!
        Kfb_x       = self.N*[np.zeros((self.nui,self.nxi))]    # this is the feedback gain
        p_grad      = (self.N+1)*[np.identity(self.n_Pauto)]
        xi_grad     = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))] 
        ui_grad     = self.N*[np.zeros((self.nui,self.n_Pauto))]
        # Vyy[self.N] = vertcat(
        #                         horzcat(np.zeros((self.n_Pauto,self.n_Pauto)),HxNp.T),
        #                         horzcat(HxNp,self.lxxN_fn(P1l0=weight1)['lxxNf'].full())
        #                     )
        Vpp[self.N] = np.zeros((self.n_Pauto,self.n_Pauto))
        Vpx[self.N] = (HxNp + scxI_grad[self.N] - pix_dis*scxi_grad[self.N]).T
        Vxp[self.N] = HxNp + scxI_grad[self.N] - pix_dis*scxi_grad[self.N]
        Vxx[self.N] = HxxN
        
       
        for k in reversed(range(self.N)):
            # Hyy_k   = vertcat(
            #     horzcat(np.zeros((self.n_Pauto,self.n_Pauto)),(Hxp[k]+ scxL_grad[k] - p1*scxl_grad[k]).T),
            #     horzcat((Hxp[k]+ scxL_grad[k] - p1*scxl_grad[k]),Hxx[k])
            # )
            # F_bar   = vertcat(
            #     horzcat(np.identity(self.n_Pauto),np.zeros((self.n_Pauto,self.n_xl))),
            #     horzcat(np.zeros((self.n_Pauto,self.n_xl)).T,F[k])
            # )
            # G_bar   = vertcat(np.zeros((self.n_Pauto,self.n_Wl)),G[k])
            # Huy_k   = horzcat((Hup[k]+ scWL_grad[k] - p1*scWl_grad[k]),Hxu[k].T)
            # Qyy_k   = Hyy_k + F_bar.T@Vyy[k+1]@F_bar
            # Quy_k   = Huy_k + G_bar.T@Vyy[k+1]@F_bar
            # Quu_k   = Huu[k] + G_bar.T@Vyy[k+1]@G_bar
            # Kfb_y[k]=-LA.inv(Quu_k)@Quy_k
            # Vyy[k]  = Qyy_k + Quy_k.T@Kfb_y[k]
            # Hpp_k    = np.zeros((self.n_Pauto,self.n_Pauto))
            Hpx_k    = (Hxp[k]+ scxI_grad[k] - pix_dis*scxi_grad[k]).T
            # Hxp_k    = Hxp[k]+ scxL_grad[k] - dis_rn*p1*scxl_grad[k]
            Hxx_k    = Hxx[k]
            Hup_k    = Hup[k]+ scuI_grad[k] - piu_dis*scui_grad[k]
            Quu_k    = Huu[k]+G[k].T@Vxx[k+1]@G[k]
            invQuu_k = LA.inv(Quu_k)
            Kfb_p[k] = -invQuu_k@(Hup_k+G[k].T@Vxp[k+1]) 
            Kfb_x[k] = -invQuu_k@(Hxu[k].T+G[k].T@Vxx[k+1]@F[k])
            Vpp[k]   = Vpp[k+1] + (Hup_k.T+Vpx[k+1]@G[k])@Kfb_p[k]
            Vpx[k]   = Hpx_k + Vpx[k+1]@F[k] + Kfb_p[k].T@(Hxu[k]+F[k].T@Vxx[k+1]@G[k]).T
            # Vxp[k]   = Hxp_k + F[k].T@Vxp[k+1] + (Hxu[k]+F[k].T@Vxx[k+1]@G[k])@Kfb_p[k]
            Vxp[k]   = Vpx[k].T
            Vxx[k]   = Hxx_k + F[k].T@Vxx[k+1]@F[k] + (Hxu[k]+F[k].T@Vxx[k+1]@G[k])@Kfb_x[k]

        for k in range(self.N):
            ui_grad[k]   = Kfb_p[k]@p_grad[k] + Kfb_x[k]@xi_grad[k]
            xi_grad[k+1] = F[k]@xi_grad[k]+G[k]@ui_grad[k]
            p_grad[k+1]  = p_grad[k]
        grad_out_cao ={"xi_grad":xi_grad,
                   "ui_grad":ui_grad,
                   "p_grad":p_grad
                }
        
        return grad_out_cao
    

    
    def PDP_Cable_Gradient(self,opt_sol,auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, pix,piu, gammaix,gammaiu,ADMM_max,i_admm):
        Hxx, Hxu, Huu, F, G  = opt_sol['Hxx'], opt_sol['Hxu'], opt_sol['Huu'], opt_sol['Fx'], opt_sol['Fu']
        HxxN, HxNp, Hxp, Hup = auxSysi['HxxN'], auxSysi['HxNp'], auxSysi['Hxp'], auxSysi['Hup']
        pix_dis     = self.open_loop_penalty(pix,gammaix,i_admm,ADMM_max)
        piu_dis     = self.open_loop_penalty(piu,gammaiu,i_admm,ADMM_max)
        P           = (self.N+1)*[np.zeros((self.nxi,self.nxi))]
        S           = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))]
        A           = self.N*[np.zeros((self.nxi,self.nxi))]
        R           = self.N*[np.zeros((self.nxi,self.nxi))]
        M_p         = self.N*[np.zeros((self.nxi,self.n_Pauto))]
        invHuu      = self.N*[np.zeros((self.nui,self.nui))]
        PinvIRP     = self.N*[np.zeros((self.nxi,self.nxi))]
        P[self.N]   = HxxN
        S[self.N]   = HxNp  + scxI_grad[self.N] - pix_dis*scxi_grad[self.N]
        xi_grad     = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))] 
        ui_grad     = self.N*[np.zeros((self.nui,self.n_Pauto))]
        I           = np.identity(self.nxi)
        for k in reversed(range(self.N)):
            P_next      = P[k+1]
            S_next      = S[k+1]
            invHuu[k]   = LA.inv(Huu[k])
            GinvHuu     = G[k]@invHuu[k]
            HxuinvHuu   = Hxu[k]@invHuu[k]
            A[k]        = F[k]-GinvHuu@Hxu[k].T
            R[k]        = GinvHuu@G[k].T
            M_p[k]      = -GinvHuu@(Hup[k] + scuI_grad[k] - piu_dis*scui_grad[k])
            Q_k         = Hxx[k]-HxuinvHuu@Hxu[k].T
            N_p_k       = Hxp[k]+ scxI_grad[k] - pix_dis*scxi_grad[k] - HxuinvHuu@(Hup[k] + scuI_grad[k] - piu_dis*scui_grad[k])
            PinvIRP[k]  = P_next@LA.inv(I+R[k]@P_next)
            P_curr      = Q_k + A[k].T@PinvIRP[k]@A[k]
            S_curr      = A[k].T@PinvIRP[k]@(M_p[k] - R[k]@S_next) + A[k].T@S_next + N_p_k
            P[k]        = P_curr
            S[k]        = S_curr
        
        for k in range(self.N):
            ui_grad[k]  = -invHuu[k]@((Hxu[k].T+G[k].T@PinvIRP[k]@A[k])@xi_grad[k] + G[k].T@PinvIRP[k]@(M_p[k]- R[k]@ S[k+1]) + G[k].T@S[k+1] + (Hup[k] + scuI_grad[k] - piu_dis*scui_grad[k]))
            xi_grad[k+1] = F[k]@xi_grad[k] + G[k]@ui_grad[k]

        grad_out ={"xi_grad":xi_grad,
                   "ui_grad":ui_grad
                }
        
        return grad_out
    
    

    def SubP2_Gradient(self,auxSys2,grad_outl,grad_outc,scxL_grad,scuL_grad,scxC_grad,scuC_grad,px,pu,gammax,gammau,pix,piu,gammaix,gammaiu,ADMM_max,i_admm):
        xl_grad      = grad_outl['xl_grad']
        ul_grad      = grad_outl['ul_grad']
        Lscxlscxl    = auxSys2['Lscxlscxl']
        Lscxlscul    = auxSys2['Lscxlscul']
        Lscxlscxc    = auxSys2['Lscxlscxc']
        Lscxlscuc    = auxSys2['Lscxlscuc']
        Lsculscul    = auxSys2['Lsculscul']
        Lsculscxc    = auxSys2['Lsculscxc']
        Lsculscuc    = auxSys2['Lsculscuc']
        Lscxcscxc    = auxSys2['Lscxcscxc']
        Lscxcscuc    = auxSys2['Lscxcscuc']
        Lscucscuc    = auxSys2['Lscucscuc']
        Lscxlp       = auxSys2['Lscxlp']
        Lsculp       = auxSys2['Lsculp']
        Lscxcp       = auxSys2['Lscxcp']
        Lscucp       = auxSys2['Lscucp']
        # hessians of the original Lagrangian
        Lscxlscxl_o  = auxSys2['Lscxlscxl_o']
        Lscxlscul_o  = auxSys2['Lscxlscul_o']
        Lscxlscxc_o  = auxSys2['Lscxlscxc_o']
        Lscxlscuc_o  = auxSys2['Lscxlscuc_o']
        Lsculscul_o  = auxSys2['Lsculscul_o']
        Lsculscxc_o  = auxSys2['Lsculscxc_o']
        Lsculscuc_o  = auxSys2['Lsculscuc_o']
        Lscxcscxc_o  = auxSys2['Lscxcscxc_o']
        Lscxcscuc_o  = auxSys2['Lscxcscuc_o']
        Lscucscuc_o  = auxSys2['Lscucscuc_o']
        I_hessian    = np.identity(self.nxl+self.nul+(self.nxi+self.nui)*int(self.nq))
        I_hess2      = np.identity(self.nxl+self.nxi*int(self.nq))
        scxl_grad    = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        scul_grad    = self.N*[np.zeros((self.nul,self.n_Pauto))]
        scxc_grad    = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.n_Pauto))]
        scuc_grad    = self.N*[np.zeros((self.nui*int(self.nq),self.n_Pauto))]
        MIN_eigen    = []
        px_dis       = self.open_loop_penalty(px,gammax,i_admm,ADMM_max)
        pu_dis       = self.open_loop_penalty(pu,gammau,i_admm,ADMM_max)
        pix_dis      = self.open_loop_penalty(pix,gammaix,i_admm,ADMM_max)
        piu_dis      = self.open_loop_penalty(piu,gammaiu,i_admm,ADMM_max)
        EigTime      = 0
        for k in range(self.N):
            L_hessian_k = vertcat(
                                horzcat(Lscxlscxl[k],   Lscxlscul[k],   Lscxlscxc[k],   Lscxlscuc[k]),
                                horzcat(Lscxlscul[k].T, Lsculscul[k],   Lsculscxc[k],   Lsculscuc[k]),
                                horzcat(Lscxlscxc[k].T, Lsculscxc[k].T, Lscxcscxc[k],   Lscxcscuc[k]),
                                horzcat(Lscxlscuc[k].T, Lsculscuc[k].T, Lscxcscuc[k].T, Lscucscuc[k])
                                )
            xl_grad_k   = xl_grad[k]
            ul_grad_k   = ul_grad[k]
            scxL_grad_k = scxL_grad[k]
            scuL_grad_k = scuL_grad[k]
            scxC_grad_k = scxC_grad[k]
            scuC_grad_k = scuC_grad[k]
            xc_grad_k   = grad_outc[0]['xi_grad'][k]
            uc_grad_k   = grad_outc[0]['ui_grad'][k]
            for i in range(1,int(self.nq)):
                xc_grad_k = np.vstack((xc_grad_k,grad_outc[i]['xi_grad'][k]))
                uc_grad_k = np.vstack((uc_grad_k,grad_outc[i]['ui_grad'][k]))
            L_trajp_k   = vertcat(
                                horzcat(Lscxlp[k] - px_dis*xl_grad_k - scxL_grad_k),
                                horzcat(Lsculp[k] - pu_dis*ul_grad_k - scuL_grad_k),
                                horzcat(Lscxcp[k] - pix_dis*xc_grad_k - scxC_grad_k),
                                horzcat(Lscucp[k] - piu_dis*uc_grad_k - scuC_grad_k)
                                )
            L_hessian_ko = vertcat(
                                horzcat(Lscxlscxl_o[k],   Lscxlscul_o[k],   Lscxlscxc_o[k],   Lscxlscuc_o[k]),
                                horzcat(Lscxlscul_o[k].T, Lsculscul_o[k],   Lsculscxc_o[k],   Lsculscuc_o[k]),
                                horzcat(Lscxlscxc_o[k].T, Lsculscxc_o[k].T, Lscxcscxc_o[k],   Lscxcscuc_o[k]),
                                horzcat(Lscxlscuc_o[k].T, Lsculscuc_o[k].T, Lscxcscuc_o[k].T, Lscucscuc_o[k])
                                )
            
            start_time = TM.time()
            min_eigval = np.min(LA.eigvalsh(L_hessian_ko))
            eigtime    = (TM.time() - start_time)*1000
            EigTime   += eigtime
            # print('ADMM iteration:',i_admm+1,"eig_time:--- %s ms ---" % format(eigtime,'.2f'))       
            
            # A = np.array(L_hessian_ko)
            # min_eigval = eigsh(A, k=1, which='SA', return_eigenvectors=False)[0]
            MIN_eigen += [min_eigval]
            if min_eigval<0:
                reg = -min_eigval+1e-4
            else:
                reg = 0
            L_hessian_k_sym = L_hessian_k + reg*I_hessian
            L, _jitter      = self.try_cholesky(L_hessian_k_sym, jitter0=0.0)
            grad_subp2_k    = self.chol_solve(L, -L_trajp_k)
            # grad_subp2_k    = LA.solve(L_hessian_k_sym,-L_trajp_k)
            scxl_grad[k]    = grad_subp2_k[0:self.nxl,:]
            scul_grad[k]    = grad_subp2_k[self.nxl:(self.nxl+self.nul),:]
            scxc_grad[k]    = grad_subp2_k[(self.nxl+self.nul):(self.nxl+self.nul+self.nxi*int(self.nq)),:]
            scuc_grad[k]    = grad_subp2_k[(self.nxl+self.nul+self.nxi*int(self.nq)):(self.nxl+self.nul+self.nxi*int(self.nq)+self.nui*int(self.nq)),:]
        # terminal gradients
        L_hessian_N = vertcat(
                                horzcat(Lscxlscxl[self.N],   Lscxlscxc[self.N]),
                                horzcat(Lscxlscxc[self.N].T, Lscxcscxc[self.N])
                                )
        xl_grad_N   = xl_grad[self.N]
        xc_grad_N   = grad_outc[0]['xi_grad'][self.N]
        for i in range(1,int(self.nq)):
            xc_grad_N = np.vstack((xc_grad_N,grad_outc[i]['xi_grad'][self.N]))
        scxL_grad_N = scxL_grad[self.N]
        scxC_grad_N = scxC_grad[self.N]
        L_trajp_N   = vertcat(
                            horzcat(Lscxlp[self.N] - px_dis*xl_grad_N - scxL_grad_N),
                            horzcat(Lscxcp[self.N] - pix_dis*xc_grad_N - scxC_grad_N)
                            )
        L_hessian_No = vertcat(
                                horzcat(Lscxlscxl_o[self.N],   Lscxlscxc_o[self.N]),
                                horzcat(Lscxlscxc_o[self.N].T, Lscxcscxc_o[self.N])
                                )
        start_time = TM.time()
        min_eigval = np.min(LA.eigvalsh(L_hessian_No))
        eigtime    = (TM.time() - start_time)*1000
        EigTime   += eigtime
        # A = np.array(L_hessian_No)
        # min_eigval = eigsh(A, k=1, which='SA', return_eigenvectors=False)[0]
        MIN_eigen += [min_eigval]
        if min_eigval<0:
            reg = -min_eigval+1e-4
        else:
            reg = 0
        L_hessian_N_sym = L_hessian_N + reg*I_hess2 
        L, _jitter      = self.try_cholesky(L_hessian_N_sym, jitter0=0.0)   
        grad_subp2_N    = self.chol_solve(L, -L_trajp_N)
        scxl_grad[self.N] = grad_subp2_N[0:self.nxl,:]
        scxc_grad[self.N] = grad_subp2_N[self.nxl:(self.nxl+self.nxi*int(self.nq)),:]
        print('min_eigen=',np.min(MIN_eigen)) 
        print('ADMM iteration:',i_admm+1,"Eig_time:--- %s ms ---" % format(EigTime,'.2f')) 
        grad_out2 = {
                    "scxl_grad":scxl_grad,
                    "scul_grad":scul_grad,
                    "scxc_grad":scxc_grad,
                    "scuc_grad":scuc_grad
                    }
        
        return grad_out2
    

    def SubP3_Gradient(self,auxSys3,grad_outl,grad_outc,grad_out2,scxL_grad,scuL_grad,scxC_grad,scuC_grad,px,pu,gammax,gammau,pix,piu,gammaix,gammaiu,ADMM_max,i_admm):
        xl_grad         = grad_outl['xl_grad']
        ul_grad         = grad_outl['ul_grad']
        scxl_grad       = grad_out2['scxl_grad']
        scul_grad       = grad_out2['scul_grad']
        scxc_grad       = grad_out2['scxc_grad']
        scuc_grad       = grad_out2['scuc_grad']
        dscxL_updatedp  = auxSys3['dscxL_updatedp']
        dscuL_updatedp  = auxSys3['dscuL_updatedp']
        dscxC_updatedp  = auxSys3['dscxC_updatedp']
        dscuC_updatedp  = auxSys3['dscuC_updatedp']
        scxL_grad_new   = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        scuL_grad_new   = self.N*[np.zeros((self.nul,self.n_Pauto))]
        scxC_grad_new   = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.n_Pauto))]
        scuC_grad_new   = self.N*[np.zeros((self.nui*int(self.nq),self.n_Pauto))]
        px_dis          = self.open_loop_penalty(px,gammax,i_admm,ADMM_max)
        pu_dis          = self.open_loop_penalty(pu,gammau,i_admm,ADMM_max)
        pix_dis         = self.open_loop_penalty(pix,gammaix,i_admm,ADMM_max)
        piu_dis         = self.open_loop_penalty(piu,gammaiu,i_admm,ADMM_max)
        for k in range(self.N):
            xc_grad_k   = grad_outc[0]['xi_grad'][k]
            uc_grad_k   = grad_outc[0]['ui_grad'][k]
            for i in range(1,int(self.nq)):
                xc_grad_k = np.vstack((xc_grad_k,grad_outc[i]['xi_grad'][k]))
                uc_grad_k = np.vstack((uc_grad_k,grad_outc[i]['ui_grad'][k]))
            scxL_grad_new[k] = scxL_grad[k] + px_dis*(xl_grad[k] - scxl_grad[k]) + dscxL_updatedp[k]
            scuL_grad_new[k] = scuL_grad[k] + pu_dis*(ul_grad[k] - scul_grad[k]) + dscuL_updatedp[k]
            scxC_grad_new[k] = scxC_grad[k] + pix_dis*(xc_grad_k - scxc_grad[k]) + dscxC_updatedp[k]
            scuC_grad_new[k] = scuC_grad[k] + piu_dis*(uc_grad_k - scuc_grad[k]) + dscuC_updatedp[k]
        # terminal gradients
        xc_grad_N   = grad_outc[0]['xi_grad'][self.N]
        for i in range(1,int(self.nq)):
            xc_grad_N = np.vstack((xc_grad_N,grad_outc[i]['xi_grad'][self.N]))
        scxL_grad_new[self.N]= scxL_grad[self.N] + px_dis*(xl_grad[self.N] - scxl_grad[self.N]) + dscxL_updatedp[self.N]
        scxC_grad_new[self.N]= scxC_grad[self.N] + pix_dis*(xc_grad_N - scxc_grad[self.N]) + dscxC_updatedp[self.N]

        grad_out3 = {
            "scxL_grad":scxL_grad_new,
            "scuL_grad":scuL_grad_new,
            "scxC_grad":scxC_grad_new,
            "scuC_grad":scuC_grad_new
        }

        return grad_out3
    

    def ADMM_Gradient_Solver(self,Opt_Sol1_l, Opt_Sol1_cddp, Opt_Sol1_c, Opt_Sol2, Opt_Sol3, Ref_xl, Ref_ul, ref_xq, ref_uq, weight1, weight2):
        # initialize the gradient trajectories of SubP2 and SubP3
        scxl_grad  = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        scul_grad  = self.N*[np.zeros((self.nul,self.n_Pauto))]
        scxL_grad  = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        scuL_grad  = self.N*[np.zeros((self.nul,self.n_Pauto))]
        scxc_grad  = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.n_Pauto))]
        scuc_grad  = self.N*[np.zeros((self.nui*int(self.nq),self.n_Pauto))]
        scxC_grad  = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.n_Pauto))]
        scuC_grad  = self.N*[np.zeros((self.nui*int(self.nq),self.n_Pauto))]
        # initial trajectories, same as those used in the ADMM recursion in the forward pass
        scxl       = np.zeros((self.N+1,self.nxl))
        scul       = np.zeros((self.N,self.nul))
        for k in range(self.N):
            scul[k,:] = Ref_ul[k*self.nul:(k+1)*self.nul]
            scxl[k,:] = np.reshape(self.model_l_fn(xl0=scxl[k,:],ul0=scul[k,:])['mdynlf'].full(),self.nxl)
        scxl[self.N,:]= Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl]
        scxc       = [np.zeros((self.N+1,self.nxi)) for _ in range(int(self.nq))] 
        scuc       = [np.zeros((self.N,self.nui)) for _ in range(int(self.nq))] 
        for i in range(int(self.nq)):
            for k in range(self.N):
                scuc[i][k,:] = ref_uq[i*self.nui:(i+1)*self.nui]
                scxc[i][k,:] = np.reshape(self.model_i_fn(xi0=scxc[i][k,:],ui0=scuc[i][k,:])['mdynif'].full(),self.nxi)
            scxc[i][self.N,:] = ref_xq[i][self.N*self.nxi:(self.N+1)*self.nxi]
        scxL       = np.zeros((self.N+1,self.nxl))
        scuL       = np.zeros((self.N,self.nul))
        scxC       = [np.zeros((self.N+1,self.nxi)) for _ in range(int(self.nq))] 
        scuC       = [np.zeros((self.N,self.nui)) for _ in range(int(self.nq))] 
        # lists for storing gradient trajectories
        Grad_Out1l     = []
        Grad_Out1c     = []
        Grad_Out2      = []
        Grad_Out3      = []
        GradTime       = []
        GradTimeCao    = []
        GradTimeCaos   = []
        GradTimePDP    = []
        GradTime_c     = []
        GradTimeCaos_c = []
        GradTimeCao_c  = []
        GradTimePDP_c  = []
        MeanerrorCao   = [] # error between gradRe and gradPDP
        MeanerrorPDP   = [] # error between gradRe and gradCao
        MeanerrorCao_c = []
        MeanerrorPDP_c = []
        Pauto      = np.concatenate((weight1,weight2))
        gMeanerror_l   = [] # error between two load gradient trajecotries at two successive ADMM iterations
        gMeanerror_c   = [] # error between two cable gradient trajecotries at two successive ADMM iterations
        for i_admm in range(self.max_iter_ADMM):
            # gradients of Subproblem1
            opt_sol        = Opt_Sol1_l[i_admm]
            auxSysl        = self.Get_AuxSys_DDP_Load(opt_sol,Ref_xl,Ref_ul,scxl,scul,scxL,scuL,weight1,i_admm)
            start_time     = TM.time()
            grad_outl      = self.DDP_Load_Gradient(opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, weight1[-4],weight1[-3],weight1[-2],weight1[-1],int(self.max_iter_ADMM),i_admm)
            gradtimeOur    = (TM.time() - start_time)*1000
            start_time     = TM.time()
            grad_outl_Caos = self.Cao_Load_Gradient_s(opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, weight1[-4],weight1[-3],weight1[-2],weight1[-1],int(self.max_iter_ADMM), i_admm)
            gradtimeCaos   = (TM.time() - start_time)*1000
            start_time     = TM.time()
            grad_outl_Cao  = self.Cao_Load_Gradient(opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, weight1[-4],weight1[-3],weight1[-2],weight1[-1],int(self.max_iter_ADMM),i_admm)
            gradtimeCao    = (TM.time() - start_time)*1000
            start_time     = TM.time()
            grad_outl_PDP  = self.PDP_Load_Gradient(opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, weight1[-4],weight1[-3],weight1[-2],weight1[-1],int(self.max_iter_ADMM), i_admm)
            gradtimePDP    = (TM.time() - start_time)*1000
            grad_outc = []
            grad_outcCao = []
            grad_outcPDP = []
            gradtimeOur_sum   = 0
            gradtimeCaos_sum  = 0
            gradtimeCao_sum   = 0
            gradtimePDP_sum   = 0
            for i in range(int(self.nq)):
                opt_solc  = Opt_Sol1_cddp[i_admm]
                Ref_xi    = ref_xq[i]
                Ref_ui    = ref_uq[i*self.nui:(i+1)*self.nui]
                scxi      = scxc[i]
                scui      = scuc[i]
                scxI      = scxC[i]
                scuI      = scuC[i]
                auxSysi   = self.Get_AuxSys_DDP_Cable(opt_solc[i],Ref_xi,Ref_ui,scxi,scui,scxI,scuI,weight2,i_admm)
                scxi_grad = (self.N+1)*[np.zeros((self.nxi, self.n_Pauto))]
                scxI_grad = (self.N+1)*[np.zeros((self.nxi, self.n_Pauto))]
                scui_grad = self.N*[np.zeros((self.nui, self.n_Pauto))]
                scuI_grad = self.N*[np.zeros((self.nui, self.n_Pauto))]
                for k in range(self.N):
                    scxi_grad[k] = np.reshape(scxc_grad[k][i*self.nxi:(i+1)*self.nxi,:],(self.nxi,self.n_Pauto))
                    scxI_grad[k] = np.reshape(scxC_grad[k][i*self.nxi:(i+1)*self.nxi,:],(self.nxi,self.n_Pauto))
                    scui_grad[k] = np.reshape(scuc_grad[k][i*self.nui:(i+1)*self.nui,:],(self.nui,self.n_Pauto))
                    scuI_grad[k] = np.reshape(scuC_grad[k][i*self.nui:(i+1)*self.nui,:],(self.nui,self.n_Pauto))
                scxi_grad[self.N]= np.reshape(scxc_grad[self.N][i*self.nxi:(i+1)*self.nxi,:],(self.nxi,self.n_Pauto))
                scxI_grad[self.N]= np.reshape(scxC_grad[self.N][i*self.nxi:(i+1)*self.nxi,:],(self.nxi,self.n_Pauto))
                start_time           = TM.time()
                grad_outi            = self.DDP_Cable_Gradient(opt_solc[i],auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, weight2[-4],weight2[-3],weight2[-2],weight2[-1],int(self.max_iter_ADMM),i_admm)
                gradtimeOur_cable    = (TM.time() - start_time)*1000
                gradtimeOur_sum     += gradtimeOur_cable
                grad_outc           += [grad_outi]
                start_time           = TM.time()
                grad_outi_Caos       = self.Cao_Cable_Gradient_s(opt_solc[i],auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, weight2[-4],weight2[-3],weight2[-2],weight2[-1],int(self.max_iter_ADMM), i_admm)
                gradtimeCaos_cable   = (TM.time() - start_time)*1000
                gradtimeCaos_sum    += gradtimeCaos_cable
                start_time           = TM.time()
                grad_outi_Cao        = self.Cao_Cable_Gradient(opt_solc[i],auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, weight2[-4],weight2[-3],weight2[-2],weight2[-1],int(self.max_iter_ADMM),i_admm)
                gradtimeCao_cable    = (TM.time() - start_time)*1000
                gradtimeCao_sum     += gradtimeCao_cable
                grad_outcCao        += [grad_outi_Cao]
                start_time           = TM.time()
                grad_outi_PDP        = self.PDP_Cable_Gradient(opt_solc[i],auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, weight2[-4],weight2[-3],weight2[-2],weight2[-1],int(self.max_iter_ADMM), i_admm)
                gradtimePDP_cable    = (TM.time() - start_time)*1000
                gradtimePDP_sum     += gradtimePDP_cable
                grad_outcPDP        += [grad_outi_PDP]
            gradtimeOur_avgcable  = gradtimeOur_sum/self.nq
            gradtimeCaos_avgcable = gradtimeCaos_sum/self.nq
            gradtimeCao_avgcable  = gradtimeCao_sum/self.nq
            gradtimePDP_avgcable  = gradtimePDP_sum/self.nq
                
            # gradients of Subproblem2
            opt_sol1_c = Opt_Sol1_c[i_admm]
            opt_sol2   = Opt_Sol2[i_admm]
            auxSys2    = self.Get_AuxSys_SubP2(opt_sol,opt_sol1_c,opt_sol2,scxL,scuL,scxC,scuC,Pauto,i_admm)
            grad_out2  = self.SubP2_Gradient(auxSys2,grad_outl,grad_outc,scxL_grad,scuL_grad,scxC_grad,scuC_grad,weight1[-4],weight1[-3],weight1[-2],weight1[-1],weight2[-4],weight2[-3],weight2[-2],weight2[-1],int(self.max_iter_ADMM),i_admm) 
            # gradients of Subproblem3
            auxSys3    = self.Get_AuxSys_SubP3(opt_sol,opt_sol1_c,opt_sol2,weight1,weight2,i_admm)
            grad_out3  = self.SubP3_Gradient(auxSys3,grad_outl,grad_outc,grad_out2,scxL_grad,scuL_grad,scxC_grad,scuC_grad,weight1[-4],weight1[-3],weight1[-2],weight1[-1],weight2[-4],weight2[-3],weight2[-2],weight2[-1],int(self.max_iter_ADMM),i_admm)
            # update
            scxl       = opt_sol2['scxl_traj']
            scul       = opt_sol2['scul_traj']
            scxc       = opt_sol2['scxc_traj']
            scuc       = opt_sol2['scuc_traj']
            opt_sol3   = Opt_Sol3[i_admm]
            scxL       = opt_sol3['scxL_traj_new']
            scuL       = opt_sol3['scuL_traj_new']
            scxC       = opt_sol3['scxC_traj_new']
            scuC       = opt_sol3['scuC_traj_new']
            scxl_grad  = grad_out2['scxl_grad']
            scul_grad  = grad_out2['scul_grad']
            scxc_grad  = grad_out2['scxc_grad']
            scuc_grad  = grad_out2['scuc_grad']
            scxL_grad  = grad_out3['scxL_grad']
            scuL_grad  = grad_out3['scuL_grad']
            scxC_grad  = grad_out3['scxC_grad']
            scuC_grad  = grad_out3['scuC_grad']
            # save the results
            Grad_Out1l     += [grad_outl]
            Grad_Out1c     += [grad_outc]
            Grad_Out2      += [grad_out2]
            Grad_Out3      += [grad_out3]
            GradTime       += [gradtimeOur]
            GradTimeCaos   += [gradtimeCaos]
            GradTimeCao    += [gradtimeCao]
            GradTimePDP    += [gradtimePDP]
            GradTime_c     += [gradtimeOur_avgcable]
            GradTimeCaos_c += [gradtimeCaos_avgcable]
            GradTimeCao_c  += [gradtimeCao_avgcable]
            GradTimePDP_c  += [gradtimePDP_avgcable] 
            

            xl_grad    = grad_outl['xl_grad']
            ul_gard    = grad_outl['ul_grad']
            xl_gradCao = grad_outl_Cao['xl_grad']
            xl_gradPDP = grad_outl_PDP['xl_grad']
            
            Error1     = 0
            Error2     = 0
            Error1_c   = 0
            Error2_c   = 0
            for k in range(self.N):
                error1 = xl_grad[k+1] - xl_gradCao[k+1]
                Error1 += (LA.norm(error1,ord='fro')/LA.norm(xl_grad[k+1],ord='fro'))
                error2 = xl_grad[k+1] - xl_gradPDP[k+1]
                Error2 += (LA.norm(error2,ord='fro')/LA.norm(xl_grad[k+1],ord='fro'))
                for i in range(int(self.nq)):
                    error1_c = grad_outc[i]['xi_grad'][k+1] - grad_outcCao[i]['xi_grad'][k+1]
                    Error1_c += (LA.norm(error1_c,ord='fro')/LA.norm(grad_outc[i]['xi_grad'][k+1],ord='fro'))
                    error2_c = grad_outc[i]['xi_grad'][k+1] - grad_outcPDP[i]['xi_grad'][k+1]
                    Error2_c += (LA.norm(error2_c,ord='fro')/LA.norm(grad_outc[i]['xi_grad'][k+1],ord='fro'))
            
           
            gError1     = 0
            gErroru1    = 0
            gError1_c   = 0
            gErroru1_c  = 0
            for k in range(self.N):
                gerror1 = xl_grad[k] - scxl_grad[k] 
                gError1 += LA.norm(gerror1,ord='fro')
                gerroru1 = ul_gard[k] - scul_grad[k]
                gErroru1 += LA.norm(gerroru1,ord='fro')
                    
                for i in range(int(self.nq)):
                    gerror1_c = grad_outc[i]['xi_grad'][k] - np.reshape(scxc_grad[k][i*self.nxi:(i+1)*self.nxi,:],(self.nxi,self.n_Pauto))
                    gError1_c += LA.norm(gerror1_c,ord='fro')
                    gerroru1_c = grad_outc[i]['ui_grad'][k] - np.reshape(scuc_grad[k][i*self.nui:(i+1)*self.nui,:],(self.nui,self.n_Pauto))
                    gErroru1_c += LA.norm(gerroru1_c,ord='fro')
            gmeanerror1 = np.sqrt(gError1**2+gErroru1**2)/self.N
            gmeanerror1_c = np.sqrt(gError1_c**2+gErroru1_c**2)/(self.N*self.nq) 
            gMeanerror_l += [gmeanerror1]
            gMeanerror_c += [gmeanerror1_c]       

            meanerror1 = Error1/self.N
            meanerror2 = Error2/self.N    
            MeanerrorCao += [meanerror1]
            MeanerrorPDP += [meanerror2]
            meanerror1_c = Error1_c/(self.N*self.nq)
            meanerror2_c = Error2_c/(self.N*self.nq)
            MeanerrorCao_c += [meanerror1_c]
            MeanerrorPDP_c += [meanerror2_c]
            if i_admm == self.max_iter_ADMM-1:
                print("g_Our:--- %s ms ---" % format(gradtimeOur,'.2f'))
                print("g_Cao_s:--- %s ms ---" % format(gradtimeCaos,'.2f'))
                print("g_Cao:--- %s ms ---" % format(gradtimeCao,'.2f'))
                print("g_PDP:--- %s ms ---" % format(gradtimePDP,'.2f'))
                print("g_Our_cable:--- %s ms ---" % format(gradtimeOur_avgcable,'.2f'))
                print("g_Cao_s_cable:--- %s ms ---" % format(gradtimeCaos_avgcable,'.2f'))
                print("g_Cao_cable:--- %s ms ---" % format(gradtimeCao_avgcable,'.2f'))
                print("g_PDP_cable:--- %s ms ---" % format(gradtimePDP_avgcable,'.2f'))
                print('meanerrorCao=',meanerror1,'meanerrorPDP=',meanerror2)
                print('meanerrorCao_c=',meanerror1_c,'meanerrorPDP_c=',meanerror2_c)
                gMeanerror_l = [float(x) for x in gMeanerror_l]
                gMeanerror_c = [float(x) for x in gMeanerror_c]
                print('gMeanerror_l=',gMeanerror_l,'gMeanerror_c=',gMeanerror_c)

        
        return Grad_Out1l, Grad_Out1c, Grad_Out2, Grad_Out3, GradTime, GradTimeCaos, GradTimeCao,  GradTimePDP,  GradTime_c, GradTimeCaos_c, GradTimeCao_c, GradTimePDP_c,  MeanerrorCao, MeanerrorPDP, MeanerrorCao_c, MeanerrorPDP_c, gMeanerror_l, gMeanerror_c 
    
    

# 这个类主要服务离线梯度/超参数训练与论文相关实验，
# 不属于当前稳定运行时 benchmark 主线。只关心 forward planner / Julia SubP2 的话可先跳过。
class Gradient_Solver:
    def __init__(self, sysm_para, horizon, xl, ul, scxl, scul, xi, ui, scxi, scui, P_auto, weight1, weight2):
        """
        [3]Kendall, A., Gal, Y. and Cipolla, R., 2018. 
        Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. 
        In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7482-7491).
        """
        def get_numel(x):
            try:
                return x.numel()
            except AttributeError:
                return x.size

        self.nxl    = get_numel(xl)
        self.nul    = get_numel(ul)
        self.nxi    = get_numel(xi)
        self.nui    = get_numel(ui)
        self.n_Pauto= get_numel(P_auto)
        self.npl    = get_numel(weight1)
        self.npi    = get_numel(weight2)
        self.nq     = int(sysm_para[6])
        self.N      = horizon
        self.xl     = xl
        self.ul     = ul
        self.xi     = xi
        self.ui     = ui
        self.scxl   = scxl
        self.scul   = scul
        self.scxi   = scxi
        self.scui   = scui
        self.Pauto  = P_auto
        self.xl_ref = SX.sym('xl_ref',self.nxl)
        self.xi_ref = SX.sym('xi_ref',self.nxi)
        # boundaries of the hyperparameters
        self.p_min  = 1e-3
        self.p_max  = 1e3
        self.gamma_min = -3
        self.gamma_max = 3
        #------------- loss definition -------------#
        # tracking loss
        track_error_l = self.xl - self.xl_ref
        track_error_i = self.xi - self.xi_ref
        self.loss_track_l = track_error_l.T@track_error_l
        # 与原版一致：14维时仅跟踪方向与张力；8维兼容旧版 JAX 状态
        if self.nxi == 14:
            self.weight_i = np.diag(np.array([1,1,1,0,0,0,0,0,0,0,0,0,1,0]))
        elif self.nxi == 8:
            self.weight_i = np.diag(np.array([1,1,1,0,0,0,1,0]))
        else:
            self.weight_i = np.eye(self.nxi)
            
        self.loss_track_i = track_error_i.T@self.weight_i@track_error_i
        # primal residual loss
        r_primal_xl     = self.xl - self.scxl
        r_primal_ul     = self.ul - self.scul
        self.loss_rpl   = r_primal_xl.T@r_primal_xl + r_primal_ul.T@r_primal_ul
        self.loss_rpl_N = r_primal_xl.T@r_primal_xl
        r_primal_xi     = self.xi - self.scxi
        r_primal_ui     = self.ui - self.scui
        self.loss_rpi   = r_primal_xi.T@r_primal_xi + r_primal_ui.T@r_primal_ui
        self.loss_rpi_N = r_primal_xi.T@r_primal_xi
    

    # def adaptive_meta_loss_weights(self,loss_t,loss_rp,wt): # using ideas from heuristic adaptive ADMM penalty parameters
    #     if loss_t > 1.25*loss_rp:
    #         wt_new = np.clip(1.5*wt,0.2,5)
    #     elif loss_rp > 1.25*loss_t:
    #         wt_new = np.clip(wt/1.5,0.2,5)
    #     else:
    #         wt_new = wt
    #     return wt_new
    
   
    def adaptive_meta_loss_weights(self, loss_t, loss_rp, g_t, g_rp, wt, alpha=7, beta_w=0.8, eps=1e-8, k_min=0.2, k_max=200.0):
        # -------- safer auto-K (clip + exponent) --------
        k_auto = (loss_t / (loss_rp + eps)) ** alpha
        k_auto = float(np.clip(k_auto, k_min, k_max))
        wt_target = 2.0 * k_auto * g_rp / (g_t + k_auto * g_rp + eps)
        # -------- slow weight update --------
        wt_new = (1 - beta_w) * wt + beta_w * wt_target
        wt_new = float(np.clip(wt_new, 0.01, 1.99))
        wrp_new = 2.0 - wt_new

        return wt_new, wrp_new, k_auto


    # def adaptive_meta_loss_weights(self, loss_t, loss_rp, wt,alpha=5, beta_w=0.6, eps=1e-8,k_min=0.25, k_max=10.0):

    #     # -------- initialize reference losses once --------
    #     if not hasattr(self, "Lt0"):
    #         self.Lt0  = float(loss_t)  + eps
    #         self.Lrp0 = float(loss_rp) + eps

    #     # -------- relative progress (dimensionless) --------
    #     r_t  = loss_t  / self.Lt0
    #     r_rp = loss_rp / self.Lrp0

    #     # -------- loss-ratio based auto-K --------
    #     k_auto = (r_t / (r_rp + eps)) ** alpha
    #     k_auto = float(np.clip(k_auto, k_min, k_max))

    #     # -------- target weights from loss ratio only --------
    #     wt_target = 2.0 * k_auto / (1.0 + k_auto)

    #     # -------- slow weight update --------
    #     wt_new = (1 - beta_w) * wt + beta_w * wt_target
    #     wt_new = float(np.clip(wt_new, 0.01, 1.99))
    #     wrp_new = 2.0 - wt_new

    #     return wt_new, wrp_new, k_auto



     


    def Set_Parameters(self,tunable_para):
        weight       = np.zeros(self.n_Pauto)
        for k in range(self.n_Pauto):
            weight[k]= self.p_min + (self.p_max - self.p_min) * 1/(1+np.exp(-tunable_para[k])) # sigmoid boundedness
            if k == self.npl-2:
                weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * 1/(1+np.exp(-tunable_para[k]))
            elif k == self.npl-1:
                weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * 1/(1+np.exp(-tunable_para[k]))
            elif k == self.n_Pauto-2:
                weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * 1/(1+np.exp(-tunable_para[k]))
            elif k == self.n_Pauto-1:
                weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * 1/(1+np.exp(-tunable_para[k]))

        return weight
    

    def Set_Parameters_nn_l(self,tunable_para):
        weight       = np.zeros(self.npl)
        for k in range(self.npl):
            weight[k]= self.p_min + (self.p_max - self.p_min) * tunable_para[0,k] # sigmoid boundedness
            if k == self.npl-2:
                weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * tunable_para[0,k]
            elif k == self.npl-1:
                weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * tunable_para[0,k]
        return weight
    
    def Set_Parameters_nn_i(self,tunable_para):
        weight       = np.zeros(self.npi)
        for k in range(self.npi):
            weight[k]= self.p_min + (self.p_max - self.p_min) * tunable_para[0,k] # sigmoid boundedness
            if k == self.npi-2:
                weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * tunable_para[0,k]
            elif k == self.npi-1:
                weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * tunable_para[0,k]
        return weight
    

    def ChainRule_Gradient(self,tunable_para):
        Tunable      = SX.sym('Tp',1,self.n_Pauto)
        Weight       = SX.sym('wp',1,self.n_Pauto)
        for k in range(self.n_Pauto):
            Weight[k]= self.p_min + (self.p_max - self.p_min) * 1/(1 + exp(-Tunable[k])) # sigmoid boundedness
            if k == self.npl-2:
                Weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * 1/(1 + exp(-Tunable[k]))
            elif k == self.npl-1:
                Weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * 1/(1 + exp(-Tunable[k]))
            elif k == self.n_Pauto-2:
                Weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * 1/(1 + exp(-Tunable[k]))
            elif k == self.n_Pauto-1:
                Weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * 1/(1 + exp(-Tunable[k]))
        dWdT         = jacobian(Weight,Tunable)
        dWdT_fn      = Function('dWdT',[Tunable],[dWdT],['Tp0'],['dWdT_f'])
        weight_grad  = dWdT_fn(Tp0=tunable_para)['dWdT_f'].full()

        return weight_grad
    
    def ChainRule_Gradient_nn_l(self,tunable_para):
        Tunable      = SX.sym('Tp',1,self.npl)
        Weight       = SX.sym('wp',1,self.npl)
        for k in range(self.npl):
            Weight[k]= self.p_min + (self.p_max - self.p_min) * Tunable[k] # sigmoid boundedness
            if k == self.npl-2:
                Weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * Tunable[k] 
            elif k == self.npl-1:
                Weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * Tunable[k] 
        dWdT         = jacobian(Weight,Tunable)
        dWdT_fn      = Function('dWdT',[Tunable],[dWdT],['Tp0'],['dWdT_f'])
        weight_grad  = dWdT_fn(Tp0=tunable_para)['dWdT_f'].full()
        return weight_grad
    
    def ChainRule_Gradient_nn_i(self,tunable_para):
        Tunable      = SX.sym('Tp',1,self.npi)
        Weight       = SX.sym('wp',1,self.npi)
        for k in range(self.npi):
            Weight[k]= self.p_min + (self.p_max - self.p_min) * Tunable[k] # sigmoid boundedness
            if k == self.npi-2:
                Weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * Tunable[k]
            elif k == self.npi-1:
                Weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * Tunable[k]
        dWdT         = jacobian(Weight,Tunable)
        dWdT_fn      = Function('dWdT',[Tunable],[dWdT],['Tp0'],['dWdT_f'])
        weight_grad  = dWdT_fn(Tp0=tunable_para)['dWdT_f'].full()
        return weight_grad
    

    def loss(self,Opt_Sol1_l,Opt_Sol1_c,Opt_Sol2,Ref_xl,ref_xq, wt, wrp):
        xl_traj   = Opt_Sol1_l[-1]['xl_traj']
        ul_traj   = Opt_Sol1_l[-1]['ul_traj']
        xc_list   = Opt_Sol1_c[-1]['xc_traj']
        uc_list   = Opt_Sol1_c[-1]['uc_traj']
        scxl_traj = Opt_Sol2[-1]['scxl_traj']
        scul_traj = Opt_Sol2[-1]['scul_traj']
        scxc_traj = Opt_Sol2[-1]['scxc_traj'] # list
        scuc_traj = Opt_Sol2[-1]['scuc_traj'] # list
        loss_track = 0
        loss_resid = 0
        for k in range(self.N):
            xl_k        = np.reshape(xl_traj[k,:],(self.nxl,1))
            ul_k        = np.reshape(ul_traj[k,:],(self.nul,1))
            scxl_k      = np.reshape(scxl_traj[k,:],(self.nxl,1))
            scul_k      = np.reshape(scul_traj[k,:],(self.nul,1))
            refxl_k     = np.reshape(Ref_xl[k*self.nxl:(k+1)*self.nxl],(self.nxl,1))
            error_k     = xl_k - refxl_k # load tracking error
            resid_xk    = xl_k - scxl_k  # load primal state residual
            resid_uk    = ul_k - scul_k  # load primal control residual
            loss_track += error_k.T@error_k # load tracking loss at k
            loss_resid += resid_xk.T@resid_xk + resid_uk.T@resid_uk
            for i in range(self.nq):
                xi_k        = np.reshape(xc_list[i][k,:],(self.nxi,1))
                ui_k        = np.reshape(uc_list[i][k,:],(self.nui,1))
                scxi_k      = np.reshape(scxc_traj[i][k,:],(self.nxi,1))
                scui_k      = np.reshape(scuc_traj[i][k,:],(self.nui,1))
                refxi_k     = np.reshape(ref_xq[i][k*self.nxi:(k+1)*self.nxi],(self.nxi,1))
                error_ik    = xi_k - refxi_k
                resid_xik   = xi_k - scxi_k
                resid_uik   = ui_k - scui_k
                loss_track += error_ik.T@self.weight_i@error_ik
                loss_resid += resid_xik.T@resid_xik + resid_uik.T@resid_uik
        xl_N        = np.reshape(xl_traj[self.N,:],(self.nxl,1))
        scxl_N      = np.reshape(scxl_traj[self.N,:],(self.nxl,1))
        refxl_N     = np.reshape(Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl],(self.nxl,1))
        error_N     = xl_N - refxl_N
        resid_xN    = xl_N - scxl_N
        loss_track += error_N.T@error_N
        loss_resid += resid_xN.T@resid_xN
        for i in range(self.nq):
            xi_N        = np.reshape(xc_list[i][self.N,:],(self.nxi,1))
            scxi_N      = np.reshape(scxc_traj[i][self.N,:],(self.nxi,1))
            refxi_N     = np.reshape(ref_xq[i][self.N*self.nxi:(self.N+1)*self.nxi],(self.nxi,1))
            error_iN    = xi_N - refxi_N
            resid_xiN   = xi_N - scxi_N
            loss_track += error_iN.T@self.weight_i@error_iN # zero weight_i
            loss_resid += resid_xiN.T@resid_xiN
        
        loss = wt*loss_track + wrp*loss_resid
        return loss, loss_track, loss_resid
    

    def ChainRule(self,Opt_Sol1_l,Opt_Sol1_c,Opt_Sol2,Ref_xl,ref_xq,Grad_Out1l,Grad_Out1c,Grad_Out2,wt,wrp):
        dltdxl          = jacobian(self.loss_track_l,self.xl)
        dltdxl_fn       = Function('dltdxl',[self.xl,self.xl_ref],[dltdxl],['xl0','refxl0'],['dltdxl_f'])
        dltdxi          = jacobian(self.loss_track_i,self.xi)
        dltdxi_fn       = Function('dltdxi',[self.xi,self.xi_ref],[dltdxi],['xi0','refxi0'],['dltdxi_f'])
        dlrpdxl         = jacobian(self.loss_rpl,self.xl)
        dlrpdxl_fn      = Function('dlrpdxl',[self.xl,self.scxl,self.ul,self.scul],[dlrpdxl],['xl0','scxl0','ul0','scul0'],['dlrpdxl_f'])
        dlrpdul         = jacobian(self.loss_rpl,self.ul)
        dlrpdul_fn      = Function('dlrpdul',[self.xl,self.scxl,self.ul,self.scul],[dlrpdul],['xl0','scxl0','ul0','scul0'],['dlrpdul_f'])
        dlrpdscxl       = jacobian(self.loss_rpl,self.scxl)
        dlrpdscxl_fn    = Function('dlrpdscxl',[self.xl,self.scxl,self.ul,self.scul],[dlrpdscxl],['xl0','scxl0','ul0','scul0'],['dlrpdscxl_f'])
        dlrpdscul       = jacobian(self.loss_rpl,self.scul)
        dlrpdscul_fn    = Function('dlrpdscul',[self.xl,self.scxl,self.ul,self.scul],[dlrpdscul],['xl0','scxl0','ul0','scul0'],['dlrpdscul_f'])
        dlrpdxi         = jacobian(self.loss_rpi,self.xi)
        dlrpdxi_fn      = Function('dlrpdxi',[self.xi,self.scxi,self.ui,self.scui],[dlrpdxi],['xi0','scxi0','ui0','scui0'],['dlrpdxi_f'])
        dlrpdui         = jacobian(self.loss_rpi,self.ui)
        dlrpdui_fn      = Function('dlrpdui',[self.xi,self.scxi,self.ui,self.scui],[dlrpdui],['xi0','scxi0','ui0','scui0'],['dlrpdui_f'])
        dlrpdscxi       = jacobian(self.loss_rpi,self.scxi)
        dlrpdscxi_fn    = Function('dlrpdscxi',[self.xi,self.scxi,self.ui,self.scui],[dlrpdscxi],['xi0','scxi0','ui0','scui0'],['dlrpdscxi_f'])
        dlrpdscui       = jacobian(self.loss_rpi,self.scui)
        dlrpdscui_fn    = Function('dlrpdscui',[self.xi,self.scxi,self.ui,self.scui],[dlrpdscui],['xi0','scxi0','ui0','scui0'],['dlrpdscui_f'])
        dlrpdxlN        = jacobian(self.loss_rpl_N,self.xl)
        dlrpdxlN_fn     = Function('dlrpdxlN',[self.xl,self.scxl],[dlrpdxlN],['xl0','scxl0'],['dlrpdxlN_f'])
        dlrpdscxlN      = jacobian(self.loss_rpl_N,self.scxl)
        dlrpdscxlN_fn   = Function('dlrpdscxlN',[self.xl,self.scxl],[dlrpdscxlN],['xl0','scxl0'],['dlrpdscxlN_f'])
        dlrpdxiN        = jacobian(self.loss_rpi_N,self.xi)
        dlrpdxiN_fn     = Function('dlrpdxiN',[self.xi,self.scxi],[dlrpdxiN],['xi0','scxi0'],['dlrpdxiN_f'])
        dlrpdscxiN      = jacobian(self.loss_rpi_N,self.scxi)
        dlrpdscxiN_fn   = Function('dlrpdscxiN',[self.xi,self.scxi],[dlrpdscxiN],['xi0','scxi0'],['dlrpdscxiN_f'])
        dltdw           = 0 # gradient of the tracking errors
        dlrpdw          = 0 # gradient of the ADMM primal residuals
        # load trajectories
        k_admm          = -1 # the last, the most recent trajectories and gradients
        xl_traj         = Opt_Sol1_l[k_admm]['xl_traj']
        ul_traj         = Opt_Sol1_l[k_admm]['ul_traj']
        scxl_traj       = Opt_Sol2[k_admm]['scxl_traj']
        scul_traj       = Opt_Sol2[k_admm]['scul_traj']
        # load gradient trajectories
        xl_grad         = Grad_Out1l[k_admm]['xl_grad']
        ul_grad         = Grad_Out1l[k_admm]['ul_grad']
        scxl_grad       = Grad_Out2[k_admm]['scxl_grad']
        scul_grad       = Grad_Out2[k_admm]['scul_grad']
        # cable trajectories
        xc_traj         = Opt_Sol1_c[k_admm]['xc_traj'] # a list
        uc_traj         = Opt_Sol1_c[k_admm]['uc_traj'] # a list
        scxc_traj       = Opt_Sol2[k_admm]['scxc_traj'] # a list
        scuc_traj       = Opt_Sol2[k_admm]['scuc_traj'] # a list
        # cable gradient trajectories
        grad_outc       = Grad_Out1c[k_admm] # a list that contains both state and control gradients
        scxc_grad       = Grad_Out2[k_admm]['scxc_grad']
        scuc_grad       = Grad_Out2[k_admm]['scuc_grad']
        # meta-loss
        loss, loss_track, loss_resid   = self.loss(Opt_Sol1_l,Opt_Sol1_c,Opt_Sol2,Ref_xl,ref_xq,wt,wrp)
        
        for k in range(self.N):
            # gradient of the load tracking errors
            dltdxl_k    = dltdxl_fn(xl0=xl_traj[k,:],refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl])['dltdxl_f'].full()
            dltldw      = dltdxl_k@xl_grad[k]
            # print('dltldwr1=',dltldw[0,2*self.nxl],'dltldwpi=',dltldw[0,self.n_Pauto-1])
            dltdw      += dltldw
            # gradient of the load primal residuals
            dlrpdxl_k   = dlrpdxl_fn(xl0=xl_traj[k,:],scxl0=scxl_traj[k,:],ul0=ul_traj[k,:],scul0=scul_traj[k,:])['dlrpdxl_f'].full()
            dlrpdscxl_k = dlrpdscxl_fn(xl0=xl_traj[k,:],scxl0=scxl_traj[k,:],ul0=ul_traj[k,:],scul0=scul_traj[k,:])['dlrpdscxl_f'].full()
            dlrpdul_k   = dlrpdul_fn(xl0=xl_traj[k,:],scxl0=scxl_traj[k,:],ul0=ul_traj[k,:],scul0=scul_traj[k,:])['dlrpdul_f'].full()
            dlrpdscul_k = dlrpdscul_fn(xl0=xl_traj[k,:],scxl0=scxl_traj[k,:],ul0=ul_traj[k,:],scul0=scul_traj[k,:])['dlrpdscul_f'].full()
            dlrpdw     += dlrpdxl_k@xl_grad[k] + dlrpdscxl_k@scxl_grad[k] + dlrpdul_k@ul_grad[k] + dlrpdscul_k@scul_grad[k]
            for i in range(self.nq):
                # gradient of the cable tracking errors
                xi_traj     = xc_traj[i]
                ui_traj     = uc_traj[i]
                scxi_traj   = scxc_traj[i]
                scui_traj   = scuc_traj[i]
                refxi_k     = ref_xq[i][k*self.nxi:(k+1)*self.nxi]
                dltdxi_k    = dltdxi_fn(xi0=xi_traj[k,:],refxi0=refxi_k)['dltdxi_f'].full()
                grad_outi   = grad_outc[i]
                xi_grad     = grad_outi['xi_grad']
                dltidw      = dltdxi_k@xi_grad[k]
                # print('dltidwr4=',dltidw[0,2*self.nxl+self.nul+2*self.nxi+self.nui],'dltidwpl=',dltidw[0,2*self.nxl+self.nul],'dltidwpi=',dltidw[0,2*self.nxl+self.nul+2*self.nxi+self.nui+1])
                dltdw      += dltidw
                # gradient of the cable primal residuals
                ui_grad     = grad_outi['ui_grad']
                scxi_grad_k = scxc_grad[k][i*self.nxi:(i+1)*self.nxi,:]
                scui_grad_k = scuc_grad[k][i*self.nui:(i+1)*self.nui,:]
                dlrpdxi_k   = dlrpdxi_fn(xi0=xi_traj[k,:],scxi0=scxi_traj[k,:],ui0=ui_traj[k,:],scui0=scui_traj[k,:])['dlrpdxi_f'].full()
                dlrpdscxi_k = dlrpdscxi_fn(xi0=xi_traj[k,:],scxi0=scxi_traj[k,:],ui0=ui_traj[k,:],scui0=scui_traj[k,:])['dlrpdscxi_f'].full()
                dlrpdui_k   = dlrpdui_fn(xi0=xi_traj[k,:],scxi0=scxi_traj[k,:],ui0=ui_traj[k,:],scui0=scui_traj[k,:])['dlrpdui_f'].full()
                dlrpdscui_k = dlrpdscui_fn(xi0=xi_traj[k,:],scxi0=scxi_traj[k,:],ui0=ui_traj[k,:],scui0=scui_traj[k,:])['dlrpdscui_f'].full()
                dlrpdw     += dlrpdxi_k@xi_grad[k] + dlrpdscxi_k@scxi_grad_k + dlrpdui_k@ui_grad[k] + dlrpdscui_k@scui_grad_k
        # -----terminal gradients-----#
        dltdxl_N    = dltdxl_fn(xl0=xl_traj[self.N,:],refxl0=Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl])['dltdxl_f'].full()
        dltdw      += dltdxl_N@xl_grad[self.N]
        dlrpdxl_N   = dlrpdxlN_fn(xl0=xl_traj[self.N,:],scxl0=scxl_traj[self.N,:])['dlrpdxlN_f'].full()
        dlrpdscxl_N = dlrpdscxlN_fn(xl0=xl_traj[self.N,:],scxl0=scxl_traj[self.N,:])['dlrpdscxlN_f'].full()
        dlrpdw     += dlrpdxl_N@xl_grad[self.N] + dlrpdscxl_N@scxl_grad[self.N]
        for i in range(self.nq):
            xi_traj     = xc_traj[i]
            scxi_traj   = scxc_traj[i]
            refxi_N     = ref_xq[i][self.N*self.nxi:(self.N+1)*self.nxi]
            dltdxi_N    = dltdxi_fn(xi0=xi_traj[self.N,:],refxi0=refxi_N)['dltdxi_f'].full()
            grad_outi   = grad_outc[i]
            xi_grad     = grad_outi['xi_grad']
            dltdw      += dltdxi_N@xi_grad[self.N]
            scxi_grad_N = scxc_grad[self.N][i*self.nxi:(i+1)*self.nxi,:]
            dlrpdxi_N   = dlrpdxiN_fn(xi0=xi_traj[self.N,:],scxi0=scxi_traj[self.N,:])['dlrpdxiN_f'].full()
            dlrpdscxi_N = dlrpdscxiN_fn(xi0=xi_traj[self.N,:],scxi0=scxi_traj[self.N,:])['dlrpdscxiN_f'].full()
            dlrpdw     += dlrpdxi_N@xi_grad[self.N] + dlrpdscxi_N@scxi_grad_N
        # total gradient
        dldw        = wt*dltdw + wrp*dlrpdw
        gloss_t     = LA.norm(dltdw)
        gloss_rp    = LA.norm(dlrpdw)

        return dldw, loss, loss_track, loss_resid, gloss_t,gloss_rp
  





















    





        

    

                


                    







    


        



    


    

    

    

        
        
            





    

# -----------------------------------------------------------------------------
# 以下是下沉到文件底部的"非当前稳定主线"顶层 helper / cache。
# - 旧 packaged SubP1 导数提取链会用到 load/cable derivs JIT cache
# - 纯 JAX SubP2 barrier 实验分支会用到 barrier runtime cache
# 它们都不是当前 persistent Julia 主线默认热路径。
# -----------------------------------------------------------------------------

_GLOBAL_SUBP2_BARRIER_RUNTIME_COMMON_CACHE = {}
_GLOBAL_SUBP2_BARRIER_RUNTIME_STAGE_CACHE = {}
_GLOBAL_LOAD_DERIVS_JIT = None
_GLOBAL_CABLE_DERIVS_JIT = None


def _get_global_load_derivs_jit():
    # 仅在旧 SubP1 packaged 路径或显式 need_derivs=True 时才会真正用到。
    # 当前稳定主线默认走 forward-only fast path，不会把这条导数提取链放进热路径。
    global _GLOBAL_LOAD_DERIVS_JIT
    if _GLOBAL_LOAD_DERIVS_JIT is None:
        _GLOBAL_LOAD_DERIVS_JIT = jax.jit(MPC_Planner._static_load_derivs)
    return _GLOBAL_LOAD_DERIVS_JIT


def _get_global_cable_derivs_jit():
    # 同上：这是 SubP1 导数导出核的全局 JIT 缓存，不是当前稳定主线默认会走到的计算。
    global _GLOBAL_CABLE_DERIVS_JIT
    if _GLOBAL_CABLE_DERIVS_JIT is None:
        _GLOBAL_CABLE_DERIVS_JIT = jax.jit(MPC_Planner._static_cable_derivs)
    return _GLOBAL_CABLE_DERIVS_JIT


# -----------------------------------------------------------------------------
# 以下是下沉到文件底部的"非当前稳定主线"顶层 helper。
# 它们只服务纯 JAX SubP2 实验分支，当前 persistent Julia 主线默认不直接走。
# -----------------------------------------------------------------------------

def _subp2_filter_accept_batch(init_metrics, cand_metrics, filter_gamma_theta, filter_gamma_phi, accept_abs, accept_ratio):
    theta_ok = cand_metrics["theta"] <= (1.0 - filter_gamma_theta) * init_metrics["theta"]
    barr_ok = cand_metrics["barr"] <= init_metrics["barr"] - filter_gamma_phi * init_metrics["theta"]
    feas_ok = cand_metrics["feas"] <= jnp.maximum(accept_abs, accept_ratio * init_metrics["feas"])
    return jnp.logical_or(theta_ok, jnp.logical_or(barr_ok, feas_ok))


def _subp2_tail_soft_mask(tail, metrics, finite_mask, soft_mu_tol, soft_comp_tol, soft_dual_tol, soft_trace_ineq_tol, soft_eq_tol, soft_raw_ineq_tol):
    return jnp.logical_and(
        finite_mask,
        jnp.logical_and(
            tail["mu"] <= soft_mu_tol,
            jnp.logical_and(
                tail["comp"] <= soft_comp_tol,
                jnp.logical_and(
                    tail["dual"] <= soft_dual_tol,
                    jnp.logical_and(
                        tail["ineq"] <= soft_trace_ineq_tol,
                        jnp.logical_and(
                            metrics["eq_inf"] <= soft_eq_tol,
                            metrics["ineq_vio"] <= soft_raw_ineq_tol,
                        ),
                    ),
                ),
            ),
        ),
    )


# 这个 kernel 只服务"纯 JAX ipoptax SubP2"分支，用来在 main / retry1 / retry2 结果里做快速筛选。
# 当前稳定的 persistent Julia 主线不会走到这里，可先跳过。
@jax.jit
def _subp2_fast_select_kernel(
    w_init_batch,
    main_x,
    main_conv,
    main_iters,
    init_metrics,
    main_metrics,
    main_tail,
    retry1_x,
    retry1_finite,
    retry1_conv,
    retry1_iters,
    retry1_metrics,
    retry1_tail,
    retry2_x,
    retry2_finite,
    retry2_conv,
    retry2_iters,
    retry2_metrics,
    retry2_tail,
    retry_on_nonconv,
    enable_retry2,
    retry_trigger_ratio,
    restoration_theta_ratio,
    filter_gamma_theta,
    filter_gamma_phi,
    accept_abs,
    accept_ratio,
    soft_mu_tol,
    soft_comp_tol,
    soft_dual_tol,
    soft_eq_tol,
    soft_raw_ineq_tol,
    soft_trace_ineq_tol,
    mode_code_main,
    mode_code_main_soft,
    mode_code_bestfeas,
    mode_code_fallback,
    mode_code_retry1,
    mode_code_retry1_soft,
    mode_code_retry1_bestfeas,
    mode_code_retry2,
    mode_code_retry2_soft,
    mode_code_retry2_bestfeas,
):
    finite_mask = jnp.all(jnp.isfinite(main_x), axis=1)
    soft_mask = _subp2_tail_soft_mask(
        main_tail,
        main_metrics,
        finite_mask,
        soft_mu_tol,
        soft_comp_tol,
        soft_dual_tol,
        soft_trace_ineq_tol,
        soft_eq_tol,
        soft_raw_ineq_tol,
    )
    conv_mask = jnp.logical_and(finite_mask, jnp.logical_or(main_conv, soft_mask))
    filter_ok = jnp.logical_and(
        finite_mask,
        _subp2_filter_accept_batch(
            init_metrics,
            main_metrics,
            filter_gamma_theta,
            filter_gamma_phi,
            accept_abs,
            accept_ratio,
        ),
    )
    resto_ok = jnp.logical_and(
        finite_mask,
        main_metrics["theta"] <= restoration_theta_ratio * init_metrics["theta"],
    )
    improved = jnp.logical_and(
        finite_mask,
        main_metrics["feas"] < (init_metrics["feas"] - 1e-6),
    )
    main_bestfeas_mask = jnp.logical_and(
        jnp.logical_not(conv_mask),
        jnp.logical_or(filter_ok, jnp.logical_and(improved, resto_ok)),
    )

    candidate_x = jnp.where(finite_mask[:, None], main_x, w_init_batch)
    candidate_metrics = {
        "eq_inf": jnp.where(finite_mask, main_metrics["eq_inf"], init_metrics["eq_inf"]),
        "ineq_vio": jnp.where(finite_mask, main_metrics["ineq_vio"], init_metrics["ineq_vio"]),
        "feas": jnp.where(finite_mask, main_metrics["feas"], jnp.inf),
        "theta": jnp.where(finite_mask, main_metrics["theta"], jnp.inf),
        "barr": jnp.where(finite_mask, main_metrics["barr"], jnp.inf),
    }
    candidate_tail = {
        "mu": jnp.where(finite_mask, main_tail["mu"], jnp.nan),
        "ineq": jnp.where(finite_mask, main_tail["ineq"], jnp.nan),
        "comp": jnp.where(finite_mask, main_tail["comp"], jnp.nan),
        "dual": jnp.where(finite_mask, main_tail["dual"], jnp.nan),
    }
    accepted_conv_mask = conv_mask
    diag_iters = main_iters
    diag_mode_codes = jnp.where(
        main_conv,
        mode_code_main,
        jnp.where(
            soft_mask,
            mode_code_main_soft,
            jnp.where(main_bestfeas_mask, mode_code_bestfeas, mode_code_fallback),
        ),
    )

    need_retry_mask = jnp.logical_or(
        jnp.logical_not(finite_mask),
        jnp.logical_and(
            jnp.logical_and(retry_on_nonconv, jnp.logical_not(conv_mask)),
            main_metrics["feas"] >= retry_trigger_ratio * init_metrics["feas"],
        ),
    )

    def _apply_candidate_update(
        success_mask,
        better_mask,
        x_new,
        metrics_new,
        tail_new,
        iters_new,
        mode_success_code,
        mode_soft_code,
        mode_better_code,
        soft_success_mask,
        conv_success_mask,
        candidate_x,
        candidate_metrics,
        candidate_tail,
        accepted_conv_mask,
        diag_iters,
        diag_mode_codes,
    ):
        update_mask = jnp.logical_or(success_mask, better_mask)
        candidate_x = jnp.where(update_mask[:, None], x_new, candidate_x)
        candidate_metrics = {
            "eq_inf": jnp.where(update_mask, metrics_new["eq_inf"], candidate_metrics["eq_inf"]),
            "ineq_vio": jnp.where(update_mask, metrics_new["ineq_vio"], candidate_metrics["ineq_vio"]),
            "feas": jnp.where(update_mask, metrics_new["feas"], candidate_metrics["feas"]),
            "theta": jnp.where(update_mask, metrics_new["theta"], candidate_metrics["theta"]),
            "barr": jnp.where(update_mask, metrics_new["barr"], candidate_metrics["barr"]),
        }
        candidate_tail = {
            "mu": jnp.where(update_mask, tail_new["mu"], candidate_tail["mu"]),
            "ineq": jnp.where(update_mask, tail_new["ineq"], candidate_tail["ineq"]),
            "comp": jnp.where(update_mask, tail_new["comp"], candidate_tail["comp"]),
            "dual": jnp.where(update_mask, tail_new["dual"], candidate_tail["dual"]),
        }
        diag_iters = jnp.where(update_mask, iters_new, diag_iters)
        diag_mode_codes = jnp.where(
            success_mask,
            jnp.where(jnp.logical_and(soft_success_mask, jnp.logical_not(conv_success_mask)), mode_soft_code, mode_success_code),
            jnp.where(better_mask, mode_better_code, diag_mode_codes),
        )
        accepted_conv_mask = jnp.logical_or(accepted_conv_mask, success_mask)
        return candidate_x, candidate_metrics, candidate_tail, accepted_conv_mask, diag_iters, diag_mode_codes

    retry1_soft = _subp2_tail_soft_mask(
        retry1_tail,
        retry1_metrics,
        retry1_finite,
        soft_mu_tol,
        soft_comp_tol,
        soft_dual_tol,
        soft_trace_ineq_tol,
        soft_eq_tol,
        soft_raw_ineq_tol,
    )
    retry1_success = jnp.logical_and(
        need_retry_mask,
        jnp.logical_and(retry1_finite, jnp.logical_or(retry1_conv, retry1_soft)),
    )
    retry1_better = jnp.logical_and(
        need_retry_mask,
        jnp.logical_and(retry1_finite, retry1_metrics["feas"] < candidate_metrics["feas"]),
    )
    candidate_x, candidate_metrics, candidate_tail, accepted_conv_mask, diag_iters, diag_mode_codes = _apply_candidate_update(
        retry1_success,
        retry1_better,
        retry1_x,
        retry1_metrics,
        retry1_tail,
        retry1_iters,
        mode_code_retry1,
        mode_code_retry1_soft,
        mode_code_retry1_bestfeas,
        retry1_soft,
        retry1_conv,
        candidate_x,
        candidate_metrics,
        candidate_tail,
        accepted_conv_mask,
        diag_iters,
        diag_mode_codes,
    )

    retry2_needed = jnp.logical_and(enable_retry2, need_retry_mask)
    retry2_soft = _subp2_tail_soft_mask(
        retry2_tail,
        retry2_metrics,
        retry2_finite,
        soft_mu_tol,
        soft_comp_tol,
        soft_dual_tol,
        soft_trace_ineq_tol,
        soft_eq_tol,
        soft_raw_ineq_tol,
    )
    retry2_success = jnp.logical_and(
        retry2_needed,
        jnp.logical_and(retry2_finite, jnp.logical_or(retry2_conv, retry2_soft)),
    )
    retry2_better = jnp.logical_and(
        retry2_needed,
        jnp.logical_and(retry2_finite, retry2_metrics["feas"] < candidate_metrics["feas"]),
    )
    candidate_x, candidate_metrics, candidate_tail, accepted_conv_mask, diag_iters, diag_mode_codes = _apply_candidate_update(
        retry2_success,
        retry2_better,
        retry2_x,
        retry2_metrics,
        retry2_tail,
        retry2_iters,
        mode_code_retry2,
        mode_code_retry2_soft,
        mode_code_retry2_bestfeas,
        retry2_soft,
        retry2_conv,
        candidate_x,
        candidate_metrics,
        candidate_tail,
        accepted_conv_mask,
        diag_iters,
        diag_mode_codes,
    )

    candidate_valid = jnp.isfinite(candidate_metrics["feas"])
    filter_ok_final = jnp.logical_and(
        candidate_valid,
        _subp2_filter_accept_batch(
            init_metrics,
            candidate_metrics,
            filter_gamma_theta,
            filter_gamma_phi,
            accept_abs,
            accept_ratio,
        ),
    )
    resto_ok_final = jnp.logical_and(
        candidate_valid,
        candidate_metrics["theta"] <= restoration_theta_ratio * init_metrics["theta"],
    )
    improved_final = jnp.logical_and(
        candidate_valid,
        candidate_metrics["feas"] < (init_metrics["feas"] - 1e-6),
    )
    bestfeas_mask = jnp.logical_and(
        jnp.logical_not(accepted_conv_mask),
        jnp.logical_or(filter_ok_final, jnp.logical_and(improved_final, resto_ok_final)),
    )
    accept_mask = jnp.logical_or(accepted_conv_mask, bestfeas_mask)
    diag_mode_codes = jnp.where(
        jnp.logical_and(jnp.logical_not(accept_mask), jnp.logical_not(accepted_conv_mask)),
        mode_code_fallback,
        diag_mode_codes,
    )

    return {
        "w_opt_batch": jnp.where(accept_mask[:, None], candidate_x, w_init_batch),
        "accept_mask": accept_mask,
        "accepted_conv_mask": accepted_conv_mask,
        "diag_mode_codes": diag_mode_codes,
        "diag_iters": jnp.where(accept_mask, diag_iters, 0),
        "diag_eq_inf": jnp.where(accept_mask, candidate_metrics["eq_inf"], init_metrics["eq_inf"]),
        "diag_ineq_vio": jnp.where(accept_mask, candidate_metrics["ineq_vio"], init_metrics["ineq_vio"]),
        "diag_tail_mu": jnp.where(candidate_valid, candidate_tail["mu"], jnp.nan),
        "diag_tail_ineq": jnp.where(candidate_valid, candidate_tail["ineq"], jnp.nan),
        "diag_tail_comp": jnp.where(candidate_valid, candidate_tail["comp"], jnp.nan),
        "diag_tail_dual": jnp.where(candidate_valid, candidate_tail["dual"], jnp.nan),
    }
