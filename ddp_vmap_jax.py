# ddp_vmap_jax.py
# DDP / iLQR with JAX + vmap batch parallelism
# Author: ChatGPT (for Huyang)
# --------------------------------------------------------------
# How to use in your project:
# 1) Replace/plug `dynamics_fn` and `cost_fn` to match your UAV model.
# 2) Keep state dim nx, action dim nu, horizon T consistent across the batch.
# 3) Call `ilqr_ddp_batched(...)` with shapes:
#       x0: [B, nx]
#       u_init: [B, T, nu]
#       params: PyTree with any leaves shaped [B, ...] or broadcastable
#    It returns optimized controls, trajectories, feedback gains, and diagnostics for all B problems in parallel.
# --------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Any, Dict

import jax
#01.10版本不匹配
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
from jax import jit, vmap, lax, tree_util
from jax import remat as checkpoint
checkpoint = jax.remat

Array = jnp.ndarray

@dataclass(frozen=True)
class ILQRConfig:
    max_iters: int = 25
    tol_g_norm: float = 1e-4          # stop when mean ||Qu|| is small
    reg_init: float = 1e-4            # Levenberg-Marquardt regularization
    reg_mult_inc: float = 2.0
    reg_mult_dec: float = 0.5
    reg_max: float = 1e8
    line_search_alphas: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05)
    relinearize_each_alpha: bool = False

@dataclass
class ILQRProblem:
    x0: Array                   # [nx]
    u_init: Array               # [T, nu]
    params: Any                 # PyTree with model/cost parameters
    nx: int
    nu: int
    T: int

@dataclass
class ILQRResult:
    xs: Array                   # [T+1, nx] or [B, T+1, nx]
    us: Array                   # [T, nu]   or [B, T, nu]
    Ks: Array                   # [T, nu, nx] or [B, T, nu, nx]
    ks: Array                   # [T, nu] or [B, T, nu]
    costs: Any                  # PyTree of costs
    iters: Any
    converged: Any
    final_reg: Any

# === 放在 ILQRResult 定义之后 ===

def _ilqr_flatten(r):
    # 把 dataclass 的字段列出来作为 pytree 的“孩子”
    children = (r.xs, r.us, r.Ks, r.ks, r.costs, r.iters, r.converged, r.final_reg)
    aux = None
    return children, aux

def _ilqr_unflatten(aux, children):
    xs, us, Ks, ks, costs, iters, converged, final_reg = children
    return ILQRResult(xs=xs, us=us, Ks=Ks, ks=ks, costs=costs,
                      iters=iters, converged=converged, final_reg=final_reg)

tree_util.register_pytree_node(ILQRResult, _ilqr_flatten, _ilqr_unflatten)

def symmetric_psd(A: Array, reg: float) -> Array:
    n = A.shape[-1]
    return A + reg * jnp.eye(n, dtype=A.dtype)

def rollout(dynamics_fn: Callable[[Array, Array, Any], Array],
            x0: Array, us: Array, params: Any) -> Array:
    def step(x, u):
        x_next = dynamics_fn(x, u, params)
        return x_next, x_next
    xT, xs_tail = lax.scan(step, x0, us)
    xs = jnp.concatenate([x0[None, :], xs_tail], axis=0)
    return xs

def total_cost(cost_fn: Callable[[Array, Array, Any], Array],
               term_cost_fn: Callable[[Array, Any], Array],
               xs: Array, us: Array, params: Any) -> Array:
    stage_costs = vmap(lambda x,u: cost_fn(x,u,params))(xs[:-1], us)
    terminal = term_cost_fn(xs[-1], params)
    return jnp.sum(stage_costs) + terminal

def derivatives(dynamics_fn, cost_fn, term_cost_fn, xs, us, params):
    fx_fn = jit(jax.jacobian(lambda x,u,p: dynamics_fn(x,u,p), 0))
    fu_fn = jit(jax.jacobian(lambda x,u,p: dynamics_fn(x,u,p), 1))
    lx_fn = jit(jax.grad(lambda x,u,p: cost_fn(x,u,p), 0))
    lu_fn = jit(jax.grad(lambda x,u,p: cost_fn(x,u,p), 1))
    lxx_fn = jit(jax.hessian(lambda x,u,p: cost_fn(x,u,p), 0))
    luu_fn = jit(jax.hessian(lambda x,u,p: cost_fn(x,u,p), 1))
    lux_fn = jit(jax.jacobian(lambda x,u,p: jax.grad(cost_fn, 1)(x,u,p), 0))
    Vx_T_fn = jit(jax.grad(lambda x,p: term_cost_fn(x,p), 0))
    Vxx_T_fn = jit(jax.hessian(lambda x,p: term_cost_fn(x,p), 0))

    # fx = vmap(fx_fn)(xs[:-1], us, jax.tree_map(lambda a: a, params))
    # fu = vmap(fu_fn)(xs[:-1], us, jax.tree_map(lambda a: a, params))

    # lx  = vmap(lx_fn)(xs[:-1], us, params)
    # lu  = vmap(lu_fn)(xs[:-1], us, params)
    # lxx = vmap(lxx_fn)(xs[:-1], us, params)
    # luu = vmap(luu_fn)(xs[:-1], us, params)
    # lux = vmap(lux_fn)(xs[:-1], us, params)

    fx = vmap(fx_fn, in_axes=(0, 0, None))(xs[:-1], us, params)
    fu = vmap(fu_fn, in_axes=(0, 0, None))(xs[:-1], us, params)

    lx  = vmap(lx_fn,  in_axes=(0, 0, None))(xs[:-1], us, params)
    lu  = vmap(lu_fn,  in_axes=(0, 0, None))(xs[:-1], us, params)
    lxx = vmap(lxx_fn, in_axes=(0, 0, None))(xs[:-1], us, params)
    luu = vmap(luu_fn, in_axes=(0, 0, None))(xs[:-1], us, params)
    lux = vmap(lux_fn, in_axes=(0, 0, None))(xs[:-1], us, params)


    Vx_T  = Vx_T_fn(xs[-1], params)
    Vxx_T = Vxx_T_fn(xs[-1], params)

    return {"fx": fx, "fu": fu, "lx": lx, "lu": lu, "lxx": lxx, "luu": luu, "lux": lux,
            "Vx_T": Vx_T, "Vxx_T": Vxx_T}

def backward_pass(derivs, reg):
    fx, fu = derivs["fx"], derivs["fu"]
    lx, lu = derivs["lx"], derivs["lu"]
    lxx, luu, lux = derivs["lxx"], derivs["luu"], derivs["lux"]
    Vx_T, Vxx_T = derivs["Vx_T"], derivs["Vxx_T"]

    T = fx.shape[0]
    nx = Vx_T.shape[-1]
    nu = lu.shape[-1]

    def scan_fun(carry, t_rev):
        t = T - 1 - t_rev
        Vx_next, Vxx_next = carry

        fx_t = fx[t]; fu_t = fu[t]
        lx_t, lu_t = lx[t], lu[t]
        lxx_t, luu_t, lux_t = lxx[t], luu[t], lux[t]

        Qx  = lx_t + fx_t.T @ Vx_next
        Qu  = lu_t + fu_t.T @ Vx_next
        Qxx = lxx_t + fx_t.T @ Vxx_next @ fx_t
        Quu = luu_t + fu_t.T @ Vxx_next @ fu_t
        Qux = lux_t + fu_t.T @ Vxx_next @ fx_t

        Quu_reg = symmetric_psd(Quu, reg)
        k = -jnp.linalg.solve(Quu_reg, Qu)
        K = -jnp.linalg.solve(Quu_reg, Qux)

        Vx = Qx + K.T @ Qu + Qux.T @ k + K.T @ Quu @ k
        Vxx = Qxx + K.T @ Qux + Qux.T @ K + K.T @ Quu @ K
        Vxx = 0.5 * (Vxx + Vxx.T)

        Qu_norm = jnp.linalg.norm(Qu)
        cond = jnp.linalg.cond(Quu_reg)

        out = (K, k, Qu_norm, cond)
        return (Vx, Vxx), out

    init = (Vx_T, Vxx_T)
    (Vx0, Vxx0), outs = lax.scan(scan_fun, init, jnp.arange(T))
    Ks, ks, Qu_norms, conds = outs
    Ks = Ks[::-1]; ks = ks[::-1]

    return Ks, ks, jnp.mean(Qu_norms), jnp.mean(conds)

def forward_apply_policy(dynamics_fn, cost_fn, term_cost_fn,
                         x0, us_nominal, xs_nominal, Ks, ks, alpha, params):
    def step(x, t):
        u_nom = us_nominal[t]
        x_nom = xs_nominal[t]
        K = Ks[t]; k = ks[t]
        u = u_nom + alpha * k + K @ (x - x_nom)
        x_next = dynamics_fn(x, u, params)
        return x_next, (x, u)

    xT, hist = lax.scan(step, x0, jnp.arange(us_nominal.shape[0]))
    xs = jnp.concatenate([hist[0], xT[None, :]], axis=0)
    us = hist[1]
    cost = total_cost(cost_fn, term_cost_fn, xs, us, params)
    return xs, us, cost

# 

def ilqr_ddp_single(problem, dynamics_fn, cost_fn, term_cost_fn, cfg):
    """
    修改版 DDP 求解器：使用 lax.scan 替代 lax.while_loop 以支持端到端反向传播。
    """
    x0, us, params, nx, nu, T = problem.x0, problem.u_init, problem.params, problem.nx, problem.nu, problem.T
    
    # 1. 初始 Rollout
    xs = rollout(dynamics_fn, x0, us, params)
    best_cost = total_cost(cost_fn, term_cost_fn, xs, us, params)
    reg = cfg.reg_init

    Ks = jnp.zeros((T, nu, nx))
    ks = jnp.zeros((T, nu))

    # 初始化状态包
    # 状态包含: (iter_count, regularization, xs, us, Ks, ks, best_cost, converged_flag)
    init_state = (jnp.array(0), reg, xs, us, Ks, ks, best_cost, jnp.array(False))

    # ==========================================================================
    # 定义单步更新函数 (用于 lax.scan)
    # ==========================================================================
    @jax.checkpoint
    def step_fn(state, _):
        (i, reg, xs, us, Ks, ks, best_cost, converged) = state

        # --- 内部更新逻辑 (只在未收敛时执行) ---
        def perform_update(s):
            (i, reg, xs, us, Ks, ks, best_cost, converged) = s
            
            # 1. Backward Pass
            derivs = derivatives(dynamics_fn, cost_fn, term_cost_fn, xs, us, params)
            Ks_new, ks_new, Qu_mean, cond_mean = backward_pass(derivs, reg)

            # 2. Line Search (Forward Pass)
            # 使用内层 scan 遍历 alphas
            def try_alpha(carry, alpha):
                best_tuple = carry
                xs_try, us_try, cost_try = forward_apply_policy(
                    dynamics_fn, cost_fn, term_cost_fn, x0, us, xs, Ks_new, ks_new, alpha, params
                )
                improved = cost_try < best_tuple[2]
                new_best = (xs_try, us_try, cost_try)
                # 如果 improved，更新 best；否则保持原样
                out = jax.tree_util.tree_map(lambda a,b: jnp.where(improved, a, b), new_best, best_tuple)
                return out, None

            init_best = (xs, us, best_cost)
            # 注意：cfg.line_search_alphas 需要是 JAX 数组
            alphas_arr = jnp.array(cfg.line_search_alphas)
            (xs_best, us_best, cost_best), _ = lax.scan(try_alpha, init_best, alphas_arr)

            # 3. Accept / Reject
            accepted = cost_best < best_cost

            xs_next = jax.tree_util.tree_map(lambda a,b: jnp.where(accepted, a, b), xs_best, xs)
            us_next = jax.tree_util.tree_map(lambda a,b: jnp.where(accepted, a, b), us_best, us)
            cost_next = jnp.where(accepted, cost_best, best_cost)
            Ks_out = jax.tree_util.tree_map(lambda a,b: jnp.where(accepted, a, b), Ks_new, Ks)
            ks_out = jax.tree_util.tree_map(lambda a,b: jnp.where(accepted, a, b), ks_new, ks)

            # 4. Update Regularization
            reg_next = jnp.where(accepted, 
                                 jnp.maximum(cfg.reg_init, reg * cfg.reg_mult_dec), 
                                 jnp.minimum(cfg.reg_max, reg * cfg.reg_mult_inc))
            
            # 5. Check Convergence
            # 如果接受了更新 且 梯度范数小于阈值，则视为收敛
            converged_next = jnp.logical_and(accepted, Qu_mean < cfg.tol_g_norm)
            
            return (i + 1, reg_next, xs_next, us_next, Ks_out, ks_out, cost_next, converged_next)
            
        # --- 核心控制流 ---
        # 如果已经收敛 (converged=True)，则直接返回原状态 (Identity)，跳过计算
        # 如果未收敛，执行 perform_update
        next_state = lax.cond(converged, 
                              lambda s: s,             # True branch (不做任何事)
                              perform_update,          # False branch (执行更新)
                              state)
        
        return next_state, None

    # ==========================================================================
    # 主循环替换：while_loop -> scan
    # ==========================================================================
    # 强制运行 max_iters 步 (Fixed computational graph structure)
    final_state, _ = lax.scan(step_fn, init_state, None, length=cfg.max_iters)

    (iters, reg_final, xs, us, Ks, ks, final_cost, converged) = final_state

    # 计算最终 Costs (用于 Logging)
    stage_costs = vmap(lambda x,u: cost_fn(x,u,params))(xs[:-1], us)
    term_c = term_cost_fn(xs[-1], params)
    costs = {"stage": stage_costs, "terminal": term_c, "total": final_cost}

    # 确保返回类型一致
    iters_j     = jnp.asarray(iters, dtype=jnp.int32)
    converged_j = jnp.asarray(converged, dtype=bool)
    reg_final_j = jnp.asarray(reg_final, dtype=jnp.float32)

    return ILQRResult(xs=xs, us=us, Ks=Ks, ks=ks, costs=costs,
                      iters=iters_j,
                      converged=converged_j,
                      final_reg=reg_final_j)

#11.09改

def ilqr_ddp_batched(x0, u_init, params, 
                     dynamics_fn, cost_fn, term_cost_fn, 
                     cfg=None):
    """
    Batched iLQR/DDP implementation.
    参数顺序已修正：数据在前，函数在后，以匹配 partial 绑定逻辑。
    """
    # ==============================================================
    # [关键修改] 强制转换为 float32
    # 解决 lax.scan 输入(float64)与输出(float32)类型不匹配的报错
    # ==============================================================
    x0 = x0.astype(jnp.float32)
    u_init = u_init.astype(jnp.float32)

    if cfg is None:
        cfg = ILQRConfig()

    # 获取 Batch 维度
    B, T, nu = u_init.shape
    nx = x0.shape[-1]

    # 定义 Problem 数据结构
    @dataclass
    class ILQRProblem:
        x0: jnp.ndarray      # [nx]
        u_init: jnp.ndarray  # [T, nu]
        params: Any          # PyTree (single sample)
        nx: int
        nu: int
        T: int

    def make_problem(x0, u_init, params):
        return ILQRProblem(x0=x0, u_init=u_init, params=params, nx=nx, nu=nu, T=T)

    # 定义单次求解的闭包
    def solve_single(x0, u_init, p):
        prob = make_problem(x0, u_init, p)
        return ilqr_ddp_single(prob, dynamics_fn, cost_fn, term_cost_fn, cfg)

    # 并行映射 (vmap)
    # in_axes=(0, 0, 0) 表示 x0, u_init, params 都有 Batch 维
    vmapped = vmap(solve_single, in_axes=(0, 0, 0))

    # 执行并行计算
    return vmapped(x0, u_init, params)


# def ilqr_ddp_batched(x0_b, u_init_b, params_b,
#                      dynamics_fn, cost_fn, term_cost_fn,
#                      cfg):
#     B, T, nu = u_init_b.shape
#     nx = x0_b.shape[-1]

#     @dataclass
#     class ILQRProblem:
#         x0: Array
#         u_init: Array
#         params: Any
#         nx: int
#         nu: int
#         T: int

#     def make_problem(x0, u_init, params):
#         return ILQRProblem(x0=x0, u_init=u_init, params=params, nx=nx, nu=nu, T=T)

#     vmapped = vmap(lambda x0, u_init, p: ilqr_ddp_single(make_problem(x0,u_init,p),
#                                                          dynamics_fn, cost_fn, term_cost_fn, cfg),
#                    in_axes=(0,0, None))

#     return vmapped(x0_b, u_init_b, params_b)

# -------- Example UAV-lite (2D double integrator) --------

def di2d_dynamics(x, u, params):
    dt = params.get("dt", 0.05)
    g  = params.get("g", 0.0)
    px, py, vx, vy = x
    ax, ay = u
    px_n = px + dt * vx
    py_n = py + dt * vy
    vx_n = vx + dt * ax
    vy_n = vy + dt * (ay - g)
    return jnp.array([px_n, py_n, vx_n, vy_n])

def di2d_stage_cost(x, u, params):
    Q = params.get("Q", jnp.diag(jnp.array([1.0, 1.0, 0.1, 0.1])))
    R = params.get("R", 0.01 * jnp.eye(2))
    x_ref = params.get("x_ref", jnp.zeros(4))
    return (x - x_ref) @ Q @ (x - x_ref) + u @ R @ u

def di2d_terminal_cost(x, params):
    Qf = params.get("Qf", 10.0 * jnp.eye(4))
    x_ref = params.get("x_ref", jnp.zeros(4))
    return (x - x_ref) @ Qf @ (x - x_ref)

# def _demo():
#     key = jax.random.PRNGKey(0)
#     B = 4
#     T = 40
#     nx, nu = 4, 2

#     key, k1, k2 = jax.random.split(key, 3)
#     x0_pos = jax.random.uniform(k1, (B, 2), minval=-5.0, maxval=5.0)
#     x0 = jnp.concatenate([x0_pos, jnp.zeros((B,2))], axis=1)
#     u_init = jnp.zeros((B, T, nu))
#     x_ref = jax.random.uniform(k2, (B, nx), minval=-1.0, maxval=1.0)

#     params = {
#         "dt": jnp.full((B,), 0.05),
#         "g": jnp.zeros((B,)),
#         "Q": jnp.tile(jnp.diag(jnp.array([1.0,1.0,0.1,0.1]))[None, ...], (B,1,1)),
#         "R": jnp.tile((0.01*jnp.eye(2))[None, ...], (B,1,1)),
#         "Qf": jnp.tile((10.0*jnp.eye(4))[None, ...], (B,1,1)),
#         "x_ref": x_ref
#     }

#     dyn = lambda x,u,p: di2d_dynamics(x,u,{"dt":p["dt"],"g":p["g"]})
#     stc = lambda x,u,p: (x - p["x_ref"]) @ p["Q"] @ (x - p["x_ref"]) + u @ p["R"] @ u
#     term = lambda x,p: (x - p["x_ref"]) @ p["Qf"] @ (x - p["x_ref"])

#     cfg = ILQRConfig(max_iters=30, tol_g_norm=1e-5)

#     results = ilqr_ddp_batched(x0, u_init, params, dyn, stc, term, cfg)
#     # Print a small summary
#     print("iters:", results.iters)
#     print("converged:", results.converged)
#     print("final positions (px,py):", results.xs[:, -1, :2])

# if __name__ == "__main__":
#     _demo()
