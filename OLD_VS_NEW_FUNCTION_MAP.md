# 新旧函数对照

这个文档只做一件事：

- 对比原版 `CasADi/DDP/Ipopt` 路径里常见的关键函数
- 和当前稳定主线 `JAX SubP1 + Python bridge + persistent Julia SubP2 + JAX SubP3`
- 说明它们现在分别在哪、是否仍在主线中、关系是什么

不追求把所有历史函数都列全，重点只放在当前最容易混淆的主链。

## 1. 总体结构

当前项目里其实同时保留了两套前向实现：

- 旧版主线：`MPC_Planner.ADMM_forward_MPC(...)`
- 当前稳定主线：`MPC_Planner.jax_ADMM_forward_MPC(...)`

其中：

- 旧版主线主要靠 CasADi/DDP/Ipopt
- 当前稳定主线主要靠 JAX 做 `SubP1/SubP3`，Julia 做 `SubP2`

## 2. 顶层总入口

| 角色 | 原版函数 | 当前函数 | 说明 |
|---|---|---|---|
| 整条前向规划入口 | [ADMM_forward_MPC](./JustWorkingOnIt.py) | [jax_ADMM_forward_MPC](./JustWorkingOnIt.py) | 两者都是“4 架飞机、horizon=100、ADMM 3 轮”的总控入口 |
| 外部正式 runner | 无单独 Python-Julia runner | [run_julia_subp2_fullflow_persistent.py](./run_julia_subp2_fullflow_persistent.py) | 当前稳定 benchmark / verify / Julia worker 启动入口 |

说明：

- `ADMM_forward_MPC(...)` 是原版完整 forward
- `jax_ADMM_forward_MPC(...)` 是现在实际维护的 Python 主线
- 根目录 runner 会 monkey-patch `jax_ADMM_SubP2(...)`，把它改接到 Julia worker

## 3. SubP1：负载与缆绳前向

### 3.1 当前稳定主线

| 子块 | 当前主线函数 | 说明 |
|---|---|---|
| Load fast path | [_solve_load_subp1_forward_only](./JustWorkingOnIt.py) | 当前默认主线直接调用，只返回前向轨迹，不提导数 |
| Cable fast path | [_solve_cable_subp1_forward_only](./JustWorkingOnIt.py) | 当前默认主线直接调用，batched/JIT 路径 |
| 旧 JAX packaged 入口 | [jax_MPC_Load_DDP_Planning_SubP1](./JustWorkingOnIt.py) | 仍保留，但不是当前稳定默认路径 |
| 旧 JAX packaged 入口 | [jax_MPC_Cable_DDP_Planning_SubP1](./JustWorkingOnIt.py) | 同上 |

### 3.2 原版对应函数

| 子块 | 原版函数 | 说明 |
|---|---|---|
| Load DDP | [DDP_Load_ADMM_Subp1](./JustWorkingOnIt.py) | 原版 `SubP1-load` 的 DDP 求解器 |
| Cable DDP | [DDP_Cable_ADMM_Subp1](./JustWorkingOnIt.py) | 原版 `SubP1-cable` 的 DDP 求解器 |

### 3.3 对照理解

- 原版：每轮 ADMM 都用 CasADi/DDP 走 load 和 4 根 cable
- 当前：默认直接走 JAX fast path，不再优先走 packaged/legacy DDP 包装器

## 4. SubP2：一致性/约束子问题

### 4.1 Python 侧入口

| 角色 | 原版函数 | 当前函数 | 说明 |
|---|---|---|---|
| SubP2 主入口 | [ADMM_SubP2](./JustWorkingOnIt.py) | [jax_ADMM_SubP2](./JustWorkingOnIt.py) | 当前 `jax_ADMM_SubP2` 在稳定 runner 下会被 monkey-patch 到 Julia |
| 打包 SubP2 输入 | 手工长向量 `Para2_cable` 拼接 | [_pack_para2](./JustWorkingOnIt.py) | 当前主线优先生成结构化 dict + `_prebuilt_subp2_batch` |
| 结构化 batch 准备 | 无对应统一 helper | [_prepare_subp2_batch](./JustWorkingOnIt.py) | 从 `Para2` 取或重建 `(w_init, params_batch)` |
| 结果解包 | 旧版直接返回各类数组 | [_unpack_subp2_results](./JustWorkingOnIt.py) | 把 Julia/新 JAX `w_opt_batch` 拆回 `scxl/scul/scxc/scuc` |

### 4.2 Runner / bridge 层

| 角色 | 当前函数 | 说明 |
|---|---|---|
| runtime payload 打包 | [build_batch_export_runtime](./run_julia_subp2_fullflow_persistent.py) | 把 Python 侧 batch 整理成发给 Julia 的紧凑 payload |
| worker 管理 | [JuliaBatchWorker](./run_julia_subp2_fullflow_persistent.py) | 启动常驻 Julia 子进程，通过 `stdin/stdout` 做 JSON IPC |
| monkey-patch 接缝 | [install_julia_subp2_patch](./run_julia_subp2_fullflow_persistent.py) | 运行时替换 `MPC_Planner.jax_ADMM_SubP2` |

### 4.3 Julia 侧真实主线

| 角色 | 当前函数 | 说明 |
|---|---|---|
| batch 主入口 | [solve_batch_payload](./julia_subp2/solve_batch_madnlp_jump_native.jl) | 当前 `SubP2` 真正的主入口 |
| 参数化 JuMP 模型 | [build_jump_model_eq_parameterized](./julia_subp2/solve_batch_madnlp_jump_native.jl) | 当前主线真实在跑的单步 JuMP 模型 |
| cached solver 构建 | [build_cached_step_solver](./julia_subp2/solve_batch_madnlp_jump_native.jl) | 给每个 step 建可复用 solver 槽位 |
| template 准备 | [_ensure_stacked_batch_template!](./julia_subp2/solve_batch_madnlp_jump_native.jl) | 准备 `101` 个 step 的 cache/template |
| 每步参数更新 | [update_parameterized_step_from_stacked!](./julia_subp2/solve_batch_madnlp_jump_native.jl) | 将当前 step 的 runtime 参数刷入模型 |
| 每步求解 | [solve_cached_step_compact!](./julia_subp2/solve_batch_madnlp_jump_native.jl) | 单步 `optimize!` + 紧凑回包 |
| 每步主线包装 | [_solve_stacked_cached_step!](./julia_subp2/solve_batch_madnlp_jump_native.jl) | 串起参数更新和求解 |

### 4.4 单步参考模型

| 角色 | 函数 | 说明 |
|---|---|---|
| 单步参考/离线版 JuMP 模型 | [build_jump_model_eq](./julia_subp2/solve_step_madnlp_jump_native_eq.jl) | 更适合阅读单个 step 的 NLP 本体，不是当前 batch 热路径 |

### 4.5 对照理解

- 原版 `ADMM_SubP2(...)`：单机 CasADi/Ipopt 路径
- 当前 `jax_ADMM_SubP2(...)`：接口名还在，但稳定主线下由 runner 替换成 Julia worker 调用
- 当前最核心的 `SubP2` 代码已经不在 Python 求解器里，而在 Julia 的 `solve_batch_payload(...)`

## 5. SubP3：ADMM 显式更新

| 角色 | 原版函数 | 当前函数 | 说明 |
|---|---|---|---|
| ADMM dual/update | [ADMM_SubP3](./JustWorkingOnIt.py) | [jax_ADMM_SubP3](./JustWorkingOnIt.py) | 两者都属于显式更新；当前 JAX 版是主线 |

说明：

- `SubP3` 本身计算量很小
- 主要作用是更新 `scxL/scuL/scxC/scuC`

## 6. 打包函数：旧接口 vs 当前接口

| 角色 | 旧函数 | 当前函数 | 状态 |
|---|---|---|---|
| SubP1 load 打包 | [_pack_paraL](./JustWorkingOnIt.py) | 无需主线依赖 | 旧 packaged 路径保留 |
| SubP1 cable 打包 | [_pack_paraC](./JustWorkingOnIt.py) | 无需主线依赖 | 旧 packaged 路径保留 |
| SubP2 结构化打包 | 无统一 helper | [_pack_para2](./JustWorkingOnIt.py) | 当前主线核心 |

说明：

- `_pack_paraL/_pack_paraC` 现在主要服务 legacy packaged `SubP1`
- `_pack_para2` 才是当前主线的 `SubP2` 输入组织器

## 7. 动力学：原版 vs JAX 版

### 原版 CasADi 动力学

文件：
- [Dynamics_load_cable_autotuning_2nd_COM_Dyn.py](./Dynamics_load_cable_autotuning_2nd_COM_Dyn.py)

关键函数：
- `payload_dyn(...)`
- `cable_dyn(...)`
- `model()`

### 当前主线 JAX 动力学

文件：
- [JustWorkingOnIt.py](./JustWorkingOnIt.py)

关键函数：
- [_jax_load_continuous_dynamics](./JustWorkingOnIt.py)
- [jax_load_dynamics](./JustWorkingOnIt.py)
- [_jax_cable_continuous_dynamics](./JustWorkingOnIt.py)
- [jax_cable_dynamics_single](./JustWorkingOnIt.py)

说明：

- 当前运行时 `SubP1` fast path 和初始化，已经优先走 JAX 动力学
- 原版 CasADi 动力学仍被旧路径和训练梯度代码继续使用

## 8. 训练求导：当前仍主要是旧体系

| 角色 | 当前位置 | 说明 |
|---|---|---|
| 训练 loss/参数映射 | [Gradient_Solver](./JustWorkingOnIt.py) | 负责 meta-loss、参数映射和链式汇总 |
| 上游梯度递推 | [MPC_Planner](./JustWorkingOnIt.py) | `Get_AuxSys_*`、`DDP_*_Gradient`、`SubP2_Gradient`、`SubP3_Gradient` 仍在旧体系里 |

当前状态：

- 运行时 forward 主线已经迁到 `JAX + Julia`
- 训练求导主线还没有完全迁到 JAX
- 训练梯度不是对当前 forward 端到端 VJP，而是旧的解析梯度/辅助系统路线

## 9. 最短阅读顺序

如果只想搞懂“当前新主线”：

1. [jax_ADMM_forward_MPC](./JustWorkingOnIt.py)
2. [_solve_load_subp1_forward_only](./JustWorkingOnIt.py)
3. [_solve_cable_subp1_forward_only](./JustWorkingOnIt.py)
4. [_pack_para2](./JustWorkingOnIt.py)
5. [_prepare_subp2_batch](./JustWorkingOnIt.py)
6. [install_julia_subp2_patch](./run_julia_subp2_fullflow_persistent.py)
7. [solve_batch_payload](./julia_subp2/solve_batch_madnlp_jump_native.jl)
8. [build_jump_model_eq_parameterized](./julia_subp2/solve_batch_madnlp_jump_native.jl)
9. [jax_ADMM_SubP3](./JustWorkingOnIt.py)

如果想搞懂“旧版训练梯度链”：

1. [ADMM_Gradient_Solver](./JustWorkingOnIt.py)
2. [Get_AuxSys_DDP_Load](./JustWorkingOnIt.py)
3. [DDP_Load_Gradient](./JustWorkingOnIt.py)
4. [Get_AuxSys_DDP_Cable](./JustWorkingOnIt.py)
5. [DDP_Cable_Gradient](./JustWorkingOnIt.py)
6. [Get_AuxSys_SubP2](./JustWorkingOnIt.py)
7. [SubP2_Gradient](./JustWorkingOnIt.py)
8. [Get_AuxSys_SubP3](./JustWorkingOnIt.py)
9. [SubP3_Gradient](./JustWorkingOnIt.py)
10. [Gradient_Solver](./JustWorkingOnIt.py)
