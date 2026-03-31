# 技术落地方案与数值流程

这个文档只描述一件事：

- 当前项目在数值上到底在算什么
- 当前稳定主线是怎么落地的
- JAX 做什么，Julia 做什么，Python/JAX 和 Julia 怎么衔接
- 训练到底在优化什么，梯度现在又是怎么来的

当前维护中的稳定运行主线不是纯 `ipoptax SubP2`，而是：

- `SubP1`: JAX
- `SubP2`: persistent Julia worker
- `SubP3`: JAX
- 外层调度与数据组织：Python

主入口见：

- [run_julia_subp2_fullflow_persistent.py](./run_julia_subp2_fullflow_persistent.py)
- [JustWorkingOnIt.py](./JustWorkingOnIt.py)

---

## 1. 数值对象：系统到底有哪些变量

项目的动力学原始定义在：

- [Dynamics_load_cable_autotuning_2nd_COM_Dyn.py](./Dynamics_load_cable_autotuning_2nd_COM_Dyn.py)

当前默认是 4 架飞机，所以一条完整规划轨迹包含：

### 1.1 负载状态与控制

- 负载状态 `xl`
  - 维度 `13`
  - 典型语义：位置 `p_l(3)`、速度 `v_l(3)`、四元数 `q_l(4)`、角速度 `w_l(3)`
- 负载控制 `ul`
  - 维度 `6`
  - 典型语义：合力 `F_l(3)`、合力矩 `M_l(3)`

### 1.2 每根缆绳/每架飞机状态与控制

- 单根缆绳状态 `xi`
  - 维度 `14`
  - 典型语义：方向 `d_i(3)`、角速度 `w_i(3)`、高阶导数项 `a_i(3), j_i(3)`、张力 `t_i(1)`、张力导数 `dt_i(1)`
- 单根缆绳控制 `ui`
  - 维度 `4`
  - 典型语义：`s_i(3)` 与张力二阶项

4 架飞机叠起来，就得到：

- `xc_traj`: 所有缆绳/飞机状态轨迹
- `uc_traj`: 所有缆绳/飞机控制轨迹

---

## 2. 参考值：项目为什么需要 reference

这个项目不是纯 regulation，而是“参考轨迹跟踪 + ADMM 分解优化”。

当前主线至少有三类参考量：

### 2.1 负载参考轨迹

- `Ref_xl`
- `Ref_ul`

这部分来自负载参考轨迹生成器，例如：

- `minisnap_load_circle(...)`

它定义了“负载应该沿什么轨迹飞”。

### 2.2 缆绳/飞机参考轨迹

- `ref_xq`
- `ref_uq`

这部分描述每根缆绳的方向、张力等参考工作点。

在更真实的 verify 或 fullflow 设置下，会从 stage-1 参考产物中读取：

- `cable_direction_*`
- `tension_magnitude_*`

### 2.3 ADMM 内部共识参考

这不是任务层 reference，而是算法内部 reference：

- `scxl`, `scul`
- `scxc`, `scuc`
- `y_*`
- `rho_*`

其中：

- `scx/scu` 是 ADMM safe-copy / 共识变量
- 各子问题在自己的 cost 里都会去靠近这些量

---

## 3. 前向到底在算什么

一次完整 forward 的输入大致包括：

- `task_idx`
- 该 task 对应的偏心载荷/任务参数 `rg_task`
- 负载参考 `Ref_xl / Ref_ul`
- 缆绳参考 `ref_xq / ref_uq`
- 初始状态 `xl_fb / xq_fb`
- 一组超参数/权重 `weight1 / weight2`

这里的 `weight1 / weight2` 不是控制输入，而是 planner 的超参数：

- 跟踪权重
- 控制权重
- ADMM penalty
- 约束/调度相关权重

前向输出不是一个标量，而是一整段 horizon 上的轨迹包：

- `xl_traj`, `ul_traj`
- `xc_traj`, `uc_traj`
- 以及 ADMM 中间变量与诊断信息

---

## 4. 当前稳定前向主线：三轮 ADMM

主调度器在：

- [jax_ADMM_forward_MPC](./JustWorkingOnIt.py)

当前稳定主线是：

1. `SubP1-load`
2. `SubP1-cable`
3. `SubP2`
4. `SubP3`
5. 重复 3 轮 ADMM

也就是现在 benchmark 里常说的：

- `nq = 4`
- `horizon = 100`
- `ADMM = 3`

### 4.1 一次完整 forward 的 15 个步骤

单次前向规划可以更细地拆成下面这 15 步：

1. 任务解析
   - 读取 `task_idx`
   - 识别当前偏心载荷任务特征 `r_g`
2. 超参生成
   - 神经网络根据任务特征输出 `tunable_l / tunable_i`
3. 参数映射
   - 将 `tunable` 映射成 planner 真正使用的 `weight1 / weight2`
4. 参考生成
   - 生成负载参考 `Ref_xl / Ref_ul`
   - 生成缆绳参考 `ref_xq / ref_uq`
5. 初始化
   - 设置当前 forward 的反馈初值 `xl_fb / xq_fb`
6. 开启 3 轮 ADMM 迭代
7. `SubP1-Load`
   - JAX 解负载轨迹
8. `SubP1-Cable`
   - JAX batched 解 4 根缆绳轨迹
9. 数据对齐
   - 调 `_pack_para2(...)` 组织 `SubP2` 输入语义
10. `SubP2` 调用
   - Python 将 batch 数据通过 bridge 发送给 Julia worker
11. Julia 并行解算
   - 多线程解算 `101` 个单步 NLP
   - 返回 safe-copy 轨迹
12. 解包还原
   - Python 将回包还原成 `scxl / scul / scxc / scuc`
13. `SubP3` 更新
   - JAX 更新对偶变量与共识副本
14. 收敛/循环
   - 进入下一轮 ADMM
15. 最终输出
   - 产出全时域轨迹
   - 同时输出 profile / timing / 误差诊断

### 4.2 SubP1：一共 5 个子问题

`SubP1` 可以理解成：

- 1 个负载子问题
- 4 个缆绳/飞机子问题

它们在每一轮 ADMM 里是分开解的，但并不是彻底互不相干，因为每个子问题都会看到：

- `scx/scu`
- `y`
- `rho`

也就是上一轮 ADMM 传下来的共识与乘子信息。

#### Load 子问题

当前稳定路径：

- [_solve_load_subp1_forward_only](./JustWorkingOnIt.py)

它解的是：

- 负载状态轨迹 `xl`
- 负载控制轨迹 `ul`

其 cost 主要包含：

- 跟踪 `Ref_xl / Ref_ul`
- 靠近 `scxl / scul`
- 乘子 `y_xl / y_ul`
- penalty `rho_lx / rho_lu`

#### Cable 子问题

当前稳定路径：

- [_solve_cable_subp1_forward_only](./JustWorkingOnIt.py)

它解的是 4 根缆绳各自的：

- 状态轨迹 `xc`
- 控制轨迹 `uc`

其 cost 主要包含：

- 跟踪 `ref_xq / ref_uq`
- 靠近 `scxc / scuc`
- 乘子 `y_xc / y_uc`
- penalty `rho_ix / rho_iu`

这里 4 根缆绳维度一致，所以可以 batched/JIT 一起算。

### 4.3 SubP2：约束一致性与投影

`SubP2` 负责处理真正重的约束一致性问题，例如：

- 缆绳方向约束
- 张力与控制边界
- 多根缆绳合成 wrench 与负载侧的一致性
- terminal / non-terminal 不同步的约束差异

当前稳定路径不在 Python 里求，而是：

- Python 打包
- Julia worker 解

### 4.4 SubP3：显式更新

`SubP3` 负责 ADMM 的显式更新：

- 更新乘子
- 更新 safe-copy / 共识相关量

当前稳定路径：

- [jax_ADMM_SubP3](./JustWorkingOnIt.py)

`SubP3` 本身计算量很小，不是当前瓶颈。

---

## 5. JAX 这边怎么建模、怎么求解

JAX 当前主要负责：

- `SubP1`
- `SubP3`
- 初始化 kernel
- 一些诊断/核对用的 objective/residual evaluator

### 5.1 JAX 动力学

JAX 版动力学在：

- [_jax_load_continuous_dynamics](./JustWorkingOnIt.py)
- [jax_load_dynamics](./JustWorkingOnIt.py)
- [_jax_cable_continuous_dynamics](./JustWorkingOnIt.py)
- [jax_cable_dynamics_single](./JustWorkingOnIt.py)

设计思路是：

- 用 `jnp` 写连续动力学
- 用 `RK4` 做离散推进
- 不再依赖 CasADi 作为运行时 forward 的动力学表达

### 5.2 JAX 的求解思路

JAX 没有用通用内点法去解 `SubP1`，而是沿用了时域最优控制的结构：

- `SubP1-load` 用 DDP/iLQR 风格方法
- `SubP1-cable` 也用 DDP/iLQR 风格方法

原因是：

- `SubP1` 有明显的时域动力学结构
- 比起把整段 horizon 摊平成一个大 NLP，DDP 更合适

### 5.3 JAX 的加速策略

当前稳定主线里，JAX 侧主要用了这些策略：

- `jit`
- 对 4 根缆绳做 batch
- forward-only fast path
- 全局 JIT 缓存初始化 kernel

需要注意：

- 不是整个项目所有部分都在 JAX `jit+vmap`
- 当前最重的 `SubP2` 已经不在 JAX 里解

---

## 6. Julia 这边怎么建模、怎么求解

Julia 当前只负责 `SubP2`，而这也是当前最重的求解层。

### 6.1 数学定义层

底层数学定义在：

- [julia_subp2/step_model.jl](./julia_subp2/step_model.jl)

这里定义了：

- 单步变量语义
- `objective`
- `equality_residual`
- `inequality_residual`

它是 Julia 侧的最低层数学描述。

### 6.2 单步参考模型

参考/离线版单步 JuMP 模型在：

- [build_jump_model_eq](./julia_subp2/solve_step_madnlp_jump_native_eq.jl)

这个版本适合阅读“单个 step 的 NLP 本体”，特点是：

- 直接把一个 `StepData` 的数值写进模型
- 更适合离线复现和理解模型结构
- 不是当前 batch 热路径

### 6.3 当前真正运行的 JuMP 模型

当前 batch 主线真正用的是：

- [build_jump_model_eq_parameterized](./julia_subp2/solve_batch_madnlp_jump_native.jl)

它和单步参考模型的差别是：

- 数学问题本体一致
- 但这里会把每轮 ADMM、每个 step 会变化的量做成 `JuMP Parameter`
- 这样可以缓存模型结构，只更新参数，不重建模型

### 6.4 Julia 的求解器栈

当前稳定配置是：

- `JuMP + MadNLP`
- 线性求解器：`mumps`

对应主文件：

- [julia_subp2/solve_batch_madnlp_jump_native.jl](./julia_subp2/solve_batch_madnlp_jump_native.jl)

### 6.5 Julia 的加速策略

Julia 侧当前真正的性能关键不是“单步更聪明”，而是：

- 一个常驻 Julia 进程
- 101 个 step 的 batch 求解
- 每个 step 一个 cached/template solver
- 每轮 ADMM 只更新参数
- 多线程并行解各 step

也就是说，当前 `SubP2` 的实现不是：

- 每个 step 每轮重新建 JuMP 模型
- 每次调用都重启 Julia

而是：

- 模型结构缓存
- 参数刷新
- 多线程并行

---

## 7. 当前 Python 和 Julia 的衔接方式

当前衔接方式是：

- persistent worker
- JSON batch payload

不是：

- 每次调用都重启一个 Julia 进程
- 共享内存
- socket RPC

而是：

- Python 启一个常驻 Julia 子进程
- 双方通过 `stdin/stdout` 管道交换 JSON

Python 侧 worker 封装在：

- [JuliaBatchWorker](./run_julia_subp2_fullflow_persistent.py)

Julia 侧 worker 在：

- [julia_subp2/worker_batch_madnlp_jump_native.jl](./julia_subp2/worker_batch_madnlp_jump_native.jl)

### 7.1 具体通信过程

1. Python 外层在 [run_julia_subp2_fullflow_persistent.py](./run_julia_subp2_fullflow_persistent.py) 里启动 `JuliaBatchWorker`
2. Python 把当前 ADMM 轮所有 step 的 `SubP2` 初值和运行时参数打成一个 batch payload
3. Python 只发送 Julia 真正会用到的字段，不再把无关数据一起打包
4. Python 通过 `stdin` 发送 `solve_batch` 请求
5. Julia worker 在 [julia_subp2/worker_batch_madnlp_jump_native.jl](./julia_subp2/worker_batch_madnlp_jump_native.jl) 里读取 JSON 命令
6. worker 调 [julia_subp2/solve_batch_madnlp_jump_native.jl](./julia_subp2/solve_batch_madnlp_jump_native.jl) 里的 `solve_batch_payload(...)` 做 batch 求解
7. Julia 返回紧凑结果：
   - `x_sol_batch`
   - `iter_batch`
   - `eq_inf_batch`
   - `ineq_vio_batch`
8. Python 从 `stdout` 读回结果，再解包回 planner 所需结构，继续后续 ADMM 流程

### 7.2 是否阻塞

当前是同步 request/response：

- Python 发请求后会阻塞等待 Julia 回包
- Julia 空闲时阻塞在 `readline(stdin)` 等下一条命令

### 7.3 是否有拷贝

有。

这条链路不是零拷贝，至少包括：

- Python/JAX 数据 materialize 到 host
- Python 对象转 JSON
- JSON 写入 pipe
- Julia 从 pipe 读出并重建数组

但当前实际 profiling 里，这层 bridge 已经不是主瓶颈，真正大的时间仍在 Julia 本体求解。

### 7.4 这套衔接方式的关键收益

这套设计里最关键的点是：

- Julia 进程常驻，避免反复启动开销
- 每个 step 的模型结构缓存下来，只更新运行时参数
- Python/Julia 通信只传必要字段，减少桥接损耗
- benchmark 会同时打印 planner 侧 `SubP2 total` 和 worker 侧 `SubP2-only`

最后这一点的意义是：

- `SubP2 total` 更接近 planner 端看到的整段 `SubP2` 时间
- `SubP2-only` 更接近纯 Julia worker 的 wall time
- 这样可以区分“Julia 真在慢”还是“桥接/重建在慢”

---

## 8. Python 侧 `SubP2` 输入到底怎么组织

当前主线不再优先走旧的 `ParaL/ParaC` 路径。

### 8.1 旧打包方式

- `_pack_paraL`
- `_pack_paraC`

这两者主要服务旧的 packaged `SubP1` 路径，现在不是当前稳定主线的核心。

### 8.2 当前 `SubP2` 打包方式

当前真正重要的是：

- [_pack_para2](./JustWorkingOnIt.py)
- [_build_subp2_batch_from_components](./JustWorkingOnIt.py)
- [_prepare_subp2_batch](./JustWorkingOnIt.py)

这条链的作用是：

1. 把 `SubP1` 输出的轨迹与参考、对偶量、penalty 组织成结构化字典
2. 预构造 `SubP2` 标准输入：
   - `w_init`
   - `params_batch`
3. runner 再把它变成发给 Julia 的 runtime payload

这里的 `_prebuilt_subp2_batch` 不是跨轮缓存，而是：

- 同一轮里提前构造好的 `(w_init, params_batch)`
- 后面直接拿来用，避免重复打包

---

## 9. 多线程并行在哪里发生

### 9.1 JAX 侧

- 4 根缆绳 `SubP1` 可以 batched
- 负载 `SubP1` 单独一个 solver

负载和缆绳不能自然合并成一个统一 batch/JIT 问题，因为：

- 状态/控制维度不同
- 动力学不同
- cost 不同

所以当前正确的结构是：

- 负载自己 JIT
- 缆绳 4 根一起 batch/JIT

### 9.2 Julia 侧

Julia 的并行发生在 `SubP2` 的 step 级别：

- 一个 batch 里有 `N+1 = 101` 个 step
- 每个 step 对应一个单步 NLP
- Julia 用 `Threads.@threads` 把这些 step 分发到多个线程

也就是说：

- 不是 101 个 Julia 进程
- 不是 Python 并行调 101 次
- 而是 1 个 Julia 进程内，多线程并行解 101 个 step

### 9.3 锁的作用

Julia 代码里有锁，但主要只保护：

- 全局 cache
- template 容器

正常的 step 求解热路径基本是：

- 每个线程拿自己的 `template[idx]`
- 写自己的 `x_sol_batch[idx, :]`

不是靠细粒度锁来做主线并行。

---

## 10. 训练在干什么

这个项目里的神经网络不是直接输出轨迹，也不是直接输出控制。

它学的是：

- 给定任务特征
- 应该如何生成 planner 的超参数/权重

也就是：

- `task -> NN -> weight1/weight2 -> planner forward -> 更好的轨迹`

当前权重映射主要在：

- [Gradient_Solver](./JustWorkingOnIt.py)

相关函数例如：

- `Set_Parameters_nn_l`
- `Set_Parameters_nn_i`

它们会把 NN 输出的 `tunable_l / tunable_i` 映射成真正给 planner 用的参数。

---

## 11. 训练 loss 在优化什么

训练里最核心的不是“让网络拟合一个动作”，而是：

- 让 planner 解出来的轨迹更接近参考
- 让 ADMM 一致性更好

大致包含两类量：

- tracking loss
- residual / consistency loss

这些最终会组合成 meta-loss。

---

## 12. 当前训练梯度是怎么来的

这里非常重要：

- 当前训练不是对整条 forward 做端到端 `jax.grad(loss_fn)`
- 不是在当前 `JAX + Julia` 运行时主线上做 VJP

当前训练梯度的本质是：

- forward 先正常求轨迹
- 再用解析敏感度公式传播梯度

### 12.1 `Get_AuxSys_*`

这组函数负责：

- 在 forward 轨迹对应的工作点上
- 计算局部 Jacobian/Hessian
- 整理成辅助导数材料包

例如：

- `Get_AuxSys_DDP_Load`
- `Get_AuxSys_DDP_Cable`
- `Get_AuxSys_SubP2`
- `Get_AuxSys_SubP3`

当前这些局部导数很多还是来自 CasADi 自动微分。

### 12.2 `*_Gradient`

这组函数负责：

- 根据上面的局部导数材料
- 按手推的解析递推公式
- 真正把参数梯度沿时间和 ADMM 轮传播下去

例如：

- `DDP_Load_Gradient`
- `DDP_Cable_Gradient`
- `SubP2_Gradient`
- `SubP3_Gradient`

所以当前训练梯度体系更准确地说是：

- 局部导数：CasADi 自动/符号微分
- 全局梯度传播：手工解析递推

不是端到端 autodiff。

---

## 13. 如果后面要把训练求导迁到 JAX，应该怎么理解

一个合理的方向不是：

- 让当前 `JAX + Julia` forward 整体变成可反传 graph

而是：

- forward 继续按现在的稳定主线跑
- 训练梯度模块单独迁到 JAX

更具体地说：

- `Get_AuxSys_*`
  - 用 JAX 纯函数重写局部数学表达式
  - 用 `jax.jacfwd / jax.hessian` 生成局部导数
- `*_Gradient`
  - 保留解析递推思路
  - 把实现改成 `jnp`、`scan`、`vmap`、JAX 线代

其中：

- `SubP2_Gradient` 数学上也是解析解
- 但工程上更难，因为它依赖的 Hessian block 最多、数值最敏感、线性代数规模最大

---

## 14. 当前主线为什么这样分工

这套 `JAX + Julia` 分工不是偶然，而是和问题结构匹配：

- `SubP1`
  - 是典型时域动力学优化
  - 适合 DDP/iLQR 风格
  - JAX 很适合做纯函数、JIT、batch
- `SubP2`
  - 是每个时间步上的 constrained NLP
  - 更适合 `JuMP + MadNLP`
  - Julia 适合参数化模型缓存和多线程 step 并行
- `SubP3`
  - 是显式更新
  - JAX 做起来最直接

所以当前主线的核心思路是：

- 把“动态结构强”的部分留给 JAX
- 把“约束重、NLP 求解重”的部分交给 Julia solver 栈

---

## 15. 一句话总结

这个项目不是“网络直接控飞机”，而是：

- 网络给 planner 生成超参数
- planner 再通过 `SubP1 + SubP2 + SubP3` 的 3 轮 ADMM
- 解出 4 架飞机和负载在 `horizon=100` 上的一整段轨迹

当前稳定落地方案就是：

- `SubP1`: JAX
- `SubP2`: Python batch + persistent Julia worker + JuMP/MadNLP
- `SubP3`: JAX
- 训练梯度：暂时仍是旧的解析梯度体系，尚未完全迁到 JAX
