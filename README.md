# meta-learn_useThis

当前还在维护的性能主线不是纯 `ipoptax SubP2`，而是：

- 外层 planner: Python/JAX
- `SubP1`: JAX
- `SubP2`: persistent Julia worker
- `SubP3`: Python/JAX

这条主线的目标很简单：保留 Python/JAX 外层流程和当前 `SubP1` 的速度优势，同时把最重的 `SubP2` 交给 Julia 里的 `JuMP + MadNLP` 去解，并且用常驻 worker 避免每轮重复建模和重复启动。

## 当前性能结论

这里有两组不能直接混在一起看的时间数据：

### 1. 历史优化基线：简化 verify/reference 条件

这组更快的数字，来自我们前一阶段的 formal benchmark。  
当时的 verify/reference 条件更简化，缆绳初值/参考还没有切到现在这套 task-specific stage-1 reference 读取逻辑，等价于“更理想化、对 `SubP2` 更顺手”的验证输入。

在那组条件下，warm `repeat=2` 的代表性结果是：

| component | original CasADi/IPOPT | current JAX + Julia |
| --- | ---: | ---: |
| `SubP1 total` | `2086.86 ms` | `133.97 ms` |
| `SubP2 total` | `3634.27 ms` | `864.91 ms` |
| `SubP2-only` | - | `817.84 ms` |
| `SubP3 total` | `13.22 ms` | `1.11 ms` |
| `fullflow total` | `5734.35 ms` | `1000.72 ms` |

这组数字对“优化过程里一度做到多快”很有价值，但现在应该理解成：

- 这是在更简化的 verify/reference 条件下得到的历史基线
- 不应和当前更真实的 stage-1 reference 条件直接混为一谈

### 2. 当前更真实的 verify 条件：读取 stage-1 cable reference

现在 `verify_jax_vs_original.py` 和 `run_julia_subp2_fullflow_persistent.py` 都会读取 stage-1 的 task-specific 缆绳参考。

按当前默认配置，代码期望的是：

- `Planning_plots_meta_COM_Dyn/cable_direction_<i_train_1>_0_4_3_n.npy`
- `Planning_plots_meta_COM_Dyn/tension_magnitude_<i_train_1>_0_4_3_n.npy`

但当前工作区里并没有这组 `model1=4` 文件，所以当前实际运行会 fallback 到：

- `Planning_plots_meta_COM_Dyn/cable_direction_18_0_2_3_n.npy`
- `Planning_plots_meta_COM_Dyn/tension_magnitude_18_0_2_3_n.npy`

也就是日志里的：

- `Using stage-1 cable references: i_train_1=18, model1=2`

在这组更真实、但又不完全匹配当前 stage-2 默认模型的 reference 条件下，最近一次 warm `repeat=2` 主线复跑结果是：

- original CasADi/IPOPT fullflow: `5590.49 ms`
- current JAX + persistent Julia fullflow: `1448.62 ms`
- current Julia `SubP2-only`: `1250.37 ms`

对应误差是：

- `load_state_xl rmse = 6.18e-05`
- `load_control_ul rmse = 6.08e-05`
- `cable_state_xc rmse = 1.25e-04`

这说明：

- 当前 Julia 主线仍然明显快于 original
- 但它比前面那组 `~1.0s` 的历史快值更慢

当前更慢的一个很可能原因是：

- 现在不再使用更简化的 verify/reference 输入
- 同时 stage-1 reference 还 fallback 到了 `model1=2`
- 这和当前默认 stage-2 权重的 `model=4` 不完全匹配，可能让 `SubP2` 更难解

## 快速理解

如果你想快速建立整体理解，建议先看：

- [`TECHNICAL_PIPELINE.md`](./TECHNICAL_PIPELINE.md)

jax 版本和原版函数的总体对应关系：
- [`OLD_VS_NEW_FUNCTION_MAP.md`](./OLD_VS_NEW_FUNCTION_MAP.md)

## 目录

- `JustWorkingOnIt.py`
  当前 JAX planner 主实现。
- `verify_jax_vs_original.py`
  对比原版 CasADi 路线和当前 JAX 路线的数值结果。
- `run_julia_subp2_fullflow_persistent.py`
  当前最重要的正式 benchmark 入口。它会跑 original，并跑 `JAX + persistent Julia SubP2`，最后给出时间和误差。
- `julia_subp2/worker_batch_madnlp_jump_native.jl`
  常驻 Julia worker，走 stdin/stdout JSON 协议。
- `julia_subp2/solve_batch_madnlp_jump_native.jl`
  Julia 侧 batch `SubP2` 求解主逻辑，负责缓存模型、更新参数、批量求解。
- `julia_subp2/step_model.jl`
  单步 `SubP2` 模型结构定义。
- [`TECHNICAL_PIPELINE.md`](./TECHNICAL_PIPELINE.md)
  当前技术落地方案与数值流程总览。
- [`OLD_VS_NEW_FUNCTION_MAP.md`](./OLD_VS_NEW_FUNCTION_MAP.md)
  原版函数和当前稳定主线函数的对照表。
- `archive/`
- `python_archive/archive/`
- `julia_subp2/archive/`
  已归档的试验脚本和一次性诊断代码，不属于当前默认主线。

## 环境准备

Python 侧当前维护机器上使用的是这个解释器：

```bash
/home/mpc/miniconda3/envs/xirui/bin/python
```

但这只是当前维护机的参考路径，不是 README 强依赖。

在别的机器上，更重要的是：

- 你要进入一个等价的 Python 环境
- 这个环境要满足下面列出的 Python 版本和关键模块版本
- 后面的命令示例都默认你已经激活好这个环境，此时直接用 `python` 即可

当前机器上这套解释器的版本是：

- `Python 3.11.13`

当前稳定主线实际依赖的关键 Python 模块版本如下：

| module | version | 说明 |
| --- | --- | --- |
| `jax` | `0.9.0.1` | JAX 主库 |
| `jaxlib` | `0.9.0.1` | XLA/JAX 后端 |
| `jaxopt` | `0.8.5` | 一些 JAX 优化器依赖 |
| `numpy` | `2.4.2` | 数组基础库 |
| `scipy` | `1.15.3` | 数值计算基础库 |
| `torch` | `2.7.0` | 当前 NN 权重加载与前向推理 |
| `matplotlib` | `3.10.0` | verify / benchmark 画图 |
| `casadi` | `3.7.1` | 原版路径与旧梯度链仍在用 |

补充两点：

- 仓库里还有一个本地目录 [`ipoptax/`](./ipoptax)，它是 vendored 源码，不是这里单独 `pip show` 出来的第三方包。
- 上面这些版本是当前 `xirui` 环境的实测结果；如果换了 conda env，README 里的 benchmark 结论就不再保证完全可复现。

Julia 侧先初始化 `julia_subp2` 环境：

```bash
JULIA_DEPOT_PATH=/tmp/julia-depot:/home/mpc/.julia julia julia_subp2/setup_env.jl
```

当前机器上的 Julia 版本是：

- `Julia 1.12.5`

`julia_subp2/Project.toml` 当前关键 Julia 包版本如下：

| package | version | 说明 |
| --- | --- | --- |
| `JuMP` | `1.30.0` | Julia 侧建模层 |
| `MadNLP` | `0.9.1` | 当前稳定 `SubP2` 主求解器 |
| `JSON3` | `1.14.3` | Python/Julia JSON IPC |
| `NLPModels` | `0.21.11` | NLP 抽象层依赖 |
| `NLPModelsJuMP` | `0.13.5` | JuMP 与 NLPModels 适配层 |
| `PythonCall` | `0.9.31` | Python/Julia 互操作依赖 |
| `CUDA` | `5.10.0` | GPU 相关依赖，当前稳定 CPU 主线默认不用 |
| `ExaModels` | `0.9.6` | 研究/实验路径依赖 |
| `CUDSS` | `0.6.7` | GPU 线性求解器实验依赖 |
| `MadNLPGPU` | `0.8.0` | GPU 实验依赖 |

补充两点：

- 上面这些版本来自当前 `julia_subp2` project 的实测 `Pkg.status()`。
- `CUDSS` 和 `MadNLPGPU` 虽然在 project 里，但当前稳定 CPU 主线默认不依赖；有些环境下它们可能尚未实际下载。


## 如何跑

### 1. 正式 benchmark

这是当前最权威的复现命令。它会同时跑：

- original CasADi/IPOPT 全流程
- 当前 JAX + persistent Julia 全流程
- 时间对比
- 轨迹误差对比

下面这条命令默认你已经激活了满足上文版本要求的 Python 环境。

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl \
VERIFY_TASK_IDX=0 VERIFY_HORIZON=100 VERIFY_ADMM_ITERS=3 \
JAX_PLATFORMS=cpu \
python -u \
run_julia_subp2_fullflow_persistent.py \
  --task-idx 0 \
  --repeats 2 \
  --julia-threads 20 \
  --julia-blas-threads 1 \
  --julia-linear-solver mumps \
  --julia-kkt-system default \
  --julia-callback default \
  --julia-thread-schedule static
```

这条命令里的每一项含义如下：

- `MPLBACKEND=Agg`
  - 让 `matplotlib` 使用无界面后端
  - 适合服务器/终端环境
- `MPLCONFIGDIR=/tmp/mpl`
  - 把 `matplotlib` 的配置和缓存写到 `/tmp/mpl`
  - 避免共享机器上的权限问题
- `VERIFY_TASK_IDX=0`
  - 传给 `verify_jax_vs_original.py` 的任务编号
  - 这里表示复现实验用 `task 0`
- `VERIFY_HORIZON=100`
  - 传给 verify 层的 MPC 时域长度
  - 这里固定成 `100`
- `VERIFY_ADMM_ITERS=3`
  - 传给 verify 层的 ADMM 轮数
  - 这里固定成 `3`
- `JAX_PLATFORMS=cpu`
  - 强制 JAX 使用 CPU backend
  - 保证和当前稳定基线一致
- `python -u`
  - 使用你当前已经激活好的 Python 环境
  - `-u` 表示无缓冲输出，方便实时看日志
- `run_julia_subp2_fullflow_persistent.py`
  - 当前正式 benchmark 入口
  - 会启动 Julia worker，并 monkey-patch `jax_ADMM_SubP2(...)`
- `--task-idx 0`
  - 传给 runner 自己的任务编号
  - 和上面的 `VERIFY_TASK_IDX=0` 保持一致
- `--repeats 2`
  - 整个 benchmark 连跑两次
  - `repeat=1` 通常包含更多冷启动成本
  - `repeat=2` 更适合当 warm run 看
- `--julia-threads 20`
  - Julia worker 的线程数
  - 当前主机的稳定 CPU 主线按 `20` 线程记录基线
- `--julia-blas-threads 1`
  - Julia 进程里底层 BLAS 线程数
  - 固定成 `1` 是为了避免和外层 Julia 线程过度竞争
- `--julia-linear-solver mumps`
  - Julia `SubP2` 里 MadNLP 使用的线性求解器
  - 当前稳定主线固定为 `mumps`
- `--julia-kkt-system default`
  - Julia `SubP2` 的 KKT system 配置
  - 当前稳定主线固定为 `default`
  - 当前代码里支持的取值有：
    - `default`
      - 不显式指定 KKT system，交给当前 `MadNLP` 后端默认配置
    - `dense`
      - 显式请求 `MadNLP.DenseKKTSystem`
    - `dense_condensed`
      - 显式请求 `MadNLP.DenseCondensedKKTSystem`
    - `sparse`
      - 显式请求 `MadNLP.SparseKKTSystem`
    - `sparse_condensed`
      - 显式请求 `MadNLP.SparseCondensedKKTSystem`
  - 这些取值最终由 Julia 侧 [`_resolve_kkt_system(...)`](./julia_subp2/solve_batch_madnlp_jump_native.jl) 解释
- `--julia-callback default`
  - Julia `SubP2` 的 callback 配置
  - 当前稳定主线固定为 `default`
  - 当前代码里支持的取值只有：
    - `default`
      - 不显式指定 callback，交给当前 `MadNLP` 后端默认配置
    - `dense`
      - 显式请求 `MadNLP.DenseCallback`
  - 这些取值最终由 Julia 侧 [`_resolve_callback(...)`](./julia_subp2/solve_batch_madnlp_jump_native.jl) 解释
- `--julia-thread-schedule static`
  - Julia worker 在 step 级并行时使用的线程调度策略
  - 当前主机上正式复现建议固定 `static`
  - 当前代码里支持的取值有：
    - `default`
      - 直接使用 `Threads.@threads` 默认调度
    - `dynamic`
      - 使用 `Threads.@threads :dynamic`
    - `static`
      - 使用 `Threads.@threads :static`
  - 当前主机上我们做过对比后，正式复现建议固定 `static`

## 跑代码的注意点

### julia是靠[`run_julia_subp2_fullflow_persistent.py`](./run_julia_subp2_fullflow_persistent.py)起的

- `verify_jax_vs_original.py` 默认 **不会** 启动 Julia worker
- 它默认走的是 Python 侧原有的 `jax_ADMM_SubP2(...)`，也就是旧的纯 JAX / `ipoptax SubP2` 验证路径

它的作用是：做数值对齐，画对比图，检查当前 JAX 路线和 original CasADi 路线的误差。

只有在根目录正式 runner：

- [`run_julia_subp2_fullflow_persistent.py`](./run_julia_subp2_fullflow_persistent.py)

里，才会：

- 启动 persistent Julia worker
- 通过 monkey-patch 把 `MPC_Planner.jax_ADMM_SubP2(...)` 偷换成 Julia IPC 版本
---
### 参考文件的问题：似乎缺了文件 

当前这条 verify 链还会读取 stage-1 的缆绳参考文件：

- `Planning_plots_meta_COM_Dyn/cable_direction_*`
- `Planning_plots_meta_COM_Dyn/tension_magnitude_*`

当前 fallback 逻辑是：

- 必须至少找到一组与当前 `task_idx / ADMM / mode` 匹配的 stage-1 reference
- 但**不要求**它一定是你指定的 `initial_model_stage1`
- 如果指定的 `initial_model_stage1` 没有对应文件，代码会 fallback 到目录里当前可用的其他 model
- 如果同一个 `task / ADMM / mode` 下完全没有任何 reference 文件，当前代码会直接报错退出

也就是说，当前要求是：

- 必须有“某个可用 reference 文件”
- 但不一定必须是“你最想要的那个 reference 文件”

按当前默认配置：

- `task_idx = 0`
- `initial_model_stage1 = 4`
- `max_iter_admm_stage1 = 3`
- `weight_mode_stage1 = n`

代码期望的 stage-1 reference 文件模式是：

- `Planning_plots_meta_COM_Dyn/cable_direction_<i_train_1>_0_4_3_n.npy`
- `Planning_plots_meta_COM_Dyn/tension_magnitude_<i_train_1>_0_4_3_n.npy`

但当前工作区里这组 `model1=4` 文件并不存在。

所以当前实际运行时，task 0 会 fallback 到目录里现有的可用 reference。
在我们最近这组实验里，最终实际使用的是：

- `Planning_plots_meta_COM_Dyn/cable_direction_18_0_2_3_n.npy`
- `Planning_plots_meta_COM_Dyn/tension_magnitude_18_0_2_3_n.npy`

日志里会表现为：

- `Using stage-1 cable references: i_train_1=18, model1=2`
---
### 参考的 horizon 似乎有 N 和N+1 的不统一

- `run_julia_subp2_fullflow_persistent.py` 内部也是调用 `verify_jax_planner(...)`
- 所以它沿用的是同一套 stage-1 reference fallback 思路
- 当前 verify / Julia 主线还会对长度只有 `N` 的 reference 做补齐，自动扩成 `N+1`

这里的 `N` / `N+1` 问题，原因是历史 reference 文件的保存约定并不完全统一：

- 对 horizon=`N` 的 MPC，通常会有：
  - `N+1` 个状态节点
  - `N` 个控制步
- `di_ref / ti_ref` 更像状态参考，所以从当前 planner 角度看，更自然希望拿到 `N+1` 个点
- 但一部分历史 stage-1 产物只存了 `N` 个点，没有单独把 terminal 状态再存一份

所以当前 verify / Julia 主线做了兼容处理：

- 如果 reference 只有 `N` 个点，就把最后一个点复制一份，补成 `N+1`
- 如果本来就是 `N+1`，就直接使用

也就是说，现在不是数学定义变了，而是历史文件有两种长度约定，当前代码在做兼容。
---
## 单独跑 JAX 或原版 forward（现在跑不通）

这两个脚本主要用于单路 forward 演示，不是正式 benchmark 入口：

```bash
python run_jax_forward_with_trained_nn.py --task-idx 0 --backend cpu
python run_original_forward_with_trained_nn.py --task-idx 0
```

它们依赖仓库里的训练权重和参考轨迹数据目录。
---
## 为什么从 `ipoptax` 切到 Julia `SubP2`

这次切换不是因为 JAX 外层没价值，而是因为真正的瓶颈逐渐收敛到了 `SubP2`。

优化过程大致是：

1. 先把 forward 逻辑和 `SubP1` 对齐到原版 CasADi/IPOPT。
2. 把 `SubP1` 做成 batched/JIT 路线，先把非 `SubP2` 成本压下去。
3. 继续尝试纯 JAX `ipoptax SubP2`。
4. 发现纯 JAX 路线在小实验上能改进，但在真实 fullflow、真实 batch 规模下，仍然不如成熟 NLP solver 栈稳定，也不够快。
5. 所以主线改成“Python/JAX 外层保留，`SubP2` 切到 Julia worker”。

这么做的核心原因有两个：

- `SubP1` 在 JAX 里已经很快，继续迁走没有必要。
- `SubP2` 本质上是大量结构相同、参数不同的 NLP，更适合交给 `JuMP + MadNLP` 这一层处理。

换句话说，这次不是“放弃 JAX”，而是把 JAX 用在它已经证明有效的部分，把最重的 `SubP2` 交给更合适的 solver 栈。


## 当前稳定配置

当前建议固定为：

- `JAX_PLATFORMS=cpu`
- `JAX_SUBP1_FORWARD_ONLY_FAST_PATH=direct`
- `JAX_CABLE_SUBP1_MODE=batched`
- Julia worker 常驻
- `--julia-linear-solver mumps`
- `--julia-kkt-system default`
- `--julia-callback default`
- `--julia-thread-schedule static`
- `--julia-threads 20`
- `--julia-blas-threads 1`

不要默认启用这些实验路径：

- 纯 JAX `SubP2 barrier/sqp`
- Julia GPU solver
- `cudss`
- `lapackcuda`
- 改动 `mumps/default/default/static` 这一组正式 benchmark 配置

## 详细日志
更详细的实验日志看这文档：
- [`JAX_IPOPT_ALIGNMENT_NOTES.md`](JAX_IPOPT_ALIGNMENT_NOTES.md)
