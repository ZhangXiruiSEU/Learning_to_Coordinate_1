# meta-learn_useThis

当前还在维护的性能主线不是纯 `ipoptax SubP2`，而是：

- 外层 planner: Python/JAX
- `SubP1`: JAX
- `SubP2`: persistent Julia worker
- `SubP3`: Python/JAX

这条主线的目标很简单：保留 Python/JAX 外层流程和当前 `SubP1` 的速度优势，同时把最重的 `SubP2` 交给 Julia 里的 `JuMP + MadNLP` 去解，并且用常驻 worker 避免每轮重复建模和重复启动。

当前稳定 CPU 基线配置下，formal benchmark 的 warm run 大约是：

- original CasADi/IPOPT fullflow: `~5.7s`
- current JAX + persistent Julia fullflow: `~0.94s` 到 `~1.00s`

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
- `archive/`
- `python_archive/archive/`
- `julia_subp2/archive/`
  已归档的试验脚本和一次性诊断代码，不属于当前默认主线。

## 环境准备

Python 侧当前实验默认使用这个解释器：

```bash
/home/mpc/miniconda3/envs/xirui/bin/python
```

Julia 侧先初始化 `julia_subp2` 环境：

```bash
JULIA_DEPOT_PATH=/tmp/julia-depot:/home/mpc/.julia julia julia_subp2/setup_env.jl
```

如果你只是复现实验结果，推荐固定这些环境变量：

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mpl
export JAX_PLATFORMS=cpu
```

当前稳定主线不建议默认切到 GPU，也不建议把 Julia 线性求解器改成 `cudss` 或 `lapackcuda`。

## 如何跑

### 1. 正式 benchmark

这是当前最权威的复现命令。它会同时跑：

- original CasADi/IPOPT 全流程
- 当前 JAX + persistent Julia 全流程
- 时间对比
- 轨迹误差对比

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl \
VERIFY_TASK_IDX=0 VERIFY_HORIZON=100 VERIFY_ADMM_ITERS=3 \
JAX_PLATFORMS=cpu \
/home/mpc/miniconda3/envs/xirui/bin/python -u \
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

解释：

- `repeat=1` 通常包含更多冷启动成本
- `repeat=2` 作为 warm run 看更有意义
- 当前主机上 Julia 线程调度对结果有影响，正式复现建议固定 `static`

### 2. 只做数值对齐验证

如果你只是想确认“当前 JAX 路线”和“原版 CasADi 路线”在同一个 task 上是否数值接近，可以直接跑：

```bash
VERIFY_JAX_BACKEND_SELECTED=1 JAX_PLATFORMS=cpu \
/home/mpc/miniconda3/envs/xirui/bin/python verify_jax_vs_original.py
```

它会打印各状态/控制量的误差，并生成对比图。

### 3. 单独跑 JAX 或原版 forward

这两个脚本主要用于单路 forward 演示，不是正式 benchmark 入口：

```bash
/home/mpc/miniconda3/envs/xirui/bin/python run_jax_forward_with_trained_nn.py --task-idx 0 --backend cpu
/home/mpc/miniconda3/envs/xirui/bin/python run_original_forward_with_trained_nn.py --task-idx 0
```

它们依赖仓库里的训练权重和参考轨迹数据目录。

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

## 当前 Python 和 Julia 的衔接方式

当前衔接方式是“persistent worker + JSON batch payload”，不是每次调用都重新起一个 Julia 进程。

主链路如下：

1. Python 外层在 [`run_julia_subp2_fullflow_persistent.py`](run_julia_subp2_fullflow_persistent.py) 里启动 `JuliaBatchWorker`。
2. Python 把当前 ADMM 轮所有 step 的 `SubP2` 初值和运行时参数打成一个 batch payload。
3. Python 只发送 Julia 真正会用到的字段，不再把无关数据一起打包。
4. Julia worker 在 [`julia_subp2/worker_batch_madnlp_jump_native.jl`](julia_subp2/worker_batch_madnlp_jump_native.jl) 里接收 JSON 命令。
5. worker 调 [`julia_subp2/solve_batch_madnlp_jump_native.jl`](julia_subp2/solve_batch_madnlp_jump_native.jl) 做 batch 求解。
6. Julia 返回紧凑结果：
   - `x_sol_batch`
   - `iter_batch`
   - `eq_inf_batch`
   - `ineq_vio_batch`
7. Python 再把这批结果重建回 planner 需要的结构，继续后续 ADMM 流程。

这套设计里最关键的点是：

- Julia 进程常驻，避免反复启动开销。
- 每个 step 的模型结构缓存下来，只更新运行时参数。
- Python/Julia 通信只传必要字段，减少桥接损耗。
- benchmark 会同时打印 planner 侧 `SubP2 total` 和 worker 侧 `SubP2-only`，方便区分“求解慢”还是“桥接慢”。

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

## 当前性能结论

同一份 warm `repeat=2` formal log 上，当前主线和 original 的对比是：

| component | original CasADi/IPOPT | current JAX + Julia |
| --- | ---: | ---: |
| `SubP1 total` | `2086.86 ms` | `133.97 ms` |
| `SubP2 total` | `3634.27 ms` | `864.91 ms` |
| `SubP2-only` | - | `817.84 ms` |
| `SubP3 total` | `13.22 ms` | `1.11 ms` |
| `fullflow total` | `5734.35 ms` | `1000.72 ms` |

更详细的实验记录看这两个文档：

- [`AI_HANDOFF_SUBP2_STATUS.md`](AI_HANDOFF_SUBP2_STATUS.md)
- [`JAX_IPOPT_ALIGNMENT_NOTES.md`](JAX_IPOPT_ALIGNMENT_NOTES.md)
