# AI Handoff: Persistent Julia SubP2 Current Status

Updated: 2026-03-12

This document replaces the older SQP-centric handoff. The current mainline is no
longer "pure JAX SubP2"; it is "Python/JAX outer planner + persistent Julia
SubP2 worker".

## 1. Current Mainline

Current default operational branch:

- outer planner: Python/JAX
- `SubP1`: JAX
- `SubP2`: persistent Julia worker
- Julia stack:
  - `JuMP`
  - `NLPModelsJuMP.MathOptNLPModel`
  - `MadNLP`
  - linear solver: `mumps`
  - `JULIA_NUM_THREADS=20`
  - `JULIA_BLAS_THREADS=1`
  - `kkt_system=default`
  - `callback=default`

Stable mainline configuration to treat as authoritative:

- backend: CPU
- `JAX_PLATFORMS=cpu`
- `JAX_SUBP1_FORWARD_ONLY_FAST_PATH=direct` (default)
- `JAX_CABLE_SUBP1_MODE=batched` (default)
- `--julia-thread-schedule static`
- do not enable experimental `JAX_SUBP2_SOLVER_MODE=barrier|sqp`
- do not switch Julia linear solver away from `mumps`
- do not enable experimental GPU options such as `cudss`, `lapackcuda`, or
  `JULIA_SUBP2_GPU_OUTER_MODE`

In other words, the stable `~0.94s` number is specifically the CPU mainline,
not any GPU experiment. On the current host, CPU `SubP2` timing is also
noticeably sensitive to Julia thread scheduling; use `static` for formal
reproduction instead of relying on the Julia default scheduler.

Current main files:

- `JustWorkingOnIt.py`
- `python_archive/run_julia_subp2_fullflow_persistent.py`
- `julia_subp2/solve_batch_madnlp_jump_native.jl`
- `julia_subp2/worker_batch_madnlp_jump_native.jl`
- `verify_jax_vs_original.py`

Historical experiment and diagnosis scripts were moved out of the main working
surface into:

- `archive/`
- `python_archive/archive/`
- `julia_subp2/archive/`

## 2. Current Best Verified State

Main benchmark definition:

- `task=0`
- `horizon=100`
- `admm=3`
- `repeats=2`
- treat `repeat=2` as warm run

Formal benchmark command:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl VERIFY_TASK_IDX=0 VERIFY_HORIZON=100 VERIFY_ADMM_ITERS=3 JAX_PLATFORMS=cpu /home/mpc/miniconda3/envs/xirui/bin/python -u python_archive/run_julia_subp2_fullflow_persistent.py --task-idx 0 --repeats 2 --julia-threads 20 --julia-blas-threads 1 --julia-linear-solver mumps --julia-kkt-system default --julia-callback default --julia-thread-schedule static
```

Interpret this command as the confirmed stable benchmark target. If a run uses a
different Julia linear solver, GPU flags, or a different `SubP2` backend, it is
not the `~0.94s` mainline result.

Current stable warm-run baseline:

- original fullflow: typically `~5.7s` to `~5.9s`
- JAX + persistent Julia fullflow: `~0.94s`
- best observed warm run on current mainline: `~937.6 ms`
- recently reproduced warm run: `~942.0 ms`
- `SubP2-only`: `~742.3 ms` best observed, `~742-757 ms` typical on the current mainline

Current reproduced accuracy:

- `load_state_xl rmse ~= 2.03e-4`
- `load_control_ul rmse ~= 1.54e-4`
- `cable_state_xc rmse ~= 4.94e-4`

Latest apples-to-apples warm `repeat=2` breakdown from the same formal log
(`task=0, horizon=100, admm=3`, CPU mainline, static thread schedule):

| component | original CasADi/IPOPT | current JAX + Julia |
| --- | ---: | ---: |
| `SubP1 total` | `2086.86 ms` | `133.97 ms` |
| `SubP2 total` | `3634.27 ms` | `864.91 ms` |
| `SubP2-only` | - | `817.84 ms` |
| `SubP3 total` | `13.22 ms` | `1.11 ms` |
| `fullflow total` | `5734.35 ms` | `1000.72 ms` |

Notes:

- current `SubP2 total` is the planner-side end-to-end time
- current `SubP2-only` is the pure Julia worker wall time
- original `SubP3 total` is not printed directly by the legacy code path; it is
  the residual `total - SubP1 - SubP2`

Interpretation:

- the project has already beaten the original fullflow wall time on the current
  formal CPU benchmark
- remaining optimization headroom is now mostly inside the Julia `SubP2`
  solving path itself

## 3. What Was Actually Optimized

### 3.1 Python <-> Julia bridge

Verified and kept:

- runtime payload filtering: Python only sends fields the Julia worker
  actually uses
- one-shot `jax.device_get(...)` for the whole runtime payload
- compact JSON payloads
- compact Julia batch result format:
  - `x_sol_batch`
  - `iter_batch`
  - `eq_inf_batch`
  - `ineq_vio_batch`
- Python side rebuild from compact batch reply

Result:

- bridge overhead is no longer the main bottleneck
- warm run `bridge_prepare + payload + rebuild` is already small

### 3.2 Julia worker hot path

Verified and kept:

- static parameters are only updated when changed
- warm-start storage cleaned up to a lightweight single-slot cache
- compact stacked-result path avoids per-step `Dict` allocation
- lighter loops for `set_start_value` and reading the solution vector

Result:

- `SubP2-only` dropped materially relative to the older baseline

### 3.3 Non-SubP2 fixed costs

Verified and kept:

- `SubP1` fast path default is now direct forward solve with host
  materialization before `SubP2`
- trajectory initialization was moved into dedicated global JIT kernels

Result:

- `init_ms` dropped from about `46 ms` to about `0.7 ms`
- `SubP1` warm cost dropped into roughly the `42-46 ms` per-ADMM-iteration
  range on the current best path

## 4. Current Default Behavior

### 4.1 `SubP1` fast path

Current default:

- `JAX_SUBP1_FORWARD_ONLY_FAST_PATH=direct`

Meaning:

- `SubP1` computes trajectories directly
- results are materialized to host before being fed into `SubP2`

Why this is the default:

- pure direct device-to-device flow made `SubP1` itself faster
- but it pushed synchronization cost into later `SubP2` batch preparation
- `direct + host materialize` was the best end-to-end balance

Fallback still available:

- `JAX_SUBP1_FORWARD_ONLY_FAST_PATH=packaged`

### 4.2 Benchmark diagnostics

Current benchmark diagnostics now report:

- `batch_total_ms`
- `batch_solve_ms_total`
- `bridge_*_ms`
- compact residual summaries

There is also a lightweight `iter_*` diagnostic path in the Python reporting
layer, but with the current `JuMP/MadNLP` route the solver iteration count is
not reliably exposed; it currently prints as `n/a` rather than misleading `-1`.

## 5. What Was Tried And Rejected

These were tested and should not be treated as the current mainline.

### 5.1 `SubP1` packaged skip-derivs path

What it did:

- keep old packaging path
- skip derivative extraction

Outcome:

- close to break-even
- not the best current path

### 5.2 `SubP1` direct path without host materialization

What it did:

- direct forward-only `SubP1`
- keep everything on device

Outcome:

- `SubP1` looked faster
- but `bridge_prepare` exploded because the sync cost moved downstream
- rejected

### 5.3 Julia stage-aware multi-slot warm cache

Intent:

- make warm start use "same ADMM stage from previous repeat"

Outcome:

- warm run regressed to about `~1099 ms`
- rolled back

Conclusion:

- do not keep pushing that cache design without much stronger evidence

### 5.4 Looser MadNLP stopping criteria

Tested example:

```bash
--max-iter 120 --acceptable-tol 3e-4 --acceptable-iter 3 --tol 1e-6
```

Outcome:

- warm run around `~969 ms`
- slightly worse than the current default path
- small but real RMSE drift appeared

Conclusion:

- not adopted

## 6. GPU Status

Short answer:

- some GPU acceleration is possible in the current Julia stack
- true "100 SubP2 problems batched together on GPU" is still not implemented

Important current facts:

- the code now has an experimental `cudss + array_type=CuArray` path in the
  current `JuMP/MadNLP` worker
- there is also an experimental `JULIA_SUBP2_GPU_OUTER_MODE=threaded` switch
  for testing outer-step concurrency on top of GPU solves
- neither path is part of the stable mainline

Latest real benchmark result on the same fullflow problem (`task=0`,
`horizon=100`, `admm=3`, `repeats=2`, `CUDA_VISIBLE_DEVICES=1`):

- `cudss + serial` warm repeat:
  - fullflow `~22.0 s`
  - `SubP2-only ~15.6 s`
- `cudss + threaded` warm repeat:
  - fullflow `~22.3 s`
  - `SubP2-only ~15.7 s`

So the current architecture supports:

- "one NLP step uses GPU-backed linear algebra internally"

It does not currently support:

- "100 same-structure NLPs are solved as a real batched GPU problem"

If true batched GPU solving is desired, that would require a different solver
layer, not a small tweak to the current `JuMP` object-per-step design. The
current worker still builds one model per step, so it does not hit the kind of
common-sparsity batched solve path that `cuDSS` is designed to accelerate.

## 7. What "SubP2 Hot Path Itself" Means

At this stage, the remaining optimization is not in:

- bridge
- payload serialization
- `init`
- `SubP1`

It is inside the actual Julia `SubP2` solve path:

- `update_parameterized_step_from_stacked!`
- `solve_cached_step_compact!`
- `optimize!`
- and the JuMP/MadNLP model defined in `solve_step_madnlp_jump_native_eq.jl`

Practical meaning:

- make each step NLP smaller
- make each step NLP easier to solve
- reduce parameter-update work per step
- improve `x_init` / warm-start quality

## 8. Current Best Judgment

What is already done:

- outer fixed costs are mostly compressed
- bridge is no longer the main bottleneck
- current mainline is already much faster than original fullflow

What remains:

- the main remaining cost is still `SubP2`
- further wins are possible, but no longer cheap

Reasonable expectation from more work:

- another `50-150 ms` may still be possible
- a dramatic additional 2x speedup is unlikely without changing the solver
  structure itself

## 9. Recommended Next Step If Work Continues

If optimization resumes, do not start by re-optimizing:

- bridge
- payload packing
- `init`
- `SubP1`
- naive warm-cache experiments

Best next direction:

- change the `SubP2` model hot path itself

Most promising sub-directions:

- split terminal and non-terminal step templates if that reduces problem size
- improve `SubP2` initial guess quality
- remove redundant variables or constraints if any are still present
- reduce per-step parameter mutation even further

Not recommended as the next step:

- solver-combo sweeps, unless explicitly requested again

## 10. Quick Handoff Summary

If someone new continues from here, assume:

- the current mainline is persistent Julia `SubP2`, not pure JAX SQP
- the stable default is already around `~0.94s` warm on the formal benchmark
- that `~0.94s` number specifically means the CPU mainline:
  - `JAX_PLATFORMS=cpu`
  - persistent Julia worker
  - `mumps/default/default`
  - `thread_schedule=static`
  - `JAX_SUBP1_FORWARD_ONLY_FAST_PATH=direct`
  - `JAX_CABLE_SUBP1_MODE=batched`
- the main remaining problem is Julia `SubP2` solve cost, not infrastructure
- experimental Julia GPU paths exist, but they are much slower than the CPU
  mainline and do not provide true batched GPU solving
- several intuitive experiments were already tried and rejected:
  - direct device-only `SubP1`
  - packaged skip-derivs as mainline
  - stage-aware Julia warm cache
  - looser MadNLP tolerances
