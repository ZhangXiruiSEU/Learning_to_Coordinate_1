# JAX Planner Alignment Notes

Last updated: 2026-03-12

## Goal

The goal of this work is to make the planner stack match the original CasADi/IPOPT planner as closely as possible while becoming faster than the original implementation at realistic scale.

The current conclusion is:

- Forward logic is mostly aligned.
- `SubP1` is mostly aligned and is no longer the main issue.
- The old pure-JAX `ipoptax` route is no longer the preferred performance mainline.
- The current **default performance branch** is:
  - Python/JAX outer planner
  - persistent Julia worker for `SubP2`
  - `JuMP -> NLPModelsJuMP.MathOptNLPModel -> MadNLP`
  - CPU, `JULIA_NUM_THREADS=20`, `JULIA_BLAS_THREADS=1`
  - slot-based parameterized reuse
- On the current formal benchmark (`task=0, horizon=100, admm=3`, warm run),
  this branch is already faster than the original fullflow while keeping good
  numerical agreement.

Recent practical conclusion from the latest experiments:

- `SubP1` is no longer where the main time goes.
- the stable performance mainline is now the CPU branch, not any GPU experiment
  and not the old pure-JAX `SubP2`
- experimental GPU routes were tested again:
  - JAX barrier on GPU can solve the batch, but is far slower than the CPU
    mainline on the real fullflow problem
  - Julia `cudss + CuArray` can be made to run, but in the current
    model-per-step architecture it is still not a true batched GPU solve and is
    also much slower than the CPU mainline
- the remaining optimization target is still the Julia `SubP2` hot path itself,
  not bridge overhead and not `SubP1`

## Current Default Mainline

The current operational default should be treated as:

- outer planner: Python/JAX
- `SubP1`: current JAX implementation
- `SubP2`: persistent Julia worker
- Julia stack:
  - `JuMP`
  - `NLPModelsJuMP.MathOptNLPModel`
  - `MadNLP`
  - linear solver `mumps`
  - `JULIA_NUM_THREADS=20`
  - `JULIA_BLAS_THREADS=1`

Older experiment and diagnosis scripts referenced throughout this note have
since been moved under:

- `archive/`
- `python_archive/archive/`
- `julia_subp2/archive/`

Current formal-scale warm benchmark (`task=0, horizon=100, admm=3`):

- best observed original fullflow: `~5709.5 ms`
- current typical original fullflow: `~5.7s` to `~5.9s`
- best observed JAX + persistent Julia fullflow: `~937.6 ms`
- recently reproduced stable run:
  - original `~5868.9 ms` to `~6122.0 ms`
  - JAX + persistent Julia `~942.0 ms`

Reproduced stable accuracy:

- `load_state_xl rmse = 2.032169e-04`
- `load_control_ul rmse = 1.542692e-04`
- `cable_state_xc rmse = 4.944731e-04`

Latest same-log warm `repeat=2` breakdown on the formal CPU benchmark:

| component | original CasADi/IPOPT | current JAX + Julia |
| --- | ---: | ---: |
| `SubP1 total` | `2086.86 ms` | `133.97 ms` |
| `SubP2 total` | `3634.27 ms` | `864.91 ms` |
| `SubP2-only` | - | `817.84 ms` |
| `SubP3 total` | `13.22 ms` | `1.11 ms` |
| `fullflow total` | `5734.35 ms` | `1000.72 ms` |

Notes:

- current `SubP2 total` is the end-to-end planner-side time
- current `SubP2-only` is the pure persistent Julia worker wall time
- original `SubP3 total` is a residual because the legacy path does not print it
  separately

Reproduced warm-run planner profile:

- `init_ms ≈ 0.7 ms`
- `SubP1 total ≈ 131-160 ms`
- `SubP2 total ≈ 742-757 ms`
- `SubP3 total ≈ 1.3 ms`

Julia worker side on the same reproduced run:

- `SubP2` pure wall-clock over 3 ADMM rounds: `~742-757 ms`
- bridge prepare/payload/rebuild totals are only on the order of `~14 ms`

Practical interpretation:

- transport is no longer the main blocker
- fixed non-`SubP2` cost is much smaller than before
- the remaining optimization work is still mostly inside Julia `SubP2`
  itself:
  - parameter update hot path
  - solver/model hot path
  - any reduction in per-step NLP difficulty

Stable mainline configuration to treat as authoritative:

- `JAX_PLATFORMS=cpu`
- `JAX_SUBP1_FORWARD_ONLY_FAST_PATH=direct`
- `JAX_CABLE_SUBP1_MODE=batched`
- persistent Julia worker
- `--julia-linear-solver mumps`
- `--julia-kkt-system default`
- `--julia-callback default`
- `--julia-thread-schedule static`
- `--julia-threads 20`
- `--julia-blas-threads 1`

On the current host, the Julia worker is materially more stable with
`thread_schedule=static`; `default` and `dynamic` can regress the warm CPU
benchmark noticeably under background load.

Formal benchmark command for the stable `~0.94s` branch:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl VERIFY_TASK_IDX=0 VERIFY_HORIZON=100 VERIFY_ADMM_ITERS=3 JAX_PLATFORMS=cpu /home/mpc/miniconda3/envs/xirui/bin/python -u run_julia_subp2_fullflow_persistent.py --task-idx 0 --repeats 2 --julia-threads 20 --julia-blas-threads 1 --julia-linear-solver mumps --julia-kkt-system default --julia-callback default --julia-thread-schedule static
```

Anything that changes the `SubP2` backend, enables JAX barrier/SQP, or switches
Julia to GPU solvers should be treated as an experiment, not the stable
mainline.

## Recent Experiment Track

The most recent optimization work followed this sequence:

### 1. First get forward alignment and `SubP1` under control

Main idea:

- do not optimize `SubP2` before the outer forward logic and `SubP1` behavior
  are numerically close to the original planner

What happened:

- forward initialization and several `SubP1` data-path mismatches were fixed
- cable-side batched DDP was brought much closer to the legacy behavior
- `SubP1` then stopped being the dominant source of mismatch

Result:

- `SubP1` is now both aligned enough and much faster than the original
- the current warm `SubP1` cost is only about `~50 ms` per ADMM round, versus
  roughly `~0.7 s` per round in the original CasADi/IPOPT planner

### 2. Test whether pure JAX `SubP2` could become the performance mainline

Main idea:

- improve `ipoptax` behavior until it looks more like IPOPT:
  - better `mu` progress
  - better line search / filter behavior
  - softer stopping logic
  - plateau-stop logic

What happened:

- on synthetic and fixed-step tests, JAX could be made much better than the old
  naive baseline
- but on the real fullflow benchmark, the pure-JAX `SubP2` route still did not
  beat the Julia worker path

Result:

- pure JAX `SubP2` is still useful as a diagnostic and research track
- it is not the preferred performance branch anymore

### 3. Move the real performance mainline to persistent Julia `SubP2`

Main idea:

- keep Python/JAX for the outer planner
- move only `SubP2` to a persistent Julia worker so that:
  - the exact real NLP still goes through a mature nonlinear solver stack
  - model construction can be cached
  - the Python/Julia bridge can be amortized

What happened:

- a persistent worker was built around:
  - `JuMP`
  - `NLPModelsJuMP.MathOptNLPModel`
  - `MadNLP`
- stacked runtime payloads and compact replies removed most bridge overhead
- hot-path parameter mutation was tightened on the Julia side

Result:

- warm fullflow dropped from the earlier `~1.2 s` range down to the current
  stable `~0.94 s` range
- `SubP2-only` dropped into the `~742-757 ms` range

### 4. Compress the non-`SubP2` fixed cost after the Julia worker was already fast

Main idea:

- once bridge overhead stopped mattering, the next obvious fixed cost was
  `init + SubP1`

What happened:

- `SubP1` default fast path became `direct + host materialize`
- trajectory initialization moved into dedicated JIT kernels

Result:

- `init_ms` fell from roughly `~46 ms` to `~0.7 ms`
- `SubP1` warm cost fell to about `~131-160 ms` over all 3 ADMM rounds

### 5. Re-test GPU routes instead of assuming they would help

Main idea:

- verify whether GPU could beat the now-optimized CPU mainline on the real
  `SubP2` batch

What happened:

- JAX barrier on GPU was run on the real fullflow batch:
  - it could solve the problem
  - but warm fullflow was still on the order of many seconds, far slower than
    the CPU mainline
- Julia `MadNLPGPU` was tested again:
  - `cudss + array_type=CuArray` now works in the current worker
  - an experimental outer-step threaded mode was also tested

Real result on the same fullflow benchmark:

- stable CPU mainline:
  - fullflow `~0.94 s`
  - `SubP2-only ~0.74-0.76 s`
- Julia `cudss + serial` warm repeat:
  - fullflow `~22.0 s`
  - `SubP2-only ~15.6 s`
- Julia `cudss + threaded` warm repeat:
  - fullflow `~22.3 s`
  - `SubP2-only ~15.7 s`

Current interpretation:

- in the present architecture, GPU does not help
- the reason is not just kernel quality; the current Julia route still solves
  `101` separate JuMP models, so it does not become a true common-pattern
  batched GPU solve
- a real batched GPU solution would require a different solver layer below the
  current JuMP object-per-step design

## Synthetic NLP Benchmark: IPOPT vs MadNLP vs JAX IPM

To separate solver-core behavior from the real `SubP2` integration overhead, a synthetic but
moderately complex nonlinear constrained problem was built and solved with:

- Python `CasADi + IPOPT`
- Julia `JuMP -> NLPModelsJuMP -> MadNLP`
- JAX `ipoptax`

Relevant scripts:

- [benchmark_madnlp_vs_ipopt_synthetic.py](/home/mpc/xirui/meta-learn_useThis/python_archive/benchmark_madnlp_vs_ipopt_synthetic.py)
- [benchmark_jax_ipoptax_synthetic.py](/home/mpc/xirui/meta-learn_useThis/python_archive/benchmark_jax_ipoptax_synthetic.py)
- [benchmark_madnlp_synthetic.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/archive/benchmark_madnlp_synthetic.jl)

The synthetic problem uses:

- `n_agents = 10`
- decision variables `[p_i in R^3, u_i in R^3]` per agent
- equality constraints:
  - `||p_i||^2 = 1`
  - `sum_i u_i = target_u`
- inequality constraints:
  - obstacle avoidance
  - pairwise spacing
  - box bounds on `u`
  - thrust upper/lower bounds

### Warm-run timing on the same synthetic problem

Observed warm-run solve times:

- IPOPT:
  - `~7.95 ms`
- MadNLP:
  - build `~0.88 ms`
  - solve `~2.13 ms`
  - total `~3.02 ms`
- JAX `ipoptax` baseline:
  - `~65.98 ms` on CPU
  - `~465.29 ms` on GPU
- JAX `ipoptax` with `soft_stop`:
  - `~3.18 ms` on CPU
  - `~18.68 ms` on GPU

Objective values on the synthetic problem were essentially identical:

- IPOPT:
  - `0.057510787727193424`
- MadNLP:
  - `0.0575107877272106`
- JAX `ipoptax` baseline:
  - `0.05751078948378563`
- JAX `ipoptax` with `soft_stop`:
  - CPU: `0.05751078948378563`
  - GPU: `0.05751078575849533`

Feasibility quality for the JAX runs remained good:

- `eq_inf = 5.96e-08`
- `ineq_vio = 0.0`

### Main synthetic benchmark conclusion

This benchmark matters because it isolates the solver from most of the real `SubP2` plumbing.

The result is:

- `MadNLP` itself is not inherently slower than IPOPT; on this synthetic problem, warm-run
  `MadNLP` was faster than IPOPT.
- `JAX ipoptax` baseline was slow mainly because it kept running the full `200` iterations.
- On the same synthetic problem, enabling `soft_stop` reduced JAX `ipoptax` from:
  - `200` iterations down to `6`
  - and reduced warm CPU solve time from `~66 ms` to `~3.18 ms`
- This strongly supports the interpretation that a major part of the JAX IPM speed problem is
  stopping / iteration policy, not only raw compiled primitive speed.

The synthetic GPU result also clarified something important:

- on this problem size, GPU is not automatically beneficial
- the warm GPU `soft_stop` result (`~18.68 ms`) was still slower than the warm CPU result
  (`~3.18 ms`)
- so small to medium NLPs can remain CPU-favored even with JAX, unless batching is large enough

## Authoritative Real `SubP2` Single-Step Benchmark

To avoid mixing together:

- snapshot capture cost
- planner/context construction cost
- and the actual single-step solver runtime

I added a cleaner benchmark:

- [benchmark_real_subp2_step.py](/home/mpc/xirui/meta-learn_useThis/python_archive/benchmark_real_subp2_step.py)

This script uses a *real* `SubP2` snapshot from the planner and reports:

- setup time:
  - `build_context_ms`
  - `capture_snapshot_ms`
  - `build_step_problem_ms`
- original IPOPT single-step solve time
- JAX `ipoptax` single-step solve time
  - `baseline`
  - `soft_stop`
  - `plateau_stop`

Tested case:

- `task=0`
- `horizon=2`
- `max_iter_admm=3`
- `target_admm_iter=1`
- `target_step=0`

### Setup cost for the real snapshot

On this real bad step, the cost of *finding* the snapshot is already significant:

- `build_context_ms ≈ 568 ms`
- `capture_snapshot_ms ≈ 20462 ms`
- `build_step_problem_ms ≈ 218 ms`

This setup cost is **not** solver-core cost. It is the cost of running the planner far enough
to extract the bad step and prepare its tensors.

### Real single-step solver timing (same fixed snapshot)

Original IPOPT single-step solve:

- `wall_ms ≈ 12.45 ms`
- `iter_count = 28`
- `eq_inf ≈ 1.88e-09`
- `ineq_vio = 0`

JAX `ipoptax` on the same fixed real snapshot:

- `baseline`
  - warm solve mean: `~6142 ms`
  - warm solve min: `~5783 ms`
  - `iterations = 120`
  - `orig_rmse ≈ 2.65e-05`
  - `ineq_vio ≈ 2.81e-04`
- `soft_stop`
  - warm solve mean: `~327 ms`
  - warm solve min: `~314 ms`
  - `iterations = 5`
  - `orig_rmse ≈ 1.16e-04`
  - `ineq_vio ≈ 1.32e-03`
- `plateau_stop`
  - warm solve mean: `~3455 ms`
  - warm solve min: `~3215 ms`
  - `iterations = 75`
  - `orig_rmse ≈ 4.52e-05`
  - `ineq_vio ≈ 5.16e-04`

### Main real-step benchmark conclusion

This benchmark is the most trustworthy answer to the question:

> “Is the real `SubP2` slowness mostly solver-core, or mostly external plumbing?”

The answer is:

- The real `SubP2` setup path is expensive, but that is **not** the main explanation for the
  JAX/IPOPT gap on a fixed step.
- On a fixed real bad snapshot, `baseline ipoptax` is still hundreds of times slower than IPOPT.
- But this is **not** a universal JAX result:
  - with `soft_stop`, the same real step drops from `~5.8–6.5 s` warm to `~0.31–0.34 s` warm
  - which shows that stopping/iteration policy is a major part of the real slowness
- Therefore the real `SubP2` speed gap is not “just JAX being slow”; it is strongly affected by:
  - solver iteration policy
  - stopping rules
  - and only secondarily by the outer benchmark plumbing

This also means that `diagnose_subp2_step.py` should be treated as a diagnostic tool, while
`benchmark_real_subp2_step.py` in `python_archive/` should be treated as the authoritative single-step timing script.

## Main Files

- Original planner: [Kinodynamic_Planning_ADMM_quaternion_DDP_autotuning_2nd_COM_Dyn.py](/home/mpc/xirui/meta-learn_useThis/Kinodynamic_Planning_ADMM_quaternion_DDP_autotuning_2nd_COM_Dyn.py)
- JAX planner: [JustWorkingOnIt.py](/home/mpc/xirui/meta-learn_useThis/JustWorkingOnIt.py)
- JAX batched DDP core: [ddp_vmap_jax.py](/home/mpc/xirui/meta-learn_useThis/ddp_vmap_jax.py)
- JAX interior-point solver: [ipoptax/solver.py](/home/mpc/xirui/meta-learn_useThis/ipoptax/solver.py)
- End-to-end verification: [verify_jax_vs_original.py](/home/mpc/xirui/meta-learn_useThis/verify_jax_vs_original.py)
- Regression summary export: [export_verify_regression_summary.py](/home/mpc/xirui/meta-learn_useThis/export_verify_regression_summary.py)
- SubP2 single-step diagnosis: [diagnose_subp2_step.py](/home/mpc/xirui/meta-learn_useThis/diagnose_subp2_step.py)
- Fullflow SubP2 diag inspection: [inspect_fullflow_subp2_diag.py](/home/mpc/xirui/meta-learn_useThis/python_archive/inspect_fullflow_subp2_diag.py)

## What Was Fixed

### 1. Forward initialization alignment

JAX safe-copy initialization was not originally matching the original ADMM forward path.

The JAX side was updated so that:

- `_initialize_trajectories(...)` now follows the original safe-copy style.
- Load and cable safe-copy trajectories are initialized and rolled forward in the same spirit as the original planner.

Relevant code:

- [JustWorkingOnIt.py](/home/mpc/xirui/meta-learn_useThis/JustWorkingOnIt.py#L1570)

### 2. Load `SubP1` parameter unpacking and warm start alignment

There was a load-side mismatch in `ParaL` unpacking and initial control usage.

The JAX side was corrected so that:

- `scxL` and `scul` are unpacked in the original order.
- Load DDP initial control guess uses `Ref_ul`, not `scul`.
- Load DDP settings were kept consistent with original-style expectations.

Relevant code:

- [JustWorkingOnIt.py](/home/mpc/xirui/meta-learn_useThis/JustWorkingOnIt.py#L552)
- [JustWorkingOnIt.py](/home/mpc/xirui/meta-learn_useThis/JustWorkingOnIt.py#L614)

### 3. Cable `SubP1` alignment

This took several rounds.

Important findings:

- The old discrepancy was not mainly from `SubP2`.
- Cable `SubP1` was the main remaining source of mismatch after load alignment.

What was done:

- Added `legacy`, `batched`, and `compare` modes for cable `SubP1`.
- Brought the batched JAX DDP logic closer to legacy DDP update rules.
- Fixed a key batched DDP bug: stage parameters were being fed with the wrong time semantics.

Current status:

- Batched JAX cable DDP is now very close to legacy cable DDP.
- Default cable mode is back to `batched`.

Relevant code:

- [JustWorkingOnIt.py](/home/mpc/xirui/meta-learn_useThis/JustWorkingOnIt.py#L399)
- [JustWorkingOnIt.py](/home/mpc/xirui/meta-learn_useThis/JustWorkingOnIt.py#L504)
- [ddp_vmap_jax.py](/home/mpc/xirui/meta-learn_useThis/ddp_vmap_jax.py#L173)
- [ddp_vmap_jax.py](/home/mpc/xirui/meta-learn_useThis/ddp_vmap_jax.py#L246)

### 4. `SubP2` numerical parameter search

We ran two types of scans:

- Snapshot scan: fixed one `SubP2` step and scanned solver configs.
- Fullflow scan: evaluated solver configs by end-to-end planner difference, not just local feasibility.

Important conclusion:

- Snapshot-optimal parameters were not always the best fullflow parameters.
- After `SubP1` was aligned, the best `SubP2` region changed.
- Fullflow ranking was more reliable than snapshot-only ranking.

Main scan scripts:

- [scan_subp2_snapshot.py](/home/mpc/xirui/meta-learn_useThis/scan_subp2_snapshot.py)
- [scan_subp2_fullflow.py](/home/mpc/xirui/meta-learn_useThis/scan_subp2_fullflow.py)

### 5. `ipoptax` solver upgrades

The JAX interior-point solver was upgraded toward IPOPT-style behavior.

Implemented directions:

- stronger numerical guards
- better `adaptive_mu`
- line-search acceptance beyond a plain single merit check
- simplified filter-style acceptance
- simplified restoration logic
- tail tracing for `mu`, `ineq`, `comp`, `dual`

Important conclusion:

- The main bottleneck was not only line search.
- A major issue was central-path progress: `mu` used to stall around `1e-5`.
- After `mu` update changes, bad single-step cases moved much closer to IPOPT.

Relevant code:

- [ipoptax/solver.py](/home/mpc/xirui/meta-learn_useThis/ipoptax/solver.py#L109)
- [ipoptax/solver.py](/home/mpc/xirui/meta-learn_useThis/ipoptax/solver.py#L377)
- [ipoptax/solver.py](/home/mpc/xirui/meta-learn_useThis/ipoptax/solver.py#L633)

### 6. Why `main-soft` was introduced

After solver improvements, the late ADMM `SubP2` steps had this behavior:

- `mu` became very small
- complementarity became very small
- dual residual became small
- trace-space inequality residual became small
- but the raw unscaled inequality violation stayed around `1e-1`

This meant:

- the solver was already near a good solution in its own internal convergence scale
- but the outer ADMM acceptance logic was still classifying it as `main-bestfeas`

The fix was not to blindly relax everything. Instead:

- record tail metrics from the solver trace
- allow a `main-soft` classification if the internal tail metrics are good enough
- keep raw feasibility as a safeguard, but not as the only acceptance gate

Current important threshold:

- `soft_dual_tol = 5e-5`

Relevant code:

- [JustWorkingOnIt.py](/home/mpc/xirui/meta-learn_useThis/JustWorkingOnIt.py#L971)
- [JustWorkingOnIt.py](/home/mpc/xirui/meta-learn_useThis/JustWorkingOnIt.py#L999)
- [JustWorkingOnIt.py](/home/mpc/xirui/meta-learn_useThis/JustWorkingOnIt.py#L1219)

## Historical Pure-JAX `SubP2` Configuration Notes

This section is kept for historical context on the old pure-JAX `SubP2`
branch. It is not the current performance mainline.

These are the currently fixed main `SubP2` solver parameters in JAX:

- `tau_min = 0.99`
- `mu_min = 1e-8`
- `min_delta = 1e-4`
- `gamma_y = 1e-4`
- `gamma_z = 1e-2`
- `line_search_factor = 0.2`
- `line_search_min_step_size = 1e-10`
- `s0_cap = 0.5`
- `z0_init = 1e-3`

These come from the current `cfg_main` defaults:

- [JustWorkingOnIt.py](/home/mpc/xirui/meta-learn_useThis/JustWorkingOnIt.py#L828)

These can also be overridden by environment variables:

- `JAX_SUBP2_MAIN_TAU_MIN`
- `JAX_SUBP2_MAIN_MU_MIN`
- `JAX_SUBP2_MAIN_MIN_DELTA`
- `JAX_SUBP2_MAIN_GAMMA_Y`
- `JAX_SUBP2_MAIN_GAMMA_Z`
- `JAX_SUBP2_MAIN_LINE_SEARCH_FACTOR`
- `JAX_SUBP2_MAIN_LINE_SEARCH_MIN_STEP`
- `JAX_SUBP2_S0_CAP`
- `JAX_SUBP2_Z0_INIT`

## Historical `main-soft` Acceptance Thresholds

These thresholds matter for the old JAX `ipoptax` acceptance logic. They are
not the configuration that defines the current `~0.94s` persistent-Julia
mainline.

These are currently fixed in the JAX planner:

- `soft_mu_tol = 1e-8`
- `soft_comp_tol = 1e-8`
- `soft_dual_tol = 5e-5`
- `soft_eq_tol = 1e-4`
- `soft_raw_ineq_tol = 2e-1`
- `soft_trace_ineq_tol = 5e-3`

Relevant code:

- [JustWorkingOnIt.py](/home/mpc/xirui/meta-learn_useThis/JustWorkingOnIt.py#L971)
- [verify_jax_vs_original.py](/home/mpc/xirui/meta-learn_useThis/verify_jax_vs_original.py#L458)

## Historical Pure-JAX Regression Status

### `horizon=2, admm=2`

Current end-to-end verification result:

- `load_state_xl rmse = 2.2290299e-06`
- `load_control_ul rmse = 1.3405229e-05`
- `cable_state_xc rmse = 2.0120679e-05`

`SubP2` behavior:

- ADMM iter 0: `main:3`
- ADMM iter 1: `main-soft:3`

Smoke regression summary:

- [verify_regression_summary_smoke.json](/home/mpc/xirui/meta-learn_useThis/archive_outputs/verify/verify_regression_summary_smoke.json)

### `horizon=3, admm=3`

Observed stable behavior:

- ADMM iter 0: all `main`
- ADMM iter 1: all `main-soft`
- ADMM iter 2: all `main-soft`

Typical end-to-end differences seen:

- `load_state_xl rmse ≈ 8.07e-06`
- `load_control_ul rmse ≈ 4.44e-05`
- `cable_state_xc rmse ≈ 1.12e-04`

### Historical plateau-stop fullflow baseline

This was a meaningful CPU baseline inside the pure-JAX branch, but it is no
longer the project-wide performance mainline.

The current best speed/accuracy tradeoff on CPU uses:

- `JAX_SUBP2_FAST_PATH=1`
- `JAX_SUBP2_SOLVER_SOFT_STOP=0`
- `JAX_SUBP2_SOLVER_PLATEAU_STOP=1`
- `JAX_SUBP2_PLATEAU_MIN_ITERS=12`
- `JAX_SUBP2_PLATEAU_PATIENCE=8`
- `JAX_SUBP2_PLATEAU_PHI_TOL=1e-7`
- `JAX_SUBP2_PLATEAU_INEQ_TOL=1e-7`

For `horizon=2, admm=3, task=0` this gives:

- `load_state_xl rmse = 9.216409e-06`
- `load_control_ul rmse = 4.676007e-05`
- `cable_state_xc rmse = 5.170613e-05`
- `SubP2` summary:
  - ADMM iter 0: `main:3`
  - ADMM iter 1: `main-soft:3`
  - ADMM iter 2: `main-soft:3`

Relevant summary artifact:

- `/tmp/verify_regression_summary_plateau_mid_h2a3.json`

## Historical pure-JAX compiled speed

To measure compiled speed rather than first-run JIT cost, the same Python process ran
`verify_jax_planner(task_idx=0, show_plots=False)` three times in a row with the plateau-stop
configuration above.

Observed wall times:

- Run 1 (includes first JIT/compile): `74.03s`
- Run 2 (compiled): `68.10s`
- Run 3 (compiled): `70.94s`

So the current realistic compiled JAX fullflow speed for:

- `horizon=2`
- `admm=3`
- `task=0`

is about:

- `~69-71s` per run on CPU

This is much faster than the earlier `~123s` high-iteration baseline, but it is still far
slower than the original CasADi/IPOPT forward on the same small test.

Artifact:

- `/tmp/jax_compiled_speed_h2a3.json`

## What was actually slow in the pure-JAX branch

The current evidence says:

- The main bottleneck is still `SubP2`.
- The issue is not primarily Python wrapper overhead anymore.
- The issue is also not mainly single-iteration cost.
- The dominant problem is that JAX `SubP2` still uses too many solver iterations compared with IPOPT.

Important caution:

- A `max_iterations=1` microbenchmark of the solver gave a much smaller per-iteration number.
- That benchmark is useful only as a very local kernel measurement.
- It does **not** predict real multi-iteration solve time well.
- Real trust should be placed on:
  - full single-step solve timing
  - full forward timing
  - repeated same-process compiled runs

This matters because:

- `max_iterations=1` suggested a much smaller effective per-iteration cost
- but real full solves still took seconds
- therefore there are substantial multi-iteration costs that are not visible in the one-iteration microbenchmark

Single-step profiling on a difficult `SubP2` snapshot showed:

- Original IPOPT single-step solve: about `11 ms`
- JAX single-step solve before early-stop changes:
  - often `121` iterations
  - steady-state several seconds
- JAX single-step solve with aggressive early-stop:
  - as low as `10` iterations
  - steady-state a few hundred milliseconds
  - but some loss in solution quality

More detailed one-iteration profiling showed:

- one compiled JAX solver iteration is on the order of tens of milliseconds
- therefore the main gap comes from iteration count, not only per-iteration math cost

Additional `max_iterations` scaling measurements on the same difficult single-step snapshot
showed the more realistic steady-state cost:

- `maxit=1`: `53.24 ms`
- `maxit=2`: `191.22 ms`
- `maxit=4`: `291.03 ms`
- `maxit=8`: `284.33 ms`
- `maxit=16`: `789.63 ms`
- `maxit=32`: `1401.75 ms`
- `maxit=64`: `3347.84 ms`
- `maxit=120`: `6524.54 ms`

Equivalent average milliseconds per realized iteration:

- `maxit=1`: `53.24 ms / iter`
- `maxit=2`: `95.61 ms / iter`
- `maxit=4`: `72.76 ms / iter`
- `maxit=8`: `35.54 ms / iter`
- `maxit=16`: `49.35 ms / iter`
- `maxit=32`: `43.81 ms / iter`
- `maxit=64`: `52.31 ms / iter`
- `maxit=120`: `54.37 ms / iter`

This implies:

- the true steady-state per-iteration cost is not the optimistic `~20 ms` suggested by the
  `max_iterations=1` microbenchmark
- on longer runs it is more like `~45-55 ms / iteration` on this snapshot
- there is no evidence yet of a catastrophic superlinear explosion with iteration count
- the dominant issue is still "too many iterations", but each iteration is also more
  expensive in the real multi-iteration solve than the one-iteration microbenchmark suggested

One explicit bug was found during this scan:

- `iteration` returned by the solver had an off-by-one error
- e.g. `max_iterations=120` reported `iteration=121`
- this came from the loop continuation condition checking `iteration < max_iterations`
  after already scheduling the next iteration
- this has been fixed in [ipoptax/solver.py](/home/mpc/xirui/meta-learn_useThis/ipoptax/solver.py)

Another major finding came from profiling the Hessian PSD projection:

- `project_psd_cone(..., use_lapack=True)` uses a full dense eigen-decomposition (`eigh`)
- on the difficult `SubP2` snapshot, this alone costs about `46 ms`
- the raw Hessian itself costs only about `0.28 ms`

Direct comparison on the same single-step solve with `max_iterations=120` and all other
settings fixed:

- eigen-decomposition PSD projection:
  - `87.6 ms`
- Gershgorin shift PSD projection without iterative refinement:
  - `5.0 ms`
- Gershgorin shift PSD projection with iterative refinement:
  - `9.9 ms`

On that snapshot, all three variants produced the same observed result:

- `iter=2`
- `conv=True`
- `eq_inf = 2.157e-04`
- `ineq_vio = 1.389e-03`
- `orig_rmse = 1.478e-04`

This strongly suggests that the dense `eigh`-based PSD projection is a major avoidable cost
in the current JAX solver.

In other words:

- compiled JAX is not slow because every single primitive is catastrophically slow
- compiled JAX is slow because the current hand-written interior-point method still needs many
  more iterations than IPOPT to reach an acceptable point

## Current strategy

The current recommended direction is:

- keep the current JAX/IPOPT-aligned path as the correctness baseline
- use plateau-based stopping to remove obviously low-value tail iterations
- avoid spending too much more effort on making the current generic hand-written JAX IPM mimic
  IPOPT exactly
- if the speed gap remains too large, consider a more structure-specific `SubP2` solver rather
  than continuing to polish a generic IPOPT-like JAX interior-point solver

## Regression Export Tooling

The verification script now returns structured regression data:

- `task_idx`
- `horizon`
- `admm_iters`
- `diff`
- `subp2_summary`
- `subp2_acceptance`
- `plots`

Relevant code:

- [verify_jax_vs_original.py](/home/mpc/xirui/meta-learn_useThis/verify_jax_vs_original.py#L81)
- [verify_jax_vs_original.py](/home/mpc/xirui/meta-learn_useThis/verify_jax_vs_original.py#L448)

The dedicated export tool is:

- [export_verify_regression_summary.py](/home/mpc/xirui/meta-learn_useThis/export_verify_regression_summary.py)

Recommended command:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl VERIFY_HORIZON=2 VERIFY_ADMM_ITERS=2 VERIFY_TASK_IDX=0 JAX_PLATFORMS=cpu \
/home/mpc/miniconda3/envs/xirui/bin/python -u export_verify_regression_summary.py verify_regression_summary.json
```

## Current Interpretation

At this point the JAX version is no longer “far from the original” in terms of end-to-end output.

The most important remaining distinction is:

- The JAX planner now produces very similar trajectories and controls.
- The late-round `SubP2` steps are accepted as `main-soft`, not fully IPOPT-style strict convergence.

This is not a blind workaround. It is based on observed solver tail metrics:

- small `mu`
- small complementarity
- small dual residual
- small trace-space inequality residual

The main reason `main-soft` was needed is scale mismatch between:

- solver internal tail metrics
- outer raw feasibility metrics

## Practical Next Steps

If work continues from here, the highest-value next steps are:

1. Keep this regression baseline and avoid regressing `main-soft` behavior.
2. If more IPOPT-like strictness is needed, continue from solver scaling and raw-vs-trace inequality consistency.
3. If acceleration becomes the priority again, focus on batching `SubP2` workloads with JAX-native execution, not on the original IPOPT path.
4. Use the JSON regression export as the standard acceptance record after every important solver change.

## Stop-Loss Assessment for the Current JAX IPM Route

At this point it is reasonable to treat the current hand-written JAX interior-point method as a
**correctness / alignment baseline**, not as the most promising final high-performance solution.

This is not because the route was useless. In fact, it achieved several important goals:

- forward logic was aligned
- `SubP1` was aligned
- `SubP2` behavior became interpretable
- end-to-end JAX outputs became close to the original planner on the tested short-horizon cases
- several real bottlenecks were identified by profiling

However, the current evidence also says:

- the remaining gap is now mostly speed, not correctness
- multiple meaningful optimizations were already made
- those optimizations helped, but did not come close to closing the IPOPT speed gap
- the speed gap does not appear to be explained by one single trivial bug

So the practical engineering conclusion is:

- continuing to polish this **generic JAX IPM** is likely to have diminishing returns
- it remains valuable as a validated baseline
- but it should not automatically be assumed to be the final production path for high-performance `SubP2`

In other words:

- this route has been successful for **alignment and diagnosis**
- it has **not yet** been successful as a convincing IPOPT-speed replacement

## Recommended Direction After This Point

If the true project goal is still:

- keep hard constraints
- use JAX / `vmap` / GPU
- and obtain a meaningful forward-speed improvement

then the more promising next-stage direction is likely to be a **structure-aware dedicated `SubP2`
solver**, not endless refinement of the current generic dense JAX IPM.

Why:

- the current `SubP2` has fixed structure
- the real target is fast batched forward rollout, not a general NLP package clone
- a dedicated solver can be designed around this specific ADMM / quaternion / COM-dynamics problem
- that is more likely to convert JAX batching into real performance gains

So the recommended high-level decision is:

- keep the current JAX IPM as a correctness reference
- but do not continue investing in it as though it is guaranteed to evolve into an IPOPT-speed replacement

## Three-Route Decision Summary

At this stage there are effectively three candidate directions for `SubP2`:

1. Continue with the current JAX `ipoptax` route.
2. Continue with the Julia `MadNLP` route.
3. Continue with the postdoc's new JAX-friendly reformulation route.

Current judgment:

- **Route 1: JAX `ipoptax`**
  - keep it as the most mature correctness / regression baseline
  - it is still useful for diagnosis and as a reference solve
  - but it should no longer be treated as the main performance bet

- **Route 2: Julia `MadNLP`**
  - keep it as a serious fallback / next-stage option
  - it has already shown that exact single-step `SubP2` modeling can match the original problem very closely
  - but the current integration is still dominated by heavy modeling / bridge overhead, so it is not yet the preferred mainline

- **Route 3: postdoc's JAX-friendly reformulation**
  - this is the best fit for the true project goal:
    - remain in JAX
    - exploit `vmap`
    - exploit GPU
    - and target real forward-speed gains rather than reproducing a generic NLP solver
  - this route is still early, but strategically it is the most aligned with the final objective

Recommended priority order:

- primary line: the postdoc's new JAX-friendly reformulation
- secondary line: keep `ipoptax` as the correctness baseline and comparison point
- tertiary / backup line: keep Julia alive, but do not make it the primary engineering investment until the JAX-friendly reformulation has been judged fairly on the real `horizon=100` benchmark

Short version:

- `ipoptax` is the current reference
- Julia is the current fallback
- the new JAX-friendly reformulation should be the main next experiment

## Postdoc JAX-Friendly Reformulation Note

The handwritten note from the postdoc proposes a different `SubP2` strategy:

- do **not** solve the original constrained NLP with a full generic interior-point/KKT stack
- instead, rewrite each time-step `SubP2` as a sequence of **unconstrained** barrier problems
- then solve those barrier subproblems with JAX-native unconstrained optimizers such as:
  - `jaxopt.LBFGS`
  - `Optimistix`
  - or a similar `jit`/`vmap`-friendly optimizer

### 1. Original constrained per-step `SubP2`

At each time step, the note views `SubP2` abstractly as:

```math
\min_{z_k} \; f_{\mathrm{ADMM}}(z_k)
```

subject to:

```math
g_r(z_k) \le 0,\quad r = 1,\dots,n_g
```

```math
h_s(z_k) = 0,\quad s = 1,\dots,n_h
```

where:

- `z_k` is the per-step decision vector
- `f_ADMM` is the original ADMM consensus / tracking objective
- `g_r` are inequality constraints
- `h_s` are equality constraints

For the actual `SubP2` in this repository, the mapping is:

- decision vector
  - `z_k = [x_l, u_l, (x_i, u_i)_{i=1}^{n_q}]`
- equality families
  - load quaternion norm
  - cable direction norm
  - wrench consensus
- inequality families
  - load obstacle constraints
  - cable obstacle constraints
  - pair spacing / inter-agent safety
  - `gio_upper / gio_lower`
  - tension bounds
  - control bounds
  - thrust bounds

### 2. Barrier / penalty reformulation

The handwritten note proposes replacing the constrained problem by an unconstrained barrier objective:

```math
F_\mu(z_k)
=
f_{\mathrm{ADMM}}(z_k)
\;+\;
\sum_{r=1}^{n_g} \left[-\mu \log(-g_r(z_k))\right]
\;+\;
\frac{1}{2\mu}\sum_{s=1}^{n_h} \|h_s(z_k)\|^2
```

Interpretation:

- inequality constraints are enforced by a standard log barrier
- equality constraints are enforced by a quadratic penalty
- the barrier parameter `\mu > 0` is progressively reduced

In practical JAX implementations, this usually becomes a slightly more robust variant:

```math
F_\mu(z_k)
=
f_{\mathrm{ADMM}}(z_k)
\;+\;
\sum_r \left[-\mu \log(-g_r(z_k))\right]
\;+\;
\frac{\lambda_h}{2}\|h(z_k)\|^2
\;+\;
\frac{\lambda_g}{2}\|\max(g(z_k), 0)\|^2
```

because:

- pure log barriers are undefined once the iterate leaves the feasible interior
- a violated-inequality quadratic penalty is often needed for numerical robustness
- equality penalties usually need a multiplier / augmented term in practice

### 3. Continuation / outer-loop logic

The postdoc note suggests a standard barrier continuation:

```math
\mu^{(0)} = \mu_0 > 0
```

For `m = 0,1,2,\dots`:

1. solve

```math
z^{(m+1)} \approx \arg\min_z F_{\mu^{(m)}}(z)
```

using a JAX-native unconstrained optimizer

2. reduce the barrier parameter:

```math
\mu^{(m+1)} = \gamma \mu^{(m)}, \qquad \gamma \in (0,1)
```

3. stop when the outer iterates change little enough, e.g.

```math
\| z^{(m+1)} - z^{(m)} \| \le \delta
```

This is attractive for JAX because:

- the inner problem is now a pure function
- it is easy to `jit`
- it is easy to `vmap` across time steps
- it is much more natural for GPU execution than a heavy generic KKT-based solver

### 4. Why this is attractive, and what the risk is

Why it is attractive in this project:

- `SubP1/DDP` already provides a strong warm start
- `SubP2` is usually a local correction problem, not a cold-start global NLP
- this makes continuation + unconstrained local optimization plausible
- it aligns much better with the real goal: fast batched JAX forward passes

Important caveat:

- the inequality treatment is still in the spirit of a hard-constraint interior method
- but the equality treatment in the handwritten note is **penalty-based**
- so, taken literally, this is **not** the same as a strict hard-equality primal-dual IPM

That is why the practical recommendation is:

- keep the note's main idea:
  - inequality barrier
  - continuation
  - JAX-native unconstrained solver
- but improve the equality treatment:
  - use stronger augmented terms
  - or reparameterize / project special equality constraints such as unit-norm constraints
  - rather than relying only on a plain quadratic penalty

### 5. Current repository status for this idea

The first JAX prototype for this route is:

- [barrier_subp2_batched_snapshot.py](/home/mpc/xirui/meta-learn_useThis/python_archive/barrier_subp2_batched_snapshot.py)

An authoritative single-step benchmark for this route on a fixed real `SubP2`
snapshot is:

- [benchmark_real_subp2_postdoc.py](/home/mpc/xirui/meta-learn_useThis/python_archive/benchmark_real_subp2_postdoc.py)

That prototype already demonstrated:

- the barrier reformulation can be implemented in JAX
- `vmap` over time steps is straightforward
- feasibility can be improved substantially

But it also showed the current weakness:

- the current implementation tends to find a feasible point that is still not close enough to the IPOPT solution
- so the next version should keep the barrier idea, but use a stronger equality treatment than simple penalties

### Real single-step benchmark for the postdoc route

Using the same fixed real snapshot:

- `task=0`
- `horizon=2`
- `target_admm_iter=1`
- `target_step=0`

the current postdoc-style JAX barrier/continuation prototype gives:

CPU (`mu=1e-1,3e-2,1e-2,3e-3`, `LBFGS maxiter=60`):

- original IPOPT:
  - `wall_ms ≈ 10.90`
  - `eq_inf ≈ 1.88e-09`
  - `ineq_vio = 0`
- postdoc barrier prototype:
  - warm solve `≈ 5.12 - 5.20 s`
  - `eq_inf ≈ 4.02e-04`
  - `ineq_vio = 0`
  - `orig_rmse ≈ 2.06e-01`

CUDA, same configuration:

- original IPOPT:
  - `wall_ms ≈ 11.49`
- postdoc barrier prototype:
  - warm solve `≈ 18.73 s`
  - `eq_inf ≈ 4.03e-04`
  - `ineq_vio = 0`
  - `orig_rmse ≈ 2.07e-01`

Additional CPU variant (`--no-use-scales`, longer schedule, `LBFGS maxiter=80`) was
not better:

- warm solve `≈ 6.41 s`
- `eq_inf ≈ 5.96e-06`
- `ineq_vio = 0`
- `orig_rmse ≈ 2.10e-01`

Interpretation:

- the postdoc route is now implemented and benchmarked on a real fixed `SubP2`
  snapshot
- it does find a feasible point
- but the current equality treatment / local model is still too weak:
  the iterate remains far from the original IPOPT solution
- on this real single-step benchmark, the current version is also much slower than
  IPOPT, and CUDA is worse than CPU for now

### Hybrid equality-treatment follow-up

To test the stronger-equality idea, the same real-step benchmark
([benchmark_real_subp2_postdoc.py](/home/mpc/xirui/meta-learn_useThis/python_archive/benchmark_real_subp2_postdoc.py))
was extended with two changes:

- **project the unit-norm equalities**
  - load quaternion `xl[6:10]`
  - each cable direction `xc[:, 0:3]`
- **keep only the wrench-consensus equalities in the augmented term**
  - the norm equalities are no longer penalized directly

This hybrid version improved the real-step result modestly:

- CPU, default hybrid configuration:
  - original IPOPT:
    - `wall_ms ≈ 10.99`
  - hybrid barrier prototype:
    - warm solve `≈ 5.89 - 6.02 s`
    - `eq_inf ≈ 1.75e-04`
    - `ineq_vio = 0`
    - `orig_rmse ≈ 1.86e-01`

Compared to the earlier pure barrier/penalty variant (`orig_rmse ≈ 2.06e-01`),
this is a real improvement, but it is still far from the IPOPT step (`~3e-05`).

More aggressive wrench-equality penalties did **not** help:

- `eq_penalty_scale = 300`, `eq_lam_update_scale = 0.5`
  - warm solve `≈ 6.54 s`
  - `eq_inf ≈ 7.56e-05`
  - `orig_rmse ≈ 2.09e-01`
- `eq_penalty_scale = 1000`, `eq_lam_update_scale = 1.0`
  - warm solve `≈ 7.92 s`
  - `eq_inf ≈ 2.78e-06`
  - `orig_rmse ≈ 2.09e-01`

Current takeaway:

- projecting the norm equalities is better than penalizing them
- but simply increasing the wrench-consensus penalty does **not** pull the
  solution toward the IPOPT point
- the next useful change would be a stronger **wrench-equality correction**
  model, not just larger penalty weights

### Wrench projection and deeper continuation

The next variant projected **both**:

- unit-norm equalities
- wrench-consensus equality

That is, after each LBFGS update:

- `xl[6:10]` is renormalized
- each cable direction `xc[:,0:3]` is renormalized
- `ul[0:6]` is overwritten so that the generated wrench exactly matches
  `[R_l^T F_l, M_l]`

With this change, equality feasibility became essentially exact, and a deeper
continuation schedule began to approach the IPOPT point much more closely.

Representative real-step results (`task=0, horizon=2, admm_iter=1, step=0`):

- schedule `1e-1, 1e-2, 1e-3, 1e-4`, `LBFGS maxiter=120`
  - warm solve `≈ 17.58 s`
  - `eq_inf ≈ 1.78e-15`
  - `ineq_vio = 0`
  - `orig_rmse ≈ 1.10e-02`

- schedule `3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4`, `LBFGS maxiter=120`
  - warm solve `≈ 22.01 s`
  - `eq_inf ≈ 6.22e-15`
  - `ineq_vio = 0`
  - `orig_rmse ≈ 2.22e-03`

- schedule `1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4`, `LBFGS maxiter=120`
  - warm solve `≈ 23.65 s`
  - `eq_inf ≈ 1.78e-15`
  - `ineq_vio = 0`
  - `orig_rmse ≈ 1.49e-03`

- longer run with the same schedule and `LBFGS maxiter=200`
  - warm solve `≈ 21.32 s`
  - `eq_inf ≈ 8.88e-16`
  - `ineq_vio = 0`
  - `orig_rmse ≈ 1.48e-03`

Interpretation:

- this confirms that the postdoc route is **not** fundamentally stuck at
  `orig_rmse ≈ 2e-01`
- once equality handling is made much stronger, it can approach the IPOPT step
  to the `1e-03` level
- however, the current implementation is still far too slow on a single real
  step compared with IPOPT (`~11 ms`)

### Optimizer and reduced-space follow-up

The same projected/barrier formulation was then tested with a few implementation
changes to see whether the remaining gap is mostly an optimizer-choice issue:

- `BFGS` instead of `LBFGS`
- eliminating `ul` from the free variables (`--eliminate-ul`) and reconstructing
  it entirely through wrench projection
- `NonlinearCG`

Results on the same real step:

- `LBFGS`, long continuation:
  - `orig_rmse ≈ 1.49e-03`
  - warm solve `≈ 23.65 s`
- `BFGS`, same continuation:
  - `orig_rmse ≈ 1.48e-03`
  - warm solve `≈ 20.77 s`
- `BFGS + eliminate-ul`:
  - `orig_rmse ≈ 1.48e-03`
  - warm solve `≈ 21.00 s`
- `NonlinearCG`:
  - regressed to `orig_rmse ≈ 2.09e-01`
  - warm solve `≈ 20.84 s`

Current takeaway:

- `BFGS` is slightly better than `LBFGS` for this barrier objective
- simply removing `ul` from the free variables does **not** materially change
  the result
- `NonlinearCG` is not suitable here
- the remaining bottleneck appears to be the **barrier continuation / local
  model itself**, not the specific choice between `LBFGS` and `BFGS`

### Batched snapshot throughput: compile vs warm-run

The next important correction was performance methodology: the first timing of
the batched barrier prototype was dominated by JIT compilation. The script

- [barrier_subp2_batched_snapshot.py](/home/mpc/xirui/meta-learn_useThis/python_archive/barrier_subp2_batched_snapshot.py)

now supports `--bench-repeats`, so the second run in the same process is the
meaningful "compiled" timing.

#### Small batched snapshot (`horizon=2`, `target_admm_iter=1`, `N=3`)

With the projected/wrench-corrected hybrid barrier model and schedule
`1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4`:

- CPU, first run:
  - `final_ms ≈ 11.97 s`
- GPU, first run:
  - `final_ms ≈ 35.48 s`

This small batch is still too small for GPU to help.

#### Medium batched snapshot (`horizon=20`, `target_admm_iter=0`, `N=20`)

With the faster schedule `1e-3,3e-4,1e-4`, `BFGS`, `maxiter=80`, `tol=1e-4`:

- CPU:
  - run 1: `11.89 s`
  - run 2: `0.710 s`
  - final quality:
    - `eq_max ≈ 5.33e-15`
    - `ineq_max = 0`
    - `rmse_mean ≈ 5.35e-03`
- GPU:
  - run 1: `34.82 s`
  - run 2: `1.063 s`
  - final quality:
    - `eq_max ≈ 7.99e-15`
    - `ineq_max = 0`
    - `rmse_mean ≈ 5.48e-03`

Interpretation:

- once compilation is amortized, the barrier route is **much faster** than the
  first-run timings suggested
- at `N=20`, GPU is still slower than CPU, but no longer by a large margin

#### Formal-scale batched snapshot (`horizon=100`, `target_admm_iter=0`, `N=100`)

This is the first benchmark that starts to look like the intended `vmap/GPU`
deployment regime.

Using schedule `1e-3,3e-4,1e-4`, `BFGS`, `maxiter=80`, `tol=1e-4`:

- CPU:
  - run 1: `13.99 s`
  - run 2: `2.684 s`
  - final quality:
    - `eq_max ≈ 7.99e-15`
    - `ineq_max ≈ 1.37e-03`
    - `rmse_mean ≈ 5.20e-03`
- GPU:
  - run 1: `45.73 s`
  - run 2: `1.899 s`
  - final quality:
    - `eq_max ≈ 1.07e-14`
    - `ineq_max ≈ 1.37e-03`
    - `rmse_mean ≈ 5.18e-03`

Adding one more continuation stage (`3e-5`) improved accuracy while staying in
the same runtime regime:

- GPU:
  - run 2: `2.042 s`
  - final quality:
    - `eq_max ≈ 8.88e-15`
    - `ineq_max ≈ 4.81e-04`
    - `rmse_mean ≈ 2.85e-03`

For comparison, the original CasADi/IPOPT `SubP2` timing decomposition on the
same formal-scale experiment was approximately:

- ADMM iter 0: `≈ 2.17 s`
- ADMM iter 1: `≈ 0.99 s`
- ADMM iter 2: `≈ 0.88 s`

So the important revised conclusion is:

- the postdoc route is **not** fundamentally too slow once it is run in the
  intended batched/JIT regime
- at `N=100`, compiled GPU performance has already reached the same order of
  magnitude as the original IPOPT first-round `SubP2` batch, and is slightly
  faster than the compiled CPU version
- the remaining problem is now a speed/accuracy tradeoff:
  - `~1.90 s` on GPU gives `rmse_mean ≈ 5.2e-03`
  - `~2.04 s` on GPU gives `rmse_mean ≈ 2.85e-03`

#### Formal-scale fullflow barrier integration (`task=0`, `horizon=100`, `admm=3`)

The barrier route was then wired back into the real planner as
`JAX_SUBP2_SOLVER_MODE=barrier`. The first integration looked much worse than
the batched snapshot benchmark because the runtime cache for the batched BFGS
solver was kept on the planner instance, so every fresh `verify_jax_planner()`
construction paid the JIT/build cost again.

This was corrected by moving the barrier runtime caches to module scope in
[JustWorkingOnIt.py](/home/mpc/xirui/meta-learn_useThis/JustWorkingOnIt.py):

- `_GLOBAL_SUBP2_BARRIER_RUNTIME_COMMON_CACHE`
- `_GLOBAL_SUBP2_BARRIER_RUNTIME_STAGE_CACHE`

This allows repeated planner constructions in the same Python process to reuse
the same staged barrier kernels.

The current formal-scale results are:

- original CasADi/IPOPT fullflow:
  - run 2: `≈ 6.19 s` to `6.26 s`

- JAX barrier, 3-stage schedule `1e-3,3e-4,1e-4`, uniform `maxiter=60`:
  - run 2 wall: `≈ 20.18 s`
  - JAX calc: `≈ 13.36 s`
  - quality:
    - `load_control_ul rmse ≈ 2.13e-04`
    - `cable_state_xc rmse ≈ 5.29e-04`
    - `max_raw_ineq` by ADMM round:
      - iter 0: `0`
      - iter 1: `1.69e-02`
      - iter 2: `7.16e-02`

- JAX barrier, 4-stage schedule `1e-3,3e-4,1e-4,3e-5`, uniform `maxiter=60`:
  - run 2 wall: `≈ 20.74 s`
  - JAX calc: `≈ 14.08 s`
  - quality:
    - `load_control_ul rmse ≈ 2.06e-04`
    - `cable_state_xc rmse ≈ 5.17e-04`
    - `max_raw_ineq` by ADMM round:
      - iter 0: `0`
      - iter 1: `4.46e-03`
      - iter 2: `4.02e-03`

- JAX barrier, 4-stage schedule with staged budget
  `JAX_SUBP2_BARRIER_MAXITER_SCHEDULE=60,50,40,30`:
  - run 2 wall: `≈ 19.97 s`
  - JAX calc: `≈ 13.38 s`
  - quality:
    - `load_control_ul rmse ≈ 2.06e-04`
    - `cable_state_xc rmse ≈ 5.16e-04`
    - `max_raw_ineq` by ADMM round:
      - iter 0: `0`
      - iter 1: `4.64e-03`
      - iter 2: `1.58e-02`

- JAX barrier on `GPU1` (`CUDA_VISIBLE_DEVICES=1`,
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`) confirms the same scale:
  - original IPOPT fullflow run 2: `≈ 6.28 s`
  - JAX barrier fullflow run 2: `≈ 19.10 s`
  - quality:
    - `load_state_xl rmse ≈ 1.98e-04`
    - `load_control_ul rmse ≈ 2.08e-04`
    - `cable_state_xc rmse ≈ 5.16e-04`
  - conclusion:
    - the remaining gap is real and is not explained by GPU0 contention
    - GPU memory pressure was not the issue; kernel granularity / host overhead
      remain the dominant suspects

- Mixed-optimizer barrier experiments on the formal batched snapshot
  (`task=0`, `horizon=100`, `target_admm_iter=0`, `GPU1`) did **not** beat the
  4-stage BFGS baseline:
  - `optimizer_schedule=lbfgs,lbfgs,bfgs,bfgs`
    - compiled `final_ms ≈ 2.46 s`
    - but `rmse_mean ≈ 4.72e-03`, `ineq_max ≈ 2.64e-04`
  - `optimizer_schedule=lbfgs,lbfgs,lbfgs,bfgs`
    - compiled `final_ms ≈ 2.64 s`
    - but `rmse_mean ≈ 6.40e-02`, `ineq_max ≈ 1.95e-03`
  - interpretation:
    - front-loading `LBFGS` does accelerate the batched snapshot
    - but it pushes the continuation into the wrong basin
    - the later `BFGS` stages do not recover IPOPT-level accuracy
    - therefore mixed `LBFGS/BFGS` schedules are currently **not** suitable as
      the production barrier baseline

Current engineering choice:

- keep the **4-stage** schedule as the more robust barrier baseline
- use the staged-budget result as the current speed frontier
- do **not** switch to mixed `LBFGS/BFGS` schedules, even though they reduce
  snapshot runtime, because they hurt accuracy too much
- do **not** use the more aggressive 3-stage variant as the main baseline,
  because its later-ADMM raw inequality violation drifts too high

What changed in interpretation:

- before the cache refactor, the barrier fullflow route looked hopelessly slow
  (`~40 s` compiled wall on GPU)
- after isolating the real batched/JIT regime and fixing cache reuse, the same
  route is now in the `~20 s` compiled-wall regime
- on the real `horizon=100` benchmark this is still about `3.2x` slower than
  original IPOPT fullflow, but no longer an order-of-magnitude miss
- the dominant remaining task is now **reducing later-ADMM residual drift while
  keeping the staged barrier runtime low**, not basic JAX/JIT viability

#### Decoupled-ALM equality schedule inside the formal-scale barrier route

The next iteration pushed the postdoc route closer to the intended hybrid:

- projection for unit-norm equalities
- barrier for inequalities
- **decoupled ALM-style equality schedule** for the remaining wrench equality

This was exposed in
[JustWorkingOnIt.py](/home/mpc/xirui/meta-learn_useThis/JustWorkingOnIt.py)
through:

- `JAX_SUBP2_BARRIER_EQ_RHO`
- `JAX_SUBP2_BARRIER_EQ_RHO_SCHEDULE`
- `JAX_SUBP2_BARRIER_EQ_RHO_COUPLED_TO_MU`

The main tested setting was:

- `JAX_SUBP2_BARRIER_EQ_RHO_SCHEDULE=300,1000,3000,10000`
- `JAX_SUBP2_BARRIER_EQ_RHO_COUPLED_TO_MU=0`
- `JAX_SUBP2_BARRIER_PROJECT_UNIT_EQUALITIES=1`
- `JAX_SUBP2_BARRIER_PROJECT_WRENCH_EQUALITY=0`
- `JAX_SUBP2_BARRIER_WRENCH_ONLY_EQUALITY=1`

On the real formal-scale benchmark
`task=0, horizon=100, admm=3, CUDA_VISIBLE_DEVICES=1`, this produced:

- original IPOPT run 2:
  - `ORIGINAL Calculation Finished in ≈ 6.23 s`
- JAX decoupled-ALM barrier run 2:
  - `JAX Calculation Finished in ≈ 15.28 s`
  - quality:
    - `load_state_xl rmse ≈ 1.75e-04`
    - `load_control_ul rmse ≈ 1.69e-04`
    - `cable_state_xc rmse ≈ 5.33e-04`
    - `max_raw_ineq` by ADMM round:
      - iter 0: `7.99e-04`
      - iter 1: `4.41e-03`
      - iter 2: `3.90e-03`

Interpretation:

- compared with the earlier pure-barrier GPU1 baseline (`≈ 19.10 s` run 2),
  the decoupled-ALM equality treatment improves both time and later-round raw
  inequality residuals
- the route is still slower than IPOPT, but it is now closer to `~2.4x`
  slower rather than the earlier `~3.0x`

Two immediate follow-up attempts were then checked:

- **Tighter later-stage budget**
  - `JAX_SUBP2_BARRIER_MAXITER_SCHEDULE=60,40,25,15`
  - run 2:
    - `JAX Calculation Finished in ≈ 15.68 s`
    - `load_control_ul rmse ≈ 3.90e-04`
    - `max_raw_ineq` remained acceptable, but the trajectory accuracy degraded
  - conclusion:
    - cutting later-stage iterations this hard saves essentially nothing and
      hurts quality

- **Looser later-stage tolerance**
  - `JAX_SUBP2_BARRIER_TOL_SCHEDULE=1e-4,1e-4,2e-4,3e-4`
  - run 2:
    - `JAX Calculation Finished in ≈ 15.71 s`
    - `load_control_ul rmse ≈ 1.93e-04`
    - `cable_state_xc rmse ≈ 5.33e-04`
  - conclusion:
    - later-stage tolerance relaxation does not unlock another clear speed gain
      once the decoupled-ALM equality schedule is already in place

Current barrier-route interpretation after these tests:

- the hybrid postdoc route is better with **decoupled ALM equality weighting**
  than with the earlier `1 / mu`-coupled equality penalty
- however, simple schedule tweaks on top of that are no longer enough to
  create another large speed jump
- the remaining bottleneck now looks more like the coarse-grain execution shape
  of the barrier/BFGS pipeline than a single obviously wrong hyperparameter

## Structured SQP Progress

The current next-stage experiment is a structure-aware hard-constraint SQP route:

- outer loop: thin problem-specific SQP
- inner QP: `jaxopt.OSQP`
- derivatives: `jax.grad`, `jax.jacfwd`, `jax.hessian`

Main prototype files:

- [structured_sqp_subp2_prototype.py](/home/mpc/xirui/meta-learn_useThis/structured_sqp_subp2_prototype.py)
- [structured_sqp_subp2_batched_snapshot.py](/home/mpc/xirui/meta-learn_useThis/structured_sqp_subp2_batched_snapshot.py)

Important current finding:

- the earlier poor SQP behavior was largely caused by using too little of the true objective Hessian
- setting `objective_weight=1.0` was a major improvement
- large proximal pullback toward the initial warm start was not helping

Current best batched snapshot configuration found so far:

- `sqp_iters = 4` with early stop enabled
- `objective_weight = 1.0`
- `prox_weight = 0.0`
- `prox_center = init`
- `step_norm_cap = 1.0`
- `osqp_warm_start = 1`
- `qp_var_scale = 1`
- `qp_row_scale = 1`
- `early_stop = 1`
- `early_stop_eq = 2.5e-4`
- `early_stop_ineq = 1e-6`
- `early_stop_step = 1e-2`

On `task=0, horizon=2, target_admm_iter=1`:

- `IPOPTAX` snapshot:
  - `solve_ms = 12959.706`
  - `rmse_mean = 1.641e-04`
  - `rmse_max = 1.763e-04`
- `OSQP-SQP` snapshot with the configuration above:
  - effective stop after 3 SQP steps
  - `final_ms ≈ 2463.875`
  - `eq_max = 2.319e-06`
  - `ineq_max = 1.169e-07`
  - `rmse_mean = 8.575e-06`
  - `rmse_max = 9.650e-06`
  - `osqp_iter_mean ≈ 43.0, 42.5, 39.0`

Interpretation:

- this SQP route is already much faster than the current JAX IPM snapshot path
- it is still slower than original IPOPT by a large margin
- but on this snapshot it is now also more accurate than the current `ipoptax` reference path
- the key improvement was QP scaling:
  - variable scaling from the QP Hessian diagonal
  - row scaling for equality / inequality Jacobians

Current SQP bottleneck:

- after scaling, `OSQP` no longer immediately saturates `maxiter=200` on this bad snapshot
- the 4th SQP step is usually unnecessary once step norm collapses; early-stop can safely cut it
- the next likely step is to test whether this scaled SQP behavior survives:
  - more snapshots
  - batched terminal / non-terminal variants
  - eventual fullflow integration

## Large-Scale SQP Fullflow Status

The small-horizon numbers were misleading. The more relevant benchmark is now:

- `task=0`
- `horizon=100`
- `admm_iters=3`
- compare the original CasADi/IPOPT forward against the JAX planner with `JAX_SUBP2_SOLVER_MODE=sqp`

### CPU

With the current SQP defaults:

- original IPOPT fullflow:
  - run 1: `~6.78 s`
  - run 2: `~6.23 s`
- JAX SQP fullflow on CPU:
  - run 1: `~29.74 s`
  - run 2 compiled: `~26.69 s`

So on realistic scale the CPU gap is no longer `26x`; it is closer to `4.3x`.

### GPU

On real NVIDIA GPU with the same configuration:

- original IPOPT fullflow:
  - run 1: `~6.30 s`
  - run 2: `~6.06 s`
- JAX SQP fullflow on GPU:
  - run 1: `~29.79 s`
  - run 2 compiled: `~18.86 s`

So the current realistic large-scale result is:

- GPU does help relative to the CPU SQP path
- but JAX SQP is still about `3.1x` slower than original IPOPT at `horizon=100`

The large-scale trajectory mismatch at this point is still moderate:

- `load_state_xl rmse ≈ 3.37e-04`
- `load_control_ul rmse ≈ 6.92e-04`
- `cable_state_xc rmse ≈ 5.02e-04`

### Active-Set / Mixed-Precision Follow-Up

Recent large-scale follow-up experiments:

- GPU `active_set_k=16`:
  - direct full active-set path hit CUDA OOM in this environment
- GPU two-stage active-set (`active_set_after_iter=2`):
  - compiled JAX time `~25.42 s`
  - worse than the current GPU baseline
  - equality residuals became noticeably worse
- GPU internal `float32` SQP kernel:
  - after fixing a dtype bug in `project_psd_cone`, the path ran successfully
  - compiled JAX time `~21.85 s`
  - still slower than the current GPU baseline `~18.86 s`
  - accuracy stayed essentially unchanged

Current conclusion:

- the current best known realistic benchmark is still the default GPU SQP path with compiled time around `18.86 s`
- `vmap` is being used in the SQP time-step batch, but these large-scale numbers are still not enough to beat the original IPOPT implementation
- future work should keep using the `horizon=100` GPU benchmark as the primary decision metric rather than relying on small-horizon timing

## Notes on GPU Messages

When running CPU verification in the current environment, JAX may still print:

- `cuInit(0) failed: CUDA_ERROR_OPERATING_SYSTEM`

In the current workflow this does not block CPU verification or JSON summary export. It is an environment/plugin issue, not a planner-logic issue.

## Current Project Judgment

At this stage, the most honest high-level judgment is:

- the JAX planner has already reached the "result alignment" goal to a useful extent
- but the original "must be faster than the original IPOPT implementation" goal is still not met

More specifically:

- for the old generic JAX IPM route, continuing to invest heavily no longer looks attractive
- for the current `SQP + jaxopt.OSQP` route, the picture is better:
  - the gap at realistic scale is no longer extreme
  - but even the best current GPU result is still about `3.1x` slower than the original IPOPT fullflow

So the correct status is not:

- "JAX as a whole is impossible"

It is closer to:

- "the current `SubP2` solver generation has not yet achieved the target speed"

This distinction matters for planning the next phase.

## Julia Assessment For `SubP2`

The current evidence suggests that Julia is no longer just a speculative fallback.
It is now a serious performance branch for `SubP2`, but only if it is used with:

- a persistent Julia worker
- parameterized model reuse
- solver/warm-start reuse
- no per-step process startup

Important judgment:

- rewriting the whole stack in Julia and still using a generic NLP solver would probably not produce a dramatic speedup by itself
- the more promising Julia path is:
  - keep the current high-level planner logic as reference
  - rewrite only `SubP2`
  - use a structure-aware hard-constraint Julia solver stack

The main reason this is plausible is that Julia has a more mature path for high-performance constrained NLP than our current hand-built JAX solver path, especially for sparse / structured problems.

The most plausible Julia stack for a `SubP2` prototype is:

- modeling:
  - `JuMP.jl`
- NLP bridge:
  - `NLPModelsJuMP.jl`
- interior-point NLP:
  - `MadNLP.jl`
- optionally later for GPU linear algebra:
  - `CUDA.jl`
  - `cuDSS`

Why this stack is relevant:

- `MadNLP.jl` is an interior-point, filter line-search style NLP solver in the Ipopt family
- `MadNLP` is built on top of the `NLPModels.jl` ecosystem
- `NLPModelsJuMP.jl` is the standard bridge that turns a `JuMP` model into an
  `AbstractNLPModel` that `MadNLP` can solve
- `MadNLP` is also interfaced with other model providers such as `ExaModels` and `Plasmo`
- for this project, the most stable and official path proved to be:
  - `JuMP -> NLPModelsJuMP.MathOptNLPModel -> MadNLP`

Important caveat:

- the likely winning Julia path is not "call Julia once per time step"
- the crossing overhead would destroy much of the gain
- if Julia is used, it should take one full `SubP2` batch per ADMM iteration, not one tiny subproblem call at a time

Current recommendation:

- keep the current JAX routes as active baselines and reformulation experiments
- treat Julia as a serious performance branch for `SubP2`
- do not rewrite the whole stack in Julia
- instead, push:
  - persistent Julia worker
  - parameterized model reuse
  - solver-state reuse
  - then GPU linear solver experiments

## Julia `SubP2` Refactor Status

The Julia branch is now past environment setup and has entered exact model replication.

Implemented bridge and validation files:

- Python exporter:
  - [export_subp2_snapshot_for_julia.py](/home/mpc/xirui/meta-learn_useThis/archive/export_subp2_snapshot_for_julia.py)
- Julia contract checker:
  - [julia_subp2/load_snapshot.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/archive/load_snapshot.jl)
- Julia step-model validator:
  - [julia_subp2/validate_step_model.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/archive/validate_step_model.jl)
- contract note:
  - [julia_subp2/SUBP2_CONTRACT.md](/home/mpc/xirui/meta-learn_useThis/julia_subp2/archive/SUBP2_CONTRACT.md)

Current status:

- the Python side can export an exact `SubP2` snapshot contract to JSON
- the Julia side can load that snapshot successfully
- for the tested snapshot (`task=0, horizon=2, admm_iter=1, step=0`), the replicated Julia step model now matches Python/JAX exactly at the level of:
  - objective value
  - equality residual vector
  - inequality residual vector

Validation result:

- objective max diff: `0.0`
- equality max diff: `~1e-17`
- inequality max diff:
  - `0.0` at `x_init`
  - `~2e-16` at original-step reference

Important implication:

- the current Julia branch has already fixed the hardest modeling-risk issue:
  - objective/constraints are now replicated correctly for one real `SubP2` step
- the next stage is no longer "figure out the model"
- the next stage is:
  - wrap the same exact model into `ExaModels + MadNLP`
  - solve the exported step
  - compare speed and solution quality against:
    - original IPOPT step
    - current JAX SQP step

## SubP1 Benchmark: Does JAX + `vmap` Actually Help?

Question checked before investing more into Julia `SubP2`:

- is JAX `SubP1` actually faster than the original?
- does `vmap` materially help, or is `SubP1` not worth worrying about?

Benchmark setup:

- script: [benchmark_subp1_vmap.py](/home/mpc/xirui/meta-learn_useThis/python_archive/benchmark_subp1_vmap.py)
- task: `task=0`
- medium scale: `horizon=20`, `admm_iters=3`
- JAX run on `cuda`
- numbers below use the **second call in the same process**, i.e. compiled steady-state

Measured times:

- original `SubP1`
  - load: `42.93 ms`
  - cable: `114.24 ms`
  - total: `157.17 ms`
- JAX load `SubP1`
  - `58.64 ms`
- JAX cable `SubP1`, `JAX_CABLE_SUBP1_MODE=batched`
  - `79.95 ms`
- JAX cable `SubP1`, `JAX_CABLE_SUBP1_MODE=legacy`
  - `119.25 ms`
- JAX total `SubP1`, cable `batched`
  - `138.59 ms`
- JAX total `SubP1`, cable `legacy`
  - `177.90 ms`

Interpretation:

- `vmap` **does help** for `SubP1`, but mainly on the cable side.
- JAX cable `batched` is clearly faster than JAX cable `legacy`:
  - `79.95 ms` vs `119.25 ms`
  - about `1.5x` faster
- JAX load `SubP1` is **not** faster than original:
  - `58.64 ms` vs `42.93 ms`
- overall `SubP1` with JAX batched cable is only **slightly** faster than original:
  - `138.59 ms` vs `157.17 ms`
  - about `12%` faster

Practical judgment:

- `SubP1` is **not** the main problem anymore.
- `vmap` is real and useful there, especially for cable DDP.
- but the gain is modest compared with the much larger `SubP2` bottleneck.
- so if effort needs to be prioritized:
  - do **not** spend much more time optimizing `SubP1`
  - keep current batched JAX `SubP1`
  - focus on `SubP2`

## Julia `SubP2`: Current State After Starting `MadNLP` Integration

New files added:

- [julia_subp2/step_model.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/step_model.jl)
- [julia_subp2/validate_step_model.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/archive/validate_step_model.jl)
- [julia_subp2/solve_step_madnlp.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/solve_step_madnlp.jl)

What is already working:

- exact Python-to-Julia snapshot export is working
- Julia-side numeric step model matches Python/JAX exactly
- shared step-model code was factored into `step_model.jl`
- numeric validation still passes after the refactor

What was attempted next:

- build a first single-step `ExaModels + MadNLP` solve prototype
- same tested snapshot:
  - `task=0`
  - `horizon=2`
  - `admm_iter=1`
  - `step=0`

Current blocker:

- the bottleneck is **not** objective/constraint correctness anymore
- the blocker is how to express the already-correct step model in a way that `ExaModels` accepts efficiently

Specific issues encountered:

- `ExaModels.Variable` does not like the earlier `reshape` / range-slice based unpacking
  - fixed by switching to explicit scalar indexing
- `ExaModels` symbolic nodes do not tolerate several `Base` array idioms:
  - `sum(...; dims=2)`
  - `zeros(Any, ...)`
  - `Float64(...)` casts on symbolic nodes
  - broadcasting a Julia vector with a symbolic scalar like `vec .* ti[i]`
  - these were progressively rewritten into explicit scalar expressions
- the bigger structural issue is with `ExaModels` constraint generators:
  - generator indices are passed in as `ExaModels.ParSource()`, not plain `Int`
  - so the current "build full residual vector, then take component `idx`" pattern does **not** work during symbolic model construction
- trying to avoid that by adding scalar constraints one-by-one caused a deep nested `Constraint{Constraint{...}}` stack-overflow during model construction

Current judgment:

- Julia `MadNLP` integration has **not** failed on math fidelity
- it is currently blocked at the **symbolic model-construction layer**
- the next viable path is:
  - either rewrite the Julia `ExaModels` build path in true generator-native form block-by-block
    - i.e. write each constraint family directly as its own generator, instead of indexing into a prebuilt residual vector
  - or switch to a different Julia NLP modeling interface that is friendlier for callback-style construction

Recommended next step for the Julia line:

- keep `step_model.jl` as the single source of truth for exact numeric parity
- for `ExaModels`, stop trying to build constraints by `component(residual, idx)`
- instead, rebuild the `MadNLP` prototype by constraint family:
  - objective
  - quaternion norm equality
  - cable norm equalities
  - wrench equalities
  - obstacle inequalities
  - pair/gio inequalities
  - tension/control/thrust inequalities
- only after that should solving speed be benchmarked

### Update: equality-only `MadNLP` prototype now solves

After starting the native `ExaModels` rewrite:

- `build_examodel(...; include_ineq=false)` now works for the equality-only model
- command:
  - `env JULIA_DEPOT_PATH=/tmp/julia-depot:/home/mpc/.julia julia julia_subp2/solve_step_madnlp.jl julia_subp2/archive/test_snapshot.json /tmp/julia_subp2_step_eq_only.json --eq-only`
- result:
  - `status = SOLVE_SUCCEEDED`
  - `iter = 3`
  - `wall_ms ≈ 51881`
  - `eq_inf ≈ 4.17e-12`
  - `orig_rmse ≈ 1.42e-04`
  - `orig_max_abs ≈ 8.71e-04`

Interpretation:

- the native equality rewrite is real and no longer blocked by the old residual-component bridge
- the remaining blocker for the full model is on the inequality families only

Full-model current blocker:

- the full solve still fails during `ExaModels` model construction
- current failing pattern is:
  - generator index is `ParSource()`
  - constant arrays such as `ra[i, :]` and other prebuilt Julia arrays/matrices cannot be indexed by `ParSource()`
- so the next step is:
  - rewrite inequality families to avoid direct Julia-array indexing by generator indices
  - either by flattening needed constants into `ExaModels.parameter(...)`
  - or by expanding families into forms that only index the decision vector `x[...]`

### Update: Python <-> Julia bridge overhead is large

I measured the current bridge overhead explicitly on the same real single-step snapshot:

- script:
  - [benchmark_python_julia_bridge.py](/home/mpc/xirui/meta-learn_useThis/python_archive/benchmark_python_julia_bridge.py)
- case:
  - `task=0`
  - `horizon=2`
  - `target_admm_iter=1`
  - `target_step=0`
  - Julia solver:
    - [julia_subp2/solve_step_madnlp_jump_native_eq.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/solve_step_madnlp_jump_native_eq.jl)

Measured averages over 3 repeats:

- Python payload construction:
  - `build_payload_ms ≈ 131`
- JSON dump:
  - `dump_ms ≈ 0.86`
- total Python subprocess call:
  - `subprocess_ms ≈ 19049.8`
- Julia-reported pure solve time:
  - `julia_wall_ms ≈ 6132.3`
- JSON load back in Python:
  - `load_ms ≈ 0.23`
- implied bridge/process overhead:
  - `bridge_overhead_ms ≈ 12917.5`

Interpretation:

- the current cross-language path is **not** cheap
- the dominant extra cost is **not** JSON serialization itself
  - JSON write/read are sub-millisecond
- the dominant extra cost is:
  - starting a fresh Julia process
  - loading packages / JIT / model construction inside that process

Practical conclusion:

- a production Julia `SubP2` path must **not** launch Julia once per time step through `subprocess`
- if Julia is kept, the integration must switch to one of:
  - a persistent Julia worker/service
  - a long-lived Julia session for the whole `SubP2` batch / ADMM iteration
  - or a batch interface that solves many time steps in one Julia call

Otherwise, the Python<->Julia bridge overhead will dominate any solver-side speedup.

### Update: single-step native JuMP model is now family-wise exact on the test snapshot

For `julia_subp2/archive/test_snapshot.json`:

- native JuMP route:
  - `JuMP -> NLPModelsJuMP.MathOptNLPModel -> MadNLP`
- audit script:
  - [julia_subp2/audit_native_families.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/archive/audit_native_families.jl)

Family-wise audit at both `x_init` and `x_orig` now matches the shared numerical model
in [julia_subp2/step_model.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/step_model.jl)
to machine precision:

- `overall max_abs_diff <= 8.88e-16`

The last issues that were fixed were:

- control-family audit ordering:
  - audit used column-major flattening instead of the residual's row-major family order
- thrust-family helper bug:
  - the Julia helper added an extra `+ g` in the third thrust component

After these fixes, all audited families match:

- `eq_load_quat`
- `eq_cable_norm`
- `eq_wrench`
- `ineq_load_obs`
- `ineq_cable_obs`
- `ineq_gio_upper`
- `ineq_gio_lower`
- `ineq_pair`
- `ineq_tension_lower`
- `ineq_tension_upper`
- `ineq_control_upper`
- `ineq_control_lower`
- `ineq_thrust_upper`
- `ineq_thrust_lower`

The single-step native JuMP/MadNLP solve remains stable after the fixes:

- `status = SOLVE_SUCCEEDED`
- `iter = 18`
- `wall_ms ≈ 5.99e3`
- `eq_inf ≈ 3.79e-10`
- `ineq_vio = 0`
- `orig_rmse ≈ 9.03e-06`

### Update: small multi-snapshot family audit also passes

New tool:

- [audit_julia_subp2_snapshots.py](/home/mpc/xirui/meta-learn_useThis/python_archive/archive/audit_julia_subp2_snapshots.py)

This exports real snapshots from the Python planner and, for each snapshot:

  - runs [julia_subp2/audit_native_families.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/archive/audit_native_families.jl)
- runs [julia_subp2/solve_step_madnlp_jump_native_eq.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/solve_step_madnlp_jump_native_eq.jl)

Verified set:

- `task=0`
- `horizon=2`
- `admm_iter in {0, 1}`
- `step in {0, 1, 2}`

Result:

- all 6 snapshots pass family-wise audit at machine precision
  - worst observed family diff: `7.105e-15`
- solve statuses:
  - mixed `SOLVE_SUCCEEDED` / `SOLVED_TO_ACCEPTABLE_LEVEL`
  - but all with very small distance to original step references
- observed `orig_rmse` range on this audit set:
  - about `1.26e-07` to `2.19e-05`

This means the current native JuMP/MadNLP single-step Julia model is no longer
validated only on one hand-picked snapshot; it is now consistent on a small real
multi-snapshot set extracted from the Python planner.

### Update: Julia fullflow experiment now uses one Julia call per ADMM `SubP2` batch

File:

- [run_julia_subp2_fullflow_experiment.py](/home/mpc/xirui/meta-learn_useThis/python_archive/run_julia_subp2_fullflow_experiment.py)
- [julia_subp2/solve_batch_madnlp_jump_native.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/solve_batch_madnlp_jump_native.jl)

The old experiment launched a fresh Julia process for every time step.
The current experiment exports one full `SubP2` batch and solves all time steps in
one Julia process per ADMM iteration.

On `task=0, horizon=100, admm=3`:

- original fullflow: about `6.41s`
- Julia batched fullflow experiment:
  - `JAX Calculation Finished in 72329.94 ms`
  - total wrapper wall time: `79606.44 ms`

Current end-to-end differences on that experiment:

- `load_state_xl rmse = 1.815e-04`
- `load_control_ul rmse = 1.551e-04`
- `cable_state_xc rmse = 4.784e-04`

and all `SubP2` steps were reported as `main:101` with very small raw feasibility:

- `max_eq` around `1e-9`
- `max_raw_ineq` around `1e-8`

So:

- the Julia native solver path is now integrated deeply enough to run a real
  fullflow replacement experiment
- but performance is still far from production-ready because the bridge is only
  partially fixed; a persistent Julia worker / in-process bridge is still needed

### Update: persistent Julia worker cuts most of the bridge/startup overhead

Files:

- [run_julia_subp2_fullflow_persistent.py](/home/mpc/xirui/meta-learn_useThis/run_julia_subp2_fullflow_persistent.py)
- [julia_subp2/worker_batch_madnlp_jump_native.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/worker_batch_madnlp_jump_native.jl)

This version keeps one Julia process alive across repeated fullflow runs and
reuses package load + JIT warmup. It still rebuilds JuMP/NLPModels models for
each step, but it removes the dominant process-boundary overhead from the older
batch-per-ADMM experiment.

Small-scale check (`task=0, horizon=2, admm=2`):

- repeat 1:
  - fullflow `18954.4 ms`
  - `SubP2` build `3257.8 ms`
  - `SubP2` solve `6101.3 ms`
  - `SubP2` total `11261.9 ms`
- repeat 2 (same Julia worker):
  - fullflow `2674.8 ms`
  - `SubP2` build `47.0 ms`
  - `SubP2` solve `64.2 ms`
  - `SubP2` total `113.8 ms`

Formal scale (`task=0, horizon=100, admm=3`):

- original CasADi/IPOPT fullflow repeat 2: about `6101 ms`
- persistent Julia fullflow
  - repeat 1:
    - fullflow `36786.0 ms`
    - `SubP2` build `5992.7 ms`
    - `SubP2` solve `9673.8 ms`
    - `SubP2` total `17725.8 ms`
  - repeat 2:
    - fullflow `20465.8 ms`
    - `SubP2` build `3000.9 ms`
    - `SubP2` solve `3437.6 ms`
    - `SubP2` total `6595.0 ms`

Current repeat-2 formal-scale quality remains unchanged:

- `load_state_xl rmse = 1.815e-04`
- `load_control_ul rmse = 1.551e-04`
- `cable_state_xc rmse = 4.784e-04`

Interpretation:

- the previous `~40.5s` Julia `SubP2` number was dominated by repeated
  Julia startup / package load / JIT warmup
- once Julia is kept alive, formal-scale `SubP2` drops to about `6.6s`
  total across 3 ADMM rounds
- that is now only about `1.6x` slower than the original CasADi/IPOPT
  `SubP2` total (`~4.04s`)
- fullflow is still slower than original because `SubP1` and the outer Python
  integration remain in the loop, but the Julia `SubP2` route is no longer
  obviously ruled out on performance grounds

### Update: cache exact step content and reuse `MadNLPSolver`

Files:

- [julia_subp2/solve_batch_madnlp_jump_native.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/solve_batch_madnlp_jump_native.jl)
- [julia_subp2/worker_batch_madnlp_jump_native.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/worker_batch_madnlp_jump_native.jl)
- [run_julia_subp2_fullflow_persistent.py](/home/mpc/xirui/meta-learn_useThis/run_julia_subp2_fullflow_persistent.py)

The next optimization step was to cache not only the Julia worker process, but
also each exact `step` model inside that worker:

- key each step by a content hash of the exported JSON payload
- build `JuMP -> NLPModelsJuMP -> MadNLPSolver` only on cache miss
- on cache hit:
  - reuse the cached `NLPModel`
  - reuse the cached `MadNLPSolver`
  - warm-start from the previous primal solution

This is not yet a fully parameterized reusable model for changing MPC states,
but it shows what happens when model build cost is driven to zero for repeated
identical formal-scale calls.

Updated measurements:

Small-scale (`task=0, horizon=2, admm=2`):

- repeat 1:
  - fullflow `19575.0 ms`
  - `SubP2` build `4311.3 ms`
  - `SubP2` solve `6826.0 ms`
  - `SubP2` total `11701.8 ms`
- repeat 2 (same worker, exact-step cache hits):
  - fullflow `2584.1 ms`
  - `SubP2` build `0.0 ms`
  - `SubP2` solve `7.4 ms`
  - `SubP2` total `8.8 ms`

Formal scale (`task=0, horizon=100, admm=3`):

- original CasADi/IPOPT fullflow repeat 2: about `6110 ms`
- original CasADi/IPOPT `SubP2` total: about `4042 ms`
- persistent Julia + exact-step cache:
  - repeat 1:
    - fullflow `39266.9 ms`
    - `SubP2` build `9186.5 ms`
    - `SubP2` solve `9929.7 ms`
    - `SubP2` total `19787.8 ms`
  - repeat 2:
    - fullflow `14299.0 ms`
    - `SubP2` build `0.0 ms`
    - `SubP2` solve `315.1 ms`
    - `SubP2` total `384.8 ms`

Quality on repeat 2 remained unchanged:

- `load_state_xl rmse = 1.815e-04`
- `load_control_ul rmse = 1.551e-04`
- `cable_state_xc rmse = 4.784e-04`

Interpretation:

- the dominant remaining Julia cost after removing process startup was still
  per-step model construction
- once that construction is eliminated for repeated identical batches, Julia
  `SubP2` becomes dramatically faster
- on the repeat-2 formal-scale benchmark, cached Julia `SubP2` (`~385 ms`)
  is now much faster than the original CasADi/IPOPT `SubP2` total (`~4042 ms`)
- this confirms that the earlier bad Julia numbers were mostly an integration
  issue, not evidence that `MadNLP` itself is intrinsically too slow
- the remaining open problem is generalization:
  - exact-step caching is excellent for repeated identical calls
  - but real MPC needs reuse for *similar but changing* step parameters, which
    requires parameterized model/state reuse rather than exact-hash reuse

### Update: persistent Julia worker with parameterized model reuse

The Julia worker was then pushed beyond:

- process reuse only
- and beyond exact-step hashing only

The key change was to maintain long-lived solver objects inside the worker and
reuse them across repeated calls, while isolating warm-start state by step slot
instead of by thread.

Important files:

- [julia_subp2/solve_batch_madnlp_jump_native.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/solve_batch_madnlp_jump_native.jl)
- [julia_subp2/worker_batch_madnlp_jump_native.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/worker_batch_madnlp_jump_native.jl)
- [run_julia_subp2_fullflow_persistent.py](/home/mpc/xirui/meta-learn_useThis/run_julia_subp2_fullflow_persistent.py)

Current implementation status:

- one persistent Julia process
- batched `SubP2` request per ADMM iteration
- threaded outer loop over time steps
- cached `JuMP -> NLPModelsJuMP -> MadNLP` step solvers
- warm-start reuse isolated by `target_step`

It is important to be precise about what this is and is not:

- it **is** persistent process reuse
- it **is** solver-state / warm-start reuse
- it **is not yet** full symbolic parameter reuse for arbitrarily changing step data
- it **is not yet** GPU-parallel MadNLP

However, this change alone dramatically improved the realistic Julia timings.

Formal-scale benchmark (`task=0, horizon=100, admm=3`):

- original CasADi/IPOPT fullflow repeat 2: `~6.1 s`
- original CasADi/IPOPT `SubP2` total: `~4.0 s`

Persistent Julia worker, slot-isolated cached run:

- repeat 2 fullflow: `~15.4 s`
- repeat 2 `SubP2` total: `~1.63 s`
- repeat 2 `SubP2` build: `0.0 ms`
- repeat 2 reported `SubP2` solve: `~11.45 s`

Quality remained stable:

- `load_state_xl rmse = 1.815e-04`
- `load_control_ul rmse = 1.551e-04`
- `cable_state_xc rmse = 4.784e-04`

Main interpretation:

- the earlier `~40 s` Julia `SubP2` number was mostly an integration artifact
- once startup and repeated model construction are removed, Julia becomes competitive
- the remaining Julia bottleneck is no longer "does MadNLP work?"
- it is now:
  - true parameterized model reuse for changing step values
  - then, after that, possible GPU linear solver acceleration

### Update: removing the old 8-thread cap helps the persistent Julia branch

The persistent Julia worker had been launched with an unnecessary cap:

- `JULIA_NUM_THREADS = min(cpu_count, 8)`

On this machine `nproc = 20`, so the Julia branch was not using all available
CPU threads. The runner and bridge benchmark were updated so that:

- the persistent worker defaults to all visible CPU cores
- worker startup reports `nthreads`
- a `--julia-threads` override is available for explicit benchmarking

Relevant files:

- [run_julia_subp2_fullflow_persistent.py](/home/mpc/xirui/meta-learn_useThis/run_julia_subp2_fullflow_persistent.py)
- [python_archive/benchmark_julia_worker_bridge.py](/home/mpc/xirui/meta-learn_useThis/python_archive/benchmark_julia_worker_bridge.py)
- [julia_subp2/worker_batch_madnlp_jump_native.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/worker_batch_madnlp_jump_native.jl)

Formal-scale benchmark (`task=0, horizon=100, admm=3`, persistent worker):

With `8` threads:

- repeat 2 fullflow: `~16.61 s`
- repeat 2 `SubP2` total (`batch_total_ms` summed over 3 ADMM rounds): `~2147 ms`

With `20` threads:

- repeat 2 fullflow: `~15.72 s`
- repeat 2 `SubP2` total (`batch_total_ms` summed over 3 ADMM rounds): `~1643 ms`

Interpretation:

- fullflow wall time improves modestly (`~5%`)
- `SubP2` wall-clock batch latency improves more clearly (`~23%`)
- multi-step CPU parallelism is therefore now a real contributor to Julia-side
  speed, and the old `8`-thread cap should not be used again

Important note:

- the summed per-step `solve_ms` values can be much larger than `batch_total_ms`
- this is expected under threaded execution
- `batch_total_ms` is the wall-clock latency that should be used for real
  performance comparisons

### Update: BLAS oversubscription was real; `20` threads + `BLAS=1` is the best CPU setting so far

`LinearAlgebra.BLAS.get_num_threads()` on this machine reported `10`, which means
the threaded Julia worker could easily oversubscribe the CPU when `Threads.@threads`
was already parallelizing over the `101` time steps. The persistent worker was
therefore updated to force:

- `JULIA_BLAS_THREADS=1` by default
- startup reporting of both `nthreads` and `blas_threads`

Relevant files:

- [julia_subp2/worker_batch_madnlp_jump_native.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/worker_batch_madnlp_jump_native.jl)
- [run_julia_subp2_fullflow_persistent.py](/home/mpc/xirui/meta-learn_useThis/run_julia_subp2_fullflow_persistent.py)

Formal-scale benchmark (`task=0, horizon=100, admm=3`, repeat-2 warm run):

- `8` Julia threads, `BLAS=1`
  - original fullflow: `~6259 ms`
  - JAX+Julia fullflow: `~6862 ms`
  - Julia `SubP2` wall-clock: `~2368 ms`
- `16` Julia threads, `BLAS=1`
  - original fullflow: `~5916 ms`
  - JAX+Julia fullflow: `~6074 ms`
  - Julia `SubP2` wall-clock: `~1780 ms`
- `20` Julia threads, `BLAS=1`
  - original fullflow: `~5928 ms`
  - JAX+Julia fullflow: `~5074 ms`
  - Julia `SubP2` wall-clock: `~1234 ms`

Interpretation:

- `20` worker threads with `BLAS=1` is currently the best known CPU configuration.
- Reducing the worker thread count hurts both fullflow time and `SubP2` latency.
- After fixing BLAS oversubscription, the persistent Julia branch is now clearly
  compute-bound on solver work rather than on transport or startup overhead.

### Update: current Python <-> Julia worker transmission overhead is small

To verify whether the next Julia priority should be:

- `PythonCall.jl`
- `JuliaCall`
- or `DLPack` zero-copy transport

the current persistent worker bridge was benchmarked directly with:

- [python_archive/benchmark_julia_worker_bridge.py](/home/mpc/xirui/meta-learn_useThis/python_archive/benchmark_julia_worker_bridge.py)

This benchmark uses the current production-style transport:

- Python writes one representative batch JSON file
- Python sends a small command over stdin/stdout to the Julia worker
- Julia only loads/parses the batch and returns a tiny response
- no actual solve is performed

Representative result for a `101`-step batch:

- batch JSON size: `~1.30 MB`
- Python write-to-disk: `~55.1 ms`
- worker `ping` RTT: `~0.22 ms`
- `load_batch_only` RTT: `~13.18 ms`
- Julia-side file read + JSON parse: `~4.96 ms`
- non-parse remainder inside roundtrip: `~8.22 ms`

Interpretation:

- the current command-channel roundtrip is already tiny
- JSON/file transport is measurable, but still only on the order of tens of milliseconds
- compared with current formal-scale Julia `SubP2` solve times, this is **not** the dominant bottleneck

Practical conclusion:

- `PythonCall.jl` / `JuliaCall` / `DLPack` may still be worthwhile later
- but they are **not** the first performance lever right now
- current priority should remain:
  - true parameterized model reuse
  - solver-state reuse
  - then, only after that, transport micro-optimization

### Update: slot-based parameterized reuse makes the persistent Julia branch actually fast

The previous persistent-worker path still left a major inefficiency:

- cache hits required exact repeated `step_json`
- similar but not identical steps still fell back to expensive rebuild-style work

This was improved by pushing the worker beyond exact-step hashing:

- keep one persistent Julia process alive
- keep one cached solver object per step slot
- update cached models from stacked runtime payloads
- reuse warm-start state by slot

Important files:

- [julia_subp2/solve_batch_madnlp_jump_native.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/solve_batch_madnlp_jump_native.jl)
- [julia_subp2/worker_batch_madnlp_jump_native.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/worker_batch_madnlp_jump_native.jl)
- [run_julia_subp2_fullflow_persistent.py](/home/mpc/xirui/meta-learn_useThis/run_julia_subp2_fullflow_persistent.py)

This is still not full symbolic parameter reuse in the strict JuMP sense, but it
is already much closer to a true "build once / solve many" execution model than
the earlier exact-hash cache.

Formal-scale benchmark (`task=0, horizon=100, admm=3`, persistent worker,
`20` Julia threads, `BLAS=1`, repeat-2 warm run):

- best observed original fullflow: `~5709.5 ms`
- best observed JAX + persistent Julia fullflow: `~1100.9 ms`
- later reproduction on the same branch: original `~5878.5 ms`, JAX `~1226.0 ms`

JAX profile breakdown on the reproduced repeat-2 run:

- `init_ms ≈ 45.0 ms`
- `SubP1 total ≈ 158.3 ms`
- `SubP2 total ≈ 1021.5 ms`
- `SubP3 total ≈ 1.3 ms`

Julia worker side on the reproduced run:

- `SubP2` wall-clock (`batch_total_ms` summed across 3 ADMM rounds): `~878.6 ms`
- `bridge_prepare_ms_total ≈ 7.4 ms`
- `bridge_payload_ms_total ≈ 1.7 ms`
- `bridge_request_ms_total ≈ 916.0 ms`
- `bridge_rebuild_ms_total ≈ 4.4 ms`

Accuracy remained stable:

- `load_state_xl rmse = 2.032169e-04`
- `load_control_ul rmse = 1.542692e-04`
- `cable_state_xc rmse = 4.944731e-04`

Interpretation:

- the old `~40 s` Julia `SubP2` figure was mostly an integration artifact
- the old `~6.6 s` persistent-worker result was still leaving too much value on the table
- once slot-based parameterized reuse is active, the Julia branch becomes a true
  high-performance path rather than just a modeling-validation branch
- at this point, `SubP2` is no longer the dominant problem in the fullflow timing
- the remaining gaps are now in:
  - residual Python/JAX packaging around the Julia worker
  - and the fixed non-`SubP2` planner cost (`init + SubP1`)

Most important current conclusion:

- pure-JAX `SubP2` is still not the fastest route
- but **JAX planner + persistent Julia `SubP2` with slot-based parameterized reuse**
  has now beaten the original fullflow wall time on the formal-scale CPU benchmark
