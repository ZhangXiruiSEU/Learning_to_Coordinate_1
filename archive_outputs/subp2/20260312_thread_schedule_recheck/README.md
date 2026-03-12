2026-03-12 thread-schedule recheck archive

Purpose:

- preserve the logs and plots behind the latest `SubP2` timing recheck
- keep the workspace root cleaner without touching the current stable mainline

Stable mainline files that remain in place and should be treated as active:

- `JustWorkingOnIt.py`
- `verify_jax_vs_original.py`
- `python_archive/run_julia_subp2_fullflow_persistent.py`
- `julia_subp2/solve_batch_madnlp_jump_native.jl`
- `julia_subp2/worker_batch_madnlp_jump_native.jl`

Archived experiment assets:

- `persistent_julia_bench4.log`
- `subp2_recheck_default_after_static_default_20260312.log`
- `subp2_recheck_dynamic_20260312.log`
- `subp2_recheck_static_20260312.log`
- verification plots were moved to:
  - `archive_outputs/verify/20260312_static_baseline/verify_compare_load_task_0.png`
  - `archive_outputs/verify/20260312_static_baseline/verify_compare_tension_task_0.png`
- GPU experiment runner moved to:
  - `python_archive/archive/run_julia_subp2_fullflow_persistent_gpuexp.py`
  - `julia_subp2/archive/worker_batch_madnlp_jump_native_gpuexp.jl`

Safety note:

- `Planning_plots_meta_COM_Dyn/` and `Planning_plots_multiagent_meta_COM_Dyn/`
  were intentionally left in place. They are not just disposable plots; other
  scripts in this repository still read from those directories.
