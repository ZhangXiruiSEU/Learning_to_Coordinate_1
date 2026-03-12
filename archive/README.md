Archived experiment scripts

This directory contains SubP2 experiment and diagnosis scripts that are no
longer part of the stable mainline.

Stable mainline to keep using:

- `JustWorkingOnIt.py`
- `verify_jax_vs_original.py`
- `python_archive/run_julia_subp2_fullflow_persistent.py`
- `julia_subp2/solve_batch_madnlp_jump_native.jl`
- `julia_subp2/worker_batch_madnlp_jump_native.jl`

Archived here on purpose:

- fixed-step `SubP2` diagnosis and export helpers
- structured SQP prototypes
- snapshot scan scripts
- one-off benchmark/export utilities

Related archived Python runners live in:

- `python_archive/archive/`

The goal is workspace cleanliness, not deletion. These files were moved rather
than removed so the experiment trail remains reproducible.
