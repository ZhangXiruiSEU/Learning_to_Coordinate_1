# Julia SubP2 Contract

Goal:

- keep the current `SubP2` problem definition unchanged
- keep input/output ordering unchanged
- replace only the solver backend with Julia `ExaModels + MadNLP`

Current bridge files:

- Python exporter: [export_subp2_snapshot_for_julia.py](/home/mpc/xirui/meta-learn_useThis/archive/export_subp2_snapshot_for_julia.py)
- Julia loader/checker: [load_snapshot.jl](/home/mpc/xirui/meta-learn_useThis/julia_subp2/archive/load_snapshot.jl)

## Decision vector ordering

The Julia implementation must preserve the exact Python/JAX ordering:

`w = [xl, ul, (xi_1, ui_1), (xi_2, ui_2), ..., (xi_nq, ui_nq)]`

where:

- `xl`: load state, size `nxl`
- `ul`: load control, size `nul`
- each cable block: `xi_i` then `ui_i`, size `nxi + nui`

This must match Python `_ipoptax_unpack`.

## Export contents

The exporter writes:

- `meta`
  - dimensions, task/horizon indices
- `step`
  - selected single-step `x_init`
  - `params_t`
  - original IPOPT reference for that exact step
  - initial and reference feasibility metrics
- `para2_contract`
  - the full raw SubP2 contract as used by Python/JAX
- optional `batch`
  - full `w_init_batch` and `params_batch`

## First Julia implementation target

Before benchmarking speed:

1. rebuild objective and constraints from `step.params_t`
2. evaluate them at:
   - `x_init`
   - `original_step_reference`
3. verify Julia matches Python/JAX values for:
   - objective
   - equality residual vector
   - inequality residual vector

Only after that should solver benchmarking start.
