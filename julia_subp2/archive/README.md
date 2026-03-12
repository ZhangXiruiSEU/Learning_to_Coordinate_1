Archived Julia SubP2 assets

Active Julia mainline files kept in `julia_subp2/`:

- `Project.toml`
- `Manifest.toml`
- `setup_env.jl`
- `step_model.jl`
- `solve_step_madnlp_jump_native_eq.jl`
- `solve_batch_madnlp_jump_native.jl`
- `worker_batch_madnlp_jump_native.jl`

Archived here:

- old single-step solver variants
- snapshot loaders and validators
- native-family audit helpers
- GPU experiment worker wrapper
- static snapshot artifacts and contract notes

These files were moved for workspace cleanliness, not because they are invalid.
Where practical, archive-side relative paths were updated so the scripts remain
executable from their new location.
