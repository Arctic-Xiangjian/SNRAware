# Visual Check Scripts

This directory contains the visual inference and PnP experiment scripts that were originally staged in `/tmp/snraware_visual_check`.

## Recommended Entry Point

Use `run_pnp_crop_domain.py` for current PnP experiments. It follows the flow_unrolled-style crop-domain setup:

- image and measurement state are `320x320`
- hard DC is applied in 320 crop-domain k-space
- SNRAware is applied through 64x64 sliding-window patches and stitched back to 320

Current visual checks assume the same single-coil FastMRI-compatible sample shape used by the training bridge. For future multi-coil experiments, add an explicit coil-combination or coil-aware reconstruction step before these scripts compare crop-domain outputs; do not feed coil stacks into the existing 2-channel visual path without updating the model and metrics assumptions.

Example:

```bash
uv run python visual_code/run_pnp_crop_domain.py \
  --experiment strong \
  --volume-names file1000000 \
  --slice-limit 1 \
  --slices-per-volume 1 \
  --device cuda:0
```

## Historical Scripts

These scripts are kept for reproducibility of earlier visual checks:

- `run_visual_check.py`
- `run_pnp_visual_check.py`
- `run_pnp_guarded_sweep.py`
- `run_pnp_long_a0p2.py`

The older PnP scripts use full-FOV/raw-k-space style diagnostics and should not be used as the primary conclusion for current crop-domain PnP results.

## Outputs

Default outputs remain under `visual_check/...`. This directory is for code only; do not store generated PNG, CSV, JSON, or smoke outputs here.
