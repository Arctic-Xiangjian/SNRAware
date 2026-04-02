# Polished Task: End-to-End LoRA Pipeline Verification

## 1. Goal
Create a standalone script `test_e2e_lora_pipeline.py` that validates LoRA integration without Lightning, and compares it against a full-parameter baseline on both 3D and 2D dummy MRI data.

This script must be runnable as:

```bash
PYTHONPATH=src python3 test_e2e_lora_pipeline.py --device cuda:1
```

---

## 2. Required Coverage

### 2.1 Inputs and Models
1. Load config from `checkpoints/snraware_large_model.yaml`.
2. Load weights from `checkpoints/snraware_large_model.pts`.
3. Gracefully fallback if either file is missing or invalid:
   - Use a hardcoded mock config.
   - Continue with random initialization if weights cannot be used.

### 2.2 Legacy Compatibility (Necessary Add-on)
Checkpoint/config may come from legacy namespace (`ifm.*`). Add target remapping:
- `ifm.model.config.* -> snraware.components.model.config.*`
- `ifm.mri.denoising.data.* -> snraware.projects.mri.denoising.data.*`

### 2.3 Dummy Data
Use random data with 10 samples for each modality:
- 3D: input `[B, 3, 16, 64, 64]`, target `[B, 2, 16, 64, 64]`
- 2D: input `[B, 3, 1, 64, 64]`, target `[B, 2, 1, 64, 64]`

### 2.4 Pipeline Stages Per Modality
For each modality, run two stages:

1. **Baseline Stage**
   - Initialize model.
   - Load pretrained weights if possible.
   - Unfreeze all parameters.
   - Run one train step: forward -> MSE -> backward -> optimizer step.
   - Run one eval pass (`torch.no_grad()`).

2. **LoRA Stage**
   - Re-initialize model.
   - Load pretrained weights if possible.
   - Apply `apply_lora_to_model(...)`.
   - Assert trainable state strictly:
     - Trainable: `*.lora_*`, `pre.*`, `post.*`
     - Frozen: everything else
   - Optimizer must use only trainable params.
   - Run one train step and one eval pass.

---

## 3. Metrics to Print
Per stage, print:
1. `total_params`, `trainable_params`, `trainable_pct`
2. Step latency (forward+backward+step) in ms
3. `initial_loss`, `updated_loss` (same batch after update), `eval_loss`
4. Peak VRAM in MB (if CUDA), else `0.0`

Also print cross-stage summary:
- trainable parameter reduction from baseline to LoRA
- baseline vs LoRA step time

---

## 4. Robustness Requirements
1. Device safety:
   - Accept `--device`.
   - If requested CUDA device is unavailable, fallback safely.
2. Weight loading safety:
   - Try `torch.jit.load(...).state_dict()` first.
   - Fallback to `torch.load(...)`.
   - Support key prefix cleanup (`module.`, `model.`, etc.).
   - Load only shape-compatible keys.
3. CUDA memory accounting:
   - `torch.cuda.reset_peak_memory_stats(device)` before stage.
   - `torch.cuda.max_memory_allocated(device)` after stage.
4. Determinism:
   - Set seed for reproducibility.

---

## 5. Acceptance Criteria
The task is complete when:
1. `test_e2e_lora_pipeline.py` runs end-to-end.
2. Both 3D and 2D modes execute baseline + LoRA stages.
3. LoRA strict trainable/frozen assertions pass.
4. Metrics are printed for every stage and final summary is shown.
5. Missing config/weights do not crash the script.

---

## 6. Optional Enhancements
1. Add `--num_samples`, `--batch_size`, `--lr`, `--lora_r`, `--lora_alpha`, `--lora_dropout` flags.
2. Add OOM fallback policy (auto-reduce batch size to 1 if needed).
3. Export final report as JSON for experiment tracking.
