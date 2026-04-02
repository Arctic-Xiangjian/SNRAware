# LoRA Fine-Tuning Guide for SNRAware

This guide explains how to:

1. start from a pretrained denoising model,
2. enable LoRA fine-tuning in this repo,
3. save/load LoRA adapters correctly,
4. prepare a custom dataset class that works with the current training pipeline.

---

## 1. What Is Implemented

LoRA integration is already wired into the MRI denoising project:

- LoRA injection utilities: `src/snraware/projects/mri/denoising/lora_utils.py`
- Training hook (auto-apply LoRA when enabled): `src/snraware/projects/mri/denoising/run.py`
- Adapter-only save logic: `src/snraware/projects/mri/denoising/lightning_denoising.py`
- LoRA-aware load logic: `src/snraware/projects/mri/denoising/inference_model.py`
- Config keys: `src/snraware/projects/mri/denoising/configs/config.yaml` under `lora:`

When `lora.enabled=true`, the model is prepared as:

- backbone frozen,
- only LoRA adapter params trainable,
- plus `pre`/`post` conv layers trainable.

Also, `.pth` save output becomes an adapter checkpoint (LoRA + `pre/post`) instead of full-model weights.

---

## 2. Environment Setup

From repo root:

```bash
PYTHONPATH=src python3 -m py_compile src/snraware/projects/mri/denoising/lora_utils.py
```

For full training/inference, ensure your runtime has required dependencies (`torch`, `lightning`, `hydra-core`, `omegaconf`, etc.).

---

## 3. Enable LoRA in Config

Edit `src/snraware/projects/mri/denoising/configs/config.yaml`:

```yaml
lora:
  enabled: true
  r: 8
  lora_alpha: 16.0
  lora_dropout: 0.0
  target_modules:
    - "\\.attn\\.key$"
    - "\\.attn\\.query$"
    - "\\.attn\\.value$"
    - "\\.attn\\.output_proj$"
    - "\\.mlp\\.0$"
    - "\\.mlp\\.2$"
```

You can override from CLI too:

```bash
PYTHONPATH=src python3 -m snraware.projects.mri.denoising.run \
  lora.enabled=true lora.r=16 lora.lora_alpha=32.0 lora.lora_dropout=0.05
```

---

## 4. Start From Pretrained Weights

Important: LoRA fine-tuning is normally most useful when starting from pretrained base weights.

Current `run.py` creates a fresh model, then applies LoRA if enabled.  
If you want true finetuning (recommended), load base weights before training starts.

### 4.1 Minimal Pattern

Use this pattern before calling `trainer.fit(...)`:

```python
status = torch.load(pretrained_path, map_location="cpu")
if isinstance(status, dict) and "model_state_dict" in status:
    status = status["model_state_dict"]
model.load_state_dict(status, strict=False)
```

Then apply LoRA:

```python
model = apply_lora_to_model(model)
```

### 4.2 `.pts` (TorchScript) Note

If your pretrained file is `.pts`, extract state dict first:

```python
scripted = torch.jit.load(pretrained_pts_path, map_location="cpu")
state_dict = scripted.state_dict()
model.load_state_dict(state_dict, strict=False)
```

If keys are from older namespace/prefixes, you may need key remap/cleanup (see `test_e2e_lora_pipeline.py` for a robust example).

---

## 5. Train and Save

Run training as usual:

```bash
PYTHONPATH=src python3 -m snraware.projects.mri.denoising.run \
  lora.enabled=true \
  train_data_dir=/path/to/train \
  test_data_dir=/path/to/test
```

Outputs from `after_training(...)`:

- `model_*.pts`: traced model artifact
- `model_*.pth`:
  - full `state_dict` when `lora.enabled=false`
  - LoRA adapter checkpoint when `lora.enabled=true` (checkpoint type: `snraware_lora_adapter_v1`)
- `config_*.yaml`: resolved config used for run

For LoRA workflows, keep:

1. base pretrained model weights,
2. adapter `.pth`,
3. matching config yaml.

---

## 6. Load for Inference

Use existing loader APIs:

- `load_model(...)` in `inference_model.py` can detect LoRA adapter checkpoints.
- `load_lit_model(...)` also supports LoRA if config has `lora.enabled=true`.

Example:

```python
from snraware.projects.mri.denoising.inference_model import load_model
model, cfg = load_model(saved_model_path="model_xxx.pth", saved_config_path="config_xxx.yaml")
```

---

## 7. Dataset Contract (Critical)

To work with `LitDenoising._decompose_batch`, your training dataset must return **4 items**:

1. `noisy`
2. `clean`
3. `noise_sigma`
4. an extra placeholder tensor (currently unused, but required by unpacking)

### 7.1 Tensor Shapes

Expected channel semantics:

- `noisy`: channels = 3 (`real`, `imag`, `gmap`)
- `clean`: channels = 2 (`real`, `imag`)

Accepted shape modes:

1. With repetition:
   - `noisy`: `[REP, 3, T, H, W]`
   - `clean`: `[REP, 2, T, H, W]`
   - DataLoader gives `[B, REP, C, T, H, W]`
   - `_decompose_batch` flattens to `[B*REP, C, T, H, W]`

2. Without repetition:
   - `noisy`: `[3, T, H, W]`
   - `clean`: `[2, T, H, W]`
   - DataLoader gives `[B, C, T, H, W]`

`noise_sigma` can be scalar-like per sample or per repetition; it is used for logging/validation scaling.

---

## 8. Custom Dataset Class Template

Your class must be Hydra-instantiable and accept `data_dir` (and ideally `rng`), because `DenoisingDataModule` calls:

```python
hydra.utils.instantiate(self.config.dataset, data_dir=..., rng=...)
```

### 8.1 Example

```python
import torch
from torch.utils.data import Dataset

class MyMRIDenoisingDataset(Dataset):
    def __init__(self, data_dir, cutout_shape=(64, 64, 16), repetition=1, rng=None, **kwargs):
        self.data_dir = data_dir
        self.cutout_shape = cutout_shape
        self.repetition = repetition
        self.rng = rng
        self.samples = self._index_samples(data_dir)

    def _index_samples(self, data_dir):
        # return list of sample descriptors
        return list(range(100))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        T = self.cutout_shape[2]
        H = self.cutout_shape[0]
        W = self.cutout_shape[1]

        noisy = torch.randn(self.repetition, 3, T, H, W, dtype=torch.float32)
        clean = torch.randn(self.repetition, 2, T, H, W, dtype=torch.float32)
        noise_sigma = torch.rand(self.repetition, dtype=torch.float32)
        noise_sigma_generated = noise_sigma.clone()  # placeholder 4th item

        return noisy, clean, noise_sigma, noise_sigma_generated
```

### 8.2 Hydra Config Example

In your run config:

```yaml
dataset:
  _target_: my_package.my_dataset.MyMRIDenoisingDataset
  cutout_shape: [64, 64, 16]
  repetition: 1
```

---

## 9. Data Preparation Checklist

Before training, verify:

1. `noisy` and `clean` are `float32`.
2. Time/depth axis is in position `T` (shape `[C, T, H, W]` or `[REP, C, T, H, W]`).
3. Channel order is correct:
   - noisy channel 0/1/2 = real/imag/gmap
   - clean channel 0/1 = real/imag
4. Dataset returns 4 values exactly.
5. `cutout_shape` in config matches your intended `H, W, T`.

---

## 10. Quick Sanity Validation

Use the standalone script:

```bash
PYTHONPATH=src python3 test_e2e_lora_pipeline.py --device cuda:0
```

It runs baseline vs LoRA passes on dummy 3D and 2D data and reports:

- parameter efficiency,
- step latency,
- loss update sanity,
- VRAM peak.

---

## 11. Common Pitfalls

1. **Forgetting pretrained init**: LoRA on random-init base is usually not meaningful.
2. **Wrong batch tuple length**: training expects 4 outputs from dataset.
3. **Wrong tensor order**: model expects `[B, C, T, H, W]`.
4. **Missing `gmap` noisy channel**: noisy input must have 3 channels.
5. **Mixing adapter-only `.pth` with unrelated base model**: adapter must match the base architecture/weights family.
6. **Test set mismatch**: current `run.py` uses `MRIDenoisingDatasetTest` for `trainer.test(...)`. If your `test_data_dir` is not in that format, either prepare compatible test data or adjust the test stage/data module logic.

---

## 12. Recommended Next Improvements

1. Add explicit CLI arg/config key in `run.py` for `pretrained_model_path` and load it before LoRA injection.
2. Add a small merge utility to export a fully merged model from base + LoRA adapter when needed for deployment.
3. Add a unit test that validates custom dataset output signatures (`tuple` length, dtype, shape, channel order).
