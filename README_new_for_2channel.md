# 2-Channel Fine-Tuning Guide For SNRAware

This note explains how to use the current FastMRI-style fine-tuning path when your input only has:

1. `real`
2. `imag`

and does not have a true `g-factor` channel.

The goal is to make clear:

1. why the extra `UNet` adapter exists,
2. what LoRA is doing in this setup,
3. what tensor contract the current trainer expects,
4. how to prepare a custom dataset if you already have 2-channel NIfTI files.

---

## 1. What Problem This Path Solves

The original SNRAware denoising model was designed around a 3-channel MRI input:

1. `real`
2. `imag`
3. `g-factor`

That third channel tells the model where noise amplification is stronger.

In many practical cases, especially with public datasets or already reconstructed image-domain data, we only have:

1. `real`
2. `imag`

and no measured `g-factor` map.

The current FastMRI fine-tuning path solves this by wrapping the pretrained SNRAware base model with an extra adapter module:

- `SNRAwareWithGFactor`
- a small `NormUnet` named `gfactor_unet`

So the runtime flow becomes:

1. input 2-channel complex image
2. `gfactor_unet` predicts a pseudo `g-factor`
3. concatenate `[real, imag, predicted_gfactor]`
4. send 3-channel tensor into the pretrained SNRAware base model

This lets us reuse the original 3-channel denoiser even when the dataset only provides 2 channels.

---

## 2. Role Of The UNet Adapter

The `UNet` adapter is not the denoiser itself. Its job is narrower:

1. receive the 2-channel image-domain input
2. estimate a nonnegative `g-factor`-like guidance map
3. make the 2-channel data compatible with the 3-channel base model

In the current implementation:

- the adapter is `NormUnet`
- it consumes `[B, 2, H, W]`
- it predicts `[B, 1, H, W]`
- the output is forced nonnegative with `abs`

Conceptually, this is a compatibility adapter, not a replacement for the SNRAware backbone.

Why this matters:

- if you only have 2 channels, the base model alone cannot be used directly as designed
- the adapter gives the base model the third channel it expects
- during fine-tuning, this adapter can learn dataset-specific noise geometry cues even without a true physics g-map

In warmup-based training, this adapter is often trained first so it can stabilize the pseudo-gfactor estimation before LoRA starts adapting the backbone behavior.

---

## 3. Role Of LoRA In This Setup

LoRA and the UNet adapter solve different problems.

### 3.1 UNet Adapter

The UNet adapter solves:

- "How do I turn 2-channel input into something the 3-channel SNRAware model can consume?"

### 3.2 LoRA

LoRA solves:

- "How do I adapt the pretrained SNRAware denoiser to a new dataset without fully retraining the backbone?"

In this repo, LoRA is injected into selected attention and MLP layers of the base denoiser. The important effect is:

1. backbone stays mostly frozen
2. only small low-rank adapter weights are updated
3. fine-tuning becomes cheaper and more stable than full-model retraining

So in the 2-channel pipeline:

- `gfactor_unet` handles input compatibility
- `LoRA` handles parameter-efficient domain adaptation of the pretrained denoiser

That is why both pieces can be valuable at the same time.

### 3.3 Practical Interpretation

If the new dataset is very different from the pretrained source:

- the `gfactor_unet` learns how to synthesize a useful third guidance channel
- LoRA learns how the denoiser should respond to this new data distribution

If you disable LoRA:

- you are mostly trusting the old pretrained denoiser and only learning the compatibility adapter

If you disable the UNet adapter:

- then this specific 2-channel path no longer works, because the base model still expects 3 channels

---

## 4. Current Data Contract For The FastMRI-Style 2-Channel Trainer

The current custom trainer in `src/snraware/projects/mri/denoising/trainer_fa.py` expects each sample to return a 4-tuple:

1. `noisy`
2. `clean`
3. `noise_sigma`
4. `metadata`

### 4.1 Expected Shapes

For the current 2-channel path:

- `noisy`: `[2, H, W]`
- `clean`: `[1, H, W]`
- `noise_sigma`: scalar tensor or zero tensor
- `metadata`: dictionary

### 4.2 Meaning Of Each Field

#### `noisy`

This is the 2-channel image-domain complex input:

- channel `0`: real
- channel `1`: imag

It should already be normalized before return.

#### `clean`

This is the target used by the trainer for loss and metrics.

Important:

- the current trainer expects **magnitude target**
- not 2-channel complex target

So `clean` must be:

- `[1, H, W]`
- magnitude domain

The trainer itself converts model output from 2-channel complex to magnitude before loss.

#### `noise_sigma`

This can be a placeholder zero tensor if you do not have a meaningful sigma value.

#### `metadata`

At minimum, the current evaluation path expects:

- `name`
- `volume_name`
- `slice_idx`
- `mean`
- `std`

Optional keys:

- `mask`
- `masked_kspace`

For non-k-space image-domain datasets, `mask` and `masked_kspace` can safely be `None`.

### 4.3 Normalization Convention

The current FastMRI path uses:

- `mean = 0`
- divide-only normalization by `std`

At evaluation time, the trainer reconstructs metric inputs by:

1. converting prediction to magnitude
2. multiplying by `std`
3. adding `mean`

So if you want to stay fully compatible with the current trainer, return:

- `metadata["mean"] = 0`
- `metadata["std"] = normalization_scale`

---

## 5. If You Already Have 2-Channel NIfTI Files

This is the most relevant case when your data is already prepared and you do not need the FastMRI k-space simulation path.

### 5.1 When No Major Trainer Change Is Needed

If your NIfTI data can provide:

1. input complex image as `real/imag`
2. target magnitude image
3. per-slice `mean/std`
4. volume name and slice index

then the current `FastMRIFineTuneTrainer` can already work with it.

In that case, the main work is writing a dataset class that returns the same 4-item contract.

### 5.2 When A Bigger Change Would Be Needed

If your NIfTI target is still 2-channel complex, not magnitude, then the current trainer is not plug-and-play.

The current trainer assumes:

- model output is converted to magnitude
- target is already magnitude

So if your ground truth is 2-channel complex and you want to keep complex-domain supervision, that would require trainer changes.

Suggested modification idea in that case:

1. add a config switch for target domain
2. support either:
   - magnitude target `[1, H, W]`
   - complex target `[2, H, W]`
3. branch the loss logic in `trainer_fa.py`

That is not required if you are willing to convert the clean target to magnitude inside the dataset.

### 5.3 Another Practical Limitation

Right now `build_fastmri_dataloaders(...)` is hard-coded to instantiate `FastMRISNRAwareDataset`.

So if you want to train directly from NIfTI files without editing code elsewhere, the current codebase does **not** yet expose a Hydra-configurable custom 2-channel dataset class through the FastMRI entrypoint.

Minimal future modification idea:

1. replace the hard-coded dataset construction with `hydra.utils.instantiate(...)`
2. allow a custom dataset target under `fastmri_finetune`
3. keep the same returned sample contract

That would let a NIfTI dataset plug in cleanly without touching the trainer core.

---

## 6. Simple Dataset Template For 2-Channel NIfTI

Below is a minimal example of what a compatible dataset class should look like.

This example assumes:

1. one NIfTI stores or yields the 2-channel noisy complex image
2. one NIfTI stores or yields the clean magnitude image
3. data is organized slice-by-slice

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class TwoChannelNiiDataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        *,
        split: str = "train",
    ):
        self.samples = samples
        self.split = split

    def __len__(self) -> int:
        return len(self.samples)

    def _load_array(self, path: str | Path) -> np.ndarray:
        return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        noisy = self._load_array(sample["noisy_path"])
        clean = self._load_array(sample["clean_path"])

        # Example expected layouts:
        # noisy: [H, W, 2] or [2, H, W]
        # clean: [H, W] or [1, H, W]
        if noisy.ndim == 3 and noisy.shape[-1] == 2:
            noisy = np.transpose(noisy, (2, 0, 1))
        if clean.ndim == 2:
            clean = clean[None, ...]

        noisy = torch.as_tensor(noisy, dtype=torch.float32)
        clean = torch.as_tensor(clean, dtype=torch.float32)

        # Current trainer expects magnitude target [1, H, W].
        if clean.shape[0] != 1:
            raise ValueError(
                f"Expected clean magnitude image with shape [1, H, W], got {tuple(clean.shape)}"
            )

        # Divide-only normalization to match current trainer contract.
        noisy_mag = torch.sqrt(noisy[0:1].square() + noisy[1:2].square())
        std = noisy_mag.mean().float()
        if torch.equal(std, torch.zeros_like(std)):
            raise ValueError(f"Zero normalization scale in sample {sample['name']}")

        noisy = noisy / std
        clean = clean / std

        metadata = {
            "name": sample["name"],
            "volume_name": sample["volume_name"],
            "slice_idx": int(sample["slice_idx"]),
            "mean": torch.tensor(0.0, dtype=torch.float32),
            "std": std,
            "mask": None,
            "masked_kspace": None,
        }

        noise_sigma = torch.tensor(0.0, dtype=torch.float32)
        return noisy, clean, noise_sigma, metadata
```

---

## 7. What You Must Return If The Clean Label Is Still Complex

If your clean NIfTI is still stored as:

- `[H, W, 2]`
- or `[2, H, W]`

then convert it to magnitude before returning it, for example:

```python
clean_mag = torch.sqrt(clean_real.square() + clean_imag.square())
clean_mag = clean_mag.unsqueeze(0)
```

Do this in the dataset if you want to remain compatible with the current trainer without further code changes.

---

## 8. Recommended Metadata For Slice-Based Volume Metrics

To keep validation/test metrics working correctly, every returned sample should carry:

1. `volume_name`
2. `slice_idx`
3. `mean`
4. `std`

The trainer uses these to:

1. un-normalize prediction and target
2. regroup slices back into volumes
3. compute PSNR / SSIM / NMSE on volume stacks

If `volume_name` and `slice_idx` are wrong or missing, metric reconstruction will be wrong even if training still runs.

---

## 9. Practical Recommendation

For a new 2-channel dataset with no true g-factor:

1. keep the current `SNRAwareWithGFactor` wrapper
2. train the `gfactor_unet` adapter
3. use LoRA to adapt the base denoiser
4. return magnitude clean targets
5. keep `mean=0`, `std=scale`
6. use `mask=None` and `masked_kspace=None` for pure image-domain NIfTI data

If you want the most checkpoint-native training setup currently available in this repo:

- use `train_patch_size=64`

That matches the current pretrained spatial size much better than larger patch settings.

---

## 10. Short Summary

In this 2-channel path:

- the `UNet` adapter makes 2-channel input compatible with the 3-channel SNRAware base model
- LoRA lets the denoiser adapt efficiently without full backbone retraining
- the current trainer expects:
  - `noisy`: `[2, H, W]`
  - `clean`: `[1, H, W]` magnitude
  - `noise_sigma`
  - `metadata`
- if your data is already 2-channel NIfTI, the main missing piece is usually the dataset class
- if your labels are still complex, either convert them to magnitude in the dataset or plan a trainer change

