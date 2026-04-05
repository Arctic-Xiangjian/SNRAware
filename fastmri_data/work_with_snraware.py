"""Bridge dataset for FastMRI-to-SNRAware fine-tuning."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset

from fastmri_data.fastmri_data import SliceDataset

__all__ = [
    "FastMRISNRAwareDataset",
    "legacy_uniform1d_mask",
]


def _seed_from_name(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def legacy_uniform1d_mask(
    img: torch.Tensor,
    size: int,
    batch_size: int,
    *,
    acc_factor: int = 4,
    center_fraction: float = 0.08,
    fix: bool = False,
    name: str | None = None,
) -> torch.Tensor:
    """Reproduce the historical uniform-1D mask math from the legacy pipeline."""
    mask = torch.zeros_like(img)

    nsamp_target = int(round(size / acc_factor))
    nsamp_center = int(round(size * center_fraction))
    if nsamp_center > nsamp_target:
        raise ValueError("center_fraction exceeds the achievable acceleration target")

    denom = size - nsamp_center
    prob = (nsamp_target - nsamp_center) / denom if denom > 0 else 0.0

    center_from = size // 2 - nsamp_center // 2
    center_to = center_from + nsamp_center

    rng = np.random.RandomState(_seed_from_name(name)) if isinstance(name, str) else None
    if fix or isinstance(name, str):
        selection = (rng.rand(size) if rng is not None else np.random.rand(size)) < prob
        if nsamp_center > 0:
            selection[center_from:center_to] = True
        idx = np.nonzero(selection)[0]
        mask[:, :, :, idx] = 1
        return mask

    for batch_idx in range(batch_size):
        selection = np.random.rand(size) < prob
        if nsamp_center > 0:
            selection[center_from:center_to] = True
        idx = np.nonzero(selection)[0]
        mask[batch_idx, :, :, idx] = 1

    return mask


def _complex_center_crop(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    height, width = shape
    if x.shape[-3] < height or x.shape[-2] < width:
        raise ValueError(f"Cannot crop shape {tuple(x.shape)} to {shape}")
    from_h = (x.shape[-3] - height) // 2
    from_w = (x.shape[-2] - width) // 2
    return x[..., from_h : from_h + height, from_w : from_w + width, :]


def _center_crop_real(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    height, width = shape
    if x.shape[-2] < height or x.shape[-1] < width:
        raise ValueError(f"Cannot crop shape {tuple(x.shape)} to {shape}")
    from_h = (x.shape[-2] - height) // 2
    from_w = (x.shape[-1] - width) // 2
    return x[..., from_h : from_h + height, from_w : from_w + width]


def _fft2c(x: torch.Tensor) -> torch.Tensor:
    complex_x = torch.view_as_complex(x.contiguous())
    complex_x = torch.fft.ifftshift(complex_x, dim=(-2, -1))
    complex_x = torch.fft.fft2(complex_x, dim=(-2, -1), norm="ortho")
    complex_x = torch.fft.fftshift(complex_x, dim=(-2, -1))
    return torch.view_as_real(complex_x)


def _ifft2c(x: torch.Tensor) -> torch.Tensor:
    complex_x = torch.view_as_complex(x.contiguous())
    complex_x = torch.fft.ifftshift(complex_x, dim=(-2, -1))
    complex_x = torch.fft.ifft2(complex_x, dim=(-2, -1), norm="ortho")
    complex_x = torch.fft.fftshift(complex_x, dim=(-2, -1))
    return torch.view_as_real(complex_x)


def _complex_abs(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.view_as_complex(x.contiguous()))


class FastMRISNRAwareDataset(Dataset):
    """A 2D FastMRI bridge dataset that returns pure 2D FastMRI fine-tune samples."""

    def __init__(
        self,
        root: str | Path | list[str] | list[Path],
        *,
        split: Literal["train", "val", "test"] = "train",
        challenge: str = "singlecoil",
        sample_rate: float | None = None,
        volume_sample_rate: float | None = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: str | Path = "dataset_cache.pkl",
        scanner_models: list[str] | None = None,
        acc_factor: int = 4,
        crop_size: tuple[int, int] = (320, 320),
        train_patch_size: tuple[int, int] | None = None,
        strict_latent_feature: bool = False,
        deterministic_mask_from_name: bool = False,
        sample_seed: int | None = None,
    ):
        super().__init__()
        if challenge != "singlecoil":
            raise NotImplementedError("FastMRI fine-tuning bridge currently supports singlecoil only")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of 'train', 'val', or 'test', got {split!r}")

        self.split = split
        self.acc_factor = int(acc_factor)
        self.crop_size = tuple(int(dim) for dim in crop_size)
        self.train_patch_size = (
            None if train_patch_size is None else tuple(int(dim) for dim in train_patch_size)
        )
        self.deterministic_mask_from_name = deterministic_mask_from_name
        if self.train_patch_size is not None:
            if any(dim <= 0 for dim in self.train_patch_size):
                raise ValueError(f"train_patch_size must contain positive integers, got {self.train_patch_size}")
            if (
                self.train_patch_size[0] > self.crop_size[0]
                or self.train_patch_size[1] > self.crop_size[1]
            ):
                raise ValueError(
                    f"train_patch_size {self.train_patch_size} must fit inside crop_size {self.crop_size}"
                )

        self.slice_dataset = SliceDataset(
            root=root,
            challenge=challenge,
            transform=None,
            use_dataset_cache=use_dataset_cache,
            sample_rate=sample_rate,
            volume_sample_rate=volume_sample_rate,
            dataset_cache_file=dataset_cache_file,
            scanner_models=scanner_models,
            num_adj_slices=1,
            strict_latent_feature=strict_latent_feature,
            sample_seed=sample_seed,
        )

    def __len__(self) -> int:
        return len(self.slice_dataset)

    def _maybe_random_crop_train_patch(
        self,
        under_recon_slice: torch.Tensor,
        clean_mag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.split != "train" or self.train_patch_size is None:
            return under_recon_slice, clean_mag

        patch_h, patch_w = self.train_patch_size
        full_h, full_w = clean_mag.shape[-2:]
        max_top = full_h - patch_h
        max_left = full_w - patch_w
        if max_top < 0 or max_left < 0:
            raise ValueError(
                f"train_patch_size {self.train_patch_size} must fit inside cropped image {(full_h, full_w)}"
            )

        top = int(torch.randint(0, max_top + 1, ()).item())
        left = int(torch.randint(0, max_left + 1, ()).item())
        under_recon_slice = under_recon_slice[:, top : top + patch_h, left : left + patch_w, :]
        clean_mag = clean_mag[:, top : top + patch_h, left : left + patch_w]
        return under_recon_slice, clean_mag

    def _build_sample(
        self,
        *,
        raw_kspace: np.ndarray,
        volume_name: str,
        slice_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        orig_kspace_slice = torch.zeros((1, raw_kspace.shape[0], raw_kspace.shape[1], 2), dtype=torch.float32)
        orig_kspace_slice[0, :, :, 0] = torch.from_numpy(raw_kspace.real.astype(np.float32))
        orig_kspace_slice[0, :, :, 1] = torch.from_numpy(raw_kspace.imag.astype(np.float32))
        original_kspace_shape = tuple(int(dim) for dim in orig_kspace_slice.shape[1:3])

        clean_recon_slice = _ifft2c(orig_kspace_slice)

        center_fraction = 0.08 if self.acc_factor == 4 else 0.04
        mask_name = volume_name if self.deterministic_mask_from_name else None
        mask = legacy_uniform1d_mask(
            img=orig_kspace_slice.permute(0, 3, 1, 2),
            size=orig_kspace_slice.shape[2],
            batch_size=1,
            acc_factor=self.acc_factor,
            center_fraction=center_fraction,
            fix=False,
            name=mask_name,
        )
        mask = mask.permute(0, 2, 3, 1)

        masked_kspace = orig_kspace_slice * mask
        under_recon_slice = _ifft2c(masked_kspace)
        under_recon_slice = _complex_center_crop(under_recon_slice, self.crop_size)
        clean_recon_slice = _complex_center_crop(clean_recon_slice, self.crop_size)

        under_recon_abs = _complex_abs(under_recon_slice).squeeze(0).unsqueeze(0)
        clean_mag = _complex_abs(clean_recon_slice).squeeze(0).unsqueeze(0)

        lq_running_mean = torch.tensor(0.0, dtype=torch.float32)
        lq_running_std = under_recon_abs.mean().float()
        if torch.equal(lq_running_std, torch.zeros_like(lq_running_std)):
            raise ValueError(f"Encountered zero normalization scale for volume {volume_name}, slice {slice_idx}")

        under_recon_slice = under_recon_slice / lq_running_std
        clean_mag = clean_mag / lq_running_std
        under_recon_slice, clean_mag = self._maybe_random_crop_train_patch(under_recon_slice, clean_mag)

        noisy = rearrange(under_recon_slice.squeeze(0), "h w c -> c h w").contiguous()
        clean = clean_mag.contiguous()

        cropped_output_shape = tuple(int(dim) for dim in clean.shape[-2:])
        if cropped_output_shape != original_kspace_shape:
            # Full-resolution mask and k-space are removed from metadata to prevent shape mismatch with the cropped 320x320 image outputs.
            metadata_mask = None
            metadata_masked_kspace = None
        else:
            metadata_mask = mask.squeeze(0).contiguous()
            metadata_masked_kspace = (masked_kspace.squeeze(0) / lq_running_std).contiguous()

        metadata = {
            "name": f"{Path(volume_name).stem}_slice_{slice_idx}",
            "volume_name": Path(volume_name).stem,
            "slice_idx": int(slice_idx),
            "mean": lq_running_mean,
            "std": lq_running_std,
            "mask": metadata_mask,
            "masked_kspace": metadata_masked_kspace,
        }
        noise_sigma = torch.tensor(0.0, dtype=torch.float32)
        return noisy.float(), clean.float(), noise_sigma, metadata

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        raw_kspace, _mask, target, _attrs, file_name, slice_idx = self.slice_dataset[idx]
        if target is None:
            raise ValueError(
                "FastMRI fine-tuning requires ground-truth reconstruction targets, "
                f"but none were found for {file_name}, slice {slice_idx}."
            )
        if np.asarray(raw_kspace).ndim != 2:
            raise ValueError(
                "FastMRI fine-tuning bridge expects single-slice 2D k-space arrays, "
                f"got shape {np.asarray(raw_kspace).shape} for {file_name}, slice {slice_idx}."
            )
        return self._build_sample(
            raw_kspace=np.asarray(raw_kspace),
            volume_name=str(file_name),
            slice_idx=int(slice_idx),
        )
