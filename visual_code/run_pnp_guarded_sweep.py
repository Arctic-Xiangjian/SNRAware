#!/usr/bin/env python
"""Run hallucination-controlled PnP sweeps outside the SNRAware source tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm


DEFAULT_REPO_ROOT = Path("/working2/arctic/snrawre/SNRAware")
DEFAULT_CHECKPOINT = (
    DEFAULT_REPO_ROOT / "checkpoints/fine_tune/warmup_then_both_20260517-145527/last.pth"
)
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "visual_check" / "PnP_guarded"
DEFAULT_VOLUMES = [
    "file1000000",
    "file1000007",
    "file1000017",
    "file1000026",
    "file1000031",
    "file1000033",
    "file1000041",
    "file1000052",
    "file1000071",
    "file1000073",
]
ALPHAS = (0.1, 0.2, 0.3)
BASE_METHODS = ("zero_filled", "one_shot", "one_shot_dc")
PNP_METHODS = (
    "pnp_k1_a0p1",
    "pnp_k2_a0p1",
    "pnp_k1_a0p2",
    "pnp_k2_a0p2",
    "pnp_k1_a0p3",
    "pnp_k2_a0p3",
    "pnp_k2_a0p2_lowpass",
)
METHODS = BASE_METHODS + PNP_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run guarded PnP sweep with held-out k-space diagnostics."
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-volumes", type=int, default=10)
    parser.add_argument("--volume-names", nargs="*", default=None)
    parser.add_argument("--slices-per-volume", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-patch-batch-size", type=int, default=None)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--lowpass-fraction", type=float, default=0.5)
    parser.add_argument(
        "--skip-top-level",
        action="store_true",
        help="Only write per-volume outputs; useful for parallel workers sharing one output dir.",
    )
    parser.add_argument("--worker-summary", type=Path, default=None)
    return parser.parse_args()


def configure_imports(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    os.chdir(repo_root)
    for path in (repo_root, repo_root / "src"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_arg)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    from snraware.projects.mri.denoising.fastmri_compat import (
        build_fastmri_wrapped_model,
        is_fastmri_finetune_checkpoint,
        load_fastmri_finetune_checkpoint,
    )
    from snraware.projects.mri.denoising.lora_utils import apply_lora_to_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not is_fastmri_finetune_checkpoint(checkpoint):
        raise ValueError(f"Not a FastMRI fine-tune checkpoint: {checkpoint_path}")

    config = OmegaConf.create(checkpoint["config"])
    ft_cfg = config.fastmri_finetune
    crop_size = tuple(int(dim) for dim in ft_cfg.crop_size)
    train_patch_size = ft_cfg.get("train_patch_size")
    model_size = crop_size if train_patch_size in (None, "", "null") else tuple(
        int(dim) for dim in train_patch_size
    )
    model, _base_config, load_info = build_fastmri_wrapped_model(
        base_config_path=config.base_model.config_path,
        base_checkpoint_path=config.base_model.checkpoint_path,
        height=model_size[0],
        width=model_size[1],
        depth=1,
        lora_config=config.get("lora"),
        gfactor_unet_kwargs=OmegaConf.to_container(ft_cfg.gfactor_unet, resolve=True),
        use_unet=bool(ft_cfg.get("use_unet", True)),
    )
    missing, unexpected = load_fastmri_finetune_checkpoint(
        model=model,
        checkpoint=checkpoint,
        apply_lora_fn=apply_lora_to_model,
        lora_config=config.get("lora"),
    )
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint restore mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model, config, checkpoint, load_info


def build_validation_dataset(config):
    from fastmri_data.work_with_snraware import FastMRISNRAwareDataset

    ft_cfg = config.fastmri_finetune
    return FastMRISNRAwareDataset(
        root=ft_cfg.val_root,
        split="val",
        challenge=ft_cfg.challenge,
        sample_rate=None,
        volume_sample_rate=None,
        use_dataset_cache=ft_cfg.use_dataset_cache,
        dataset_cache_file=ft_cfg.dataset_cache_file,
        scanner_models=ft_cfg.scanner_models,
        acc_factor=int(ft_cfg.acc_factor),
        crop_size=tuple(int(dim) for dim in ft_cfg.crop_size),
        train_patch_size=None,
        strict_latent_feature=False,
        deterministic_mask_from_name=bool(ft_cfg.deterministic_mask_from_name),
        sample_seed=ft_cfg.sample_seed,
    )


def select_volume_indices(dataset, requested: list[str] | None, num_volumes: int):
    examples = dataset.slice_dataset.examples
    ordered_volumes: list[str] = []
    indices_by_volume: dict[str, list[int]] = {}
    for index, (path, _slice_idx, _metadata) in enumerate(examples):
        volume = Path(path).stem
        if volume not in indices_by_volume:
            ordered_volumes.append(volume)
            indices_by_volume[volume] = []
        indices_by_volume[volume].append(index)
    selected = [name.removesuffix(".h5") for name in requested] if requested else [
        name for name in DEFAULT_VOLUMES if name in indices_by_volume
    ][:num_volumes]
    missing = [name for name in selected if name not in indices_by_volume]
    if missing:
        raise ValueError(f"Requested volumes not found in validation dataset: {missing}")
    return selected, {name: indices_by_volume[name] for name in selected}


def sliding_window_positions(size: int, patch_size: int, overlap: int) -> list[int]:
    stride = patch_size - overlap
    if patch_size > size or stride <= 0:
        raise ValueError(f"Invalid sliding window size={size}, patch={patch_size}, overlap={overlap}")
    positions = list(range(0, size - patch_size + 1, stride))
    final = size - patch_size
    if not positions or positions[-1] != final:
        positions.append(final)
    return positions


def predict_gfactor_for_patch(model, patch_tensor: torch.Tensor) -> torch.Tensor:
    patch_5d = model._prepare_2ch_input(patch_tensor)
    scale = model._fastmri_current_mean_scale(patch_5d)
    return model.predict_gfactor(patch_5d / scale).float()


def run_sliding_window_slice(
    model,
    noisy: torch.Tensor,
    *,
    patch_size: tuple[int, int],
    overlap: tuple[int, int],
    patch_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, full_h, full_w = noisy.shape
    patch_h, patch_w = patch_size
    top_positions = sliding_window_positions(full_h, patch_h, overlap[0])
    left_positions = sliding_window_positions(full_w, patch_w, overlap[1])
    pred_sum = torch.zeros((2, full_h, full_w), device=device, dtype=torch.float32)
    gfactor_sum = torch.zeros((1, full_h, full_w), device=device, dtype=torch.float32)
    weight_sum = torch.zeros((1, full_h, full_w), device=device, dtype=torch.float32)
    patch_buffer: list[torch.Tensor] = []
    coord_buffer: list[tuple[int, int]] = []

    def flush() -> None:
        if not patch_buffer:
            return
        patch_tensor = torch.stack(patch_buffer, dim=0).to(device=device, dtype=torch.float32)
        patch_output = model(patch_tensor, checkpoint_base_model=False).squeeze(2).float()
        patch_gfactor = predict_gfactor_for_patch(model, patch_tensor)
        for idx, (top, left) in enumerate(coord_buffer):
            pred_sum[:, top : top + patch_h, left : left + patch_w] += patch_output[idx]
            gfactor_sum[:, top : top + patch_h, left : left + patch_w] += patch_gfactor[idx]
            weight_sum[:, top : top + patch_h, left : left + patch_w] += 1.0
        patch_buffer.clear()
        coord_buffer.clear()

    for top in top_positions:
        for left in left_positions:
            patch_buffer.append(noisy[:, top : top + patch_h, left : left + patch_w])
            coord_buffer.append((top, left))
            if len(patch_buffer) >= patch_batch_size:
                flush()
    flush()
    if not torch.all(weight_sum > 0):
        raise RuntimeError("Sliding-window inference left uncovered pixels")
    return pred_sum / weight_sum, gfactor_sum / weight_sum


def fft2c_complex(image: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c_complex(kspace: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[0].float(), x[1].float())


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([x.real.float(), x.imag.float()], dim=0)


def center_crop_complex(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    top = (x.shape[-2] - shape[0]) // 2
    left = (x.shape[-1] - shape[1]) // 2
    return x[..., top : top + shape[0], left : left + shape[1]]


def center_embed_complex(crop: torch.Tensor, full_shape: tuple[int, int]) -> torch.Tensor:
    full = torch.zeros(full_shape, device=crop.device, dtype=crop.dtype)
    top = (full_shape[0] - crop.shape[-2]) // 2
    left = (full_shape[1] - crop.shape[-1]) // 2
    full[top : top + crop.shape[-2], left : left + crop.shape[-1]] = crop
    return full


def seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def make_masks(
    *,
    raw_kspace: np.ndarray,
    file_name: str,
    slice_idx: int,
    acc_factor: int,
    heldout_fraction: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from fastmri_data.work_with_snraware import legacy_uniform1d_mask

    raw = np.asarray(raw_kspace)
    raw_complex = torch.from_numpy(raw.astype(np.complex64)).to(device=device)
    raw_complex_last = torch.zeros((1, raw.shape[0], raw.shape[1], 2), dtype=torch.float32)
    raw_complex_last[0, :, :, 0] = torch.from_numpy(raw.real.astype(np.float32))
    raw_complex_last[0, :, :, 1] = torch.from_numpy(raw.imag.astype(np.float32))
    center_fraction = 0.08 if int(acc_factor) == 4 else 0.04
    full_mask = legacy_uniform1d_mask(
        img=raw_complex_last.permute(0, 3, 1, 2),
        size=raw_complex_last.shape[2],
        batch_size=1,
        acc_factor=int(acc_factor),
        center_fraction=center_fraction,
        fix=False,
        name=file_name,
    )[0, 0].to(device=device, dtype=torch.bool)
    sampled_cols = torch.where(full_mask.any(dim=0))[0].detach().cpu().numpy()
    width = raw.shape[1]
    n_center = int(round(width * center_fraction))
    center_from = width // 2 - n_center // 2
    center_to = center_from + n_center
    non_center = [int(col) for col in sampled_cols if not (center_from <= int(col) < center_to)]
    rng = np.random.RandomState(seed_from_text(f"{file_name}:{slice_idx}:heldout"))
    rng.shuffle(non_center)
    n_hold = max(1, int(round(len(non_center) * float(heldout_fraction)))) if non_center else 0
    heldout_cols = set(non_center[:n_hold])
    val_mask = torch.zeros_like(full_mask)
    if heldout_cols:
        val_mask[:, sorted(heldout_cols)] = full_mask[:, sorted(heldout_cols)]
    dc_mask = full_mask & ~val_mask
    return raw_complex, dc_mask, val_mask


def apply_full_fov_dc(
    prediction_crop: torch.Tensor,
    *,
    observed_kspace: torch.Tensor,
    mask: torch.Tensor,
    crop_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_full = center_embed_complex(channels_to_complex(prediction_crop.to(observed_kspace.device)), tuple(observed_kspace.shape[-2:]))
    pred_kspace = fft2c_complex(pred_full)
    dc_kspace = torch.where(mask, observed_kspace, pred_kspace)
    dc_full = ifft2c_complex(dc_kspace)
    return complex_to_channels(center_crop_complex(dc_full, crop_size)), dc_full


def crop_to_full_complex(prediction_crop: torch.Tensor, full_shape: tuple[int, int]) -> torch.Tensor:
    return center_embed_complex(channels_to_complex(prediction_crop), full_shape)


def kspace_relative_residual(full_image: torch.Tensor, observed_kspace: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any().item()):
        return float("nan")
    pred_kspace = fft2c_complex(full_image.to(observed_kspace.device))
    diff = pred_kspace[mask] - observed_kspace[mask]
    denom = torch.linalg.vector_norm(observed_kspace[mask]).clamp_min(1e-12)
    return float((torch.linalg.vector_norm(diff) / denom).detach().cpu().item())


def lowpass_residual(residual: torch.Tensor, keep_fraction: float) -> torch.Tensor:
    residual_complex = channels_to_complex(residual)
    kspace = fft2c_complex(residual_complex)
    height, width = residual_complex.shape
    keep_h = max(1, int(round(height * keep_fraction)))
    keep_w = max(1, int(round(width * keep_fraction)))
    top = (height - keep_h) // 2
    left = (width - keep_w) // 2
    mask = torch.zeros((height, width), device=residual.device, dtype=torch.bool)
    mask[top : top + keep_h, left : left + keep_w] = True
    return complex_to_channels(ifft2c_complex(torch.where(mask, kspace, torch.zeros_like(kspace))))


def highpass_residual_magnitude(residual: torch.Tensor, keep_fraction: float) -> np.ndarray:
    high = residual - lowpass_residual(residual, keep_fraction)
    return magnitude_from_complex_channels(high)


def magnitude_from_complex_channels(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().float().numpy()
    return np.sqrt(np.square(x[0]) + np.square(x[1])).astype(np.float32)


def minmax_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = float(finite.min()), float(finite.max())
    return np.zeros_like(image, dtype=np.float32) if hi <= lo else (image - lo) / (hi - lo)


def choose_visual_indices(num_slices: int, requested: int) -> list[int]:
    return sorted(set(int(idx) for idx in np.linspace(0, num_slices - 1, max(1, min(requested, num_slices)))))


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def compute_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    from fastmri.evaluate import nmse, psnr, ssim

    return {
        "psnr": float(np.asarray(psnr(target, prediction)).mean()),
        "ssim": float(np.asarray(ssim(target, prediction)).mean()),
        "nmse": float(np.asarray(nmse(target, prediction)).mean()),
    }


def method_display_name(method: str) -> str:
    return {
        "zero_filled": "Zero-filled",
        "one_shot": "One-shot",
        "one_shot_dc": "One-shot+DC",
        "pnp_k1_a0p1": "K1 a0.1",
        "pnp_k2_a0p1": "K2 a0.1",
        "pnp_k1_a0p2": "K1 a0.2",
        "pnp_k2_a0p2": "K2 a0.2",
        "pnp_k1_a0p3": "K1 a0.3",
        "pnp_k2_a0p3": "K2 a0.3",
        "pnp_k2_a0p2_lowpass": "K2 a0.2 LP",
    }.get(method, method)


def select_best_methods(method_metrics: dict[str, dict[str, float]]) -> list[str]:
    candidates = list(PNP_METHODS)
    ranked = sorted(
        candidates,
        key=lambda name: (
            method_metrics[name]["heldout_residual"],
            -method_metrics[name]["psnr"],
        ),
    )
    return ranked[:3]


def save_comparison_png(
    volume_result: dict[str, Any],
    output_path: Path,
    visual_rows: list[int],
    best_methods: list[str],
) -> None:
    columns = ["one_shot", "one_shot_dc", *best_methods, "target"]
    fig, axes = plt.subplots(
        len(visual_rows),
        len(columns) * 2 - 1,
        figsize=(2.45 * (len(columns) * 2 - 1), 2.7 * len(visual_rows)),
        dpi=130,
        squeeze=False,
    )
    panel_cols: list[tuple[str, str, bool]] = []
    for method in columns:
        panel_cols.append((method_display_name(method), method, False))
        if method != "target":
            panel_cols.append((f"Err {method_display_name(method)}", method, True))
    for row_idx, slice_pos in enumerate(visual_rows):
        target = volume_result["target"][slice_pos]
        for col_idx, (title, method, is_error) in enumerate(panel_cols):
            ax = axes[row_idx][col_idx]
            image = np.abs(volume_result[method][slice_pos] - target) if is_error else volume_result[method][slice_pos]
            ax.imshow(minmax_image(image), cmap="magma" if is_error else "gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"slice {volume_result['slice_indices'][slice_pos]}", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_residual_maps_png(volume_result: dict[str, Any], output_path: Path, visual_rows: list[int]) -> None:
    panels = [
        ("Residual K1 a0.2", "residual_mag_a0p2"),
        ("High-pass residual", "residual_high_a0p2"),
        ("Low-pass residual", "residual_low_mag_a0p2"),
    ]
    fig, axes = plt.subplots(len(visual_rows), len(panels), figsize=(3.1 * len(panels), 2.8 * len(visual_rows)), dpi=130, squeeze=False)
    for row_idx, slice_pos in enumerate(visual_rows):
        for col_idx, (title, key) in enumerate(panels):
            ax = axes[row_idx][col_idx]
            ax.imshow(minmax_image(volume_result[key][slice_pos]), cmap="magma", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"slice {volume_result['slice_indices'][slice_pos]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_alpha_pnp(
    model,
    *,
    x0: torch.Tensor,
    observed_kspace: torch.Tensor,
    dc_mask: torch.Tensor,
    crop_size: tuple[int, int],
    patch_size: tuple[int, int],
    overlap: tuple[int, int],
    patch_batch_size: int,
    alpha: float,
    device: torch.device,
    lowpass: bool,
    lowpass_fraction: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    outputs: dict[str, torch.Tensor] = {}
    full_images: dict[str, torch.Tensor] = {}
    residuals: dict[str, torch.Tensor] = {}
    x = x0
    for iteration in (1, 2):
        z, _g = run_sliding_window_slice(
            model,
            x,
            patch_size=patch_size,
            overlap=overlap,
            patch_batch_size=patch_batch_size,
            device=device,
        )
        residual = z - x
        if lowpass:
            residual = lowpass_residual(residual, lowpass_fraction)
        damped = x + alpha * residual
        x, full = apply_full_fov_dc(damped, observed_kspace=observed_kspace, mask=dc_mask, crop_size=crop_size)
        key = f"k{iteration}"
        outputs[key] = x
        full_images[key] = full
        residuals[key] = residual
    return outputs, full_images, residuals


def run_volume(
    *,
    model,
    dataset,
    volume_name: str,
    dataset_indices: list[int],
    patch_size: tuple[int, int],
    overlap: tuple[int, int],
    patch_batch_size: int,
    crop_size: tuple[int, int],
    acc_factor: int,
    heldout_fraction: float,
    lowpass_fraction: float,
    device: torch.device,
    output_dir: Path,
    slices_per_volume: int,
) -> dict[str, Any]:
    volume_dir = output_dir / volume_name
    volume_dir.mkdir(parents=True, exist_ok=True)

    method_stacks: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    full_images_by_method: dict[str, list[torch.Tensor]] = {method: [] for method in METHODS}
    dc_residuals: dict[str, list[float]] = {method: [] for method in METHODS}
    heldout_residuals: dict[str, list[float]] = {method: [] for method in METHODS}
    residual_mag_a0p2: list[np.ndarray] = []
    residual_high_a0p2: list[np.ndarray] = []
    residual_low_mag_a0p2: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    slice_indices: list[int] = []

    progress = tqdm(dataset_indices, desc=volume_name, leave=False)
    with torch.inference_mode():
        for dataset_index in progress:
            noisy, clean, _noise_sigma, metadata = dataset[dataset_index]
            path, slice_idx, _example_metadata = dataset.slice_dataset.examples[dataset_index]
            with h5py.File(path, "r") as hf:
                raw_kspace = np.asarray(hf["kspace"][slice_idx])

            observed_kspace, dc_mask, val_mask = make_masks(
                raw_kspace=raw_kspace,
                file_name=Path(path).name,
                slice_idx=int(slice_idx),
                acc_factor=acc_factor,
                heldout_fraction=heldout_fraction,
                device=device,
            )

            _ = noisy
            zero_full = ifft2c_complex(torch.where(dc_mask, observed_kspace, torch.zeros_like(observed_kspace)))
            zero_crop = complex_to_channels(center_crop_complex(zero_full, crop_size)).to(dtype=torch.float32)
            one_shot, _g = run_sliding_window_slice(
                model,
                zero_crop,
                patch_size=patch_size,
                overlap=overlap,
                patch_batch_size=patch_batch_size,
                device=device,
            )
            one_shot_full = crop_to_full_complex(one_shot, tuple(observed_kspace.shape[-2:]))
            one_shot_dc, one_shot_dc_full = apply_full_fov_dc(one_shot, observed_kspace=observed_kspace, mask=dc_mask, crop_size=crop_size)

            method_tensors: dict[str, torch.Tensor] = {
                "zero_filled": zero_crop,
                "one_shot": one_shot,
                "one_shot_dc": one_shot_dc,
            }
            full_tensors: dict[str, torch.Tensor] = {
                "zero_filled": zero_full,
                "one_shot": one_shot_full,
                "one_shot_dc": one_shot_dc_full,
            }

            for alpha in ALPHAS:
                suffix = str(alpha).replace(".", "p")
                outputs, fulls, residuals = run_alpha_pnp(
                    model,
                    x0=one_shot_dc,
                    observed_kspace=observed_kspace,
                    dc_mask=dc_mask,
                    crop_size=crop_size,
                    patch_size=patch_size,
                    overlap=overlap,
                    patch_batch_size=patch_batch_size,
                    alpha=float(alpha),
                    device=device,
                    lowpass=False,
                    lowpass_fraction=lowpass_fraction,
                )
                method_tensors[f"pnp_k1_a{suffix}"] = outputs["k1"]
                method_tensors[f"pnp_k2_a{suffix}"] = outputs["k2"]
                full_tensors[f"pnp_k1_a{suffix}"] = fulls["k1"]
                full_tensors[f"pnp_k2_a{suffix}"] = fulls["k2"]
                if abs(alpha - 0.2) < 1e-6:
                    residual_mag_a0p2.append(magnitude_from_complex_channels(residuals["k1"]))
                    residual_high_a0p2.append(highpass_residual_magnitude(residuals["k1"], lowpass_fraction))
                    residual_low_mag_a0p2.append(magnitude_from_complex_channels(lowpass_residual(residuals["k1"], lowpass_fraction)))

            outputs_lp, fulls_lp, _residuals_lp = run_alpha_pnp(
                model,
                x0=one_shot_dc,
                observed_kspace=observed_kspace,
                dc_mask=dc_mask,
                crop_size=crop_size,
                patch_size=patch_size,
                overlap=overlap,
                patch_batch_size=patch_batch_size,
                alpha=0.2,
                device=device,
                lowpass=True,
                lowpass_fraction=lowpass_fraction,
            )
            method_tensors["pnp_k2_a0p2_lowpass"] = outputs_lp["k2"]
            full_tensors["pnp_k2_a0p2_lowpass"] = fulls_lp["k2"]

            for method in METHODS:
                method_stacks[method].append(magnitude_from_complex_channels(method_tensors[method]))
                full_images_by_method[method].append(full_tensors[method].detach().cpu())
                dc_residuals[method].append(kspace_relative_residual(full_tensors[method], observed_kspace, dc_mask))
                heldout_residuals[method].append(kspace_relative_residual(full_tensors[method], observed_kspace, val_mask))

            targets.append(clean.squeeze(0).detach().cpu().float().numpy().astype(np.float32))
            slice_indices.append(int(metadata["slice_idx"]))

    order = np.argsort(np.asarray(slice_indices))
    slice_indices = [slice_indices[int(idx)] for idx in order]
    method_arrays = {
        method: np.stack([values[int(idx)] for idx in order], axis=0).astype(np.float32)
        for method, values in method_stacks.items()
    }
    target_stack = np.stack([targets[int(idx)] for idx in order], axis=0).astype(np.float32)
    residual_arrays = {
        "residual_mag_a0p2": np.stack([residual_mag_a0p2[int(idx)] for idx in order], axis=0).astype(np.float32),
        "residual_high_a0p2": np.stack([residual_high_a0p2[int(idx)] for idx in order], axis=0).astype(np.float32),
        "residual_low_mag_a0p2": np.stack([residual_low_mag_a0p2[int(idx)] for idx in order], axis=0).astype(np.float32),
    }

    method_metrics: dict[str, dict[str, float]] = {}
    for method in METHODS:
        image_metrics = compute_metrics(target_stack, method_arrays[method])
        method_metrics[method] = {
            **image_metrics,
            "dc_residual": finite_mean(dc_residuals[method]),
            "heldout_residual": finite_mean(heldout_residuals[method]),
        }
    baseline = method_metrics["one_shot"]
    deltas = {
        method: {
            key: method_metrics[method][key] - baseline[key]
            for key in ("psnr", "ssim", "nmse", "heldout_residual")
        }
        for method in METHODS
        if method != "one_shot"
    }
    best_methods = select_best_methods(method_metrics)
    best_method = best_methods[0]

    volume_result = {
        "slice_indices": slice_indices,
        "target": target_stack,
        **method_arrays,
        **residual_arrays,
    }
    visual_rows = choose_visual_indices(len(slice_indices), slices_per_volume)
    save_comparison_png(volume_result, volume_dir / "comparison.png", visual_rows, best_methods)
    save_residual_maps_png(volume_result, volume_dir / "residual_maps.png", visual_rows)

    metrics_payload = {
        "volume": volume_name,
        "num_slices": len(slice_indices),
        "slice_indices": slice_indices,
        "visualized_slice_indices": [slice_indices[idx] for idx in visual_rows],
        "metrics": method_metrics,
        "deltas_vs_one_shot": deltas,
        "best_method_by_heldout_residual": best_method,
        "displayed_pnp_methods": best_methods,
    }
    with (volume_dir / "metrics.json").open("w") as json_file:
        json.dump(json_ready(metrics_payload), json_file, indent=2)

    row: dict[str, Any] = {"volume": volume_name, "num_slices": len(slice_indices), "best_method": best_method}
    for method in METHODS:
        for key, value in method_metrics[method].items():
            row[f"{method}_{key}"] = value
    for method, values in deltas.items():
        for key, value in values.items():
            row[f"{method}_delta_{key}"] = value
    return row


def write_metrics_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["volume", "num_slices", "best_method"]
    for method in METHODS:
        fieldnames.extend([f"{method}_psnr", f"{method}_ssim", f"{method}_nmse", f"{method}_dc_residual", f"{method}_heldout_residual"])
    for method in METHODS:
        if method != "one_shot":
            fieldnames.extend([f"{method}_delta_psnr", f"{method}_delta_ssim", f"{method}_delta_nmse", f"{method}_delta_heldout_residual"])
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def summarize_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mean_metrics: dict[str, Any] = {}
    rankings: list[dict[str, Any]] = []
    for method in METHODS:
        entry = {
            key: float(np.mean([row[f"{method}_{key}"] for row in rows])) if rows else float("nan")
            for key in ("psnr", "ssim", "nmse", "dc_residual", "heldout_residual")
        }
        mean_metrics[method] = entry
        rankings.append({"method": method, **entry})
    rankings.sort(key=lambda row: (row["heldout_residual"], -row["psnr"]))
    return mean_metrics, rankings


def write_rankings_csv(output_path: Path, rankings: list[dict[str, Any]]) -> None:
    fieldnames = ["rank", "method", "psnr", "ssim", "nmse", "dc_residual", "heldout_residual"]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rankings, start=1):
            writer.writerow({"rank": idx, **{key: row.get(key) for key in fieldnames if key != "rank"}})


def load_previous_pnp_summary(repo_root: Path) -> dict[str, Any] | None:
    path = repo_root / "visual_check" / "PnP" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("mean_metrics")


def main() -> int:
    args = parse_args()
    configure_imports(args.repo_root)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")

    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)
    model, config, checkpoint, load_info = load_model_from_checkpoint(args.checkpoint, device)
    dataset = build_validation_dataset(config)
    selected_volumes, indices_by_volume = select_volume_indices(dataset, args.volume_names, args.num_volumes)

    ft_cfg = config.fastmri_finetune
    patch_size = tuple(int(dim) for dim in ft_cfg.train_patch_size)
    crop_size = tuple(int(dim) for dim in ft_cfg.crop_size)
    overlap_values = config.get("overlap_for_inference", [16, 16, 0])
    overlap = (int(overlap_values[0]), int(overlap_values[1]))
    patch_batch_size = int(args.eval_patch_batch_size or ft_cfg.get("eval_patch_batch_size", 64))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_top_level:
        (args.output_dir / "selected_volumes.txt").write_text("\n".join(selected_volumes) + "\n")

    rows = []
    for volume_name in tqdm(selected_volumes, desc="volumes"):
        rows.append(
            run_volume(
                model=model,
                dataset=dataset,
                volume_name=volume_name,
                dataset_indices=indices_by_volume[volume_name],
                patch_size=patch_size,
                overlap=overlap,
                patch_batch_size=patch_batch_size,
                crop_size=crop_size,
                acc_factor=int(ft_cfg.acc_factor),
                heldout_fraction=float(args.heldout_fraction),
                lowpass_fraction=float(args.lowpass_fraction),
                device=device,
                output_dir=args.output_dir,
                slices_per_volume=args.slices_per_volume,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_metrics, rankings = summarize_rows(rows)
    previous_pnp = load_previous_pnp_summary(args.repo_root)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "selected_volumes": selected_volumes,
        "num_volumes": len(selected_volumes),
        "patch_size": patch_size,
        "crop_size": crop_size,
        "overlap": overlap,
        "eval_patch_batch_size": patch_batch_size,
        "heldout_fraction": float(args.heldout_fraction),
        "lowpass_fraction": float(args.lowpass_fraction),
        "load_info": load_info,
        "mean_metrics": mean_metrics,
        "method_rankings": rankings,
        "previous_pnp_mean_metrics": previous_pnp,
        "volumes": rows,
    }
    if args.skip_top_level:
        if args.worker_summary is not None:
            args.worker_summary.parent.mkdir(parents=True, exist_ok=True)
            args.worker_summary.write_text(json.dumps(json_ready(summary), indent=2))
    else:
        write_metrics_csv(args.output_dir / "metrics.csv", rows)
        write_rankings_csv(args.output_dir / "method_rankings.csv", rankings)
        (args.output_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2))

    print("Finished guarded PnP sweep.", flush=True)
    print(json.dumps(json_ready({"top_ranked": rankings[:5]}), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
