#!/usr/bin/env python
"""Run PnP FastMRI validation inference and visual checks outside SNRAware."""

from __future__ import annotations

import argparse
import csv
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
    DEFAULT_REPO_ROOT
    / "checkpoints/fine_tune/warmup_then_both_20260517-145527/last.pth"
)
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "visual_check" / "PnP"
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
METHODS = ["zero_filled", "one_shot", "one_shot_dc", "pnp_k2", "pnp_k3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-shot+DC and damped PnP inference for FastMRI SNRAware visual checks."
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-volumes", type=int, default=10)
    parser.add_argument("--volume-names", nargs="*", default=None)
    parser.add_argument("--slices-per-volume", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-patch-batch-size", type=int, default=None)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--max-pnp-iter", type=int, default=3)
    parser.add_argument(
        "--skip-top-level",
        action="store_true",
        help="Only write per-volume outputs; useful for parallel workers sharing one output dir.",
    )
    parser.add_argument(
        "--worker-summary",
        type=Path,
        default=None,
        help="Optional JSON path for a worker summary when --skip-top-level is used.",
    )
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
    gfactor_kwargs = OmegaConf.to_container(ft_cfg.gfactor_unet, resolve=True)

    model, _base_config, load_info = build_fastmri_wrapped_model(
        base_config_path=config.base_model.config_path,
        base_checkpoint_path=config.base_model.checkpoint_path,
        height=model_size[0],
        width=model_size[1],
        depth=1,
        lora_config=config.get("lora"),
        gfactor_unet_kwargs=gfactor_kwargs,
        use_unet=bool(ft_cfg.get("use_unet", True)),
    )
    missing, unexpected = load_fastmri_finetune_checkpoint(
        model=model,
        checkpoint=checkpoint,
        apply_lora_fn=apply_lora_to_model,
        lora_config=config.get("lora"),
    )
    if missing or unexpected:
        raise RuntimeError(
            "Unexpected checkpoint restore mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

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

    if requested:
        selected = [name.removesuffix(".h5") for name in requested]
    else:
        selected = [name for name in DEFAULT_VOLUMES if name in indices_by_volume][:num_volumes]

    missing = [name for name in selected if name not in indices_by_volume]
    if missing:
        raise ValueError(f"Requested volumes not found in validation dataset: {missing}")
    return selected, {name: indices_by_volume[name] for name in selected}


def sliding_window_positions(size: int, patch_size: int, overlap: int) -> list[int]:
    if patch_size > size:
        raise ValueError(f"Patch size {patch_size} cannot exceed image size {size}")
    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError(f"Invalid sliding-window stride from patch={patch_size}, overlap={overlap}")
    positions = list(range(0, size - patch_size + 1, stride))
    if not positions:
        return [0]
    final = size - patch_size
    if positions[-1] != final:
        positions.append(final)
    return positions


def predict_gfactor_for_patch(model, patch_tensor: torch.Tensor) -> torch.Tensor:
    patch_5d = model._prepare_2ch_input(patch_tensor)
    current_mean_scale = model._fastmri_current_mean_scale(patch_5d)
    return model.predict_gfactor(patch_5d / current_mean_scale).float()


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
    overlap_h, overlap_w = overlap
    top_positions = sliding_window_positions(full_h, patch_h, overlap_h)
    left_positions = sliding_window_positions(full_w, patch_w, overlap_w)

    pred_sum = torch.zeros((2, full_h, full_w), device=device, dtype=torch.float32)
    gfactor_sum = torch.zeros((1, full_h, full_w), device=device, dtype=torch.float32)
    weight_sum = torch.zeros((1, full_h, full_w), device=device, dtype=torch.float32)
    patch_buffer: list[torch.Tensor] = []
    coord_buffer: list[tuple[int, int]] = []

    def flush_patch_buffer() -> None:
        if not patch_buffer:
            return
        patch_tensor = torch.stack(patch_buffer, dim=0).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        patch_output = model(patch_tensor, checkpoint_base_model=False).squeeze(2).float()
        patch_gfactor = predict_gfactor_for_patch(model, patch_tensor)
        for patch_index, (top, left) in enumerate(coord_buffer):
            pred_sum[:, top : top + patch_h, left : left + patch_w] += patch_output[patch_index]
            gfactor_sum[:, top : top + patch_h, left : left + patch_w] += patch_gfactor[patch_index]
            weight_sum[:, top : top + patch_h, left : left + patch_w] += 1.0
        patch_buffer.clear()
        coord_buffer.clear()

    for top in top_positions:
        for left in left_positions:
            patch_buffer.append(noisy[:, top : top + patch_h, left : left + patch_w])
            coord_buffer.append((top, left))
            if len(patch_buffer) >= patch_batch_size:
                flush_patch_buffer()
    flush_patch_buffer()

    if not torch.all(weight_sum > 0):
        raise RuntimeError("Sliding-window inference left uncovered pixels")
    return pred_sum / weight_sum, gfactor_sum / weight_sum


def fft2c_complex(image: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(image, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def ifft2c_complex(kspace: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[0].float(), x[1].float())


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([x.real.float(), x.imag.float()], dim=0)


def center_crop_complex(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    height, width = shape
    top = (x.shape[-2] - height) // 2
    left = (x.shape[-1] - width) // 2
    return x[..., top : top + height, left : left + width]


def center_embed_complex(crop: torch.Tensor, full_shape: tuple[int, int]) -> torch.Tensor:
    full = torch.zeros(full_shape, device=crop.device, dtype=crop.dtype)
    top = (full_shape[0] - crop.shape[-2]) // 2
    left = (full_shape[1] - crop.shape[-1]) // 2
    full[top : top + crop.shape[-2], left : left + crop.shape[-1]] = crop
    return full


def make_dc_inputs(
    *,
    raw_kspace: np.ndarray,
    file_name: str,
    acc_factor: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    from fastmri_data.work_with_snraware import legacy_uniform1d_mask

    raw = np.asarray(raw_kspace)
    raw_complex = torch.from_numpy(raw.astype(np.complex64)).to(device=device)
    raw_complex_last = torch.zeros((1, raw.shape[0], raw.shape[1], 2), dtype=torch.float32)
    raw_complex_last[0, :, :, 0] = torch.from_numpy(raw.real.astype(np.float32))
    raw_complex_last[0, :, :, 1] = torch.from_numpy(raw.imag.astype(np.float32))

    center_fraction = 0.08 if int(acc_factor) == 4 else 0.04
    mask = legacy_uniform1d_mask(
        img=raw_complex_last.permute(0, 3, 1, 2),
        size=raw_complex_last.shape[2],
        batch_size=1,
        acc_factor=int(acc_factor),
        center_fraction=center_fraction,
        fix=False,
        name=file_name,
    )
    mask_bool = mask[0, 0].to(device=device, dtype=torch.bool)
    return raw_complex, mask_bool


def apply_full_fov_dc(
    prediction_crop: torch.Tensor,
    *,
    observed_kspace: torch.Tensor,
    mask: torch.Tensor,
    crop_size: tuple[int, int],
) -> torch.Tensor:
    pred_complex = channels_to_complex(prediction_crop.to(device=observed_kspace.device))
    pred_full = center_embed_complex(pred_complex, tuple(observed_kspace.shape[-2:]))
    pred_kspace = fft2c_complex(pred_full)
    dc_kspace = torch.where(mask, observed_kspace, pred_kspace)
    dc_full = ifft2c_complex(dc_kspace)
    return complex_to_channels(center_crop_complex(dc_full, crop_size))


def magnitude_from_complex_channels(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().float().numpy()
    return np.sqrt(np.square(x[0]) + np.square(x[1])).astype(np.float32)


def minmax_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    min_value = float(finite.min())
    max_value = float(finite.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def choose_visual_indices(num_slices: int, requested: int) -> list[int]:
    count = max(1, min(int(requested), num_slices))
    return sorted(set(int(idx) for idx in np.linspace(0, num_slices - 1, count)))


def finite_stats(stack: np.ndarray) -> dict[str, float]:
    finite = np.asarray(stack, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"min": float("nan"), "mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "p95": float(np.percentile(finite, 95)),
        "max": float(finite.max()),
    }


def compute_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    from fastmri.evaluate import nmse, psnr, ssim

    return {
        "psnr": float(np.asarray(psnr(target, prediction)).mean()),
        "ssim": float(np.asarray(ssim(target, prediction)).mean()),
        "nmse": float(np.asarray(nmse(target, prediction)).mean()),
    }


def save_comparison_png(volume_result: dict[str, Any], output_path: Path, visual_rows: list[int]) -> None:
    columns = [
        ("Zero-filled", "zero_filled", "image"),
        ("One-shot", "one_shot", "image"),
        ("Err one-shot", "one_shot", "error"),
        ("One-shot+DC", "one_shot_dc", "image"),
        ("Err DC", "one_shot_dc", "error"),
        ("PnP K2", "pnp_k2", "image"),
        ("Err K2", "pnp_k2", "error"),
        ("PnP K3", "pnp_k3", "image"),
        ("Err K3", "pnp_k3", "error"),
        ("Target", "target", "image"),
    ]
    fig, axes = plt.subplots(
        len(visual_rows),
        len(columns),
        figsize=(2.5 * len(columns), 2.7 * len(visual_rows)),
        dpi=130,
        squeeze=False,
    )
    for row_index, slice_position in enumerate(visual_rows):
        slice_idx = volume_result["slice_indices"][slice_position]
        target = volume_result["target"][slice_position]
        for col_index, (title, key, panel_type) in enumerate(columns):
            axis = axes[row_index][col_index]
            if panel_type == "error":
                image = np.abs(volume_result[key][slice_position] - target)
                cmap = "magma"
            else:
                image = volume_result[key][slice_position]
                cmap = "gray"
            axis.imshow(minmax_image(image), cmap=cmap, vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(title, fontsize=9)
            if col_index == 0:
                axis.set_ylabel(f"slice {slice_idx}", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_gfactor_png(volume_result: dict[str, Any], output_path: Path, visual_rows: list[int]) -> None:
    fig, axes = plt.subplots(
        len(visual_rows),
        2,
        figsize=(6.4, 2.8 * len(visual_rows)),
        dpi=130,
        squeeze=False,
    )
    gfactor_stack = np.concatenate(
        [
            np.asarray(volume_result["gfactor_one_shot"], dtype=np.float32).reshape(-1),
            np.asarray(volume_result["gfactor_pnp_k3"], dtype=np.float32).reshape(-1),
        ]
    )
    finite = gfactor_stack[np.isfinite(gfactor_stack)]
    vmin = float(np.percentile(finite, 1)) if finite.size else 1.0
    vmax = float(np.percentile(finite, 99)) if finite.size else 5.0
    if vmax <= vmin:
        vmin, vmax = 1.0, 5.0

    last_image = None
    for row_index, slice_position in enumerate(visual_rows):
        slice_idx = volume_result["slice_indices"][slice_position]
        for col_index, (title, key) in enumerate(
            [("One-shot g-factor", "gfactor_one_shot"), ("PnP K3 g-factor", "gfactor_pnp_k3")]
        ):
            axis = axes[row_index][col_index]
            last_image = axis.imshow(volume_result[key][slice_position], cmap="viridis", vmin=vmin, vmax=vmax)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(title, fontsize=10)
            if col_index == 0:
                axis.set_ylabel(f"slice {slice_idx}", fontsize=9)
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


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
    damping: float,
    max_pnp_iter: int,
    device: torch.device,
    output_dir: Path,
    slices_per_volume: int,
) -> dict[str, Any]:
    volume_dir = output_dir / volume_name
    volume_dir.mkdir(parents=True, exist_ok=True)

    method_stacks: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    target_images: list[np.ndarray] = []
    gfactor_one_shot: list[np.ndarray] = []
    gfactor_pnp_k3: list[np.ndarray] = []
    slice_indices: list[int] = []

    progress = tqdm(dataset_indices, desc=volume_name, leave=False)
    with torch.inference_mode():
        for dataset_index in progress:
            noisy, clean, _noise_sigma, metadata = dataset[dataset_index]
            path, slice_idx, _example_metadata = dataset.slice_dataset.examples[dataset_index]
            with h5py.File(path, "r") as hf:
                raw_kspace = np.asarray(hf["kspace"][slice_idx])

            observed_kspace, mask = make_dc_inputs(
                raw_kspace=raw_kspace,
                file_name=Path(path).name,
                acc_factor=acc_factor,
                device=device,
            )

            noisy = noisy.to(dtype=torch.float32)
            one_shot, g_one = run_sliding_window_slice(
                model,
                noisy,
                patch_size=patch_size,
                overlap=overlap,
                patch_batch_size=patch_batch_size,
                device=device,
            )
            one_shot_dc = apply_full_fov_dc(
                one_shot,
                observed_kspace=observed_kspace,
                mask=mask,
                crop_size=crop_size,
            )

            x = one_shot_dc
            pnp_outputs: dict[int, torch.Tensor] = {}
            pnp_gfactor: dict[int, torch.Tensor] = {}
            for iteration in range(1, max_pnp_iter + 1):
                z, g_iter = run_sliding_window_slice(
                    model,
                    x,
                    patch_size=patch_size,
                    overlap=overlap,
                    patch_batch_size=patch_batch_size,
                    device=device,
                )
                damped = x + float(damping) * (z - x)
                x = apply_full_fov_dc(
                    damped,
                    observed_kspace=observed_kspace,
                    mask=mask,
                    crop_size=crop_size,
                )
                pnp_outputs[iteration] = x
                pnp_gfactor[iteration] = g_iter

            pnp_k2 = pnp_outputs[2 if max_pnp_iter >= 2 else max_pnp_iter]
            pnp_k3 = pnp_outputs[3 if max_pnp_iter >= 3 else max_pnp_iter]
            g_pnp_final = pnp_gfactor[3 if max_pnp_iter >= 3 else max_pnp_iter]

            method_tensors = {
                "zero_filled": noisy,
                "one_shot": one_shot,
                "one_shot_dc": one_shot_dc,
                "pnp_k2": pnp_k2,
                "pnp_k3": pnp_k3,
            }
            for method, tensor in method_tensors.items():
                method_stacks[method].append(magnitude_from_complex_channels(tensor))
            target_images.append(clean.squeeze(0).detach().cpu().float().numpy().astype(np.float32))
            gfactor_one_shot.append(g_one.squeeze(0).detach().cpu().float().numpy().astype(np.float32))
            gfactor_pnp_k3.append(g_pnp_final.squeeze(0).detach().cpu().float().numpy().astype(np.float32))
            slice_indices.append(int(metadata["slice_idx"]))

    order = np.argsort(np.asarray(slice_indices))
    slice_indices = [slice_indices[int(index)] for index in order]
    method_arrays = {
        method: np.stack([values[int(index)] for index in order], axis=0).astype(np.float32)
        for method, values in method_stacks.items()
    }
    target_stack = np.stack([target_images[int(index)] for index in order], axis=0).astype(np.float32)
    g_one_stack = np.stack([gfactor_one_shot[int(index)] for index in order], axis=0).astype(np.float32)
    g_k3_stack = np.stack([gfactor_pnp_k3[int(index)] for index in order], axis=0).astype(np.float32)

    metrics = {method: compute_metrics(target_stack, method_arrays[method]) for method in METHODS}
    baseline = metrics["one_shot"]
    deltas = {
        method: {metric: metrics[method][metric] - baseline[metric] for metric in ("psnr", "ssim", "nmse")}
        for method in ("one_shot_dc", "pnp_k2", "pnp_k3")
    }

    volume_result = {
        "volume": volume_name,
        "slice_indices": slice_indices,
        "target": target_stack,
        "gfactor_one_shot": g_one_stack,
        "gfactor_pnp_k3": g_k3_stack,
        **method_arrays,
    }
    visual_rows = choose_visual_indices(len(slice_indices), slices_per_volume)
    save_comparison_png(volume_result, volume_dir / "comparison.png", visual_rows)
    save_gfactor_png(volume_result, volume_dir / "gfactor_map.png", visual_rows)

    metrics_payload = {
        "volume": volume_name,
        "num_slices": len(slice_indices),
        "slice_indices": slice_indices,
        "visualized_slice_indices": [slice_indices[index] for index in visual_rows],
        "metrics": metrics,
        "deltas_vs_one_shot": deltas,
        "gfactor_one_shot": finite_stats(g_one_stack),
        "gfactor_pnp_k3": finite_stats(g_k3_stack),
        "pnp": {"damping": float(damping), "max_pnp_iter": int(max_pnp_iter)},
    }
    with (volume_dir / "metrics.json").open("w") as json_file:
        json.dump(json_ready(metrics_payload), json_file, indent=2)

    row: dict[str, Any] = {"volume": volume_name, "num_slices": len(slice_indices)}
    for method in METHODS:
        for metric_name, metric_value in metrics[method].items():
            row[f"{method}_{metric_name}"] = metric_value
    for method in ("one_shot_dc", "pnp_k2", "pnp_k3"):
        for metric_name, metric_value in deltas[method].items():
            row[f"{method}_delta_{metric_name}"] = metric_value
    return row


def write_metrics_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["volume", "num_slices"]
    for method in METHODS:
        fieldnames.extend([f"{method}_psnr", f"{method}_ssim", f"{method}_nmse"])
    for method in ("one_shot_dc", "pnp_k2", "pnp_k3"):
        fieldnames.extend(
            [f"{method}_delta_psnr", f"{method}_delta_ssim", f"{method}_delta_nmse"]
        )
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in METHODS:
        summary[method] = {
            metric: float(np.mean([row[f"{method}_{metric}"] for row in rows])) if rows else float("nan")
            for metric in ("psnr", "ssim", "nmse")
        }
    for method in ("one_shot_dc", "pnp_k2", "pnp_k3"):
        summary[f"{method}_delta_vs_one_shot"] = {
            metric: float(np.mean([row[f"{method}_delta_{metric}"] for row in rows]))
            if rows
            else float("nan")
            for metric in ("psnr", "ssim", "nmse")
        }
    return summary


def main() -> int:
    args = parse_args()
    configure_imports(args.repo_root)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")

    if args.max_pnp_iter < 2:
        raise ValueError("--max-pnp-iter must be at least 2 so PnP_K2 can be evaluated")

    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)
    model, config, checkpoint, load_info = load_model_from_checkpoint(args.checkpoint, device)
    dataset = build_validation_dataset(config)
    selected_volumes, indices_by_volume = select_volume_indices(
        dataset,
        args.volume_names,
        args.num_volumes,
    )

    ft_cfg = config.fastmri_finetune
    patch_size = tuple(int(dim) for dim in ft_cfg.train_patch_size)
    crop_size = tuple(int(dim) for dim in ft_cfg.crop_size)
    overlap_cfg = config.get("overlap_for_inference", [16, 16, 0])
    overlap = (int(overlap_cfg[0]), int(overlap_cfg[1]))
    patch_batch_size = int(args.eval_patch_batch_size or ft_cfg.get("eval_patch_batch_size", 64))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_top_level:
        with (args.output_dir / "selected_volumes.txt").open("w") as selected_file:
            selected_file.write("\n".join(selected_volumes) + "\n")

    rows: list[dict[str, Any]] = []
    for volume_name in tqdm(selected_volumes, desc="volumes"):
        row = run_volume(
            model=model,
            dataset=dataset,
            volume_name=volume_name,
            dataset_indices=indices_by_volume[volume_name],
            patch_size=patch_size,
            overlap=overlap,
            patch_batch_size=patch_batch_size,
            crop_size=crop_size,
            acc_factor=int(ft_cfg.acc_factor),
            damping=float(args.damping),
            max_pnp_iter=int(args.max_pnp_iter),
            device=device,
            output_dir=args.output_dir,
            slices_per_volume=args.slices_per_volume,
        )
        rows.append(row)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metric_summary = summarize_rows(rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics", {}),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "selected_volumes": selected_volumes,
        "num_volumes": len(selected_volumes),
        "patch_size": patch_size,
        "crop_size": crop_size,
        "overlap": overlap,
        "eval_patch_batch_size": patch_batch_size,
        "pnp": {"damping": float(args.damping), "max_pnp_iter": int(args.max_pnp_iter)},
        "load_info": load_info,
        "mean_metrics": metric_summary,
        "volumes": rows,
    }

    if args.skip_top_level:
        if args.worker_summary is not None:
            args.worker_summary.parent.mkdir(parents=True, exist_ok=True)
            with args.worker_summary.open("w") as json_file:
                json.dump(json_ready(summary), json_file, indent=2)
    else:
        write_metrics_csv(args.output_dir / "metrics.csv", rows)
        with (args.output_dir / "summary.json").open("w") as json_file:
            json.dump(json_ready(summary), json_file, indent=2)

    print("Finished PnP visual check.", flush=True)
    print(json.dumps(json_ready(metric_summary), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
