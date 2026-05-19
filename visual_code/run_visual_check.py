#!/usr/bin/env python
"""Run FastMRI validation inference and g-factor visualization outside SNRAware."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

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
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "visual_check"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a FastMRI SNRAware fine-tune checkpoint and save PNG visual checks."
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-volumes", type=int, default=10)
    parser.add_argument("--volume-names", nargs="*", default=None)
    parser.add_argument("--slices-per-volume", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-patch-batch-size", type=int, default=None)
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
        selected = ordered_volumes[:num_volumes]

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


def save_comparison_png(volume_result: dict[str, Any], output_path: Path, visual_rows: list[int]) -> None:
    columns = [
        ("Zero-filled", "input", "gray"),
        ("Prediction", "prediction", "gray"),
        ("Target", "target", "gray"),
        ("Abs error", "error", "magma"),
        ("G-factor", "gfactor", "viridis"),
    ]
    fig, axes = plt.subplots(
        len(visual_rows),
        len(columns),
        figsize=(3.0 * len(columns), 3.0 * len(visual_rows)),
        dpi=130,
        squeeze=False,
    )
    for row_index, slice_position in enumerate(visual_rows):
        slice_idx = volume_result["slice_indices"][slice_position]
        for col_index, (title, key, cmap) in enumerate(columns):
            axis = axes[row_index][col_index]
            image = volume_result[key][slice_position]
            axis.imshow(minmax_image(image), cmap=cmap, vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(title, fontsize=10)
            if col_index == 0:
                axis.set_ylabel(f"slice {slice_idx}", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_gfactor_png(volume_result: dict[str, Any], output_path: Path, visual_rows: list[int]) -> None:
    fig, axes = plt.subplots(
        1,
        len(visual_rows),
        figsize=(3.2 * len(visual_rows), 3.4),
        dpi=130,
        squeeze=False,
    )
    gfactor_stack = np.asarray(volume_result["gfactor"], dtype=np.float32)
    finite = gfactor_stack[np.isfinite(gfactor_stack)]
    vmin = float(np.percentile(finite, 1)) if finite.size else 1.0
    vmax = float(np.percentile(finite, 99)) if finite.size else 5.0
    if vmax <= vmin:
        vmin, vmax = 1.0, 5.0
    last_image = None
    for col_index, slice_position in enumerate(visual_rows):
        axis = axes[0][col_index]
        slice_idx = volume_result["slice_indices"][slice_position]
        last_image = axis.imshow(
            volume_result["gfactor"][slice_position],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(f"slice {slice_idx}", fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


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
        "psnr": float(psnr(target, prediction)),
        "ssim": float(ssim(target, prediction)),
        "nmse": float(nmse(target, prediction)),
    }


def write_metrics_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "volume",
        "num_slices",
        "psnr",
        "ssim",
        "nmse",
        "gfactor_min",
        "gfactor_mean",
        "gfactor_p95",
        "gfactor_max",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def run_volume(
    *,
    model,
    dataset,
    volume_name: str,
    dataset_indices: list[int],
    patch_size: tuple[int, int],
    overlap: tuple[int, int],
    patch_batch_size: int,
    device: torch.device,
    output_dir: Path,
    slices_per_volume: int,
) -> dict[str, Any]:
    volume_dir = output_dir / volume_name
    volume_dir.mkdir(parents=True, exist_ok=True)

    input_images: list[np.ndarray] = []
    prediction_images: list[np.ndarray] = []
    target_images: list[np.ndarray] = []
    error_images: list[np.ndarray] = []
    gfactor_images: list[np.ndarray] = []
    slice_indices: list[int] = []

    progress = tqdm(dataset_indices, desc=volume_name, leave=False)
    with torch.inference_mode():
        for dataset_index in progress:
            noisy, clean, _noise_sigma, metadata = dataset[dataset_index]
            noisy = noisy.to(dtype=torch.float32)
            prediction_complex, gfactor = run_sliding_window_slice(
                model,
                noisy,
                patch_size=patch_size,
                overlap=overlap,
                patch_batch_size=patch_batch_size,
                device=device,
            )
            prediction_mag = magnitude_from_complex_channels(prediction_complex)
            input_mag = magnitude_from_complex_channels(noisy)
            target_mag = clean.squeeze(0).detach().cpu().float().numpy().astype(np.float32)

            mean = float(torch.as_tensor(metadata["mean"]).item())
            std = float(torch.as_tensor(metadata["std"]).item())
            prediction_mag = prediction_mag * std + mean
            target_mag = target_mag * std + mean
            input_mag = input_mag * std + mean
            error_mag = np.abs(prediction_mag - target_mag).astype(np.float32)

            input_images.append(input_mag)
            prediction_images.append(prediction_mag.astype(np.float32))
            target_images.append(target_mag.astype(np.float32))
            error_images.append(error_mag)
            gfactor_images.append(gfactor.squeeze(0).detach().cpu().float().numpy().astype(np.float32))
            slice_indices.append(int(metadata["slice_idx"]))

    order = np.argsort(np.asarray(slice_indices))
    slice_indices = [slice_indices[int(index)] for index in order]
    input_stack = np.stack([input_images[int(index)] for index in order], axis=0)
    prediction_stack = np.stack([prediction_images[int(index)] for index in order], axis=0)
    target_stack = np.stack([target_images[int(index)] for index in order], axis=0)
    error_stack = np.stack([error_images[int(index)] for index in order], axis=0)
    gfactor_stack = np.stack([gfactor_images[int(index)] for index in order], axis=0)

    metrics = compute_metrics(target_stack, prediction_stack)
    gfactor_stats = finite_stats(gfactor_stack)
    volume_result = {
        "volume": volume_name,
        "slice_indices": slice_indices,
        "input": input_stack,
        "prediction": prediction_stack,
        "target": target_stack,
        "error": error_stack,
        "gfactor": gfactor_stack,
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
        "gfactor": gfactor_stats,
    }
    with (volume_dir / "metrics.json").open("w") as json_file:
        json.dump(json_ready(metrics_payload), json_file, indent=2)

    return {
        "volume": volume_name,
        "num_slices": len(slice_indices),
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "nmse": metrics["nmse"],
        "gfactor_min": gfactor_stats["min"],
        "gfactor_mean": gfactor_stats["mean"],
        "gfactor_p95": gfactor_stats["p95"],
        "gfactor_max": gfactor_stats["max"],
    }


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
    selected_volumes, indices_by_volume = select_volume_indices(
        dataset,
        args.volume_names,
        args.num_volumes,
    )

    ft_cfg = config.fastmri_finetune
    patch_size = tuple(int(dim) for dim in ft_cfg.train_patch_size)
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
            device=device,
            output_dir=args.output_dir,
            slices_per_volume=args.slices_per_volume,
        )
        rows.append(row)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metric_summary = {
        name: float(np.mean([row[name] for row in rows])) if rows else float("nan")
        for name in ("psnr", "ssim", "nmse")
    }
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics", {}),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "selected_volumes": selected_volumes,
        "num_volumes": len(selected_volumes),
        "patch_size": patch_size,
        "overlap": overlap,
        "eval_patch_batch_size": patch_batch_size,
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

    print("Finished visual check.", flush=True)
    print(json.dumps(json_ready(metric_summary), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
