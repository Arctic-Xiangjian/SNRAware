#!/usr/bin/env python
"""Run crop-domain PnP experiments using the flow_unrolled FastMRI data path."""

from __future__ import annotations

import argparse
import csv
import importlib.util
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
from fastmri import fft2c, ifft2c
from tqdm.auto import tqdm


HELPER_PATH = Path(__file__).resolve().parent / "run_pnp_guarded_sweep.py"
DEFAULT_REPO_ROOT = Path("/working2/arctic/snrawre/SNRAware")
DEFAULT_FLOW_ROOT = Path("/working2/arctic/flow_unrolled")
DEFAULT_CHECKPOINT = (
    DEFAULT_REPO_ROOT / "checkpoints/fine_tune/warmup_then_both_20260517-145527/last.pth"
)
DEFAULT_VAL_ROOT = Path("/working2/arctic/VAR/data_fastmri/single_coil_knee/val")
DEFAULT_OUTPUT_BASE = DEFAULT_REPO_ROOT / "visual_check"
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
METRIC_KEYS = ("psnr", "ssim", "nmse", "dc_residual")


def load_helper():
    if not HELPER_PATH.exists():
        raise FileNotFoundError(f"Required helper script not found: {HELPER_PATH}")
    spec = importlib.util.spec_from_file_location("pnp_guarded_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper script: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pnp_guarded_helper"] = module
    spec.loader.exec_module(module)
    return module


g = load_helper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run crop-domain PnP FastMRI experiments.")
    parser.add_argument("--experiment", choices=["strong", "guarded", "long"], required=True)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--flow-root", type=Path, default=DEFAULT_FLOW_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--val-root", type=Path, default=DEFAULT_VAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-volumes", type=int, default=10)
    parser.add_argument("--volume-names", nargs="*", default=None)
    parser.add_argument("--slices-per-volume", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-patch-batch-size", type=int, default=None)
    parser.add_argument("--slice-limit", type=int, default=None)
    return parser.parse_args()


def configure_paths(repo_root: Path, flow_root: Path) -> None:
    g.configure_imports(repo_root)
    flow_root = flow_root.resolve()
    flow_text = str(flow_root)
    if flow_text in sys.path:
        sys.path.remove(flow_text)
    sys.path.insert(0, flow_text)
    os.chdir(repo_root)


def build_crop_domain_dataset(val_root: Path, acc_factor: int):
    from data.fastmri_data import SliceDataset
    from data.warp_fastmri_singlecoil import MRIDataset

    return MRIDataset(
        SliceDataset(root=val_root, challenge="singlecoil", sample_seed=1234),
        acc_factor=int(acc_factor),
        require_latent_feature=False,
        crop_size=(320, 320),
        use_seed=True,
    )


def select_volume_indices(dataset, requested: list[str] | None, num_volumes: int) -> tuple[list[str], dict[str, list[int]]]:
    examples = dataset.given_dataset.examples
    ordered: list[str] = []
    indices: dict[str, list[int]] = {}
    for idx, (path, _slice_idx, _metadata) in enumerate(examples):
        volume = Path(path).stem
        if volume not in indices:
            ordered.append(volume)
            indices[volume] = []
        indices[volume].append(idx)
    selected = [name.removesuffix(".h5") for name in requested] if requested else [
        name for name in DEFAULT_VOLUMES if name in indices
    ][:num_volumes]
    missing = [name for name in selected if name not in indices]
    if missing:
        raise ValueError(f"Requested volumes not found: {missing}")
    return selected, {name: indices[name] for name in selected}


def experiment_output_dir(experiment: str) -> Path:
    return {
        "strong": DEFAULT_OUTPUT_BASE / "PnP_cropDC",
        "guarded": DEFAULT_OUTPUT_BASE / "PnP_guarded_cropDC",
        "long": DEFAULT_OUTPUT_BASE / "PnP_long_a0p2_cropDC",
    }[experiment]


def experiment_methods(experiment: str) -> list[str]:
    if experiment == "strong":
        return ["one_shot", "one_shot_dc", "pnp_k2_alpha0p5", "pnp_k3_alpha0p5"]
    if experiment == "guarded":
        return [
            "one_shot",
            "one_shot_dc",
            "pnp_k1_alpha0p1",
            "pnp_k2_alpha0p1",
            "pnp_k1_alpha0p2",
            "pnp_k2_alpha0p2",
            "pnp_k1_alpha0p3",
            "pnp_k2_alpha0p3",
            "pnp_k2_alpha0p2_lowpass",
        ]
    if experiment == "long":
        return ["one_shot", "one_shot_dc", "pnp_k1", "pnp_k2", "pnp_k4", "pnp_k6", "pnp_k8", "pnp_k10"]
    raise ValueError(f"Unknown experiment: {experiment}")


def display_name(method: str) -> str:
    return (
        method.replace("one_shot_dc", "One-shot+DC")
        .replace("one_shot", "One-shot")
        .replace("alpha0p", " a0.")
        .replace("pnp_k", "K")
        .replace("_lowpass", " LP")
        .replace("_", " ")
    )


def channels_to_last(x: torch.Tensor) -> torch.Tensor:
    return x.permute(1, 2, 0).contiguous()


def last_to_channels(x: torch.Tensor) -> torch.Tensor:
    return x.permute(2, 0, 1).contiguous()


def crop_domain_dc(pred_crop: torch.Tensor, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_kspace = fft2c(channels_to_last(pred_crop))
    dc_kspace = torch.where(mask.bool(), masked_kspace, pred_kspace)
    return last_to_channels(ifft2c(dc_kspace)).float()


def crop_domain_residual(pred_crop: torch.Tensor, masked_kspace: torch.Tensor, mask: torch.Tensor) -> float:
    pred_kspace = fft2c(channels_to_last(pred_crop.to(masked_kspace.device)))
    sampled = mask.bool()
    if not bool(sampled.any().item()):
        return float("nan")
    diff = pred_kspace[sampled] - masked_kspace[sampled]
    denom = torch.linalg.vector_norm(masked_kspace[sampled]).clamp_min(1e-12)
    return float((torch.linalg.vector_norm(diff) / denom).detach().cpu().item())


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def magnitude_from_channels(x: torch.Tensor | np.ndarray) -> np.ndarray:
    return g.magnitude_from_complex_channels(x)


def run_pnp_iterations(
    model,
    *,
    x0: torch.Tensor,
    masked_kspace: torch.Tensor,
    mask: torch.Tensor,
    patch_size: tuple[int, int],
    overlap: tuple[int, int],
    patch_batch_size: int,
    alpha: float,
    max_k: int,
    device: torch.device,
    lowpass: bool = False,
    lowpass_fraction: float = 0.5,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    outputs: dict[int, torch.Tensor] = {}
    residuals: dict[int, torch.Tensor] = {}
    x = x0
    for k in range(1, max_k + 1):
        z, _gfactor = g.run_sliding_window_slice(
            model,
            x,
            patch_size=patch_size,
            overlap=overlap,
            patch_batch_size=patch_batch_size,
            device=device,
        )
        residual = z - x
        if lowpass:
            residual = g.lowpass_residual(residual, lowpass_fraction)
        x = crop_domain_dc(x + alpha * residual, masked_kspace, mask)
        outputs[k] = x
        residuals[k] = residual
    return outputs, residuals


def run_methods_for_slice(
    model,
    *,
    experiment: str,
    img_lq: torch.Tensor,
    masked_kspace: torch.Tensor,
    mask: torch.Tensor,
    patch_size: tuple[int, int],
    overlap: tuple[int, int],
    patch_batch_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    one_shot, _gfactor = g.run_sliding_window_slice(
        model,
        img_lq,
        patch_size=patch_size,
        overlap=overlap,
        patch_batch_size=patch_batch_size,
        device=device,
    )
    one_shot_dc = crop_domain_dc(one_shot, masked_kspace, mask)
    methods = {"one_shot": one_shot, "one_shot_dc": one_shot_dc}
    residuals: dict[str, torch.Tensor] = {}

    if experiment == "strong":
        outputs, pnp_residuals = run_pnp_iterations(
            model,
            x0=one_shot_dc,
            masked_kspace=masked_kspace,
            mask=mask,
            patch_size=patch_size,
            overlap=overlap,
            patch_batch_size=patch_batch_size,
            alpha=0.5,
            max_k=3,
            device=device,
        )
        methods["pnp_k2_alpha0p5"] = outputs[2]
        methods["pnp_k3_alpha0p5"] = outputs[3]
        residuals = {f"residual_k{k}_alpha0p5": value for k, value in pnp_residuals.items()}
    elif experiment == "guarded":
        for alpha in (0.1, 0.2, 0.3):
            suffix = str(alpha).replace(".", "p")
            outputs, pnp_residuals = run_pnp_iterations(
                model,
                x0=one_shot_dc,
                masked_kspace=masked_kspace,
                mask=mask,
                patch_size=patch_size,
                overlap=overlap,
                patch_batch_size=patch_batch_size,
                alpha=alpha,
                max_k=2,
                device=device,
            )
            methods[f"pnp_k1_alpha{suffix}"] = outputs[1]
            methods[f"pnp_k2_alpha{suffix}"] = outputs[2]
            if abs(alpha - 0.2) < 1e-6:
                residuals["residual_k1_alpha0p2"] = pnp_residuals[1]
                residuals["residual_k2_alpha0p2"] = pnp_residuals[2]
        outputs_lp, residuals_lp = run_pnp_iterations(
            model,
            x0=one_shot_dc,
            masked_kspace=masked_kspace,
            mask=mask,
            patch_size=patch_size,
            overlap=overlap,
            patch_batch_size=patch_batch_size,
            alpha=0.2,
            max_k=2,
            device=device,
            lowpass=True,
        )
        methods["pnp_k2_alpha0p2_lowpass"] = outputs_lp[2]
        residuals["residual_k2_alpha0p2_lowpass"] = residuals_lp[2]
    elif experiment == "long":
        outputs, pnp_residuals = run_pnp_iterations(
            model,
            x0=one_shot_dc,
            masked_kspace=masked_kspace,
            mask=mask,
            patch_size=patch_size,
            overlap=overlap,
            patch_batch_size=patch_batch_size,
            alpha=0.2,
            max_k=10,
            device=device,
        )
        for k in (1, 2, 4, 6, 8, 10):
            methods[f"pnp_k{k}"] = outputs[k]
        residuals = {f"residual_k{k}": pnp_residuals[k] for k in (1, 4, 10)}
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    return methods, residuals


def residual_keys_for_experiment(experiment: str) -> list[str]:
    if experiment == "strong":
        return ["residual_k1_alpha0p5", "residual_k2_alpha0p5", "residual_k3_alpha0p5"]
    if experiment == "guarded":
        return ["residual_k1_alpha0p2", "residual_k2_alpha0p2", "residual_k2_alpha0p2_lowpass"]
    if experiment == "long":
        return ["residual_k1", "residual_k4", "residual_k10"]
    raise ValueError(f"Unknown experiment: {experiment}")


def save_comparison_png(volume_result: dict[str, Any], output_path: Path, visual_rows: list[int], methods: list[str]) -> None:
    panel_cols: list[tuple[str, str, bool]] = []
    for method in [*methods, "target"]:
        panel_cols.append((display_name(method), method, False))
        if method != "target":
            panel_cols.append((f"Err {display_name(method)}", method, True))
    fig, axes = plt.subplots(
        len(visual_rows),
        len(panel_cols),
        figsize=(2.15 * len(panel_cols), 2.65 * len(visual_rows)),
        dpi=130,
        squeeze=False,
    )
    for row_idx, slice_pos in enumerate(visual_rows):
        target = volume_result["target"][slice_pos]
        for col_idx, (title, method, is_error) in enumerate(panel_cols):
            ax = axes[row_idx][col_idx]
            image = np.abs(volume_result[method][slice_pos] - target) if is_error else volume_result[method][slice_pos]
            ax.imshow(g.minmax_image(image), cmap="magma" if is_error else "gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"slice {volume_result['slice_indices'][slice_pos]}", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_residual_maps_png(
    volume_result: dict[str, Any],
    output_path: Path,
    visual_rows: list[int],
    residual_keys: list[str],
) -> None:
    panels: list[tuple[str, str]] = [(key.replace("residual_", "Residual ").replace("_", " "), key) for key in residual_keys]
    panels.append((f"High-pass {residual_keys[-1]}", f"{residual_keys[-1]}_high"))
    fig, axes = plt.subplots(
        len(visual_rows),
        len(panels),
        figsize=(3.05 * len(panels), 2.75 * len(visual_rows)),
        dpi=130,
        squeeze=False,
    )
    for row_idx, slice_pos in enumerate(visual_rows):
        for col_idx, (title, key) in enumerate(panels):
            ax = axes[row_idx][col_idx]
            ax.imshow(g.minmax_image(volume_result[key][slice_pos]), cmap="magma", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"slice {volume_result['slice_indices'][slice_pos]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_iteration_trend_png(metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    x_values = np.asarray([0, 1, 2, 4, 6, 8, 10])
    method_for_k = {0: "one_shot_dc", 1: "pnp_k1", 2: "pnp_k2", 4: "pnp_k4", 6: "pnp_k6", 8: "pnp_k8", 10: "pnp_k10"}
    series = [("PSNR", "psnr"), ("SSIM", "ssim"), ("NMSE", "nmse"), ("DC residual", "dc_residual")]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), dpi=130, squeeze=False)
    for ax, (title, key) in zip(axes.ravel(), series, strict=True):
        values = [metrics[method_for_k[int(k)]][key] for k in x_values]
        ax.plot(x_values, values, marker="o", linewidth=1.8, label="DC + PnP")
        ax.axhline(metrics["one_shot"][key], color="tab:gray", linestyle="--", linewidth=1.0, label="one-shot")
        ax.set_title(title)
        ax.set_xlabel("PnP iteration K")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def compute_method_metrics(
    target_stack: np.ndarray,
    method_arrays: dict[str, np.ndarray],
    dc_residuals: dict[str, list[float]],
    methods: list[str],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for method in methods:
        image_metrics = g.compute_metrics(target_stack, method_arrays[method])
        metrics[method] = {
            **image_metrics,
            "dc_residual": finite_mean(dc_residuals[method]),
        }
    return metrics


def run_volume(
    *,
    model,
    dataset,
    volume_name: str,
    dataset_indices: list[int],
    experiment: str,
    patch_size: tuple[int, int],
    overlap: tuple[int, int],
    patch_batch_size: int,
    device: torch.device,
    output_dir: Path,
    slices_per_volume: int,
    slice_limit: int | None,
) -> dict[str, Any]:
    volume_dir = output_dir / volume_name
    volume_dir.mkdir(parents=True, exist_ok=True)

    if slice_limit is not None:
        dataset_indices = dataset_indices[: max(1, int(slice_limit))]

    methods = experiment_methods(experiment)
    residual_keys = residual_keys_for_experiment(experiment)
    method_stacks: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    dc_residuals: dict[str, list[float]] = {method: [] for method in methods}
    residual_stacks: dict[str, list[np.ndarray]] = {key: [] for key in residual_keys}
    residual_stacks[f"{residual_keys[-1]}_high"] = []
    targets: list[np.ndarray] = []
    slice_indices: list[int] = []

    progress = tqdm(dataset_indices, desc=volume_name, leave=False)
    with torch.inference_mode():
        for dataset_index in progress:
            img_lq, target, sample = dataset[dataset_index]
            _path, slice_idx, _metadata = dataset.given_dataset.examples[dataset_index]
            img_lq = img_lq.to(device=device, dtype=torch.float32)
            masked_kspace = sample["masked_kspace"].to(device=device, dtype=torch.float32)
            mask = sample["mask"].to(device=device, dtype=torch.float32)

            method_tensors, residual_tensors = run_methods_for_slice(
                model,
                experiment=experiment,
                img_lq=img_lq,
                masked_kspace=masked_kspace,
                mask=mask,
                patch_size=patch_size,
                overlap=overlap,
                patch_batch_size=patch_batch_size,
                device=device,
            )

            for method in methods:
                method_stacks[method].append(magnitude_from_channels(method_tensors[method]))
                dc_residuals[method].append(crop_domain_residual(method_tensors[method], masked_kspace, mask))
            for key in residual_keys:
                residual_stacks[key].append(magnitude_from_channels(residual_tensors[key]))
            residual_stacks[f"{residual_keys[-1]}_high"].append(
                g.highpass_residual_magnitude(residual_tensors[residual_keys[-1]], keep_fraction=0.5)
            )
            targets.append(target.squeeze(0).detach().cpu().float().numpy().astype(np.float32))
            slice_indices.append(int(slice_idx))

    order = np.argsort(np.asarray(slice_indices))
    slice_indices = [slice_indices[int(idx)] for idx in order]
    method_arrays = {
        method: np.stack([values[int(idx)] for idx in order], axis=0).astype(np.float32)
        for method, values in method_stacks.items()
    }
    residual_arrays = {
        key: np.stack([values[int(idx)] for idx in order], axis=0).astype(np.float32)
        for key, values in residual_stacks.items()
    }
    target_stack = np.stack([targets[int(idx)] for idx in order], axis=0).astype(np.float32)
    method_metrics = compute_method_metrics(target_stack, method_arrays, dc_residuals, methods)

    baseline = method_metrics["one_shot"]
    deltas = {
        method: {
            key: method_metrics[method][key] - baseline[key]
            for key in ("psnr", "ssim", "nmse", "dc_residual")
        }
        for method in methods
        if method != "one_shot"
    }
    rankings = sorted(methods, key=lambda method: (-method_metrics[method]["psnr"], method_metrics[method]["dc_residual"]))

    volume_result = {
        "slice_indices": slice_indices,
        "target": target_stack,
        **method_arrays,
        **residual_arrays,
    }
    visual_rows = g.choose_visual_indices(len(slice_indices), slices_per_volume)
    save_comparison_png(volume_result, volume_dir / "comparison.png", visual_rows, methods)
    save_residual_maps_png(volume_result, volume_dir / "residual_maps.png", visual_rows, residual_keys)
    if experiment == "long":
        save_iteration_trend_png(method_metrics, volume_dir / "iteration_trend.png")

    metrics_payload = {
        "volume": volume_name,
        "experiment": experiment,
        "num_slices": len(slice_indices),
        "slice_indices": slice_indices,
        "visualized_slice_indices": [slice_indices[idx] for idx in visual_rows],
        "measurement_domain": "flow_unrolled_crop_320",
        "metrics": method_metrics,
        "deltas_vs_one_shot": deltas,
        "method_ranking_by_psnr": rankings,
    }
    with (volume_dir / "metrics.json").open("w") as json_file:
        json.dump(g.json_ready(metrics_payload), json_file, indent=2)

    row: dict[str, Any] = {"volume": volume_name, "num_slices": len(slice_indices), "best_method": rankings[0]}
    for method in methods:
        for key, value in method_metrics[method].items():
            row[f"{method}_{key}"] = value
    for method, values in deltas.items():
        for key, value in values.items():
            row[f"{method}_delta_{key}"] = value
    return row


def summarize_rows(rows: list[dict[str, Any]], methods: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mean_metrics: dict[str, Any] = {}
    rankings: list[dict[str, Any]] = []
    for method in methods:
        entry = {
            key: float(np.mean([row[f"{method}_{key}"] for row in rows])) if rows else float("nan")
            for key in METRIC_KEYS
        }
        mean_metrics[method] = entry
        rankings.append({"method": method, **entry})
    rankings.sort(key=lambda row: (-row["psnr"], row["dc_residual"]))
    return mean_metrics, rankings


def write_metrics_csv(output_path: Path, rows: list[dict[str, Any]], methods: list[str]) -> None:
    fieldnames = ["volume", "num_slices", "best_method"]
    for method in methods:
        fieldnames.extend([f"{method}_{key}" for key in METRIC_KEYS])
    for method in methods:
        if method != "one_shot":
            fieldnames.extend([f"{method}_delta_psnr", f"{method}_delta_ssim", f"{method}_delta_nmse", f"{method}_delta_dc_residual"])
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_rankings_csv(output_path: Path, rankings: list[dict[str, Any]]) -> None:
    fieldnames = ["rank", "method", *METRIC_KEYS]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rankings, start=1):
            writer.writerow({"rank": idx, **{key: row.get(key) for key in fieldnames if key != "rank"}})


def write_top_level(
    *,
    output_dir: Path,
    experiment: str,
    rows: list[dict[str, Any]],
    selected_volumes: list[str],
    checkpoint: Path,
    checkpoint_epoch: Any,
    device: str,
    patch_size: tuple[int, int],
    overlap: tuple[int, int],
    patch_batch_size: int,
    load_info: Any,
) -> dict[str, Any]:
    methods = experiment_methods(experiment)
    mean_metrics, rankings = summarize_rows(rows, methods)
    summary = {
        "experiment": experiment,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "output_dir": str(output_dir),
        "device": device,
        "selected_volumes": selected_volumes,
        "num_volumes": len(selected_volumes),
        "measurement_domain": "flow_unrolled_crop_320",
        "patch_size": patch_size,
        "crop_size": (320, 320),
        "overlap": overlap,
        "eval_patch_batch_size": patch_batch_size,
        "load_info": load_info,
        "mean_metrics": mean_metrics,
        "method_rankings": rankings,
        "volumes": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_volumes.txt").write_text("\n".join(selected_volumes) + "\n")
    write_metrics_csv(output_dir / "metrics.csv", rows, methods)
    write_rankings_csv(output_dir / "method_rankings.csv", rankings)
    (output_dir / "summary.json").write_text(json.dumps(g.json_ready(summary), indent=2))
    return summary


def main() -> int:
    args = parse_args()
    configure_paths(args.repo_root, args.flow_root)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")

    device = g.resolve_device(args.device)
    print(f"Using device: {device}", flush=True)
    model, config, checkpoint, load_info = g.load_model_from_checkpoint(args.checkpoint, device)
    ft_cfg = config.fastmri_finetune
    dataset = build_crop_domain_dataset(args.val_root, acc_factor=int(ft_cfg.acc_factor))
    selected_volumes, indices_by_volume = select_volume_indices(dataset, args.volume_names, args.num_volumes)

    patch_size = tuple(int(dim) for dim in ft_cfg.train_patch_size)
    overlap_values = config.get("overlap_for_inference", [16, 16, 0])
    overlap = (int(overlap_values[0]), int(overlap_values[1]))
    patch_batch_size = int(args.eval_patch_batch_size or ft_cfg.get("eval_patch_batch_size", 64))
    output_dir = args.output_dir if args.output_dir is not None else experiment_output_dir(args.experiment)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_volumes.txt").write_text("\n".join(selected_volumes) + "\n")

    rows: list[dict[str, Any]] = []
    for volume_name in tqdm(selected_volumes, desc=f"{args.experiment} volumes"):
        rows.append(
            run_volume(
                model=model,
                dataset=dataset,
                volume_name=volume_name,
                dataset_indices=indices_by_volume[volume_name],
                experiment=args.experiment,
                patch_size=patch_size,
                overlap=overlap,
                patch_batch_size=patch_batch_size,
                device=device,
                output_dir=output_dir,
                slices_per_volume=args.slices_per_volume,
                slice_limit=args.slice_limit,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = write_top_level(
        output_dir=output_dir,
        experiment=args.experiment,
        rows=rows,
        selected_volumes=selected_volumes,
        checkpoint=args.checkpoint,
        checkpoint_epoch=checkpoint.get("epoch"),
        device=str(device),
        patch_size=patch_size,
        overlap=overlap,
        patch_batch_size=patch_batch_size,
        load_info=load_info,
    )
    print(f"Finished crop-domain {args.experiment} PnP.", flush=True)
    print(json.dumps(g.json_ready({"top_ranked": summary["method_rankings"][:5]}), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
