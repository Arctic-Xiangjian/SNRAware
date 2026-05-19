#!/usr/bin/env python
"""Run long alpha=0.2 guarded PnP iterations for FastMRI visual checks."""

from __future__ import annotations

import argparse
import csv
import importlib.util
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
from tqdm.auto import tqdm


HELPER_PATH = Path(__file__).resolve().parent / "run_pnp_guarded_sweep.py"
DEFAULT_REPO_ROOT = Path("/working2/arctic/snrawre/SNRAware")
DEFAULT_CHECKPOINT = (
    DEFAULT_REPO_ROOT / "checkpoints/fine_tune/warmup_then_both_20260517-145527/last.pth"
)
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "visual_check" / "PnP_long_a0p2"
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
ALPHA = 0.2
DEFAULT_REPORT_KS = (1, 2, 4, 6, 8, 10)
METRIC_KEYS = ("psnr", "ssim", "nmse", "dc_residual", "heldout_residual")


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
    parser = argparse.ArgumentParser(description="Run long alpha=0.2 guarded PnP.")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-volumes", type=int, default=10)
    parser.add_argument("--volume-names", nargs="*", default=None)
    parser.add_argument("--slices-per-volume", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-patch-batch-size", type=int, default=None)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--report-ks", nargs="*", type=int, default=list(DEFAULT_REPORT_KS))
    parser.add_argument(
        "--skip-top-level",
        action="store_true",
        help="Only write per-volume outputs; useful for parallel workers sharing one output dir.",
    )
    parser.add_argument("--worker-summary", type=Path, default=None)
    parser.add_argument(
        "--merge-worker-summaries",
        nargs="*",
        type=Path,
        default=None,
        help="Merge worker summary JSON files and exit without loading a model.",
    )
    return parser.parse_args()


def pnp_method(k: int) -> str:
    return f"pnp_k{k}"


def report_methods(report_ks: list[int]) -> list[str]:
    return ["one_shot", "one_shot_dc", *[pnp_method(k) for k in report_ks]]


def display_name(method: str) -> str:
    if method == "one_shot":
        return "One-shot"
    if method == "one_shot_dc":
        return "One-shot+DC"
    if method.startswith("pnp_k"):
        return method.replace("pnp_k", "K")
    return method


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def choose_residual_ks(max_k: int) -> list[int]:
    return sorted(set(k for k in (1, 4, max_k) if 1 <= k <= max_k))


def run_long_pnp_slice(
    model,
    *,
    one_shot_dc: torch.Tensor,
    observed_kspace: torch.Tensor,
    dc_mask: torch.Tensor,
    crop_size: tuple[int, int],
    patch_size: tuple[int, int],
    overlap: tuple[int, int],
    patch_batch_size: int,
    alpha: float,
    max_k: int,
    report_ks: set[int],
    residual_ks: set[int],
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    report_outputs: dict[int, torch.Tensor] = {}
    report_fulls: dict[int, torch.Tensor] = {}
    all_outputs: dict[int, torch.Tensor] = {}
    all_fulls: dict[int, torch.Tensor] = {}
    residuals: dict[int, torch.Tensor] = {}
    x = one_shot_dc
    for k in range(1, max_k + 1):
        z, _g = g.run_sliding_window_slice(
            model,
            x,
            patch_size=patch_size,
            overlap=overlap,
            patch_batch_size=patch_batch_size,
            device=device,
        )
        residual = z - x
        x, full = g.apply_full_fov_dc(
            x + alpha * residual,
            observed_kspace=observed_kspace,
            mask=dc_mask,
            crop_size=crop_size,
        )
        all_outputs[k] = x
        all_fulls[k] = full
        if k in report_ks:
            report_outputs[k] = x
            report_fulls[k] = full
        if k in residual_ks:
            residuals[k] = residual
    return report_outputs, report_fulls, all_outputs, all_fulls, residuals


def save_comparison_png(
    volume_result: dict[str, Any],
    output_path: Path,
    visual_rows: list[int],
    columns: list[str],
) -> None:
    panel_cols: list[tuple[str, str, bool]] = []
    for method in [*columns, "target"]:
        panel_cols.append((display_name(method), method, False))
        if method != "target":
            panel_cols.append((f"Err {display_name(method)}", method, True))
    fig, axes = plt.subplots(
        len(visual_rows),
        len(panel_cols),
        figsize=(2.2 * len(panel_cols), 2.65 * len(visual_rows)),
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
    residual_ks: list[int],
) -> None:
    panels: list[tuple[str, str]] = []
    for k in residual_ks:
        panels.append((f"Residual K{k}", f"residual_mag_k{k}"))
    panels.append((f"High-pass K{residual_ks[-1]}", f"residual_high_k{residual_ks[-1]}"))
    fig, axes = plt.subplots(
        len(visual_rows),
        len(panels),
        figsize=(3.0 * len(panels), 2.75 * len(visual_rows)),
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
                ax.set_title(title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"slice {volume_result['slice_indices'][slice_pos]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_iteration_trend_png(
    metrics: dict[str, dict[str, float]],
    output_path: Path,
    max_k: int,
) -> None:
    x_values = np.arange(0, max_k + 1)
    method_for_k = {0: "one_shot_dc", **{k: pnp_method(k) for k in range(1, max_k + 1)}}
    series = [
        ("PSNR", "psnr"),
        ("SSIM", "ssim"),
        ("NMSE", "nmse"),
        ("Held-out k-space", "heldout_residual"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), dpi=130, squeeze=False)
    for ax, (title, key) in zip(axes.ravel(), series, strict=True):
        values = [metrics[method_for_k[k]][key] for k in x_values]
        ax.plot(x_values, values, marker="o", linewidth=1.8, label="DC + PnP")
        ax.axhline(metrics["one_shot"][key], color="tab:gray", linestyle="--", linewidth=1.0, label="one-shot")
        ax.set_title(title)
        ax.set_xlabel("PnP iteration K")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
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
    heldout_fraction: float,
    alpha: float,
    max_k: int,
    report_ks: list[int],
    device: torch.device,
    output_dir: Path,
    slices_per_volume: int,
) -> dict[str, Any]:
    volume_dir = output_dir / volume_name
    volume_dir.mkdir(parents=True, exist_ok=True)

    methods_for_metrics = ["one_shot", "one_shot_dc", *[pnp_method(k) for k in range(1, max_k + 1)]]
    methods_for_report = report_methods(report_ks)
    residual_ks = choose_residual_ks(max_k)

    method_stacks: dict[str, list[np.ndarray]] = {method: [] for method in methods_for_metrics}
    dc_residuals: dict[str, list[float]] = {method: [] for method in methods_for_metrics}
    heldout_residuals: dict[str, list[float]] = {method: [] for method in methods_for_metrics}
    residual_arrays_by_key: dict[str, list[np.ndarray]] = {
        **{f"residual_mag_k{k}": [] for k in residual_ks},
        f"residual_high_k{residual_ks[-1]}": [],
    }
    targets: list[np.ndarray] = []
    slice_indices: list[int] = []

    report_set = set(report_ks)
    residual_set = set(residual_ks)

    progress = tqdm(dataset_indices, desc=volume_name, leave=False)
    with torch.inference_mode():
        for dataset_index in progress:
            noisy, clean, _noise_sigma, metadata = dataset[dataset_index]
            path, slice_idx, _example_metadata = dataset.slice_dataset.examples[dataset_index]
            with h5py.File(path, "r") as hf:
                raw_kspace = np.asarray(hf["kspace"][slice_idx])

            observed_kspace, dc_mask, val_mask = g.make_masks(
                raw_kspace=raw_kspace,
                file_name=Path(path).name,
                slice_idx=int(slice_idx),
                acc_factor=acc_factor,
                heldout_fraction=heldout_fraction,
                device=device,
            )

            _ = noisy
            zero_full = g.ifft2c_complex(torch.where(dc_mask, observed_kspace, torch.zeros_like(observed_kspace)))
            zero_crop = g.complex_to_channels(g.center_crop_complex(zero_full, crop_size)).to(dtype=torch.float32)
            one_shot, _g = g.run_sliding_window_slice(
                model,
                zero_crop,
                patch_size=patch_size,
                overlap=overlap,
                patch_batch_size=patch_batch_size,
                device=device,
            )
            one_shot_full = g.crop_to_full_complex(one_shot, tuple(observed_kspace.shape[-2:]))
            one_shot_dc, one_shot_dc_full = g.apply_full_fov_dc(
                one_shot,
                observed_kspace=observed_kspace,
                mask=dc_mask,
                crop_size=crop_size,
            )
            report_outputs, report_fulls, all_outputs, all_fulls, residuals = run_long_pnp_slice(
                model,
                one_shot_dc=one_shot_dc,
                observed_kspace=observed_kspace,
                dc_mask=dc_mask,
                crop_size=crop_size,
                patch_size=patch_size,
                overlap=overlap,
                patch_batch_size=patch_batch_size,
                alpha=alpha,
                max_k=max_k,
                report_ks=report_set,
                residual_ks=residual_set,
                device=device,
            )

            method_tensors: dict[str, torch.Tensor] = {
                "one_shot": one_shot,
                "one_shot_dc": one_shot_dc,
                **{pnp_method(k): all_outputs[k] for k in range(1, max_k + 1)},
            }
            full_tensors: dict[str, torch.Tensor] = {
                "one_shot": one_shot_full,
                "one_shot_dc": one_shot_dc_full,
                **{pnp_method(k): all_fulls[k] for k in range(1, max_k + 1)},
            }

            for method in methods_for_metrics:
                method_stacks[method].append(g.magnitude_from_complex_channels(method_tensors[method]))
                dc_residuals[method].append(g.kspace_relative_residual(full_tensors[method], observed_kspace, dc_mask))
                heldout_residuals[method].append(g.kspace_relative_residual(full_tensors[method], observed_kspace, val_mask))

            for k in residual_ks:
                residual_arrays_by_key[f"residual_mag_k{k}"].append(g.magnitude_from_complex_channels(residuals[k]))
            residual_arrays_by_key[f"residual_high_k{residual_ks[-1]}"].append(
                g.highpass_residual_magnitude(residuals[residual_ks[-1]], keep_fraction=0.5)
            )

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
        key: np.stack([values[int(idx)] for idx in order], axis=0).astype(np.float32)
        for key, values in residual_arrays_by_key.items()
    }

    method_metrics: dict[str, dict[str, float]] = {}
    for method in methods_for_metrics:
        image_metrics = g.compute_metrics(target_stack, method_arrays[method])
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
        for method in methods_for_metrics
        if method != "one_shot"
    }

    volume_result = {
        "slice_indices": slice_indices,
        "target": target_stack,
        **{method: method_arrays[method] for method in methods_for_report},
        **residual_arrays,
    }
    visual_rows = g.choose_visual_indices(len(slice_indices), slices_per_volume)
    save_comparison_png(volume_result, volume_dir / "comparison.png", visual_rows, methods_for_report)
    save_residual_maps_png(volume_result, volume_dir / "residual_maps.png", visual_rows, residual_ks)
    save_iteration_trend_png(method_metrics, volume_dir / "iteration_trend.png", max_k)

    ranked_report = sorted(
        methods_for_report,
        key=lambda name: (method_metrics[name]["heldout_residual"], -method_metrics[name]["psnr"]),
    )
    metrics_payload = {
        "volume": volume_name,
        "num_slices": len(slice_indices),
        "slice_indices": slice_indices,
        "visualized_slice_indices": [slice_indices[idx] for idx in visual_rows],
        "alpha": alpha,
        "max_k": max_k,
        "report_ks": report_ks,
        "residual_ks": residual_ks,
        "metrics": method_metrics,
        "report_metrics": {method: method_metrics[method] for method in methods_for_report},
        "deltas_vs_one_shot": deltas,
        "best_report_method_by_heldout_residual": ranked_report[0],
        "report_method_ranking": ranked_report,
    }
    with (volume_dir / "metrics.json").open("w") as json_file:
        json.dump(g.json_ready(metrics_payload), json_file, indent=2)

    row: dict[str, Any] = {"volume": volume_name, "num_slices": len(slice_indices), "best_method": ranked_report[0]}
    for method in methods_for_report:
        for key, value in method_metrics[method].items():
            row[f"{method}_{key}"] = value
    for method in methods_for_report:
        if method == "one_shot":
            continue
        for key, value in deltas[method].items():
            row[f"{method}_delta_{key}"] = value
    row["all_iteration_metrics"] = method_metrics
    return row


def summarize_rows(rows: list[dict[str, Any]], report_ks: list[int], max_k: int) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    methods_for_report = report_methods(report_ks)
    mean_metrics: dict[str, Any] = {}
    rankings: list[dict[str, Any]] = []
    for method in methods_for_report:
        entry = {
            key: float(np.mean([row[f"{method}_{key}"] for row in rows])) if rows else float("nan")
            for key in METRIC_KEYS
        }
        mean_metrics[method] = entry
        rankings.append({"method": method, **entry})
    rankings.sort(key=lambda row: (row["heldout_residual"], -row["psnr"]))

    all_iteration_mean_metrics: dict[str, Any] = {}
    for method in ["one_shot", "one_shot_dc", *[pnp_method(k) for k in range(1, max_k + 1)]]:
        all_iteration_mean_metrics[method] = {
            key: float(np.mean([row["all_iteration_metrics"][method][key] for row in rows])) if rows else float("nan")
            for key in METRIC_KEYS
        }
    return mean_metrics, all_iteration_mean_metrics, rankings


def write_metrics_csv(output_path: Path, rows: list[dict[str, Any]], report_ks: list[int]) -> None:
    methods_for_report = report_methods(report_ks)
    fieldnames = ["volume", "num_slices", "best_method"]
    for method in methods_for_report:
        fieldnames.extend([f"{method}_{key}" for key in METRIC_KEYS])
    for method in methods_for_report:
        if method != "one_shot":
            fieldnames.extend([f"{method}_delta_psnr", f"{method}_delta_ssim", f"{method}_delta_nmse", f"{method}_delta_heldout_residual"])
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


def load_reference_summaries(repo_root: Path) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for name, rel in {
        "pnp_guarded": "visual_check/PnP_guarded/summary.json",
        "pnp_strong": "visual_check/PnP/summary.json",
    }.items():
        path = repo_root / rel
        refs[name] = json.loads(path.read_text()).get("mean_metrics") if path.exists() else None
    return refs


def write_top_level_outputs(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    selected_volumes: list[str],
    report_ks: list[int],
    max_k: int,
    alpha: float,
    checkpoint: Path,
    checkpoint_epoch: Any = None,
    device: str | None = None,
    patch_size: tuple[int, int] | None = None,
    crop_size: tuple[int, int] | None = None,
    overlap: tuple[int, int] | None = None,
    eval_patch_batch_size: int | None = None,
    heldout_fraction: float | None = None,
    load_info: Any = None,
    repo_root: Path = DEFAULT_REPO_ROOT,
    worker_summaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    mean_metrics, all_iteration_mean_metrics, rankings = summarize_rows(rows, report_ks, max_k)
    references = load_reference_summaries(repo_root)
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "output_dir": str(output_dir),
        "device": device,
        "selected_volumes": selected_volumes,
        "num_volumes": len(selected_volumes),
        "alpha": alpha,
        "max_k": max_k,
        "report_ks": report_ks,
        "patch_size": patch_size,
        "crop_size": crop_size,
        "overlap": overlap,
        "eval_patch_batch_size": eval_patch_batch_size,
        "heldout_fraction": heldout_fraction,
        "load_info": load_info,
        "mean_metrics": mean_metrics,
        "all_iteration_mean_metrics": all_iteration_mean_metrics,
        "method_rankings": rankings,
        "reference_mean_metrics": references,
        "volumes": rows,
        "worker_summaries": worker_summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_volumes.txt").write_text("\n".join(selected_volumes) + "\n")
    write_metrics_csv(output_dir / "metrics.csv", rows, report_ks)
    write_rankings_csv(output_dir / "method_rankings.csv", rankings)
    (output_dir / "summary.json").write_text(json.dumps(g.json_ready(summary), indent=2))
    return summary


def merge_worker_summaries(args: argparse.Namespace) -> int:
    summaries = [json.loads(path.read_text()) for path in args.merge_worker_summaries]
    rows: list[dict[str, Any]] = []
    selected_volumes: list[str] = []
    for summary in summaries:
        selected_volumes.extend(summary["selected_volumes"])
        rows.extend(summary["volumes"])
    order = {name: idx for idx, name in enumerate(DEFAULT_VOLUMES)}
    selected_volumes = sorted(selected_volumes, key=lambda name: order.get(name, len(order)))
    rows = sorted(rows, key=lambda row: order.get(row["volume"], len(order)))
    first = summaries[0]
    summary = write_top_level_outputs(
        output_dir=args.output_dir,
        rows=rows,
        selected_volumes=selected_volumes,
        report_ks=first["report_ks"],
        max_k=int(first["max_k"]),
        alpha=float(first["alpha"]),
        checkpoint=Path(first["checkpoint"]),
        checkpoint_epoch=first.get("checkpoint_epoch"),
        device=",".join(str(summary.get("device")) for summary in summaries),
        patch_size=first.get("patch_size"),
        crop_size=first.get("crop_size"),
        overlap=first.get("overlap"),
        eval_patch_batch_size=first.get("eval_patch_batch_size"),
        heldout_fraction=first.get("heldout_fraction"),
        load_info=first.get("load_info"),
        repo_root=args.repo_root,
        worker_summaries=summaries,
    )
    print(json.dumps(g.json_ready({"top_ranked": summary["method_rankings"][:5]}), indent=2), flush=True)
    return 0


def normalize_report_ks(report_ks: list[int], max_k: int) -> list[int]:
    normalized = sorted(set(k for k in report_ks if 1 <= int(k) <= max_k))
    if not normalized:
        raise ValueError(f"No valid report ks for max_k={max_k}: {report_ks}")
    return normalized


def main() -> int:
    args = parse_args()
    if args.merge_worker_summaries is not None:
        return merge_worker_summaries(args)

    g.configure_imports(args.repo_root)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")

    if args.max_k < 1:
        raise ValueError("--max-k must be >= 1")
    report_ks = normalize_report_ks(args.report_ks, args.max_k)

    device = g.resolve_device(args.device)
    print(f"Using device: {device}", flush=True)
    model, config, checkpoint, load_info = g.load_model_from_checkpoint(args.checkpoint, device)
    dataset = g.build_validation_dataset(config)
    selected_volumes, indices_by_volume = g.select_volume_indices(dataset, args.volume_names, args.num_volumes)

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
                alpha=float(args.alpha),
                max_k=int(args.max_k),
                report_ks=report_ks,
                device=device,
                output_dir=args.output_dir,
                slices_per_volume=args.slices_per_volume,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_metrics, all_iteration_mean_metrics, rankings = summarize_rows(rows, report_ks, int(args.max_k))
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "selected_volumes": selected_volumes,
        "num_volumes": len(selected_volumes),
        "alpha": float(args.alpha),
        "max_k": int(args.max_k),
        "report_ks": report_ks,
        "patch_size": patch_size,
        "crop_size": crop_size,
        "overlap": overlap,
        "eval_patch_batch_size": patch_batch_size,
        "heldout_fraction": float(args.heldout_fraction),
        "load_info": load_info,
        "mean_metrics": mean_metrics,
        "all_iteration_mean_metrics": all_iteration_mean_metrics,
        "method_rankings": rankings,
        "reference_mean_metrics": load_reference_summaries(args.repo_root),
        "volumes": rows,
    }
    if args.skip_top_level:
        if args.worker_summary is not None:
            args.worker_summary.parent.mkdir(parents=True, exist_ok=True)
            args.worker_summary.write_text(json.dumps(g.json_ready(summary), indent=2))
    else:
        write_top_level_outputs(
            output_dir=args.output_dir,
            rows=rows,
            selected_volumes=selected_volumes,
            report_ks=report_ks,
            max_k=int(args.max_k),
            alpha=float(args.alpha),
            checkpoint=args.checkpoint,
            checkpoint_epoch=checkpoint.get("epoch"),
            device=str(device),
            patch_size=patch_size,
            crop_size=crop_size,
            overlap=overlap,
            eval_patch_batch_size=patch_batch_size,
            heldout_fraction=float(args.heldout_fraction),
            load_info=load_info,
            repo_root=args.repo_root,
        )

    print("Finished long alpha=0.2 PnP.", flush=True)
    print(json.dumps(g.json_ready({"top_ranked": rankings[:5]}), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
