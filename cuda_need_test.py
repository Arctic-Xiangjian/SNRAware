"""CPU RAM estimator and CUDA training-VRAM profiler for FastMRI wrapped SNRAware."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from snraware.projects.mri.denoising.base_model_resolver import resolve_base_model_paths
from snraware.projects.mri.denoising.fastmri_compat import build_fastmri_wrapped_model
from snraware.projects.mri.denoising.trainer_fa import (
    build_fastmri_optimizer,
    complex_output_to_magnitude,
    configure_model_for_finetune_mode,
    fastmri_autocast_context,
    resolve_fastmri_precision,
    should_checkpoint_frozen_base,
)

PROFILE_TARGETS = ("cpu_infer_estimate", "cuda_train_peak", "both")
TRAIN_MODES = ("unet_only", "unet_and_lora", "lora_only", "warmup_then_both")
FASTMRI_CONFIG_PATH = Path("src/snraware/projects/mri/denoising/configs/fastmri_finetune.yaml")


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU RAM estimator and CUDA training-VRAM profiler for FastMRI wrapped SNRAware models"
    )
    parser.add_argument("--profile_target", choices=PROFILE_TARGETS, default="both")
    parser.add_argument("--model_size", choices=("small", "large"), default="small")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--weight_path", type=str, default=None)
    parser.add_argument("--train_mode", choices=TRAIN_MODES, default="unet_and_lora")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--crop_size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_bf16", type=_parse_optional_bool, default=None)
    parser.add_argument("--gradient_checkpoint_frozen_base", type=_parse_optional_bool, default=None)
    parser.add_argument("--report_json", type=str, default=None)
    return parser.parse_args()



def _rss_bytes() -> int:
    process = psutil.Process(os.getpid())
    return int(process.memory_info().rss)



def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.name == "posix" and os.uname().sysname == "Darwin":
        return int(usage)
    return int(usage) * 1024



def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()



def _bytes_to_gb(value: int | float) -> float:
    return float(value) / (1024.0**3)



def _repo_root() -> Path:
    return Path(__file__).resolve().parent



def _load_fastmri_runtime_config(repo_root: Path | None = None) -> DictConfig:
    root = repo_root or _repo_root()
    return OmegaConf.load(root / FASTMRI_CONFIG_PATH)



def _resolve_runtime_paths(
    *,
    model_size: str,
    config_path: str | None,
    weight_path: str | None,
    repo_root: Path | None = None,
) -> tuple[Path, str, str]:
    root = repo_root or _repo_root()
    resolved_config_path, resolved_weight_path = resolve_base_model_paths(
        variant=model_size,
        config_path=config_path,
        checkpoint_path=weight_path,
        repo_root=root,
    )
    return root, resolved_config_path, resolved_weight_path



def _resolved_bool(value: bool | None, default: bool) -> bool:
    return default if value is None else bool(value)



def _gfactor_unet_config(runtime_config: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(runtime_config.fastmri_finetune.gfactor_unet, resolve=True)



def _lora_config(runtime_config: DictConfig) -> Any:
    return runtime_config.get("lora")



def _collect_static_stats(model: torch.nn.Module, dummy_input: torch.Tensor, output: torch.Tensor) -> dict[str, Any]:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    buffer_count = sum(buffer.numel() for buffer in model.buffers())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    input_bytes = _tensor_bytes(dummy_input)
    output_bytes = _tensor_bytes(output)

    return {
        "parameter_count": int(parameter_count),
        "parameter_bytes": int(parameter_bytes),
        "buffer_count": int(buffer_count),
        "buffer_bytes": int(buffer_bytes),
        "input_bytes": int(input_bytes),
        "output_bytes": int(output_bytes),
        "model_load_estimate_bytes": int(parameter_bytes + buffer_bytes),
        "forward_resident_estimate_bytes": int(parameter_bytes + buffer_bytes + input_bytes + output_bytes),
    }



def _build_wrapped_model(
    *,
    resolved_config_path: str,
    resolved_weight_path: str,
    crop_size: tuple[int, int],
    gfactor_unet_kwargs: dict[str, Any],
    lora_config: Any | None,
):
    return build_fastmri_wrapped_model(
        base_config_path=resolved_config_path,
        base_checkpoint_path=resolved_weight_path,
        height=int(crop_size[0]),
        width=int(crop_size[1]),
        depth=1,
        lora_config=lora_config,
        gfactor_unet_kwargs=gfactor_unet_kwargs,
    )



def estimate_fastmri_wrapped_cpu_infer_memory(
    *,
    model_size: str,
    config_path: str | None,
    weight_path: str | None,
    crop_size: tuple[int, int],
    batch_size: int,
) -> dict[str, Any]:
    repo_root, resolved_config_path, resolved_weight_path = _resolve_runtime_paths(
        model_size=model_size,
        config_path=config_path,
        weight_path=weight_path,
    )
    runtime_config = _load_fastmri_runtime_config(repo_root)

    rss_before_build = _rss_bytes()
    peak_before_build = _peak_rss_bytes()

    model, _base_config, load_info = _build_wrapped_model(
        resolved_config_path=resolved_config_path,
        resolved_weight_path=resolved_weight_path,
        crop_size=crop_size,
        gfactor_unet_kwargs=_gfactor_unet_config(runtime_config),
        lora_config=None,
    )
    model = model.to(device=torch.device("cpu"), dtype=torch.float32)
    model.eval()

    rss_after_build = _rss_bytes()
    peak_after_build = _peak_rss_bytes()

    dummy_input = torch.randn(batch_size, 2, int(crop_size[0]), int(crop_size[1]), dtype=torch.float32)
    rss_after_input = _rss_bytes()
    peak_after_input = _peak_rss_bytes()

    with torch.inference_mode():
        output = model(dummy_input)

    rss_after_forward = _rss_bytes()
    peak_after_forward = _peak_rss_bytes()

    static_stats = _collect_static_stats(model, dummy_input, output)
    checkpoint_file_bytes = Path(resolved_weight_path).stat().st_size

    return {
        "mode": "fastmri_wrapped_cpu_infer_estimate",
        "note": "CPU load/inference RAM estimate only; this is not a training-time CUDA VRAM number.",
        "model_size": model_size,
        "resolved_config_path": resolved_config_path,
        "resolved_weight_path": resolved_weight_path,
        "crop_size": [int(crop_size[0]), int(crop_size[1])],
        "batch_size": int(batch_size),
        "checkpoint_file_bytes": int(checkpoint_file_bytes),
        "checkpoint_file_gb": _bytes_to_gb(int(checkpoint_file_bytes)),
        "load_info": load_info,
        "static": {
            **static_stats,
            "parameter_gb": _bytes_to_gb(static_stats["parameter_bytes"]),
            "buffer_gb": _bytes_to_gb(static_stats["buffer_bytes"]),
            "input_gb": _bytes_to_gb(static_stats["input_bytes"]),
            "output_gb": _bytes_to_gb(static_stats["output_bytes"]),
            "model_load_estimate_gb": _bytes_to_gb(static_stats["model_load_estimate_bytes"]),
            "forward_resident_estimate_gb": _bytes_to_gb(static_stats["forward_resident_estimate_bytes"]),
        },
        "measured_rss": {
            "before_build_bytes": int(rss_before_build),
            "after_build_bytes": int(rss_after_build),
            "after_input_bytes": int(rss_after_input),
            "after_forward_bytes": int(rss_after_forward),
            "peak_before_build_bytes": int(peak_before_build),
            "peak_after_build_bytes": int(peak_after_build),
            "peak_after_input_bytes": int(peak_after_input),
            "peak_after_forward_bytes": int(peak_after_forward),
            "build_delta_bytes": int(rss_after_build - rss_before_build),
            "input_delta_bytes": int(rss_after_input - rss_after_build),
            "forward_delta_bytes": int(rss_after_forward - rss_after_input),
            "peak_delta_from_start_bytes": int(max(peak_after_forward, rss_after_forward) - rss_before_build),
            "build_delta_gb": _bytes_to_gb(int(rss_after_build - rss_before_build)),
            "input_delta_gb": _bytes_to_gb(int(rss_after_input - rss_after_build)),
            "forward_delta_gb": _bytes_to_gb(int(rss_after_forward - rss_after_input)),
            "peak_delta_from_start_gb": _bytes_to_gb(int(max(peak_after_forward, rss_after_forward) - rss_before_build)),
        },
    }



def estimate_fastmri_wrapped_ram(
    *,
    model_size: str,
    config_path: str | None,
    weight_path: str | None,
    crop_size: tuple[int, int],
    batch_size: int,
) -> dict[str, Any]:
    """Backward-compatible alias for the CPU inference estimate."""
    return estimate_fastmri_wrapped_cpu_infer_memory(
        model_size=model_size,
        config_path=config_path,
        weight_path=weight_path,
        crop_size=crop_size,
        batch_size=batch_size,
    )



def _require_cuda_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError(
            "cuda_train_peak profiling requires a CUDA device, "
            f"but got device={device}. Use --device cuda:0 or switch to --profile_target cpu_infer_estimate."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "cuda_train_peak profiling requires CUDA, but torch.cuda.is_available() is False. "
            "Use --profile_target cpu_infer_estimate for the offline CPU estimate."
        )



def _safe_cuda_sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _set_cuda_device(device: torch.device) -> None:
    if device.index is not None:
        torch.cuda.set_device(device)


def _clear_cuda_case(device: torch.device) -> None:
    _set_cuda_device(device)
    gc.collect()
    torch.cuda.empty_cache()
    _safe_cuda_sync(device)



def _cuda_memory_snapshot(device: torch.device) -> dict[str, Any]:
    _safe_cuda_sync(device)
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "allocated_gb": _bytes_to_gb(allocated),
        "reserved_gb": _bytes_to_gb(reserved),
        "peak_allocated_gb": _bytes_to_gb(peak_allocated),
        "peak_reserved_gb": _bytes_to_gb(peak_reserved),
    }



def _trainable_parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        total += int(parameter.numel())
        if parameter.requires_grad:
            trainable += int(parameter.numel())
    return {"trainable_parameters": trainable, "total_parameters": total}



def _dummy_clean_target(batch_size: int, crop_size: tuple[int, int], device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, 1, int(crop_size[0]), int(crop_size[1]), device=device, dtype=torch.float32)



def _profile_single_cuda_train_case(
    *,
    case_name: str,
    model_size: str,
    resolved_config_path: str,
    resolved_weight_path: str,
    crop_size: tuple[int, int],
    batch_size: int,
    device: torch.device,
    train_mode: str,
    runtime_config: DictConfig,
    use_bf16: bool,
    gradient_checkpoint_frozen_base: bool,
    adapters_active: bool | None,
) -> dict[str, Any]:
    _require_cuda_device(device)
    _set_cuda_device(device)
    _clear_cuda_case(device)
    torch.cuda.reset_peak_memory_stats(device)

    lora_config = _lora_config(runtime_config)
    model, _base_config, load_info = _build_wrapped_model(
        resolved_config_path=resolved_config_path,
        resolved_weight_path=resolved_weight_path,
        crop_size=crop_size,
        gfactor_unet_kwargs=_gfactor_unet_config(runtime_config),
        lora_config=lora_config,
    )
    model = model.to(device=device, dtype=torch.float32)
    model.train()

    mode_state = configure_model_for_finetune_mode(
        model,
        mode=train_mode,
        lora_config=lora_config,
        adapters_active=adapters_active,
    )
    optimizer = build_fastmri_optimizer(model, runtime_config.fastmri_finetune, mode_state)
    precision_state = resolve_fastmri_precision(device, use_bf16=use_bf16)
    loss_fn = nn.L1Loss()

    snapshots = {
        "after_model_load": _cuda_memory_snapshot(device),
    }

    noisy = torch.randn(
        batch_size,
        2,
        int(crop_size[0]),
        int(crop_size[1]),
        device=device,
        dtype=torch.float32,
    )
    clean = _dummy_clean_target(batch_size, crop_size, device)
    snapshots["after_input_transfer"] = _cuda_memory_snapshot(device)

    optimizer.zero_grad(set_to_none=True)
    checkpoint_base_model = should_checkpoint_frozen_base(
        model,
        gradient_checkpoint_frozen_base=gradient_checkpoint_frozen_base,
    )
    with fastmri_autocast_context(device, enabled=bool(precision_state["use_bf16"])):
        output = model(noisy, checkpoint_base_model=checkpoint_base_model)
        magnitude_output = complex_output_to_magnitude(output)
    snapshots["after_forward"] = _cuda_memory_snapshot(device)

    magnitude_output = magnitude_output.float()
    loss = loss_fn(magnitude_output, clean)
    snapshots["after_loss"] = _cuda_memory_snapshot(device)

    loss.backward()
    snapshots["after_backward"] = _cuda_memory_snapshot(device)

    optimizer.step()
    snapshots["after_optimizer_step"] = _cuda_memory_snapshot(device)

    final_snapshot = snapshots["after_optimizer_step"]
    report = {
        "case_name": case_name,
        "model_size": model_size,
        "train_mode": train_mode,
        "adapters_active": bool(mode_state["adapters_active"]),
        "has_lora": bool(mode_state["has_lora"]),
        "checkpoint_base_model": bool(checkpoint_base_model),
        "precision_mode": str(precision_state["mode"]),
        "use_bf16": bool(precision_state["use_bf16"]),
        "gradient_checkpoint_frozen_base": bool(gradient_checkpoint_frozen_base),
        "resolved_config_path": resolved_config_path,
        "resolved_weight_path": resolved_weight_path,
        "crop_size": [int(crop_size[0]), int(crop_size[1])],
        "batch_size": int(batch_size),
        "loss": float(loss.detach().item()),
        "load_info": load_info,
        "snapshots": snapshots,
        "headline": {
            "peak_allocated_bytes": int(final_snapshot["peak_allocated_bytes"]),
            "peak_reserved_bytes": int(final_snapshot["peak_reserved_bytes"]),
            "peak_allocated_gb": _bytes_to_gb(final_snapshot["peak_allocated_bytes"]),
            "peak_reserved_gb": _bytes_to_gb(final_snapshot["peak_reserved_bytes"]),
        },
        **_trainable_parameter_counts(model),
    }

    del loss, magnitude_output, output, clean, noisy, optimizer, model
    _clear_cuda_case(device)
    return report



def profile_fastmri_wrapped_cuda_train_peak(
    *,
    model_size: str,
    config_path: str | None,
    weight_path: str | None,
    crop_size: tuple[int, int],
    batch_size: int,
    device: str | torch.device,
    train_mode: str,
    use_bf16: bool | None,
    gradient_checkpoint_frozen_base: bool | None,
) -> dict[str, Any]:
    runtime_device = torch.device(device)
    _require_cuda_device(runtime_device)

    repo_root, resolved_config_path, resolved_weight_path = _resolve_runtime_paths(
        model_size=model_size,
        config_path=config_path,
        weight_path=weight_path,
    )
    runtime_config = _load_fastmri_runtime_config(repo_root)
    resolved_use_bf16 = _resolved_bool(use_bf16, bool(runtime_config.fastmri_finetune.use_bf16))
    resolved_checkpointing = _resolved_bool(
        gradient_checkpoint_frozen_base,
        bool(runtime_config.fastmri_finetune.gradient_checkpoint_frozen_base),
    )

    if train_mode == "warmup_then_both":
        cases = [
            ("warmup", False),
            ("after_warmup", True),
        ]
    else:
        cases = [(train_mode, None)]

    profiled_cases = {}
    for case_name, adapters_active in cases:
        profiled_cases[case_name] = _profile_single_cuda_train_case(
            case_name=case_name,
            model_size=model_size,
            resolved_config_path=resolved_config_path,
            resolved_weight_path=resolved_weight_path,
            crop_size=crop_size,
            batch_size=batch_size,
            device=runtime_device,
            train_mode=train_mode,
            runtime_config=runtime_config,
            use_bf16=resolved_use_bf16,
            gradient_checkpoint_frozen_base=resolved_checkpointing,
            adapters_active=adapters_active,
        )

    return {
        "mode": "fastmri_wrapped_cuda_train_peak",
        "note": "Single-step synthetic training VRAM profile; peak_reserved_gb is the best fit check.",
        "model_size": model_size,
        "train_mode": train_mode,
        "device": str(runtime_device),
        "resolved_config_path": resolved_config_path,
        "resolved_weight_path": resolved_weight_path,
        "crop_size": [int(crop_size[0]), int(crop_size[1])],
        "batch_size": int(batch_size),
        "use_bf16": bool(resolved_use_bf16),
        "gradient_checkpoint_frozen_base": bool(resolved_checkpointing),
        "cases": profiled_cases,
    }



def profile_fastmri_wrapped_model(
    *,
    profile_target: str,
    model_size: str,
    config_path: str | None,
    weight_path: str | None,
    crop_size: tuple[int, int],
    batch_size: int,
    device: str | torch.device,
    train_mode: str,
    use_bf16: bool | None,
    gradient_checkpoint_frozen_base: bool | None,
) -> dict[str, Any]:
    report = {
        "mode": "fastmri_wrapped_profiler",
        "profile_target": profile_target,
        "model_size": model_size,
        "train_mode": train_mode,
        "crop_size": [int(crop_size[0]), int(crop_size[1])],
        "batch_size": int(batch_size),
        "requested_device": str(device),
    }

    if profile_target in {"cpu_infer_estimate", "both"}:
        report["cpu_infer_estimate"] = estimate_fastmri_wrapped_cpu_infer_memory(
            model_size=model_size,
            config_path=config_path,
            weight_path=weight_path,
            crop_size=crop_size,
            batch_size=batch_size,
        )

    if profile_target in {"cuda_train_peak", "both"}:
        report["cuda_train_peak"] = profile_fastmri_wrapped_cuda_train_peak(
            model_size=model_size,
            config_path=config_path,
            weight_path=weight_path,
            crop_size=crop_size,
            batch_size=batch_size,
            device=device,
            train_mode=train_mode,
            use_bf16=use_bf16,
            gradient_checkpoint_frozen_base=gradient_checkpoint_frozen_base,
        )

    return report



def _print_cpu_infer_report(report: dict[str, Any]) -> None:
    print("=" * 84)
    print("CPU Load/Inference RAM Estimate For FastMRI Wrapped SNRAware")
    print("=" * 84)
    print(report["note"])
    print(f"Model preset        : {report['model_size']}")
    print(f"Config path         : {report['resolved_config_path']}")
    print(f"Weight path         : {report['resolved_weight_path']}")
    print(f"Crop size           : {tuple(report['crop_size'])}")
    print(f"Batch size          : {report['batch_size']}")
    print(f"Checkpoint on disk  : {report['checkpoint_file_gb']:.3f} GB")
    print("-" * 84)
    print(f"Parameters          : {report['static']['parameter_count']:,}")
    print(f"Parameter memory    : {report['static']['parameter_gb']:.3f} GB")
    print(f"Buffer memory       : {report['static']['buffer_gb']:.3f} GB")
    print(f"Input tensor        : {report['static']['input_gb']:.6f} GB")
    print(f"Output tensor       : {report['static']['output_gb']:.6f} GB")
    print(f"Static load estimate: {report['static']['model_load_estimate_gb']:.3f} GB")
    print(f"Static forward est. : {report['static']['forward_resident_estimate_gb']:.3f} GB")
    print("-" * 84)
    print(f"RSS build delta     : {report['measured_rss']['build_delta_gb']:.3f} GB")
    print(f"RSS input delta     : {report['measured_rss']['input_delta_gb']:.6f} GB")
    print(f"RSS forward delta   : {report['measured_rss']['forward_delta_gb']:.3f} GB")
    print(f"RSS peak delta      : {report['measured_rss']['peak_delta_from_start_gb']:.3f} GB")
    print("=" * 84)



def _print_cuda_case(case: dict[str, Any]) -> None:
    print(f"Case                : {case['case_name']}")
    print(f"Precision           : {case['precision_mode']}")
    print(f"Adapters active     : {case['adapters_active']}")
    print(f"Checkpoint base     : {case['checkpoint_base_model']}")
    print(f"Trainable params    : {case['trainable_parameters']:,} / {case['total_parameters']:,}")
    print(f"Peak allocated      : {case['headline']['peak_allocated_gb']:.3f} GB")
    print(f"Peak reserved       : {case['headline']['peak_reserved_gb']:.3f} GB")
    print("Phase snapshots:")
    for phase_name, snapshot in case["snapshots"].items():
        print(
            f"  {phase_name:<20} allocated={snapshot['allocated_gb']:.3f} GB "
            f"reserved={snapshot['reserved_gb']:.3f} GB "
            f"peak_alloc={snapshot['peak_allocated_gb']:.3f} GB "
            f"peak_resv={snapshot['peak_reserved_gb']:.3f} GB"
        )



def _print_cuda_train_report(report: dict[str, Any]) -> None:
    print("=" * 84)
    print("CUDA Single-Step Training VRAM Profile For FastMRI Wrapped SNRAware")
    print("=" * 84)
    print(report["note"])
    print(f"Model preset        : {report['model_size']}")
    print(f"Train mode          : {report['train_mode']}")
    print(f"Device              : {report['device']}")
    print(f"Config path         : {report['resolved_config_path']}")
    print(f"Weight path         : {report['resolved_weight_path']}")
    print(f"Crop size           : {tuple(report['crop_size'])}")
    print(f"Batch size          : {report['batch_size']}")
    print(f"BF16                : {report['use_bf16']}")
    print(f"Frozen-base ckpt    : {report['gradient_checkpoint_frozen_base']}")
    print("-" * 84)
    for index, case in enumerate(report["cases"].values()):
        if index > 0:
            print("-" * 84)
        _print_cuda_case(case)
    print("=" * 84)



def _print_report(report: dict[str, Any]) -> None:
    if "cpu_infer_estimate" in report:
        _print_cpu_infer_report(report["cpu_infer_estimate"])
    if "cuda_train_peak" in report:
        if "cpu_infer_estimate" in report:
            print()
        _print_cuda_train_report(report["cuda_train_peak"])



def main() -> None:
    args = parse_args()
    report = profile_fastmri_wrapped_model(
        profile_target=args.profile_target,
        model_size=args.model_size,
        config_path=args.config_path,
        weight_path=args.weight_path,
        crop_size=(int(args.crop_size[0]), int(args.crop_size[1])),
        batch_size=int(args.batch_size),
        device=args.device,
        train_mode=args.train_mode,
        use_bf16=args.use_bf16,
        gradient_checkpoint_frozen_base=args.gradient_checkpoint_frozen_base,
    )
    _print_report(report)

    if args.report_json:
        output_path = Path(args.report_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"Saved JSON report to {output_path}")


if __name__ == "__main__":
    main()
