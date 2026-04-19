#!/usr/bin/env python
"""Python CLI for FastMRI single-coil SNRAware fine-tuning."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from snraware.projects.mri.denoising.base_model_resolver import (
    VALID_BASE_MODEL_VARIANTS,
    resolve_base_model_paths,
)
from snraware.projects.mri.denoising.train import _resolve_device, run_fastmri_finetuning_from_config
from snraware.projects.mri.denoising.trainer_fa import resolve_fastmri_precision, resolve_train_patch_size

VALID_FINE_TUNE_MODES = ("unet_only", "unet_and_lora", "lora_only", "warmup_then_both")
DEFAULT_NUM_WORKERS = min(8, os.cpu_count() or 4)


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "null":
        return None
    return text


def parse_spatial_size(value: str) -> tuple[int, int]:
    normalized = str(value).strip().replace("[", "").replace("]", "").replace(" ", "")
    if "x" in normalized.lower():
        parts = normalized.lower().split("x")
    elif "," in normalized:
        parts = normalized.split(",")
    else:
        parts = [normalized, normalized]
    if len(parts) != 2 or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError(
            f"Invalid spatial size '{value}'. Use formats like 64, 64x96, or 64,96."
        )
    try:
        height, width = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid spatial size '{value}'. Use integer values like 64 or 64x96."
        ) from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError(f"Spatial size must be positive, got '{value}'.")
    return height, width


def parse_optional_spatial_size(value: str) -> tuple[int, int] | None:
    if _normalize_optional_text(value) is None:
        return None
    return parse_spatial_size(value)


def parse_optional_float(value: str) -> float | None:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        return None
    try:
        return float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a float or null, got '{value}'.") from exc


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer, got '{value}'.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got '{value}'.")
    return parsed


def _format_spatial_size(size: tuple[int, int] | list[int] | None) -> str:
    if size is None:
        return "null"
    if isinstance(size, tuple) and len(size) == 2:
        return f"{int(size[0])}x{int(size[1])}"
    if isinstance(size, list) and len(size) >= 2:
        return f"{int(size[0])}x{int(size[1])}"
    return str(size)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _config_dir() -> Path:
    return _repo_root() / "src" / "snraware" / "projects" / "mri" / "denoising" / "configs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SNRAware on FastMRI single-coil data with a maintainable Python CLI."
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--train-root", required=True, help="FastMRI single-coil train directory.")
    data_group.add_argument("--val-root", required=True, help="FastMRI single-coil validation directory.")
    data_group.add_argument(
        "--test-root",
        default=None,
        help="FastMRI single-coil test directory. Defaults to --val-root.",
    )

    model_group = parser.add_argument_group("base model")
    model_group.add_argument(
        "--model-size",
        default="large",
        choices=VALID_BASE_MODEL_VARIANTS,
        help="Named base-model preset. Ignored when explicit base-model paths are provided.",
    )
    model_group.add_argument(
        "--base-model-config",
        default=None,
        help="Explicit base-model YAML path. Must be paired with --base-model-checkpoint.",
    )
    model_group.add_argument(
        "--base-model-checkpoint",
        default=None,
        help="Explicit base-model checkpoint path. Must be paired with --base-model-config.",
    )

    strategy_group = parser.add_argument_group("training strategy")
    strategy_group.add_argument(
        "--mode",
        default="warmup_then_both",
        choices=VALID_FINE_TUNE_MODES,
        help="FastMRI fine-tuning mode.",
    )
    strategy_group.add_argument(
        "--use-unet",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable the FastMRI g-factor prediction wrapper.",
    )
    strategy_group.add_argument(
        "--train-pre-post",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Train SNRAware pre/post layers together with adapters when adapters are active.",
    )
    strategy_group.add_argument(
        "--gradient-checkpoint-frozen-base",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Keep base-model gradient checkpointing enabled during FastMRI fine-tuning.",
    )

    patch_group = parser.add_argument_group("patch and crop")
    patch_group.add_argument(
        "--train-patch-size",
        type=parse_optional_spatial_size,
        default=(64, 64),
        help="Training patch size. Accepts 64, 64x96, 64,96, or null to disable patch training.",
    )
    patch_group.add_argument(
        "--crop-size",
        type=parse_spatial_size,
        default=(320, 320),
        help="Evaluation crop size. Accepts 320, 320x320, or 320,320.",
    )
    patch_group.add_argument(
        "--patch-overlap",
        type=parse_spatial_size,
        default=(16, 16),
        help="Sliding-window overlap used for validation/test patch inference.",
    )
    patch_group.add_argument(
        "--eval-patch-batch-size",
        type=parse_positive_int,
        default=64,
        help="Patch batch size for evaluation-time sliding-window inference.",
    )

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--unet-lr", type=float, default=1.0e-4)
    optimization_group.add_argument("--adapter-lr", type=float, default=1.0e-5)
    optimization_group.add_argument("--weight-decay", type=float, default=0.0)
    optimization_group.add_argument("--max-epochs", type=parse_positive_int, default=20)
    optimization_group.add_argument("--warmup-epochs", type=parse_positive_int, default=5)
    optimization_group.add_argument("--evaluate-every", type=parse_positive_int, default=2)
    optimization_group.add_argument("--batch-size", type=parse_positive_int, default=1)
    optimization_group.add_argument("--num-workers", type=parse_positive_int, default=DEFAULT_NUM_WORKERS)
    optimization_group.add_argument("--acc-factor", type=parse_positive_int, default=8)
    optimization_group.add_argument("--scheduler-t-max", type=int, default=0)
    optimization_group.add_argument("--log-every-n-steps", type=parse_positive_int, default=50)
    optimization_group.add_argument(
        "--train-sample-rate",
        type=parse_optional_float,
        default=None,
        help="Optional training-only slice sample rate. Use null for the full training set.",
    )
    optimization_group.add_argument(
        "--train-volume-sample-rate",
        type=parse_optional_float,
        default=None,
        help="Optional training-only volume sample rate. Use null for the full training set.",
    )
    optimization_group.add_argument(
        "--shuffle-train",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Shuffle the FastMRI training loader.",
    )

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument(
        "--device",
        default="auto",
        help="Torch device string, for example auto, cpu, cuda, or cuda:0.",
    )
    runtime_group.add_argument(
        "--cuda-device",
        default=None,
        help="Physical GPU id(s) passed through CUDA_VISIBLE_DEVICES before training starts.",
    )
    runtime_group.add_argument(
        "--precision",
        default="auto",
        choices=("auto", "bf16", "fp32"),
        help="Training precision policy. auto prefers bf16 on supported CUDA devices, otherwise fp32.",
    )
    runtime_group.add_argument("--save-root", default="./checkpoints/fine_tune")
    runtime_group.add_argument("--run-name", default=None)
    runtime_group.add_argument(
        "--wandb",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable Weights & Biases logging.",
    )
    runtime_group.add_argument("--project", default="fastmri-snraware")
    runtime_group.add_argument("--seed", type=int, default=None)
    runtime_group.add_argument("--sample-seed", type=int, default=1234)
    runtime_group.add_argument(
        "--deterministic-mask",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Derive the undersampling mask deterministically from the sample name.",
    )
    runtime_group.add_argument(
        "--pin-memory",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Override DataLoader pin_memory. Defaults to true on CUDA and false on CPU.",
    )
    runtime_group.add_argument(
        "--persistent-workers",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Override DataLoader persistent_workers. Defaults to num_workers > 0.",
    )
    runtime_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved training plan and config, then exit without starting training.",
    )
    runtime_group.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved OmegaConf config before launching training.",
    )

    advanced_group = parser.add_argument_group("advanced")
    advanced_group.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Advanced OmegaConf override, for example --override lora.r=16.",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _compose_base_config() -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_config_dir())):
        config = compose(config_name="fastmri_finetune")
    OmegaConf.set_struct(config, False)
    return config


def _validate_existing_dir(path_value: str, *, label: str) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (_repo_root() / path).resolve()
    else:
        path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")
    return str(path)


def _resolve_path_string(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (_repo_root() / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def _normalize_config_pair(
    value: object,
    *,
    label: str,
    allow_null: bool = False,
    allow_zero: bool = False,
) -> tuple[int, int] | None:
    if value in (None, "", "null"):
        if allow_null:
            return None
        raise ValueError(f"{label} cannot be null.")
    if OmegaConf.is_list(value) or isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{label} must contain exactly two values, got {value}.")
        dims = tuple(int(dim) for dim in value)
    else:
        dims = parse_spatial_size(str(value))
    if not allow_zero and any(dim <= 0 for dim in dims):
        raise ValueError(f"{label} must be positive, got {value}.")
    if allow_zero and any(dim < 0 for dim in dims):
        raise ValueError(f"{label} must be non-negative, got {value}.")
    return dims


def _normalize_overlap_pair(value: object) -> tuple[int, int]:
    if OmegaConf.is_list(value) or isinstance(value, (list, tuple)):
        if len(value) < 2:
            raise ValueError(f"overlap_for_inference must contain at least two values, got {value}.")
        dims = (int(value[0]), int(value[1]))
    else:
        dims = parse_spatial_size(str(value))
    if any(dim < 0 for dim in dims):
        raise ValueError(f"PATCH_OVERLAP must be non-negative, got {value}.")
    return dims


def _override_touches_key(overrides: list[str], key: str) -> bool:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected KEY=VALUE.")
        override_key = override.split("=", 1)[0].strip()
        if override_key == key:
            return True
    return False


def resolve_cli_precision(device_str: str, precision: str) -> dict[str, Any]:
    device = _resolve_device(device_str)
    if precision == "fp32":
        precision_state = {"mode": "fp32", "use_bf16": False}
    elif precision == "bf16":
        precision_state = resolve_fastmri_precision(device, use_bf16=True)
    else:
        if device.type != "cuda":
            precision_state = {"mode": "fp32", "use_bf16": False}
        else:
            try:
                precision_state = resolve_fastmri_precision(device, use_bf16=True)
            except RuntimeError:
                precision_state = {"mode": "fp32", "use_bf16": False}
    return {"device": device, **precision_state}


def _native_patch_size_from_base_config(config_path: str) -> tuple[int, int] | None:
    try:
        base_config = OmegaConf.load(config_path)
    except Exception:
        return None
    cutout_shape = OmegaConf.select(base_config, "dataset.cutout_shape")
    if not (OmegaConf.is_list(cutout_shape) or isinstance(cutout_shape, (list, tuple))) or len(cutout_shape) < 2:
        return None
    return int(cutout_shape[0]), int(cutout_shape[1])


def build_config_from_args(args: argparse.Namespace) -> tuple[DictConfig, dict[str, Any]]:
    config = _compose_base_config()

    config.fastmri_finetune.train_root = _validate_existing_dir(args.train_root, label="train_root")
    config.fastmri_finetune.val_root = _validate_existing_dir(args.val_root, label="val_root")
    test_root = args.val_root if args.test_root is None else args.test_root
    config.fastmri_finetune.test_root = _validate_existing_dir(test_root, label="test_root")

    config.base_model.variant = args.model_size
    config.base_model.config_path = _normalize_optional_text(args.base_model_config)
    config.base_model.checkpoint_path = _normalize_optional_text(args.base_model_checkpoint)

    config.fastmri_finetune.mode = args.mode
    config.fastmri_finetune.use_unet = bool(args.use_unet)
    config.fastmri_finetune.train_pre_post = bool(args.train_pre_post)
    config.fastmri_finetune.gradient_checkpoint_frozen_base = bool(
        args.gradient_checkpoint_frozen_base
    )
    config.fastmri_finetune.train_patch_size = (
        None if args.train_patch_size is None else list(args.train_patch_size)
    )
    config.fastmri_finetune.crop_size = list(args.crop_size)
    config.overlap_for_inference = [int(args.patch_overlap[0]), int(args.patch_overlap[1]), 0]
    config.fastmri_finetune.eval_patch_batch_size = int(args.eval_patch_batch_size)

    config.fastmri_finetune.unet_lr = float(args.unet_lr)
    config.fastmri_finetune.adapter_lr = float(args.adapter_lr)
    config.fastmri_finetune.weight_decay = float(args.weight_decay)
    config.fastmri_finetune.max_epochs = int(args.max_epochs)
    config.fastmri_finetune.warmup_epochs = int(args.warmup_epochs)
    config.fastmri_finetune.evaluate_every_n_epochs = int(args.evaluate_every)
    config.fastmri_finetune.batch_size = int(args.batch_size)
    config.fastmri_finetune.num_workers = int(args.num_workers)
    config.fastmri_finetune.acc_factor = int(args.acc_factor)
    config.fastmri_finetune.scheduler_t_max = int(args.scheduler_t_max)
    config.fastmri_finetune.log_every_n_steps = int(args.log_every_n_steps)
    config.fastmri_finetune.train_sample_rate = args.train_sample_rate
    config.fastmri_finetune.train_volume_sample_rate = args.train_volume_sample_rate
    config.fastmri_finetune.shuffle_train = bool(args.shuffle_train)

    config.fastmri_finetune.device = str(args.device)
    config.fastmri_finetune.save_root = _resolve_path_string(str(args.save_root))
    config.fastmri_finetune.run_name = _normalize_optional_text(args.run_name)
    config.logging.use_wandb = bool(args.wandb)
    config.logging.project = str(args.project)
    config.seed = args.seed
    config.fastmri_finetune.sample_seed = int(args.sample_seed)
    config.fastmri_finetune.deterministic_mask_from_name = bool(args.deterministic_mask)

    if args.override:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.override))

    config.fastmri_finetune.save_root = _resolve_path_string(str(config.fastmri_finetune.save_root))
    planned_run_name = config.fastmri_finetune.run_name or f"{config.fastmri_finetune.mode}_auto"
    config.logging.run_name = planned_run_name
    config.logging.output_dir = str(Path(config.fastmri_finetune.save_root) / planned_run_name)

    if (
        config.fastmri_finetune.mode == "unet_only"
        and not bool(config.fastmri_finetune.get("use_unet", True))
    ):
        raise ValueError("mode=unet_only cannot be combined with --no-use-unet.")

    resolved_crop_size = _normalize_config_pair(
        config.fastmri_finetune.crop_size,
        label="fastmri_finetune.crop_size",
    )
    config.fastmri_finetune.crop_size = list(resolved_crop_size)

    resolved_train_patch = resolve_train_patch_size(config.fastmri_finetune)
    if resolved_train_patch is not None:
        if resolved_train_patch[0] > resolved_crop_size[0] or resolved_train_patch[1] > resolved_crop_size[1]:
            raise ValueError(
                "TRAIN_PATCH_SIZE must fit inside CROP_SIZE, "
                f"got TRAIN_PATCH_SIZE={_format_spatial_size(resolved_train_patch)} and "
                f"CROP_SIZE={_format_spatial_size(resolved_crop_size)}."
            )
        config.fastmri_finetune.train_patch_size = list(resolved_train_patch)
    else:
        config.fastmri_finetune.train_patch_size = None

    resolved_patch_overlap = _normalize_overlap_pair(config.get("overlap_for_inference", [0, 0, 0]))
    if resolved_train_patch is not None and (
        resolved_patch_overlap[0] >= resolved_train_patch[0]
        or resolved_patch_overlap[1] >= resolved_train_patch[1]
    ):
        raise ValueError(
            "PATCH_OVERLAP must be smaller than TRAIN_PATCH_SIZE, "
            f"got PATCH_OVERLAP={_format_spatial_size(resolved_patch_overlap)} and "
            f"TRAIN_PATCH_SIZE={_format_spatial_size(resolved_train_patch)}."
        )
    config.overlap_for_inference = [int(resolved_patch_overlap[0]), int(resolved_patch_overlap[1]), 0]

    resolved_config_path, resolved_checkpoint_path = resolve_base_model_paths(
        variant=config.base_model.get("variant"),
        config_path=config.base_model.get("config_path"),
        checkpoint_path=config.base_model.get("checkpoint_path"),
        repo_root=_repo_root(),
    )
    config.base_model.config_path = resolved_config_path
    config.base_model.checkpoint_path = resolved_checkpoint_path

    precision_override = _override_touches_key(args.override, "fastmri_finetune.use_bf16")
    pin_memory_override = _override_touches_key(args.override, "fastmri_finetune.pin_memory")
    persistent_workers_override = _override_touches_key(
        args.override, "fastmri_finetune.persistent_workers"
    )
    if precision_override:
        precision_state = resolve_fastmri_precision(
            _resolve_device(str(config.fastmri_finetune.device)),
            use_bf16=bool(config.fastmri_finetune.use_bf16),
        )
        resolved_device = _resolve_device(str(config.fastmri_finetune.device))
    else:
        resolved_precision = resolve_cli_precision(str(config.fastmri_finetune.device), args.precision)
        resolved_device = resolved_precision["device"]
        precision_state = {
            "mode": resolved_precision["mode"],
            "use_bf16": resolved_precision["use_bf16"],
        }
        config.fastmri_finetune.use_bf16 = bool(precision_state["use_bf16"])

    if args.pin_memory is None and not pin_memory_override:
        config.fastmri_finetune.pin_memory = bool(resolved_device.type == "cuda")
    elif args.pin_memory is not None:
        config.fastmri_finetune.pin_memory = bool(args.pin_memory)
    if args.persistent_workers is None and not persistent_workers_override:
        config.fastmri_finetune.persistent_workers = int(config.fastmri_finetune.num_workers) > 0
    elif args.persistent_workers is not None:
        config.fastmri_finetune.persistent_workers = bool(args.persistent_workers)

    native_patch_size = _native_patch_size_from_base_config(resolved_config_path)
    train_input_size = resolved_train_patch or resolved_crop_size
    patch_inference_enabled = (
        resolved_train_patch is not None and tuple(resolved_train_patch) != tuple(resolved_crop_size)
    )

    summary = {
        "DRY_RUN": str(bool(args.dry_run)).lower(),
        "MODEL_SIZE": str(config.base_model.get("variant")),
        "MODE": str(config.fastmri_finetune.mode),
        "TRAIN_ROOT": str(config.fastmri_finetune.train_root),
        "VAL_ROOT": str(config.fastmri_finetune.val_root),
        "TEST_ROOT": str(config.fastmri_finetune.test_root),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "DEVICE_REQUEST": str(config.fastmri_finetune.device),
        "RESOLVED_DEVICE": str(resolved_device),
        "TRAINING_PRECISION": str(precision_state["mode"]),
        "TRAIN_INPUT_SIZE": _format_spatial_size(train_input_size),
        "EVAL_CROP_SIZE": _format_spatial_size(resolved_crop_size),
        "PATCH_INFERENCE_FOR_VAL_TEST": str(bool(patch_inference_enabled)).lower(),
        "PATCH_OVERLAP": (
            f"{_format_spatial_size(resolved_patch_overlap)} (unused without train_patch_size)"
            if resolved_train_patch is None
            else _format_spatial_size(resolved_patch_overlap)
        ),
        "EVAL_PATCH_BATCH_SIZE": str(config.fastmri_finetune.eval_patch_batch_size),
        "EVALUATE_EVERY": str(config.fastmri_finetune.evaluate_every_n_epochs),
        "USE_UNET": str(bool(config.fastmri_finetune.use_unet)).lower(),
        "TRAIN_PRE_POST": str(bool(config.fastmri_finetune.train_pre_post)).lower(),
        "GRADIENT_CHECKPOINT_FROZEN_BASE": str(
            bool(config.fastmri_finetune.gradient_checkpoint_frozen_base)
        ).lower(),
        "BASE_MODEL_CONFIG": resolved_config_path,
        "BASE_MODEL_CHECKPOINT": resolved_checkpoint_path,
        "CHECKPOINT_NATIVE_SPATIAL_SIZE": _format_spatial_size(native_patch_size),
    }

    return config, summary


def _print_summary(summary: dict[str, Any], config: DictConfig) -> None:
    print("FastMRI single-coil training plan:")
    for key, value in summary.items():
        print(f"{key}={value}")
    print("Resolved config:")
    print(OmegaConf.to_yaml(config, resolve=True))


def _maybe_set_cuda_visible_devices(cuda_device: str | None) -> None:
    normalized = _normalize_optional_text(cuda_device)
    if normalized is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = normalized


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _maybe_set_cuda_visible_devices(args.cuda_device)

    try:
        config, summary = build_config_from_args(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    if args.dry_run or args.print_config:
        _print_summary(summary, config)
    if args.dry_run:
        print("Dry run requested; training was not started.")
        return 0

    return run_fastmri_finetuning_from_config(config)


if __name__ == "__main__":
    main()
