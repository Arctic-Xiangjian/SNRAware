"""Custom FastMRI fine-tuning loop for SNRAware."""

from __future__ import annotations

import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from snraware.projects.mri.denoising.fastmri_compat import (
    SNRAwareWithGFactor,
    has_lora_adapters,
    load_fastmri_finetune_checkpoint,
    save_fastmri_finetune_checkpoint,
)
from snraware.projects.mri.denoising.lora_utils import (
    apply_lora_to_model,
    count_trainable_parameters,
    resolve_lora_config,
)

__all__ = [
    "FastMRIFineTuneTrainer",
    "build_fastmri_optimizer",
    "build_fastmri_dataloaders",
    "complex_output_to_magnitude",
    "compute_volume_metrics",
    "configure_model_for_finetune_mode",
    "fastmri_autocast_context",
    "group_slices_into_volumes",
    "resolve_fastmri_precision",
    "resolve_train_patch_size",
    "seed_everything",
    "should_checkpoint_frozen_base",
]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def complex_output_to_magnitude(output: torch.Tensor) -> torch.Tensor:
    """Convert SNRAware 2-channel complex output to a 2D magnitude tensor."""
    if output.ndim != 5 or output.shape[1] != 2:
        raise ValueError(f"Expected [B, 2, T, H, W] output, got {tuple(output.shape)}")
    real = output[:, 0:1, ...]
    imag = output[:, 1:2, ...]
    magnitude = torch.sqrt(real.square() + imag.square())
    if magnitude.shape[2] != 1:
        raise ValueError(
            f"FastMRI fine-tuning expects singleton T=1, got magnitude shape {tuple(magnitude.shape)}"
        )
    return magnitude.squeeze(2)


def resolve_train_patch_size(ft_cfg: Any) -> tuple[int, int] | None:
    """Normalize the optional FastMRI training patch size config."""
    value = ft_cfg.get("train_patch_size", None)
    if value in (None, "", "null"):
        return None
    patch_size = tuple(int(dim) for dim in value)
    if len(patch_size) != 2:
        raise ValueError(f"fastmri_finetune.train_patch_size must contain exactly two values, got {value}")
    if any(dim <= 0 for dim in patch_size):
        raise ValueError(f"fastmri_finetune.train_patch_size must be positive, got {patch_size}")
    return patch_size


def resolve_fastmri_precision(device: torch.device, *, use_bf16: bool) -> dict[str, Any]:
    """Resolve the FastMRI fine-tuning precision policy for the selected device."""
    if not use_bf16:
        return {"mode": "fp32", "use_bf16": False}

    if device.type != "cuda":
        raise ValueError(
            "fastmri_finetune.use_bf16=true requires a CUDA device, "
            f"but got device={device}. Set fastmri_finetune.use_bf16=false "
            "(or USE_BF16=false) to run in fp32."
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            f"Selected device {device} does not report bf16 support. "
            "Set fastmri_finetune.use_bf16=false (or USE_BF16=false) to run in fp32."
        )

    return {"mode": "bf16", "use_bf16": True}


def fastmri_autocast_context(device: torch.device, *, enabled: bool):
    """Return the autocast context used by the FastMRI fine-tune path."""
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


def _is_lora_parameter(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name


def _is_pre_post_parameter(name: str) -> bool:
    return name.startswith("pre.") or name.startswith("post.")


def _set_module_trainable(module: nn.Module, flag: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = flag


def _set_lora_trainable(base_model: nn.Module, flag: bool) -> None:
    for name, parameter in base_model.named_parameters():
        if _is_lora_parameter(name):
            parameter.requires_grad = flag


def _set_pre_post_trainable(base_model: nn.Module, flag: bool) -> None:
    for name, parameter in base_model.named_parameters():
        if _is_pre_post_parameter(name):
            parameter.requires_grad = flag


def _get_fastmri_adapter_parameters(base_model: nn.Module, *, train_pre_post: bool) -> list[nn.Parameter]:
    return [
        parameter
        for name, parameter in base_model.named_parameters()
        if _is_lora_parameter(name) or (train_pre_post and _is_pre_post_parameter(name))
    ]


def _configure_fastmri_base_model_trainable_state(
    base_model: nn.Module,
    *,
    train_lora: bool,
    train_pre_post: bool,
) -> None:
    _set_module_trainable(base_model, False)
    _set_lora_trainable(base_model, train_lora)
    _set_pre_post_trainable(base_model, train_pre_post)


def _resolve_train_pre_post(ft_cfg: Any) -> bool:
    return bool(ft_cfg.get("train_pre_post", False))


def _resolve_eval_patch_overlap(config: DictConfig, patch_size: tuple[int, int]) -> tuple[int, int]:
    overlap_values = config.get("overlap_for_inference", [0, 0, 0])
    if len(overlap_values) < 2:
        raise ValueError("overlap_for_inference must provide at least two spatial values")
    overlap = (int(overlap_values[0]), int(overlap_values[1]))
    if overlap[0] < 0 or overlap[1] < 0:
        raise ValueError(f"overlap_for_inference must be non-negative, got {overlap}")
    if overlap[0] >= patch_size[0] or overlap[1] >= patch_size[1]:
        raise ValueError(
            "FastMRI spatial overlap_for_inference must be smaller than train_patch_size, "
            f"got overlap={overlap}, train_patch_size={patch_size}"
        )
    return overlap


def _stack_fastmri_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    noisy = torch.stack([sample[0] for sample in batch], dim=0)
    clean = torch.stack([sample[1] for sample in batch], dim=0)
    noise_sigma = torch.stack([torch.as_tensor(sample[2], dtype=torch.float32) for sample in batch], dim=0)

    metadata_entries = [sample[3] for sample in batch]
    metadata: dict[str, Any] = {}
    for key in metadata_entries[0]:
        values = [entry[key] for entry in metadata_entries]
        if key in {"name", "volume_name"}:
            metadata[key] = [str(value) for value in values]
        elif key == "slice_idx":
            metadata[key] = torch.tensor([int(value) for value in values], dtype=torch.int64)
        elif all(value is None for value in values):
            metadata[key] = None
        elif any(value is None for value in values):
            raise ValueError(f"Mixed None/non-None values for metadata['{key}'] in the same batch")
        else:
            metadata[key] = torch.stack([torch.as_tensor(value) for value in values], dim=0)

    return noisy, clean, noise_sigma, metadata


def build_fastmri_optimizer(
    model: SNRAwareWithGFactor,
    ft_cfg: Any,
    mode_state: dict[str, Any],
) -> torch.optim.Optimizer:
    """Build the FastMRI AdamW optimizer with trainer-matching parameter groups."""
    param_groups = []
    gfactor_params = list(model.gfactor_unet.parameters())
    adapter_params = _get_fastmri_adapter_parameters(
        model.base_model,
        train_pre_post=_resolve_train_pre_post(ft_cfg),
    )

    if any(parameter.requires_grad for parameter in gfactor_params) or ft_cfg.mode == "warmup_then_both":
        param_groups.append(
            {
                "name": "gfactor_unet",
                "params": gfactor_params,
                "lr": float(ft_cfg.unet_lr),
                "weight_decay": float(ft_cfg.weight_decay),
            }
        )

    if adapter_params:
        adapter_lr = (
            0.0 if ft_cfg.mode == "warmup_then_both" and not mode_state["adapters_active"] else float(ft_cfg.adapter_lr)
        )
        param_groups.append(
            {
                "name": "adapter",
                "params": adapter_params,
                "lr": adapter_lr,
                "weight_decay": float(ft_cfg.weight_decay),
            }
        )

    if not param_groups:
        raise RuntimeError("No parameter groups were created for FastMRI fine-tuning")

    return torch.optim.AdamW(param_groups)


def should_checkpoint_frozen_base(
    model: SNRAwareWithGFactor,
    *,
    gradient_checkpoint_frozen_base: bool,
) -> bool:
    """Checkpoint the SNRAware base-model segment during training whenever enabled."""
    return gradient_checkpoint_frozen_base and model.training


def configure_model_for_finetune_mode(
    model: SNRAwareWithGFactor,
    *,
    mode: str,
    lora_config: Any | None = None,
    adapters_active: bool | None = None,
    train_pre_post: bool = False,
) -> dict[str, Any]:
    """Configure trainable parameters for the requested FastMRI fine-tune mode."""
    supported_modes = {"unet_only", "unet_and_lora", "lora_only", "warmup_then_both"}
    if mode not in supported_modes:
        raise ValueError(f"Unsupported fine-tune mode: {mode}")

    requested_adapters_active = adapters_active
    adapters_active = False
    if mode == "unet_only":
        _set_module_trainable(model.base_model, False)
        _set_module_trainable(model.gfactor_unet, True)
    else:
        resolved = resolve_lora_config(
            model_config=getattr(model.base_model, "config", None),
            lora_config=lora_config,
        )
        if not resolved.enabled:
            raise ValueError(f"LoRA must be enabled for fine-tune mode '{mode}'")

        if not has_lora_adapters(model.base_model):
            apply_lora_to_model(model.base_model, lora_config=lora_config)
        _configure_fastmri_base_model_trainable_state(
            model.base_model,
            train_lora=False,
            train_pre_post=False,
        )

        if mode == "unet_and_lora":
            _set_module_trainable(model.gfactor_unet, True)
            _configure_fastmri_base_model_trainable_state(
                model.base_model,
                train_lora=True,
                train_pre_post=bool(train_pre_post),
            )
            adapters_active = True
        elif mode == "lora_only":
            _set_module_trainable(model.gfactor_unet, False)
            _configure_fastmri_base_model_trainable_state(
                model.base_model,
                train_lora=True,
                train_pre_post=bool(train_pre_post),
            )
            adapters_active = True
        else:
            _set_module_trainable(model.gfactor_unet, True)
            adapters_active = (
                bool(requested_adapters_active) if requested_adapters_active is not None else False
            )
            _configure_fastmri_base_model_trainable_state(
                model.base_model,
                train_lora=adapters_active,
                train_pre_post=bool(train_pre_post) and adapters_active,
            )

    return {
        "mode": mode,
        "adapters_active": adapters_active,
        "has_lora": has_lora_adapters(model.base_model),
        "train_pre_post": bool(train_pre_post),
    }


def _normalize_metric_sources(metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]] | None):
    if metric_fns is not None:
        required = {"psnr", "ssim", "nmse"}
        if set(metric_fns) != required:
            raise ValueError(f"metric_fns must provide exactly {sorted(required)}")
        return metric_fns

    try:
        from fastmri.evaluate import nmse, psnr, ssim
    except ImportError as exc:
        raise ImportError(
            "The FastMRI fine-tuning path requires the `fastmri` package for volume metrics. "
            "Install it and re-run training."
        ) from exc

    return {"psnr": psnr, "ssim": ssim, "nmse": nmse}


def group_slices_into_volumes(
    volume_names: Iterable[str],
    slice_indices: Iterable[int],
    predictions: Iterable[np.ndarray],
    targets: Iterable[np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Group per-slice predictions and targets back into volume stacks."""
    grouped: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for volume_name, slice_idx, prediction, target in zip(
        volume_names, slice_indices, predictions, targets, strict=True
    ):
        grouped[str(volume_name)].append((int(slice_idx), prediction.astype(np.float32), target.astype(np.float32)))

    volumes: dict[str, dict[str, np.ndarray]] = {}
    for volume_name, items in grouped.items():
        items.sort(key=lambda item: item[0])
        volumes[volume_name] = {
            "prediction": np.stack([item[1] for item in items], axis=0),
            "target": np.stack([item[2] for item in items], axis=0),
        }
    return volumes


def compute_volume_metrics(
    grouped_volumes: dict[str, dict[str, np.ndarray]],
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
) -> dict[str, float]:
    """Compute FastMRI volume-level metrics from grouped magnitude volumes."""
    metric_fns = _normalize_metric_sources(metric_fns)
    if not grouped_volumes:
        return {"psnr": float("nan"), "ssim": float("nan"), "nmse": float("nan")}

    metric_lists: dict[str, list[float]] = {"psnr": [], "ssim": [], "nmse": []}
    for volume in grouped_volumes.values():
        target = volume["target"]
        prediction = volume["prediction"]
        metric_lists["psnr"].append(float(metric_fns["psnr"](target, prediction)))
        metric_lists["ssim"].append(float(metric_fns["ssim"](target, prediction)))
        metric_lists["nmse"].append(float(metric_fns["nmse"](target, prediction)))

    return {name: float(np.mean(values)) for name, values in metric_lists.items()}


def build_fastmri_dataloaders(
    config: DictConfig,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """Build FastMRI bridge dataloaders for train/val/test."""
    try:
        from fastmri_data.work_with_snraware import FastMRISNRAwareDataset
    except ImportError as exc:
        raise ImportError(
            "Could not import the FastMRI bridge dataset. Make sure you run from the repository root."
        ) from exc

    ft_cfg = config.fastmri_finetune

    def _resolve_train_sampling() -> tuple[Any | None, Any | None]:
        train_sample_rate = ft_cfg.get("train_sample_rate", None)
        train_volume_sample_rate = ft_cfg.get("train_volume_sample_rate", None)

        # Backward compatibility: legacy `sample_rate` / `volume_sample_rate`
        # are interpreted as training-only knobs.
        if train_sample_rate is None:
            train_sample_rate = ft_cfg.sample_rate
        if train_volume_sample_rate is None:
            train_volume_sample_rate = ft_cfg.volume_sample_rate

        return train_sample_rate, train_volume_sample_rate

    def _build_dataset(
        root: str | Path | None,
        *,
        split: str,
        sample_rate: Any | None,
        volume_sample_rate: Any | None,
    ):
        if root in (None, "", "null"):
            return None
        train_patch_size = resolve_train_patch_size(ft_cfg) if split == "train" else None
        return FastMRISNRAwareDataset(
            root=root,
            split=split,
            challenge=ft_cfg.challenge,
            sample_rate=sample_rate,
            volume_sample_rate=volume_sample_rate,
            use_dataset_cache=ft_cfg.use_dataset_cache,
            dataset_cache_file=ft_cfg.dataset_cache_file,
            scanner_models=ft_cfg.scanner_models,
            acc_factor=ft_cfg.acc_factor,
            crop_size=tuple(ft_cfg.crop_size),
            train_patch_size=train_patch_size,
            strict_latent_feature=False,
            deterministic_mask_from_name=bool(ft_cfg.deterministic_mask_from_name),
            sample_seed=ft_cfg.sample_seed,
        )

    train_sample_rate, train_volume_sample_rate = _resolve_train_sampling()
    train_dataset = _build_dataset(
        ft_cfg.train_root,
        split="train",
        sample_rate=train_sample_rate,
        volume_sample_rate=train_volume_sample_rate,
    )
    if train_dataset is None:
        raise ValueError("fastmri_finetune.train_root must be provided")

    # Validation and test always use the full dataset by default.
    val_dataset = _build_dataset(ft_cfg.val_root, split="val", sample_rate=None, volume_sample_rate=None)
    test_dataset = _build_dataset(ft_cfg.test_root, split="test", sample_rate=None, volume_sample_rate=None)

    loader_kwargs = {
        "batch_size": int(ft_cfg.batch_size),
        "num_workers": int(ft_cfg.num_workers),
        "pin_memory": bool(ft_cfg.pin_memory),
        "persistent_workers": bool(ft_cfg.persistent_workers) if int(ft_cfg.num_workers) > 0 else False,
        "drop_last": False,
        "collate_fn": _stack_fastmri_batch,
    }

    train_loader = DataLoader(train_dataset, shuffle=bool(ft_cfg.shuffle_train), **loader_kwargs)
    val_loader = (
        DataLoader(val_dataset, shuffle=False, **loader_kwargs) if val_dataset is not None else None
    )
    test_loader = (
        DataLoader(test_dataset, shuffle=False, **loader_kwargs) if test_dataset is not None else None
    )
    return train_loader, val_loader, test_loader


class FastMRIFineTuneTrainer:
    """A small, explicit trainer for FastMRI fine-tuning with magnitude-domain loss."""

    def __init__(
        self,
        *,
        model: SNRAwareWithGFactor,
        config: DictConfig,
        device: torch.device,
        run_dir: str | Path,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
        wandb_run: Any | None = None,
        precision_state: dict[str, Any] | None = None,
    ):
        self.config = config
        self.ft_cfg = config.fastmri_finetune
        self.device = device
        self.precision_state = precision_state or resolve_fastmri_precision(
            device,
            use_bf16=bool(self.ft_cfg.use_bf16),
        )
        self.use_bf16 = bool(self.precision_state["use_bf16"])
        self.precision_mode = str(self.precision_state["mode"])
        self.gradient_checkpoint_frozen_base = bool(
            self.ft_cfg.get("gradient_checkpoint_frozen_base", True)
        )
        self.train_patch_size = resolve_train_patch_size(self.ft_cfg)
        self.eval_patch_overlap = (
            _resolve_eval_patch_overlap(config, self.train_patch_size)
            if self.train_patch_size is not None
            else None
        )
        self.model = model.to(device=device, dtype=torch.float32)
        self.run_dir = Path(run_dir)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.metric_fns = metric_fns
        self.wandb_run = wandb_run
        self.loss_fn = nn.L1Loss()
        self._use_tqdm = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self.mode_state = configure_model_for_finetune_mode(
            self.model,
            mode=self.ft_cfg.mode,
            lora_config=config.get("lora"),
            train_pre_post=_resolve_train_pre_post(self.ft_cfg),
        )
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.current_epoch = 0
        self.best_val_psnr = float("-inf")

        if self.ft_cfg.resume_from:
            self._load_resume_state(self.ft_cfg.resume_from)

        self._print_startup_summary()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return build_fastmri_optimizer(self.model, self.ft_cfg, self.mode_state)

    def _build_scheduler(self) -> Any | None:
        t_max = int(self.ft_cfg.scheduler_t_max)
        if t_max <= 0:
            return None
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max)

    def _sync_optimizer_learning_rates(self) -> None:
        for group in self.optimizer.param_groups:
            group["weight_decay"] = float(self.ft_cfg.weight_decay)
            if group.get("name") == "gfactor_unet":
                group["lr"] = float(self.ft_cfg.unet_lr)
            elif group.get("name") == "adapter":
                group["lr"] = (
                    0.0
                    if self.ft_cfg.mode == "warmup_then_both" and not self.mode_state["adapters_active"]
                    else float(self.ft_cfg.adapter_lr)
                )
        if self.scheduler is not None and hasattr(self.scheduler, "base_lrs"):
            for index, group in enumerate(self.optimizer.param_groups):
                if group.get("name") == "gfactor_unet":
                    self.scheduler.base_lrs[index] = float(self.ft_cfg.unet_lr)
                elif group.get("name") == "adapter":
                    self.scheduler.base_lrs[index] = float(group["lr"])

    def _resolve_resume_adapters_active(self, payload: dict[str, Any]) -> bool:
        if self.ft_cfg.mode != "warmup_then_both":
            return self.ft_cfg.mode != "unet_only"

        stored = payload.get("adapters_active", None)
        if stored is not None:
            return bool(stored)

        saved_epoch = int(payload.get("epoch", -1))
        return saved_epoch >= int(self.ft_cfg.warmup_epochs)

    def _reconcile_mode_state_after_resume(self, payload: dict[str, Any]) -> None:
        adapters_active = self._resolve_resume_adapters_active(payload)
        self.mode_state = configure_model_for_finetune_mode(
            self.model,
            mode=self.ft_cfg.mode,
            lora_config=self.config.get("lora"),
            adapters_active=adapters_active,
            train_pre_post=_resolve_train_pre_post(self.ft_cfg),
        )
        self._sync_optimizer_learning_rates()

    def _load_resume_state(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_mode = payload.get("mode")
        if checkpoint_mode is not None and checkpoint_mode != self.ft_cfg.mode:
            raise ValueError(
                "FastMRI fine-tune resume mode mismatch: "
                f"checkpoint mode '{checkpoint_mode}' != configured mode '{self.ft_cfg.mode}'"
            )
        load_fastmri_finetune_checkpoint(
            self.model,
            payload,
            apply_lora_fn=apply_lora_to_model,
            lora_config=self.config.get("lora"),
        )
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in payload:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.current_epoch = int(payload.get("epoch", -1)) + 1
        self._reconcile_mode_state_after_resume(payload)

    def _maybe_activate_adapters(self, epoch: int) -> None:
        if self.ft_cfg.mode != "warmup_then_both":
            return
        if self.mode_state["adapters_active"]:
            return
        if epoch < int(self.ft_cfg.warmup_epochs):
            return

        _configure_fastmri_base_model_trainable_state(
            self.model.base_model,
            train_lora=True,
            train_pre_post=_resolve_train_pre_post(self.ft_cfg),
        )
        self.mode_state["adapters_active"] = True
        self._sync_optimizer_learning_rates()

    def _should_checkpoint_frozen_base(self) -> bool:
        return should_checkpoint_frozen_base(
            self.model,
            gradient_checkpoint_frozen_base=self.gradient_checkpoint_frozen_base,
        )

    def _log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log(payload, step=step)

    def _console_print(self, message: str) -> None:
        print(message, flush=True)

    @staticmethod
    def _format_lr(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.2e}"

    @staticmethod
    def _format_metric_value(value: float) -> str:
        return "nan" if not np.isfinite(value) else f"{value:.4f}"

    def _current_group_lr(self, group_name: str) -> float | None:
        for group in self.optimizer.param_groups:
            if group.get("name") == group_name:
                return float(group["lr"])
        return None

    def _print_startup_summary(self) -> None:
        crop_size_cfg = self.ft_cfg.get("crop_size", None)
        eval_crop_size = (
            tuple(int(dim) for dim in crop_size_cfg)
            if crop_size_cfg is not None
            else self.train_patch_size
        )
        patch_inference_enabled = self.train_patch_size is not None and (
            eval_crop_size is None or self.train_patch_size != eval_crop_size
        )
        overlap_text = (
            f"{self.eval_patch_overlap[0]}x{self.eval_patch_overlap[1]}"
            if self.eval_patch_overlap is not None
            else "n/a"
        )
        self._console_print(
            "[FastMRI trainer] "
            f"mode={self.ft_cfg.mode} | "
            f"precision={self.precision_mode} | "
            f"train_patch_size={self.train_patch_size or 'full'} | "
            f"eval_crop_size={eval_crop_size or 'unknown'} | "
            f"eval_patch_inference={'yes' if patch_inference_enabled else 'no'} | "
            f"eval_patch_overlap={overlap_text} | "
            f"checkpoint_base_model={'yes' if self.gradient_checkpoint_frozen_base else 'no'}"
        )

    def _print_epoch_header(self, epoch: int) -> None:
        self._console_print(
            "[Epoch "
            f"{epoch + 1}/{int(self.ft_cfg.max_epochs)}] "
            f"mode={self.ft_cfg.mode} | "
            f"adapters_active={self.mode_state['adapters_active']} | "
            f"checkpoint_base_model={self._should_checkpoint_frozen_base()} | "
            f"lr_unet={self._format_lr(self._current_group_lr('gfactor_unet'))} | "
            f"lr_adapter={self._format_lr(self._current_group_lr('adapter'))}"
        )

    def _print_metric_summary(self, split: str, metrics: dict[str, float]) -> None:
        if split == "train":
            self._console_print(
                "[Train] "
                f"loss={self._format_metric_value(metrics['loss'])}"
            )
            return

        self._console_print(
            f"[{split.capitalize()}] "
            f"loss={self._format_metric_value(metrics['loss'])} | "
            f"psnr={self._format_metric_value(metrics['psnr'])} | "
            f"ssim={self._format_metric_value(metrics['ssim'])} | "
            f"nmse={self._format_metric_value(metrics['nmse'])}"
        )

    def _autocast_context(self, *, enabled: bool):
        return fastmri_autocast_context(self.device, enabled=enabled)

    def _metadata_to_cpu_numpy(
        self,
        magnitude_prediction: torch.Tensor,
        magnitude_target: torch.Tensor,
        metadata: dict[str, Any],
    ) -> tuple[list[str], list[int], list[np.ndarray], list[np.ndarray]]:
        prediction = magnitude_prediction.detach().cpu().float()
        target = magnitude_target.detach().cpu().float()
        if prediction.ndim != 4 or target.ndim != 4:
            raise ValueError(
                "FastMRI evaluation expects [B, 1, H, W] magnitude tensors, "
                f"got prediction {tuple(prediction.shape)} and target {tuple(target.shape)}"
            )

        mean = torch.as_tensor(metadata["mean"], dtype=torch.float32).reshape(-1, 1, 1, 1)
        std = torch.as_tensor(metadata["std"], dtype=torch.float32).reshape(-1, 1, 1, 1)

        prediction = (prediction * std + mean).squeeze(1)
        target = (target * std + mean).squeeze(1)

        volume_names = [str(name) for name in metadata["volume_name"]]
        slice_indices = [int(idx) for idx in torch.as_tensor(metadata["slice_idx"]).tolist()]
        prediction_list = [prediction[idx].numpy() for idx in range(prediction.shape[0])]
        target_list = [target[idx].numpy() for idx in range(target.shape[0])]
        return volume_names, slice_indices, prediction_list, target_list

    def _should_use_eval_patch_inference(self, noisy: torch.Tensor) -> bool:
        if self.train_patch_size is None:
            return False
        return tuple(int(dim) for dim in noisy.shape[-2:]) != self.train_patch_size

    @staticmethod
    def _sliding_window_positions(size: int, patch_size: int, overlap: int) -> list[int]:
        if patch_size > size:
            raise ValueError(f"Patch size {patch_size} cannot exceed image size {size}")
        stride = patch_size - overlap
        if stride <= 0:
            raise ValueError(
                f"Sliding-window stride must be positive, got patch_size={patch_size}, overlap={overlap}"
            )
        positions = list(range(0, size - patch_size + 1, stride))
        if not positions:
            return [0]
        final_position = size - patch_size
        if positions[-1] != final_position:
            positions.append(final_position)
        return positions

    def _run_eval_patch_inference(self, noisy: torch.Tensor) -> torch.Tensor:
        if self.train_patch_size is None or self.eval_patch_overlap is None:
            raise RuntimeError("Patch inference requested without an active train_patch_size configuration")

        patch_h, patch_w = self.train_patch_size
        overlap_h, overlap_w = self.eval_patch_overlap
        patch_batch_size = max(1, int(self.ft_cfg.get("batch_size", 1)))
        batch_predictions: list[torch.Tensor] = []

        for sample in noisy:
            _, full_h, full_w = sample.shape
            top_positions = self._sliding_window_positions(full_h, patch_h, overlap_h)
            left_positions = self._sliding_window_positions(full_w, patch_w, overlap_w)

            pred_sum = torch.zeros((2, full_h, full_w), device=self.device, dtype=torch.float32)
            weight_sum = torch.zeros((1, full_h, full_w), device=self.device, dtype=torch.float32)
            patch_buffer: list[torch.Tensor] = []
            coord_buffer: list[tuple[int, int]] = []

            def flush_patch_buffer() -> None:
                if not patch_buffer:
                    return
                patch_tensor = torch.stack(patch_buffer, dim=0).to(
                    self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                patch_output = self.model(patch_tensor, checkpoint_base_model=False).squeeze(2).float()
                for index, (top, left) in enumerate(coord_buffer):
                    pred_sum[:, top : top + patch_h, left : left + patch_w] += patch_output[index]
                    weight_sum[:, top : top + patch_h, left : left + patch_w] += 1.0
                patch_buffer.clear()
                coord_buffer.clear()

            for top in top_positions:
                for left in left_positions:
                    patch_buffer.append(sample[:, top : top + patch_h, left : left + patch_w])
                    coord_buffer.append((top, left))
                    if len(patch_buffer) >= patch_batch_size:
                        flush_patch_buffer()

            flush_patch_buffer()
            batch_predictions.append(pred_sum / weight_sum.clamp_min(1.0))

        output = torch.stack(batch_predictions, dim=0).unsqueeze(2)
        if output.shape[0] != noisy.shape[0] or output.shape[-2:] != noisy.shape[-2:]:
            raise RuntimeError(
                "FastMRI patch inference stitch produced an unexpected output shape: "
                f"got {tuple(output.shape)}, expected batch={noisy.shape[0]} "
                f"and spatial={tuple(noisy.shape[-2:])}"
            )
        return output

    def _forward_eval_batch(self, noisy: torch.Tensor) -> torch.Tensor:
        if self._should_use_eval_patch_inference(noisy):
            return self._run_eval_patch_inference(noisy)
        output = self.model(noisy, checkpoint_base_model=False)
        if output.shape[-2:] != noisy.shape[-2:]:
            raise RuntimeError(
                "FastMRI evaluation forward changed the spatial size unexpectedly: "
                f"input {tuple(noisy.shape)}, output {tuple(output.shape)}"
            )
        return output

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self._maybe_activate_adapters(epoch)
        self.model.train()
        checkpoint_frozen_base = self._should_checkpoint_frozen_base()

        loss_sum = 0.0
        step_count = 0
        step_start = time.time()

        progress = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc=f"train {epoch + 1}/{int(self.ft_cfg.max_epochs)}",
            dynamic_ncols=True,
            leave=False,
            disable=not self._use_tqdm,
        )

        for step_idx, (noisy, clean, _noise_sigma, _metadata) in enumerate(progress):
            noisy = noisy.to(self.device, dtype=torch.float32, non_blocking=True)
            clean = clean.to(self.device, dtype=torch.float32, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast_context(enabled=self.use_bf16):
                output = self.model(noisy, checkpoint_base_model=checkpoint_frozen_base)
                magnitude_output = complex_output_to_magnitude(output)
            magnitude_output = magnitude_output.float()
            loss = self.loss_fn(magnitude_output, clean)

            loss.backward()
            self.optimizer.step()

            loss_sum += float(loss.item())
            step_count += 1

            mean_loss = loss_sum / max(step_count, 1)
            if self._use_tqdm:
                progress.set_postfix(
                    loss=f"{float(loss.item()):.4f}",
                    avg=f"{mean_loss:.4f}",
                    lr_u=self._format_lr(self._current_group_lr("gfactor_unet")),
                    lr_a=self._format_lr(self._current_group_lr("adapter")),
                    prec=self.precision_mode,
                    adapters="on" if self.mode_state["adapters_active"] else "off",
                    ckpt="on" if checkpoint_frozen_base else "off",
                )

            if (step_idx + 1) % int(self.ft_cfg.log_every_n_steps) == 0:
                elapsed = time.time() - step_start
                self._log(
                    {
                        "train/loss": mean_loss,
                        "train/epoch": epoch,
                        "train/steps_per_sec": step_count / max(elapsed, 1e-12),
                    }
                )
                if not self._use_tqdm:
                    self._console_print(
                        "[Train step "
                        f"{step_idx + 1}/{len(self.train_loader)}] "
                        f"loss={float(loss.item()):.4f} | "
                        f"avg_loss={mean_loss:.4f} | "
                        f"lr_unet={self._format_lr(self._current_group_lr('gfactor_unet'))} | "
                        f"lr_adapter={self._format_lr(self._current_group_lr('adapter'))} | "
                        f"precision={self.precision_mode} | "
                        f"adapters_active={self.mode_state['adapters_active']} | "
                        f"checkpoint_base_model={checkpoint_frozen_base}"
                    )

        if self._use_tqdm:
            progress.close()

        if self.scheduler is not None:
            self.scheduler.step()

        train_metrics = {"loss": loss_sum / max(step_count, 1)}
        self._log({"train/loss_epoch": train_metrics["loss"], "epoch": epoch})
        return train_metrics

    def evaluate_loader(self, loader: DataLoader | None, *, split: str) -> dict[str, float]:
        if loader is None:
            return {"loss": float("nan"), "psnr": float("nan"), "ssim": float("nan"), "nmse": float("nan")}

        self.model.eval()
        loss_sum = 0.0
        batch_count = 0
        volume_names: list[str] = []
        slice_indices: list[int] = []
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        progress = tqdm(
            loader,
            total=len(loader),
            desc=f"{split} eval",
            dynamic_ncols=True,
            leave=False,
            disable=not self._use_tqdm,
        )

        with torch.inference_mode():
            for batch_idx, (noisy, clean, _noise_sigma, metadata) in enumerate(progress):
                noisy = noisy.to(self.device, dtype=torch.float32, non_blocking=True)
                clean = clean.to(self.device, dtype=torch.float32, non_blocking=True)

                with self._autocast_context(enabled=False):
                    output = self._forward_eval_batch(noisy)
                    magnitude_output = complex_output_to_magnitude(output)
                magnitude_output = magnitude_output.float()
                loss = self.loss_fn(magnitude_output, clean)

                loss_sum += float(loss.item())
                batch_count += 1

                batch_volume_names, batch_slice_indices, batch_predictions, batch_targets = self._metadata_to_cpu_numpy(
                    magnitude_output,
                    clean,
                    metadata,
                )
                volume_names.extend(batch_volume_names)
                slice_indices.extend(batch_slice_indices)
                predictions.extend(batch_predictions)
                targets.extend(batch_targets)
                if self._use_tqdm:
                    progress.set_postfix(loss=f"{loss_sum / max(batch_count, 1):.4f}")
                elif (batch_idx + 1) % max(1, int(self.ft_cfg.log_every_n_steps)) == 0:
                    self._console_print(
                        f"[{split.capitalize()} step {batch_idx + 1}/{len(loader)}] "
                        f"loss={loss_sum / max(batch_count, 1):.4f}"
                    )

        if self._use_tqdm:
            progress.close()

        grouped = group_slices_into_volumes(volume_names, slice_indices, predictions, targets)
        metric_values = compute_volume_metrics(grouped, metric_fns=self.metric_fns)
        summary = {"loss": loss_sum / max(batch_count, 1), **metric_values}
        self._log({f"{split}/{key}": value for key, value in summary.items()} | {"epoch": self.current_epoch})
        return summary

    def _save_checkpoint(self, path: str | Path, *, epoch: int, metrics: dict[str, float]) -> Path:
        return save_fastmri_finetune_checkpoint(
            self.model,
            path,
            mode=self.ft_cfg.mode,
            adapters_active=self.mode_state["adapters_active"],
            config=self.config,
            lora_config=self.config.get("lora"),
            epoch=epoch,
            metrics=metrics,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

    def train(self) -> dict[str, dict[str, float]]:
        results: dict[str, dict[str, float]] = {}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        last_ckpt_path = self.run_dir / "last.pth"
        best_ckpt_path = self.run_dir / "best_psnr.pth"

        trainable, total = count_trainable_parameters(self.model)
        self._log(
            {
                "model/trainable_parameters": trainable,
                "model/total_parameters": total,
                "model/trainable_ratio": trainable / max(total, 1),
                "model/training_precision_bf16": float(self.use_bf16),
            }
        )

        for epoch in range(self.current_epoch, int(self.ft_cfg.max_epochs)):
            self.current_epoch = epoch
            self._print_epoch_header(epoch)
            train_metrics = self.train_one_epoch(epoch)
            results[f"train_epoch_{epoch}"] = train_metrics
            self._print_metric_summary("train", train_metrics)

            val_metrics = {"loss": float("nan"), "psnr": float("nan"), "ssim": float("nan"), "nmse": float("nan")}
            if (
                self.val_loader is not None
                and int(self.ft_cfg.evaluate_every_n_epochs) > 0
                and (epoch + 1) % int(self.ft_cfg.evaluate_every_n_epochs) == 0
            ):
                val_metrics = self.evaluate_loader(self.val_loader, split="val")
                results[f"val_epoch_{epoch}"] = val_metrics
                self._print_metric_summary("val", val_metrics)

            self._save_checkpoint(last_ckpt_path, epoch=epoch, metrics=val_metrics)
            self._console_print(f"[Checkpoint] Saved last checkpoint to {last_ckpt_path}")
            if val_metrics["psnr"] > self.best_val_psnr:
                self.best_val_psnr = val_metrics["psnr"]
                self._save_checkpoint(best_ckpt_path, epoch=epoch, metrics=val_metrics)
                self._console_print(
                    f"[Checkpoint] New best val PSNR {self._format_metric_value(self.best_val_psnr)} -> {best_ckpt_path}"
                )

        if self.test_loader is not None:
            results["test"] = self.evaluate_loader(self.test_loader, split="test")
            self._print_metric_summary("test", results["test"])

        return results
