"""Custom FastMRI fine-tuning loop for SNRAware."""

from __future__ import annotations

import random
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

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
    "build_fastmri_dataloaders",
    "complex_output_to_magnitude",
    "compute_volume_metrics",
    "configure_model_for_finetune_mode",
    "group_slices_into_volumes",
    "seed_everything",
]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def complex_output_to_magnitude(output: torch.Tensor) -> torch.Tensor:
    """Convert SNRAware 2-channel complex output to magnitude without epsilon."""
    if output.ndim != 5 or output.shape[1] != 2:
        raise ValueError(f"Expected [B, 2, T, H, W] output, got {tuple(output.shape)}")
    real = output[:, 0:1, ...]
    imag = output[:, 1:2, ...]
    return torch.sqrt(real.square() + imag.square())


def _is_adapter_parameter(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name or name.startswith("pre.") or name.startswith("post.")


def _set_module_trainable(module: nn.Module, flag: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = flag


def _set_adapter_trainable(base_model: nn.Module, flag: bool) -> None:
    for name, parameter in base_model.named_parameters():
        if _is_adapter_parameter(name):
            parameter.requires_grad = flag


def _get_adapter_parameters(base_model: nn.Module) -> list[nn.Parameter]:
    return [parameter for name, parameter in base_model.named_parameters() if _is_adapter_parameter(name)]


def configure_model_for_finetune_mode(
    model: SNRAwareWithGFactor,
    *,
    mode: str,
    lora_config: Any | None = None,
) -> dict[str, Any]:
    """Configure trainable parameters for the requested FastMRI fine-tune mode."""
    supported_modes = {"unet_only", "unet_and_lora", "lora_only", "warmup_then_both"}
    if mode not in supported_modes:
        raise ValueError(f"Unsupported fine-tune mode: {mode}")

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

        apply_lora_to_model(model.base_model, lora_config=lora_config)

        if mode == "unet_and_lora":
            _set_module_trainable(model.gfactor_unet, True)
            _set_adapter_trainable(model.base_model, True)
            adapters_active = True
        elif mode == "lora_only":
            _set_module_trainable(model.gfactor_unet, False)
            _set_adapter_trainable(model.base_model, True)
            adapters_active = True
        else:
            _set_module_trainable(model.gfactor_unet, True)
            _set_adapter_trainable(model.base_model, False)

    return {
        "mode": mode,
        "adapters_active": adapters_active,
        "has_lora": has_lora_adapters(model.base_model),
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

    def _build_dataset(root: str | Path | None):
        if root in (None, "", "null"):
            return None
        return FastMRISNRAwareDataset(
            root=root,
            challenge=ft_cfg.challenge,
            sample_rate=ft_cfg.sample_rate,
            volume_sample_rate=ft_cfg.volume_sample_rate,
            use_dataset_cache=ft_cfg.use_dataset_cache,
            dataset_cache_file=ft_cfg.dataset_cache_file,
            scanner_models=ft_cfg.scanner_models,
            acc_factor=ft_cfg.acc_factor,
            crop_size=tuple(ft_cfg.crop_size),
            strict_latent_feature=False,
            deterministic_mask_from_name=bool(ft_cfg.deterministic_mask_from_name),
            sample_seed=ft_cfg.sample_seed,
        )

    train_dataset = _build_dataset(ft_cfg.train_root)
    if train_dataset is None:
        raise ValueError("fastmri_finetune.train_root must be provided")

    val_dataset = _build_dataset(ft_cfg.val_root)
    test_dataset = _build_dataset(ft_cfg.test_root)

    loader_kwargs = {
        "batch_size": int(ft_cfg.batch_size),
        "num_workers": int(ft_cfg.num_workers),
        "pin_memory": bool(ft_cfg.pin_memory),
        "persistent_workers": bool(ft_cfg.persistent_workers) if int(ft_cfg.num_workers) > 0 else False,
        "drop_last": False,
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
    ):
        self.model = model.to(device=device, dtype=torch.float32)
        self.config = config
        self.device = device
        self.run_dir = Path(run_dir)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.metric_fns = metric_fns
        self.wandb_run = wandb_run
        self.ft_cfg = config.fastmri_finetune
        self.loss_fn = nn.L1Loss()
        self.mode_state = configure_model_for_finetune_mode(
            self.model,
            mode=self.ft_cfg.mode,
            lora_config=config.get("lora"),
        )
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.current_epoch = 0
        self.best_val_psnr = float("-inf")

        if self.ft_cfg.resume_from:
            self._load_resume_state(self.ft_cfg.resume_from)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        param_groups = []
        gfactor_params = list(self.model.gfactor_unet.parameters())
        adapter_params = _get_adapter_parameters(self.model.base_model)

        if any(parameter.requires_grad for parameter in gfactor_params) or self.ft_cfg.mode == "warmup_then_both":
            param_groups.append(
                {
                    "name": "gfactor_unet",
                    "params": gfactor_params,
                    "lr": float(self.ft_cfg.unet_lr),
                    "weight_decay": float(self.ft_cfg.weight_decay),
                }
            )

        if adapter_params:
            adapter_lr = (
                0.0
                if self.ft_cfg.mode == "warmup_then_both" and not self.mode_state["adapters_active"]
                else float(self.ft_cfg.adapter_lr)
            )
            param_groups.append(
                {
                    "name": "adapter",
                    "params": adapter_params,
                    "lr": adapter_lr,
                    "weight_decay": float(self.ft_cfg.weight_decay),
                }
            )

        if not param_groups:
            raise RuntimeError("No parameter groups were created for FastMRI fine-tuning")

        return torch.optim.AdamW(param_groups)

    def _build_scheduler(self) -> Any | None:
        t_max = int(self.ft_cfg.scheduler_t_max)
        if t_max <= 0:
            return None
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max)

    def _load_resume_state(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
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

    def _maybe_activate_adapters(self, epoch: int) -> None:
        if self.ft_cfg.mode != "warmup_then_both":
            return
        if self.mode_state["adapters_active"]:
            return
        if epoch < int(self.ft_cfg.warmup_epochs):
            return

        _set_adapter_trainable(self.model.base_model, True)
        self.mode_state["adapters_active"] = True
        for group in self.optimizer.param_groups:
            if group.get("name") == "adapter":
                group["lr"] = float(self.ft_cfg.adapter_lr)

    def _log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log(payload, step=step)

    def _metadata_to_cpu_numpy(
        self,
        magnitude_prediction: torch.Tensor,
        magnitude_target: torch.Tensor,
        metadata: dict[str, Any],
    ) -> tuple[list[str], list[int], list[np.ndarray], list[np.ndarray]]:
        prediction = magnitude_prediction.detach().cpu().float().squeeze(1).squeeze(1).numpy()
        target = magnitude_target.detach().cpu().float().squeeze(1).squeeze(1).numpy()

        mean = torch.as_tensor(metadata["mean"], dtype=torch.float32).reshape(-1, 1, 1).numpy()
        std = torch.as_tensor(metadata["std"], dtype=torch.float32).reshape(-1, 1, 1).numpy()

        prediction = prediction * std + mean
        target = target * std + mean

        volume_names = [str(name) for name in metadata["volume_name"]]
        slice_indices = [int(idx) for idx in torch.as_tensor(metadata["slice_idx"]).tolist()]
        prediction_list = [prediction[idx] for idx in range(prediction.shape[0])]
        target_list = [target[idx] for idx in range(target.shape[0])]
        return volume_names, slice_indices, prediction_list, target_list

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self._maybe_activate_adapters(epoch)
        self.model.train()

        loss_sum = 0.0
        step_count = 0
        step_start = time.time()

        for step_idx, (noisy, clean, _noise_sigma, _metadata) in enumerate(self.train_loader):
            noisy = noisy.to(self.device, dtype=torch.float32, non_blocking=True)
            clean = clean.to(self.device, dtype=torch.float32, non_blocking=True)

            output = self.model(noisy)
            magnitude_output = complex_output_to_magnitude(output)
            loss = self.loss_fn(magnitude_output, clean)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            loss_sum += float(loss.item())
            step_count += 1

            if (step_idx + 1) % int(self.ft_cfg.log_every_n_steps) == 0:
                elapsed = time.time() - step_start
                self._log(
                    {
                        "train/loss": loss_sum / step_count,
                        "train/epoch": epoch,
                        "train/steps_per_sec": step_count / max(elapsed, 1e-12),
                    }
                )

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

        with torch.inference_mode():
            for noisy, clean, _noise_sigma, metadata in loader:
                noisy = noisy.to(self.device, dtype=torch.float32, non_blocking=True)
                clean = clean.to(self.device, dtype=torch.float32, non_blocking=True)

                output = self.model(noisy)
                magnitude_output = complex_output_to_magnitude(output)
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
            }
        )

        for epoch in range(self.current_epoch, int(self.ft_cfg.max_epochs)):
            self.current_epoch = epoch
            train_metrics = self.train_one_epoch(epoch)
            results[f"train_epoch_{epoch}"] = train_metrics

            val_metrics = {"loss": float("nan"), "psnr": float("nan"), "ssim": float("nan"), "nmse": float("nan")}
            if (
                self.val_loader is not None
                and int(self.ft_cfg.evaluate_every_n_epochs) > 0
                and (epoch + 1) % int(self.ft_cfg.evaluate_every_n_epochs) == 0
            ):
                val_metrics = self.evaluate_loader(self.val_loader, split="val")
                results[f"val_epoch_{epoch}"] = val_metrics

            self._save_checkpoint(last_ckpt_path, epoch=epoch, metrics=val_metrics)
            if val_metrics["psnr"] > self.best_val_psnr:
                self.best_val_psnr = val_metrics["psnr"]
                self._save_checkpoint(best_ckpt_path, epoch=epoch, metrics=val_metrics)

        if self.test_loader is not None:
            results["test"] = self.evaluate_loader(self.test_loader, split="test")

        return results
