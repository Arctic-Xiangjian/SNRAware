"""FastMRI compatibility helpers for SNRAware fine-tuning."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from omegaconf import DictConfig, OmegaConf

from snraware.projects.mri.denoising.lora_utils import (
    is_lora_checkpoint,
    resolve_lora_config,
)
from snraware.projects.mri.denoising.model import DenoisingModel

TARGET_REPLACEMENTS = {
    "ifm.model.config.": "snraware.components.model.config.",
    "ifm.mri.denoising.data.": "snraware.projects.mri.denoising.data.",
}

FASTMRI_FINETUNE_CHECKPOINT_TYPE = "snraware_fastmri_finetune_v1"

__all__ = [
    "FASTMRI_FINETUNE_CHECKPOINT_TYPE",
    "NormUnet",
    "SNRAwareWithGFactor",
    "build_fastmri_wrapped_model",
    "has_lora_adapters",
    "is_fastmri_finetune_checkpoint",
    "load_fastmri_finetune_checkpoint",
    "load_model_config_with_legacy_fixes",
    "load_pretrained_base_weights",
    "save_fastmri_finetune_checkpoint",
]


class ConvBlock(nn.Module):
    """A small convolution block used by the FastMRI-style U-Net."""

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransposeConvBlock(nn.Module):
    """Upsampling block used by the FastMRI-style U-Net."""

    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Unet2D(nn.Module):
    """A compact 2D U-Net matching the FastMRI NormUnet structure."""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        if num_pool_layers < 1:
            raise ValueError("num_pool_layers must be >= 1")

        self.down_sample_layers = nn.ModuleList()
        current_chans = chans
        self.down_sample_layers.append(ConvBlock(in_chans, current_chans, drop_prob))
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(current_chans, current_chans * 2, drop_prob))
            current_chans *= 2

        self.conv = ConvBlock(current_chans, current_chans * 2, drop_prob)

        self.up_transpose_conv = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(current_chans * 2, current_chans))
            self.up_conv.append(ConvBlock(current_chans * 2, current_chans, drop_prob))
            current_chans //= 2

        self.up_transpose_conv.append(TransposeConvBlock(current_chans * 2, current_chans))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(current_chans * 2, current_chans, drop_prob),
                nn.Conv2d(current_chans, out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stack = []
        output = x

        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, ceil_mode=True)

        output = self.conv(output)

        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv, strict=True):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            if output.shape[-2:] != downsample_layer.shape[-2:]:
                output = F.interpolate(
                    output,
                    size=downsample_layer.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class NormUnet(nn.Module):
    """FastMRI-style U-Net adapted for real-valued g-factor prediction.

    Input is expected in complex-last format: ``[B, C, H, W, 2]``.
    Output is a real-valued tensor of shape ``[B, out_chans, H, W]``.
    """

    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 1,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.unet = Unet2D(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
        )

    @staticmethod
    def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[-1] != 2:
            raise ValueError(f"Expected complex-last tensor [B, C, H, W, 2], got {tuple(x.shape)}")
        batch, chans, height, width, two = x.shape
        return x.permute(0, 1, 4, 2, 3).reshape(batch, chans * two, height, width)

    @staticmethod
    def norm(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, chans, height, width = x.shape
        if chans % 2 != 0:
            raise ValueError("NormUnet expects an even number of channels after complex packing")
        x_grouped = x.reshape(batch, 2, (chans // 2) * height * width)
        mean = x_grouped.mean(dim=2).reshape(batch, 2, 1, 1)
        std = x_grouped.std(dim=2).reshape(batch, 2, 1, 1)
        mean = mean.repeat_interleave(chans // 2, dim=1)
        std = std.repeat_interleave(chans // 2, dim=1)
        return (x - mean) / std, mean, std

    @staticmethod
    def pad(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        _, _, height, width = x.shape
        height_mult = ((height - 1) | 15) + 1
        width_mult = ((width - 1) | 15) + 1
        pad_height = [math.floor((height_mult - height) / 2), math.ceil((height_mult - height) / 2)]
        pad_width = [math.floor((width_mult - width) / 2), math.ceil((width_mult - width) / 2)]
        x = F.pad(x, [pad_width[0], pad_width[1], pad_height[0], pad_height[1]])
        return x, (pad_height[0], pad_height[1], pad_width[0], pad_width[1])

    @staticmethod
    def unpad(x: torch.Tensor, pad_sizes: tuple[int, int, int, int]) -> torch.Tensor:
        pad_top, pad_bottom, pad_left, pad_right = pad_sizes
        if pad_bottom == 0:
            h_slice = slice(pad_top, None)
        else:
            h_slice = slice(pad_top, -pad_bottom)
        if pad_right == 0:
            w_slice = slice(pad_left, None)
        else:
            w_slice = slice(pad_left, -pad_right)
        return x[..., h_slice, w_slice]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.complex_to_chan_dim(x)
        # FastMRI samples are already normalized in the dataset. Avoid a second
        # per-sample std normalization here so low-variance background patches
        # do not divide by zero and produce NaNs during warmup training.
        x, pad_sizes = self.pad(x)
        x = self.unet(x)
        return self.unpad(x, pad_sizes)


class SNRAwareWithGFactor(nn.Module):
    """Wrap SNRAware with a g-factor predictor for 2-channel FastMRI inputs."""

    def __init__(
        self,
        base_model: DenoisingModel,
        gfactor_unet: nn.Module | None = None,
        *,
        use_unet: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.gfactor_unet = gfactor_unet or NormUnet()
        self.use_unet = bool(use_unet)
        self.config = getattr(base_model, "config", None)
        self.last_gfactor_stats: dict[str, float] | None = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _prepare_2ch_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == 2:
            return x.unsqueeze(2)
        if x.ndim == 5 and x.shape[1] == 2:
            if x.shape[2] != 1:
                raise ValueError(
                    "FastMRI compatibility path only supports 2D slices with T=1 for 2-channel input"
                )
            return x
        raise ValueError(f"Unsupported 2-channel input shape: {tuple(x.shape)}")

    def predict_gfactor(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_2ch_input(x)
        if not self.use_unet:
            return torch.ones(
                (x.shape[0], 1, x.shape[-2], x.shape[-1]),
                device=x.device,
                dtype=x.dtype,
            )
        x_2d = x.squeeze(2)
        complex_last = x_2d.permute(0, 2, 3, 1).unsqueeze(1).contiguous()
        gfactor = torch.abs(self.gfactor_unet(complex_last))
        return gfactor

    def _record_gfactor_stats(self, gfactor: torch.Tensor) -> None:
        flat = gfactor.detach().float().reshape(-1)
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            self.last_gfactor_stats = {"mean": float("nan"), "p95": float("nan"), "max": float("nan")}
            return
        self.last_gfactor_stats = {
            "mean": float(finite.mean().item()),
            "p95": float(torch.quantile(finite, 0.95).item()),
            "max": float(finite.max().item()),
        }

    def _forward_base_model(
        self,
        x_with_gfactor: torch.Tensor,
        *,
        checkpoint_base_model: bool = False,
    ) -> torch.Tensor:
        if checkpoint_base_model:
            return checkpoint_utils.checkpoint(
                self.base_model,
                x_with_gfactor,
                use_reentrant=False,
            )
        return self.base_model(x_with_gfactor)

    def forward_fastmri_2ch(
        self,
        x: torch.Tensor,
        *,
        checkpoint_base_model: bool = False,
    ) -> torch.Tensor:
        x = self._prepare_2ch_input(x)
        gfactor = self.predict_gfactor(x).unsqueeze(2)
        self._record_gfactor_stats(gfactor)
        x_with_gfactor = torch.cat([x, gfactor], dim=1)
        return self._forward_base_model(
            x_with_gfactor,
            checkpoint_base_model=checkpoint_base_model,
        )

    def forward(self, x: torch.Tensor, *, checkpoint_base_model: bool = False) -> torch.Tensor:
        if x.ndim == 5 and x.shape[1] == 3:
            return self.base_model(x)
        if x.ndim == 4 and x.shape[1] == 3:
            return self.base_model(x.unsqueeze(2))
        if x.shape[1] != 2:
            raise ValueError(f"Expected 2 or 3 channels, got shape {tuple(x.shape)}")
        return self.forward_fastmri_2ch(
            x,
            checkpoint_base_model=checkpoint_base_model,
        )


def _replace_legacy_targets(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _replace_legacy_targets(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_replace_legacy_targets(value) for value in obj]
    if isinstance(obj, str):
        for old, new in TARGET_REPLACEMENTS.items():
            if obj.startswith(old):
                return new + obj[len(old) :]
    return obj


def load_model_config_with_legacy_fixes(config_path: str | Path) -> DictConfig:
    """Load a model config and remap legacy `_target_` values when needed."""
    raw_config = OmegaConf.load(config_path)
    config = OmegaConf.create(_replace_legacy_targets(OmegaConf.to_container(raw_config, resolve=False)))
    if not isinstance(config, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(config).__name__}")
    return config


def _load_raw_state_dict(weight_path: str | Path) -> tuple[dict[str, torch.Tensor] | None, str]:
    path = Path(weight_path)
    if not path.exists():
        raise FileNotFoundError(f"Base model checkpoint does not exist: {path}")

    if path.suffix == ".pth":
        status = torch.load(path, map_location="cpu")
        if is_lora_checkpoint(status):
            raise ValueError(
                "Expected a frozen base-model checkpoint, but received a LoRA adapter checkpoint."
            )
        if isinstance(status, dict) and "model_state_dict" in status:
            status = status["model_state_dict"]
        if isinstance(status, dict):
            tensors = {key: value.detach().cpu() for key, value in status.items() if torch.is_tensor(value)}
            if tensors:
                return tensors, "torch"

    try:
        scripted = torch.jit.load(str(path), map_location="cpu")
        return {key: value.detach().cpu() for key, value in scripted.state_dict().items()}, "jit"
    except Exception:
        pass

    try:
        status = torch.load(path, map_location="cpu")
        if isinstance(status, dict) and "model_state_dict" in status:
            status = status["model_state_dict"]
        if isinstance(status, dict):
            tensors = {key: value.detach().cpu() for key, value in status.items() if torch.is_tensor(value)}
            if tensors:
                return tensors, "torch"
    except Exception as exc:
        raise RuntimeError(f"Could not load model weights from {path}") from exc

    raise RuntimeError(f"No compatible state dict could be loaded from {path}")


def _strip_known_prefixes(
    state_dict: dict[str, torch.Tensor], prefixes: Iterable[str]
) -> dict[str, torch.Tensor]:
    stripped = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        stripped[new_key] = value
    return stripped


def _best_shape_compatible_state(
    model: nn.Module, raw_state: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], int, int, list[str]]:
    model_state = model.state_dict()
    candidates = [
        raw_state,
        _strip_known_prefixes(raw_state, ["module."]),
        _strip_known_prefixes(raw_state, ["model."]),
        _strip_known_prefixes(raw_state, ["base_model."]),
        _strip_known_prefixes(raw_state, ["base_model.model."]),
        _strip_known_prefixes(raw_state, ["module.model."]),
    ]

    best_filtered: dict[str, torch.Tensor] = {}
    best_match = -1
    best_mismatch = math.inf
    best_mismatch_keys: list[str] = []
    for candidate in candidates:
        filtered = {}
        mismatched_shapes = 0
        mismatch_keys: list[str] = []
        for key, value in candidate.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    filtered[key] = value
                else:
                    mismatched_shapes += 1
                    mismatch_keys.append(key)
        if len(filtered) > best_match or (
            len(filtered) == best_match and mismatched_shapes < best_mismatch
        ):
            best_filtered = filtered
            best_match = len(filtered)
            best_mismatch = mismatched_shapes
            best_mismatch_keys = mismatch_keys

    return best_filtered, int(best_match), int(best_mismatch), best_mismatch_keys


def load_pretrained_base_weights(
    model: nn.Module, weight_path: str | Path
) -> tuple[int, int, int, str, list[str]]:
    """Load the best shape-compatible weights from a checkpoint into a base model."""
    raw_state, source = _load_raw_state_dict(weight_path)
    filtered, matched, mismatched, mismatch_keys = _best_shape_compatible_state(model, raw_state)
    if matched <= 0:
        raise RuntimeError(f"No compatible weights matched the current model from {weight_path}")
    model.load_state_dict(filtered, strict=False)
    return matched, mismatched, len(model.state_dict()), source, mismatch_keys


def has_lora_adapters(model: nn.Module) -> bool:
    """Return whether the model currently contains LoRA parameters."""
    return any(".lora_A." in name or ".lora_B." in name for name, _ in model.named_parameters())


def _resolve_fastmri_train_pre_post(config: Any | None) -> bool:
    if config is None:
        return False
    if isinstance(config, DictConfig):
        value = OmegaConf.select(config, "fastmri_finetune.train_pre_post")
        return bool(value) if value is not None else False
    if isinstance(config, dict):
        fastmri_config = config.get("fastmri_finetune", {})
        if isinstance(fastmri_config, dict):
            return bool(fastmri_config.get("train_pre_post", False))
    return False


def _resolve_fastmri_use_unet(config: Any | None) -> bool:
    if config is None:
        return True
    if isinstance(config, DictConfig):
        value = OmegaConf.select(config, "fastmri_finetune.use_unet")
        return bool(value) if value is not None else True
    if isinstance(config, dict):
        fastmri_config = config.get("fastmri_finetune", {})
        if isinstance(fastmri_config, dict):
            return bool(fastmri_config.get("use_unet", True))
    return True


def extract_fastmri_adapter_state(
    model: SNRAwareWithGFactor,
    *,
    include_pre_post: bool,
) -> dict[str, torch.Tensor]:
    """Extract FastMRI adapter state, optionally including pre/post layers."""
    if not has_lora_adapters(model.base_model):
        return {}
    selected = {}
    for name, tensor in model.base_model.state_dict().items():
        if ".lora_A." in name or ".lora_B." in name or (
            include_pre_post and (name.startswith("pre.") or name.startswith("post."))
        ):
            selected[name] = tensor.detach().cpu()
    return selected


def _is_adapter_state_key(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name or name.startswith("pre.") or name.startswith("post.")


def is_fastmri_finetune_checkpoint(payload: Any) -> bool:
    """Return whether a payload matches the FastMRI fine-tune checkpoint format."""
    return isinstance(payload, dict) and payload.get("checkpoint_type") == FASTMRI_FINETUNE_CHECKPOINT_TYPE


def save_fastmri_finetune_checkpoint(
    model: SNRAwareWithGFactor,
    checkpoint_path: str | Path,
    *,
    mode: str,
    adapters_active: bool | None = None,
    config: Any | None = None,
    lora_config: Any | None = None,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> Path:
    """Save a FastMRI fine-tune checkpoint without including frozen backbone weights."""
    checkpoint_path = Path(checkpoint_path)
    resolved_lora_config = resolve_lora_config(
        model_config=getattr(model.base_model, "config", None),
        lora_config=lora_config,
    )
    train_pre_post = _resolve_fastmri_train_pre_post(config)
    payload = {
        "checkpoint_type": FASTMRI_FINETUNE_CHECKPOINT_TYPE,
        "mode": mode,
        "adapters_active": bool(adapters_active) if adapters_active is not None else None,
        "use_unet": _resolve_fastmri_use_unet(config),
        "epoch": epoch,
        "metrics": metrics or {},
        "config": OmegaConf.to_container(config, resolve=False) if isinstance(config, DictConfig) else config,
        "lora_config": asdict(resolved_lora_config),
        "train_pre_post": train_pre_post,
        "gfactor_unet": {key: value.detach().cpu() for key, value in model.gfactor_unet.state_dict().items()},
        "lora_adapter": extract_fastmri_adapter_state(
            model,
            include_pre_post=train_pre_post,
        ),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_fastmri_finetune_checkpoint(
    model: SNRAwareWithGFactor,
    checkpoint: str | Path | dict[str, Any],
    *,
    apply_lora_fn: Any | None = None,
    lora_config: Any | None = None,
) -> tuple[list[str], list[str]]:
    """Load a FastMRI fine-tune checkpoint into a wrapped model."""
    payload = checkpoint
    if isinstance(checkpoint, str | Path):
        payload = torch.load(checkpoint, map_location="cpu")

    if not is_fastmri_finetune_checkpoint(payload):
        raise ValueError("Provided checkpoint is not a FastMRI fine-tune checkpoint")

    model.gfactor_unet.load_state_dict(payload["gfactor_unet"], strict=True)

    lora_state = payload.get("lora_adapter", {})
    if not lora_state:
        return [], []

    if apply_lora_fn is None:
        raise ValueError("apply_lora_fn is required to restore a checkpoint containing LoRA adapters")

    resolved_lora_config = lora_config if lora_config is not None else payload.get("lora_config")
    if lora_config is not None and not resolve_lora_config(lora_config=lora_config).enabled:
        resolved_lora_config = payload.get("lora_config", lora_config)
    if not has_lora_adapters(model.base_model):
        apply_lora_fn(model.base_model, lora_config=resolved_lora_config)
    missing, unexpected = model.base_model.load_state_dict(lora_state, strict=False)
    expected_adapter_keys = set(lora_state)
    filtered_missing = [name for name in missing if name in expected_adapter_keys]
    return filtered_missing, unexpected


def build_fastmri_wrapped_model(
    *,
    base_config_path: str | Path,
    base_checkpoint_path: str | Path,
    height: int,
    width: int,
    depth: int = 1,
    lora_config: Any | None = None,
    gfactor_unet_kwargs: dict[str, Any] | None = None,
    use_unet: bool = True,
) -> tuple[SNRAwareWithGFactor, DictConfig, dict[str, Any]]:
    """Build a FastMRI fine-tune model from a base SNRAware checkpoint."""
    config = load_model_config_with_legacy_fixes(base_config_path)
    native_cutout_shape = OmegaConf.select(config, "dataset.cutout_shape")
    native_spatial_size = None
    if native_cutout_shape is not None and len(native_cutout_shape) >= 2:
        native_spatial_size = [int(native_cutout_shape[0]), int(native_cutout_shape[1])]
    if OmegaConf.select(config, "dataset") is None:
        config.dataset = OmegaConf.create({})
    config.dataset.cutout_shape = [height, width, depth]
    if lora_config is not None:
        config.lora = OmegaConf.create(
            OmegaConf.to_container(lora_config, resolve=True) if isinstance(lora_config, DictConfig) else lora_config
        )

    base_model = DenoisingModel(config=config, D=depth, H=height, W=width)
    matched, mismatched, total, source, mismatch_keys = load_pretrained_base_weights(
        base_model, base_checkpoint_path
    )
    wrapped_model = SNRAwareWithGFactor(
        base_model=base_model,
        gfactor_unet=NormUnet(**(gfactor_unet_kwargs or {})),
        use_unet=use_unet,
    )

    load_info = {
        "matched_keys": matched,
        "mismatched_keys": mismatched,
        "total_model_keys": total,
        "weight_source": source,
        "mismatch_key_examples": mismatch_keys[:10],
        "native_spatial_size": native_spatial_size,
        "model_spatial_size": [int(height), int(width)],
    }
    return wrapped_model, config, load_info
