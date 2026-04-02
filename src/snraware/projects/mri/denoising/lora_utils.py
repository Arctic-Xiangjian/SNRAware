"""LoRA utilities for parameter-efficient finetuning of the denoising model."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from snraware.components.model import Conv2DExt, Conv3DExt, LinearGrid3DExt, LinearGridExt

DEFAULT_TARGET_MODULES = [
    r"\.attn\.key$",
    r"\.attn\.query$",
    r"\.attn\.value$",
    r"\.attn\.output_proj$",
    r"\.mlp\.0$",
    r"\.mlp\.2$",
]

LORA_CHECKPOINT_TYPE = "snraware_lora_adapter_v1"

__all__ = [
    "DEFAULT_TARGET_MODULES",
    "LORA_CHECKPOINT_TYPE",
    "LoraConfig",
    "apply_lora_to_model",
    "count_trainable_parameters",
    "extract_lora_state_dict",
    "is_lora_checkpoint",
    "is_lora_enabled",
    "load_lora_checkpoint",
    "resolve_lora_config",
    "save_lora_checkpoint",
]


@dataclass
class LoraConfig:
    """Configuration for LoRA injection."""

    enabled: bool = False
    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))


class LoRALinear(nn.Module):
    """LoRA wrapper for ``nn.Linear``."""

    def __init__(self, base_layer: nn.Linear, r: int, lora_alpha: float, lora_dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0 for linear layers")

        self.base_layer = base_layer
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)

        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


class LoRAConv2d(nn.Module):
    """LoRA wrapper for ``nn.Conv2d``."""

    def __init__(self, base_layer: nn.Conv2d, r: int, lora_alpha: float, lora_dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0 for conv2d layers")
        if base_layer.groups != 1:
            raise NotImplementedError("LoRAConv2d currently supports groups=1 only")

        self.base_layer = base_layer
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        self.lora_A = nn.Conv2d(
            in_channels=base_layer.in_channels,
            out_channels=r,
            kernel_size=base_layer.kernel_size,
            stride=base_layer.stride,
            padding=base_layer.padding,
            dilation=base_layer.dilation,
            groups=base_layer.groups,
            bias=False,
            padding_mode=base_layer.padding_mode,
        )
        self.lora_B = nn.Conv2d(
            in_channels=r,
            out_channels=base_layer.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)

        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


class LoRAConv3d(nn.Module):
    """LoRA wrapper for ``nn.Conv3d``."""

    def __init__(self, base_layer: nn.Conv3d, r: int, lora_alpha: float, lora_dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0 for conv3d layers")
        if base_layer.groups != 1:
            raise NotImplementedError("LoRAConv3d currently supports groups=1 only")

        self.base_layer = base_layer
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        self.lora_A = nn.Conv3d(
            in_channels=base_layer.in_channels,
            out_channels=r,
            kernel_size=base_layer.kernel_size,
            stride=base_layer.stride,
            padding=base_layer.padding,
            dilation=base_layer.dilation,
            groups=base_layer.groups,
            bias=False,
            padding_mode=base_layer.padding_mode,
        )
        self.lora_B = nn.Conv3d(
            in_channels=r,
            out_channels=base_layer.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)

        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


def _get_config_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, LoraConfig):
        return asdict(config)
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    if isinstance(config, dict):
        return config
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return {}


def resolve_lora_config(
    model_config: Any | None = None, lora_config: Any | None = None
) -> LoraConfig:
    """Resolve LoRA config from explicit value or model config."""
    config_obj = lora_config
    if config_obj is None and model_config is not None:
        if isinstance(model_config, DictConfig):
            config_obj = OmegaConf.select(model_config, "lora")
        elif hasattr(model_config, "lora"):
            config_obj = model_config.lora
        elif isinstance(model_config, dict):
            config_obj = model_config.get("lora")

    config_dict = _get_config_dict(config_obj)
    if not config_dict:
        return LoraConfig()

    target_modules = config_dict.get("target_modules", None)
    if not target_modules:
        target_modules = list(DEFAULT_TARGET_MODULES)

    enabled = config_dict.get("enabled", config_dict.get("enable", False))
    r = int(config_dict.get("r", 8))
    lora_alpha = float(config_dict.get("lora_alpha", 16.0))
    lora_dropout = float(config_dict.get("lora_dropout", 0.0))

    if r <= 0 and bool(enabled):
        raise ValueError(f"LoRA rank must be > 0 when enabled, got {r}")

    return LoraConfig(
        enabled=bool(enabled),
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[str(t) for t in target_modules],
    )


def is_lora_enabled(model_config: Any | None = None, lora_config: Any | None = None) -> bool:
    """Return whether LoRA is enabled by config."""
    return resolve_lora_config(model_config=model_config, lora_config=lora_config).enabled


def _get_child_module(parent: nn.Module, name: str) -> nn.Module:
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and name.isdigit():
        return parent[int(name)]
    if isinstance(parent, nn.ModuleDict):
        return parent[name]
    return getattr(parent, name)


def _set_child_module(parent: nn.Module, name: str, module: nn.Module) -> None:
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and name.isdigit():
        parent[int(name)] = module
        return
    if isinstance(parent, nn.ModuleDict):
        parent[name] = module
        return
    setattr(parent, name, module)


def _find_parent_module(root: nn.Module, module_path: str) -> tuple[nn.Module, str]:
    parts = module_path.split(".")
    if len(parts) == 1:
        return root, parts[0]

    parent = root
    for part in parts[:-1]:
        parent = _get_child_module(parent, part)
    return parent, parts[-1]


def _matches_any_pattern(name: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, name) is not None for pattern in patterns)


def _inject_lora_into_module(module: nn.Module, lora_cfg: LoraConfig) -> bool:
    if isinstance(module, nn.Identity):
        return False

    if isinstance(module, LoRALinear | LoRAConv2d | LoRAConv3d):
        return True

    if isinstance(module, Conv2DExt):
        if isinstance(module.conv, LoRAConv2d):
            return True
        module.conv = LoRAConv2d(
            base_layer=module.conv,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
        )
        return True

    if isinstance(module, Conv3DExt):
        if isinstance(module.conv, LoRAConv3d):
            return True
        module.conv = LoRAConv3d(
            base_layer=module.conv,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
        )
        return True

    if isinstance(module, LinearGridExt):
        if isinstance(module.linear, LoRALinear):
            return True
        module.linear = LoRALinear(
            base_layer=module.linear,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
        )
        return True

    if isinstance(module, LinearGrid3DExt):
        if isinstance(module.linear, LoRALinear):
            return True
        module.linear = LoRALinear(
            base_layer=module.linear,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
        )
        return True

    return False


def _wrap_module_if_needed(module: nn.Module, lora_cfg: LoraConfig) -> nn.Module:
    if isinstance(module, nn.Linear):
        return LoRALinear(
            base_layer=module,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
        )
    if isinstance(module, nn.Conv2d):
        return LoRAConv2d(
            base_layer=module,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
        )
    if isinstance(module, nn.Conv3d):
        return LoRAConv3d(
            base_layer=module,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
        )
    return module


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _mark_lora_trainable(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            param.requires_grad = True


def apply_lora_to_model(
    model: nn.Module, lora_config: Any | None = None, target_root_attr: str = "bk"
) -> nn.Module:
    """
    Inject LoRA adapters into attention projections and FFN/mixer layers.

    This function freezes all model weights, injects LoRA into the target backbone modules,
    then unfreezes only LoRA parameters plus ``pre``/``post`` layers.
    """

    lora_cfg = resolve_lora_config(model_config=getattr(model, "config", None), lora_config=lora_config)
    if not lora_cfg.enabled:
        return model

    if not hasattr(model, target_root_attr):
        raise AttributeError(f"Expected model to have '{target_root_attr}' for LoRA injection")

    for p in model.parameters():
        p.requires_grad = False

    target_root = getattr(model, target_root_attr)
    named_modules = list(target_root.named_modules())

    matched_paths = [
        name
        for name, _module in named_modules
        if name and _matches_any_pattern(name, lora_cfg.target_modules)
    ]

    injected_count = 0
    unsupported_paths = []
    for path in matched_paths:
        parent, child_name = _find_parent_module(target_root, path)
        module = _get_child_module(parent, child_name)

        injected_as_leaf = _inject_lora_into_module(module, lora_cfg)
        if injected_as_leaf:
            injected_count += 1
            continue

        wrapped = _wrap_module_if_needed(module, lora_cfg)
        if wrapped is not module:
            _set_child_module(parent, child_name, wrapped)
            injected_count += 1
        elif not isinstance(module, nn.Identity):
            unsupported_paths.append((path, type(module).__name__))

    if injected_count == 0:
        raise RuntimeError(
            f"No LoRA adapters were injected. Check target modules: {lora_cfg.target_modules}"
        )
    if unsupported_paths:
        unsupported_str = ", ".join([f"{name} ({mod_type})" for name, mod_type in unsupported_paths])
        raise RuntimeError(f"Unsupported LoRA target modules found: {unsupported_str}")

    if hasattr(model, "pre"):
        _set_requires_grad(model.pre, True)
    if hasattr(model, "post"):
        _set_requires_grad(model.post, True)
    _mark_lora_trainable(model)

    return model


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def extract_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract LoRA weights and pre/post layers only."""
    full_state_dict = model.state_dict()
    selected = {}
    for name, tensor in full_state_dict.items():
        if ".lora_A." in name or ".lora_B." in name or name.startswith("pre.") or name.startswith("post."):
            selected[name] = tensor.detach().cpu()
    return selected


def save_lora_checkpoint(
    model: nn.Module, checkpoint_path: str | Path, lora_config: Any | None = None
) -> Path:
    """Save adapter-only checkpoint (LoRA + pre/post)."""
    lora_cfg = resolve_lora_config(model_config=getattr(model, "config", None), lora_config=lora_config)
    payload = {
        "checkpoint_type": LORA_CHECKPOINT_TYPE,
        "lora_config": asdict(lora_cfg),
        "model_state_dict": extract_lora_state_dict(model),
    }
    checkpoint_path = Path(checkpoint_path)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def is_lora_checkpoint(payload: Any) -> bool:
    """Return whether a loaded object is a LoRA adapter checkpoint."""
    return isinstance(payload, dict) and payload.get("checkpoint_type") == LORA_CHECKPOINT_TYPE


def load_lora_checkpoint(
    model: nn.Module,
    checkpoint: str | Path | dict[str, Any],
    lora_config: Any | None = None,
) -> tuple[list[str], list[str]]:
    """
    Load adapter-only checkpoint into a model.

    The model must already contain the compatible frozen backbone weights.
    """
    payload = checkpoint
    if isinstance(checkpoint, str | Path):
        payload = torch.load(checkpoint, map_location="cpu")

    if not is_lora_checkpoint(payload):
        raise ValueError("Provided checkpoint is not a LoRA adapter checkpoint")

    resolved_lora_config = lora_config if lora_config is not None else payload.get("lora_config")
    apply_lora_to_model(model, lora_config=resolved_lora_config)

    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    return missing, unexpected
