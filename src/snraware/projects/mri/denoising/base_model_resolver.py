"""Resolve named SNRAware base-model checkpoint presets."""

from __future__ import annotations

from pathlib import Path

DEFAULT_BASE_MODEL_VARIANT = "small"
VALID_BASE_MODEL_VARIANTS = ("small", "large")

BASE_MODEL_PRESETS = {
    "small": {
        "config": Path("checkpoints/small/snraware_small_model.yaml"),
        "checkpoint": Path("checkpoints/small/snraware_small_model.pts"),
    },
    "large": {
        "config": Path("checkpoints/large/snraware_large_model.yaml"),
        "checkpoint": Path("checkpoints/large/snraware_large_model.pts"),
    },
}

__all__ = [
    "BASE_MODEL_PRESETS",
    "DEFAULT_BASE_MODEL_VARIANT",
    "VALID_BASE_MODEL_VARIANTS",
    "resolve_base_model_paths",
]


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _normalize_optional_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "null":
        return None
    return text


def _normalize_variant(variant: str | None) -> str:
    text = DEFAULT_BASE_MODEL_VARIANT if variant is None else str(variant).strip().lower()
    if text == "":
        text = DEFAULT_BASE_MODEL_VARIANT
    if text not in BASE_MODEL_PRESETS:
        supported = ", ".join(VALID_BASE_MODEL_VARIANTS)
        raise ValueError(
            f"Unsupported base model variant '{variant}'. Expected one of: {supported}"
        )
    return text


def _resolve_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def resolve_base_model_paths(
    variant: str | None,
    config_path: str | Path | None,
    checkpoint_path: str | Path | None,
    repo_root: str | Path | None = None,
) -> tuple[str, str]:
    """Resolve final base-model config/checkpoint paths for training or utility scripts."""
    repo_root_path = _resolve_path(repo_root or _default_repo_root(), _default_repo_root())

    explicit_config = _normalize_optional_path(config_path)
    explicit_checkpoint = _normalize_optional_path(checkpoint_path)
    has_explicit_config = explicit_config is not None
    has_explicit_checkpoint = explicit_checkpoint is not None

    if has_explicit_config != has_explicit_checkpoint:
        raise ValueError(
            "base_model.config_path and base_model.checkpoint_path must either both be set "
            "or both be unset."
        )

    if has_explicit_config and has_explicit_checkpoint:
        resolved_config = _resolve_path(explicit_config, repo_root_path)
        resolved_checkpoint = _resolve_path(explicit_checkpoint, repo_root_path)
    else:
        normalized_variant = _normalize_variant(variant)
        preset = BASE_MODEL_PRESETS[normalized_variant]
        resolved_config = _resolve_path(preset["config"], repo_root_path)
        resolved_checkpoint = _resolve_path(preset["checkpoint"], repo_root_path)

    if not resolved_config.is_file():
        raise FileNotFoundError(f"Base model config not found: {resolved_config}")
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"Base model checkpoint not found: {resolved_checkpoint}")

    return str(resolved_config), str(resolved_checkpoint)
