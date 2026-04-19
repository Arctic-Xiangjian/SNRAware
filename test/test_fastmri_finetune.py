from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

pytest.importorskip("hydra")
from hydra import compose, initialize

import fastmri_data.work_with_snraware as fastmri_dataset_module
from fastmri_data.work_with_snraware import FastMRISNRAwareDataset
import train_fastmri_single_coil as fastmri_cli_module
from snraware.projects.mri.denoising.base_model_resolver import (
    DEFAULT_BASE_MODEL_VARIANT,
    resolve_base_model_paths,
)
from snraware.projects.mri.denoising.fastmri_compat import (
    NormUnet,
    SNRAwareWithGFactor,
    is_fastmri_finetune_checkpoint,
    load_fastmri_finetune_checkpoint,
    save_fastmri_finetune_checkpoint,
)
import snraware.projects.mri.denoising.fastmri_compat as fastmri_compat_module
from snraware.projects.mri.denoising.lora_utils import apply_lora_to_model
from snraware.projects.mri.denoising.model import DenoisingModel
import snraware.projects.mri.denoising.train as fastmri_train_module
import snraware.projects.mri.denoising.trainer_fa as trainer_fa_module
from snraware.projects.mri.denoising.trainer_fa import (
    FastMRIFineTuneTrainer,
    build_fastmri_dataloaders,
    complex_output_to_magnitude,
    configure_model_for_finetune_mode,
    fastmri_autocast_context,
    resolve_fastmri_precision,
)


def _tiny_config(mode: str):
    with initialize(version_base=None, config_path="../src/snraware/projects/mri/denoising/configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "backbone=soanet",
                "backbone.num_of_channels=8",
                "backbone.block_str=[T1]",
                "backbone.num_stages=1",
                "backbone.downsample=False",
                "backbone.block.cell.n_head=4",
                "backbone.block.cell.window_size=[4,4,1]",
                "backbone.block.cell.patch_size=[2,2,1]",
                "dataset.cutout_shape=[32,32,1]",
                "lora.enabled=True",
                "lora.r=2",
                "lora.lora_alpha=8.0",
                "lora.lora_dropout=0.0",
            ],
        )

    OmegaConf.set_struct(cfg, False)
    cfg.fastmri_finetune = OmegaConf.create(
        {
            "mode": mode,
            "warmup_epochs": 1,
            "unet_lr": 1e-3,
            "adapter_lr": 1e-3,
            "gradient_clip_val": 1.0,
            "weight_decay": 0.0,
            "scheduler_t_max": 0,
            "resume_from": None,
            "max_epochs": 1,
            "batch_size": 1,
            "log_every_n_steps": 100,
            "evaluate_every_n_epochs": 1,
            "use_bf16": False,
            "use_unet": True,
            "gradient_checkpoint_frozen_base": True,
            "train_pre_post": False,
            "train_patch_size": None,
            "eval_patch_batch_size": 64,
        }
    )
    return cfg


class DummyBaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        return x[:, :2, ...]


class DummyGFactor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return -torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], dtype=x.dtype, device=x.device)


class ToyFastMRIDataset(Dataset):
    def __init__(self, num_samples: int, size: int = 32):
        super().__init__()
        generator = torch.Generator().manual_seed(42)
        self.size = size
        self.noisy = torch.randn(num_samples, 2, size, size, generator=generator)
        self.clean = torch.sqrt(
            self.noisy[:, 0:1, ...].square() + self.noisy[:, 1:2, ...].square()
        )
        self.volume_names = [f"volume_{idx // 2}" for idx in range(num_samples)]

    def __len__(self) -> int:
        return len(self.noisy)

    def __getitem__(self, index: int):
        metadata = {
            "name": f"{self.volume_names[index]}_slice_{index}",
            "volume_name": self.volume_names[index],
            "slice_idx": index % 2,
            "mean": torch.tensor(0.0, dtype=torch.float32),
            "std": torch.tensor(1.0, dtype=torch.float32),
            "mask": torch.ones(self.size, self.size, 2, dtype=torch.float32),
            "masked_kspace": torch.zeros(self.size, self.size, 2, dtype=torch.float32),
        }
        return self.noisy[index], self.clean[index], torch.tensor(0.0), metadata


class FlatFastMRIDataset(Dataset):
    def __init__(self, num_samples: int, *, size: int = 64, fill_value: float = 0.0):
        super().__init__()
        self.size = size
        self.noisy = torch.full((num_samples, 2, size, size), fill_value, dtype=torch.float32)
        self.clean = torch.sqrt(
            self.noisy[:, 0:1, ...].square() + self.noisy[:, 1:2, ...].square()
        )

    def __len__(self) -> int:
        return len(self.noisy)

    def __getitem__(self, index: int):
        metadata = {
            "name": f"flat_volume_0_slice_{index}",
            "volume_name": "flat_volume_0",
            "slice_idx": index,
            "mean": torch.tensor(0.0, dtype=torch.float32),
            "std": torch.tensor(1.0, dtype=torch.float32),
            "mask": torch.ones(self.size, self.size, 2, dtype=torch.float32),
            "masked_kspace": torch.zeros(self.size, self.size, 2, dtype=torch.float32),
        }
        return self.noisy[index], self.clean[index], torch.tensor(0.0), metadata


def _build_wrapped_model(mode: str, size: int = 32, *, use_unet: bool = True) -> SNRAwareWithGFactor:
    config = _tiny_config(mode)
    base_model = DenoisingModel(config=config, D=1, H=size, W=size)
    return SNRAwareWithGFactor(base_model=base_model, use_unet=use_unet)


def _write_fake_fastmri_volume(path: Path, *, size: int = 320) -> None:
    image = torch.zeros(2, size, size, dtype=torch.complex64)
    image[0, size // 2, size // 2] = 1 + 2j
    image[1, size // 2 - 5, size // 2 + 3] = 2 - 1j

    kspace = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(image, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    ).numpy()
    target = torch.abs(image).numpy().astype(np.float32)

    header = f"""
    <ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
      <encoding>
        <encodedSpace><matrixSize><x>{size}</x><y>{size}</y><z>1</z></matrixSize></encodedSpace>
        <reconSpace><matrixSize><x>{size}</x><y>{size}</y><z>1</z></matrixSize></reconSpace>
        <encodingLimits><kspace_encoding_step_1><center>{size // 2}</center><maximum>{size - 1}</maximum></kspace_encoding_step_1></encodingLimits>
      </encoding>
      <acquisitionSystemInformation><systemModel>TestScanner</systemModel></acquisitionSystemInformation>
    </ismrmrdHeader>
    """.strip()

    with h5py.File(path, "w") as hf:
        hf.create_dataset("kspace", data=kspace)
        hf.create_dataset("reconstruction_esc", data=target)
        hf.create_dataset("ismrmrd_header", data=np.bytes_(header))


class CaptureL1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pred_dtype = None
        self.target_dtype = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.pred_dtype = pred.dtype
        self.target_dtype = target.dtype
        return torch.nn.functional.l1_loss(pred, target)


def _adapter_trainable_names(model: SNRAwareWithGFactor) -> list[str]:
    return [
        name
        for name, parameter in model.base_model.named_parameters()
        if ".lora_" in name and parameter.requires_grad
    ]


def _pre_post_trainable_names(model: SNRAwareWithGFactor) -> list[str]:
    return [
        name
        for name, parameter in model.base_model.named_parameters()
        if (name.startswith("pre.") or name.startswith("post.")) and parameter.requires_grad
    ]


def _clean_lora_wrapped_name(name: str) -> str:
    return name.replace(".base_layer", "")


def _backbone_trainable_names(model: SNRAwareWithGFactor) -> list[str]:
    return [
        name
        for name, parameter in model.base_model.named_parameters()
        if ".lora_" not in name and not name.startswith("pre.") and not name.startswith("post.")
        and parameter.requires_grad
    ]


def _gfactor_is_trainable(model: SNRAwareWithGFactor) -> bool:
    return any(parameter.requires_grad for parameter in model.gfactor_unet.parameters())


def _domain_scale_is_trainable(model: SNRAwareWithGFactor) -> bool:
    return bool(model.domain_scale.requires_grad)


def _group_param_ids(trainer: FastMRIFineTuneTrainer, group_name: str) -> set[int]:
    return {
        id(parameter)
        for group in trainer.optimizer.param_groups
        if group.get("name") == group_name
        for parameter in group["params"]
    }


def _adapter_group_lr(trainer: FastMRIFineTuneTrainer) -> float | None:
    for group in trainer.optimizer.param_groups:
        if group.get("name") == "adapter":
            return float(group["lr"])
    return None


def test_wrapper_routes_2ch_input_through_gfactor_head():
    base_model = DummyBaseModel()
    gfactor = DummyGFactor()
    model = SNRAwareWithGFactor(base_model=base_model, gfactor_unet=gfactor)

    noisy = torch.randn(2, 2, 8, 8)
    output = model(noisy)

    assert gfactor.calls == 1
    assert output.shape == (2, 2, 1, 8, 8)
    assert base_model.last_input is not None
    assert base_model.last_input.shape == (2, 3, 1, 8, 8)
    assert torch.all(base_model.last_input[:, 2] >= 1.0)
    assert torch.all(base_model.last_input[:, 2] <= 5.0)


def test_wrapper_bounds_gfactor_predictions_to_physical_range():
    class ExtremeGFactor(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), -100.0, dtype=x.dtype, device=x.device)
            out[..., 1::2, :] = 100.0
            return out

    base_model = DummyBaseModel()
    model = SNRAwareWithGFactor(base_model=base_model, gfactor_unet=ExtremeGFactor())

    noisy = torch.randn(1, 2, 8, 8)
    model(noisy)

    assert base_model.last_input is not None
    predicted_gfactor = base_model.last_input[:, 2]
    assert torch.all(predicted_gfactor >= 1.0)
    assert torch.all(predicted_gfactor <= 5.0)


def test_wrapper_domain_scale_scales_fastmri_input_and_restores_output_scale():
    base_model = DummyBaseModel()
    gfactor = DummyGFactor()
    model = SNRAwareWithGFactor(base_model=base_model, gfactor_unet=gfactor)
    with torch.no_grad():
        model.domain_scale.fill_(3.5)

    noisy = torch.randn(2, 2, 8, 8)
    output = model(noisy)

    assert base_model.last_input is not None
    expected_scale = torch.tensor(3.5, dtype=noisy.dtype)
    expected_gfactor = 4.0 * torch.sigmoid(torch.tensor(-1.0, dtype=noisy.dtype)) + 1.0
    assert torch.allclose(base_model.last_input[:, :2], noisy.unsqueeze(2) * expected_scale)
    assert torch.allclose(
        base_model.last_input[:, 2],
        torch.full_like(base_model.last_input[:, 2], float(expected_gfactor.item())),
    )
    assert torch.allclose(output, noisy.unsqueeze(2))


def test_wrapper_bypasses_gfactor_for_native_3ch_input():
    base_model = DummyBaseModel()
    gfactor = DummyGFactor()
    model = SNRAwareWithGFactor(base_model=base_model, gfactor_unet=gfactor)

    noisy = torch.randn(2, 3, 1, 8, 8)
    output = model(noisy)

    assert gfactor.calls == 0
    assert output.shape == (2, 2, 1, 8, 8)
    assert torch.equal(base_model.last_input, noisy)


def test_wrapper_can_short_circuit_gfactor_unet_with_ones_map():
    base_model = DummyBaseModel()
    gfactor = DummyGFactor()
    model = SNRAwareWithGFactor(base_model=base_model, gfactor_unet=gfactor, use_unet=False)

    noisy = torch.randn(2, 2, 8, 8)
    output = model(noisy)

    assert gfactor.calls == 0
    assert output.shape == (2, 2, 1, 8, 8)
    assert base_model.last_input is not None
    assert torch.allclose(base_model.last_input[:, 2], torch.ones_like(base_model.last_input[:, 2]))
    assert model.last_gfactor_stats == {"mean": 1.0, "p95": 1.0, "max": 1.0}


@pytest.mark.parametrize(
    ("name", "input_tensor"),
    [
        ("zeros", torch.zeros(1, 1, 64, 64, 2, dtype=torch.float32)),
        ("constant", torch.ones(1, 1, 64, 64, 2, dtype=torch.float32)),
        ("tiny_variance", torch.randn(1, 1, 64, 64, 2, dtype=torch.float32) * 1e-8),
    ],
)
def test_normunet_adapter_remains_finite_on_low_variance_inputs(name: str, input_tensor: torch.Tensor):
    model = NormUnet()
    output = model(input_tensor)
    assert output.shape == (1, 1, 64, 64), name
    assert torch.isfinite(output).all(), name


def test_complex_output_to_magnitude_uses_exact_formula_without_epsilon():
    output = torch.zeros(1, 2, 1, 1, 2, dtype=torch.float32)
    output[0, 0, 0, 0, 0] = 3.0
    output[0, 1, 0, 0, 0] = 4.0
    magnitude = complex_output_to_magnitude(output)
    expected = torch.tensor([[[[5.0, 0.0]]]], dtype=torch.float32)
    assert torch.equal(magnitude, expected)


def test_resolve_fastmri_precision_rejects_cpu_bf16():
    with pytest.raises(ValueError, match="USE_BF16=false"):
        resolve_fastmri_precision(torch.device("cpu"), use_bf16=True)


def test_resolve_fastmri_precision_rejects_unsupported_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    with pytest.raises(RuntimeError, match="USE_BF16=false"):
        resolve_fastmri_precision(torch.device("cuda:0"), use_bf16=True)


def test_resolve_fastmri_precision_allows_fp32_opt_out():
    precision_state = resolve_fastmri_precision(torch.device("cpu"), use_bf16=False)
    assert precision_state == {"mode": "fp32", "use_bf16": False}


def test_fastmri_autocast_context_uses_cuda_bf16(monkeypatch):
    calls = []

    @contextmanager
    def fake_autocast(*, device_type, dtype, enabled):
        calls.append(
            {
                "device_type": device_type,
                "dtype": dtype,
                "enabled": enabled,
            }
        )
        yield

    monkeypatch.setattr(torch, "autocast", fake_autocast)

    with fastmri_autocast_context(torch.device("cuda:0"), enabled=True):
        pass

    with fastmri_autocast_context(torch.device("cuda:0"), enabled=False):
        pass

    assert calls == [
        {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": True},
        {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": False},
    ]


@pytest.mark.parametrize(
    ("mode", "expect_unet_trainable", "expect_adapter_trainable", "expect_pre_post_trainable"),
    [
        ("unet_only", True, False, False),
        ("unet_and_lora", True, True, False),
        ("lora_only", False, True, False),
        ("warmup_then_both", True, False, False),
    ],
)
def test_configure_model_for_modes(
    mode: str,
    expect_unet_trainable: bool,
    expect_adapter_trainable: bool,
    expect_pre_post_trainable: bool,
):
    model = _build_wrapped_model(mode)
    state = configure_model_for_finetune_mode(
        model,
        mode=mode,
        lora_config=model.base_model.config.lora,
        train_pre_post=False,
    )

    unet_trainable = any(parameter.requires_grad for parameter in model.gfactor_unet.parameters())
    adapter_trainable = bool(_adapter_trainable_names(model))
    pre_post_trainable = bool(_pre_post_trainable_names(model))
    frozen_backbone = all(
        not parameter.requires_grad
        for name, parameter in model.base_model.named_parameters()
        if ".lora_" not in name and not name.startswith("pre.") and not name.startswith("post.")
    )

    assert unet_trainable is expect_unet_trainable
    assert _domain_scale_is_trainable(model) is True
    assert adapter_trainable is expect_adapter_trainable
    assert pre_post_trainable is expect_pre_post_trainable
    assert frozen_backbone
    assert state["mode"] == mode
    assert state["train_pre_post"] is False


@pytest.mark.parametrize("mode", ["unet_and_lora", "lora_only", "warmup_then_both"])
def test_configure_model_for_modes_can_enable_pre_post(mode: str):
    model = _build_wrapped_model(mode)
    state = configure_model_for_finetune_mode(
        model,
        mode=mode,
        lora_config=model.base_model.config.lora,
        adapters_active=(mode != "warmup_then_both"),
        train_pre_post=True,
    )

    expect_pre_post_trainable = mode != "warmup_then_both" or state["adapters_active"]
    assert bool(_pre_post_trainable_names(model)) is expect_pre_post_trainable
    assert state["train_pre_post"] is True


def test_fastmri_checkpoint_round_trip_restores_gfactor_and_lora(tmp_path: Path):
    torch.manual_seed(123)
    model = _build_wrapped_model("unet_and_lora")
    configure_model_for_finetune_mode(
        model,
        mode="unet_and_lora",
        lora_config=model.base_model.config.lora,
        train_pre_post=False,
    )

    with torch.no_grad():
        first_unet_param = next(model.gfactor_unet.parameters())
        first_unet_param.add_(0.5)
        model.domain_scale.fill_(2.5)
        first_adapter_param = next(
            parameter
            for name, parameter in model.base_model.named_parameters()
            if ".lora_" in name
        )
        first_adapter_param.add_(0.25)

    checkpoint_path = tmp_path / "fastmri_roundtrip.pth"
    save_fastmri_finetune_checkpoint(
        model,
        checkpoint_path,
        mode="unet_and_lora",
        config=model.base_model.config,
        lora_config=model.base_model.config.lora,
        epoch=0,
        metrics={"psnr": 1.0, "ssim": 0.5, "nmse": 0.1},
    )

    payload = torch.load(checkpoint_path, map_location="cpu")
    assert is_fastmri_finetune_checkpoint(payload)
    assert torch.equal(payload["domain_scale"], model.domain_scale.detach().cpu())

    fresh_model = _build_wrapped_model("unet_and_lora")
    frozen_snapshot = {
        _clean_lora_wrapped_name(name): parameter.detach().clone()
        for name, parameter in fresh_model.base_model.named_parameters()
        if ".lora_" not in name
    }

    missing, unexpected = load_fastmri_finetune_checkpoint(
        fresh_model,
        payload,
        apply_lora_fn=apply_lora_to_model,
        lora_config=fresh_model.base_model.config.lora,
    )

    assert not unexpected
    assert not missing
    for key, value in model.gfactor_unet.state_dict().items():
        assert torch.equal(value, fresh_model.gfactor_unet.state_dict()[key])
    assert torch.equal(model.domain_scale, fresh_model.domain_scale)
    for name, parameter in fresh_model.base_model.named_parameters():
        if ".lora_" in name:
            assert torch.equal(parameter, model.base_model.state_dict()[name])
        else:
            assert torch.equal(parameter, frozen_snapshot[_clean_lora_wrapped_name(name)])


def test_fastmri_checkpoint_loader_defaults_missing_domain_scale_to_one(tmp_path: Path):
    torch.manual_seed(9)
    model = _build_wrapped_model("unet_and_lora")
    configure_model_for_finetune_mode(
        model,
        mode="unet_and_lora",
        lora_config=model.base_model.config.lora,
        train_pre_post=False,
    )

    checkpoint_path = tmp_path / "legacy_domain_scale_missing.pth"
    save_fastmri_finetune_checkpoint(
        model,
        checkpoint_path,
        mode="unet_and_lora",
        adapters_active=True,
        config=model.base_model.config,
        lora_config=model.base_model.config.lora,
        epoch=0,
    )

    payload = torch.load(checkpoint_path, map_location="cpu")
    payload.pop("domain_scale", None)

    fresh_model = _build_wrapped_model("unet_and_lora")
    assert fresh_model.domain_scale.item() == pytest.approx(1.0)

    missing, unexpected = load_fastmri_finetune_checkpoint(
        fresh_model,
        payload,
        apply_lora_fn=apply_lora_to_model,
        lora_config=fresh_model.base_model.config.lora,
    )

    assert missing == []
    assert unexpected == []
    assert fresh_model.domain_scale.item() == pytest.approx(1.0)


def test_fastmri_checkpoint_loader_rejects_nonfinite_domain_scale(tmp_path: Path):
    model = _build_wrapped_model("unet_and_lora")
    configure_model_for_finetune_mode(
        model,
        mode="unet_and_lora",
        lora_config=model.base_model.config.lora,
        train_pre_post=False,
    )

    checkpoint_path = tmp_path / "bad_domain_scale.pth"
    save_fastmri_finetune_checkpoint(
        model,
        checkpoint_path,
        mode="unet_and_lora",
        adapters_active=True,
        config=model.base_model.config,
        lora_config=model.base_model.config.lora,
        epoch=0,
    )

    payload = torch.load(checkpoint_path, map_location="cpu")
    payload["domain_scale"] = torch.tensor(float("nan"))

    with pytest.raises(ValueError, match="non-finite domain_scale"):
        load_fastmri_finetune_checkpoint(
            _build_wrapped_model("unet_and_lora"),
            payload,
            apply_lora_fn=apply_lora_to_model,
            lora_config=model.base_model.config.lora,
        )


def test_fastmri_checkpoint_payload_records_adapters_active(tmp_path: Path):
    model = _build_wrapped_model("warmup_then_both")
    configure_model_for_finetune_mode(
        model,
        mode="warmup_then_both",
        lora_config=model.base_model.config.lora,
        adapters_active=False,
        train_pre_post=False,
    )

    checkpoint_path = tmp_path / "warmup_state.pth"
    save_fastmri_finetune_checkpoint(
        model,
        checkpoint_path,
        mode="warmup_then_both",
        adapters_active=False,
        config=model.base_model.config,
        lora_config=model.base_model.config.lora,
        epoch=0,
    )

    payload = torch.load(checkpoint_path, map_location="cpu")
    assert payload["adapters_active"] is False
    assert payload["train_pre_post"] is False
    assert all(".lora_" in key for key in payload["lora_adapter"])


def test_fastmri_checkpoint_payload_records_use_unet(tmp_path: Path):
    model = _build_wrapped_model("lora_only", use_unet=False)
    checkpoint_path = tmp_path / "short_circuit_state.pth"
    save_fastmri_finetune_checkpoint(
        model,
        checkpoint_path,
        mode="lora_only",
        adapters_active=True,
        config={"fastmri_finetune": {"train_pre_post": False, "use_unet": False}},
        lora_config={"enabled": True, "r": 2, "lora_alpha": 8.0, "lora_dropout": 0.0},
        epoch=0,
    )

    payload = torch.load(checkpoint_path, map_location="cpu")
    assert payload["use_unet"] is False


def test_fastmri_checkpoint_loader_skips_duplicate_lora_injection(tmp_path: Path):
    torch.manual_seed(7)
    model = _build_wrapped_model("unet_and_lora")
    configure_model_for_finetune_mode(
        model,
        mode="unet_and_lora",
        lora_config=model.base_model.config.lora,
        train_pre_post=False,
    )

    checkpoint_path = tmp_path / "already_adapted.pth"
    save_fastmri_finetune_checkpoint(
        model,
        checkpoint_path,
        mode="unet_and_lora",
        adapters_active=True,
        config=model.base_model.config,
        lora_config=model.base_model.config.lora,
        epoch=0,
    )
    payload = torch.load(checkpoint_path, map_location="cpu")

    fresh_model = _build_wrapped_model("unet_and_lora")
    configure_model_for_finetune_mode(
        fresh_model,
        mode="unet_and_lora",
        lora_config=fresh_model.base_model.config.lora,
        train_pre_post=False,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("LoRA injection should be skipped when adapters already exist")

    missing, unexpected = load_fastmri_finetune_checkpoint(
        fresh_model,
        payload,
        apply_lora_fn=fail_if_called,
        lora_config=fresh_model.base_model.config.lora,
    )

    assert missing == []
    assert unexpected == []


def test_fastmri_checkpoint_loader_accepts_legacy_pre_post_adapter_state(tmp_path: Path):
    torch.manual_seed(11)
    source_model = _build_wrapped_model("unet_and_lora")
    configure_model_for_finetune_mode(
        source_model,
        mode="unet_and_lora",
        lora_config=source_model.base_model.config.lora,
        train_pre_post=True,
    )

    with torch.no_grad():
        first_pre_post = next(
            parameter
            for name, parameter in source_model.base_model.named_parameters()
            if name.startswith("pre.") or name.startswith("post.")
        )
        first_pre_post.add_(0.75)

    checkpoint_path = tmp_path / "legacy_pre_post.pth"
    save_fastmri_finetune_checkpoint(
        source_model,
        checkpoint_path,
        mode="unet_and_lora",
        adapters_active=True,
        config=OmegaConf.create(
            {
                "fastmri_finetune": {"train_pre_post": True},
            }
        ),
        lora_config=source_model.base_model.config.lora,
        epoch=0,
    )
    payload = torch.load(checkpoint_path, map_location="cpu")
    assert any(key.startswith("pre.") or key.startswith("post.") for key in payload["lora_adapter"])

    fresh_model = _build_wrapped_model("unet_and_lora")
    configure_model_for_finetune_mode(
        fresh_model,
        mode="unet_and_lora",
        lora_config=fresh_model.base_model.config.lora,
        train_pre_post=False,
    )

    missing, unexpected = load_fastmri_finetune_checkpoint(
        fresh_model,
        payload,
        apply_lora_fn=apply_lora_to_model,
        lora_config=fresh_model.base_model.config.lora,
    )

    assert missing == []
    assert unexpected == []
    for name, parameter in fresh_model.base_model.named_parameters():
        if name.startswith("pre.") or name.startswith("post."):
            assert torch.equal(parameter, source_model.base_model.state_dict()[name])


def test_fastmri_bridge_dataset_returns_expected_shapes_and_uses_full_resolution_masking(tmp_path: Path):
    data_dir = tmp_path / "singlecoil_train"
    data_dir.mkdir()
    volume_path = data_dir / "case001.h5"
    _write_fake_fastmri_volume(volume_path, size=384)

    dataset = FastMRISNRAwareDataset(
        root=data_dir,
        split="train",
        acc_factor=4,
        deterministic_mask_from_name=True,
        use_dataset_cache=False,
    )
    noisy, clean, noise_sigma, metadata = dataset[0]

    assert noisy.shape == (2, 320, 320)
    assert clean.shape == (1, 320, 320)
    assert noisy.dtype == torch.float32
    assert clean.dtype == torch.float32
    assert noise_sigma.dtype == torch.float32
    assert {"name", "volume_name", "slice_idx", "mean", "std", "mask", "masked_kspace"} <= set(metadata)
    assert metadata["mask"] is None
    assert metadata["masked_kspace"] is None

    noisy_mag = torch.sqrt(noisy[0:1].square() + noisy[1:2].square())
    assert torch.allclose(noisy_mag.mean(), torch.tensor(1.0), atol=1e-4)
    assert metadata["mean"].item() == 0.0

    with h5py.File(volume_path, "r") as hf:
        raw_kspace = np.asarray(hf["kspace"][0])

    orig_kspace_slice = torch.zeros((1, raw_kspace.shape[0], raw_kspace.shape[1], 2), dtype=torch.float32)
    orig_kspace_slice[0, :, :, 0] = torch.from_numpy(raw_kspace.real.astype(np.float32))
    orig_kspace_slice[0, :, :, 1] = torch.from_numpy(raw_kspace.imag.astype(np.float32))
    clean_recon_slice = fastmri_dataset_module._ifft2c(orig_kspace_slice)

    mask = fastmri_dataset_module.legacy_uniform1d_mask(
        img=orig_kspace_slice.permute(0, 3, 1, 2),
        size=orig_kspace_slice.shape[2],
        batch_size=1,
        acc_factor=4,
        center_fraction=0.08,
        fix=False,
        name="case001.h5",
    ).permute(0, 2, 3, 1)
    masked_kspace = orig_kspace_slice * mask
    under_recon_slice = fastmri_dataset_module._ifft2c(masked_kspace)

    clean_recon_slice = fastmri_dataset_module._complex_center_crop(clean_recon_slice, (320, 320))
    under_recon_slice = fastmri_dataset_module._complex_center_crop(under_recon_slice, (320, 320))

    under_recon_abs = fastmri_dataset_module._complex_abs(under_recon_slice).squeeze(0).unsqueeze(0)
    expected_std = under_recon_abs.mean().float()
    expected_noisy = (under_recon_slice / expected_std).squeeze(0).permute(2, 0, 1).contiguous()
    expected_clean = (
        fastmri_dataset_module._complex_abs(clean_recon_slice).squeeze(0).unsqueeze(0) / expected_std
    ).contiguous()

    assert torch.allclose(noisy, expected_noisy, atol=1e-5)
    assert torch.allclose(clean, expected_clean, atol=1e-5)


def test_fastmri_bridge_dataset_keeps_mask_metadata_when_no_crop(tmp_path: Path):
    data_dir = tmp_path / "singlecoil_val"
    data_dir.mkdir()
    _write_fake_fastmri_volume(data_dir / "case001.h5", size=320)

    dataset = FastMRISNRAwareDataset(
        root=data_dir,
        split="val",
        acc_factor=4,
        deterministic_mask_from_name=True,
        use_dataset_cache=False,
    )
    noisy, clean, _noise_sigma, metadata = dataset[0]

    assert dataset.split == "val"
    assert noisy.shape == (2, 320, 320)
    assert clean.shape == (1, 320, 320)
    assert metadata["mask"] is not None
    assert metadata["masked_kspace"] is not None
    assert metadata["mask"].shape == (320, 320, 2)
    assert metadata["masked_kspace"].shape == (320, 320, 2)

    center_fraction = 0.08
    nsamp_center = int(round(320 * center_fraction))
    center_from = 320 // 2 - nsamp_center // 2
    center_to = center_from + nsamp_center
    assert torch.all(metadata["mask"][:, center_from:center_to, :] == 1)


def test_fastmri_bridge_dataset_applies_train_patch_crop_after_normalization(monkeypatch, tmp_path: Path):
    data_dir = tmp_path / "singlecoil_train_patch"
    data_dir.mkdir()
    _write_fake_fastmri_volume(data_dir / "case001.h5", size=320)

    full_dataset = FastMRISNRAwareDataset(
        root=data_dir,
        split="train",
        acc_factor=4,
        deterministic_mask_from_name=True,
        use_dataset_cache=False,
    )
    full_noisy, full_clean, _noise_sigma, _metadata = full_dataset[0]

    randint_values = iter([torch.tensor(11), torch.tensor(17)])
    monkeypatch.setattr(
        fastmri_dataset_module.torch,
        "randint",
        lambda low, high, size, **kwargs: next(randint_values),
    )

    patch_dataset = FastMRISNRAwareDataset(
        root=data_dir,
        split="train",
        acc_factor=4,
        crop_size=(320, 320),
        train_patch_size=(128, 160),
        deterministic_mask_from_name=True,
        use_dataset_cache=False,
    )
    noisy, clean, _noise_sigma, metadata = patch_dataset[0]

    assert noisy.shape == (2, 128, 160)
    assert clean.shape == (1, 128, 160)
    assert torch.equal(noisy, full_noisy[:, 11:139, 17:177])
    assert torch.equal(clean, full_clean[:, 11:139, 17:177])
    assert metadata["mask"] is None
    assert metadata["masked_kspace"] is None


def test_fastmri_bridge_dataset_ignores_train_patch_size_for_validation(tmp_path: Path):
    data_dir = tmp_path / "singlecoil_val_patch"
    data_dir.mkdir()
    _write_fake_fastmri_volume(data_dir / "case001.h5", size=320)

    dataset = FastMRISNRAwareDataset(
        root=data_dir,
        split="val",
        acc_factor=4,
        crop_size=(320, 320),
        train_patch_size=(128, 128),
        deterministic_mask_from_name=True,
        use_dataset_cache=False,
    )
    noisy, clean, _noise_sigma, metadata = dataset[0]

    assert noisy.shape == (2, 320, 320)
    assert clean.shape == (1, 320, 320)
    assert metadata["mask"] is not None
    assert metadata["masked_kspace"] is not None


def test_build_fastmri_dataloaders_applies_sample_rate_to_train_only(monkeypatch):
    created = []

    class FakeDataset(Dataset):
        def __init__(self, **kwargs):
            created.append(kwargs)

        def __len__(self):
            return 1

        def __getitem__(self, index: int):
            metadata = {
                "name": "vol0_slice_0",
                "volume_name": "vol0",
                "slice_idx": 0,
                "mean": torch.tensor(0.0, dtype=torch.float32),
                "std": torch.tensor(1.0, dtype=torch.float32),
                "mask": torch.ones(4, 4, 2, dtype=torch.float32),
                "masked_kspace": torch.zeros(4, 4, 2, dtype=torch.float32),
            }
            noisy = torch.zeros(2, 4, 4, dtype=torch.float32)
            clean = torch.zeros(1, 4, 4, dtype=torch.float32)
            return noisy, clean, torch.tensor(0.0), metadata

    fake_module = types.ModuleType("fastmri_data.work_with_snraware")
    fake_module.FastMRISNRAwareDataset = FakeDataset
    monkeypatch.setitem(sys.modules, "fastmri_data.work_with_snraware", fake_module)

    with initialize(version_base=None, config_path="../src/snraware/projects/mri/denoising/configs"):
        config = compose(config_name="fastmri_finetune")

    config.fastmri_finetune.train_root = "/tmp/train"
    config.fastmri_finetune.val_root = "/tmp/val"
    config.fastmri_finetune.test_root = "/tmp/test"
    config.fastmri_finetune.num_workers = 0
    config.fastmri_finetune.sample_rate = 0.02
    config.fastmri_finetune.volume_sample_rate = 0.5
    config.fastmri_finetune.train_sample_rate = None
    config.fastmri_finetune.train_volume_sample_rate = None
    config.fastmri_finetune.train_patch_size = [128, 128]

    build_fastmri_dataloaders(config)

    assert len(created) == 3
    assert created[0]["split"] == "train"
    assert created[1]["split"] == "val"
    assert created[2]["split"] == "test"
    assert created[0]["train_patch_size"] == (128, 128)
    assert created[1]["train_patch_size"] is None
    assert created[2]["train_patch_size"] is None
    assert created[0]["sample_rate"] == 0.02
    assert created[0]["volume_sample_rate"] == 0.5
    assert created[1]["sample_rate"] is None
    assert created[1]["volume_sample_rate"] is None
    assert created[2]["sample_rate"] is None
    assert created[2]["volume_sample_rate"] is None


def test_fastmri_collate_fn_preserves_optional_none_metadata():
    batch = [
        (
            torch.zeros(2, 8, 8, dtype=torch.float32),
            torch.zeros(1, 8, 8, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            {
                "name": "vol0_slice_0",
                "volume_name": "vol0",
                "slice_idx": 0,
                "mean": torch.tensor(0.0, dtype=torch.float32),
                "std": torch.tensor(1.0, dtype=torch.float32),
                "mask": None,
                "masked_kspace": None,
            },
        ),
        (
            torch.ones(2, 8, 8, dtype=torch.float32),
            torch.ones(1, 8, 8, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            {
                "name": "vol0_slice_1",
                "volume_name": "vol0",
                "slice_idx": 1,
                "mean": torch.tensor(0.0, dtype=torch.float32),
                "std": torch.tensor(1.0, dtype=torch.float32),
                "mask": None,
                "masked_kspace": None,
            },
        ),
    ]

    noisy, clean, noise_sigma, metadata = trainer_fa_module._stack_fastmri_batch(batch)

    assert noisy.shape == (2, 2, 8, 8)
    assert clean.shape == (2, 1, 8, 8)
    assert noise_sigma.shape == (2,)
    assert metadata["mask"] is None
    assert metadata["masked_kspace"] is None
    assert metadata["volume_name"] == ["vol0", "vol0"]


@pytest.mark.parametrize(
    ("variant", "config_suffix", "checkpoint_suffix"),
    [
        ("small", "checkpoints/small/snraware_small_model.yaml", "checkpoints/small/snraware_small_model.pts"),
        ("large", "checkpoints/large/snraware_large_model.yaml", "checkpoints/large/snraware_large_model.pts"),
    ],
)
def test_resolve_base_model_paths_for_presets(
    variant: str,
    config_suffix: str,
    checkpoint_suffix: str,
):
    repo_root = Path(__file__).resolve().parents[1]
    config_path, checkpoint_path = resolve_base_model_paths(
        variant=variant,
        config_path=None,
        checkpoint_path=None,
        repo_root=repo_root,
    )

    assert config_path.endswith(config_suffix)
    assert checkpoint_path.endswith(checkpoint_suffix)
    assert Path(config_path).is_file()
    assert Path(checkpoint_path).is_file()


def test_resolve_base_model_paths_rejects_invalid_variant():
    with pytest.raises(ValueError, match="Unsupported base model variant"):
        resolve_base_model_paths("medium", None, None, repo_root=Path(__file__).resolve().parents[1])


def test_resolve_base_model_paths_rejects_partial_override():
    with pytest.raises(ValueError, match="must either both be set or both be unset"):
        resolve_base_model_paths(
            DEFAULT_BASE_MODEL_VARIANT,
            "./checkpoints/small/snraware_small_model.yaml",
            None,
            repo_root=Path(__file__).resolve().parents[1],
        )


def test_resolve_base_model_paths_uses_explicit_pair_override():
    repo_root = Path(__file__).resolve().parents[1]
    config_path, checkpoint_path = resolve_base_model_paths(
        variant="large",
        config_path="./checkpoints/small/snraware_small_model.yaml",
        checkpoint_path="./checkpoints/small/snraware_small_model.pts",
        repo_root=repo_root,
    )

    assert config_path.endswith("checkpoints/small/snraware_small_model.yaml")
    assert checkpoint_path.endswith("checkpoints/small/snraware_small_model.pts")


def test_resolve_base_model_paths_fails_with_exact_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Base model config not found"):
        resolve_base_model_paths("small", None, None, repo_root=tmp_path)


def test_fastmri_config_defaults_to_small_base_model():
    with initialize(version_base=None, config_path="../src/snraware/projects/mri/denoising/configs"):
        config = compose(config_name="fastmri_finetune")

    assert config.base_model.variant == "small"
    assert config.base_model.config_path is None
    assert config.base_model.checkpoint_path is None
    assert int(config.fastmri_finetune.evaluate_every_n_epochs) == 10
    assert bool(config.fastmri_finetune.use_unet) is True
    assert config.fastmri_finetune.train_pre_post is False
    assert list(config.fastmri_finetune.train_patch_size) == [64, 64]
    assert int(config.fastmri_finetune.eval_patch_batch_size) == 64


def _make_fastmri_cli_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    test_root = tmp_path / "test"
    train_root.mkdir()
    val_root.mkdir()
    test_root.mkdir()
    return train_root, val_root, test_root


def _parse_fastmri_cli_args(tmp_path: Path, *extra_args: str) -> argparse.Namespace:
    train_root, val_root, _ = _make_fastmri_cli_roots(tmp_path)
    return fastmri_cli_module.parse_args(
        ["--train-root", str(train_root), "--val-root", str(val_root), *extra_args]
    )


def test_fastmri_cli_defaults(tmp_path: Path):
    args = _parse_fastmri_cli_args(tmp_path)

    assert args.model_size == "large"
    assert args.mode == "warmup_then_both"
    assert args.use_unet is True
    assert args.train_pre_post is True
    assert args.precision == "auto"
    assert args.train_patch_size == (64, 64)
    assert args.crop_size == (320, 320)
    assert args.patch_overlap == (16, 16)
    assert args.eval_patch_batch_size == 64
    assert args.max_epochs == 20
    assert args.warmup_epochs == 5
    assert args.evaluate_every == 2


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("64", (64, 64)), ("64x96", (64, 96)), ("64,96", (64, 96))],
)
def test_fastmri_cli_parses_spatial_sizes(raw_value: str, expected: tuple[int, int]):
    assert fastmri_cli_module.parse_spatial_size(raw_value) == expected


def test_fastmri_cli_parses_optional_null_patch_size():
    assert fastmri_cli_module.parse_optional_spatial_size("null") is None
    assert fastmri_cli_module.parse_optional_spatial_size("NULL") is None


def test_fastmri_cli_rejects_partial_explicit_model_override(tmp_path: Path):
    base_model_config = tmp_path / "base_model.yaml"
    base_model_config.write_text("dummy: true\n")
    args = _parse_fastmri_cli_args(tmp_path, "--base-model-config", str(base_model_config), "--device", "cpu")

    with pytest.raises(ValueError, match="must either both be set or both be unset"):
        fastmri_cli_module.build_config_from_args(args)


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (("--train-patch-size", "400x400", "--crop-size", "320"), "TRAIN_PATCH_SIZE must fit inside CROP_SIZE"),
        (("--patch-overlap", "128"), "PATCH_OVERLAP must be smaller than TRAIN_PATCH_SIZE"),
        (("--mode", "unet_only", "--no-use-unet"), "mode=unet_only cannot be combined with --no-use-unet"),
    ],
)
def test_fastmri_cli_rejects_invalid_patch_or_mode(
    tmp_path: Path,
    extra_args: tuple[str, ...],
    message: str,
):
    args = _parse_fastmri_cli_args(tmp_path, "--device", "cpu", *extra_args)

    with pytest.raises(ValueError, match=message):
        fastmri_cli_module.build_config_from_args(args)


def test_fastmri_cli_precision_auto_resolves_cpu_to_fp32(tmp_path: Path):
    args = _parse_fastmri_cli_args(tmp_path, "--device", "cpu")
    config, summary = fastmri_cli_module.build_config_from_args(args)

    assert summary["TRAINING_PRECISION"] == "fp32"
    assert config.fastmri_finetune.use_bf16 is False
    assert config.fastmri_finetune.pin_memory is False
    assert config.fastmri_finetune.persistent_workers is True


def test_fastmri_cli_precision_auto_prefers_bf16_when_supported(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(fastmri_cli_module, "_resolve_device", lambda device_str: torch.device("cuda:0"))
    monkeypatch.setattr(
        fastmri_cli_module,
        "resolve_fastmri_precision",
        lambda device, use_bf16: {"mode": "bf16", "use_bf16": True},
    )

    args = _parse_fastmri_cli_args(tmp_path)
    config, summary = fastmri_cli_module.build_config_from_args(args)

    assert summary["TRAINING_PRECISION"] == "bf16"
    assert config.fastmri_finetune.use_bf16 is True
    assert config.fastmri_finetune.pin_memory is True


def test_fastmri_cli_overrides_apply_after_defaults(tmp_path: Path):
    args = _parse_fastmri_cli_args(
        tmp_path,
        "--device",
        "cpu",
        "--override",
        "fastmri_finetune.batch_size=3",
        "--override",
        "lora.r=16",
    )
    config, _ = fastmri_cli_module.build_config_from_args(args)

    assert int(config.fastmri_finetune.batch_size) == 3
    assert int(config.lora.r) == 16


def test_fastmri_cli_dry_run_prints_summary_and_config(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    train_root, val_root, _ = _make_fastmri_cli_roots(tmp_path)
    monkeypatch.setattr(
        fastmri_cli_module,
        "run_fastmri_finetuning_from_config",
        lambda config: pytest.fail("dry-run should not launch training"),
    )

    result = fastmri_cli_module.main(
        [
            "--train-root",
            str(train_root),
            "--val-root",
            str(val_root),
            "--device",
            "cpu",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "MODEL_SIZE=large" in captured.out
    assert "MODE=warmup_then_both" in captured.out
    assert "TRAINING_PRECISION=fp32" in captured.out
    assert "TRAIN_INPUT_SIZE=64x64" in captured.out
    assert "EVALUATE_EVERY=2" in captured.out
    assert "USE_UNET=true" in captured.out
    assert "TRAIN_PRE_POST=true" in captured.out
    assert "CHECKPOINT_NATIVE_SPATIAL_SIZE=64x64" in captured.out
    assert "Resolved config:" in captured.out
    assert "train_pre_post: true" in captured.out
    assert "Dry run requested; training was not started." in captured.out


def _run_fastmri_launcher_dry_run(
    tmp_path: Path,
    *script_args: str,
    **env_overrides: str,
):
    script_path = Path(__file__).resolve().parents[1] / "run_fast_mri_single_coil.sh"
    train_root, val_root, test_root = _make_fastmri_cli_roots(tmp_path)

    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "UV_CACHE_DIR": str(Path(__file__).resolve().parents[1] / ".uv-cache"),
        "DRY_RUN": "true",
        "TRAIN_ROOT": str(train_root),
        "VAL_ROOT": str(val_root),
        "TEST_ROOT": str(test_root),
        "DEVICE": "cpu",
        "USE_WANDB": "false",
    }
    env.update({key: str(value) for key, value in env_overrides.items()})

    return subprocess.run(
        ["bash", str(script_path), *script_args],
        cwd=str(script_path.parent),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"MODE": "bad_mode"}, "invalid choice"),
        ({"TRAIN_PATCH_SIZE": "400x400"}, "TRAIN_PATCH_SIZE must fit inside CROP_SIZE"),
        ({"PATCH_OVERLAP": "128"}, "PATCH_OVERLAP must be smaller than TRAIN_PATCH_SIZE"),
    ],
)
def test_run_fastmri_launcher_rejects_invalid_patch_or_mode(tmp_path: Path, overrides: dict[str, str], message: str):
    result = _run_fastmri_launcher_dry_run(tmp_path, **overrides)

    assert result.returncode != 0
    assert message in result.stderr


def test_run_fastmri_launcher_dry_run_uses_new_python_cli_defaults(tmp_path: Path):
    if shutil.which("uv") is None:
        pytest.skip("uv not installed")

    result = _run_fastmri_launcher_dry_run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "MODEL_SIZE=large" in result.stdout
    assert "TRAIN_INPUT_SIZE=64x64" in result.stdout
    assert "EVAL_CROP_SIZE=320x320" in result.stdout
    assert "EVAL_PATCH_BATCH_SIZE=64" in result.stdout
    assert "EVALUATE_EVERY=2" in result.stdout
    assert "USE_UNET=true" in result.stdout
    assert "TRAIN_PRE_POST=true" in result.stdout
    assert "CHECKPOINT_NATIVE_SPATIAL_SIZE=64x64" in result.stdout
    assert "Primary FastMRI single-coil entrypoint: train_fastmri_single_coil.py" in result.stderr


def test_run_fastmri_launcher_dry_run_emits_patch_overrides(tmp_path: Path):
    if shutil.which("uv") is None:
        pytest.skip("uv not installed")

    result = _run_fastmri_launcher_dry_run(
        tmp_path,
        TRAIN_PATCH_SIZE="128",
        CROP_SIZE="320",
        PATCH_OVERLAP="16",
    )

    assert result.returncode == 0, result.stderr
    assert "TRAIN_INPUT_SIZE=128x128" in result.stdout
    assert "PATCH_INFERENCE_FOR_VAL_TEST=true" in result.stdout
    assert "PATCH_OVERLAP=16x16" in result.stdout
    assert "train_patch_size:" in result.stdout


def test_run_fastmri_launcher_dry_run_can_disable_patch_mode(tmp_path: Path):
    if shutil.which("uv") is None:
        pytest.skip("uv not installed")

    result = _run_fastmri_launcher_dry_run(tmp_path, TRAIN_PATCH_SIZE="null")

    assert result.returncode == 0, result.stderr
    assert "TRAIN_INPUT_SIZE=320x320" in result.stdout
    assert "PATCH_INFERENCE_FOR_VAL_TEST=false" in result.stdout
    assert "PATCH_OVERLAP=16x16 (unused without train_patch_size)" in result.stdout
    assert "train_patch_size: null" in result.stdout


def test_run_fastmri_launcher_dry_run_accepts_legacy_eval_every_env_var(tmp_path: Path):
    if shutil.which("uv") is None:
        pytest.skip("uv not installed")

    result = _run_fastmri_launcher_dry_run(tmp_path, EVALUATE_EVERY_N_EPOCHS="7")

    assert result.returncode == 0, result.stderr
    assert "EVALUATE_EVERY=7" in result.stdout
    assert "evaluate_every_n_epochs: 7" in result.stdout


def test_run_fastmri_launcher_dry_run_exposes_use_unet_override(tmp_path: Path):
    if shutil.which("uv") is None:
        pytest.skip("uv not installed")

    result = _run_fastmri_launcher_dry_run(tmp_path, USE_UNET="false", MODE="lora_only")

    assert result.returncode == 0, result.stderr
    assert "USE_UNET=false" in result.stdout
    assert "use_unet: false" in result.stdout


def test_run_fastmri_launcher_supports_model_size_presets(tmp_path: Path):
    if shutil.which("uv") is None:
        pytest.skip("uv not installed")

    result = _run_fastmri_launcher_dry_run(tmp_path, MODEL_SIZE="small")

    assert result.returncode == 0, result.stderr
    assert "MODEL_SIZE=small" in result.stdout
    assert "snraware_small_model.yaml" in result.stdout


def test_run_fastmri_launcher_supports_positional_cuda_device(tmp_path: Path):
    if shutil.which("uv") is None:
        pytest.skip("uv not installed")

    result = _run_fastmri_launcher_dry_run(tmp_path, "3")

    assert result.returncode == 0, result.stderr
    assert "CUDA_VISIBLE_DEVICES=3" in result.stdout


def test_fastmri_train_entrypoint_builds_model_at_train_patch_size(monkeypatch, tmp_path: Path):
    with initialize(version_base=None, config_path="../src/snraware/projects/mri/denoising/configs"):
        config = compose(config_name="fastmri_finetune")

    config.logging.use_wandb = False
    config.fastmri_finetune.train_root = "/tmp/train"
    config.fastmri_finetune.val_root = "/tmp/val"
    config.fastmri_finetune.test_root = None
    config.fastmri_finetune.save_root = str(tmp_path)
    config.fastmri_finetune.device = "cpu"
    config.fastmri_finetune.use_bf16 = False
    config.fastmri_finetune.train_patch_size = [128, 128]
    config.fastmri_finetune.crop_size = [320, 320]

    fake_fastmri = types.ModuleType("fastmri")
    fake_fastmri_evaluate = types.ModuleType("fastmri.evaluate")
    fake_fastmri_evaluate.nmse = lambda *args, **kwargs: 0.0
    monkeypatch.setitem(sys.modules, "fastmri", fake_fastmri)
    monkeypatch.setitem(sys.modules, "fastmri.evaluate", fake_fastmri_evaluate)

    monkeypatch.setattr(
        fastmri_train_module,
        "resolve_base_model_paths",
        lambda **kwargs: ("./cfg.yaml", "./weights.pts"),
    )
    monkeypatch.setattr(
        fastmri_train_module,
        "build_fastmri_dataloaders",
        lambda config: ("train_loader", "val_loader", None),
    )
    monkeypatch.setattr(
        fastmri_train_module,
        "_resolve_device",
        lambda device_str: torch.device("cpu"),
    )
    monkeypatch.setattr(
        fastmri_train_module,
        "resolve_fastmri_precision",
        lambda device, use_bf16: {"mode": "fp32", "use_bf16": False},
    )
    monkeypatch.setattr(fastmri_train_module, "seed_everything", lambda seed: None)

    build_calls = {}

    def fake_build_fastmri_wrapped_model(**kwargs):
        build_calls.update(kwargs)
        return (
            torch.nn.Identity(),
            OmegaConf.create({}),
            {
                "weight_source": "jit",
                "matched_keys": 0,
                "mismatched_keys": 0,
                "total_model_keys": 0,
            },
        )

    monkeypatch.setattr(fastmri_train_module, "build_fastmri_wrapped_model", fake_build_fastmri_wrapped_model)

    trainer_kwargs = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            trainer_kwargs.update(kwargs)

        def train(self):
            return {}

    monkeypatch.setattr(fastmri_train_module, "FastMRIFineTuneTrainer", FakeTrainer)

    fastmri_train_module.run_fastmri_finetuning_from_config(config)

    assert build_calls["height"] == 128
    assert build_calls["width"] == 128
    assert build_calls["use_unet"] is True
    assert trainer_kwargs["train_loader"] == "train_loader"
    assert trainer_kwargs["val_loader"] == "val_loader"


def test_fastmri_train_entrypoint_prints_native_size_and_partial_load_warning(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    with initialize(version_base=None, config_path="../src/snraware/projects/mri/denoising/configs"):
        config = compose(config_name="fastmri_finetune")

    config.logging.use_wandb = False
    config.fastmri_finetune.train_root = "/tmp/train"
    config.fastmri_finetune.val_root = None
    config.fastmri_finetune.test_root = None
    config.fastmri_finetune.save_root = str(tmp_path)
    config.fastmri_finetune.device = "cpu"
    config.fastmri_finetune.use_bf16 = False
    config.fastmri_finetune.train_patch_size = [128, 128]
    config.fastmri_finetune.crop_size = [320, 320]

    fake_fastmri = types.ModuleType("fastmri")
    fake_fastmri_evaluate = types.ModuleType("fastmri.evaluate")
    fake_fastmri_evaluate.nmse = lambda *args, **kwargs: 0.0
    monkeypatch.setitem(sys.modules, "fastmri", fake_fastmri)
    monkeypatch.setitem(sys.modules, "fastmri.evaluate", fake_fastmri_evaluate)

    monkeypatch.setattr(
        fastmri_train_module,
        "resolve_base_model_paths",
        lambda **kwargs: ("./cfg.yaml", "./weights.pts"),
    )
    monkeypatch.setattr(
        fastmri_train_module,
        "build_fastmri_dataloaders",
        lambda config: ("train_loader", None, None),
    )
    monkeypatch.setattr(
        fastmri_train_module,
        "_resolve_device",
        lambda device_str: torch.device("cpu"),
    )
    monkeypatch.setattr(
        fastmri_train_module,
        "resolve_fastmri_precision",
        lambda device, use_bf16: {"mode": "fp32", "use_bf16": False},
    )
    monkeypatch.setattr(fastmri_train_module, "seed_everything", lambda seed: None)

    monkeypatch.setattr(
        fastmri_train_module,
        "build_fastmri_wrapped_model",
        lambda **kwargs: (
            torch.nn.Identity(),
            OmegaConf.create({}),
            {
                "weight_source": "jit",
                "matched_keys": 218,
                "mismatched_keys": 70,
                "total_model_keys": 288,
                "native_spatial_size": [64, 64],
                "model_spatial_size": [128, 128],
                "mismatch_key_examples": ["bk.B00.block.cell_0.n1.ln.weight"],
            },
        ),
    )

    class FakeTrainer:
        def __init__(self, **kwargs):
            pass

        def train(self):
            return {}

    monkeypatch.setattr(fastmri_train_module, "FastMRIFineTuneTrainer", FakeTrainer)

    fastmri_train_module.run_fastmri_finetuning_from_config(config)

    captured = capsys.readouterr()
    assert "FastMRI train input: 128x128 | eval crop: 320x320 | eval sliding-window: enabled" in captured.out
    assert "Checkpoint native spatial size: 64x64 | current model spatial size: 128x128 | full match: no" in captured.out
    assert "Partial pretrained load detected." in captured.out
    assert "Example mismatched keys: bk.B00.block.cell_0.n1.ln.weight" in captured.out


def test_fastmri_hydra_entrypoint_delegates_to_shared_runner():
    config = OmegaConf.create({"foo": "bar"})
    expected = {"status": "ok"}
    captured = {}

    def fake_runner(incoming_config):
        captured["config"] = incoming_config
        return expected

    original_runner = fastmri_train_module.run_fastmri_finetuning_from_config
    fastmri_train_module.run_fastmri_finetuning_from_config = fake_runner
    try:
        result = fastmri_train_module.run_fastmri_finetuning.__wrapped__(config)
    finally:
        fastmri_train_module.run_fastmri_finetuning_from_config = original_runner

    assert captured["config"] is config
    assert result == expected


def test_e2e_pipeline_defaults_to_small_model_preset(monkeypatch):
    import test_e2e_lora_pipeline as e2e_module

    monkeypatch.setattr(sys, "argv", ["test_e2e_lora_pipeline.py"])
    args = e2e_module.parse_args()

    assert args.model_size == "small"
    assert args.config_path is None
    assert args.weight_path is None

    config_path, checkpoint_path = resolve_base_model_paths(
        variant=args.model_size,
        config_path=args.config_path,
        checkpoint_path=args.weight_path,
        repo_root=Path(__file__).resolve().parents[1],
    )
    assert config_path.endswith("checkpoints/small/snraware_small_model.yaml")
    assert checkpoint_path.endswith("checkpoints/small/snraware_small_model.pts")


def test_e2e_pipeline_allows_large_model_preset(monkeypatch):
    import test_e2e_lora_pipeline as e2e_module

    monkeypatch.setattr(sys, "argv", ["test_e2e_lora_pipeline.py", "--model_size", "large"])
    args = e2e_module.parse_args()

    config_path, checkpoint_path = resolve_base_model_paths(
        variant=args.model_size,
        config_path=args.config_path,
        checkpoint_path=args.weight_path,
        repo_root=Path(__file__).resolve().parents[1],
    )
    assert config_path.endswith("checkpoints/large/snraware_large_model.yaml")
    assert checkpoint_path.endswith("checkpoints/large/snraware_large_model.pts")


@pytest.mark.parametrize(
    ("mode", "expect_gfactor_trainable", "expect_adapter_trainable"),
    [
        ("unet_only", True, False),
        ("unet_and_lora", True, True),
        ("lora_only", False, True),
    ],
)
def test_resume_reconciles_mode_trainable_state(
    mode: str,
    expect_gfactor_trainable: bool,
    expect_adapter_trainable: bool,
    tmp_path: Path,
):
    config = _tiny_config(mode)
    train_loader = DataLoader(ToyFastMRIDataset(2), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model(mode),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / f"{mode}_save",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )
    checkpoint_path = trainer._save_checkpoint(tmp_path / f"{mode}.pth", epoch=0, metrics={})

    resumed_config = _tiny_config(mode)
    resumed_config.fastmri_finetune.resume_from = str(checkpoint_path)
    resumed_trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model(mode),
        config=resumed_config,
        device=torch.device("cpu"),
        run_dir=tmp_path / f"{mode}_resume",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )

    assert _gfactor_is_trainable(resumed_trainer.model) is expect_gfactor_trainable
    assert _domain_scale_is_trainable(resumed_trainer.model) is True
    assert bool(_adapter_trainable_names(resumed_trainer.model)) is expect_adapter_trainable
    assert _pre_post_trainable_names(resumed_trainer.model) == []
    assert _backbone_trainable_names(resumed_trainer.model) == []


def test_resume_reconciles_warmup_then_both_before_activation(tmp_path: Path):
    config = _tiny_config("warmup_then_both")
    config.fastmri_finetune.warmup_epochs = 2
    train_loader = DataLoader(ToyFastMRIDataset(2), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("warmup_then_both"),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "warmup_save",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )
    checkpoint_path = trainer._save_checkpoint(tmp_path / "warmup_false.pth", epoch=0, metrics={})

    resumed_config = _tiny_config("warmup_then_both")
    resumed_config.fastmri_finetune.warmup_epochs = 2
    resumed_config.fastmri_finetune.resume_from = str(checkpoint_path)
    resumed_trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("warmup_then_both"),
        config=resumed_config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "warmup_resume",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )

    assert resumed_trainer.mode_state["adapters_active"] is False
    assert _gfactor_is_trainable(resumed_trainer.model) is True
    assert _domain_scale_is_trainable(resumed_trainer.model) is True
    assert _adapter_trainable_names(resumed_trainer.model) == []
    assert _pre_post_trainable_names(resumed_trainer.model) == []
    assert _backbone_trainable_names(resumed_trainer.model) == []
    assert _adapter_group_lr(resumed_trainer) == 0.0


def test_resume_reconciles_warmup_then_both_after_activation(tmp_path: Path):
    config = _tiny_config("warmup_then_both")
    config.fastmri_finetune.warmup_epochs = 1
    train_loader = DataLoader(ToyFastMRIDataset(2), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("warmup_then_both"),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "warmup_active_save",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )
    trainer._maybe_activate_adapters(epoch=1)
    checkpoint_path = trainer._save_checkpoint(tmp_path / "warmup_true.pth", epoch=1, metrics={})

    resumed_config = _tiny_config("warmup_then_both")
    resumed_config.fastmri_finetune.warmup_epochs = 1
    resumed_config.fastmri_finetune.resume_from = str(checkpoint_path)
    resumed_trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("warmup_then_both"),
        config=resumed_config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "warmup_active_resume",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )

    assert resumed_trainer.mode_state["adapters_active"] is True
    assert _gfactor_is_trainable(resumed_trainer.model) is True
    assert _domain_scale_is_trainable(resumed_trainer.model) is True
    assert bool(_adapter_trainable_names(resumed_trainer.model)) is True
    assert _pre_post_trainable_names(resumed_trainer.model) == []
    assert _backbone_trainable_names(resumed_trainer.model) == []
    assert _adapter_group_lr(resumed_trainer) == pytest.approx(float(config.fastmri_finetune.adapter_lr))


def test_resume_rejects_mode_mismatch(tmp_path: Path):
    source_config = _tiny_config("unet_and_lora")
    train_loader = DataLoader(ToyFastMRIDataset(2), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("unet_and_lora"),
        config=source_config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "mismatch_save",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )
    checkpoint_path = trainer._save_checkpoint(tmp_path / "mismatch.pth", epoch=0, metrics={})

    resumed_config = _tiny_config("lora_only")
    resumed_config.fastmri_finetune.resume_from = str(checkpoint_path)

    with pytest.raises(ValueError, match="resume mode mismatch"):
        FastMRIFineTuneTrainer(
            model=_build_wrapped_model("lora_only"),
            config=resumed_config,
            device=torch.device("cpu"),
            run_dir=tmp_path / "mismatch_resume",
            train_loader=train_loader,
            val_loader=None,
            test_loader=None,
        )


def test_resume_rejects_use_unet_mismatch(tmp_path: Path):
    source_config = _tiny_config("lora_only")
    source_config.fastmri_finetune.use_unet = False
    train_loader = DataLoader(ToyFastMRIDataset(2), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("lora_only", use_unet=False),
        config=source_config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "use_unet_mismatch_save",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )
    checkpoint_path = trainer._save_checkpoint(tmp_path / "use_unet_mismatch.pth", epoch=0, metrics={})

    resumed_config = _tiny_config("lora_only")
    resumed_config.fastmri_finetune.use_unet = True
    resumed_config.fastmri_finetune.resume_from = str(checkpoint_path)

    with pytest.raises(ValueError, match="resume USE_UNET mismatch"):
        FastMRIFineTuneTrainer(
            model=_build_wrapped_model("lora_only", use_unet=True),
            config=resumed_config,
            device=torch.device("cpu"),
            run_dir=tmp_path / "use_unet_mismatch_resume",
            train_loader=train_loader,
            val_loader=None,
            test_loader=None,
        )


def test_resume_allows_legacy_checkpoint_without_use_unet(tmp_path: Path):
    source_config = _tiny_config("lora_only")
    source_config.fastmri_finetune.use_unet = False
    train_loader = DataLoader(ToyFastMRIDataset(2), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("lora_only", use_unet=False),
        config=source_config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "legacy_use_unet_save",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )
    checkpoint_path = trainer._save_checkpoint(tmp_path / "legacy_use_unet.pth", epoch=0, metrics={})

    payload = torch.load(checkpoint_path, map_location="cpu")
    payload.pop("use_unet", None)
    torch.save(payload, checkpoint_path)

    resumed_config = _tiny_config("lora_only")
    resumed_config.fastmri_finetune.use_unet = False
    resumed_config.fastmri_finetune.resume_from = str(checkpoint_path)
    resumed_trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("lora_only", use_unet=False),
        config=resumed_config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "legacy_use_unet_resume",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )

    assert resumed_trainer.use_unet is False
    assert resumed_trainer.effective_mode == "lora_only"
    assert _domain_scale_is_trainable(resumed_trainer.model) is True


def test_resume_skips_legacy_optimizer_state_without_domain_scale(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    config = _tiny_config("lora_only")
    train_loader = DataLoader(ToyFastMRIDataset(2), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("lora_only"),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "legacy_optimizer_save",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )
    checkpoint_path = trainer._save_checkpoint(tmp_path / "legacy_optimizer_state.pth", epoch=0, metrics={})

    payload = torch.load(checkpoint_path, map_location="cpu")
    payload.pop("domain_scale", None)
    payload["optimizer_state_dict"]["param_groups"][0]["params"] = payload["optimizer_state_dict"][
        "param_groups"
    ][0]["params"][:-1]
    torch.save(payload, checkpoint_path)

    resumed_config = _tiny_config("lora_only")
    resumed_config.fastmri_finetune.resume_from = str(checkpoint_path)
    resumed_trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("lora_only"),
        config=resumed_config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "legacy_optimizer_resume",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )

    captured = capsys.readouterr()
    assert "Skipping optimizer state restore" in captured.out
    assert resumed_trainer.current_epoch == 1
    assert resumed_trainer.model.domain_scale.item() == pytest.approx(1.0)
    assert _domain_scale_is_trainable(resumed_trainer.model) is True


@pytest.mark.parametrize(
    ("mode", "epoch", "expect_checkpoint"),
    [
        ("unet_only", 0, True),
        ("unet_and_lora", 0, True),
        ("lora_only", 0, True),
        ("warmup_then_both", 0, True),
        ("warmup_then_both", 1, True),
    ],
)
def test_train_gradient_checkpointing_runs_whenever_enabled_during_training(
    mode: str,
    epoch: int,
    expect_checkpoint: bool,
    monkeypatch,
    tmp_path: Path,
):
    config = _tiny_config(mode)
    config.fastmri_finetune.warmup_epochs = 1
    train_loader = DataLoader(ToyFastMRIDataset(1), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model(mode),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / f"{mode}_gc",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )

    checkpoint_calls = []

    def fake_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(kwargs)
        return function(*args)

    monkeypatch.setattr(fastmri_compat_module.checkpoint_utils, "checkpoint", fake_checkpoint)

    trainer.train_one_epoch(epoch)

    assert bool(checkpoint_calls) is expect_checkpoint
    if expect_checkpoint:
        assert checkpoint_calls == [{"use_reentrant": False}]


def test_eval_never_uses_gradient_checkpointing(monkeypatch, tmp_path: Path):
    config = _tiny_config("unet_only")
    loader = DataLoader(ToyFastMRIDataset(1), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("unet_only"),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "eval_gc",
        train_loader=loader,
        val_loader=loader,
        test_loader=None,
        metric_fns={
            "psnr": lambda gt, pred: float(np.mean(gt - pred)),
            "ssim": lambda gt, pred: float(np.mean(gt + pred)),
            "nmse": lambda gt, pred: float(np.mean(pred * 0)),
        },
    )

    checkpoint_calls = []

    def fake_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(kwargs)
        return function(*args)

    monkeypatch.setattr(fastmri_compat_module.checkpoint_utils, "checkpoint", fake_checkpoint)

    trainer.evaluate_loader(loader, split="val")

    assert checkpoint_calls == []


def test_should_checkpoint_frozen_base_stays_enabled_with_active_lora():
    model = _build_wrapped_model("unet_and_lora")
    configure_model_for_finetune_mode(
        model,
        mode="unet_and_lora",
        lora_config=model.base_model.config.lora,
        train_pre_post=False,
    )

    model.train()
    assert trainer_fa_module.should_checkpoint_frozen_base(
        model,
        gradient_checkpoint_frozen_base=True,
    ) is True


def test_gradient_checkpointing_keeps_gfactor_gradients():
    model = _build_wrapped_model("unet_only")
    configure_model_for_finetune_mode(model, mode="unet_only", lora_config=model.base_model.config.lora)

    noisy = torch.randn(1, 2, 32, 32, dtype=torch.float32)
    output = model(noisy, checkpoint_base_model=True)
    loss = complex_output_to_magnitude(output).sum()
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.gfactor_unet.parameters())
    assert model.domain_scale.grad is not None
    assert torch.isfinite(model.domain_scale.grad).all()


def test_gradient_checkpointing_keeps_lora_gradients_when_adapters_active():
    model = _build_wrapped_model("unet_and_lora")
    configure_model_for_finetune_mode(
        model,
        mode="unet_and_lora",
        lora_config=model.base_model.config.lora,
        train_pre_post=False,
    )

    noisy = torch.randn(1, 2, 32, 32, dtype=torch.float32)
    output = model(noisy, checkpoint_base_model=True)
    loss = complex_output_to_magnitude(output).sum()
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.gfactor_unet.parameters())
    assert model.domain_scale.grad is not None
    assert torch.isfinite(model.domain_scale.grad).all()
    assert any(
        parameter.grad is not None
        for name, parameter in model.base_model.named_parameters()
        if ".lora_" in name
    )


def test_lora_only_without_unet_keeps_domain_scale_gradients():
    model = _build_wrapped_model("lora_only", use_unet=False)
    configure_model_for_finetune_mode(
        model,
        mode="lora_only",
        lora_config=model.base_model.config.lora,
        train_pre_post=False,
        use_unet=False,
    )

    noisy = torch.randn(1, 2, 32, 32, dtype=torch.float32)
    output = model(noisy, checkpoint_base_model=True)
    loss = complex_output_to_magnitude(output).sum()
    loss.backward()

    assert model.domain_scale.grad is not None
    assert torch.isfinite(model.domain_scale.grad).all()
    assert all(parameter.grad is None for parameter in model.gfactor_unet.parameters())


def test_denoising_model_forward_backward_smoke():
    config = _tiny_config("unet_only")
    model = DenoisingModel(config=config, D=1, H=32, W=32)
    noisy = torch.randn(1, 3, 1, 32, 32, dtype=torch.float32, requires_grad=True)

    output = model(noisy)
    loss = output.sum()
    loss.backward()

    assert output.shape == (1, 2, 1, 32, 32)
    assert noisy.grad is not None


def test_warmup_activation_updates_scheduler_base_lrs(tmp_path: Path):
    config = _tiny_config("warmup_then_both")
    config.fastmri_finetune.warmup_epochs = 1
    config.fastmri_finetune.scheduler_t_max = 4
    train_loader = DataLoader(ToyFastMRIDataset(1), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("warmup_then_both"),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "warmup_sched",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )

    assert trainer.scheduler is not None
    assert _adapter_group_lr(trainer) == 0.0
    assert trainer.scheduler.base_lrs[1] == 0.0

    trainer._maybe_activate_adapters(epoch=1)

    assert _adapter_group_lr(trainer) == pytest.approx(float(config.fastmri_finetune.adapter_lr))
    assert trainer.scheduler.base_lrs[1] == pytest.approx(float(config.fastmri_finetune.adapter_lr))


def test_fastmri_optimizer_honors_train_pre_post_boundary(tmp_path: Path):
    config = _tiny_config("unet_and_lora")
    config.fastmri_finetune.train_pre_post = False
    train_loader = DataLoader(ToyFastMRIDataset(1), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("unet_and_lora"),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "optimizer_boundary",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )

    adapter_param_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        if group.get("name") == "adapter"
        for parameter in group["params"]
    }
    gfactor_param_ids = _group_param_ids(trainer, "gfactor_unet")
    pre_post_param_ids = {
        id(parameter)
        for name, parameter in trainer.model.base_model.named_parameters()
        if name.startswith("pre.") or name.startswith("post.")
    }

    assert adapter_param_ids
    assert id(trainer.model.domain_scale) in gfactor_param_ids
    assert adapter_param_ids.isdisjoint(pre_post_param_ids)

    config.fastmri_finetune.train_pre_post = True
    trainer_with_pre_post = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("unet_and_lora"),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "optimizer_boundary_with_pre_post",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )

    adapter_param_ids_with_pre_post = {
        id(parameter)
        for group in trainer_with_pre_post.optimizer.param_groups
        if group.get("name") == "adapter"
        for parameter in group["params"]
    }
    gfactor_param_ids_with_pre_post = _group_param_ids(trainer_with_pre_post, "gfactor_unet")
    pre_post_param_ids_with_pre_post = {
        id(parameter)
        for name, parameter in trainer_with_pre_post.model.base_model.named_parameters()
        if name.startswith("pre.") or name.startswith("post.")
    }

    assert id(trainer_with_pre_post.model.domain_scale) in gfactor_param_ids_with_pre_post
    assert pre_post_param_ids_with_pre_post <= adapter_param_ids_with_pre_post


@pytest.mark.parametrize("mode", ["unet_only", "unet_and_lora"])
def test_fastmri_trainer_smoke_step(mode: str, tmp_path: Path):
    config = _tiny_config(mode)
    config.fastmri_finetune.max_epochs = 1
    config.fastmri_finetune.log_every_n_steps = 100
    config.fastmri_finetune.evaluate_every_n_epochs = 1
    config.fastmri_finetune.unet_lr = 1e-3
    config.fastmri_finetune.adapter_lr = 1e-3
    config.fastmri_finetune.scheduler_t_max = 0

    metric_fns = {
        "psnr": lambda gt, pred: float(-np.mean((gt - pred) ** 2)),
        "ssim": lambda gt, pred: float(1.0 / (1.0 + np.mean((gt - pred) ** 2))),
        "nmse": lambda gt, pred: float(np.sum((gt - pred) ** 2) / max(np.sum(gt**2), 1e-12)),
    }

    train_loader = DataLoader(ToyFastMRIDataset(4), batch_size=2, shuffle=False)
    val_loader = DataLoader(ToyFastMRIDataset(4), batch_size=2, shuffle=False)

    model = _build_wrapped_model(mode)
    trainer = FastMRIFineTuneTrainer(
        model=model,
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / mode,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        metric_fns=metric_fns,
    )

    train_metrics = trainer.train_one_epoch(0)
    val_metrics = trainer.evaluate_loader(val_loader, split="val")

    assert "loss" in train_metrics
    assert np.isfinite(train_metrics["loss"])
    assert set(val_metrics) == {"loss", "psnr", "ssim", "nmse"}
    assert all(np.isfinite(value) for value in val_metrics.values())


def test_fastmri_trainer_smoke_step_stays_finite_on_zero_background_patches(tmp_path: Path):
    config = _tiny_config("unet_only")
    config.fastmri_finetune.max_epochs = 1
    config.fastmri_finetune.log_every_n_steps = 100
    config.fastmri_finetune.evaluate_every_n_epochs = 1
    config.fastmri_finetune.unet_lr = 1e-3
    config.fastmri_finetune.scheduler_t_max = 0

    train_loader = DataLoader(FlatFastMRIDataset(4, size=64, fill_value=0.0), batch_size=2, shuffle=False)
    model = _build_wrapped_model("unet_only", size=64)
    trainer = FastMRIFineTuneTrainer(
        model=model,
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "zero_background",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )

    train_metrics = trainer.train_one_epoch(0)

    assert "loss" in train_metrics
    assert np.isfinite(train_metrics["loss"])
    assert train_metrics["num_skipped_nonfinite_steps"] == 0
    assert np.isfinite(train_metrics["mean_grad_norm_gfactor_unet"])
    assert np.isfinite(train_metrics["mean_gfactor_mean"])


def test_fastmri_trainer_eval_uses_patch_inference_for_full_resolution_metrics(monkeypatch, tmp_path: Path):
    config = _tiny_config("unet_only")
    config.fastmri_finetune.train_patch_size = [32, 32]
    config.overlap_for_inference = [8, 8, 0]

    train_loader = DataLoader(ToyFastMRIDataset(2, size=32), batch_size=1, shuffle=False)
    val_loader = DataLoader(ToyFastMRIDataset(2, size=64), batch_size=1, shuffle=False)

    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("unet_only", size=32),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "patch_eval",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        metric_fns={
            "psnr": lambda gt, pred: float(-np.mean((gt - pred) ** 2)),
            "ssim": lambda gt, pred: float(1.0 / (1.0 + np.mean((gt - pred) ** 2))),
            "nmse": lambda gt, pred: float(np.sum((gt - pred) ** 2) / max(np.sum(gt**2), 1e-12)),
        },
    )

    patch_calls = []
    original_patch_inference = trainer._run_eval_patch_inference

    def wrapped_patch_inference(noisy: torch.Tensor) -> torch.Tensor:
        patch_calls.append(tuple(noisy.shape))
        return original_patch_inference(noisy)

    monkeypatch.setattr(trainer, "_run_eval_patch_inference", wrapped_patch_inference)

    metrics = trainer.evaluate_loader(val_loader, split="val")

    assert patch_calls == [(1, 2, 64, 64), (1, 2, 64, 64)]
    assert set(metrics) == {"loss", "psnr", "ssim", "nmse"}
    assert all(np.isfinite(value) for value in metrics.values())


def test_fastmri_warmup_keeps_base_model_in_eval_mode_until_adapters_activate(monkeypatch, tmp_path: Path):
    config = _tiny_config("warmup_then_both")
    config.fastmri_finetune.warmup_epochs = 1
    train_loader = DataLoader(ToyFastMRIDataset(2, size=32), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("warmup_then_both", size=32),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "warmup_base_eval",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )

    original_forward = trainer.model.base_model.forward
    epoch0_flags: list[bool] = []
    epoch1_flags: list[bool] = []

    def wrapped_forward_epoch0(x: torch.Tensor):
        epoch0_flags.append(trainer.model.base_model.training)
        return original_forward(x)

    monkeypatch.setattr(trainer.model.base_model, "forward", wrapped_forward_epoch0)
    trainer.train_one_epoch(0)

    def wrapped_forward_epoch1(x: torch.Tensor):
        epoch1_flags.append(trainer.model.base_model.training)
        return original_forward(x)

    monkeypatch.setattr(trainer.model.base_model, "forward", wrapped_forward_epoch1)
    trainer.train_one_epoch(1)

    assert epoch0_flags
    assert all(flag is False for flag in epoch0_flags)
    assert epoch1_flags
    assert all(flag is True for flag in epoch1_flags)


def test_fastmri_use_unet_false_short_circuits_warmup_to_adapter_training(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    config = _tiny_config("warmup_then_both")
    config.fastmri_finetune.use_unet = False

    train_loader = DataLoader(ToyFastMRIDataset(2, size=32), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("warmup_then_both", size=32, use_unet=False),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "use_unet_false_warmup",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )

    captured = capsys.readouterr()
    assert "USE_UNET=false bypasses the FastMRI g-factor U-Net" in captured.out
    assert trainer.requested_mode == "warmup_then_both"
    assert trainer.effective_mode == "lora_only"
    assert trainer.warmup_bypassed is True
    assert trainer.mode_state["adapters_active"] is True
    assert _gfactor_is_trainable(trainer.model) is False
    assert _domain_scale_is_trainable(trainer.model) is True
    assert trainer._current_group_lr("gfactor_unet") == pytest.approx(float(config.fastmri_finetune.unet_lr))
    assert _adapter_group_lr(trainer) == pytest.approx(float(config.fastmri_finetune.adapter_lr))


def test_fastmri_use_unet_false_turns_unet_and_lora_into_lora_only(tmp_path: Path):
    config = _tiny_config("unet_and_lora")
    config.fastmri_finetune.use_unet = False

    train_loader = DataLoader(ToyFastMRIDataset(2, size=32), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("unet_and_lora", size=32, use_unet=False),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "use_unet_false_unet_and_lora",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )

    assert trainer.requested_mode == "unet_and_lora"
    assert trainer.effective_mode == "lora_only"
    assert trainer.mode_state["adapters_active"] is True
    assert _gfactor_is_trainable(trainer.model) is False
    assert _domain_scale_is_trainable(trainer.model) is True


def test_fastmri_use_unet_false_preserves_train_pre_post_for_adapter_phase(tmp_path: Path):
    config = _tiny_config("warmup_then_both")
    config.fastmri_finetune.use_unet = False
    config.fastmri_finetune.train_pre_post = True

    train_loader = DataLoader(ToyFastMRIDataset(2, size=32), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("warmup_then_both", size=32, use_unet=False),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "use_unet_false_train_pre_post",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )

    assert trainer.effective_mode == "lora_only"
    assert _gfactor_is_trainable(trainer.model) is False
    assert _domain_scale_is_trainable(trainer.model) is True
    assert _pre_post_trainable_names(trainer.model)
    assert _adapter_trainable_names(trainer.model)


def test_fastmri_use_unet_false_rejects_unet_only(tmp_path: Path):
    config = _tiny_config("unet_only")
    config.fastmri_finetune.use_unet = False

    train_loader = DataLoader(ToyFastMRIDataset(2, size=32), batch_size=1, shuffle=False)

    with pytest.raises(ValueError, match="USE_UNET=false is incompatible with MODE=unet_only"):
        FastMRIFineTuneTrainer(
            model=_build_wrapped_model("unet_only", size=32, use_unet=False),
            config=config,
            device=torch.device("cpu"),
            run_dir=tmp_path / "use_unet_false_unet_only",
            train_loader=train_loader,
            val_loader=None,
            test_loader=None,
        )


def test_fastmri_eval_uses_inference_mode_and_eval_flag(monkeypatch, tmp_path: Path):
    config = _tiny_config("unet_only")
    config.fastmri_finetune.train_patch_size = [32, 32]
    config.fastmri_finetune.eval_patch_batch_size = 4
    config.overlap_for_inference = [8, 8, 0]

    train_loader = DataLoader(ToyFastMRIDataset(1, size=32), batch_size=1, shuffle=False)
    val_loader = DataLoader(ToyFastMRIDataset(1, size=64), batch_size=1, shuffle=False)

    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("unet_only", size=32),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "eval_mode_check",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        metric_fns={
            "psnr": lambda gt, pred: float(-np.mean((gt - pred) ** 2)),
            "ssim": lambda gt, pred: float(1.0 / (1.0 + np.mean((gt - pred) ** 2))),
            "nmse": lambda gt, pred: float(np.sum((gt - pred) ** 2) / max(np.sum(gt**2), 1e-12)),
        },
    )

    flags: list[tuple[bool, bool]] = []
    original_forward = trainer.model.forward

    def wrapped_forward(*args, **kwargs):
        flags.append((trainer.model.training, torch.is_inference_mode_enabled()))
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(trainer.model, "forward", wrapped_forward)

    trainer.evaluate_loader(val_loader, split="val")

    assert flags
    assert all(training_flag is False for training_flag, _ in flags)
    assert all(inference_flag is True for _, inference_flag in flags)


def test_fastmri_trainer_skips_nonfinite_steps_without_updating_optimizer(
    monkeypatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    config = _tiny_config("unet_only")
    config.fastmri_finetune.log_every_n_steps = 1

    train_loader = DataLoader(ToyFastMRIDataset(2, size=32), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("unet_only", size=32),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "skip_nonfinite",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
    )

    original_forward = trainer.model.forward
    call_state = {"count": 0}

    def wrapped_forward(x: torch.Tensor, **kwargs):
        call_state["count"] += 1
        if call_state["count"] == 1:
            return torch.full((x.shape[0], 2, 1, x.shape[-2], x.shape[-1]), float("nan"), dtype=torch.float32)
        return original_forward(x, **kwargs)

    monkeypatch.setattr(trainer.model, "forward", wrapped_forward)

    train_metrics = trainer.train_one_epoch(0)

    captured = capsys.readouterr()
    assert train_metrics["num_skipped_nonfinite_steps"] == 1
    assert np.isfinite(train_metrics["loss"])
    assert "[Train skip 1/2]" in captured.out
    assert "reason=nonfinite_output" in captured.out


def test_fastmri_eval_patch_inference_uses_eval_patch_batch_size(monkeypatch, tmp_path: Path):
    config = _tiny_config("unet_only")
    config.fastmri_finetune.train_patch_size = [32, 32]
    config.fastmri_finetune.batch_size = 1
    config.fastmri_finetune.eval_patch_batch_size = 3
    config.overlap_for_inference = [8, 8, 0]

    train_loader = DataLoader(ToyFastMRIDataset(1, size=32), batch_size=1, shuffle=False)
    val_loader = DataLoader(ToyFastMRIDataset(1, size=64), batch_size=1, shuffle=False)

    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("unet_only", size=32),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "eval_patch_batch_size",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        metric_fns={
            "psnr": lambda gt, pred: float(-np.mean((gt - pred) ** 2)),
            "ssim": lambda gt, pred: float(1.0 / (1.0 + np.mean((gt - pred) ** 2))),
            "nmse": lambda gt, pred: float(np.sum((gt - pred) ** 2) / max(np.sum(gt**2), 1e-12)),
        },
    )

    patch_batch_sizes: list[int] = []
    original_forward = trainer.model.forward

    def wrapped_forward(x: torch.Tensor, **kwargs):
        patch_batch_sizes.append(int(x.shape[0]))
        return original_forward(x, **kwargs)

    monkeypatch.setattr(trainer.model, "forward", wrapped_forward)

    trainer.evaluate_loader(val_loader, split="val")

    assert patch_batch_sizes == [3, 3, 3]


def test_fastmri_trainer_prints_console_progress_summaries(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config = _tiny_config("unet_only")
    config.fastmri_finetune.max_epochs = 1
    config.fastmri_finetune.log_every_n_steps = 1
    config.fastmri_finetune.evaluate_every_n_epochs = 1

    metric_fns = {
        "psnr": lambda gt, pred: float(-np.mean((gt - pred) ** 2)),
        "ssim": lambda gt, pred: float(1.0 / (1.0 + np.mean((gt - pred) ** 2))),
        "nmse": lambda gt, pred: float(np.sum((gt - pred) ** 2) / max(np.sum(gt**2), 1e-12)),
    }

    train_loader = DataLoader(ToyFastMRIDataset(2), batch_size=1, shuffle=False)
    val_loader = DataLoader(ToyFastMRIDataset(2), batch_size=1, shuffle=False)

    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("unet_only"),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "console_progress",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        metric_fns=metric_fns,
    )

    trainer.train()

    captured = capsys.readouterr()
    assert "[FastMRI trainer]" in captured.out
    assert "[Epoch 1/1]" in captured.out
    assert "[Train step 1/2]" in captured.out
    assert "[Train]" in captured.out
    assert "[Val]" in captured.out
    assert "[Checkpoint] Saved last checkpoint" in captured.out
    assert "gfactor_max=" in captured.out
    assert "grad_unet=" in captured.out
    assert "skipped_nonfinite_steps=0" in captured.out


@pytest.mark.parametrize(
    ("evaluate_every", "max_epochs", "expected_val_epochs"),
    [
        (10, 4, [3]),
        (2, 5, [1, 3, 4]),
        (1, 3, [0, 1, 2]),
        (0, 4, [3]),
    ],
)
def test_fastmri_trainer_runs_validation_on_interval_and_final_epoch(
    monkeypatch,
    tmp_path: Path,
    evaluate_every: int,
    max_epochs: int,
    expected_val_epochs: list[int],
):
    config = _tiny_config("unet_only")
    config.fastmri_finetune.max_epochs = max_epochs
    config.fastmri_finetune.evaluate_every_n_epochs = evaluate_every

    train_loader = DataLoader(ToyFastMRIDataset(2), batch_size=1, shuffle=False)
    val_loader = DataLoader(ToyFastMRIDataset(2), batch_size=1, shuffle=False)

    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("unet_only"),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / f"eval_every_{evaluate_every}_{max_epochs}",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
    )

    monkeypatch.setattr(trainer, "train_one_epoch", lambda epoch: {"loss": float(epoch)})
    val_calls: list[tuple[int, str]] = []

    def fake_evaluate_loader(loader, *, split: str):
        val_calls.append((trainer.current_epoch, split))
        return {"loss": 0.0, "psnr": 1.0, "ssim": 1.0, "nmse": 0.0}

    monkeypatch.setattr(trainer, "evaluate_loader", fake_evaluate_loader)
    monkeypatch.setattr(trainer, "_save_checkpoint", lambda path, *, epoch, metrics: Path(path))

    results = trainer.train()

    assert [epoch for epoch, split in val_calls if split == "val"] == expected_val_epochs
    assert sorted(int(key.split("_")[-1]) for key in results if key.startswith("val_epoch_")) == expected_val_epochs


def test_fastmri_trainer_train_uses_bf16_context_but_loss_stays_fp32(monkeypatch, tmp_path: Path):
    config = _tiny_config("unet_only")
    config.fastmri_finetune.max_epochs = 1
    config.fastmri_finetune.use_bf16 = True

    train_loader = DataLoader(ToyFastMRIDataset(2), batch_size=1, shuffle=False)
    model = _build_wrapped_model("unet_only")
    trainer = FastMRIFineTuneTrainer(
        model=model,
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "bf16_train",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
        precision_state={"mode": "bf16", "use_bf16": True},
    )

    autocast_calls = []

    @contextmanager
    def fake_context(*, enabled: bool):
        autocast_calls.append(enabled)
        yield

    monkeypatch.setattr(trainer, "_autocast_context", lambda *, enabled: fake_context(enabled=enabled))
    monkeypatch.setattr(
        trainer_fa_module,
        "complex_output_to_magnitude",
        lambda output: complex_output_to_magnitude(output).to(torch.bfloat16),
    )
    capture_loss = CaptureL1Loss()
    trainer.loss_fn = capture_loss

    trainer.train_one_epoch(0)

    assert autocast_calls == [True, True]
    assert capture_loss.pred_dtype == torch.float32
    assert capture_loss.target_dtype == torch.float32


def test_fastmri_trainer_joint_bf16_path_keeps_checkpointing_enabled(monkeypatch, tmp_path: Path):
    config = _tiny_config("unet_and_lora")
    config.fastmri_finetune.max_epochs = 1
    config.fastmri_finetune.use_bf16 = True

    train_loader = DataLoader(ToyFastMRIDataset(2), batch_size=1, shuffle=False)
    trainer = FastMRIFineTuneTrainer(
        model=_build_wrapped_model("unet_and_lora"),
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "bf16_joint_train",
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
        precision_state={"mode": "bf16", "use_bf16": True},
    )

    autocast_calls = []
    checkpoint_calls = []

    @contextmanager
    def fake_context(*, enabled: bool):
        autocast_calls.append(enabled)
        yield

    def fake_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(kwargs)
        return function(*args)

    monkeypatch.setattr(trainer, "_autocast_context", lambda *, enabled: fake_context(enabled=enabled))
    monkeypatch.setattr(fastmri_compat_module.checkpoint_utils, "checkpoint", fake_checkpoint)

    trainer.train_one_epoch(0)

    assert autocast_calls == [True, True]
    assert checkpoint_calls == [{"use_reentrant": False}, {"use_reentrant": False}]


def test_fastmri_trainer_eval_disables_bf16_and_keeps_metric_arrays_fp32(monkeypatch, tmp_path: Path):
    config = _tiny_config("unet_only")
    config.fastmri_finetune.max_epochs = 1
    config.fastmri_finetune.use_bf16 = True

    val_loader = DataLoader(ToyFastMRIDataset(2), batch_size=1, shuffle=False)
    model = _build_wrapped_model("unet_only")
    trainer = FastMRIFineTuneTrainer(
        model=model,
        config=config,
        device=torch.device("cpu"),
        run_dir=tmp_path / "bf16_eval",
        train_loader=val_loader,
        val_loader=val_loader,
        test_loader=None,
        metric_fns={
            "psnr": lambda gt, pred: float(np.mean(gt - pred)),
            "ssim": lambda gt, pred: float(np.mean(gt + pred)),
            "nmse": lambda gt, pred: float(np.mean(gt * 0 + pred * 0)),
        },
        precision_state={"mode": "bf16", "use_bf16": True},
    )

    autocast_calls = []

    @contextmanager
    def fake_context(*, enabled: bool):
        autocast_calls.append(enabled)
        yield

    monkeypatch.setattr(trainer, "_autocast_context", lambda *, enabled: fake_context(enabled=enabled))
    monkeypatch.setattr(
        trainer_fa_module,
        "complex_output_to_magnitude",
        lambda output: complex_output_to_magnitude(output).to(torch.bfloat16),
    )

    original_metadata_to_cpu_numpy = trainer._metadata_to_cpu_numpy
    captured_prediction_dtypes = []
    captured_target_dtypes = []

    def wrapped_metadata_to_cpu_numpy(magnitude_prediction, magnitude_target, metadata):
        volume_names, slice_indices, predictions, targets = original_metadata_to_cpu_numpy(
            magnitude_prediction,
            magnitude_target,
            metadata,
        )
        captured_prediction_dtypes.extend(pred.dtype for pred in predictions)
        captured_target_dtypes.extend(target.dtype for target in targets)
        return volume_names, slice_indices, predictions, targets

    monkeypatch.setattr(trainer, "_metadata_to_cpu_numpy", wrapped_metadata_to_cpu_numpy)

    metrics = trainer.evaluate_loader(val_loader, split="val")

    assert autocast_calls == [False, False]
    assert set(metrics) == {"loss", "psnr", "ssim", "nmse"}
    assert captured_prediction_dtypes == [np.float32, np.float32]
    assert captured_target_dtypes == [np.float32, np.float32]
