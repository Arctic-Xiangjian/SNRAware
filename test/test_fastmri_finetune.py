from __future__ import annotations

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

from fastmri_data.work_with_snraware import FastMRISNRAwareDataset
from snraware.projects.mri.denoising.fastmri_compat import (
    SNRAwareWithGFactor,
    is_fastmri_finetune_checkpoint,
    load_fastmri_finetune_checkpoint,
    save_fastmri_finetune_checkpoint,
)
from snraware.projects.mri.denoising.lora_utils import apply_lora_to_model
from snraware.projects.mri.denoising.model import DenoisingModel
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
            "weight_decay": 0.0,
            "scheduler_t_max": 0,
            "resume_from": None,
            "max_epochs": 1,
            "log_every_n_steps": 100,
            "evaluate_every_n_epochs": 1,
            "use_bf16": False,
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


def _build_wrapped_model(mode: str) -> SNRAwareWithGFactor:
    config = _tiny_config(mode)
    base_model = DenoisingModel(config=config, D=1, H=32, W=32)
    return SNRAwareWithGFactor(base_model=base_model)


def _write_fake_fastmri_volume(path: Path) -> None:
    size = 320
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


def _clean_lora_wrapped_name(name: str) -> str:
    return name.replace(".base_layer", "")


class CaptureL1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pred_dtype = None
        self.target_dtype = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.pred_dtype = pred.dtype
        self.target_dtype = target.dtype
        return torch.nn.functional.l1_loss(pred, target)


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
    assert torch.all(base_model.last_input[:, 2] >= 0)


def test_wrapper_bypasses_gfactor_for_native_3ch_input():
    base_model = DummyBaseModel()
    gfactor = DummyGFactor()
    model = SNRAwareWithGFactor(base_model=base_model, gfactor_unet=gfactor)

    noisy = torch.randn(2, 3, 1, 8, 8)
    output = model(noisy)

    assert gfactor.calls == 0
    assert output.shape == (2, 2, 1, 8, 8)
    assert torch.equal(base_model.last_input, noisy)


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
    ("mode", "expect_unet_trainable", "expect_adapter_trainable"),
    [
        ("unet_only", True, False),
        ("unet_and_lora", True, True),
        ("lora_only", False, True),
        ("warmup_then_both", True, False),
    ],
)
def test_configure_model_for_modes(mode: str, expect_unet_trainable: bool, expect_adapter_trainable: bool):
    model = _build_wrapped_model(mode)
    state = configure_model_for_finetune_mode(model, mode=mode, lora_config=model.base_model.config.lora)

    unet_trainable = any(parameter.requires_grad for parameter in model.gfactor_unet.parameters())
    adapter_trainable = any(
        parameter.requires_grad
        for name, parameter in model.base_model.named_parameters()
        if ".lora_" in name or name.startswith("pre.") or name.startswith("post.")
    )
    frozen_backbone = all(
        not parameter.requires_grad
        for name, parameter in model.base_model.named_parameters()
        if ".lora_" not in name and not name.startswith("pre.") and not name.startswith("post.")
    )

    assert unet_trainable is expect_unet_trainable
    assert adapter_trainable is expect_adapter_trainable
    assert frozen_backbone
    assert state["mode"] == mode


def test_fastmri_checkpoint_round_trip_restores_gfactor_and_lora(tmp_path: Path):
    torch.manual_seed(123)
    model = _build_wrapped_model("unet_and_lora")
    configure_model_for_finetune_mode(
        model,
        mode="unet_and_lora",
        lora_config=model.base_model.config.lora,
    )

    with torch.no_grad():
        first_unet_param = next(model.gfactor_unet.parameters())
        first_unet_param.add_(0.5)
        first_adapter_param = next(
            parameter
            for name, parameter in model.base_model.named_parameters()
            if ".lora_" in name or name.startswith("pre.") or name.startswith("post.")
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

    fresh_model = _build_wrapped_model("unet_and_lora")
    backbone_snapshot = {
        _clean_lora_wrapped_name(name): parameter.detach().clone()
        for name, parameter in fresh_model.base_model.named_parameters()
        if ".lora_" not in name and not name.startswith("pre.") and not name.startswith("post.")
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
    for name, parameter in fresh_model.base_model.named_parameters():
        if ".lora_" in name or name.startswith("pre.") or name.startswith("post."):
            assert torch.equal(parameter, model.base_model.state_dict()[name])
        else:
            assert torch.equal(parameter, backbone_snapshot[_clean_lora_wrapped_name(name)])


def test_fastmri_bridge_dataset_returns_expected_shapes_and_legacy_normalization(tmp_path: Path):
    data_dir = tmp_path / "singlecoil_train"
    data_dir.mkdir()
    _write_fake_fastmri_volume(data_dir / "case001.h5")

    dataset = FastMRISNRAwareDataset(
        root=data_dir,
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

    noisy_mag = torch.sqrt(noisy[0:1].square() + noisy[1:2].square())
    assert torch.allclose(noisy_mag.mean(), torch.tensor(1.0), atol=1e-4)
    assert metadata["mean"].item() == 0.0

    center_fraction = 0.08
    nsamp_center = int(round(320 * center_fraction))
    center_from = 320 // 2 - nsamp_center // 2
    center_to = center_from + nsamp_center
    assert torch.all(metadata["mask"][:, center_from:center_to, :] == 1)


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

    build_fastmri_dataloaders(config)

    assert len(created) == 3
    assert created[0]["sample_rate"] == 0.02
    assert created[0]["volume_sample_rate"] == 0.5
    assert created[1]["sample_rate"] is None
    assert created[1]["volume_sample_rate"] is None
    assert created[2]["sample_rate"] is None
    assert created[2]["volume_sample_rate"] is None


def test_run_fastmri_launcher_exposes_use_bf16():
    script = (Path(__file__).resolve().parents[1] / "run_fast_mri_single_coil.sh").read_text()

    assert 'USE_BF16="${USE_BF16:-true}"' in script
    assert '"fastmri_finetune.use_bf16=${USE_BF16}"' in script
    assert "USE_BF16=false ./run_fast_mri_single_coil.sh" in script


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
