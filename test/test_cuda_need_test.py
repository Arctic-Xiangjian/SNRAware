from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import cuda_need_test as cuda_need_test_module


class _FakeCpuWrappedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(8, dtype=torch.float32))
        self.register_buffer("scale", torch.ones(4, dtype=torch.float32))
        self.forward_grad_enabled = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_grad_enabled = torch.is_grad_enabled()
        return torch.zeros(x.shape[0], 2, 1, x.shape[2], x.shape[3], dtype=x.dtype)


class _FakeLeaf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))


class _FakeCudaWrappedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = _FakeLeaf()
        self.gfactor_unet = _FakeLeaf()
        self.forward_checkpoint_flags: list[bool] = []

    def to(self, *args, **kwargs):
        return self

    def forward(self, x: torch.Tensor, *, checkpoint_base_model: bool = False) -> torch.Tensor:
        self.forward_checkpoint_flags.append(bool(checkpoint_base_model))
        scale = self.base_model.weight + self.gfactor_unet.weight
        return scale * torch.ones(x.shape[0], 2, 1, x.shape[2], x.shape[3], dtype=x.dtype, device=x.device)


class _FakeOptimizer:
    def __init__(self):
        self.zero_grad_calls = 0
        self.step_calls = 0
        self.param_groups = [
            {"name": "gfactor_unet", "lr": 1.0e-4, "weight_decay": 0.0},
            {"name": "adapter", "lr": 1.0e-5, "weight_decay": 0.0},
        ]

    def zero_grad(self, set_to_none: bool = False):
        self.zero_grad_calls += 1

    def step(self):
        self.step_calls += 1



def test_estimator_returns_static_and_measured_fields_and_uses_wrapper(monkeypatch, tmp_path: Path):
    fake_model = _FakeCpuWrappedModel()
    resolved = {
        "config": str(tmp_path / "fake.yaml"),
        "weight": str(tmp_path / "fake.pts"),
    }
    Path(resolved["config"]).write_text("dummy: true\n")
    Path(resolved["weight"]).write_bytes(b"fake-weights")

    calls = {}

    monkeypatch.setattr(
        cuda_need_test_module,
        "resolve_base_model_paths",
        lambda variant, config_path, checkpoint_path, repo_root=None: (
            calls.__setitem__("variant", variant) or resolved["config"],
            resolved["weight"],
        ),
    )

    def fake_build_fastmri_wrapped_model(**kwargs):
        calls["build_kwargs"] = kwargs
        return fake_model, None, {"matched_keys": 1, "mismatched_keys": 0, "total_model_keys": 1, "weight_source": "fake"}

    monkeypatch.setattr(cuda_need_test_module, "build_fastmri_wrapped_model", fake_build_fastmri_wrapped_model)

    report = cuda_need_test_module.estimate_fastmri_wrapped_cpu_infer_memory(
        model_size="small",
        config_path=None,
        weight_path=None,
        crop_size=(64, 64),
        batch_size=1,
    )

    assert calls["variant"] == "small"
    assert calls["build_kwargs"]["height"] == 64
    assert calls["build_kwargs"]["width"] == 64
    assert calls["build_kwargs"]["depth"] == 1
    assert calls["build_kwargs"]["lora_config"] is None
    assert "gfactor_unet_kwargs" in calls["build_kwargs"]
    assert report["mode"] == "fastmri_wrapped_cpu_infer_estimate"
    assert "not a training-time CUDA VRAM number" in report["note"]
    assert report["static"]["parameter_count"] == 8
    assert report["static"]["buffer_count"] == 4
    assert report["static"]["input_bytes"] > 0
    assert report["static"]["output_bytes"] > 0
    assert "peak_delta_from_start_bytes" in report["measured_rss"]
    assert fake_model.forward_grad_enabled is False
    assert fake_model.weight.grad is None



def test_estimator_does_not_call_torch_cuda(monkeypatch, tmp_path: Path):
    fake_model = _FakeCpuWrappedModel()
    resolved = {
        "config": str(tmp_path / "fake.yaml"),
        "weight": str(tmp_path / "fake.pts"),
    }
    Path(resolved["config"]).write_text("dummy: true\n")
    Path(resolved["weight"]).write_bytes(b"fake-weights")

    monkeypatch.setattr(
        cuda_need_test_module,
        "resolve_base_model_paths",
        lambda variant, config_path, checkpoint_path, repo_root=None: (resolved["config"], resolved["weight"]),
    )
    monkeypatch.setattr(
        cuda_need_test_module,
        "build_fastmri_wrapped_model",
        lambda **kwargs: (
            fake_model,
            None,
            {"matched_keys": 1, "mismatched_keys": 0, "total_model_keys": 1, "weight_source": "fake"},
        ),
    )

    def fail(*args, **kwargs):
        raise AssertionError("torch.cuda should not be called by the CPU inference estimator")

    monkeypatch.setattr(torch.cuda, "is_available", fail)
    monkeypatch.setattr(torch.cuda, "is_initialized", fail)
    monkeypatch.setattr(torch.cuda, "memory_allocated", fail)

    report = cuda_need_test_module.estimate_fastmri_wrapped_cpu_infer_memory(
        model_size="small",
        config_path=None,
        weight_path=None,
        crop_size=(64, 64),
        batch_size=1,
    )

    assert report["model_size"] == "small"



def test_profile_fastmri_wrapped_model_combines_cpu_and_cuda_sections(monkeypatch):
    cpu_report = {"mode": "cpu", "note": "cpu", "static": {}, "measured_rss": {}}
    cuda_report = {"mode": "cuda", "cases": {"warmup": {"headline": {}}}}

    monkeypatch.setattr(
        cuda_need_test_module,
        "estimate_fastmri_wrapped_cpu_infer_memory",
        lambda **kwargs: cpu_report,
    )
    monkeypatch.setattr(
        cuda_need_test_module,
        "profile_fastmri_wrapped_cuda_train_peak",
        lambda **kwargs: cuda_report,
    )

    report = cuda_need_test_module.profile_fastmri_wrapped_model(
        profile_target="both",
        model_size="small",
        config_path=None,
        weight_path=None,
        crop_size=(64, 64),
        batch_size=1,
        device="cuda:0",
        train_mode="warmup_then_both",
        use_bf16=True,
        gradient_checkpoint_frozen_base=True,
    )

    assert report["mode"] == "fastmri_wrapped_profiler"
    assert report["cpu_infer_estimate"] is cpu_report
    assert report["cuda_train_peak"] is cuda_report



def test_main_writes_json_report(monkeypatch, tmp_path: Path):
    report_path = tmp_path / "report.json"
    report_payload = {
        "mode": "fastmri_wrapped_profiler",
        "profile_target": "cpu_infer_estimate",
        "cpu_infer_estimate": {
            "mode": "fastmri_wrapped_cpu_infer_estimate",
            "note": "cpu only",
            "model_size": "small",
            "resolved_config_path": "/tmp/fake.yaml",
            "resolved_weight_path": "/tmp/fake.pts",
            "crop_size": [64, 64],
            "batch_size": 1,
            "checkpoint_file_bytes": 1,
            "checkpoint_file_gb": 0.0,
            "load_info": {},
            "static": {
                "parameter_count": 1,
                "parameter_bytes": 4,
                "buffer_count": 0,
                "buffer_bytes": 0,
                "input_bytes": 1,
                "output_bytes": 1,
                "model_load_estimate_bytes": 4,
                "forward_resident_estimate_bytes": 6,
                "parameter_gb": 0.0,
                "buffer_gb": 0.0,
                "input_gb": 0.0,
                "output_gb": 0.0,
                "model_load_estimate_gb": 0.0,
                "forward_resident_estimate_gb": 0.0,
            },
            "measured_rss": {
                "before_build_bytes": 0,
                "after_build_bytes": 0,
                "after_input_bytes": 0,
                "after_forward_bytes": 0,
                "peak_before_build_bytes": 0,
                "peak_after_build_bytes": 0,
                "peak_after_input_bytes": 0,
                "peak_after_forward_bytes": 0,
                "build_delta_bytes": 0,
                "input_delta_bytes": 0,
                "forward_delta_bytes": 0,
                "peak_delta_from_start_bytes": 0,
                "build_delta_gb": 0.0,
                "input_delta_gb": 0.0,
                "forward_delta_gb": 0.0,
                "peak_delta_from_start_gb": 0.0,
            },
        },
    }

    monkeypatch.setattr(cuda_need_test_module, "profile_fastmri_wrapped_model", lambda **kwargs: report_payload)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_need_test.py",
            "--profile_target",
            "cpu_infer_estimate",
            "--model_size",
            "small",
            "--crop_size",
            "64",
            "64",
            "--batch_size",
            "1",
            "--report_json",
            str(report_path),
        ],
    )

    cuda_need_test_module.main()

    saved = json.loads(report_path.read_text())
    assert saved["mode"] == "fastmri_wrapped_profiler"
    assert saved["cpu_infer_estimate"]["mode"] == "fastmri_wrapped_cpu_infer_estimate"



def test_cuda_profiler_warmup_then_both_records_phases_and_toggles_checkpoint(monkeypatch, tmp_path: Path):
    resolved_config = tmp_path / "fake.yaml"
    resolved_weight = tmp_path / "fake.pts"
    resolved_config.write_text("dummy: true\n")
    resolved_weight.write_bytes(b"fake")

    fake_config = OmegaConf.create(
        {
            "fastmri_finetune": {
                "use_bf16": True,
                "gradient_checkpoint_frozen_base": True,
                "unet_lr": 1.0e-4,
                "adapter_lr": 1.0e-5,
                "weight_decay": 0.0,
                "gfactor_unet": {
                    "in_chans": 2,
                    "out_chans": 1,
                    "chans": 8,
                    "num_pool_layers": 2,
                    "drop_prob": 0.0,
                },
            },
            "lora": {"enabled": True},
        }
    )

    autocast_calls = []
    optimizer_calls = []
    checkpoint_queries = []
    built_models = []
    snapshot_values = iter(
        [
            {"allocated_bytes": 10, "reserved_bytes": 20, "peak_allocated_bytes": 10, "peak_reserved_bytes": 20},
            {"allocated_bytes": 30, "reserved_bytes": 40, "peak_allocated_bytes": 30, "peak_reserved_bytes": 40},
            {"allocated_bytes": 50, "reserved_bytes": 60, "peak_allocated_bytes": 70, "peak_reserved_bytes": 80},
            {"allocated_bytes": 55, "reserved_bytes": 65, "peak_allocated_bytes": 75, "peak_reserved_bytes": 85},
            {"allocated_bytes": 90, "reserved_bytes": 100, "peak_allocated_bytes": 110, "peak_reserved_bytes": 120},
            {"allocated_bytes": 95, "reserved_bytes": 105, "peak_allocated_bytes": 115, "peak_reserved_bytes": 120},
            {"allocated_bytes": 12, "reserved_bytes": 22, "peak_allocated_bytes": 12, "peak_reserved_bytes": 22},
            {"allocated_bytes": 32, "reserved_bytes": 42, "peak_allocated_bytes": 32, "peak_reserved_bytes": 42},
            {"allocated_bytes": 52, "reserved_bytes": 62, "peak_allocated_bytes": 72, "peak_reserved_bytes": 82},
            {"allocated_bytes": 57, "reserved_bytes": 67, "peak_allocated_bytes": 77, "peak_reserved_bytes": 87},
            {"allocated_bytes": 92, "reserved_bytes": 102, "peak_allocated_bytes": 112, "peak_reserved_bytes": 122},
            {"allocated_bytes": 94, "reserved_bytes": 104, "peak_allocated_bytes": 114, "peak_reserved_bytes": 122},
        ]
    )

    def with_gb(snapshot: dict[str, int]) -> dict[str, float]:
        return snapshot | {
            "allocated_gb": cuda_need_test_module._bytes_to_gb(snapshot["allocated_bytes"]),
            "reserved_gb": cuda_need_test_module._bytes_to_gb(snapshot["reserved_bytes"]),
            "peak_allocated_gb": cuda_need_test_module._bytes_to_gb(snapshot["peak_allocated_bytes"]),
            "peak_reserved_gb": cuda_need_test_module._bytes_to_gb(snapshot["peak_reserved_bytes"]),
        }

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)

    monkeypatch.setattr(
        cuda_need_test_module,
        "resolve_base_model_paths",
        lambda variant, config_path, checkpoint_path, repo_root=None: (str(resolved_config), str(resolved_weight)),
    )
    monkeypatch.setattr(cuda_need_test_module, "_load_fastmri_runtime_config", lambda repo_root=None: fake_config)

    def fake_build_fastmri_wrapped_model(**kwargs):
        model = _FakeCudaWrappedModel()
        built_models.append(model)
        return model, None, {"matched_keys": 1, "mismatched_keys": 0, "total_model_keys": 1, "weight_source": "fake"}

    monkeypatch.setattr(cuda_need_test_module, "build_fastmri_wrapped_model", fake_build_fastmri_wrapped_model)

    def fake_configure_model_for_finetune_mode(model, *, mode, lora_config=None, adapters_active=None):
        if mode == "warmup_then_both":
            active = bool(adapters_active)
            model.base_model.weight.requires_grad = active
            model.gfactor_unet.weight.requires_grad = True
            model._adapters_active = active
            return {"mode": mode, "adapters_active": active, "has_lora": True}
        raise AssertionError("Unexpected mode for this test")

    monkeypatch.setattr(cuda_need_test_module, "configure_model_for_finetune_mode", fake_configure_model_for_finetune_mode)

    def fake_build_fastmri_optimizer(model, ft_cfg, mode_state):
        optimizer_calls.append(dict(mode_state))
        return _FakeOptimizer()

    monkeypatch.setattr(cuda_need_test_module, "build_fastmri_optimizer", fake_build_fastmri_optimizer)
    monkeypatch.setattr(
        cuda_need_test_module,
        "resolve_fastmri_precision",
        lambda device, *, use_bf16: {"mode": "bf16" if use_bf16 else "fp32", "use_bf16": bool(use_bf16)},
    )

    @contextmanager
    def fake_autocast(*, enabled: bool):
        autocast_calls.append(enabled)
        yield

    monkeypatch.setattr(
        cuda_need_test_module,
        "fastmri_autocast_context",
        lambda device, *, enabled: fake_autocast(enabled=enabled),
    )
    monkeypatch.setattr(
        cuda_need_test_module,
        "should_checkpoint_frozen_base",
        lambda model, *, gradient_checkpoint_frozen_base: checkpoint_queries.append(bool(model.base_model.weight.requires_grad))
        or gradient_checkpoint_frozen_base,
    )
    monkeypatch.setattr(cuda_need_test_module, "_cuda_memory_snapshot", lambda device: with_gb(next(snapshot_values)))

    real_randn = torch.randn

    def fake_randn(*args, **kwargs):
        kwargs.pop("device", None)
        return real_randn(*args, **kwargs)

    monkeypatch.setattr(cuda_need_test_module.torch, "randn", fake_randn)

    report = cuda_need_test_module.profile_fastmri_wrapped_cuda_train_peak(
        model_size="small",
        config_path=None,
        weight_path=None,
        crop_size=(32, 32),
        batch_size=1,
        device="cuda:0",
        train_mode="warmup_then_both",
        use_bf16=True,
        gradient_checkpoint_frozen_base=True,
    )

    assert report["mode"] == "fastmri_wrapped_cuda_train_peak"
    assert set(report["cases"]) == {"warmup", "after_warmup"}
    assert report["cases"]["warmup"]["checkpoint_base_model"] is True
    assert report["cases"]["after_warmup"]["checkpoint_base_model"] is True
    assert report["cases"]["warmup"]["headline"]["peak_reserved_bytes"] == 120
    assert report["cases"]["after_warmup"]["headline"]["peak_reserved_bytes"] == 122
    assert list(report["cases"]["warmup"]["snapshots"]) == [
        "after_model_load",
        "after_input_transfer",
        "after_forward",
        "after_loss",
        "after_backward",
        "after_optimizer_step",
    ]
    assert autocast_calls == [True, True]
    assert checkpoint_queries == [False, True]
    assert optimizer_calls == [
        {"mode": "warmup_then_both", "adapters_active": False, "has_lora": True},
        {"mode": "warmup_then_both", "adapters_active": True, "has_lora": True},
    ]
    assert built_models[0].forward_checkpoint_flags == [True]
    assert built_models[1].forward_checkpoint_flags == [True]



def test_cuda_profiler_unet_only_uses_checkpoint_when_base_is_frozen(monkeypatch, tmp_path: Path):
    resolved_config = tmp_path / "fake.yaml"
    resolved_weight = tmp_path / "fake.pts"
    resolved_config.write_text("dummy: true\n")
    resolved_weight.write_bytes(b"fake")

    fake_config = OmegaConf.create(
        {
            "fastmri_finetune": {
                "use_bf16": False,
                "gradient_checkpoint_frozen_base": True,
                "unet_lr": 1.0e-4,
                "adapter_lr": 1.0e-5,
                "weight_decay": 0.0,
                "gfactor_unet": {
                    "in_chans": 2,
                    "out_chans": 1,
                    "chans": 8,
                    "num_pool_layers": 2,
                    "drop_prob": 0.0,
                },
            },
            "lora": {"enabled": True},
        }
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(
        cuda_need_test_module,
        "resolve_base_model_paths",
        lambda variant, config_path, checkpoint_path, repo_root=None: (str(resolved_config), str(resolved_weight)),
    )
    monkeypatch.setattr(cuda_need_test_module, "_load_fastmri_runtime_config", lambda repo_root=None: fake_config)
    monkeypatch.setattr(
        cuda_need_test_module,
        "build_fastmri_wrapped_model",
        lambda **kwargs: (
            _FakeCudaWrappedModel(),
            None,
            {"matched_keys": 1, "mismatched_keys": 0, "total_model_keys": 1, "weight_source": "fake"},
        ),
    )
    monkeypatch.setattr(
        cuda_need_test_module,
        "configure_model_for_finetune_mode",
        lambda model, *, mode, lora_config=None, adapters_active=None: (
            setattr(model.base_model.weight, "requires_grad", False)
            or setattr(model.gfactor_unet.weight, "requires_grad", True)
            or {"mode": mode, "adapters_active": False, "has_lora": False}
        ),
    )
    monkeypatch.setattr(cuda_need_test_module, "build_fastmri_optimizer", lambda model, ft_cfg, mode_state: _FakeOptimizer())
    monkeypatch.setattr(
        cuda_need_test_module,
        "resolve_fastmri_precision",
        lambda device, *, use_bf16: {"mode": "fp32", "use_bf16": False},
    )

    @contextmanager
    def fake_autocast(*, enabled: bool):
        yield

    monkeypatch.setattr(
        cuda_need_test_module,
        "fastmri_autocast_context",
        lambda device, *, enabled: fake_autocast(enabled=enabled),
    )
    monkeypatch.setattr(
        cuda_need_test_module,
        "should_checkpoint_frozen_base",
        lambda model, *, gradient_checkpoint_frozen_base: gradient_checkpoint_frozen_base,
    )
    monkeypatch.setattr(
        cuda_need_test_module,
        "_cuda_memory_snapshot",
        lambda device: {
            "allocated_bytes": 1,
            "reserved_bytes": 2,
            "peak_allocated_bytes": 3,
            "peak_reserved_bytes": 4,
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "peak_allocated_gb": 0.0,
            "peak_reserved_gb": 0.0,
        },
    )

    real_randn = torch.randn
    monkeypatch.setattr(
        cuda_need_test_module.torch,
        "randn",
        lambda *args, **kwargs: real_randn(*args, **{k: v for k, v in kwargs.items() if k != "device"}),
    )

    report = cuda_need_test_module.profile_fastmri_wrapped_cuda_train_peak(
        model_size="small",
        config_path=None,
        weight_path=None,
        crop_size=(16, 16),
        batch_size=1,
        device="cuda:0",
        train_mode="unet_only",
        use_bf16=False,
        gradient_checkpoint_frozen_base=True,
    )

    assert list(report["cases"]) == ["unet_only"]
    assert report["cases"]["unet_only"]["checkpoint_base_model"] is True
    assert report["cases"]["unet_only"]["precision_mode"] == "fp32"



def test_cuda_profiler_rejects_non_cuda_device():
    with pytest.raises(ValueError, match="requires a CUDA device"):
        cuda_need_test_module.profile_fastmri_wrapped_cuda_train_peak(
            model_size="small",
            config_path=None,
            weight_path=None,
            crop_size=(64, 64),
            batch_size=1,
            device="cpu",
            train_mode="unet_only",
            use_bf16=False,
            gradient_checkpoint_frozen_base=True,
        )



def test_estimator_cpu_smoke_small_model():
    report = cuda_need_test_module.estimate_fastmri_wrapped_cpu_infer_memory(
        model_size="small",
        config_path=None,
        weight_path=None,
        crop_size=(64, 64),
        batch_size=1,
    )

    assert report["model_size"] == "small"
    assert report["static"]["parameter_count"] > 0
    assert report["static"]["model_load_estimate_bytes"] > 0
    assert report["measured_rss"]["peak_delta_from_start_bytes"] >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_profiler_smoke_small_model():
    report = cuda_need_test_module.profile_fastmri_wrapped_cuda_train_peak(
        model_size="small",
        config_path=None,
        weight_path=None,
        crop_size=(64, 64),
        batch_size=1,
        device="cuda:0",
        train_mode="unet_only",
        use_bf16=False,
        gradient_checkpoint_frozen_base=True,
    )

    case = report["cases"]["unet_only"]
    assert case["headline"]["peak_allocated_bytes"] > 0
    assert case["headline"]["peak_reserved_bytes"] > 0
