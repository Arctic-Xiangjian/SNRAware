import pytest
import torch

pytest.importorskip("hydra")
from hydra import compose, initialize

from snraware.projects.mri.denoising.lora_utils import apply_lora_to_model
from snraware.projects.mri.denoising.model import DenoisingModel


def get_dummy_config():
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
                "dataset.cutout_shape=[16,16,1]",
                "lora.enabled=True",
                "lora.r=2",
                "lora.lora_alpha=8.0",
                "lora.lora_dropout=0.0",
            ],
        )

    return cfg


def _clean_name_for_snapshot(name: str) -> str:
    return name.replace(".base_layer", "")


def test_lora_integration():
    torch.manual_seed(1234)

    config = get_dummy_config()
    base_model = DenoisingModel(config=config, D=1, H=16, W=16)
    base_weights_snapshot = {
        name: param.detach().clone() for name, param in base_model.named_parameters()
    }

    lora_model = apply_lora_to_model(base_model)

    lora_param_names = [name for name, _ in lora_model.named_parameters() if ".lora_" in name]
    assert lora_param_names, "No LoRA parameters were attached to the model"

    for name, param in lora_model.named_parameters():
        if ".lora_" in name or name.startswith("pre.") or name.startswith("post."):
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"

    dummy_input = torch.randn(2, 3, 1, 16, 16)
    dummy_target = torch.randn(2, 2, 1, 16, 16)

    optimizer = torch.optim.Adam((p for p in lora_model.parameters() if p.requires_grad), lr=1e-3)
    output = lora_model(dummy_input)
    loss = torch.nn.functional.mse_loss(output, dummy_target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    for name, param in lora_model.named_parameters():
        if ".lora_" in name or name.startswith("pre.") or name.startswith("post."):
            continue

        clean_name = _clean_name_for_snapshot(name)
        assert clean_name in base_weights_snapshot, f"Missing {clean_name} in baseline snapshot"
        assert torch.equal(param, base_weights_snapshot[clean_name]), (
            f"Backbone weight changed unexpectedly for {name}"
        )
