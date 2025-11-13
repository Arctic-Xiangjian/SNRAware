# test the parameter configuration with hydra
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf


def test_config_with_overrides():
    with initialize(version_base=None, config_path="../../src/snraware/components/configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "backbone=hrnet",
                "backbone.block.cell.attention_type=conv",
                "backbone.num_of_channels=15",
                "backbone.block.cell_type=parallel",
                "backbone.block.cell.att_dropout_p=0.7",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
        assert cfg.backbone.num_of_channels == 15
        assert cfg.backbone.block.cell_type == "parallel"
        assert cfg.backbone.block.cell.att_dropout_p == 0.7

    config = hydra.utils.instantiate(cfg.backbone)
    assert config.num_of_channels == 15
    assert config.block.cell_type == "parallel"
    assert config.block.cell.att_dropout_p == 0.7
