import os
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from colorama import Fore, Style
from hydra import compose, initialize
from omegaconf import OmegaConf

from snraware.components.model import HRnet
from snraware.components.setup import end_timer, get_device, set_seed, start_timer

# -----------------------------------------------------------------


class TestHRnet:
    def setup_class(self):
        set_seed(785456)
        torch.set_printoptions(precision=10)

        Current_DIR = Path(__file__).parents[0].resolve()

        self.test_path = str(Current_DIR)
        self.data_root = str(Current_DIR) + "/../data/backbone_hrnet"

        self.test_in = np.load(os.path.join(self.data_root, "test_in.npy"))

        with initialize(version_base=None, config_path="../../src/snraware/components/configs"):
            cfg = compose(config_name="config", overrides=["backbone=hrnet"])
            print(OmegaConf.to_yaml(cfg))

        self.cfg = cfg
        self.config = hydra.utils.instantiate(cfg.backbone)

    def teardown_class(self):
        self.config = None

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "backbone",
        [
            ["S3ShT1L1G1V2V3", "V2V3T1L1G1S3Sh"],
            ["S3ShS3ShS3Sh", "S3ShS3ShS3Sh"],
            ["V3V3V3", "V3V3V3"],
            ["T1L1G1T1", "T1L1G1T1"],
            ["T1L1G1", "T1L1G1"],
            ["T1T1T1", "T1T1T1"],
            ["T1V2V2", "T1V2V2"],
            ["T1L1G1T1L1G1", "T1L1G1T1L1G1T1L1G1"],
            ["C2C2C2", "C2C2C2"],
            ["C3C3C3", "C3C3C3"],
        ],
    )
    def test(self, backbone):
        with_timer = True
        device = get_device()

        _B, C, T, H, W = 1, 2, 16, 16, 16
        test_in = torch.from_numpy(self.test_in).to(dtype=torch.float32, device=device)
        assert np.linalg.norm(self.test_in - test_in.cpu().numpy()) < 1e-3

        self.cfg.backbone.block.cell.window_size = [H // 8, W // 8, T // 8]
        self.cfg.backbone.block.cell.patch_size = [2, 2, 2]

        set_seed(785456)

        print(
            f"{Fore.GREEN}-------------> HRNet -- {backbone} <----------------------{Style.RESET_ALL}"
        )

        self.cfg.backbone.block_str = backbone
        self.config = hydra.utils.instantiate(self.cfg.backbone)

        model = HRnet(config=self.config, input_feature_channels=C, H=H, W=W, D=T)
        model.to(device=device)

        t0 = start_timer(enable=with_timer)
        test_out = model(test_in)[-1]
        end_timer(
            enable=with_timer, t=t0, msg=f"{backbone} - forward pass - test_in {test_in.shape}"
        )

        gt_fname = os.path.join(self.data_root, f"test_out_{backbone}.npy")
        # np.save(gt_fname, test_out.detach().cpu().numpy())
        assert os.path.exists(gt_fname)
        test_out_gt = np.load(gt_fname)
        assert (
            np.linalg.norm(test_out_gt - test_out.detach().cpu().numpy())
            / np.linalg.norm(test_out_gt)
            < 2e-3
        )

        loss = torch.nn.MSELoss()
        t0 = start_timer(enable=with_timer)
        mse = loss(test_in, test_out[:, :C, :, :, :])
        mse.backward()
        end_timer(enable=with_timer, t=t0, msg=f"{backbone} - backward pass")

        del model
        torch.cuda.empty_cache()

        print(f"{'***' * 20}")
