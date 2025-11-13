import os
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from colorama import Fore, Style
from hydra import compose, initialize
from omegaconf import OmegaConf

from snraware.components.model import Unet
from snraware.components.setup import end_timer, get_device, set_seed, start_timer

# -----------------------------------------------------------------


class TestUnet:
    def setup_class(self):
        set_seed(7878756)
        torch.set_printoptions(precision=10)

        Current_DIR = Path(__file__).parents[0].resolve()

        self.test_path = str(Current_DIR)
        self.data_root = str(Current_DIR) + "/../data/backbone_unet"

        self.test_in_fname = os.path.join(self.data_root, "test_in.npy")
        self.test_in = np.load(self.test_in_fname)

        with initialize(version_base=None, config_path="../../src/snraware/components/configs"):
            cfg = compose(config_name="config", overrides=["backbone=unet"])
            print(OmegaConf.to_yaml(cfg))

        self.cfg = cfg
        self.config = hydra.utils.instantiate(cfg.backbone)

    def teardown_class(self):
        pass

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "backbone",
        [
            ["S3ShT1L1G1V2V3", "V2V3T1L1G1S3Sh"],
            ["V3V3V3", "V3V3V3"],
            ["T1L1G1T1", "T1L1G1T1"],
            ["T1L1G1", "T1L1G1"],
            ["T1T1T1", "T1T1T1"],
            ["T1V2V2", "T1V2V2"],
            ["C2C2C2", "C2C2C2"],
            ["C3C3C3", "C3C3C3"],
        ],
    )
    def test(self, backbone):
        with_timer = True
        device = get_device()

        _B, C, T, H, W = 1, 2, 16, 32, 32
        test_in = torch.from_numpy(self.test_in).to(dtype=torch.float32, device=device)

        self.cfg.backbone.block.cell.window_size = [H // 4, W // 4, T // 4]
        self.cfg.backbone.block.cell.patch_size = [2, 2, 2]

        # run the tests
        bk = backbone

        set_seed(7878756)

        print(f"{Fore.GREEN}-------------> UNet -- {bk} <----------------------{Style.RESET_ALL}")

        self.cfg.backbone.block_str = bk
        self.cfg.backbone.num_resolution_levels = len(bk)
        self.config = hydra.utils.instantiate(self.cfg.backbone)

        model = Unet(config=self.config, input_feature_channels=C, H=H, W=W, D=T)
        model.to(device=device)

        t0 = start_timer(enable=with_timer)
        test_out = model(test_in)[0]
        end_timer(enable=with_timer, t=t0, msg=f"{bk} - forward pass - test_in {test_in.shape}")

        loss = torch.nn.MSELoss()
        t0 = start_timer(enable=with_timer)
        mse = loss(test_in, test_out[:, :C, :, :, :])
        mse.backward()
        end_timer(enable=with_timer, t=t0, msg=f"{bk} - backward pass")

        fname = f"{self.cfg.backbone.block_str[0]}_{self.cfg.backbone.num_of_channels}"
        gt_fname = os.path.join(self.data_root, f"test_out_{fname}.npy")
        # np.save(gt_fname, test_out.detach().cpu().numpy())
        assert os.path.exists(gt_fname)
        test_out_gt = np.load(gt_fname)
        assert (
            np.linalg.norm(test_out_gt - test_out.detach().cpu().numpy())
            / np.linalg.norm(test_out_gt)
            < 2e-3
        )

        del model
        torch.cuda.empty_cache()
