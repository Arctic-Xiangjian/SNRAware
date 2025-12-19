import os
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from colorama import Fore, Style
from hydra import compose, initialize
from omegaconf import OmegaConf

from snraware.components.model import Block
from snraware.components.model.config import create_block_config
from snraware.components.setup import get_device, set_seed

# -----------------------------------------------------------------


class TestBlock:
    def setup_class(self):
        set_seed(4587962)
        torch.set_printoptions(precision=10)

        Current_DIR = Path(__file__).parents[0].resolve()

        self.test_path = str(Current_DIR)
        self.data_root = str(Current_DIR) + "/../data/block"

        with initialize(version_base=None, config_path="../../src/snraware/components/configs"):
            cfg = compose(config_name="config")
            print(OmegaConf.to_yaml(cfg))

        self.cfg = hydra.utils.instantiate(cfg.backbone)

    def teardown_class(self):
        pass

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "block_str",
        [
            "V3",
            "C2",
            "C3",
            "T1",
            "T0",
            "L1",
            "L0",
            "G1",
            "G0",
            "V2",
            "V3",
            "S3",
            "Sh",
            "L1G1T1",
            "T1L1G1T1",
            "T1T1T1",
            "V2V2V2V2V2",
            "V3V3V3V3V3",
            "S3ShS3ShS3ShS3ShS3Sh",
            "T1L1G1V2V3",
            "S3ShT1L1G1",
            "V2V3S3ShS3ShT1T1T1",
            "T1V2S3ShL1V2S3ShG1V2S3Sh",
        ],
    )
    def test(self, block_str):
        set_seed(7878756)

        B, T, C, H, W = 1, 16, 3, 64, 64
        C_out = 8
        test_in = torch.rand(B, T, C, H, W)

        device = get_device()
        if device != "cuda":
            pytest.skip("GPU only test")

        test_in = test_in.to(device=device, dtype=torch.float32)
        test_in = torch.permute(test_in, [0, 2, 1, 3, 4])

        self.cfg.block.cell.n_head = 8
        self.cfg.block.cell.cosine_att = True
        self.cfg.block.cell.norm_mode = "layer"
        self.cfg.block.block_dense_connection = True

        with torch.inference_mode():
            fname = f"{block_str}"
            print(f"{Fore.YELLOW}--> {fname}{Style.RESET_ALL}")

            block_config = create_block_config(block_str=[block_str], block=self.cfg.block)
            a_block = Block(
                block_config=block_config[0],
                C_in=C,
                C_out=C_out,
                H=H,
                W=W,
                D=T,
                config=self.cfg.block,
            )
            a_block.to(device=device)
            test_out = a_block(torch.clone(test_in))

            gt_fname = os.path.join(self.data_root, f"test_out_{fname}.npy")
            # np.save(gt_fname, test_out.detach().cpu().numpy())
            assert os.path.exists(gt_fname)
            test_out_gt = np.load(gt_fname)
            assert (
                np.linalg.norm(test_out_gt - test_out.detach().cpu().numpy())
                / np.linalg.norm(test_out_gt)
                < 2e-3
            )
