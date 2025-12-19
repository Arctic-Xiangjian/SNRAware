import os
from pathlib import Path

import numpy as np
import pytest
import torch

from snraware.components.model import Cell, Parallel_Cell
from snraware.components.setup import get_device, set_seed

# -----------------------------------------------------------------


class TestCell:
    def setup_class(self):
        set_seed(1854417)
        torch.set_printoptions(precision=10)

        Current_DIR = Path(__file__).parents[0].resolve()

        self.test_path = str(Current_DIR)
        self.data_root = str(Current_DIR) + "/../data/cell"

    def teardown_class(self):
        pass

    @pytest.mark.gpu
    def test(self):
        B, T, C, H, W = 1, 16, 2, 64, 64
        C_out = 8
        test_in = torch.rand(B, T, C, H, W).to(torch.float32)

        device = get_device()
        if device != "cuda":
            pytest.skip("GPU only test")

        test_in = test_in.to(device=device)

        att_types = [
            "vit_3d",
            "local",
            "global",
            "vit_2d",
            "swin_3d",
            "swin_3d_shifted",
            "temporal",
            "conv2d",
            "conv3d",
        ]
        norm_types = ["instance2d", "batch2d", "layer", "instance3d", "batch3d"]

        att_types = ["local", "global", "temporal", "conv2d", "conv3d"]
        norm_types = ["instance2d", "batch2d", "layer"]
        cosine_atts = ["True", "False"]
        att_with_relative_position_biases = ["True", "False"]
        mixer_types = ["conv"]
        with_flash_attentions = [False]
        stride_ss = [[1, 1, 1]]

        test_in = torch.permute(test_in, [0, 2, 1, 3, 4])

        with torch.inference_mode():
            for with_flash_attention in with_flash_attentions:
                for norm_type in norm_types:
                    for cosine_att in cosine_atts:
                        for att_with_relative_position_bias in att_with_relative_position_biases:
                            for mixer_type in mixer_types:
                                for stride_s in stride_ss:
                                    for att_type in att_types:
                                        fname = f"{norm_type}_{att_type}_{mixer_type}_{cosine_att}_{att_with_relative_position_bias}_{stride_s}_{with_flash_attention}_False"
                                        print(fname)

                                        x = torch.clone(test_in)

                                        CNNT_Cell = Cell(
                                            C_in=C,
                                            C_out=C_out,
                                            H=H,
                                            W=W,
                                            D=T,
                                            window_size=[H // 8, W // 8, T // 4],
                                            patch_size=[4, 4, 2],
                                            num_wind=[8, 8, 4],
                                            num_patch=[H // 32, W // 32, T // 8],
                                            att_mode=att_type,
                                            mixer_type=mixer_type,
                                            norm_mode=norm_type,
                                            stride_s=stride_s,
                                            cosine_att=cosine_att,
                                            att_with_relative_position_bias=att_with_relative_position_bias,
                                            use_flash_attention=with_flash_attention,
                                        ).to(device=device)
                                        test_out = CNNT_Cell(x)

                                        gt_fname = os.path.join(
                                            self.data_root, f"test_out_STCNNT_Cell_{fname}.npy"
                                        )
                                        # np.save(gt_fname, test_out.detach().cpu().numpy())
                                        assert os.path.exists(gt_fname)
                                        test_out_gt = np.load(
                                            os.path.join(
                                                self.data_root, f"test_out_STCNNT_Cell_{fname}.npy"
                                            )
                                        )

                                        assert (
                                            np.linalg.norm(
                                                test_out_gt - test_out.detach().cpu().numpy()
                                            )
                                            / np.linalg.norm(test_out_gt)
                                            < 1e-3
                                        )

        for with_flash_attention in with_flash_attentions:
            for att_type in att_types:
                for norm_type in norm_types:
                    for cosine_att in cosine_atts:
                        for att_with_relative_position_bias in att_with_relative_position_biases:
                            for mixer_type in mixer_types:
                                for stride_s in stride_ss:
                                    fname = f"{norm_type}_{att_type}_{mixer_type}_{cosine_att}_{att_with_relative_position_bias}_{stride_s}_{with_flash_attention}_False"
                                    print(fname)

                                    p_cell = Parallel_Cell(
                                        C_in=C,
                                        C_out=C_out,
                                        H=H,
                                        W=W,
                                        D=T,
                                        window_size=[H // 8, W // 8, T // 4],
                                        patch_size=[4, 4, 2],
                                        num_wind=[8, 8, 4],
                                        num_patch=[H // 32, W // 32, T // 8],
                                        att_mode=att_type,
                                        mixer_type=mixer_type,
                                        norm_mode=norm_type,
                                        stride_s=stride_s,
                                        cosine_att=cosine_att,
                                        att_with_relative_position_bias=att_with_relative_position_bias,
                                        use_flash_attention=with_flash_attention,
                                    ).to(device=device)
                                    test_out = p_cell(test_in)

                                    gt_fname = os.path.join(
                                        self.data_root,
                                        f"test_out_STCNNT_Parallel_Cell_{fname}.npy",
                                    )
                                    # np.save(gt_fname, test_out.detach().cpu().numpy())
                                    assert os.path.exists(gt_fname)
                                    test_out_gt = np.load(
                                        os.path.join(
                                            self.data_root,
                                            f"test_out_STCNNT_Parallel_Cell_{fname}.npy",
                                        )
                                    )

                                    assert (
                                        np.linalg.norm(
                                            test_out_gt - test_out.detach().cpu().numpy()
                                        )
                                        / np.linalg.norm(test_out_gt)
                                        < 1e-3
                                    )
