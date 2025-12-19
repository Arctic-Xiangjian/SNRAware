import os
from pathlib import Path

import numpy as np
import pytest
import torch

from snraware.components.model.attention import SpatialGlobalAttention
from snraware.components.setup import end_timer, get_device, set_seed, start_timer

# -----------------------------------------------------------------


class TestSpatialGlobalAttention:
    def setup_class(self):
        set_seed(358)
        torch.set_printoptions(precision=10)

        Current_DIR = Path(__file__).parents[0].resolve()

        self.test_path = str(Current_DIR)
        self.data_root = str(Current_DIR) + "/../data/spatial_global"

        self.test_in = np.load(os.path.join(self.data_root, "test_in.npy"))
        self.test_in2 = np.load(os.path.join(self.data_root, "test_in2.npy"))

    def teardown_class(self):
        pass

    @pytest.mark.gpu
    def test(self):
        device = get_device()
        if device != "cuda":
            pytest.skip("GPU only test")

        t = np.arange(256)
        t = np.reshape(t, (16, 16))

        t = torch.from_numpy(t).to(dtype=torch.float32)
        t = torch.cat((t[None, :], t[None, :]), dim=0)

        B, T, C, H, W = 2, 4, 2, 16, 16
        C_out = 8
        test_in = t.repeat(B, T, 1, 1, 1)
        print(test_in.shape)

        spacial_vit = SpatialGlobalAttention(
            H=H,
            W=W,
            window_size=[8, 8],
            patch_size=[4, 4],
            stride_qk=(2, 2),
            num_wind=None,
            num_patch=None,
            attention_type="conv",
            C_in=C,
            C_out=C_out,
        )

        a = spacial_vit.im2grid(test_in)
        b = spacial_vit.grid2im(a)

        # gt = torch.tensor(
        #     [
        #         [
        #             [
        #                 [64.0, 65.0, 66.0, 67.0],
        #                 [80.0, 81.0, 82.0, 83.0],
        #                 [96.0, 97.0, 98.0, 99.0],
        #                 [112.0, 113.0, 114.0, 115.0],
        #             ],
        #             [
        #                 [72.0, 73.0, 74.0, 75.0],
        #                 [88.0, 89.0, 90.0, 91.0],
        #                 [104.0, 105.0, 106.0, 107.0],
        #                 [120.0, 121.0, 122.0, 123.0],
        #             ],
        #         ],
        #         [
        #             [
        #                 [192.0, 193.0, 194.0, 195.0],
        #                 [208.0, 209.0, 210.0, 211.0],
        #                 [224.0, 225.0, 226.0, 227.0],
        #                 [240.0, 241.0, 242.0, 243.0],
        #             ],
        #             [
        #                 [200.0, 201.0, 202.0, 203.0],
        #                 [216.0, 217.0, 218.0, 219.0],
        #                 [232.0, 233.0, 234.0, 235.0],
        #                 [248.0, 249.0, 250.0, 251.0],
        #             ],
        #         ],
        #     ]
        # )

        # assert torch.norm(a[0, 0, 1, 0, :, :, :, :, 0] - gt) < 1e-3
        assert torch.norm(b - test_in) < 1e-3

        attention_types = ["conv", "lin"]
        normalize_Q_Ks = [True, False]
        cosine_atts = [True, False]
        att_with_relative_position_biases = [True, False]
        att_with_output_projs = [True, False]
        stride_qks = [[1, 1]]

        with_timer = True

        B, C, T, H1, W1 = 1, 2, 16, 64, 64
        C_out = 4
        test_in = torch.rand(B, C, T, H1, W1).to(device=device)
        print(test_in.shape)
        np.save(os.path.join(self.data_root, "test_in.npy"), test_in.cpu().numpy())
        self.test_in = np.load(os.path.join(self.data_root, "test_in.npy"))
        assert np.linalg.norm(self.test_in - test_in.cpu().numpy()) < 1e-3

        for attention_type in attention_types:
            for normalize_Q_K in normalize_Q_Ks:
                for att_with_output_proj in att_with_output_projs:
                    for cosine_att in cosine_atts:
                        for att_with_relative_position_bias in att_with_relative_position_biases:
                            for stride_qk in stride_qks:
                                m = SpatialGlobalAttention(
                                    window_size=[16, 16],
                                    patch_size=[8, 8],
                                    num_wind=None,
                                    num_patch=None,
                                    attention_type=attention_type,
                                    C_in=C,
                                    C_out=C_out,
                                    H=H1,
                                    W=W1,
                                    stride_qk=stride_qk,
                                    cosine_att=cosine_att,
                                    normalize_Q_K=normalize_Q_K,
                                    att_with_relative_position_bias=att_with_relative_position_bias,
                                    att_with_output_proj=att_with_output_proj,
                                )

                                m.to(device=device)
                                t0 = start_timer(enable=with_timer)
                                test_out = m(test_in)
                                end_timer(
                                    enable=with_timer, t=t0, msg=f"forward pass - {test_in.shape}"
                                )

                                fname = f"{attention_type}_{normalize_Q_K}_{att_with_output_proj}_{cosine_att}_{att_with_relative_position_bias}_{stride_qk}"
                                gt_fname = os.path.join(self.data_root, f"test_out_{fname}.npy")
                                # np.save(gt_fname, test_out.detach().cpu().numpy())
                                assert os.path.exists(gt_fname)
                                test_out_gt = np.load(
                                    os.path.join(self.data_root, f"test_out_{fname}.npy")
                                )
                                # test_out_gt = np.transpose(test_out_gt, [0, 2, 1, 3, 4])
                                assert (
                                    np.linalg.norm(test_out_gt - test_out.detach().cpu().numpy())
                                    / np.linalg.norm(test_out_gt)
                                    < 1e-3
                                )

                                loss = torch.nn.MSELoss()
                                t0 = start_timer(enable=with_timer)
                                mse = loss(test_in, test_out[:, :C])
                                mse.backward()
                                end_timer(enable=with_timer, t=t0, msg="backward pass")
