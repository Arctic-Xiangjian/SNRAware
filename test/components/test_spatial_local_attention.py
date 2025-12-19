import os
from pathlib import Path

import numpy as np
import pytest
import torch

from snraware.components.model.attention import SpatialLocalAttention
from snraware.components.setup import end_timer, get_device, set_seed, start_timer

# -----------------------------------------------------------------


class TestSpatialLocalAttention:
    def setup_class(self):
        set_seed(358)
        torch.set_printoptions(precision=10)

        Current_DIR = Path(__file__).parents[0].resolve()

        self.test_path = str(Current_DIR)
        self.data_root = str(Current_DIR) + "/../data/spatial_local"

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

        spacial_vit = SpatialLocalAttention(
            H=H,
            W=W,
            window_size=None,
            patch_size=None,
            num_wind=[2, 2],
            num_patch=[2, 2],
            attention_type="conv",
            C_in=C,
            C_out=C_out,
        )

        a = spacial_vit.im2grid(
            test_in
        )  # b t num_win_h num_win_w num_patch_h num_patch_w patch_size_h patch_size_w c
        b = spacial_vit.grid2im(a)

        assert torch.allclose(test_in, b)

        gt = torch.tensor(
            [
                [
                    [
                        [128.0, 129.0, 130.0, 131.0],
                        [144.0, 145.0, 146.0, 147.0],
                        [160.0, 161.0, 162.0, 163.0],
                        [176.0, 177.0, 178.0, 179.0],
                    ],
                    [
                        [132.0, 133.0, 134.0, 135.0],
                        [148.0, 149.0, 150.0, 151.0],
                        [164.0, 165.0, 166.0, 167.0],
                        [180.0, 181.0, 182.0, 183.0],
                    ],
                ],
                [
                    [
                        [192.0, 193.0, 194.0, 195.0],
                        [208.0, 209.0, 210.0, 211.0],
                        [224.0, 225.0, 226.0, 227.0],
                        [240.0, 241.0, 242.0, 243.0],
                    ],
                    [
                        [196.0, 197.0, 198.0, 199.0],
                        [212.0, 213.0, 214.0, 215.0],
                        [228.0, 229.0, 230.0, 231.0],
                        [244.0, 245.0, 246.0, 247.0],
                    ],
                ],
            ]
        )

        assert torch.norm(a[0, 0, 1, 0, :, :, :, :, 1] - gt) < 1e-3
        assert torch.norm(b - test_in) < 1e-3

        attention_types = ["conv", "lin"]
        normalize_Q_Ks = [True, False]
        cosine_atts = [True, False]
        att_with_relative_position_biases = [True, False]
        att_with_output_projs = [True, False]
        stride_qks = [[1, 1], [2, 2]]

        with_timer = True

        B, T, C, H1, W1 = 1, 4, 2, 64, 64
        C_out = 16
        n_head = 16
        test_in = torch.rand(B, T, C, H1, W1).to(device=device)
        print(test_in.shape)

        B, T, C, H2, W2 = 1, 4, 2, 32, 32
        C_out = 16
        test_in2 = torch.rand(B, T, C, H2, W2).to(device=device)
        print(test_in2.shape)

        test_in = torch.permute(test_in, [0, 2, 1, 3, 4])
        test_in2 = torch.permute(test_in2, [0, 2, 1, 3, 4])

        for attention_type in attention_types:
            for normalize_Q_K in normalize_Q_Ks:
                for att_with_output_proj in att_with_output_projs:
                    for cosine_att in cosine_atts:
                        for att_with_relative_position_bias in att_with_relative_position_biases:
                            for stride_qk in stride_qks:
                                m = SpatialLocalAttention(
                                    H=H1,
                                    W=W1,
                                    window_size=None,
                                    patch_size=None,
                                    num_wind=[4, 4],
                                    num_patch=[2, 2],
                                    attention_type=attention_type,
                                    C_in=C,
                                    C_out=C_out,
                                    n_head=n_head,
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
                                assert (
                                    np.linalg.norm(test_out_gt - test_out.detach().cpu().numpy())
                                    / np.linalg.norm(test_out_gt)
                                    < 1e-3
                                )

                                t0 = start_timer(enable=with_timer)
                                loss = torch.nn.MSELoss()
                                mse = loss(test_in, test_out[:, :C])
                                mse.backward()
                                end_timer(
                                    enable=with_timer,
                                    t=t0,
                                    msg=f"backward pass - stride_qk {stride_qk}",
                                )

                                t0 = start_timer(enable=with_timer)
                                test_out = m(test_in2)
                                end_timer(
                                    enable=with_timer, t=t0, msg=f"forward pass -{test_in2.shape}"
                                )
