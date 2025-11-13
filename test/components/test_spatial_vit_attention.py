import os
from pathlib import Path

import numpy as np
import pytest
import torch

from snraware.components.model.attention import SpatialViTAttention
from snraware.components.setup import end_timer, get_device, set_seed, start_timer

# -----------------------------------------------------------------


class TestSpatialViT:
    def setup_class(self):
        set_seed(1232543)
        torch.set_printoptions(precision=10)

        Current_DIR = Path(__file__).parents[0].resolve()

        self.test_path = str(Current_DIR)
        self.data_root = str(Current_DIR) + "/../data/vit2d"

        self.test_in = np.load(os.path.join(self.data_root, "test_in.npy"))
        self.test_in2 = np.load(os.path.join(self.data_root, "test_in2.npy"))

    def teardown_class(self):
        pass

    @pytest.mark.gpu
    def test(self):
        print("Begin Testing")

        t = np.arange(256)
        t = np.reshape(t, (16, 16))

        w = 8

        t = torch.from_numpy(t).to(dtype=torch.float32)
        t = torch.cat((t[None, :], t[None, :]), dim=0)

        B, T, C, H, W = 2, 16, 2, 64, 64
        C_out = 8
        test_in = t.repeat(B, T, 1, 1, 1)
        print(test_in.shape)

        vit = SpatialViTAttention(
            window_size=[w, w],
            num_wind=None,
            attention_type="conv",
            C_in=C,
            C_out=C_out,
            H=H,
            W=W,
            stride_qk=(1, 1),
            n_head=8,
            cosine_att=True,
            normalize_Q_K=True,
            att_with_relative_position_bias=False,
            att_with_output_proj=True,
            with_timer=True,
        )

        a = vit.im2grid(test_in)
        b = vit.grid2im(a)

        assert torch.allclose(test_in, b)

        attention_types = ["conv", "lin"]
        normalize_Q_Ks = [True, False]
        cosine_atts = [True, False]
        att_with_relative_position_biases = [False]
        att_with_output_projs = [True, False]
        stride_qks = [[1, 1, 1], [2, 2, 2]]

        with_timer = True

        device = get_device()

        B, T, C, H1, W1 = 1, 16, 2, 32, 32
        C_out = 8
        test_in = torch.rand(B, T, C, H1, W1).to(device=device)
        print(test_in.shape)
        assert np.linalg.norm(self.test_in - test_in.cpu().numpy()) < 1e-3

        B, T, C, H2, W2 = 1, 16, 2, 128, 128
        C_out = 8
        test_in2 = torch.rand(B, T, C, H2, W2).to(device=device)
        print(test_in2.shape)
        assert np.linalg.norm(self.test_in2 - test_in2.cpu().numpy()) < 1e-3

        test_in = torch.permute(test_in, [0, 2, 1, 3, 4])
        test_in2 = torch.permute(test_in2, [0, 2, 1, 3, 4])

        for attention_type in attention_types:
            for normalize_Q_K in normalize_Q_Ks:
                for att_with_output_proj in att_with_output_projs:
                    for cosine_att in cosine_atts:
                        for att_with_relative_position_bias in att_with_relative_position_biases:
                            for stride_qk in stride_qks:
                                vit = SpatialViTAttention(
                                    window_size=[w, w],
                                    num_wind=None,
                                    attention_type=attention_type,
                                    C_in=C,
                                    C_out=C_out,
                                    H=H1,
                                    W=W1,
                                    stride_qk=stride_qk[:2],
                                    n_head=8,
                                    cosine_att=cosine_att,
                                    normalize_Q_K=normalize_Q_K,
                                    att_with_relative_position_bias=att_with_relative_position_bias,
                                    att_with_output_proj=att_with_output_proj,
                                    with_timer=False,
                                )
                                vit.to(device=device)

                                t0 = start_timer(enable=with_timer)
                                test_out = vit(test_in)
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
                                test_out_gt = np.transpose(test_out_gt, [0, 2, 1, 3, 4])
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

                                print(f"attention_type is {attention_type}, mse is {mse.item()}")

                                # -------------------------------------------------

                                vit = SpatialViTAttention(
                                    window_size=[w, w],
                                    num_wind=None,
                                    attention_type=attention_type,
                                    C_in=C,
                                    C_out=C_out,
                                    H=H2,
                                    W=W2,
                                    stride_qk=stride_qk[:2],
                                    n_head=8,
                                    cosine_att=cosine_att,
                                    normalize_Q_K=normalize_Q_K,
                                    att_with_relative_position_bias=att_with_relative_position_bias,
                                    att_with_output_proj=att_with_output_proj,
                                    with_timer=False,
                                )
                                vit.to(device=device)

                                t0 = start_timer(enable=with_timer)
                                test_out = vit(test_in2)
                                end_timer(
                                    enable=with_timer, t=t0, msg=f"forward pass - {test_in2.shape}"
                                )
