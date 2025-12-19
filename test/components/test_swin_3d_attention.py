import os
from pathlib import Path

import numpy as np
import pytest
import torch
from colorama import Fore, Style

from snraware.components.model.attention import Swin3DAttention
from snraware.components.setup import end_timer, get_device, set_seed, start_timer

# -----------------------------------------------------------------


class TestSwin3DAttention:
    def setup_class(self):
        set_seed(1467865)
        torch.set_printoptions(precision=10)

        Current_DIR = Path(__file__).parents[0].resolve()

        self.test_path = str(Current_DIR)
        self.data_root = str(Current_DIR) + "/../data/swin_3d"

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

        B, D = 1, 16
        C_out = 4
        test_in = t.repeat(B, D, 1, 1, 1)
        print(test_in.shape)

        C = test_in.shape[-3]
        H = test_in.shape[-2]
        W = test_in.shape[-1]

        att = Swin3DAttention(
            H=H,
            W=W,
            D=D,
            window_size=None,
            num_wind=[4, 4, 2],
            attention_type="conv",
            C_in=C,
            C_out=C_out,
        )

        a = att.im2grid(torch.permute(test_in, [0, 2, 1, 3, 4]))
        b = att.grid2im(a)
        b = torch.permute(b, [0, 2, 1, 3, 4])

        assert torch.allclose(test_in, b)

        attention_types = ["conv", "lin"]
        normalize_Q_Ks = [True, False]
        cosine_atts = [True, False]
        att_with_relative_position_biases = [False]
        att_with_output_projs = [True, False]
        stride_qks = [[1, 1, 1]]

        with_timer = True

        B, T, C, H1, W1 = 1, 16, 2, 32, 32
        C_out = 32
        test_in = torch.rand(B, T, C, H1, W1).to(device=device)
        print(test_in.shape)
        # np.save(os.path.join(self.data_root, "test_in.npy"), test_in.detach().cpu().numpy())
        # self.test_in = np.load(os.path.join(self.data_root, 'test_in.npy'))
        assert np.linalg.norm(self.test_in - test_in.cpu().numpy()) < 1e-3

        test_in = torch.permute(test_in, [0, 2, 1, 3, 4])

        for attention_type in attention_types:
            for normalize_Q_K in normalize_Q_Ks:
                for att_with_output_proj in att_with_output_projs:
                    for cosine_att in cosine_atts:
                        for att_with_relative_position_bias in att_with_relative_position_biases:
                            for stride_qk in stride_qks:
                                fname = f"{attention_type}_{normalize_Q_K}_{att_with_output_proj}_{cosine_att}_{att_with_relative_position_bias}_{stride_qk}"
                                print(
                                    f"{Fore.YELLOW}run - {fname}, {torch.linalg.norm(test_in)}{Style.RESET_ALL}"
                                )

                                att = Swin3DAttention(
                                    window_size=None,
                                    num_wind=[4, 4, 2],
                                    attention_type=attention_type,
                                    C_in=C,
                                    C_out=C_out,
                                    H=H1,
                                    W=W1,
                                    D=T,
                                    stride_qk=stride_qk,
                                    cosine_att=cosine_att,
                                    normalize_Q_K=normalize_Q_K,
                                    att_with_relative_position_bias=att_with_relative_position_bias,
                                    att_with_output_proj=att_with_output_proj,
                                )
                                att.to(device=device)

                                t0 = start_timer(enable=with_timer)
                                test_out = att(test_in)
                                end_timer(
                                    enable=with_timer,
                                    t=t0,
                                    msg=f"forward pass - test_in {test_in.shape}",
                                )

                                gt_fname = os.path.join(self.data_root, f"test_out_{fname}.npy")
                                # np.save(gt_fname, test_out.detach().cpu().numpy())
                                assert os.path.exists(gt_fname)
                                test_out_gt = np.load(gt_fname)
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
