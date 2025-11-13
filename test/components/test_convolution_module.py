import os
from pathlib import Path

import numpy as np
import pytest
import torch

from snraware.components.model.attention import ConvolutionModule
from snraware.components.setup import end_timer, get_device, set_seed, start_timer

# -----------------------------------------------------------------


class TestConvolutionModule:
    def setup_class(self):
        set_seed(78562)
        torch.set_printoptions(precision=10)

        Current_DIR = Path(__file__).parents[0].resolve()

        self.test_path = str(Current_DIR)
        self.data_root = str(Current_DIR) + "/../data/convolution"

        self.test_in = np.load(os.path.join(self.data_root, "test_in.npy"))

    def teardown_class(self):
        pass

    @pytest.mark.gpu
    def test(self):
        import itertools

        B, T, C, H, W = 2, 16, 3, 16, 16
        C_out = 32

        kernel_size = (3, 3, 3)
        stride = (1, 1, 1)
        padding = (1, 1, 1)

        with_timer = True
        device = get_device()

        test_in = torch.rand(B, T, C, H, W, device=device)
        assert np.linalg.norm(self.test_in - test_in.cpu().numpy()) < 1e-3

        test_in = torch.permute(test_in, [0, 2, 1, 3, 4])

        conv_types = ["conv2d", "conv3d"]
        separables = [True, False]
        norm_modes = ["instance2d", "batch2d", "layer"]
        activation_funcs = ["prelu", "gelu", "relu"]

        for conv_type, separable, norm_mode, activation_func in itertools.product(
            conv_types, separables, norm_modes, activation_funcs
        ):
            print(conv_type, separable, norm_mode, activation_func)

            model = ConvolutionModule(
                conv_type=conv_type,
                C_in=C,
                C_out=C_out,
                H=H,
                W=W,
                D=T,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                norm_mode=norm_mode,
                activation_func=activation_func,
            )
            model.to(device)

            t0 = start_timer(enable=with_timer)
            test_out = model(test_in)
            end_timer(enable=with_timer, t=t0, msg=f"forward pass - {test_in.shape}")

            fname = f"{conv_type}_{separable}_{norm_mode}_{activation_func}"
            gt_fname = os.path.join(self.data_root, f"test_out_{fname}.npy")
            # np.save(gt_fname, test_out.detach().cpu().numpy())
            assert os.path.exists(gt_fname)
            test_out_gt = np.load(os.path.join(self.data_root, f"test_out_{fname}.npy"))
            test_out_gt = np.transpose(test_out_gt, [0, 2, 1, 3, 4])
            assert (
                np.linalg.norm(test_out_gt - test_out.detach().cpu().numpy())
                / np.linalg.norm(test_out_gt)
                < 1e-3
            )

            t0 = start_timer(enable=with_timer)
            loss = torch.nn.MSELoss()
            mse = loss(test_in, test_out[:, :C])
            mse.backward()
            end_timer(enable=with_timer, t=t0, msg="backward pass")

        print("Passed all tests")
