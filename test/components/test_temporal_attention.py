import os
from pathlib import Path

import numpy as np
import pytest
import torch

from snraware.components.model.attention import TemporalChannelCnnAttention, TemporalCnnAttention
from snraware.components.setup import end_timer, get_device, set_seed, start_timer

# -----------------------------------------------------------------


class TestTemporalAttention:
    def setup_class(self):
        set_seed(89845)
        torch.set_printoptions(precision=10)

        Current_DIR = Path(__file__).parents[0].resolve()

        self.test_path = str(Current_DIR)
        self.data_root = str(Current_DIR) + "/../data/temporal"

        self.test_in = np.load(os.path.join(self.data_root, "test_in.npy"))

    def teardown_class(self):
        pass

    @pytest.mark.gpu
    def test(self):
        device = get_device()
        if device != "cuda":
            pytest.skip("GPU only test")

        B, T, C, H, W = 2, 16, 3, 16, 16
        C_out = 32

        with_timer = True

        test_in = torch.rand(B, T, C, H, W, device=device)
        assert np.linalg.norm(self.test_in - test_in.cpu().numpy()) < 1e-3

        print("Begin Testing")

        test_in = torch.permute(test_in, [0, 2, 1, 3, 4])

        causals = [True, False]
        normalize_Q_Ks = [True, False]
        att_with_output_projs = [True, False]
        stride_qks = [[1, 1], [2, 2]]
        for causal in causals:
            for normalize_Q_K in normalize_Q_Ks:
                for stride_qk in stride_qks:
                    for att_with_output_proj in att_with_output_projs:
                        # --------------------------------------
                        t0 = start_timer(enable=with_timer)
                        temporal = TemporalCnnAttention(
                            C,
                            C_out=C_out,
                            is_causal=causal,
                            normalize_Q_K=normalize_Q_K,
                            att_with_output_proj=att_with_output_proj,
                        ).to(device=device)
                        test_out = temporal(test_in)
                        end_timer(enable=with_timer, t=t0, msg=f"forward pass - {test_in.shape}")

                        fname = f"TemporalCnnStandardAttention_{normalize_Q_K}_{att_with_output_proj}_{causal}_{stride_qk}"
                        gt_fname = os.path.join(self.data_root, f"test_out_{fname}.npy")
                        # np.save(gt_fname, test_out.detach().cpu().numpy())
                        assert os.path.exists(gt_fname)
                        test_out_gt = np.load(
                            os.path.join(self.data_root, f"test_out_{fname}.npy")
                        )
                        test_out_gt = np.transpose(test_out_gt, [0, 2, 1, 3, 4])
                        # assert np.linalg.norm(test_out_gt - test_out.detach().cpu().numpy())/np.linalg.norm(test_out_gt) < 1e-3

                        t0 = start_timer(enable=with_timer)
                        loss = torch.nn.MSELoss()
                        mse = loss(test_in, test_out[:, :C])
                        mse.backward()
                        end_timer(enable=with_timer, t=t0, msg="backward pass")

                        # --------------------------------------
                        t0 = start_timer(enable=with_timer)
                        temporal = TemporalChannelCnnAttention(
                            C,
                            C_out=C_out,
                            is_causal=causal,
                            normalize_Q_K=normalize_Q_K,
                            att_with_output_proj=att_with_output_proj,
                        ).to(device=device)
                        test_out = temporal(test_in)
                        end_timer(enable=with_timer, t=t0, msg=f"forward pass - {test_in.shape}")

                        fname = f"TemporalChannelCnnAttention_{normalize_Q_K}_{att_with_output_proj}_{causal}_{stride_qk}"
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
                        end_timer(enable=with_timer, t=t0, msg="backward pass")
