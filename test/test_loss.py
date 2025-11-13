import os
from pathlib import Path

import numpy as np
import scipy
import torch
from colorama import Fore, Style

from snraware.projects.loss.imaging_loss import Combined_Loss

# -----------------------------------------------------------------


class TestLoss:
    test_path = None
    data_root = None
    data = None
    inputs = None
    noise = None
    noise2 = None
    noise3 = None

    def setup_class(self):
        torch.set_printoptions(precision=10)

        Current_DIR = Path(__file__).parents[0].resolve()

        self.test_path = str(Current_DIR)
        self.data_root = str(Current_DIR) + "/data/loss"

        self.data = np.load(os.path.join(self.data_root, "data_real.npy")) + 1j * np.load(
            os.path.join(self.data_root, "data_imag.npy")
        )

        # create degraded version
        sigmas = [1.0, 2.0, 3.0, 8.0, 20.0]

        self.inputs = list()
        for sig in sigmas:
            self.inputs.append(scipy.ndimage.gaussian_filter(self.data, sig))

        self.noise = list()
        self.noise2 = list()
        self.noise3 = list()
        for ind, sd in enumerate([1.0, 5.0, 10.0, 15.0, 40.0]):
            self.noise.append(sd * np.random.randn(*self.inputs[ind].shape))
            self.noise2.append(sd * np.random.randn(*self.inputs[ind].shape))
            self.noise3.append(sd * np.random.randn(*self.inputs[ind].shape))

    def teardown_class(self):
        pass

    def run_loss(self, complex=True):
        complex_losses = [
            "msssim",
            "ssim",
            "ssim3D",
            "mse",
            "rmse",
            "l1",
            "charbonnier",
            "psnr",
            "perpendicular",
            "gaussian",
            "gaussian3D",
            "dwt",
            ["mse", "rmse", "l1", "charbonnier"],
            ["ssim", "ssim3D", "psnr"],
            ["perpendicular", "ssim", "dwt"],
        ]
        real_losses = [
            "ssim",
            "ssim3D",
            "mse",
            "rmse",
            "l1",
            "charbonnier",
            "psnr",
            "msssim",
            "gaussian",
            "gaussian3D",
            "dwt",
            ["mse", "rmse", "l1", "charbonnier"],
            ["ssim", "ssim3D", "psnr"],
            ["ssim", "dwt"],
        ]

        for a_loss in complex_losses if complex else real_losses:
            print(f"test_loss, complex {complex}, test {Fore.YELLOW}{a_loss}{Style.RESET_ALL}...")
            loss_f = Combined_Loss(
                losses=a_loss if isinstance(a_loss, list) else [a_loss],
                loss_weights=[1.0] * 6,
                complex_i=complex,
                device="cpu",
            )
            v = list()
            for im, nn in zip(self.inputs, self.noise, strict=False):
                data = np.copy(self.data)
                data = np.expand_dims(np.transpose(data, [2, 0, 1]), axis=[0, 1])
                x = np.expand_dims(np.transpose(im, [2, 0, 1]), axis=[0, 1])
                n = np.expand_dims(np.transpose(nn, [2, 0, 1]), axis=[0, 1])

                if complex:
                    data = np.concatenate([np.real(data), np.imag(data)], axis=1)
                    x = np.concatenate([np.real(x), np.imag(x)], axis=1)
                    n = np.concatenate([np.real(n), np.imag(n)], axis=1)
                    a_v = loss_f(
                        outputs=torch.from_numpy(x + n).type(torch.FloatTensor),
                        targets=torch.from_numpy(data).type(torch.FloatTensor),
                    )
                else:
                    a_v = loss_f(
                        outputs=torch.from_numpy(np.abs(x + n)).type(torch.FloatTensor),
                        targets=torch.from_numpy(np.abs(data)).type(torch.FloatTensor),
                    )

                v.append(a_v.cpu().item())

            print(f"loss values: {v}")
            assert v[-1] > v[-2] and v[-2] > v[-3] and v[-3] > v[-4] and v[-4] > v[-5]

    def test_loss_complex(self):
        self.run_loss(complex=True)

    def test_loss_magnitude(self):
        self.run_loss(complex=False)
