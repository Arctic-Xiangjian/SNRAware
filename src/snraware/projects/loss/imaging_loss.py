"""
Losses for imaging applications.

Provides implmentation of the following types of losses:
    - FSIM: Feature Similarity Index Measure
    - SSIM: Structural Similarity Index Measure for 2D
    - SSIM3D: Structural Similarity Index Measure for 3D
    - MSSSIM: multi-scale SSIM for 2D
    - L1: Mean Absolute Error
    - MSE: Mean Squared Error
    - Perpendicular: perp loss for complex images
    - Combined: Any weighed combination of the above

Allows custom weights for each indvidual loss calculation as well.

The input tensor has the shape [B, C, T/F, H, W]. It can be complex or real in data types.

A weights tensor can be supplied to provide different weighting for each sample. weights can be [B], or [B, T] in shape.
In the former case, every batch can have different weighting. In the latter, each batch and frame step can have different weighting.
"""

import numpy as np
import piq
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision
from pytorch_wavelets import DWTForward, DWTInverse
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_functional

from snraware.projects.loss.gaussian import (
    create_gaussian_window_1d,
    create_gaussian_window_2d,
    create_gaussian_window_3d,
)

__all__ = [
    "PSNR",
    "Charbonnier_Loss",
    "Combined_Loss",
    "FSIM_Loss",
    "GaussianDeriv1D_Loss",
    "GaussianDeriv3D_Loss",
    "GaussianDeriv_Loss",
    "L1_Loss",
    "MSE_Loss",
    "MSSSIM_Loss",
    "PSNR_Loss",
    "Perpendicular_Loss",
    "SSIM3D_Loss",
    "SSIM_Loss",
    "VGGPerceptualLoss",
]


# -------------------------------------------------------------------------------------------------
def _get_magnitude_tensor(outputs, targets, is_complex=True):
    if is_complex:
        assert targets.shape[1] == 2, (
            f"Complex type requires image to have C=2, given C={targets.shape[1]}"
        )
        outputs_mag = torch.sqrt(outputs[:, :1] * outputs[:, :1] + outputs[:, 1:] * outputs[:, 1:])
        targets_mag = torch.sqrt(targets[:, :1] * targets[:, :1] + targets[:, 1:] * targets[:, 1:])
    else:
        outputs_mag = outputs
        targets_mag = targets

    return outputs_mag, targets_mag


# -------------------------------------------------------------------------------------------------
# Feature Similarity Index Measure (FSIM) loss


class FSIM_Loss:
    """Weighted FSIM loss."""

    def __init__(self, chromatic=False, data_range=None, complex_i=False, device="cpu"):
        """
        @args:
            - chromatic (bool) : flag to compute FSIMc, which also takes into account chromatic components
            - data_range (float): max data value in the training; if none, determine from data
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on.
        """
        self.complex_i = complex_i
        self.chromatic = chromatic
        self.data_range = data_range

    def __call__(self, outputs, targets, weights=None):
        outputs_im, targets_im = _get_magnitude_tensor(outputs, targets, is_complex=self.complex_i)

        B, C, T, H, W = targets_im.shape
        outputs_im = torch.reshape(torch.permute(outputs_im, (0, 2, 1, 3, 4)), (B * T, C, H, W))
        targets_im = torch.reshape(torch.permute(targets_im, (0, 2, 1, 3, 4)), (B * T, C, H, W))

        data_range = self.data_range
        if self.data_range is None:
            data_range = torch.max(torch.cat((targets_im, outputs_im), dim=0))

        loss = piq.fsim(
            outputs_im,
            targets_im,
            reduction="none",
            data_range=data_range,
            chromatic=self.chromatic,
        )

        if weights is not None:
            if weights.ndim == 1:
                weights_used = weights.expand(T, B).permute(1, 0).reshape(B * T)
            elif weights.ndim == 2:
                weights_used = weights.reshape(B * T)
            else:
                raise NotImplementedError(
                    "Only support 1D(Batch) or 2D(Batch+Time) weights for FSIM_Loss"
                )

            v = torch.sum(weights_used * loss) / (
                torch.sum(weights_used) + torch.finfo(torch.float32).eps
            )
        else:
            v = torch.mean(loss)

        if torch.any(torch.isnan(v)):
            v = torch.tensor(1.0, requires_grad=True)

        return 1.0 - v


# -------------------------------------------------------------------------------------------------
# SSIM loss


class SSIM_Loss:
    """Weighted SSIM loss."""

    def __init__(self, window_size=11, complex_i=False, device="cpu"):
        """
        @args:
            - window_size (int): size of the window to use for loss computation
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on.
        """
        self.complex_i = complex_i
        self.window_size = window_size

    def __call__(self, outputs, targets, weights=None):
        outputs_im, targets_im = _get_magnitude_tensor(outputs, targets, is_complex=self.complex_i)

        B, C, T, H, W = targets_im.shape
        outputs_im = torch.reshape(torch.permute(outputs_im, (0, 2, 1, 3, 4)), (B * T, C, H, W))
        targets_im = torch.reshape(torch.permute(targets_im, (0, 2, 1, 3, 4)), (B * T, C, H, W))

        if weights is not None:
            if weights.ndim == 1:
                weights_used = weights.expand(T, B).permute(1, 0).reshape(B * T)
            elif weights.ndim == 2:
                weights_used = weights.reshape(B * T)
            else:
                raise NotImplementedError(
                    "Only support 1D(Batch) or 2D(Batch+Time) weights for SSIM_Loss"
                )

            v_ssim = torch.sum(
                weights_used
                * ssim_functional(
                    outputs_im,
                    targets_im,
                    gaussian_kernel=False,
                    kernel_size=self.window_size,
                    reduction=None,
                )
            ) / (torch.sum(weights_used) + torch.finfo(torch.float32).eps)
        else:
            v_ssim = torch.mean(
                ssim_functional(
                    outputs_im,
                    targets_im,
                    gaussian_kernel=False,
                    kernel_size=self.window_size,
                    reduction=None,
                )
            )

        if torch.any(torch.isnan(v_ssim)):
            v_ssim = torch.tensor(1.0, requires_grad=True)

        return 1.0 - v_ssim


# -------------------------------------------------------------------------------------------------
# SSIM3D loss


class SSIM3D_Loss(SSIM_Loss):
    """Weighted SSIM3D loss."""

    def __init__(self, window_size=11, complex_i=False, device="cpu"):
        super().__init__(window_size=window_size, complex_i=complex_i, device=device)

    def __call__(self, outputs, targets, weights=None):
        outputs_im, targets_im = _get_magnitude_tensor(outputs, targets, is_complex=self.complex_i)

        _B, _C, _T, _H, _W = targets_im.shape

        if weights is not None:
            if not weights.ndim == 1:
                raise NotImplementedError("Only support 1D(Batch) weights for SSIM3D_Loss")
            v_ssim = torch.sum(
                weights
                * ssim_functional(
                    outputs_im,
                    targets_im,
                    gaussian_kernel=False,
                    kernel_size=self.window_size,
                    reduction=None,
                )
            ) / (torch.sum(weights) + torch.finfo(torch.float32).eps)
        else:
            v_ssim = torch.mean(
                ssim_functional(
                    outputs_im,
                    targets_im,
                    gaussian_kernel=False,
                    kernel_size=self.window_size,
                    reduction=None,
                )
            )

        if torch.any(torch.isnan(v_ssim)):
            v_ssim = torch.tensor(1.0, requires_grad=True)

        return 1.0 - v_ssim


# -------------------------------------------------------------------------------------------------
# MSSSIM loss


class MSSSIM_Loss:
    """Weighted MSSSIM loss."""

    def __init__(self, window_size=5, complex_i=False, data_range=256.0, device="cpu"):
        """
        @args:
            - window_size (int): size of the window to use for loss computation
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on.
        """
        self.complex_i = complex_i
        self.data_range = data_range
        self.msssim_loss = torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure(
            sigma=window_size, reduction=None
        )

    def __call__(self, outputs, targets, weights=None):
        outputs_im, targets_im = _get_magnitude_tensor(outputs, targets, is_complex=self.complex_i)

        B, C, T, H, W = targets_im.shape
        outputs_im = torch.reshape(torch.permute(outputs_im, (0, 2, 1, 3, 4)), (B * T, C, H, W))
        targets_im = torch.reshape(torch.permute(targets_im, (0, 2, 1, 3, 4)), (B * T, C, H, W))

        v = self.msssim_loss(outputs_im, targets_im)

        if weights is not None:
            if weights.ndim == 1:
                weights_used = weights.expand(T, B).permute(1, 0).reshape(B * T)
            elif weights.ndim == 2:
                weights_used = weights.reshape(B * T)
            else:
                raise NotImplementedError(
                    "Only support 1D(Batch) or 2D(Batch+Time) weights for MSSSIM_Loss"
                )

            v_ssim = torch.sum(weights_used * v) / (
                torch.sum(weights_used) + torch.finfo(torch.float32).eps
            )
        else:
            v_ssim = torch.mean(v)

        v_ssim = torch.clamp(v_ssim, 0.0, 1.0)

        self.msssim_loss.reset()
        return 1.0 - v_ssim


# -------------------------------------------------------------------------------------------------
# L1/mae loss


class L1_Loss:
    """Weighted L1 loss."""

    def __init__(self, complex_i=False):
        """
        @args:
            - complex_i (bool): whether images are 2 channelled for complex data.
        """
        self.complex_i = complex_i

    def __call__(self, outputs, targets, weights=None):
        B, C, T, _H, _W = targets.shape
        if self.complex_i:
            assert C == 2, f"Complex type requires image to have C=2, given C={C}"
            diff_L1 = torch.abs(outputs[:, 0] - targets[:, 0]) + torch.abs(
                outputs[:, 1] - targets[:, 1]
            )
        else:
            diff_L1 = torch.abs(outputs - targets)

        if weights is not None:
            if weights.ndim == 1:
                weights = weights.reshape(B, 1, 1, 1, 1)
            elif weights.ndim == 2:
                weights = weights.reshape(B, T, 1, 1, 1)
            else:
                raise NotImplementedError(
                    "Only support 1D(Batch) or 2D(Batch+Time) weights for L1_Loss"
                )

            v_l1 = torch.sum(weights * diff_L1) / (
                torch.sum(weights) + torch.finfo(torch.float32).eps
            )
        else:
            v_l1 = torch.sum(diff_L1)

        if torch.any(torch.isnan(v_l1)):
            v_l1 = torch.mean(0.0 * outputs)

        return v_l1 / diff_L1.numel()


# -------------------------------------------------------------------------------------------------
# MSE loss


class MSE_Loss:
    """Weighted MSE loss."""

    def __init__(self, rmse_mode=False, complex_i=False):
        """
        @args:
            - rmse_mode (bool): whether to turn root mean sequare error
            - complex_i (bool): whether images are 2 channelled for complex data.
        """
        self.rmse_mode = rmse_mode
        self.complex_i = complex_i

    def __call__(self, outputs, targets, weights=None):
        outputs = outputs.to(torch.float32)
        targets = targets.to(torch.float32)

        B, C, T, _H, _W = targets.shape
        if self.complex_i:
            assert C == 2, f"Complex type requires image to have C=2, given C={C}"
            diff_mag_square = torch.square(outputs[:, 0] - targets[:, 0]) + torch.square(
                outputs[:, 1] - targets[:, 1]
            )
        else:
            diff_mag_square = torch.square(outputs - targets)

        if self.rmse_mode:
            diff_mag_square = torch.sqrt(diff_mag_square)

        if weights is not None:
            if weights.ndim == 1:
                weights = weights.reshape(B, 1, 1, 1, 1)
            elif weights.ndim == 2:
                weights = weights.reshape(B, T, 1, 1, 1)
            else:
                raise NotImplementedError(
                    "Only support 1D(Batch) or 2D(Batch+Time) weights for MSE_Loss"
                )

            v_l2 = torch.sum(weights * diff_mag_square) / (
                torch.sum(weights) + torch.finfo(torch.float32).eps
            )
        else:
            return torch.mean(diff_mag_square)

        if torch.any(torch.isnan(v_l2)):
            v_l2 = torch.mean(0.0 * outputs)

        return v_l2 / diff_mag_square.numel()


# -------------------------------------------------------------------------------------------------
# PSNR


class PSNR:
    """PSNR as a comparison metric."""

    def __init__(self, range=1.0):
        """
        @args:
            - range (float): max range of the values in the images.
        """
        self.range = range

    def __call__(self, outputs, targets):
        num = self.range * self.range
        den = torch.mean(torch.square(targets - outputs)) + torch.finfo(torch.float32).eps

        return 10 * torch.log10(num / den)


class PSNR_Loss:
    """PSNR as a comparison metric."""

    def __init__(self, range=1.0):
        """
        @args:
            - range (float): max range of the values in the images.
        """
        self.range = range

    def __call__(self, outputs, targets, weights=None):
        B, _C, T, _H, _W = targets.shape

        num = self.range * self.range
        den = torch.square(targets - outputs) + torch.finfo(torch.float32).eps

        if weights is not None:
            if weights.ndim == 1:
                weights = weights.reshape(B, 1, 1, 1, 1)
            elif weights.ndim == 2:
                weights = weights.reshape(B, T, 1, 1, 1)
            else:
                raise NotImplementedError(
                    "Only support 1D(Batch) or 2D(Batch+Time) weights for PSNR_Loss"
                )

            v_l2 = torch.sum(weights * torch.log10(num / den)) / (
                torch.sum(weights) + torch.finfo(torch.float32).eps
            )
        else:
            v_l2 = torch.sum(torch.log10(num / den))

        if torch.any(torch.isnan(v_l2)):
            v_l2 = torch.mean(0.0 * outputs)

        return 10 - v_l2 / den.numel()


# -------------------------------------------------------------------------------------------------


def perpendicular_loss_complex(X, Y):
    """
    Perpendicular loss for complex MR images.

    from https://gitlab.com/computational-imaging-lab/perp_loss/-/blob/main/PerpLoss_-_Image_reconstruction.ipynb

    Args:
        X (complex images): torch complex images
        Y (complex images): torch complex images

    Outputs:
        final_term: the loss in the same size as X and Y
    """
    assert X.is_complex()
    assert Y.is_complex()

    mag_input = torch.abs(X)
    mag_target = torch.abs(Y)
    cross = torch.abs(X.real * Y.imag - X.imag * Y.real)

    eps_float32 = torch.finfo(torch.float32).eps

    angle = torch.atan2(X.imag, X.real + eps_float32) - torch.atan2(Y.imag, Y.real + eps_float32)
    ploss = torch.abs(cross) / (mag_input + eps_float32)

    aligned_mask = (torch.cos(angle) < 0).bool()

    final_term = torch.zeros_like(ploss)
    final_term[aligned_mask] = mag_target[aligned_mask] + (
        mag_target[aligned_mask] - ploss[aligned_mask]
    )
    final_term[~aligned_mask] = ploss[~aligned_mask]

    return final_term


class Perpendicular_Loss:
    """Perpendicular loss."""

    def __init__(self):
        pass

    def __call__(self, outputs, targets, weights=None):
        B, C, T, _H, _W = targets.shape

        if C == 1:
            loss = perpendicular_loss_complex(
                outputs[:, 0] + 1j * outputs[:, 0], targets[:, 0] + 1j * targets[:, 0]
            )
        else:
            loss = perpendicular_loss_complex(
                outputs[:, 0] + 1j * outputs[:, 1], targets[:, 0] + 1j * targets[:, 1]
            )

        if weights is not None:
            if weights.ndim == 1:
                weights = weights.reshape(B, 1, 1, 1, 1)
            elif weights.ndim == 2:
                weights = weights.reshape(B, T, 1, 1, 1)
            else:
                raise NotImplementedError(
                    "Only support 1D(Batch) or 2D(Batch+Time) weights for Perpendicular_Loss"
                )

            v = torch.sum(weights * loss) / (torch.sum(weights) + torch.finfo(torch.float32).eps)
        else:
            return torch.mean(loss)

        if torch.any(torch.isnan(v)):
            v = torch.mean(0.0 * outputs)

        return v / targets.numel()


# -------------------------------------------------------------------------------------------------
# Charbonnier Loss


class Charbonnier_Loss:
    """Charbonnier Loss (L1)."""

    def __init__(self, complex_i=False, eps=1e-3):
        """
        @args:
            - complex_i (bool): whether images are 2 channelled for complex data
            - eps (float): epsilon, different values can be tried here.
        """
        self.complex_i = complex_i
        self.eps = eps

    def __call__(self, outputs, targets, weights=None):
        B, C, T, _H, _W = targets.shape
        if self.complex_i:
            assert C == 2, f"Complex type requires image to have C=2, given C={C}"
            diff_L1_real = torch.abs(outputs[:, 0] - targets[:, 0])
            diff_L1_imag = torch.abs(outputs[:, 1] - targets[:, 1])
            loss = torch.sqrt(
                diff_L1_real * diff_L1_real + diff_L1_imag * diff_L1_imag + self.eps * self.eps
            )
        else:
            diff_L1 = torch.abs(outputs - targets)
            loss = torch.sqrt(diff_L1 * diff_L1 + self.eps * self.eps)

        if weights is not None:
            if weights.ndim == 1:
                weights = weights.reshape(B, 1, 1, 1, 1)
            elif weights.ndim == 2:
                weights = weights.reshape(B, T, 1, 1, 1)
            else:
                raise NotImplementedError(
                    "Only support 1D(Batch) or 2D(Batch+Time) weights for L1_Loss"
                )

            v_l1 = torch.sum(weights * loss) / torch.sum(weights)
        else:
            v_l1 = torch.sum(loss)

        return v_l1 / loss.numel()


# -------------------------------------------------------------------------------------------------
# Perceptual Loss


class VGGPerceptualLoss(torch.nn.Module):
    """Perceptual Loss (VGG Loss)."""

    def __init__(self, complex_i=False, resize=False, interpolate_mode="bilinear"):
        super().__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights="DEFAULT").features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights="DEFAULT").features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights="DEFAULT").features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights="DEFAULT").features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.interpolate_mode = interpolate_mode
        self.complex_i = complex_i
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    # can also try feature_layers=[2], style_layers=[0, 1, 2, 3]
    def __call__(self, outputs, targets, feature_layers=None, style_layers=None, weights=None):
        if style_layers is None:
            style_layers = []
        if feature_layers is None:
            feature_layers = [0, 1, 2, 3]
        B, C, T, H, W = targets.shape
        if self.complex_i:
            assert C == 2, f"Complex type requires image to have C=2, given C={C}"
            outputs_im = torch.sqrt(outputs[:, 0] * outputs[:, 0] + outputs[:, 1] * outputs[:, 1])
            targets_im = torch.sqrt(targets[:, 0] * targets[:, 0] + targets[:, 1] * targets[:, 1])
            outputs_im = torch.unsqueeze(outputs_im, dim=1)
            targets_im = torch.unsqueeze(targets_im, dim=1)
        else:
            outputs_im = outputs
            targets_im = targets

        B, C, T, H, W = targets_im.shape
        outputs_im = torch.reshape(torch.permute(outputs_im, (0, 2, 1, 3, 4)), (B * T, C, H, W))
        targets_im = torch.reshape(torch.permute(targets_im, (0, 2, 1, 3, 4)), (B * T, C, H, W))

        if outputs_im.shape[1] != 3:
            outputs_im = outputs_im.repeat(1, 3, 1, 1)
            targets_im = targets_im.repeat(1, 3, 1, 1)

        if self.resize:
            outputs_im = self.transform(
                outputs_im, mode=self.interpolate_mode, size=(224, 224), align_corners=False
            )
            targets_im = self.transform(
                targets_im, mode=self.interpolate_mode, size=(224, 224), align_corners=False
            )

        outputs_im = outputs_im.to(dtype=torch.float32)
        targets_im = targets_im.to(dtype=torch.float32)

        loss = 0.0
        x = outputs_im
        y = targets_im
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)

        if weights is not None:
            if weights.ndim == 1:
                weights_used = weights.expand(T, B).permute(1, 0).reshape(B * T)
            elif weights.ndim == 2:
                weights_used = weights.reshape(B * T)
            else:
                raise NotImplementedError("Only support 1D(Batch) or 2D(Batch+Time) weights")
            v_vgg = torch.sum(weights_used * loss) / (
                torch.sum(weights_used) + torch.finfo(torch.float16).eps
            )
        else:
            v_vgg = torch.mean(loss)

        return v_vgg


# -------------------------------------------------------------------------------------------------


class GaussianDeriv1D_Loss:
    """Weighted gaussian derivative loss for 1D."""

    def __init__(self, sigmas=None, complex_i=False, device="cpu"):
        """
        @args:
            - sigmas (list): sigma for every scale
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on.
        """
        if sigmas is None:
            sigmas = [0.5, 1.0, 1.25]
        self.complex_i = complex_i
        self.sigmas = sigmas

        # compute kernels
        self.kernels = []
        for sigma in sigmas:
            k_1d = create_gaussian_window_1d(sigma=sigma, halfwidth=3, voxelsize=1.0, order=1)
            kx = k_1d.shape
            k_1d = torch.from_numpy(np.reshape(k_1d, (1, 1, kx))).to(torch.float32)
            self.kernels.append(k_1d.to(device=device))

    def __call__(self, outputs, targets, weights=None):
        outputs_im, targets_im = _get_magnitude_tensor(outputs, targets, is_complex=self.complex_i)

        B, _T, C = targets_im.shape
        outputs_im = torch.permute(outputs_im, (0, 2, 1))
        targets_im = torch.permute(targets_im, (0, 2, 1))

        loss = 0
        for k_1d in self.kernels:
            grad_outputs_im = F.conv1d(
                outputs_im, k_1d, bias=None, stride=1, padding="same", groups=C
            )
            grad_targets_im = F.conv1d(
                targets_im, k_1d, bias=None, stride=1, padding="same", groups=C
            )
            loss += torch.mean(
                torch.abs(grad_outputs_im - grad_targets_im), dim=(1, 2), keepdim=True
            )

        loss /= len(self.kernels)

        if weights is not None:
            if weights.ndim == 1:
                weights_used = weights.reshape((B, 1))
            else:
                raise NotImplementedError(
                    "Only support 1D(Batch) weights for GaussianDeriv1D_Loss"
                )

            v = torch.sum(weights_used * loss) / (
                torch.sum(weights_used) + torch.finfo(torch.float32).eps
            )
        else:
            v = torch.mean(loss)

        if torch.any(torch.isnan(v)):
            v = torch.mean(0.0 * outputs)

        return v


# -------------------------------------------------------------------------------------------------


class GaussianDeriv_Loss:
    """
    Weighted gaussian derivative loss for 2D
    For every sigma, the gaussian derivatives are computed for outputs and targets along the magnitude of H and W
    The l1 loss are computed to measure the agreement of gaussian derivatives.

    If sigmas have more than one value, every sigma in sigmas are used to compute a guassian derivative tensor
    The mean l1 is returned
    """

    def __init__(self, sigmas=None, complex_i=False, device="cpu"):
        """
        @args:
            - sigmas (list): sigma for every scale
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on.
        """
        if sigmas is None:
            sigmas = [0.5, 1.0, 1.25]
        self.complex_i = complex_i
        self.sigmas = sigmas

        # compute kernels
        self.kernels = []
        for sigma in sigmas:
            k_2d = create_gaussian_window_2d(
                sigma=(sigma, sigma), halfwidth=(3, 3), voxelsize=(1.0, 1.0), order=(1, 1)
            )
            kx, ky = k_2d.shape
            k_2d = torch.from_numpy(np.reshape(k_2d, (1, 1, kx, ky))).to(torch.float32)
            self.kernels.append(k_2d.to(device=device))

    def __call__(self, outputs, targets, weights=None):
        outputs_im, targets_im = _get_magnitude_tensor(outputs, targets, is_complex=self.complex_i)

        B, C, T, H, W = targets_im.shape
        outputs_im = torch.reshape(torch.permute(outputs_im, (0, 2, 1, 3, 4)), (B * T, C, H, W))
        targets_im = torch.reshape(torch.permute(targets_im, (0, 2, 1, 3, 4)), (B * T, C, H, W))

        loss = 0
        for k_2d in self.kernels:
            grad_outputs_im = F.conv2d(
                outputs_im, k_2d, bias=None, stride=1, padding="same", groups=C
            )
            grad_targets_im = F.conv2d(
                targets_im, k_2d, bias=None, stride=1, padding="same", groups=C
            )
            loss += torch.mean(
                torch.abs(grad_outputs_im - grad_targets_im), dim=(1, 2, 3), keepdim=True
            )

        loss /= len(self.kernels)

        if weights is not None:
            if weights.ndim == 1:
                weights_used = weights.expand(T, B).permute(1, 0).reshape(B * T)
            elif weights.ndim == 2:
                weights_used = weights.reshape(B * T)
            else:
                raise NotImplementedError(
                    "Only support 1D(Batch) or 2D(Batch+Time) weights for GaussianDeriv_Loss"
                )

            v = torch.sum(weights_used * loss) / (
                torch.sum(weights_used) + torch.finfo(torch.float32).eps
            )
        else:
            v = torch.mean(loss)

        if torch.any(torch.isnan(v)):
            v = torch.mean(0.0 * outputs)

        return v


# -------------------------------------------------------------------------------------------------


class GaussianDeriv3D_Loss:
    """
    Weighted gaussian derivative loss for 3D
    For every sigma, the gaussian derivatives are computed for outputs and targets along the magnitude of T, H, W
    The l1 loss are computed to measure the agreement of gaussian derivatives.

    If sigmas have more than one value, every sigma in sigmas are used to compute a guassian derivative tensor
    The mean l1 is returned
    """

    def __init__(self, sigmas=None, sigmas_T=None, complex_i=False, device="cpu"):
        """
        @args:
            - sigmas (list): sigma for every scale along H and W
            - sigmas_T (list): sigma for every scale along T
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on.
        """
        if sigmas_T is None:
            sigmas_T = [0.5, 1.0, 1.25]
        if sigmas is None:
            sigmas = [0.5, 1.0, 1.25]
        self.complex_i = complex_i
        self.sigmas = sigmas
        self.sigmas_T = sigmas_T

        assert len(self.sigmas_T) == len(self.sigmas)

        # compute kernels
        self.kernels = []
        for sigma, sigma_T in zip(sigmas, sigmas_T, strict=True):
            k_3d = create_gaussian_window_3d(
                sigma=(sigma, sigma, sigma_T),
                halfwidth=(3, 3, 3),
                voxelsize=(1.0, 1.0, 1.0),
                order=(1, 1, 1),
            )
            kx, ky, kz = k_3d.shape
            k_3d = torch.from_numpy(np.reshape(k_3d, (1, 1, kx, ky, kz))).to(torch.float32)
            k_3d = torch.permute(k_3d, [0, 1, 4, 2, 3])
            self.kernels.append(k_3d.to(device=device))

    def __call__(self, outputs, targets, weights=None):
        outputs_im, targets_im = _get_magnitude_tensor(outputs, targets, is_complex=self.complex_i)

        _B, C, _T, _H, _W = targets_im.shape

        loss = 0
        for k_3d in self.kernels:
            grad_outputs_im = F.conv3d(
                outputs_im, k_3d, bias=None, stride=1, padding="same", groups=C
            )
            grad_targets_im = F.conv3d(
                targets_im, k_3d, bias=None, stride=1, padding="same", groups=C
            )
            loss += torch.mean(
                torch.abs(grad_outputs_im - grad_targets_im), dim=(1, 2, 3, 4), keepdim=True
            )

        loss /= len(self.kernels)

        if weights is not None:
            if not weights.ndim == 1:
                raise NotImplementedError(
                    "Only support 1D(Batch) weights for GaussianDeriv3D_Loss"
                )
            v = torch.sum(weights * loss) / (torch.sum(weights) + torch.finfo(torch.float32).eps)
        else:
            v = torch.mean(loss)

        if torch.any(torch.isnan(v)):
            v = torch.mean(0.0 * outputs)

        return v


# -------------------------------------------------------------------------------------------------


class Spectral_Loss:
    """
    Test Spectral loss based on fft.

    Mask out center and edges of the shifted square fft and
    use different methods to computer the loss
    """

    def __init__(self, dim=None, min_bound=5, max_bound=95, complex_i=False, device="cpu"):
        """
        @args:
            - dim (int list): the dimension to take fft on
            - min_bound (int): the percentage of middle mask
            - max_bound (int): the percentage of edge mask
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on.
        """
        if dim is None:
            dim = [-2, -1]
        self.dim = dim
        self.eps = 1e-7

        self.min_bound = min_bound
        self.max_bound = max_bound

        self.complex_i = complex_i
        self.device = device

    def create_mask(self, shape):
        _B, _C, _T, H, W = shape

        mask = torch.zeros(*shape)

        # keep edges at 0 make everything inside 1
        upper_bound_H = int(self.max_bound * H / 100)
        upper_bound_W = int(self.max_bound * W / 100)
        mask[:, :, :, (H - upper_bound_H) : upper_bound_H, (W - upper_bound_W) : upper_bound_W] = 1

        lower_bound_H = int(self.min_bound * H / 100)
        lower_bound_W = int(self.min_bound * W / 100)
        ch, cw = H // 2, W // 2

        # make the inner square 0
        # with the edge case to keep the middle single pixel
        if self.min_bound != -1:
            mask[
                :,
                :,
                :,
                (ch - lower_bound_H) : (ch + lower_bound_H + 1),
                (cw - lower_bound_W) : (cw + lower_bound_W + 1),
            ] = 0

        return mask

    def __call__(self, outputs, targets, weights=None):
        _B, C, _T, _H, _W = targets.shape
        if self.complex_i:
            assert C == 2, f"Complex type requires image to have C=2, given C={C}"
            outputs_im = outputs[:, 0] + 1j * outputs[:, 1]
            targets_im = targets[:, 0] + 1j * targets[:, 1]
            outputs_im = torch.unsqueeze(outputs_im, dim=1)
            targets_im = torch.unsqueeze(targets_im, dim=1)
        else:
            outputs_im = outputs
            targets_im = targets

        outputs_im = outputs_im.to(device=self.device)
        targets_im = targets_im.to(device=self.device)

        outputs_fft = torch.fft.fftshift(
            torch.fft.fftn(outputs_im, dim=self.dim, norm="backward"), dim=self.dim
        )
        targets_fft = torch.fft.fftshift(
            torch.fft.fftn(targets_im, dim=self.dim, norm="backward"), dim=self.dim
        )

        outputs_fft[outputs_fft == 0] = self.eps
        targets_fft[targets_fft == 0] = self.eps

        outputs_fft_log = torch.log(outputs_fft)
        targets_fft_log = torch.log(targets_fft)

        mask = self.create_mask(outputs_fft.shape).to(self.device)

        outputs_fft_log_masked = outputs_fft_log * mask
        targets_fft_log_masked = targets_fft_log * mask

        # loss = perpendicular_loss_complex(outputs_fft_masked, targets_fft_masked)

        loss = torch.abs(targets_fft_log_masked - outputs_fft_log_masked)

        if weights is not None:
            if not weights.ndim == 1:
                raise NotImplementedError("Only support 1D(Batch) weights for spectral loss")
            v = torch.sum(weights * loss) / (torch.sum(weights) + torch.finfo(torch.float16).eps)
        else:
            v = torch.sum(loss)

        if torch.any(torch.isnan(v)):
            v = torch.mean(0.0 * outputs)

        return v / targets.numel()


# -------------------------------------------------------------------------------------------------


class Wavelet_Loss:
    """
    Test wavelet loss based on pytorch_wavelets.
    This version performs 2D wavelet transformation on the last two dimensions [-2, -1].
    """

    def __init__(
        self,
        J=1,
        wave="db3",
        mode="reflect",
        separable=False,
        only_h=True,
        complex_i=False,
        device="cpu",
    ):
        """
        @args:
            - J (int): number of wavelet levels
            - wave (str): wavelet type, 'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'
            - mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'
            - only_h (bool): if True, only count high frequency bands for loss
            - complex_i (bool): whether images are 2 channelled for complex data.
        """
        self.J = J
        self.wave = wave
        self.mode = mode
        self.separable = separable
        self.only_h = only_h
        self.complex_i = complex_i
        self.device = device

        self.wt = DWTForward(J=J, wave=wave, mode=mode).to(device=self.device)
        self.iwt = DWTInverse(wave=wave, mode=mode).to(device=self.device)

    def __call__(self, outputs, targets, weights=None):
        B, C, T, H, W = targets.shape
        if self.complex_i:
            assert C == 2, f"Complex type requires image to have C=2, given C={C}"

        outputs_im = outputs
        targets_im = targets

        outputs_im = outputs_im.to(device=self.device)
        targets_im = targets_im.to(device=self.device)

        ol, oh = self.wt(torch.reshape(outputs_im, (B * C, T, H, W)))
        tl, th = self.wt(torch.reshape(targets_im, (B * C, T, H, W)))

        if weights is not None:
            if not weights.ndim == 1:
                raise NotImplementedError("Only support 1D(Batch) weights for spectral loss")

            loss = 0
            for a, b in zip(oh, th, strict=False):
                v = torch.abs(a - b)
                loss += torch.sum(weights * v) / (
                    torch.sum(weights) + torch.finfo(torch.float16).eps
                )

            if not self.only_h:
                v = torch.mean(torch.abs(ol - tl))
                loss += torch.sum(weights * v) / (
                    torch.sum(weights) + torch.finfo(torch.float16).eps
                )

            loss /= targets.numel()
        else:
            loss = 0
            for a, b in zip(oh, th, strict=False):
                loss += torch.mean(torch.abs(a - b))

            if not self.only_h:
                loss += torch.mean(torch.abs(ol - tl))

        if torch.any(torch.isnan(loss)):
            loss = torch.mean(0.0 * outputs)

        return loss


# -------------------------------------------------------------------------------------------------
# Combined loss class


class Combined_Loss:
    """
    Combined loss for image enhancement
    Sums multiple loss with their respective weights.
    """

    def __init__(self, losses, loss_weights, complex_i=False, device="cpu") -> None:
        """
        @args:
            - losses (list of "ssim", "ssim3D", "l1", "mse"):
                list of losses to be combined
            - loss_weights (list of floats)
                weights of the losses in the combined loss
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on.
        """
        assert len(losses) > 0, "At least one loss is required to setup"
        assert len(losses) <= len(loss_weights), "Each loss should have its weight"

        self.complex_i = complex_i
        self.device = device

        losses = [self.str_to_loss(loss) for loss in losses]
        self.losses = list(zip(losses, loss_weights, strict=False))

    def str_to_loss(self, loss_name):
        if loss_name.lower() == "mse":
            loss_f = MSE_Loss(rmse_mode=False, complex_i=self.complex_i)
        elif loss_name.lower() == "rmse":
            loss_f = MSE_Loss(rmse_mode=True, complex_i=self.complex_i)
        elif loss_name.lower() == "l1":
            loss_f = L1_Loss(complex_i=self.complex_i)
        elif loss_name.lower() == "charbonnier":
            loss_f = Charbonnier_Loss(complex_i=self.complex_i)
        elif loss_name.lower() == "perceptual":
            loss_f = VGGPerceptualLoss(complex_i=self.complex_i)
            loss_f.to(self.device)
        elif loss_name.lower() == "ssim":
            loss_f = SSIM_Loss(window_size=7, complex_i=self.complex_i, device=self.device)
        elif loss_name.lower() == "ssim3d":
            loss_f = SSIM3D_Loss(window_size=5, complex_i=self.complex_i, device=self.device)
        elif loss_name == "psnr":
            loss_f = PSNR_Loss(range=2048.0)
        elif loss_name.lower() == "perpendicular":
            loss_f = Perpendicular_Loss()
        elif loss_name.lower() == "msssim":
            loss_f = MSSSIM_Loss(
                window_size=3, complex_i=self.complex_i, data_range=256, device=self.device
            )
        elif loss_name.lower() == "gaussian":
            loss_f = GaussianDeriv_Loss(
                sigmas=[0.25, 0.5, 1.0, 1.5], complex_i=self.complex_i, device=self.device
            )
        elif loss_name.lower() == "gaussian3d":
            loss_f = GaussianDeriv3D_Loss(
                sigmas=[0.25, 0.5, 1.0],
                sigmas_T=[0.25, 0.5, 0.5],
                complex_i=self.complex_i,
                device=self.device,
            )
        elif loss_name.lower() == "spec":
            loss_f = Spectral_Loss(
                dim=[-2, -1],
                min_bound=5,
                max_bound=95,
                complex_i=self.complex_i,
                device=self.device,
            )
        elif loss_name.lower() == "dwt":
            loss_f = Wavelet_Loss(
                J=2,
                wave="db3",
                mode="reflect",
                separable=False,
                only_h=True,
                complex_i=self.complex_i,
                device=self.device,
            )
        else:
            raise NotImplementedError(f"Loss type not implemented: {loss_name}")

        return loss_f

    def __call__(self, outputs, targets, weights=None):
        outputs = outputs.to(torch.float32)
        targets = targets.to(torch.float32)

        combined_loss = 0
        for loss_f, weight in self.losses:
            v = weight * loss_f(outputs=outputs, targets=targets, weights=weights)
            if not torch.any(torch.isnan(v)):
                combined_loss += v

        return combined_loss


# --------------------------------------------------------------------
