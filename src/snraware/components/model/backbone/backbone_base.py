"""
Define the base class for the backbones.
Contain common methods and utilities for the backbone models.

The output of a backbone is a list of tensors. If more than one tensor is returned, this list contains the outputs from all resolution levels.
The first tensor res[0] has the highest resolution.

For the hrnet, the res[-1] is the aggregated tensor from all resolutoin levels.

For more information, please refer to the backbone section in the documentation.
"""

import copy
import inspect
from abc import ABC, abstractmethod
from collections import OrderedDict

import interpol
import torch
import torch.nn as nn
from torch import Tensor

from snraware.components.model.attention import *

from .blocks import *

__all__ = [
    "BackboneBase",
    "DownSample",
    "UpSample",
    "set_window_patch_sizes_keep_num_window",
    "set_window_patch_sizes_keep_window_size",
]

# -------------------------------------------------------------------------------------------------


def set_window_patch_sizes_keep_num_window(kwargs, HW, num_wind, num_patch, module_name=None):
    """
    Set window and patch sizes based on the image size and number of windows and patches.

    Args:
        kwargs (dict): dictonary containing the following keys:
            window_size (list): list of two integers representing the window size [H, W].
            patch_size (list): list of two integers representing the patch size [H, W].
            num_wind (list): list of two integers representing the number of windows [num_wind_H, num_wind_W].
            num_patch (list): list of two integers representing the number of patches [num_patch_H, num_patch_W].
            num_wind (list): number of windows.
            num_patch (list): number of patches.

    Returns:
        kwargs: updated dictionary
    """
    num_wind = [2 if v < 2 else v for v in num_wind]
    num_patch = [2 if v < 2 else v for v in num_patch]

    kwargs["window_size"][0] = max(HW[0] // num_wind[0], 1)
    kwargs["window_size"][1] = max(HW[1] // num_wind[1], 1)

    kwargs["patch_size"][0] = max(kwargs["window_size"][0] // num_patch[0], 1)
    kwargs["patch_size"][1] = max(kwargs["window_size"][1] // num_patch[1], 1)

    kwargs["num_wind"] = num_wind
    kwargs["num_patch"] = num_patch

    info_str = f" --> image size {HW} - windows size {kwargs['window_size']} - patch size {kwargs['patch_size']} - num windows {kwargs['num_wind']} - num patch {kwargs['num_patch']}"

    if module_name is not None:
        info_str = module_name + info_str

    print(info_str)

    return kwargs


# -------------------------------------------------------------------------------------------------


def set_window_patch_sizes_keep_window_size(
    kwargs, HW, window_size_input, patch_size_input, module_name=None
):
    """Set window and patch sizes according to image sizes and number of windows and patches."""
    window_size = copy.deepcopy(window_size_input)
    patch_size = copy.deepcopy(patch_size_input)

    if HW[0] // window_size[0] < 2:
        window_size[0] = max(HW[0] // 2, 1)

    if HW[1] // window_size[1] < 2:
        window_size[1] = max(HW[1] // 2, 1)

    if window_size[0] // patch_size[0] < 2:
        patch_size[0] = max(window_size[0] // 2, 1)

    if window_size[1] // patch_size[1] < 2:
        patch_size[1] = max(window_size[1] // 2, 1)

    kwargs["window_size"] = window_size
    kwargs["patch_size"] = patch_size

    kwargs["num_wind"][0] = max(HW[0] // window_size[0], 1)
    kwargs["num_wind"][1] = max(HW[1] // window_size[1], 1)

    kwargs["num_patch"][0] = max(window_size[0] // patch_size[0], 1)
    kwargs["num_patch"][1] = max(window_size[1] // patch_size[1], 1)

    info_str = f" --> image size {HW} - windows size {kwargs['window_size']} - patch size {kwargs['patch_size']} - num windows {kwargs['num_wind']} - num patch {kwargs['num_patch']}"

    if module_name is not None:
        info_str = module_name + info_str

    print(info_str)

    return kwargs


# -------------------------------------------------------------------------------------------------
# building blocks


class _D2(nn.Module):
    """
    Downsample by 2 layer.

    This module takes in a [B, T, C, H, W] tensor and downsample it to [B, T, C, H//2, W//2]

    By default, the operation is performed with a bilinear interpolation.
    If with_conv is True, a 1x1 convolution is added after interpolation.
    If with_interpolation is False, the stride convolution is used.
    """

    def __init__(self, C_in=16, C_out=-1, use_interpolation=True, with_conv=True) -> None:
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out if C_out > 0 else C_in

        self.use_interpolation = use_interpolation
        self.with_conv = with_conv

        self.stride_conv = None
        self.conv = None

        if not self.use_interpolation:
            self.stride_conv = Conv2DExt(
                in_channels=self.C_in,
                out_channels=self.C_out,
                kernel_size=[3, 3],
                stride=[2, 2],
                padding=[1, 1],
            )
        elif self.with_conv or (self.C_in != self.C_out):
            self.conv = Conv2DExt(
                in_channels=self.C_in,
                out_channels=self.C_out,
                kernel_size=[1, 1],
                stride=[1, 1],
                padding=[0, 0],
            )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape
        if self.use_interpolation:
            y = F.interpolate(
                x.view((B * T, C, H, W)),
                scale_factor=(0.5, 0.5),
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            y = torch.reshape(y, (B, T, *y.shape[1:]))
            if self.with_conv:
                y = self.conv(y)
        else:
            y = self.stride_conv(x)

        return y


# -------------------------------------------------------------------------------------------------


class _D2_patch_merging(nn.Module):
    """
    Downsample by 2 layer using patch merging.

    This module takes in a [B, T, C, H, W] tensor and first reformat it to [B, T, 4*C_in, H//2, W//2],
    then a conv is used to get C_out channels.
    """

    def __init__(self, C_in=16, C_out=64) -> None:
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out if C_out > 0 else C_in

        self.conv = Conv2DExt(
            in_channels=4 * self.C_in,
            out_channels=self.C_out,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
        )

    def forward(self, x: Tensor) -> Tensor:
        _B, _T, _C, _H, _W = x.shape

        x0 = x[:, :, :, 0::2, 0::2]  # B T C, H/2 W/2
        x1 = x[:, :, :, 1::2, 0::2]
        x2 = x[:, :, :, 0::2, 1::2]
        x3 = x[:, :, :, 1::2, 1::2]

        y = torch.cat([x0, x1, x2, x3], dim=2)  # B T 4*C H/2 W/2
        y = self.conv(y)

        return y


# -------------------------------------------------------------------------------------------------


class _D2_3D(nn.Module):
    """
    Downsample by 2.

    This module takes in a [B, T, C, H, W] tensor and downsample it to [B, T//2, C, H//2, W//2]

    By default, the operation is performed with a trilinear interpolation.
    If with_conv is True, a 1x1 convolution is added after interpolation.
    If with_interpolation is False, the stride convolution is used.
    """

    def __init__(self, C_in=16, C_out=-1, use_interpolation=True, with_conv=True) -> None:
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out if C_out > 0 else C_in

        self.use_interpolation = use_interpolation
        self.with_conv = with_conv

        self.stride_conv = None
        self.conv = None

        if not self.use_interpolation:
            self.stride_conv = Conv3DExt(
                in_channels=self.C_in,
                out_channels=self.C_out,
                kernel_size=[3, 3, 3],
                stride=[2, 2, 2],
                padding=[1, 1, 1],
            )
        elif self.with_conv or (self.C_in != self.C_out):
            self.conv = Conv3DExt(
                in_channels=self.C_in,
                out_channels=self.C_out,
                kernel_size=[1, 1, 1],
                stride=[1, 1, 1],
                padding=[0, 0, 0],
            )

    def forward(self, x: Tensor) -> Tensor:
        _B, _T, _C, _H, _W = x.shape
        if self.use_interpolation:
            y = F.interpolate(
                torch.permute(x, (0, 2, 1, 3, 4)),
                scale_factor=(0.5, 0.5, 0.5),
                mode="trilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            y = torch.permute(y, (0, 2, 1, 3, 4))
            if self.with_conv:
                y = self.conv(y)
        else:
            y = self.stride_conv(x)

        return y


# -------------------------------------------------------------------------------------------------


class _D2_patch_merging_3D(nn.Module):
    """
    Downsample by 2 layer using patch merging.

    This module takes in a [B, T, C, H, W] tensor and first reformat it to [B, T//2, 8*C_in, H//2, W//2],
    then a conv is used to get C_out channels.
    """

    def __init__(self, C_in=16, C_out=64) -> None:
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out if C_out > 0 else C_in

        self.conv = Conv3DExt(
            in_channels=8 * self.C_in,
            out_channels=self.C_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
        )

    def forward(self, x: Tensor) -> Tensor:
        _B, _T, _C, _H, _W = x.shape

        x0 = x[:, 0::2, :, 0::2, 0::2]  # B T C, H/2 W/2
        x1 = x[:, 0::2, :, 1::2, 0::2]
        x2 = x[:, 0::2, :, 0::2, 1::2]
        x3 = x[:, 0::2, :, 1::2, 1::2]
        x4 = x[:, 1::2, :, 0::2, 0::2]
        x5 = x[:, 1::2, :, 1::2, 0::2]
        x6 = x[:, 1::2, :, 0::2, 1::2]
        x7 = x[:, 1::2, :, 1::2, 1::2]

        y = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=2)
        y = self.conv(y)

        return y


# -------------------------------------------------------------------------------------------------


class DownSample(nn.Module):
    """Downsample by x=2^N, by using N D2 layers."""

    def __init__(
        self, N=2, C_in=16, C_out=-1, use_interpolation=True, with_conv=True, is_3D=False
    ) -> None:
        super().__init__()

        C_out = C_out if C_out > 0 else C_in

        self.N = N
        self.C_in = C_in
        self.C_out = C_out
        self.use_interpolation = use_interpolation
        self.with_conv = with_conv
        self.is_3D = is_3D

        DownSampleLayer = _D2_patch_merging
        if is_3D:
            DownSampleLayer = _D2_patch_merging_3D

        layers = [("D2_0", DownSampleLayer(C_in=C_in, C_out=C_out))]

        for n in range(1, N):
            layers.append((f"D2_{n}", DownSampleLayer(C_in=C_out, C_out=C_out)))

        self.block = nn.Sequential(OrderedDict(layers))

    def forward(self, x: Tensor) -> Tensor:
        return permute_to_B_C_T_H_W(self.block(permute_to_B_T_C_H_W(x)))


# -------------------------------------------------------------------------------------------------


class _U2(nn.Module):
    """
    Upsample by 2.

    This module takes in a [B, T, Cin, H, W] tensor and upsample it to [B, T, Cout, 2*H, 2*W], if channel_first is False
    This module takes in a [B, Cin, T, H, W] tensor and upsample it to [B, Cout, T, 2*H, 2*W], if channel_first is True

    By default, the operation is performed with a bilinear interpolation.
    If with_conv is True, a 1x1 convolution is added after interpolation.
    """

    def __init__(
        self, C_in=16, C_out=-1, method="linear", with_conv=True, channel_first=False
    ) -> None:
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out if C_out > 0 else C_in

        self.method = method

        self.with_conv = with_conv
        self.channel_first = channel_first

        self.conv = None
        if self.with_conv or (self.C_in != self.C_out):
            self.conv = Conv2DExt(
                in_channels=self.C_in,
                out_channels=self.C_out,
                kernel_size=[3, 3],
                stride=[1, 1],
                padding=[1, 1],
                channel_first=self.channel_first,
            )

    def forward(self, x: Tensor) -> Tensor:
        B, D1, D2, H, W = x.shape

        if self.method == "NN":
            y = F.interpolate(
                x.reshape((B * D1, D2, H, W)),
                size=(2 * H, 2 * W),
                mode="nearest",
                recompute_scale_factor=False,
            )
        elif self.method == "linear":
            y = F.interpolate(
                x.reshape((B * D1, D2, H, W)),
                size=(2 * H, 2 * W),
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        else:
            opt = dict(shape=[2 * H, 2 * W], anchor="first", bound="replicate")
            y = interpol.resize(x.reshape((B * D1, D2, H, W)), **opt, interpolation=5)

        y = torch.reshape(y, (B, D1, *y.shape[1:]))
        if self.with_conv:
            y = self.conv(y)

        return y


class _U2_3D(nn.Module):
    """
    Upsample by 2 for 3D tensors.

    By default, the operation is performed with a trilinear interpolation.
    If with_conv is True, a 1x1 convolution is added after interpolation.
    """

    def __init__(
        self, C_in=16, C_out=-1, method="linear", with_conv=True, channel_first=False
    ) -> None:
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out if C_out > 0 else C_in
        self.method = method
        self.with_conv = with_conv
        self.channel_first = channel_first

        self.conv = None
        if self.with_conv or (self.C_in != self.C_out):
            self.conv = Conv3DExt(
                in_channels=self.C_in,
                out_channels=self.C_out,
                kernel_size=[3, 3, 3],
                stride=[1, 1, 1],
                padding=[1, 1, 1],
                channel_first=self.channel_first,
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.channel_first:
            x_channel_first = x
        else:
            x_channel_first = torch.permute(x, (0, 2, 1, 3, 4))

        _B, _C, T, H, W = x_channel_first.shape

        if self.method == "NN":
            y = F.interpolate(
                x_channel_first,
                size=(2 * T, 2 * H, 2 * W),
                mode="nearest",
                recompute_scale_factor=False,
            )
        elif self.method == "linear":
            y = F.interpolate(
                x_channel_first,
                size=(2 * T, 2 * H, 2 * W),
                mode="trilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        else:
            opt = dict(shape=[2 * T, 2 * H, 2 * W], anchor="first", bound="replicate")
            y = interpol.resize(x_channel_first, **opt, interpolation=5)

        if not self.channel_first:
            y = torch.permute(y, (0, 2, 1, 3, 4))

        if self.with_conv:
            y = self.conv(y)

        return y


# -------------------------------------------------------------------------------------------------


class UpSample(nn.Module):
    """Upsample by x(2^N), by using N U2 layers."""

    def __init__(
        self,
        N=2,
        C_in=16,
        C_out=-1,
        method="linear",
        with_conv=True,
        is_3D=False,
        channel_first=False,
    ) -> None:
        super().__init__()

        C_out = C_out if C_out > 0 else C_in

        self.N = N
        self.C_in = C_in
        self.C_out = C_out
        self.with_conv = with_conv
        self.is_3D = is_3D
        self.method = method
        self.channel_first = channel_first

        UpSampleLayer = _U2
        if is_3D:
            UpSampleLayer = _U2_3D

        layers = [
            (
                "U2_0",
                UpSampleLayer(
                    C_in=C_in,
                    C_out=C_out,
                    method=method,
                    with_conv=with_conv,
                    channel_first=self.channel_first,
                ),
            )
        ]
        for n in range(1, N):
            layers.append(
                (
                    f"U2_{n}",
                    UpSampleLayer(
                        C_in=C_out,
                        C_out=C_out,
                        method=method,
                        with_conv=with_conv,
                        channel_first=self.channel_first,
                    ),
                )
            )

        self.block = nn.Sequential(OrderedDict(layers))

    def forward(self, x: Tensor) -> Tensor:
        return permute_to_B_C_T_H_W(self.block(permute_to_B_T_C_H_W(x)))


# -------------------------------------------------------------------------------------------------
class BackboneBase(nn.Module, ABC):
    """Base class for backbone models."""

    def __init__(self, config) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup.
        """
        super().__init__()
        self.config = config

        # preset some default parameters
        self.use_interpolation = True  # whether to use interpolation for up/downsampling
        self.with_conv = True  # whether to add convolution after up/downsampling

    @abstractmethod
    def get_number_of_output_channels(self) -> int:
        """Derived class needs to return number of output channels."""
        pass

    def permute(self, x):
        return torch.permute(x, (0, 2, 1, 3, 4))

    def get_block_parameters(self, model_config):
        """
        Helper function for block instantiation.
        This function extracts the parameters used to define a block and store them as a dictionary.
        """
        sig = inspect.signature(Block.__init__)
        params = dict()
        for name in sig.parameters.keys():
            if name in model_config:
                params[name] = model_config[name]

        block_config = copy.copy(self.config.block)
        block_config.cell.window_size = model_config["window_size"]
        block_config.cell.patch_size = model_config["patch_size"]
        params["config"] = block_config

        return params

    @property
    def device(self):
        return next(self.parameters()).device


# -------------------------------------------------------------------------------------------------
