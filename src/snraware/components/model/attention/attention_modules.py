"""
Attention modules for the Imaging-FM project.

This module contains base classes and utility functions for implementing various attention mechanisms.

To find more information, please visit the https://health-futures.github.io/resys, imaging fm model section.
"""

import torch.nn as nn
from torch.nn import functional as F

from snraware.components.setup.status import *

# -------------------------------------------------------------------------------------------------


def compute_conv_output_shape(h_w, kernel_size, stride, pad, dilation):
    """
    Utility function for computing output of convolutions given the setup
    @args:
        - h_w (int, int): 2-tuple of height, width of input
        - kernel_size, stride, pad (int, int): 2-tuple of conv parameters
        - dilation (int): dilation conv parameter
    @rets:
        - h, w (int, int): 2-tuple of height, width of image returned by the conv.
    """
    h_0 = h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1
    w_0 = h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1

    h = torch.div(h_0, stride[0], rounding_mode="floor") + 1
    w = torch.div(w_0, stride[1], rounding_mode="floor") + 1

    return h, w


# -------------------------------------------------------------------------------------------------
# Create activation functions


def create_activation_func(name="gelu"):
    if name == "elu":
        return nn.modules.ELU(alpha=1, inplace=False)
    elif name == "relu":
        return nn.modules.ReLU(inplace=False)
    elif name == "leakyrelu":
        return nn.modules.LeakyReLU(negative_slope=0.1, inplace=False)
    elif name == "prelu":
        return nn.modules.PReLU(num_parameters=1, init=0.25)
    elif name == "relu6":
        return nn.modules.ReLU6(inplace=False)
    elif name == "selu":
        return nn.modules.SELU(inplace=False)
    elif name == "celu":
        return nn.modules.CELU(alpha=1, inplace=False)
    elif name == "gelu":
        return nn.modules.GELU(approximate="tanh")
    else:
        return nn.Identity()


# -------------------------------------------------------------------------------------------------


def permute_to_B_T_C_H_W(x):
    """
    X : [B, T/D/Z, C, ...]
    res: [B, C, T, ...].
    """
    return torch.transpose(x, 1, 2)


def permute_to_B_C_T_H_W(x):
    """
    X : [B, C, T/D/Z, ...]
    res: [B, T, C, ...].
    """
    return torch.transpose(x, 1, 2)


# -------------------------------------------------------------------------------------------------
class Conv2DExt(nn.Module):
    # Extends torch 2D conv to support 5D inputs
    # if channel_first is True, input x is [B, C, T, H, W]
    # if channel_first is False, input x is [B, T, C, H, W]
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=None,
        stride=None,
        padding=None,
        padding_mode="reflect",
        bias=False,
        channel_first=False,
    ):
        if padding is None:
            padding = [1, 1]
        if stride is None:
            stride = [1, 1]
        if kernel_size is None:
            kernel_size = [3, 3]
        super().__init__()
        self.channel_first = channel_first
        if not isinstance(kernel_size, int):
            kernel_size = kernel_size[:2]
        if not isinstance(stride, int):
            stride = stride[:2]
        if not isinstance(padding, int):
            padding = padding[:2]

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        # requires input to have 5 dimensions
        if self.channel_first:
            B, C, T, H, W = input.shape
            x = torch.permute(input, [0, 2, 1, 3, 4])
        else:
            B, T, C, H, W = input.shape
            x = input

        y = self.conv(x.reshape((B * T, C, H, W)))
        y = y.reshape([B, T, *y.shape[1:]])

        if self.channel_first:
            y = torch.permute(y, [0, 2, 1, 3, 4])

        return y


# -------------------------------------------------------------------------------------------------
class Conv3DExt(nn.Module):
    # Extends torch 3D conv to support 5D inputs

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=None,
        stride=None,
        padding=None,
        bias=False,
        padding_mode="reflect",
        channel_first=False,
    ):
        if padding is None:
            padding = [1, 1, 1]
        if stride is None:
            stride = [1, 1, 1]
        if kernel_size is None:
            kernel_size = [3, 3, 3]
        super().__init__()
        self.channel_first = channel_first
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        # requires input to have 5 dimensions
        if self.channel_first:
            y = self.conv(input)
            return y
        else:
            _B, _T, _C, _H, _W = input.shape
            x = torch.permute(input, (0, 2, 1, 3, 4))
            y = self.conv(x)
            return torch.permute(y, (0, 2, 1, 3, 4))


# -------------------------------------------------------------------------------------------------
class LinearGridExt(nn.Module):
    # Helper module for linear head in imaging attentions.

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, input):
        # requires input to have 7 dimensions
        *S, Gh, Gw, C = input.shape
        y = self.linear(input.reshape((-1, C * Gh * Gw)))
        y = y.reshape((*S, Gh, Gw, -1))

        return y


# -------------------------------------------------------------------------------------------------


class LinearGrid3DExt(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, input):
        *S, Gd, Gh, Gw, C = input.shape
        y = self.linear(input.reshape((-1, C * Gd * Gh * Gw)))
        y = y.reshape((*S, Gd, Gh, Gw, -1))

        return y


# -------------------------------------------------------------------------------------------------


class PixelShuffle2DExt(nn.Module):
    # Extends torch 2D pixel shuffle

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ps = nn.PixelShuffle(*args, **kwargs)

    def forward(self, input):
        input = permute_to_B_T_C_H_W(input)
        B, T, C, H, W = input.shape
        y = self.ps(input.reshape((B * T, C, H, W)))
        return permute_to_B_C_T_H_W(y.reshape([B, T, *y.shape[1:]]))


# -------------------------------------------------------------------------------------------------


class LayerNorm2DExt(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm(*args, **kwargs)

    def forward(self, input):
        input = permute_to_B_T_C_H_W(input)
        _B, _T, _C, _H, _W = input.shape
        norm_input = self.ln(input)
        return permute_to_B_C_T_H_W(norm_input)


# -------------------------------------------------------------------------------------------------
class LayerNorm3DExt(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm(*args, **kwargs)

    def forward(self, input):
        norm_input = self.ln(input)
        return norm_input


# -------------------------------------------------------------------------------------------------
class BatchNorm2DExt(nn.Module):
    # Extends BatchNorm2D to 5D inputs

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(*args, **kwargs)

    def forward(self, input):
        # requires input to have 5 dimensions
        input = permute_to_B_T_C_H_W(input)
        B, T, C, H, W = input.shape
        norm_input = self.bn(input.reshape(B * T, C, H, W))
        return permute_to_B_C_T_H_W(norm_input.reshape(input.shape))


# -------------------------------------------------------------------------------------------------
class BatchNorm3DExt(nn.Module):
    # Corrects BatchNorm3D, switching first and second dimension

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm3d(*args, **kwargs)

    def forward(self, input):
        norm_input = self.bn(input)
        return norm_input


# -------------------------------------------------------------------------------------------------
class InstanceNorm2DExt(nn.Module):
    # Extends InstanceNorm2D to 5D inputs

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.inst = nn.InstanceNorm2d(*args, **kwargs)

    def forward(self, input):
        input = permute_to_B_T_C_H_W(input)
        B, T, C, H, W = input.shape
        norm_input = self.inst(input.reshape(B * T, C, H, W))
        return permute_to_B_C_T_H_W(norm_input.reshape(input.shape))


# -------------------------------------------------------------------------------------------------
class InstanceNorm3DExt(nn.Module):
    # Corrects InstanceNorm3D, switching first and second dimension

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.inst = nn.InstanceNorm3d(*args, **kwargs)

    def forward(self, input):
        norm_input = self.inst(input)
        return norm_input


# -------------------------------------------------------------------------------------------------
class AvgPool2DExt(nn.Module):
    # Extends torch 2D averaging pooling to support 5D inputs

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.avg_pool_2d = nn.AvgPool2d(*args, **kwargs)

    def forward(self, input):
        input = permute_to_B_T_C_H_W(input)
        B, T, C, H, W = input.shape
        y = self.avg_pool_2d(input.reshape((B * T, C, H, W)))
        return permute_to_B_C_T_H_W(torch.reshape(y, [B, T, *y.shape[1:]]))


# -------------------------------------------------------------------------------------------------
def create_norm(norm_mode="instance2d", C=64, H=32, W=32, D=32):
    if norm_mode == "layer":
        n = LayerNorm2DExt([C, H, W])

    elif norm_mode == "layer3d":
        n = LayerNorm3DExt([C, D, H, W])

    elif norm_mode == "batch2d":
        n = BatchNorm2DExt(C)

    elif norm_mode == "batch3d":
        n = BatchNorm3DExt(C)

    elif norm_mode == "instance2d":
        n = InstanceNorm2DExt(C)

    else:
        n = InstanceNorm3DExt(C)

    return n


# -------------------------------------------------------------------------------------------------


def _get_relative_position_bias_tensor(
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, N: int
) -> torch.Tensor:
    """From pytorch source code."""
    relative_position_bias = relative_position_bias_table[
        relative_position_index[:N, :N].reshape(-1)
    ]
    relative_position_bias = torch.reshape(relative_position_bias, (N, N, -1))
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    return relative_position_bias.unsqueeze(0)


def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor,
    relative_position_index: torch.Tensor,
    window_size: tuple[int],
) -> torch.Tensor:
    """From pytorch source code."""
    N = window_size[0] * window_size[1]
    return _get_relative_position_bias_tensor(
        relative_position_bias_table, relative_position_index, N
    )


def _get_relative_position_bias_3D(
    relative_position_bias_table: torch.Tensor,
    relative_position_index: torch.Tensor,
    window_size: tuple[int],
) -> torch.Tensor:
    """From pytorch source code."""
    N = window_size[0] * window_size[1] * window_size[2]
    return _get_relative_position_bias_tensor(
        relative_position_bias_table, relative_position_index, N
    )


# -------------------------------------------------------------------------------------------------


def cosine_attention(q, k):
    """Computes cosine attention between query and key tensors."""
    att = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
    return att


# -------------------------------------------------------------------------------------------------


def normalize_qk(q, k):
    """
    Normalizes query and key tensors.
    If cosine attention is used, normalization is not applied.
    """
    eps = torch.finfo(k.dtype).eps
    k = (k - torch.mean(k, dim=-1, keepdim=True)) / (
        torch.sqrt(torch.var(k, dim=-1, keepdim=True) + eps)
    )
    q = (q - torch.mean(q, dim=-1, keepdim=True)) / (
        torch.sqrt(torch.var(q, dim=-1, keepdim=True) + eps)
    )
    return q, k


# -------------------------------------------------------------------------------------------------
class CnnAttentionBase(nn.Module):
    """Base class for attention layers."""

    def __init__(
        self,
        C_in,
        C_out=16,
        H=128,
        W=128,
        D=1,
        n_head=8,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        stride_qk=(1, 1, 1),
        att_dropout_p=0.0,
        cosine_att=False,
        normalize_Q_K=False,
        att_with_relative_position_bias=True,
        att_with_output_proj=True,
        flash_att=False,
        with_timer=False,
    ):
        """
        Base class for the cnn attentions.

        Input to the attention layer has the size [B, C, T, H, W]
        Output has the size [B, output_channels, T, H', W']

        @args:
            - C_in (int): number of input channels
            - C_out (int): number of output channels
            - H, W (int): image height and width
            - D (int): number of frames in the input, 1 for 2D attention
            - n_head (int): number of heads in self attention
            - kernel_size, stride, padding (int, int, int): convolution parameters, in the order of (H, W, D), not (D, H, W)!
            - stride_qk (int, int, int): stride to compute q and k
            - att_dropout_p (float): probability of dropout for the attention matrix
            - cosine_att (bool): whether to perform cosine attention; if True, normalize_Q_K will be ignored, as Q and K are already normalized; https://arxiv.org/pdf/2409.18747
            - normalize_Q_K (bool): whether to add normalization for Q and K matrix
            - att_with_relative_position_bias (bool): whether to add relative positional bias
            - att_with_output_proj (bool): whether to add output projection
        """
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.H = H
        self.W = W
        self.D = D
        self.n_head = n_head
        self.kernel_size = (kernel_size,)
        self.stride = stride
        self.stride_qk = stride_qk
        self.padding = padding
        self.att_dropout_p = att_dropout_p
        self.cosine_att = cosine_att
        self.normalize_Q_K = normalize_Q_K
        self.att_with_relative_position_bias = att_with_relative_position_bias
        self.att_with_output_proj = att_with_output_proj
        self.with_timer = with_timer

        if att_with_output_proj:
            if self.D > 1:
                self.output_proj = Conv3DExt(
                    C_out,
                    C_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=True,
                    channel_first=True,
                )
            else:
                self.output_proj = Conv2DExt(
                    C_out,
                    C_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=True,
                    channel_first=True,
                )
        else:
            self.output_proj = nn.Identity()

        if att_dropout_p > 0:
            self.attn_drop = nn.Dropout(p=att_dropout_p)
        else:
            self.attn_drop = nn.Identity()

        self.has_flash_attention = flash_att

    def perform_flash_atten(self, k, q, v):
        softmax_scale = None
        if self.cosine_att:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            softmax_scale = 1
        elif self.normalize_Q_K:
            eps = torch.finfo(k.dtype).eps
            k = (k - torch.mean(k, dim=-1, keepdim=True)) / (
                torch.sqrt(torch.var(k, dim=-1, keepdim=True) + eps)
            )
            q = (q - torch.mean(q, dim=-1, keepdim=True)) / (
                torch.sqrt(torch.var(q, dim=-1, keepdim=True) + eps)
            )

        if self.training:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.att_dropout_p,
                is_causal=self.is_causal,
                scale=softmax_scale,
            )
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=self.is_causal,
                scale=softmax_scale,
            )

        return y

    def get_dimension_for_linear_mixer(self):
        """If linear mixer is to be used, return the dimension of attention computation, as the input dimension for linear mixer."""
        return self.C_out * self.patch_size[0] * self.patch_size[1]

    def define_relative_position_bias_table_3D(self, num_win_h=100, num_win_w=100, num_win_d=100):
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * num_win_h - 1) * (2 * num_win_w - 1) * (2 * num_win_d - 1), self.n_head
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index_3D(self, num_win_h=100, num_win_w=100, num_win_d=100):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(num_win_h)
        coords_w = torch.arange(num_win_w)
        coords_d = torch.arange(num_win_d)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, coords_d, indexing="ij")
        )  # 3, Wh, Ww, Wd
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wd
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 3, Wh*Ww*Wd, Wh*Ww*Wd
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wd, Wh*Ww*Wd, 3
        relative_coords[:, :, 0] += num_win_h - 1  # shift to start from 0
        relative_coords[:, :, 1] += num_win_w - 1
        relative_coords[:, :, 2] += num_win_d - 1
        relative_coords[:, :, 0] *= (2 * num_win_w - 1) * (2 * num_win_d - 1)
        relative_coords[:, :, 1] *= 2 * num_win_d - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias_3D(self, num_win_h, num_win_w, num_win_d) -> torch.Tensor:
        return _get_relative_position_bias_3D(
            self.relative_position_bias_table,
            self.relative_position_index,
            (num_win_h, num_win_w, num_win_d),
        )

    def define_relative_position_bias_table(self, num_win_h=100, num_win_w=100):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * num_win_h - 1) * (2 * num_win_w - 1), self.n_head)
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self, num_win_h=100, num_win_w=100):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(num_win_h)
        coords_w = torch.arange(num_win_w)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += num_win_h - 1  # shift to start from 0
        relative_coords[:, :, 1] += num_win_w - 1
        relative_coords[:, :, 0] *= 2 * num_win_w - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self, num_win_h, num_win_w) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table,
            self.relative_position_index,
            (num_win_h, num_win_w),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def set_and_check_wind(self):
        if self.D > 1:
            if self.num_wind is not None:
                self.window_size = [
                    self.H // self.num_wind[0],
                    self.W // self.num_wind[1],
                    self.D // self.num_wind[2],
                ]
                self.window_size = [max(w, 1) for w in self.window_size]
                self.num_wind = [
                    self.H // self.window_size[0],
                    self.W // self.window_size[1],
                    self.D // self.window_size[2],
                ]
            else:
                self.num_wind = [
                    self.H // self.window_size[0],
                    self.W // self.window_size[1],
                    self.D // self.window_size[2],
                ]
                self.num_wind = [max(n, 1) for n in self.num_wind]
                self.window_size = [
                    self.H // self.num_wind[0],
                    self.W // self.num_wind[1],
                    self.D // self.num_wind[2],
                ]

            assert self.num_wind[2] * self.window_size[2] == self.D, (
                f"self.num_wind[2]*self.window_size[2] == self.D, num_wind {self.num_wind}, window_size {self.window_size}, D {self.D}"
            )
        else:
            if self.num_wind is not None:
                self.window_size = [self.H // self.num_wind[0], self.W // self.num_wind[1]]
                self.window_size = [max(w, 1) for w in self.window_size]
                self.num_wind = [self.H // self.window_size[0], self.W // self.window_size[1]]
            else:
                self.num_wind = [self.H // self.window_size[0], self.W // self.window_size[1]]
                self.num_wind = [max(n, 1) for n in self.num_wind]
                self.window_size = [self.H // self.num_wind[0], self.W // self.num_wind[1]]

        assert self.num_wind[0] * self.window_size[0] == self.H, (
            f"self.num_wind[0]*self.window_size[0] == self.H, num_wind {self.num_wind}, window_size {self.window_size}, H {self.H}"
        )

        assert self.num_wind[1] * self.window_size[1] == self.W, (
            f"self.num_wind[1]*self.window_size[1] == self.W, num_wind {self.num_wind}, window_size {self.window_size}, W {self.W}"
        )

    def set_and_check_patch(self):
        if self.D > 1:
            if self.num_patch is not None:
                self.patch_size = [
                    self.window_size[0] // self.num_patch[0],
                    self.window_size[1] // self.num_patch[1],
                    self.window_size[2] // self.num_patch[2],
                ]
                self.patch_size = [max(v, 1) for v in self.patch_size]
                self.num_patch = [
                    w // p for w, p in zip(self.window_size, self.patch_size, strict=False)
                ]
            else:
                self.num_patch = [
                    self.window_size[0] // self.patch_size[0],
                    self.window_size[1] // self.patch_size[1],
                    self.window_size[2] // self.patch_size[2],
                ]
                self.num_patch = [max(v, 1) for v in self.num_patch]
                self.patch_size = [
                    w // n for w, n in zip(self.window_size, self.num_patch, strict=False)
                ]

            assert (
                (self.patch_size[0] * self.num_patch[0] == self.window_size[0])
                and (self.patch_size[1] * self.num_patch[1] == self.window_size[1])
                and (self.patch_size[2] * self.num_patch[2] == self.window_size[2])
            ), (
                f"self.patch_size*self.num_patch == self.window_size, patch_size {self.patch_size}, num_patch {self.num_patch}, window_size {self.window_size}"
            )
        else:
            if self.num_patch is not None:
                self.patch_size = [
                    self.window_size[0] // self.num_patch[0],
                    self.window_size[1] // self.num_patch[1],
                ]
                self.patch_size = [max(v, 1) for v in self.patch_size]
                self.num_patch = [
                    w // p for w, p in zip(self.window_size, self.patch_size, strict=False)
                ]
            else:
                self.num_patch = [
                    self.window_size[0] // self.patch_size[0],
                    self.window_size[1] // self.patch_size[1],
                ]
                self.num_patch = [max(v, 1) for v in self.num_patch]
                self.patch_size = [
                    w // n for w, n in zip(self.window_size, self.num_patch, strict=False)
                ]

            assert (self.patch_size[0] * self.num_patch[0] == self.window_size[0]) and (
                self.patch_size[1] * self.num_patch[1] == self.window_size[1]
            ), (
                f"self.patch_size*self.num_patch == self.window_size, patch_size {self.patch_size}, num_patch {self.num_patch}, window_size {self.window_size}"
            )

    def validate_window_patch(self):
        assert self.window_size[0] * self.num_wind[0] == self.H, (
            "self.window_size[0]*self.num_wind[0] == self.H"
        )
        assert self.window_size[1] * self.num_wind[1] == self.W, (
            "self.window_size[1]*self.num_wind[1] == self.W"
        )
        assert self.patch_size[0] * self.num_patch[0] == self.window_size[0], (
            "self.patch_size[0]*self.num_patch[0] == self.window_size[0]"
        )
        assert self.patch_size[0] * self.num_patch[1] == self.window_size[1], (
            "self.patch_size[1]*self.num_patch[1] == self.window_size[1]"
        )

        if self.D > 1:
            assert self.window_size[2] * self.num_wind[2] == self.D, (
                "self.window_size[2]*self.num_wind[2] == self.D"
            )
            assert self.patch_size[2] * self.num_patch[2] == self.window_size[2], (
                "self.patch_size[2]*self.num_patch[2] == self.window_size[2]"
            )

    def __str__(self):
        res = create_generic_class_str(self)
        return res
