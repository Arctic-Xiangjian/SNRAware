"""
Post components.

All modules take in a 5D tensor [B, C, T, H, W] and output a 2D logits [B, num_classes]
"""

import torch
import torch.nn as nn

from snraware.components.model import Conv2DExt

# -------------------------------------------------------------------------------------------------


class PreConv2D(nn.Module):
    """A conv2d head to convert C channels to Cout channels."""

    def __init__(self, C, C_out, bias=True):
        super().__init__()

        self.layer = Conv2DExt(
            in_channels=C,
            out_channels=C_out,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            padding_mode="reflect",
            bias=bias,
            channel_first=True,
        )

    def forward(self, x):
        return self.layer(x)


# -------------------------------------------------------------------------------------------------


class PoolLinear(nn.Module):
    def __init__(self, config, C, num_classes, add_tanh=False):
        """
        For an input [B, C, D, H, W] tensor, first avgpool over [D, H, W], then convert [B, C, 1] tensor to [B, num_classes] logits.

        add_tanh (bool): whether to add tanh after linear layer.
        """
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Linear(C, num_classes))
        if add_tanh:
            self.head.append(nn.Tanh())

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.head(x)
        return x


# -------------------------------------------------------------------------------------------------


class SimpleConv2d(nn.Module):
    def __init__(self, config, C, num_classes):
        """
        Takes in features [B, C, D, H, W] from backbone model and produces an output of [B, num_classes, D, H, W]
        num_classes: the number of segmentation class (e.g. 1 for binary segmentation, N for N-target segmentation).
        """
        super().__init__()
        self.conv2d = Conv2DExt(
            in_channels=C,
            out_channels=num_classes,
            kernel_size=[3, 3],
            padding=[1, 1],
            stride=[1, 1],
            bias=True,
            channel_first=True,
        )

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]
        x = self.conv2d(x)
        return x


# -------------------------------------------------------------------------------------------------
