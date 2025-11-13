"""Convolution module to process 5D tensors."""

from .attention_modules import *


class ConvolutionModule(nn.Module):
    """
    Either 2d or 3d depending on the argument.
    Add residual connection if C_in == C_out.
    """

    def __init__(
        self,
        conv_type,
        C_in,
        C_out,
        H=128,
        W=128,
        D=32,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        norm_mode="instance2d",
        activation_func="prelu",
    ):
        """
        @args:
            - conv_type ("conv2d" or "conv3d"): the type of conv.
        """
        super().__init__()

        assert conv_type == "conv2d" or conv_type == "conv3d", (
            f"Conv type not implemented: {conv_type}"
        )

        self.C_in = C_in
        self.C_out = C_out

        if conv_type == "conv2d":
            if len(kernel_size) == 3:
                kernel_size = (kernel_size[0], kernel_size[1])
            if len(stride) == 3:
                stride = (stride[0], stride[1])
            if len(padding) == 3:
                padding = (padding[0], padding[1])

            self.conv = Conv2DExt(
                in_channels=C_in,
                out_channels=C_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                channel_first=True,
            )
        elif conv_type == "conv3d":
            if len(kernel_size) == 2:
                kernel_size = (*kernel_size, kernel_size[0])
            if len(stride) == 2:
                stride = (*stride, stride[0])
            if len(padding) == 2:
                padding = (*padding, padding[0])

            self.conv = Conv3DExt(
                in_channels=C_in,
                out_channels=C_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                channel_first=True,
            )
        else:
            raise NotImplementedError(f"Conv type not implemented: {conv_type}")

        self.n1 = create_norm(norm_mode=norm_mode, C=C_out, H=H, W=W, D=D)
        self.act_func = create_activation_func(name=activation_func)

    def forward(self, x):
        """
        @args:
            x ([B, C_in, T, H, W]): Input of a batch of time series.

        @rets:
            y ([B, C_out, T, H', W']): Output of the batch
        """
        res = self.act_func(self.n1(self.conv(x)))
        if self.C_in == self.C_out:
            res += x

        return res


# -------------------------------------------------------------------------------------------------
