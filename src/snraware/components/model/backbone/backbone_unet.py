"""Backbone model - UNet architecture, with attention."""

from collections import OrderedDict

import torch
import torch.nn as nn
from colorama import Fore, Style

from snraware.components.model.attention import *
from snraware.components.model.backbone.backbone_base import (
    BackboneBase,
    DownSample,
    UpSample,
    set_window_patch_sizes_keep_num_window,
    set_window_patch_sizes_keep_window_size,
)
from snraware.components.model.backbone.blocks import *
from snraware.components.model.backbone.cells import *

__all__ = ["Unet"]


# -------------------------------------------------------------------------------------------------
class _unet_attention(nn.Module):
    """
    Unet attention scheme.

    The query q is from the lower resolution level [B, C_q, T, H, W];
    The value x is from the higher resolution level [B, C, T, H, W]

    Output is a gated value tensor [B, C, T, H, W]
    """

    def __init__(self, C_q=32, C=16) -> None:
        super().__init__()

        self.C_q = C_q
        self.C = C

        self.conv_query = Conv2DExt(
            in_channels=self.C_q,
            out_channels=self.C,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            channel_first=False,
        )
        self.conv_x = Conv2DExt(
            in_channels=self.C,
            out_channels=self.C,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            channel_first=False,
        )

        self.conv_gate = Conv2DExt(
            in_channels=self.C,
            out_channels=1,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            channel_first=False,
        )

    def forward(self, q, x):
        q = permute_to_B_T_C_H_W(q)
        x = permute_to_B_T_C_H_W(x)

        v = F.relu(self.conv_query(q) + self.conv_x(x), inplace=False)
        g = torch.sigmoid(self.conv_gate(v))  # [B, T, 1, H, W]

        y = x * g

        y = permute_to_B_C_T_H_W(y)

        return y


# -------------------------------------------------------------------------------------------------
# stcnnt unet


class Unet(BackboneBase):
    """
    This class implemented an Unet with maximal 5 down/upsample levels.

    The attention window_size and patch_size are in the unit of pixels and set for the top level resolution. For every downsample level, they are reduced by x2 to keep the number of windows roughly the same.
    """

    def __init__(self, config, input_feature_channels, H, W, D):
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - input_feature_channels (List[int]): list of ints indicating the number of channels in each input tensor.

        @args (from config):

            ---------------------------------------------------------------
            model specific arguments
            ---------------------------------------------------------------

            - C (int): number of operation channels, when resolution is reduced by x2, number of channels will increase by x2
            - num_resolution_levels (int): number of resolution levels; each deeper level will reduce spatial size by x2

            - block_str (str | list of strings): This string is the "Block string" to define the attention layers in a block. If a list of string is given, each string defines the attention structure for a resolution level. The last string is the bridge structure.

              During the configuration process, the block_str is parsed and converted into a list of block configurations for each resolution level.

            - use_unet_attention (bool): whether to use unet attention from lower resolution to higher resolution
        """
        super().__init__(config)

        if isinstance(input_feature_channels, list):
            C_in = input_feature_channels[-1]
        else:
            C_in = input_feature_channels

        assert config.num_resolution_levels <= 5 and config.num_resolution_levels >= 1, (
            "Maximal number of resolution levels is 5"
        )

        self.C = config.num_of_channels
        self.num_resolution_levels = config.num_resolution_levels
        self.use_unet_attention = config.use_unet_attention

        self.block_config = config.block_config
        if (
            len(self.block_config) < self.num_resolution_levels + 1
        ):  # num_resolution_levels+1 includes the bridge
            self.block_config.extend(
                [
                    config.block_config[-1]
                    for _i in range(self.num_resolution_levels + 1 - len(self.block_config))
                ]
            )

        # compute number of windows and patches
        self.num_wind = [
            H // config.block.cell.window_size[0],
            W // config.block.cell.window_size[1],
        ]
        self.num_patch = [
            config.block.cell.window_size[0] // config.block.cell.patch_size[0],
            config.block.cell.window_size[1] // config.block.cell.patch_size[1],
        ]

        if len(config.block.cell.window_size) == 3:
            self.num_wind.append(D // config.block.cell.window_size[2])
            self.num_patch.append(
                config.block.cell.window_size[2] // config.block.cell.patch_size[2]
            )

        model_config = dict()
        model_config["C_in"] = C_in
        model_config["C_out"] = self.C
        model_config["H"] = H
        model_config["W"] = W
        model_config["D"] = D
        model_config["num_wind"] = self.num_wind
        model_config["num_patch"] = self.num_patch
        model_config["block_config"] = config.block_config[0]
        model_config["window_size"] = self.config.block.cell.window_size
        model_config["patch_size"] = self.config.block.cell.patch_size

        print(
            f"{Fore.CYAN}=====> Create UNET, {self.config.block.cell_type}, {self.config.block.cell.attention_type} - {H} - {W} - {D} - {self.config.block.cell.window_size} - {self.config.block.cell.patch_size} - {self.config.block.block_dense_connection} <====={Style.RESET_ALL}"
        )

        window_sizes = []
        patch_sizes = []

        if self.num_resolution_levels >= 1:
            # define D0
            model_config["C_in"] = C_in
            model_config["C_out"] = self.C
            model_config["H"] = H
            model_config["W"] = W

            if self.config.block.cell.window_sizing_method == "keep_num_window":
                model_config = set_window_patch_sizes_keep_num_window(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    self.num_wind,
                    self.num_patch,
                    module_name="D0",
                )
            elif self.config.block.cell.window_sizing_method == "keep_window_size":
                model_config = set_window_patch_sizes_keep_window_size(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    model_config["window_size"],
                    model_config["patch_size"],
                    module_name="D0",
                )
            else:  # mixed
                model_config = set_window_patch_sizes_keep_num_window(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    self.num_wind,
                    self.num_patch,
                    module_name="D0",
                )

            window_sizes.append(model_config["window_size"])
            patch_sizes.append(model_config["patch_size"])

            model_config["block_config"] = self.block_config[0]
            self.D0 = Block(**self.get_block_parameters(model_config))

            self.down_0 = DownSample(
                N=1,
                C_in=model_config["C_out"],
                C_out=model_config["C_out"],
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )

        if self.num_resolution_levels >= 2:
            # define D1
            model_config["C_in"] = self.C
            model_config["C_out"] = 2 * self.C
            model_config["H"] = H // 2
            model_config["W"] = W // 2

            if self.config.block.cell.window_sizing_method == "keep_num_window":
                model_config = set_window_patch_sizes_keep_num_window(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    self.num_wind,
                    self.num_patch,
                    module_name="D1",
                )
            elif self.config.block.cell.window_sizing_method == "keep_window_size":
                model_config = set_window_patch_sizes_keep_window_size(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    window_sizes[0],
                    patch_sizes[0],
                    module_name="D1",
                )
            else:  # mixed
                model_config = set_window_patch_sizes_keep_window_size(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    window_sizes[0],
                    patch_sizes[0],
                    module_name="D1",
                )

            window_sizes.append(model_config["window_size"])
            patch_sizes.append(model_config["patch_size"])

            model_config["block_config"] = self.block_config[1]
            self.D1 = Block(**self.get_block_parameters(model_config))

            self.down_1 = DownSample(
                N=1,
                C_in=model_config["C_out"],
                C_out=model_config["C_out"],
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )

        if self.num_resolution_levels >= 3:
            # define D2
            model_config["C_in"] = 2 * self.C
            model_config["C_out"] = 4 * self.C
            model_config["H"] = H // 4
            model_config["W"] = W // 4

            if self.config.block.cell.window_sizing_method == "keep_num_window":
                model_config = set_window_patch_sizes_keep_num_window(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    self.num_wind,
                    self.num_patch,
                    module_name="D2",
                )
            elif self.config.block.cell.window_sizing_method == "keep_window_size":
                model_config = set_window_patch_sizes_keep_window_size(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    window_sizes[1],
                    patch_sizes[1],
                    module_name="D2",
                )
            else:  # mixed
                model_config = set_window_patch_sizes_keep_num_window(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    [v // 2 for v in self.num_wind],
                    self.num_patch,
                    module_name="D2",
                )

            window_sizes.append(model_config["window_size"])
            patch_sizes.append(model_config["patch_size"])

            model_config["block_config"] = self.block_config[2]
            self.D2 = Block(**self.get_block_parameters(model_config))

            self.down_2 = DownSample(
                N=1,
                C_in=model_config["C_out"],
                C_out=model_config["C_out"],
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )

        if self.num_resolution_levels >= 4:
            # define D3
            model_config["C_in"] = 4 * self.C
            model_config["C_out"] = 8 * self.C
            model_config["H"] = H // 8
            model_config["W"] = W // 8

            if self.config.block.cell.window_sizing_method == "keep_num_window":
                model_config = set_window_patch_sizes_keep_num_window(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    self.num_wind,
                    self.num_patch,
                    module_name="D3",
                )
            elif self.config.block.cell.window_sizing_method == "keep_window_size":
                model_config = set_window_patch_sizes_keep_window_size(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    window_sizes[2],
                    patch_sizes[2],
                    module_name="D3",
                )
            else:  # mixed
                model_config = set_window_patch_sizes_keep_window_size(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    window_sizes[2],
                    patch_sizes[2],
                    module_name="D3",
                )

            window_sizes.append(model_config["window_size"])
            patch_sizes.append(model_config["patch_size"])

            model_config["block_config"] = self.block_config[3]
            self.D3 = Block(**self.get_block_parameters(model_config))

            self.down_3 = DownSample(
                N=1,
                C_in=model_config["C_out"],
                C_out=model_config["C_out"],
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )

        if self.num_resolution_levels >= 5:
            # define D4
            model_config["C_in"] = 8 * self.C
            model_config["C_out"] = 16 * self.C
            model_config["H"] = H // 16
            model_config["W"] = W // 16

            if self.config.block.cell.window_sizing_method == "keep_num_window":
                model_config = set_window_patch_sizes_keep_num_window(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    self.num_wind,
                    self.num_patch,
                    module_name="D4",
                )
            elif self.config.block.cell.window_sizing_method == "keep_window_size":
                model_config = set_window_patch_sizes_keep_window_size(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    window_sizes[3],
                    patch_sizes[3],
                    module_name="D4",
                )
            else:  # mixed
                model_config = set_window_patch_sizes_keep_num_window(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    [v // 4 for v in self.num_wind],
                    self.num_patch,
                    module_name="D4",
                )

            window_sizes.append(model_config["window_size"])
            patch_sizes.append(model_config["patch_size"])

            model_config["block_config"] = self.block_config[4]
            self.D4 = Block(**self.get_block_parameters(model_config))

            self.down_4 = DownSample(
                N=1,
                C_in=model_config["C_out"],
                C_out=model_config["C_out"],
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )

        # define the bridge
        model_config["C_in"] = model_config["C_out"]
        model_config["block_config"] = self.block_config[-1]
        model_config["H"] //= 2
        model_config["W"] //= 2
        model_config = set_window_patch_sizes_keep_window_size(
            model_config,
            [model_config["H"], model_config["W"]],
            window_sizes[-1],
            patch_sizes[-1],
            module_name="bridge",
        )
        self.bridge = Block(**self.get_block_parameters(model_config))

        if self.num_resolution_levels >= 5:
            self.up_4 = UpSample(
                N=1,
                C_in=16 * self.C,
                C_out=16 * self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )
            if self.use_unet_attention:
                self.attention_4 = _unet_attention(C_q=16 * self.C, C=16 * self.C)

            model_config["C_in"] = 32 * self.C
            model_config["C_out"] = 8 * self.C
            model_config["H"] = H // 16
            model_config["W"] = W // 16
            model_config = set_window_patch_sizes_keep_window_size(
                model_config,
                [model_config["H"], model_config["W"]],
                window_sizes[3],
                patch_sizes[3],
                module_name="U4",
            )
            model_config["block_config"] = self.block_config[4]
            self.U4 = Block(**self.get_block_parameters(model_config))

        if self.num_resolution_levels >= 4:
            self.up_3 = UpSample(
                N=1,
                C_in=8 * self.C,
                C_out=8 * self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )
            if self.use_unet_attention:
                self.attention_3 = _unet_attention(C_q=8 * self.C, C=8 * self.C)

            model_config["C_in"] = 16 * self.C
            model_config["C_out"] = 4 * self.C
            model_config["H"] = H // 8
            model_config["W"] = W // 8
            model_config = set_window_patch_sizes_keep_window_size(
                model_config,
                [model_config["H"], model_config["W"]],
                window_sizes[2],
                patch_sizes[2],
                module_name="U3",
            )
            model_config["block_config"] = self.block_config[3]
            self.U3 = Block(**self.get_block_parameters(model_config))

        if self.num_resolution_levels >= 3:
            self.up_2 = UpSample(
                N=1,
                C_in=4 * self.C,
                C_out=4 * self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )
            if self.use_unet_attention:
                self.attention_2 = _unet_attention(C_q=4 * self.C, C=4 * self.C)

            model_config["C_in"] = 8 * self.C
            model_config["C_out"] = 2 * self.C
            model_config["H"] = H // 4
            model_config["W"] = W // 4
            model_config = set_window_patch_sizes_keep_window_size(
                model_config,
                [model_config["H"], model_config["W"]],
                window_sizes[1],
                patch_sizes[1],
                module_name="U2",
            )
            model_config["block_config"] = self.block_config[2]
            self.U2 = Block(**self.get_block_parameters(model_config))

        if self.num_resolution_levels >= 2:
            self.up_1 = UpSample(
                N=1,
                C_in=2 * self.C,
                C_out=2 * self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )
            if self.use_unet_attention:
                self.attention_1 = _unet_attention(C_q=2 * self.C, C=2 * self.C)

            model_config["C_in"] = 4 * self.C
            model_config["C_out"] = self.C
            model_config["H"] = H // 2
            model_config["W"] = W // 2
            model_config = set_window_patch_sizes_keep_window_size(
                model_config,
                [model_config["H"], model_config["W"]],
                window_sizes[0],
                patch_sizes[0],
                module_name="U1",
            )
            model_config["block_config"] = self.block_config[1]
            self.U1 = Block(**self.get_block_parameters(model_config))

        if self.num_resolution_levels >= 1:
            self.up_0 = UpSample(
                N=1,
                C_in=self.C,
                C_out=self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )
            if self.use_unet_attention:
                self.attention_0 = _unet_attention(C_q=self.C, C=self.C)

            model_config["C_in"] = 2 * self.C
            model_config["C_out"] = self.C
            model_config["H"] = H
            model_config["W"] = W
            model_config = set_window_patch_sizes_keep_window_size(
                model_config,
                [model_config["H"], model_config["W"]],
                window_sizes[0],
                patch_sizes[0],
                module_name="U0",
            )
            model_config["block_config"] = self.block_config[0]
            self.U0 = Block(**self.get_block_parameters(model_config))

        print(f"{Fore.CYAN}=====> Completed, create UNET <====={Style.RESET_ALL}")

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]

        _B, _D, _Cin, _H, _W = x.shape

        # first we go down the resolution ...
        if self.num_resolution_levels >= 1:
            x_0 = self.D0(x)
            x_d_0 = self.down_0(x_0)

        if self.num_resolution_levels >= 2:
            x_1 = self.D1(x_d_0)
            x_d_1 = self.down_1(x_1)

        if self.num_resolution_levels >= 3:
            x_2 = self.D2(x_d_1)
            x_d_2 = self.down_2(x_2)

        if self.num_resolution_levels >= 4:
            x_3 = self.D3(x_d_2)
            x_d_3 = self.down_3(x_3)

        if self.num_resolution_levels >= 5:
            x_4 = self.D4(x_d_3)
            x_d_4 = self.down_4(x_4)

        # now we go up the resolution ...
        if self.num_resolution_levels == 1:
            y_d_0 = self.bridge(x_d_0)
            y_0 = self.up_0(y_d_0)
            x_gated_0 = self.attention_0(q=y_0, x=x_0) if self.use_unet_attention else x_0
            y_hat = self.U0(torch.cat((x_gated_0, y_0), dim=1))

        if self.num_resolution_levels == 2:
            y_d_1 = self.bridge(x_d_1)
            y_1 = self.up_1(y_d_1)
            x_gated_1 = self.attention_1(q=y_1, x=x_1) if self.use_unet_attention else x_1
            y_d_0 = self.U1(torch.cat((x_gated_1, y_1), dim=1))

            y_0 = self.up_0(y_d_0)
            x_gated_0 = self.attention_0(q=y_0, x=x_0) if self.use_unet_attention else x_0
            y_hat = self.U0(torch.cat((x_gated_0, y_0), dim=1))

        if self.num_resolution_levels == 3:
            y_d_2 = self.bridge(x_d_2)
            y_2 = self.up_2(y_d_2)
            x_gated_2 = self.attention_2(q=y_2, x=x_2) if self.use_unet_attention else x_2
            y_d_1 = self.U2(torch.cat((x_gated_2, y_2), dim=1))

            y_1 = self.up_1(y_d_1)
            x_gated_1 = self.attention_1(q=y_1, x=x_1) if self.use_unet_attention else x_1
            y_d_0 = self.U1(torch.cat((x_gated_1, y_1), dim=1))

            y_0 = self.up_0(y_d_0)
            x_gated_0 = self.attention_0(q=y_0, x=x_0) if self.use_unet_attention else x_0
            y_hat = self.U0(torch.cat((x_gated_0, y_0), dim=1))

        if self.num_resolution_levels == 4:
            y_d_3 = self.bridge(x_d_3)
            y_3 = self.up_3(y_d_3)
            x_gated_3 = self.attention_3(q=y_3, x=x_3) if self.use_unet_attention else x_3
            y_d_2 = self.U3(torch.cat((x_gated_3, y_3), dim=1))

            y_2 = self.up_2(y_d_2)
            x_gated_2 = self.attention_2(q=y_2, x=x_2) if self.use_unet_attention else x_2
            y_d_1 = self.U2(torch.cat((x_gated_2, y_2), dim=1))

            y_1 = self.up_1(y_d_1)
            x_gated_1 = self.attention_1(q=y_1, x=x_1) if self.use_unet_attention else x_1
            y_d_0 = self.U1(torch.cat((x_gated_1, y_1), dim=1))

            y_0 = self.up_0(y_d_0)
            x_gated_0 = self.attention_0(q=y_0, x=x_0) if self.use_unet_attention else x_0
            y_hat = self.U0(torch.cat((x_gated_0, y_0), dim=1))

        if self.num_resolution_levels == 5:
            y_d_4 = self.bridge(x_d_4)
            y_4 = self.up_4(y_d_4)
            x_gated_4 = self.attention_4(q=y_4, x=x_4) if self.use_unet_attention else x_4
            y_d_3 = self.U4(torch.cat((x_gated_4, y_4), dim=1))

            y_3 = self.up_3(y_d_3)
            x_gated_3 = self.attention_3(q=y_3, x=x_3) if self.use_unet_attention else x_3
            y_d_2 = self.U3(torch.cat((x_gated_3, y_3), dim=1))

            y_2 = self.up_2(y_d_2)
            x_gated_2 = self.attention_2(q=y_2, x=x_2) if self.use_unet_attention else x_2
            y_d_1 = self.U2(torch.cat((x_gated_2, y_2), dim=1))

            y_1 = self.up_1(y_d_1)
            x_gated_1 = self.attention_1(q=y_1, x=x_1) if self.use_unet_attention else x_1
            y_d_0 = self.U1(torch.cat((x_gated_1, y_1), dim=1))

            y_0 = self.up_0(y_d_0)
            x_gated_0 = self.attention_0(q=y_0, x=x_0) if self.use_unet_attention else x_0
            y_hat = self.U0(torch.cat((x_gated_0, y_0), dim=1))

        return [y_hat]

    def __str__(self):
        return create_generic_class_str(
            obj=self,
            exclusion_list=[nn.Module, OrderedDict, Block, DownSample, UpSample, _unet_attention],
        )

    def get_number_of_output_channels(self):
        return int(self.config.num_of_channels)
