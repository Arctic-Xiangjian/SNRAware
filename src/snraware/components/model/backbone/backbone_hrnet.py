"""
Backbone model - High-resolution, HRNet architecture.

This file implements a HRNet design for the imaging backbone.
The input to the model is [B, C_in, T, H, W]. The output of the model is [B, C_out, D, H, W].
For every resolution level, the image size will be reduced by x2, with the number of channels increasing by x2.

Besides the aggregated output tensor, this backbone model also outputs the per-resolution-level feature maps as a list.

Please refer to the backbone section in the documentation for more details on the model structure.
"""

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

__all__ = ["HRnet"]


# -------------------------------------------------------------------------------------
class HRnet(BackboneBase):
    """This class implemented the HRnet with maximal 5 resolution levels."""

    def __init__(self, config, input_feature_channels, H, W, D):
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - input_feature_channels (List[int] or int): number of channels in each input tensor
            - H, W, D (int): H, W, and depth of the input tensor [B, input_feature_channels, D, H, W].

        @args (from config):
            ---------------------------------------------------------------
            model specific arguments
            ---------------------------------------------------------------

            - C (int): number of operation channels; the input channel is first scaled up to C channels. The downsample will double the number of operation channels,
            and upsample will halve the number of operation channels.

            - num_resolution_levels (int): number of resolution levels; each deeper level will reduce spatial size by x2

            - block_str (str | list of strings): This string is the "Block string" to define the attention layers in a block.
                If a list of string is given, each string defines the attention structure for a resolution level. For example,
                ["T1L1G1", "T1L1G1T1L1G1"] means the blocks at the first resolution level has three attention layers, and the second resolution level has six attention layers.

                If one string is given, e.g. ["T1L1G1"], it means all resolution levels have the same attention structure, i.e. three attention layers.

                During the configuration process, the block_str is parsed and converted into a list of block configurations for each resolution level.

            ---------------------------------------------------------------
        @outputs:
            A list of tensors for each resolution level; the last tensor is the aggregation of all resolution levels.
        """
        super().__init__(config)

        # set up the model specific parameters
        self.C = config.num_of_channels

        self.num_resolution_levels = config.num_resolution_levels
        assert self.num_resolution_levels <= 5 and self.num_resolution_levels >= 2, (
            "Maximal number of resolution levels is 5"
        )

        self.block_config = config.block_config
        if len(self.block_config) < self.num_resolution_levels:
            self.block_config.extend(
                [
                    config.block_config[-1]
                    for _i in range(self.num_resolution_levels - len(self.block_config))
                ]
            )

        if isinstance(input_feature_channels, list):
            C_in = input_feature_channels[-1]
        else:
            C_in = input_feature_channels

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

        # dict to store the parameters used for making blocks
        # these values can change from block to block
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

        window_sizes = []
        patch_sizes = []

        print(
            f"{Fore.CYAN}=====> Create HRNET, {self.config.block.cell_type}, {self.config.block.cell.attention_type} - {H} - {W} - {D} - {self.config.block.cell.window_size} - {self.config.block.cell.patch_size} - {self.config.block.block_dense_connection} <====={Style.RESET_ALL}"
        )

        if self.num_resolution_levels >= 1:
            # define B00
            self.B00, model_config = self.add_a_block(
                level=0,
                block_name="B00",
                model_config=model_config,
                c_in=C_in,
                c_out=self.C,
                h=H,
                w=W,
                window_sizes=model_config["window_size"],
                patch_sizes=model_config["patch_size"],
                block_config=self.block_config[0],
                window_sizing_method=self.config.block.cell.window_sizing_method,
            )
            window_sizes.append(model_config["window_size"])
            patch_sizes.append(model_config["patch_size"])

            # output stage 0
            self.output_B0, model_config = self.add_a_block(
                level=0,
                block_name="output_0",
                model_config=model_config,
                c_in=self.C,
                c_out=self.C,
                h=H,
                w=W,
                window_sizes=window_sizes[-1],
                patch_sizes=patch_sizes[-1],
                block_config=self.block_config[0],
                window_sizing_method="keep_window_size",
            )

        if self.num_resolution_levels >= 2:
            # define B01
            self.B01, model_config = self.add_a_block(
                level=0,
                block_name="B01",
                model_config=model_config,
                c_in=self.C,
                c_out=self.C,
                h=H,
                w=W,
                window_sizes=window_sizes[0],
                patch_sizes=patch_sizes[0],
                block_config=self.block_config[0],
                window_sizing_method="keep_window_size",
            )

            # define B11
            self.B11, model_config = self.add_a_block(
                level=1,
                block_name="B11",
                model_config=model_config,
                c_in=2 * self.C,
                c_out=2 * self.C,
                h=H // 2,
                w=W // 2,
                window_sizes=window_sizes[0],
                patch_sizes=patch_sizes[0],
                block_config=self.block_config[1],
                window_sizing_method=self.config.block.cell.window_sizing_method,
            )
            window_sizes.append(model_config["window_size"])
            patch_sizes.append(model_config["patch_size"])

            # define down sample
            self.down_00_11 = DownSample(
                N=1,
                C_in=self.C,
                C_out=2 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )

            # define output B1
            self.output_B1, model_config = self.add_a_block(
                level=1,
                block_name="output_1",
                model_config=model_config,
                c_in=2 * self.C,
                c_out=2 * self.C,
                h=H // 2,
                w=W // 2,
                window_sizes=window_sizes[-1],
                patch_sizes=patch_sizes[-1],
                block_config=self.block_config[1],
                window_sizing_method="keep_window_size",
            )

        if self.num_resolution_levels >= 3:
            # define B02
            self.B02, model_config = self.add_a_block(
                level=0,
                block_name="B02",
                model_config=model_config,
                c_in=self.C,
                c_out=self.C,
                h=H,
                w=W,
                window_sizes=window_sizes[0],
                patch_sizes=patch_sizes[0],
                block_config=self.block_config[0],
                window_sizing_method="keep_window_size",
            )

            # define B12
            self.B12, model_config = self.add_a_block(
                level=1,
                block_name="B12",
                model_config=model_config,
                c_in=2 * self.C,
                c_out=2 * self.C,
                h=H // 2,
                w=W // 2,
                window_sizes=window_sizes[1],
                patch_sizes=patch_sizes[1],
                block_config=self.block_config[1],
                window_sizing_method="keep_window_size",
            )

            # define B22
            self.B22, model_config = self.add_a_block(
                level=2,
                block_name="B22",
                model_config=model_config,
                c_in=4 * self.C,
                c_out=4 * self.C,
                h=H // 4,
                w=W // 4,
                window_sizes=window_sizes[1],
                patch_sizes=patch_sizes[1],
                block_config=self.block_config[2],
                window_sizing_method=self.config.block.cell.window_sizing_method,
            )
            window_sizes.append(model_config["window_size"])
            patch_sizes.append(model_config["patch_size"])

            # define down sample
            self.down_01_12 = DownSample(
                N=1,
                C_in=self.C,
                C_out=2 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
            self.down_01_22 = DownSample(
                N=2,
                C_in=self.C,
                C_out=4 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
            self.down_11_22 = DownSample(
                N=1,
                C_in=2 * self.C,
                C_out=4 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )

            # define output B2
            self.output_B2, model_config = self.add_a_block(
                level=2,
                block_name="output_B2",
                model_config=model_config,
                c_in=4 * self.C,
                c_out=4 * self.C,
                h=H // 4,
                w=W // 4,
                window_sizes=window_sizes[-1],
                patch_sizes=patch_sizes[-1],
                block_config=self.block_config[2],
                window_sizing_method="keep_window_size",
            )

        if self.num_resolution_levels >= 4:
            # define B03
            self.B03, model_config = self.add_a_block(
                level=0,
                block_name="B03",
                model_config=model_config,
                c_in=self.C,
                c_out=self.C,
                h=H,
                w=W,
                window_sizes=window_sizes[0],
                patch_sizes=patch_sizes[0],
                block_config=self.block_config[0],
                window_sizing_method="keep_window_size",
            )

            # define B13
            self.B13, model_config = self.add_a_block(
                level=1,
                block_name="B13",
                model_config=model_config,
                c_in=2 * self.C,
                c_out=2 * self.C,
                h=H // 2,
                w=W // 2,
                window_sizes=window_sizes[1],
                patch_sizes=patch_sizes[1],
                block_config=self.block_config[1],
                window_sizing_method="keep_window_size",
            )

            # define B23
            self.B23, model_config = self.add_a_block(
                level=2,
                block_name="B23",
                model_config=model_config,
                c_in=4 * self.C,
                c_out=4 * self.C,
                h=H // 4,
                w=W // 4,
                window_sizes=window_sizes[2],
                patch_sizes=patch_sizes[2],
                block_config=self.block_config[2],
                window_sizing_method="keep_window_size",
            )

            # define B33
            self.B33, model_config = self.add_a_block(
                level=3,
                block_name="B33",
                model_config=model_config,
                c_in=8 * self.C,
                c_out=8 * self.C,
                h=H // 8,
                w=W // 8,
                window_sizes=window_sizes[2],
                patch_sizes=patch_sizes[2],
                block_config=self.block_config[3],
                window_sizing_method=self.config.block.cell.window_sizing_method,
            )
            window_sizes.append(model_config["window_size"])
            patch_sizes.append(model_config["patch_size"])

            # define down sample
            self.down_02_13 = DownSample(
                N=1,
                C_in=self.C,
                C_out=2 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
            self.down_02_23 = DownSample(
                N=2,
                C_in=self.C,
                C_out=4 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
            self.down_02_33 = DownSample(
                N=3,
                C_in=self.C,
                C_out=8 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
            self.down_12_23 = DownSample(
                N=1,
                C_in=2 * self.C,
                C_out=4 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
            self.down_12_33 = DownSample(
                N=2,
                C_in=2 * self.C,
                C_out=8 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
            self.down_22_33 = DownSample(
                N=1,
                C_in=4 * self.C,
                C_out=8 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )

            # define output B3
            self.output_B3, model_config = self.add_a_block(
                level=3,
                block_name="output_B3",
                model_config=model_config,
                c_in=8 * self.C,
                c_out=8 * self.C,
                h=H // 8,
                w=W // 8,
                window_sizes=window_sizes[-1],
                patch_sizes=patch_sizes[-1],
                block_config=self.block_config[3],
                window_sizing_method="keep_window_size",
            )

        if self.num_resolution_levels >= 5:
            # define B04
            self.B04, model_config = self.add_a_block(
                level=0,
                block_name="B04",
                model_config=model_config,
                c_in=self.C,
                c_out=self.C,
                h=H,
                w=W,
                window_sizes=window_sizes[0],
                patch_sizes=patch_sizes[0],
                block_config=self.block_config[0],
                window_sizing_method="keep_window_size",
            )

            # define B14
            self.B14, model_config = self.add_a_block(
                level=1,
                block_name="B14",
                model_config=model_config,
                c_in=2 * self.C,
                c_out=2 * self.C,
                h=H // 2,
                w=W // 2,
                window_sizes=window_sizes[1],
                patch_sizes=patch_sizes[1],
                block_config=self.block_config[1],
                window_sizing_method="keep_window_size",
            )

            # define B24
            self.B24, model_config = self.add_a_block(
                level=2,
                block_name="B24",
                model_config=model_config,
                c_in=4 * self.C,
                c_out=4 * self.C,
                h=H // 4,
                w=W // 4,
                window_sizes=window_sizes[2],
                patch_sizes=patch_sizes[2],
                block_config=self.block_config[2],
                window_sizing_method="keep_window_size",
            )

            # define B34
            self.B34, model_config = self.add_a_block(
                level=3,
                block_name="B34",
                model_config=model_config,
                c_in=8 * self.C,
                c_out=8 * self.C,
                h=H // 8,
                w=W // 8,
                window_sizes=window_sizes[3],
                patch_sizes=patch_sizes[3],
                block_config=self.block_config[3],
                window_sizing_method="keep_window_size",
            )

            # define B44
            self.B44, model_config = self.add_a_block(
                level=4,
                block_name="B44",
                model_config=model_config,
                c_in=16 * self.C,
                c_out=16 * self.C,
                h=H // 16,
                w=W // 16,
                window_sizes=window_sizes[2],
                patch_sizes=patch_sizes[2],
                block_config=self.block_config[4],
                window_sizing_method=self.config.block.cell.window_sizing_method,
            )
            window_sizes.append(model_config["window_size"])
            patch_sizes.append(model_config["patch_size"])

            # define down sample
            self.down_03_14 = DownSample(
                N=1,
                C_in=self.C,
                C_out=2 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
            self.down_03_24 = DownSample(
                N=2,
                C_in=self.C,
                C_out=4 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
            self.down_03_34 = DownSample(
                N=3,
                C_in=self.C,
                C_out=8 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
            self.down_03_44 = DownSample(
                N=4,
                C_in=self.C,
                C_out=16 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )

            self.down_13_24 = DownSample(
                N=1,
                C_in=2 * self.C,
                C_out=4 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
            self.down_13_34 = DownSample(
                N=2,
                C_in=2 * self.C,
                C_out=8 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
            self.down_13_44 = DownSample(
                N=3,
                C_in=2 * self.C,
                C_out=16 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )

            self.down_23_34 = DownSample(
                N=1,
                C_in=4 * self.C,
                C_out=8 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
            self.down_23_44 = DownSample(
                N=2,
                C_in=4 * self.C,
                C_out=16 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )

            self.down_33_44 = DownSample(
                N=1,
                C_in=8 * self.C,
                C_out=16 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )

            # define output B4
            self.output_B4, model_config = self.add_a_block(
                level=4,
                block_name="output_B4",
                model_config=model_config,
                c_in=16 * self.C,
                c_out=16 * self.C,
                h=H // 16,
                w=W // 16,
                window_sizes=window_sizes[-1],
                patch_sizes=patch_sizes[-1],
                block_config=self.block_config[4],
                window_sizing_method="keep_window_size",
            )

        # fusion stage
        if self.num_resolution_levels >= 2:
            self.down_0_1 = DownSample(
                N=1,
                C_in=self.C,
                C_out=2 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
        if self.num_resolution_levels >= 3:
            self.down_0_2 = DownSample(
                N=2,
                C_in=self.C,
                C_out=4 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
        if self.num_resolution_levels >= 4:
            self.down_0_3 = DownSample(
                N=3,
                C_in=self.C,
                C_out=8 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
        if self.num_resolution_levels >= 5:
            self.down_0_4 = DownSample(
                N=4,
                C_in=self.C,
                C_out=16 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )

        if self.num_resolution_levels >= 3:
            self.down_1_2 = DownSample(
                N=1,
                C_in=2 * self.C,
                C_out=4 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
        if self.num_resolution_levels >= 4:
            self.down_1_3 = DownSample(
                N=2,
                C_in=2 * self.C,
                C_out=8 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
        if self.num_resolution_levels >= 5:
            self.down_1_4 = DownSample(
                N=3,
                C_in=2 * self.C,
                C_out=16 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )

        if self.num_resolution_levels >= 4:
            self.down_2_3 = DownSample(
                N=1,
                C_in=4 * self.C,
                C_out=8 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )
        if self.num_resolution_levels >= 5:
            self.down_2_4 = DownSample(
                N=2,
                C_in=4 * self.C,
                C_out=16 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )

        if self.num_resolution_levels >= 5:
            self.down_3_4 = DownSample(
                N=1,
                C_in=8 * self.C,
                C_out=16 * self.C,
                use_interpolation=self.use_interpolation,
                with_conv=self.with_conv,
            )

        if self.num_resolution_levels >= 2:
            self.up_1_0 = UpSample(
                N=1,
                C_in=2 * self.C,
                C_out=self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )
        if self.num_resolution_levels >= 3:
            self.up_2_0 = UpSample(
                N=2,
                C_in=4 * self.C,
                C_out=self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )
        if self.num_resolution_levels >= 4:
            self.up_3_0 = UpSample(
                N=3,
                C_in=8 * self.C,
                C_out=self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )
        if self.num_resolution_levels >= 5:
            self.up_4_0 = UpSample(
                N=4,
                C_in=16 * self.C,
                C_out=self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )

        if self.num_resolution_levels >= 3:
            self.up_2_1 = UpSample(
                N=1,
                C_in=4 * self.C,
                C_out=2 * self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )
        if self.num_resolution_levels >= 4:
            self.up_3_1 = UpSample(
                N=2,
                C_in=8 * self.C,
                C_out=2 * self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )
        if self.num_resolution_levels >= 5:
            self.up_4_1 = UpSample(
                N=3,
                C_in=16 * self.C,
                C_out=2 * self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )

        if self.num_resolution_levels >= 4:
            self.up_3_2 = UpSample(
                N=1,
                C_in=8 * self.C,
                C_out=4 * self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )
        if self.num_resolution_levels >= 5:
            self.up_4_2 = UpSample(
                N=2,
                C_in=16 * self.C,
                C_out=4 * self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )

        if self.num_resolution_levels >= 5:
            self.up_4_3 = UpSample(
                N=1,
                C_in=16 * self.C,
                C_out=8 * self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )

        if self.num_resolution_levels >= 2:
            self.up_1 = UpSample(
                N=1,
                C_in=2 * self.C,
                C_out=2 * self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )
        if self.num_resolution_levels >= 3:
            self.up_2 = UpSample(
                N=2,
                C_in=4 * self.C,
                C_out=4 * self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )
        if self.num_resolution_levels >= 4:
            self.up_3 = UpSample(
                N=3,
                C_in=8 * self.C,
                C_out=8 * self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )
        if self.num_resolution_levels >= 5:
            self.up_4 = UpSample(
                N=4,
                C_in=16 * self.C,
                C_out=16 * self.C,
                method=self.config.block.cell.upsample_method,
                with_conv=self.with_conv,
            )

        print(f"{Fore.CYAN}=====> Completed, create HRNET <====={Style.RESET_ALL}")

    def add_a_block(
        self,
        level,
        block_name,
        model_config,
        c_in,
        c_out,
        h,
        w,
        window_sizes,
        patch_sizes,
        block_config,
        window_sizing_method,
    ):
        model_config["C_in"] = c_in
        model_config["C_out"] = c_out
        model_config["H"] = h
        model_config["W"] = w

        if window_sizing_method == "keep_num_window":
            model_config = set_window_patch_sizes_keep_num_window(
                model_config,
                [model_config["H"], model_config["W"]],
                self.num_wind,
                self.num_patch,
                module_name=block_name,
            )
        elif window_sizing_method == "keep_window_size":
            model_config = set_window_patch_sizes_keep_window_size(
                model_config,
                [model_config["H"], model_config["W"]],
                window_sizes,
                patch_sizes,
                module_name=block_name,
            )
        else:  # mixed
            if level == 0:
                model_config = set_window_patch_sizes_keep_num_window(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    self.num_wind,
                    self.num_patch,
                    module_name=block_name,
                )
            elif level == 1:
                model_config = set_window_patch_sizes_keep_window_size(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    window_sizes,
                    patch_sizes,
                    module_name=block_name,
                )
            elif level == 2:
                model_config = set_window_patch_sizes_keep_num_window(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    [v // 2 for v in self.num_wind],
                    self.num_patch,
                    module_name=block_name,
                )
            elif level == 3:
                model_config = set_window_patch_sizes_keep_window_size(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    window_sizes,
                    patch_sizes,
                    module_name=block_name,
                )
            else:  # level == 4
                model_config = set_window_patch_sizes_keep_num_window(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    [v // 4 for v in self.num_wind],
                    self.num_patch,
                    module_name=block_name,
                )

        model_config["block_config"] = block_config

        print(
            f"{Fore.BLUE}{block_name} -- {model_config['C_in']} to {model_config['C_out']} --- {[type(m).__name__ for m in model_config['block_config']]}{Style.RESET_ALL}"
        )

        params = self.get_block_parameters(model_config)
        m = Block(**params)

        return m, model_config

    def forward(self, x):
        """
        @args:
            - x (list of 5D torch.Tensor): the input image, [B, Cin, D, H, W].

        @rets:
            - y_hat (5D torch.Tensor): aggregated output tensor
            - y_level_outputs (Tuple): tuple of tensor for every resolution level
        """
        if isinstance(x, list):
            x = x[-1]

        _B, _D, _Cin, _H, _W = x.shape

        y_hat = None
        y_level_outputs = None

        # compute the block outputs
        if self.num_resolution_levels >= 1:
            x_00 = self.B00(x)

        if self.num_resolution_levels >= 2:
            x_01 = self.B01(x_00)
            x_11 = self.B11(self.down_00_11(x_00))

        if self.num_resolution_levels >= 3:
            x_02 = self.B02(x_01 + x_00)

            x_12 = self.B12(x_11 + self.down_01_12(x_01))

            x_22 = self.B22(self.down_11_22(x_11) + self.down_01_22(x_01))

        if self.num_resolution_levels >= 4:
            x_03 = self.B03(x_02 + x_00 + x_01)

            x_13 = self.B13(x_12 + self.down_02_13(x_02) + x_11)

            x_23 = self.B23(x_22 + self.down_12_23(x_12) + self.down_02_23(x_02))

            x_33 = self.B33(self.down_22_33(x_22) + self.down_12_33(x_12) + self.down_02_33(x_02))

        if self.num_resolution_levels >= 5:
            x_04 = self.B04(x_03 + x_02 + x_01 + x_00)

            x_14 = self.B14(x_13 + self.down_03_14(x_03) + x_12 + x_11)

            x_24 = self.B24(x_23 + self.down_13_24(x_13) + self.down_03_24(x_03) + x_22)

            x_34 = self.B34(
                x_33 + self.down_23_34(x_23) + self.down_13_34(x_13) + self.down_03_34(x_03)
            )

            x_44 = self.B44(
                self.down_33_44(x_33)
                + self.down_23_44(x_23)
                + self.down_13_44(x_13)
                + self.down_03_44(x_03)
            )

        if self.num_resolution_levels == 1:
            y_hat_0 = self.output_B0(x_00)
            y_hat = y_hat_0
            y_level_outputs = [y_hat_0]

        if self.num_resolution_levels == 2:
            y_hat_0 = x_01 + self.up_1_0(x_11)
            y_hat_1 = x_11 + self.down_0_1(x_01)

            y_hat_0 = self.output_B0(y_hat_0)
            y_hat_1 = self.output_B1(y_hat_1)

            y_hat = torch.cat((y_hat_0, self.up_1(y_hat_1)), dim=1)

            y_level_outputs = [y_hat_0, y_hat_1]

        if self.num_resolution_levels == 3:
            y_hat_0 = x_02 + self.up_1_0(x_12) + self.up_2_0(x_22)
            y_hat_1 = self.down_0_1(x_02) + x_12 + self.up_2_1(x_22)
            y_hat_2 = self.down_0_2(x_02) + self.down_1_2(x_12) + x_22

            y_hat_0 = self.output_B0(y_hat_0)
            y_hat_1 = self.output_B1(y_hat_1)
            y_hat_2 = self.output_B2(y_hat_2)

            y_hat = torch.cat((y_hat_0, self.up_1(y_hat_1), self.up_2(y_hat_2)), dim=1)
            y_level_outputs = [y_hat_0, y_hat_1, y_hat_2]

        if self.num_resolution_levels == 4:
            y_hat_0 = x_03 + self.up_1_0(x_13) + self.up_2_0(x_23) + self.up_3_0(x_33)
            y_hat_1 = self.down_0_1(x_03) + x_13 + self.up_2_1(x_23) + self.up_3_1(x_33)
            y_hat_2 = self.down_0_2(x_03) + self.down_1_2(x_13) + x_23 + self.up_3_2(x_33)
            y_hat_3 = self.down_0_3(x_03) + self.down_1_3(x_13) + self.down_2_3(x_23) + x_33

            y_hat_0 = self.output_B0(y_hat_0)
            y_hat_1 = self.output_B1(y_hat_1)
            y_hat_2 = self.output_B2(y_hat_2)
            y_hat_3 = self.output_B3(y_hat_3)

            y_hat = torch.cat(
                (y_hat_0, self.up_1(y_hat_1), self.up_2(y_hat_2), self.up_3(y_hat_3)), dim=1
            )

            y_level_outputs = [y_hat_0, y_hat_1, y_hat_2, y_hat_3]

        if self.num_resolution_levels == 5:
            y_hat_0 = (
                x_04
                + self.up_1_0(x_14)
                + self.up_2_0(x_24)
                + self.up_3_0(x_34)
                + self.up_4_0(x_44)
            )
            y_hat_1 = (
                self.down_0_1(x_04)
                + x_14
                + self.up_2_1(x_24)
                + self.up_3_1(x_34)
                + self.up_4_1(x_44)
            )
            y_hat_2 = (
                self.down_0_2(x_04)
                + self.down_1_2(x_14)
                + x_24
                + self.up_3_2(x_34)
                + self.up_4_2(x_44)
            )
            y_hat_3 = (
                self.down_0_3(x_04)
                + self.down_1_3(x_14)
                + self.down_2_3(x_24)
                + x_34
                + self.up_4_3(x_44)
            )
            y_hat_4 = (
                self.down_0_4(x_04)
                + self.down_1_4(x_14)
                + self.down_2_4(x_24)
                + self.down_3_4(x_34)
                + x_44
            )

            y_hat_0 = self.output_B0(y_hat_0)
            y_hat_1 = self.output_B1(y_hat_1)
            y_hat_2 = self.output_B2(y_hat_2)
            y_hat_3 = self.output_B3(y_hat_3)
            y_hat_4 = self.output_B4(y_hat_4)

            y_hat = torch.cat(
                (
                    y_hat_0,
                    self.up_1(y_hat_1),
                    self.up_2(y_hat_2),
                    self.up_3(y_hat_3),
                    self.up_4(y_hat_4),
                ),
                dim=1,
            )

            y_level_outputs = [y_hat_0, y_hat_1, y_hat_2, y_hat_3, y_hat_4]

        return [*y_level_outputs, y_hat]

    def __str__(self):
        return create_generic_class_str(
            obj=self, exclusion_list=[nn.Module, OrderedDict, Block, DownSample, UpSample]
        )

    def get_number_of_output_channels(self):
        return int(
            self.config.num_of_channels
            * sum([np.power(2, k) for k in range(self.config.num_resolution_levels)])
        )
