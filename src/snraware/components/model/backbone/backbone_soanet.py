"""Backbone model - Stack of attention architecture."""

from collections import OrderedDict

import numpy as np
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

__all__ = ["SOAnet"]


# -------------------------------------------------------------------------------------
class SOAnet(BackboneBase):
    """This class implemented a multi-stage attention network, which is a stack of attention blocks. Every stage only has one block."""

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
            - self.num_stages (int): number of stages; each deeper stage will reduce spatial size by x2 and increase number of channels by x2.
        """
        super().__init__(config)

        if isinstance(input_feature_channels, list):
            C_in = input_feature_channels[-1]
        else:
            C_in = input_feature_channels

        self.C = config.num_of_channels
        self.num_stages = config.num_stages
        self.downsample = config.downsample

        self.block_config = config.block_config
        if len(self.block_config) < self.num_stages:
            self.block_config.extend(
                [config.block_config[-1] for _i in range(self.num_stages - len(self.block_config))]
            )

        # compute number of windows and patches
        self.num_wind = [
            H // self.config.block.cell.window_size[0],
            W // self.config.block.cell.window_size[1],
        ]
        self.num_patch = [
            self.config.block.cell.window_size[0] // self.config.block.cell.patch_size[0],
            self.config.block.cell.window_size[1] // self.config.block.cell.patch_size[1],
        ]

        if len(self.config.block.cell.window_size) == 3:
            self.num_wind.append(D // self.config.block.cell.window_size[2])
            self.num_patch.append(
                self.config.block.cell.window_size[2] // self.config.block.cell.patch_size[2]
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
            f"{Fore.CYAN}=====> Create SOANET, {self.config.block.cell_type}, {self.config.block.cell.attention_type} - {H} - {W} - {D} - {self.config.block.cell.window_size} - {self.config.block.cell.patch_size} - {self.config.block.cell_type} - {self.config.block.block_dense_connection} <====={Style.RESET_ALL}"
        )

        self.layers = nn.ModuleList()

        stage_C_in = C_in
        stage_C_out = self.C

        for stage in range(self.num_stages):
            module_name = f"Stage{stage}"

            model_config["C_in"] = stage_C_in
            model_config["C_out"] = stage_C_out
            model_config["H"] = H if not self.downsample else H // int(np.power(2, stage))
            model_config["W"] = W if not self.downsample else W // int(np.power(2, stage))

            if self.config.block.cell.window_sizing_method == "keep_num_window":
                model_config = set_window_patch_sizes_keep_num_window(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    self.num_wind,
                    self.num_patch,
                    module_name=module_name,
                )
            elif self.config.block.cell.window_sizing_method == "keep_window_size":
                model_config = set_window_patch_sizes_keep_window_size(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    model_config["window_size"],
                    model_config["patch_size"],
                    module_name=module_name,
                )
            else:  # mixed
                model_config = set_window_patch_sizes_keep_window_size(
                    model_config,
                    [model_config["H"], model_config["W"]],
                    model_config["window_size"],
                    model_config["patch_size"],
                    module_name=module_name,
                )

            model_config["block_config"] = self.block_config[stage]
            self.layers.append(Block(**self.get_block_parameters(model_config)))

            print(
                f"{Fore.BLUE}{module_name} -- {model_config['C_in']} to {model_config['C_out']} --- {[type(m).__name__ for m in model_config['block_config']]}{Style.RESET_ALL}"
            )

            if self.downsample:
                if stage < self.num_stages - 1:
                    self.downsample = DownSample(
                        N=1,
                        C_in=stage_C_out,
                        C_out=2 * stage_C_out,
                        use_interpolation=self.use_interpolation,
                        with_conv=self.with_conv,
                    )
                    self.layers.append(self.downsample)

                    print(
                        f"{Fore.BLUE}{module_name}, self.downsample -- {stage_C_out} to {2 * stage_C_out}{Style.RESET_ALL}"
                    )

                    # update the stage input and output channels
                    stage_C_in = 2 * stage_C_out
                    stage_C_out = 2 * stage_C_out
            else:
                stage_C_in = stage_C_out
                stage_C_out = 2 * stage_C_out

        print(f"{Fore.CYAN}=====> Completed, create SOANET <====={Style.RESET_ALL}")

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]

        y_level_outputs = []

        y_hat = x
        for a_layer in self.layers:
            y_hat = a_layer(y_hat)
            y_level_outputs.append(y_hat)

        return y_level_outputs

    def __str__(self):
        return create_generic_class_str(
            obj=self, exclusion_list=[nn.Module, OrderedDict, Block, DownSample, UpSample]
        )

    def get_number_of_output_channels(self):
        return int(self.config.num_of_channels * np.power(2, self.config.num_stages - 1))
