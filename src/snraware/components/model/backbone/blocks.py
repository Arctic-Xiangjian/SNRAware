"""
A block contains a set of cells. Block structure is configurable by the 'block string'.
For example, 'L1T1G1' means to configure with a local attention (L1) with mixer (1 after 'L'), followed by a temporal attention with mixer (T1)
and a global attention with mixer (G1).

For more details on the configuration, please refer to the Cell and Block section of the documentation.

"""

from snraware.components.model.attention import *
from snraware.components.model.config import (
    ConvolutionConfig,
    SpatialGlobal3DConfig,
    SpatialGlobalConfig,
    SpatialLocal3DConfig,
    SpatialLocalConfig,
    SpatialViTConfig,
    Swin3DConfig,
    TemporalAttentionConfig,
    ViT3DConfig,
)
from snraware.components.setup.status import create_generic_class_str

from .cells import *

__all__ = ["Block"]

# -------------------------------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, block_config, C_in, C_out, H, W, D, config):
        """
        Block of multiple transformer cells stacked on top of each other.

        @args:
            - block_config (list[CellTypeVariant]): block configuration defines the structure of the block.
                every element of block config is for an attention types defined in the snraware.components.model.config.

            - C_in (int): number of input channels.
            - C_out (int): number of output channels.
            - H (int): height of the input feature map.
            - W (int): width of the input feature map.
            - D (int): depth of the input feature map, default is 1 for 2D data.
            - config: configuration object containing additional parameters
        """
        super().__init__()

        self.block_config = block_config
        self.C_in = C_in
        self.C_out = C_out
        self.H = H
        self.W = W
        self.D = D
        self.config = config

        window_size = self.config.cell.window_size
        patch_size = self.config.cell.patch_size

        self.window_size = window_size
        self.patch_size = patch_size

        num_wind = [
            max(H // window_size[0], 1),
            max(W // window_size[1], 1),
            max(D // window_size[2], 1),
        ]
        num_patch = [max(a // b, 1) for a, b in zip(window_size, patch_size, strict=False)]

        self.num_wind = num_wind
        self.num_patch = num_patch

        self.cell_type = self.config.cell_type
        self.block_dense_connection = self.config.block_dense_connection

        # fixed parameters
        kernel_size = [3, 3, 3]
        stride = [1, 1, 1]
        padding = [1, 1, 1]
        stride_s = [1, 1, 1]
        mixer_kernel_size = [3, 3, 3]
        mixer_stride = [1, 1, 1]
        mixer_padding = [1, 1, 1]
        is_causal = False
        shuffle_in_window = False
        use_flash_attention = False
        with_mixer = True

        self.cells = []

        for i, atten_config in enumerate(block_config):
            if isinstance(atten_config, TemporalAttentionConfig):
                att_type = "temporal"
            elif isinstance(atten_config, SpatialLocalConfig):
                att_type = "local"
            elif isinstance(atten_config, SpatialGlobalConfig):
                att_type = "global"
                shuffle_in_window = atten_config.shuffle_in_window
            elif isinstance(atten_config, ConvolutionConfig):
                if atten_config.conv_type == "conv2d":
                    att_type = "conv2d"
                elif atten_config.conv_type == "conv3d":
                    att_type = "conv3d"
                else:
                    raise ValueError(f"Incorrect conv_type: {atten_config.conv_type}")
            elif isinstance(atten_config, SpatialLocal3DConfig):
                att_type = "local_3d"
            elif isinstance(atten_config, SpatialGlobal3DConfig):
                att_type = "global_3d"
                shuffle_in_window = atten_config.shuffle_in_window
            elif isinstance(atten_config, SpatialViTConfig):
                att_type = "vit_2d"
            elif isinstance(atten_config, ViT3DConfig):
                att_type = "vit_3d"
            elif isinstance(atten_config, Swin3DConfig):
                if atten_config.shifted_window_attention:
                    att_type = "swin_3d_shifted"
                else:
                    att_type = "swin_3d"
            else:
                raise ValueError(f"Incorrect atten_config: {atten_config}")

            C = C_in if i == 0 else C_out

            if self.cell_type.lower() == "sequential":
                self.cells.append(
                    (
                        f"cell_{i}",
                        Cell(
                            C_in=C,
                            C_out=C_out,
                            H=H,
                            W=W,
                            D=D,
                            att_mode=att_type,
                            attention_type=self.config.cell.attention_type,
                            mixer_type=self.config.cell.mixer_type,
                            window_size=window_size,
                            patch_size=patch_size,
                            num_wind=num_wind,
                            num_patch=num_patch,
                            is_causal=is_causal,
                            n_head=self.config.cell.n_head,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            stride_s=stride_s,
                            stride_t=self.config.cell.temporal.stride_qk,
                            activation_func=self.config.cell.activation_func,
                            mixer_kernel_size=mixer_kernel_size,
                            mixer_stride=mixer_stride,
                            mixer_padding=mixer_padding,
                            normalize_Q_K=self.config.cell.normalize_Q_K,
                            att_dropout_p=self.config.cell.att_dropout_p,
                            dropout_p=self.config.cell.dropout_p,
                            cosine_att=self.config.cell.cosine_att,
                            temporal_multi_head_att_on_C_H_W=self.config.cell.temporal.temporal_multi_head_att_on_C_H_W,
                            att_with_relative_position_bias=self.config.cell.att_with_relative_position_bias,
                            att_with_output_proj=self.config.cell.att_with_output_proj,
                            scale_ratio_in_mixer=self.config.cell.scale_ratio_in_mixer,
                            with_mixer=with_mixer,
                            norm_mode=self.config.cell.norm_mode,
                            shuffle_in_window=shuffle_in_window,
                            use_flash_attention=use_flash_attention,
                        ),
                    )
                )
            else:
                self.cells.append(
                    (
                        f"cell_{i}",
                        Parallel_Cell(
                            C_in=C,
                            C_out=C_out,
                            H=H,
                            W=W,
                            D=D,
                            att_mode=att_type,
                            attention_type=self.config.cell.attention_type,
                            mixer_type=self.config.cell.mixer_type,
                            window_size=self.config.cell.window_size,
                            patch_size=self.config.cell.patch_size,
                            num_wind=self.config.cell.num_wind,
                            num_patch=self.config.cell.num_patch,
                            is_causal=is_causal,
                            n_head=self.config.cell.n_head,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            stride_s=stride_s,
                            stride_t=self.config.cell.temporal.stride_qk,
                            activation_func=self.config.cell.activation_func,
                            mixer_kernel_size=mixer_kernel_size,
                            mixer_stride=mixer_stride,
                            mixer_padding=mixer_padding,
                            normalize_Q_K=self.config.cell.normalize_Q_K,
                            att_dropout_p=self.config.cell.att_dropout_p,
                            dropout_p=self.config.cell.dropout_p,
                            cosine_att=self.config.cell.cosine_att,
                            temporal_multi_head_att_on_C_H_W=self.config.cell.temporal.temporal_multi_head_att_on_C_H_W,
                            att_with_relative_position_bias=self.config.cell.att_with_relative_position_bias,
                            att_with_output_proj=self.config.cell.att_with_output_proj,
                            scale_ratio_in_mixer=self.config.cell.scale_ratio_in_mixer,
                            with_mixer=with_mixer,
                            norm_mode=self.config.cell.norm_mode,
                            shuffle_in_window=shuffle_in_window,
                            use_flash_attention=use_flash_attention,
                        ),
                    )
                )

        self.make_block()

    @property
    def device(self):
        return next(self.parameters()).device

    def make_block(self):
        self.block = nn.ModuleDict(OrderedDict(self.cells))

    def forward(self, x):
        num_cells = len(self.block)

        if self.block_dense_connection:
            block_res = []

            for c in range(num_cells):
                if c == 0:
                    block_res.append(self.block[f"cell_{c}"](x))
                else:
                    input = 0
                    for k in block_res:
                        input = input + k
                    block_res.append(self.block[f"cell_{c}"](input))

            x = block_res[-1]
        else:
            for c in range(num_cells):
                x = self.block[f"cell_{c}"](x)

        return x

    def __str__(self):
        res = create_generic_class_str(self)
        return res


# -------------------------------------------------------------------------------------------------
