"""Configuration classes for model parameters."""

from dataclasses import dataclass, field

import omegaconf

__all__ = [
    "BackboneConfig",
    "BlockConfig",
    "CellConfig",
    "ConvolutionConfig",
    "HRNetConfig",
    "SOANetConfig",
    "SpatialGlobal3DConfig",
    "SpatialGlobalConfig",
    "SpatialLocal3DConfig",
    "SpatialLocalConfig",
    "SpatialViTConfig",
    "Swin3DConfig",
    "TemporalAttentionConfig",
    "UNetConfig",
    "ViT3DConfig",
    "create_block_config",
]


# -------------------------------------------------------------------------------------------------
@dataclass
class CellConfig:
    # parameters for the cell

    # conv or lin, type of attention in the spatial attention modules
    attention_type: str = "conv"

    # conv or lin, type of mixer in the spatial attention modules; only conv is possible for the temporal attention
    mixer_type: str = "conv"

    # size of window for spatial attention. This is the number of pixels in a window. Given image height and weight H and W, number of windows is H/windows_size[0] * W/windows_size[1]; for 3D usecase, window_size order is [H, W, D]
    window_size: list[int] = field(default_factory=lambda: [16, 16, 1])

    # size of patch for spatial attention. This is the number of pixels in a patch. An image is first split into windows. Every window is further split into patches.
    patch_size: list[int] = field(default_factory=lambda: [2, 2, 1])

    # method to adjust window_size between resolution levels, "keep_window_size", "keep_num_window", "mixed".
    # "keep_window_size" means number of pixels in a window is kept after down/upsample the image; "keep_num_window" means the number of windows is kept after down/upsample the image; "mixed" means interleave both methods.
    window_sizing_method: str = "mixed"

    # number of transformer heads
    n_head: int = 16

    # query, key and value conv setting for the cells
    kernel_size: list[int] = field(default_factory=lambda: [3, 3, 3])
    stride: list[int] = field(default_factory=lambda: [1, 1, 1])
    padding: list[int] = field(default_factory=lambda: [1, 1, 1])

    # stride for spatial dimensions of query and key tensors
    stride_s: list[int] = field(default_factory=lambda: [1, 1, 1])

    # conv setting for the mixers in the cells
    mixer_kernel_size: list[int] = field(default_factory=lambda: [3, 3, 3])
    mixer_stride: list[int] = field(default_factory=lambda: [1, 1, 1])
    mixer_padding: list[int] = field(default_factory=lambda: [1, 1, 1])

    # he scaling ratio to increase/decrease dimensions in the mixer of an attention layer
    scale_ratio_in_mixer: float = 4.0

    # whether to normalize Q and K before computing attention matrix
    normalize_Q_K: bool = True
    # whether to use cosine attention for computing attention matrix
    cosine_att: bool = True
    # whether to add relative position bias to the attention matrix
    att_with_relative_position_bias: bool = True
    # dropout probability for attention matrix
    att_dropout_p: float = 0.0
    # dropout probability after applying the attention matrix; aka stochastic residual connections
    dropout_p: float = 0.1
    # whether to add output projection in attention layer
    att_with_output_proj: bool = True

    # normalization mode
    norm_mode: str = "instance2d"  # "layer", "batch2d", "instance2d", "batch3d", "instance3d"

    # whether to treat timed data as causal and mask future entries
    is_causal: bool = False

    # nonlinear activation function, "elu", "relu", "leakyrelu", "prelu", "relu6", "selu", "celu", "gelu"
    activation_func: str = "prelu"

    # upsampling method in backbone, "NN", "linear", "bspline"
    upsample_method: str = "linear"


# configuration for cells with specific attention types
# the name string of attention modules are defined in the documentation


@dataclass
class TemporalAttentionConfig(CellConfig):
    # If True, multi-head attention is applied to the C, H, W dimensions for the temporal attention.
    # If False, multi-head attention is applied to the C dimension only.
    temporal_multi_head_att_on_C_H_W: bool = False

    # whether to use flash attention for the temporal attention layers
    flash_att: bool = False

    # stride for temporal dimension of query and key tensors
    stride_qk: list[int] = field(default_factory=lambda: [1, 1, 1])


@dataclass
class SpatialLocalConfig(CellConfig):
    pass


@dataclass
class SpatialGlobalConfig(CellConfig):
    # whether to shuffle patches in a window for the global attention
    shuffle_in_window: bool = False


@dataclass
class ConvolutionConfig(CellConfig):
    conv_type: str = "conv2d"  # conv2d or conv3d


@dataclass
class SpatialLocal3DConfig(CellConfig):
    pass


@dataclass
class SpatialGlobal3DConfig(CellConfig):
    # whether to shuffle patches in a window for the global attention
    shuffle_in_window: bool = False


@dataclass
class SpatialViTConfig(CellConfig):
    pass


@dataclass
class ViT3DConfig(CellConfig):
    pass


@dataclass
class Swin3DConfig(CellConfig):
    shifted_window_attention: bool = False


CellTypeVariant = (
    TemporalAttentionConfig
    | SpatialLocalConfig
    | SpatialGlobalConfig
    | ConvolutionConfig
    | SpatialLocal3DConfig
    | SpatialGlobal3DConfig
    | SpatialViTConfig
    | ViT3DConfig
    | Swin3DConfig
)

# ---------------------------------------------------------------------------------------------


@dataclass
class BlockConfig:
    # block contains cell parameters
    cell: CellConfig = field(default_factory=CellConfig)

    # sequential or parallel cells
    cell_type: str = "sequential"

    # whether to add dense connections between cells in a block
    block_dense_connection: bool = False


# ---------------------------------------------------------------------------------------------
def create_block_config(block_str: list[str], block: BlockConfig) -> list[list[CellTypeVariant]]:
    block_config = []
    for att_types in block_str:
        assert len(att_types) >= 1, "At least one attention module is required to build the model"
        assert not (len(att_types) % 2), "require attention and mixer info for each cell"

        a_block_config = []

        for i in range(0, len(att_types), 2):
            att_type = att_types[i]
            mixer = att_types[i + 1]

            if att_type == "L":
                if mixer == "3":
                    a_block_config.append(SpatialLocal3DConfig())
                else:
                    a_block_config.append(SpatialLocalConfig())
            elif att_type == "G":
                if mixer == "3":
                    a_block_config.append(
                        SpatialGlobal3DConfig(
                            shuffle_in_window=block.cell.spatial_global_3d.shuffle_in_window
                        )
                    )
                else:
                    a_block_config.append(
                        SpatialGlobalConfig(
                            shuffle_in_window=block.cell.spatial_global.shuffle_in_window
                        )
                    )
            elif att_type == "T":
                a_block_config.append(
                    TemporalAttentionConfig(
                        temporal_multi_head_att_on_C_H_W=block.cell.temporal.temporal_multi_head_att_on_C_H_W,
                        flash_att=block.cell.temporal.flash_att,
                        stride_qk=block.cell.temporal.stride_qk,
                    )
                )
            elif att_type == "V" and mixer == "2":
                a_block_config.append(SpatialViTConfig())
            elif att_type == "V" and mixer == "3":
                a_block_config.append(ViT3DConfig())
            elif att_type == "s" and mixer == "h":
                a_block_config.append(Swin3DConfig())
            elif att_type == "S" and mixer == "3":
                a_block_config.append(Swin3DConfig())
            elif att_type == "S" and mixer == "h":
                a_block_config.append(Swin3DConfig(shifted_window_attention=True))
            elif att_type == "C" and mixer == "2":
                a_block_config.append(ConvolutionConfig(conv_type="conv2d"))
            elif att_type == "C" and mixer == "3":
                a_block_config.append(ConvolutionConfig(conv_type="conv3d"))
            else:
                raise ValueError(f"Incorrect att_type: {att_type}, mixer: {mixer}")

        block_config.append(a_block_config)

    return block_config


@dataclass
class BackboneConfig:
    block: BlockConfig = field(default_factory=BlockConfig)

    # name of the backbone
    name: str = "HRnet"

    # number of channels in main body of backbone
    num_of_channels: int = 32

    # number of resolution levels; image size reduce by x2 for every level
    num_resolution_levels: int = 2

    # block string to define the attention layers in blocks; if multiple strings are given, each is for a resolution level
    block_str: list[str] = field(default_factory=lambda: ["T1L1G1"])

    # block config to store the configuration objects for all cells
    # this field is created from the block_str field in the __post_init__
    block_config: list[list[CellTypeVariant]] = field(
        default_factory=lambda: [
            [TemporalAttentionConfig(), SpatialLocalConfig(), SpatialGlobalConfig()]
        ]
    )

    # whether to use interpolation in downsample layer; if False, use stride convolution
    use_interpolation: bool = True

    # whether to add conv in downsample layers; if False, only interpolation is performed
    with_conv: bool = True

    # whether to add unet channel attention on the skip connections between resolution levels
    use_unet_attention: bool = True

    # number of attention stages in the soanet backbone
    # optionally, tensor size can be reduced by x2 for every stage
    num_stages: int = 2

    # whether to downsample between stages
    downsample: bool = True

    def __post_init__(self):
        assert isinstance(self.block_str, list) or isinstance(
            self.block_str, omegaconf.listconfig.ListConfig
        ), "block_str must be a list of strings"
        self.block_config = create_block_config(self.block_str, self.block)


# ---------------------------------------------------------------------------------------------


@dataclass
class HRNetConfig(BackboneConfig):
    num_resolution_levels: int = 2


# ---------------------------------------------------------------------------------------------


@dataclass
class UNetConfig(BackboneConfig):
    # number of resolution levels; image size reduce by x2 for every level
    num_resolution_levels: int = 2

    # whether to add unet channel attention on the skip connections between resolution levels
    use_unet_attention: bool = True


# ---------------------------------------------------------------------------------------------


@dataclass
class SOANetConfig(BackboneConfig):
    # number of attention stages in the soanet backbone
    # optionally, tensor size can be reduced by x2 for every stage
    num_stages: int = 2

    # whether to downsample between stages
    downsample: bool = True


# ---------------------------------------------------------------------------------------------
