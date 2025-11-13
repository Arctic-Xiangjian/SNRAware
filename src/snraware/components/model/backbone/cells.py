"""
This file implements the cell structure in the model architecture. A cell is a 'transformer module' consisting
of attention layers, normalization layers and mixers with non-linearities.

Two type of  cells are implemented here:

- sequential norm first, transformer model
- Parallel cell, proposed in https://arxiv.org/abs/2302.05442

"""

import torchvision

from snraware.components.model.attention import *
from snraware.components.setup.status import create_generic_class_str

__all__ = ["Cell", "Parallel_Cell"]

# -------------------------------------------------------------------------------------------------


class Cell_Base(nn.Module):
    """Base class for the cell."""

    def __init__(
        self,
        C_in,
        C_out,
        H,
        W,
        D,
        att_mode,
        attention_type="conv",
        mixer_type="conv",
        window_size=None,
        patch_size=None,
        num_wind=None,
        num_patch=None,
        is_causal=False,
        n_head=8,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        activation_func="prelu",
        stride_s=(1, 1, 1),
        stride_t=(1, 1, 1),
        mixer_kernel_size=(5, 5, 5),
        mixer_stride=(1, 1, 1),
        mixer_padding=(2, 2, 2),
        normalize_Q_K=False,
        att_dropout_p=0.0,
        dropout_p=0.1,
        cosine_att=True,
        temporal_multi_head_att_on_C_H_W=False,
        att_with_relative_position_bias=True,
        att_with_output_proj=True,
        scale_ratio_in_mixer=4.0,
        with_mixer=True,
        norm_mode="layer",
        shuffle_in_window=False,
        use_flash_attention=False,
    ):
        """
        @args:
            - C_in (int): number of input channels
            - C_out (int): number of output channels
            - H, W, D (int): expected height, width and depth of the input, [B, C, D, H, W]
            - att_mode ("local", "global", "temporal", "local_3d", "global_3d", "vit_2d", "vit_3d", "swin_3d", "swin_3d_shifted", "conv2d", "conv3d"): different attention modules
            - attention_type ("conv", "lin"): what to use in attention modules to compute Q/K/V
            - mixer_type ("conv", "lin"): type of mixers; for temporal attention, only conv mixer is possible
            - window_size (int): size of window in the order of [H, W, D]
            - patch_size (int): size of patches
            - is_causal (bool): whether to mask attention to imply causality
            - n_head (int): number of heads in self attention
            - kernel_size, stride, padding (int, int): convolution parameters
            - stride_s (int, int): stride for spatial attention k,q matrices
            - stride_t (int, int): stride for temporal/frame attention k,q matrices
            - normalize_Q_K (bool): whether to use layernorm to normalize Q and K, as in 22B ViT paper
            - att_dropout_p (float): probability of dropout for attention coefficients
            - dropout_p (float): probability of dropout for attention output
            - cosine_att(bool): whether to use cosine attention
            - temporal_multi_head_att_on_C_H_W (bool): whether to use temporal multi-head attention on C,H,W dimensions; if false, with channel dimension for the multi-head attention
            - att_with_relative_position_bias (bool): whether to use relative position bias in attention
            - with_mixer (bool): whether to add a conv2D mixer after attention
            - att_with_output_proj (bool): whether to add output projection in the attention layer
            - scale_ratio_in_mixer (float): channel scaling ratio in the mixer
            - norm_mode ("layer", "batch2d", "instance2d", "batch3d", "instance3d"):
                - layer: each C,H,W
                - batch2d: along B*T
                - instance2d: each H,W
                - batch3d: along B
                - instance3d: each T,H,W
            - shuffle_in_window (bool): whether to shuffle the input in the window for global attention
            - use_flash_attention (bool): whether to use flash attention for the attention module.
        """
        if num_patch is None:
            num_patch = [4, 4, 1]
        if num_wind is None:
            num_wind = [8, 8, 1]
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.H = H
        self.W = W
        self.D = D
        self.att_mode = att_mode
        self.attention_type = attention_type
        self.mixer_type = mixer_type
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_wind = num_wind
        self.num_patch = num_patch
        self.is_causal = is_causal
        self.n_head = n_head

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.activation_func = activation_func

        self.stride_s = stride_s
        self.stride_t = stride_t

        self.mixer_kernel_size = mixer_kernel_size
        self.mixer_stride = mixer_stride
        self.mixer_padding = mixer_padding

        self.normalize_Q_K = normalize_Q_K
        self.cosine_att = cosine_att
        self.att_with_relative_position_bias = att_with_relative_position_bias
        self.att_dropout_p = att_dropout_p
        self.dropout_p = dropout_p
        self.att_with_output_proj = att_with_output_proj
        self.with_mixer = with_mixer
        self.scale_ratio_in_mixer = scale_ratio_in_mixer
        self.norm_mode = norm_mode
        self.shuffle_in_window = shuffle_in_window

        self.use_flash_attention = use_flash_attention

        self.temporal_multi_head_att_on_C_H_W = temporal_multi_head_att_on_C_H_W

        self.n1 = create_norm(norm_mode=norm_mode, C=C_in, H=H, W=W, D=D)
        self.n2 = create_norm(norm_mode=norm_mode, C=C_out, H=H, W=W, D=D)

        if C_in != C_out:
            self.input_proj = Conv2DExt(
                C_in,
                C_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
                channel_first=True,
            )
        else:
            self.input_proj = nn.Identity()

        if att_mode == "temporal":
            if self.temporal_multi_head_att_on_C_H_W:
                self.attn = TemporalCnnAttention(
                    C_in=C_in,
                    C_out=C_out,
                    H=self.H,
                    W=self.W,
                    is_causal=is_causal,
                    n_head=n_head,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    stride_qk=stride_t,
                    cosine_att=cosine_att,
                    normalize_Q_K=normalize_Q_K,
                    att_dropout_p=att_dropout_p,
                    att_with_output_proj=att_with_output_proj,
                    flash_att=self.use_flash_attention,
                )
            else:
                self.attn = TemporalChannelCnnAttention(
                    C_in=C_in,
                    C_out=C_out,
                    H=self.H,
                    W=self.W,
                    is_causal=is_causal,
                    n_head=n_head,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    stride_qk=stride_t,
                    cosine_att=cosine_att,
                    normalize_Q_K=normalize_Q_K,
                    att_dropout_p=att_dropout_p,
                    att_with_output_proj=att_with_output_proj,
                    flash_att=self.use_flash_attention,
                )
        elif att_mode == "local":
            self.attn = SpatialLocalAttention(
                C_in=C_in,
                C_out=C_out,
                H=self.H,
                W=self.W,
                window_size=window_size,
                patch_size=patch_size,
                num_wind=num_wind,
                num_patch=num_patch,
                attention_type=attention_type,
                n_head=n_head,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                stride_qk=self.stride_s,
                normalize_Q_K=normalize_Q_K,
                cosine_att=cosine_att,
                att_with_relative_position_bias=att_with_relative_position_bias,
                att_dropout_p=att_dropout_p,
                att_with_output_proj=att_with_output_proj,
            )
        elif att_mode == "local_3d":
            self.attn = Local3DAttention(
                C_in=C_in,
                C_out=C_out,
                H=H,
                W=W,
                D=D,
                window_size=window_size,
                patch_size=patch_size,
                num_wind=num_wind,
                num_patch=num_patch,
                attention_type=attention_type,
                n_head=n_head,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                stride_qk=self.stride_s,
                att_dropout_p=att_dropout_p,
                cosine_att=cosine_att,
                normalize_Q_K=normalize_Q_K,
                att_with_relative_position_bias=att_with_relative_position_bias,
                att_with_output_proj=att_with_output_proj,
            )
        elif att_mode == "global":
            self.attn = SpatialGlobalAttention(
                C_in=C_in,
                C_out=C_out,
                H=self.H,
                W=self.W,
                window_size=[k // 2 for k in window_size],
                patch_size=patch_size,
                num_wind=[2 * n for n in num_wind],
                num_patch=[p // 2 for p in num_patch],
                attention_type=attention_type,
                n_head=n_head,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                stride_qk=self.stride_s,
                normalize_Q_K=normalize_Q_K,
                cosine_att=cosine_att,
                att_with_relative_position_bias=att_with_relative_position_bias,
                att_dropout_p=att_dropout_p,
                att_with_output_proj=att_with_output_proj,
                shuffle_in_window=shuffle_in_window,
            )
        elif att_mode == "global_3d":
            self.attn = Global3DAttention(
                C_in=C_in,
                C_out=C_out,
                H=H,
                W=H,
                D=D,
                window_size=window_size,
                patch_size=patch_size,
                num_wind=num_wind,
                num_patch=num_patch,
                attention_type=attention_type,
                n_head=n_head,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                stride_qk=self.stride_s,
                att_dropout_p=att_dropout_p,
                cosine_att=cosine_att,
                normalize_Q_K=normalize_Q_K,
                att_with_relative_position_bias=att_with_relative_position_bias,
                att_with_output_proj=att_with_output_proj,
                shuffle_in_window=shuffle_in_window,
            )
        elif att_mode == "vit_2d":
            self.attn = SpatialViTAttention(
                C_in=C_in,
                C_out=C_out,
                H=self.H,
                W=self.W,
                window_size=window_size,
                num_wind=num_wind,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                stride_qk=self.stride_s,
                normalize_Q_K=normalize_Q_K,
                cosine_att=cosine_att,
                att_with_relative_position_bias=att_with_relative_position_bias,
                att_dropout_p=att_dropout_p,
                att_with_output_proj=att_with_output_proj,
            )
        elif att_mode == "vit_3d":
            self.attn = ViT3DAttention(
                C_in=C_in,
                C_out=C_out,
                H=self.H,
                W=self.W,
                D=self.D,
                window_size=window_size,
                num_wind=num_wind,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                stride_qk=self.stride_s,
                normalize_Q_K=normalize_Q_K,
                cosine_att=cosine_att,
                att_with_relative_position_bias=att_with_relative_position_bias,
                att_dropout_p=att_dropout_p,
                att_with_output_proj=att_with_output_proj,
            )
        elif att_mode == "swin_3d":
            self.attn = Swin3DAttention(
                C_in=C_in,
                C_out=C_out,
                H=self.H,
                W=self.W,
                D=self.D,
                window_size=window_size,
                num_wind=num_wind,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                shifted=False,
                stride_qk=self.stride_s,
                normalize_Q_K=normalize_Q_K,
                cosine_att=cosine_att,
                att_with_relative_position_bias=att_with_relative_position_bias,
                att_dropout_p=att_dropout_p,
                att_with_output_proj=att_with_output_proj,
            )
        elif att_mode == "swin_3d_shifted":
            self.attn = Swin3DAttention(
                C_in=C_in,
                C_out=C_out,
                H=self.H,
                W=self.W,
                D=self.D,
                window_size=window_size,
                num_wind=num_wind,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                shifted=True,
                stride_qk=self.stride_s,
                normalize_Q_K=normalize_Q_K,
                cosine_att=cosine_att,
                att_with_relative_position_bias=att_with_relative_position_bias,
                att_dropout_p=att_dropout_p,
                att_with_output_proj=att_with_output_proj,
            )
        elif att_mode == "conv2d" or att_mode == "conv3d":
            self.attn = ConvolutionModule(
                conv_type=att_mode,
                C_in=C_in,
                C_out=C_out,
                H=self.H,
                W=self.W,
                D=self.D,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                norm_mode=norm_mode,
                activation_func=self.activation_func,
            )
        else:
            raise NotImplementedError(f"Attention mode not implemented: {att_mode}")

        self.stochastic_depth = torchvision.ops.StochasticDepth(p=self.dropout_p, mode="row")

        self.with_mixer = with_mixer
        if self.with_mixer:
            self.create_mixer()

    def create_mixer(self):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    def __str__(self):
        res = create_generic_class_str(self)
        return res


# -------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------
class Cell(Cell_Base):
    """
    The Pre-Norm implementation is used here.

    x-> Norm -> attention -> + -> Norm -> mixer -> + -> logits
    |------------------------| |-----------------------|
    """

    def __init__(
        self,
        C_in,
        C_out,
        H,
        W,
        D,
        att_mode,
        attention_type="conv",
        mixer_type="conv",
        window_size=None,
        patch_size=None,
        num_wind=None,
        num_patch=None,
        is_causal=False,
        n_head=8,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        activation_func="prelu",
        stride_s=(1, 1),
        stride_t=(1, 1),
        mixer_kernel_size=(5, 5),
        mixer_stride=(1, 1),
        mixer_padding=(2, 2),
        normalize_Q_K=False,
        att_dropout_p=0.0,
        dropout_p=0.1,
        cosine_att=True,
        temporal_multi_head_att_on_C_H_W=False,
        att_with_relative_position_bias=True,
        att_with_output_proj=True,
        scale_ratio_in_mixer=4.0,
        with_mixer=True,
        norm_mode="layer",
        shuffle_in_window=False,
        use_flash_attention=False,
    ):
        if num_patch is None:
            num_patch = [4, 4]
        if num_wind is None:
            num_wind = [8, 8]
        super().__init__(
            C_in=C_in,
            C_out=C_out,
            H=H,
            W=W,
            D=D,
            att_mode=att_mode,
            attention_type=attention_type,
            mixer_type=mixer_type,
            window_size=window_size,
            patch_size=patch_size,
            num_wind=num_wind,
            num_patch=num_patch,
            is_causal=is_causal,
            n_head=n_head,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation_func=activation_func,
            stride_s=stride_s,
            stride_t=stride_t,
            mixer_kernel_size=mixer_kernel_size,
            mixer_stride=mixer_stride,
            mixer_padding=mixer_padding,
            normalize_Q_K=normalize_Q_K,
            att_dropout_p=att_dropout_p,
            dropout_p=dropout_p,
            cosine_att=cosine_att,
            temporal_multi_head_att_on_C_H_W=temporal_multi_head_att_on_C_H_W,
            att_with_relative_position_bias=att_with_relative_position_bias,
            att_with_output_proj=att_with_output_proj,
            scale_ratio_in_mixer=scale_ratio_in_mixer,
            with_mixer=with_mixer,
            norm_mode=norm_mode,
            shuffle_in_window=shuffle_in_window,
            use_flash_attention=use_flash_attention,
        )

    def create_mixer(self):
        act_func = create_activation_func(name=self.activation_func)

        if (
            self.mixer_type == "conv"
            or self.att_mode == "temporal"
            or self.att_mode == "conv2d"
            or self.att_mode == "conv3d"
        ):
            mixer_cha = int(self.scale_ratio_in_mixer * self.C_out)

            self.mlp = nn.Sequential(
                Conv2DExt(
                    self.C_out,
                    mixer_cha,
                    kernel_size=self.mixer_kernel_size,
                    stride=self.mixer_stride,
                    padding=self.mixer_padding,
                    bias=True,
                    channel_first=True,
                ),
                act_func,
                Conv2DExt(
                    mixer_cha,
                    self.C_out,
                    kernel_size=self.mixer_kernel_size,
                    stride=self.mixer_stride,
                    padding=self.mixer_padding,
                    bias=True,
                    channel_first=True,
                ),
            )
        elif self.mixer_type == "lin":
            # apply mixer on every patch
            D = self.attn.get_dimension_for_linear_mixer()
            D_prime = int(self.scale_ratio_in_mixer * D)

            self.mlp = nn.Sequential(
                nn.Linear(D, D_prime, bias=True), act_func, nn.Linear(D_prime, D, bias=True)
            )
        else:
            raise NotImplementedError(f"Mixer mode not implemented: {self.mixer_type}")

    def forward(self, x):
        x = self.input_proj(x) + self.stochastic_depth(self.attn(self.n1(x)))

        if self.with_mixer:
            if (
                self.mixer_type == "conv"
                or self.att_mode == "temporal"
                or self.att_mode == "conv2d"
                or self.att_mode == "conv3d"
            ):
                x = x + self.stochastic_depth(self.mlp(self.n2(x)))
            elif self.mixer_type == "lin":
                x = self.n2(x)
                if "3d" in self.att_mode:
                    x = self.attn.im2grid(x)
                    *Dim, a, b, c, d = x.shape
                    x = self.mlp(x.reshape((*Dim, -1)))
                    x = torch.reshape(x, (*Dim, a, b, c, d))
                    x = self.attn.grid2im(x)
                    x = self.stochastic_depth(x)
                else:
                    x = permute_to_B_T_C_H_W(x)
                    x = self.attn.im2grid(x)
                    *Dim, C, wh, ww = x.shape
                    x = self.mlp(x.reshape((*Dim, -1)))
                    x = self.attn.grid2im(x.reshape((*Dim, C, wh, ww)))
                    x = permute_to_B_C_T_H_W(x)
                    x = self.stochastic_depth(x)

        return x


# -------------------------------------------------------------------------------------------------


class Parallel_Cell(Cell_Base):
    """
    Parallel transformer cell.

    x -> Norm ----> attention --> + -> + -> logits
      |        |--> CNN mixer-----|    |
      |----------> input_proj ----> ---|

    """

    def __init__(
        self,
        C_in,
        C_out,
        H,
        W,
        D,
        att_mode,
        attention_type="conv",
        mixer_type="conv",
        window_size=None,
        patch_size=None,
        num_wind=None,
        num_patch=None,
        is_causal=False,
        n_head=8,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        stride_s=(1, 1),
        stride_t=(1, 1),
        activation_func="prelu",
        mixer_kernel_size=(5, 5),
        mixer_stride=(1, 1),
        mixer_padding=(2, 2),
        normalize_Q_K=False,
        att_dropout_p=0.0,
        dropout_p=0.1,
        cosine_att=True,
        temporal_multi_head_att_on_C_H_W=False,
        att_with_relative_position_bias=True,
        att_with_output_proj=True,
        scale_ratio_in_mixer=4.0,
        with_mixer=True,
        norm_mode="layer",
        shuffle_in_window=False,
        use_flash_attention=False,
    ):
        """Complete transformer parallel cell."""
        if num_patch is None:
            num_patch = [4, 4]
        if num_wind is None:
            num_wind = [8, 8]
        super().__init__(
            C_in=C_in,
            C_out=C_out,
            H=H,
            W=W,
            D=D,
            att_mode=att_mode,
            attention_type=attention_type,
            mixer_type=mixer_type,
            window_size=window_size,
            patch_size=patch_size,
            num_wind=num_wind,
            num_patch=num_patch,
            is_causal=is_causal,
            n_head=n_head,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation_func=activation_func,
            stride_s=stride_s,
            stride_t=stride_t,
            mixer_kernel_size=mixer_kernel_size,
            mixer_stride=mixer_stride,
            mixer_padding=mixer_padding,
            normalize_Q_K=normalize_Q_K,
            att_dropout_p=att_dropout_p,
            dropout_p=dropout_p,
            cosine_att=cosine_att,
            temporal_multi_head_att_on_C_H_W=temporal_multi_head_att_on_C_H_W,
            att_with_relative_position_bias=att_with_relative_position_bias,
            att_with_output_proj=att_with_output_proj,
            scale_ratio_in_mixer=scale_ratio_in_mixer,
            with_mixer=with_mixer,
            norm_mode=norm_mode,
            shuffle_in_window=shuffle_in_window,
            use_flash_attention=use_flash_attention,
        )

    def create_mixer(self):
        act_func = create_activation_func(name=self.activation_func)

        if (
            self.mixer_type == "conv"
            or self.att_mode == "temporal"
            or self.att_mode == "conv2d"
            or self.att_mode == "conv3d"
        ):
            mixer_cha = int(self.scale_ratio_in_mixer * self.C_out)

            self.mlp = nn.Sequential(
                Conv2DExt(
                    self.C_in,
                    mixer_cha,
                    kernel_size=self.mixer_kernel_size,
                    stride=self.mixer_stride,
                    padding=self.mixer_padding,
                    bias=True,
                    channel_first=True,
                ),
                act_func,
                Conv2DExt(
                    mixer_cha,
                    self.C_out,
                    kernel_size=self.mixer_kernel_size,
                    stride=self.mixer_stride,
                    padding=self.mixer_padding,
                    bias=True,
                    channel_first=True,
                ),
            )
        elif self.mixer_type == "lin":
            D = self.attn.get_dimension_for_linear_mixer()
            D = D // self.C_out
            D *= self.C_in

            D_out = int(D // self.C_in * self.C_out)
            D_prime = int(self.scale_ratio_in_mixer * D_out)

            self.mlp = nn.Sequential(
                nn.Linear(D, D_prime, bias=True), act_func, nn.Linear(D_prime, D_out, bias=True)
            )
        else:
            raise NotImplementedError(f"Mixer mode not implemented: {self.mixer_type}")

    def forward(self, x):
        x_normed = self.n1(x)
        y = self.stochastic_depth(self.attn(x_normed))

        if self.with_mixer:
            if (
                self.mixer_type == "conv"
                or self.att_mode == "temporal"
                or self.att_mode == "conv2d"
                or self.att_mode == "conv3d"
            ):
                res_mixer = self.stochastic_depth(self.mlp(x_normed))
            else:
                if "3d" in self.att_mode:
                    t = self.attn.im2grid(x_normed)
                    *Dim, a, b, c, _d = t.shape
                    t = self.mlp(t.reshape((*Dim, -1)))
                    t = torch.reshape(t, (*Dim, a, b, c, -1))
                    t = self.attn.grid2im(t)
                    res_mixer = self.stochastic_depth(t)
                else:
                    res_mixer = self.attn.im2grid(permute_to_B_T_C_H_W(x_normed))
                    *Dim, wh, ww, _C = res_mixer.shape
                    res_mixer = self.mlp(res_mixer.reshape((*Dim, -1)))
                    res_mixer = self.attn.grid2im(res_mixer.reshape((*Dim, wh, ww, self.C_out)))
                    res_mixer = permute_to_B_C_T_H_W(res_mixer)
                    res_mixer = self.stochastic_depth(res_mixer)

            y += res_mixer

        y += self.input_proj(x)

        return y


# -------------------------------------------------------------------------------------------------
