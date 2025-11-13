"""Local 3D attention modules computes attentions among all patches within a windows."""

from einops import rearrange

from .attention_modules import *


# -------------------------------------------------------------------------------------------------
class Local3DAttention(CnnAttentionBase):
    """Multi-head attention among all patches in a window."""

    def __init__(
        self,
        C_in,
        C_out=16,
        H=256,
        W=256,
        D=128,
        window_size=None,
        patch_size=None,
        num_wind=None,
        num_patch=None,
        attention_type="conv",
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
        with_timer=False,
    ):
        """
        @args:
            - window_size (int): number of pixels in a window
            - patch_size(int): number of pixels in a patch.
        """
        if num_patch is None:
            num_patch = [4, 4, 4]
        if num_wind is None:
            num_wind = [8, 8, 8]
        if patch_size is None:
            patch_size = [4, 4, 4]
        if window_size is None:
            window_size = [32, 32, 8]
        super().__init__(
            C_in=C_in,
            C_out=C_out,
            H=H,
            W=W,
            D=D,
            n_head=n_head,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            stride_qk=stride_qk,
            att_dropout_p=att_dropout_p,
            cosine_att=cosine_att,
            normalize_Q_K=normalize_Q_K,
            att_with_relative_position_bias=att_with_relative_position_bias,
            att_with_output_proj=att_with_output_proj,
            with_timer=with_timer,
        )

        self.attention_type = attention_type
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_wind = num_wind
        self.num_patch = num_patch

        self.set_and_check_wind()
        self.set_and_check_patch()

        if len(self.window_size) == 2:
            self.window_size.append(1)
        if len(self.num_wind) == 2:
            self.num_wind.append(1)

        if len(self.patch_size) == 2:
            self.patch_size.append(1)
        if len(self.num_patch) == 2:
            self.num_patch.append(1)

        self.validate_window_patch()

        num_pixel_patch = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        assert self.C_out * num_pixel_patch % self.n_head == 0, (
            f"Number of pixels in a patch {self.C_out * num_pixel_patch} should be divisible by number of heads {self.n_head}"
        )

        print(
            f"{Fore.YELLOW}--> Spatial, local, H {H}, W {W}, D {D}, win size {self.window_size}, patch size {self.patch_size}{Style.RESET_ALL}, num_wind {self.num_wind}, num_patch {self.num_patch}"
        )

        if attention_type == "conv":
            # key, query, value projections convolution
            # Wk, Wq, Wv
            if self.D > 1:
                self.key = Conv3DExt(
                    C_in,
                    C_out,
                    kernel_size=kernel_size,
                    stride=stride_qk,
                    padding=padding,
                    bias=True,
                    channel_first=True,
                )
                self.query = Conv3DExt(
                    C_in,
                    C_out,
                    kernel_size=kernel_size,
                    stride=stride_qk,
                    padding=padding,
                    bias=True,
                    channel_first=True,
                )
                self.value = Conv3DExt(
                    C_in,
                    C_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=True,
                    channel_first=True,
                )
            else:
                self.key = Conv2DExt(
                    C_in,
                    C_out,
                    kernel_size=kernel_size,
                    stride=stride_qk,
                    padding=padding,
                    bias=True,
                    channel_first=True,
                )
                self.query = Conv2DExt(
                    C_in,
                    C_out,
                    kernel_size=kernel_size,
                    stride=stride_qk,
                    padding=padding,
                    bias=True,
                    channel_first=True,
                )
                self.value = Conv2DExt(
                    C_in,
                    C_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=True,
                    channel_first=True,
                )
        elif attention_type == "lin":
            # linear projections

            self.key = LinearGrid3DExt(C_in * num_pixel_patch, C_out * num_pixel_patch, bias=True)
            self.query = LinearGrid3DExt(
                C_in * num_pixel_patch, C_out * num_pixel_patch, bias=True
            )
            self.value = LinearGrid3DExt(
                C_in * num_pixel_patch, C_out * num_pixel_patch, bias=True
            )
        else:
            raise NotImplementedError(f"Attention type not implemented: {attention_type}")

        if self.att_with_relative_position_bias:
            self.define_relative_position_bias_table_3D(
                num_win_h=self.num_patch[0],
                num_win_w=self.num_patch[1],
                num_win_d=self.num_patch[2],
            )
            self.define_relative_position_index_3D(
                num_win_h=self.num_patch[0],
                num_win_w=self.num_patch[1],
                num_win_d=self.num_patch[2],
            )

    def attention(self, k, q, v):
        (
            B,
            num_win_d,
            num_win_h,
            num_win_w,
            num_patch_d_per_win,
            num_patch_h_per_win,
            num_patch_w_per_win,
            pd,
            ph,
            pw,
            C,
        ) = k.shape
        *_, pd_v, ph_v, pw_v, _ = v.shape

        assert self.num_patch[0] == num_patch_h_per_win
        assert self.num_patch[1] == num_patch_w_per_win
        assert self.num_patch[2] == num_patch_d_per_win

        # format the window
        # use torch.sym_... to support onnx conversion
        hc = (C * pd * ph * pw) // self.n_head
        hc_v = (C * pd_v * ph_v * pw_v) // self.n_head

        # k, q, v will be [B, T, num_win_h*num_win_w, self.n_head, num_patch_h_per_win*num_patch_w_per_win, hc]
        tm = start_timer(enable=self.with_timer)
        k = k.reshape(
            (
                B,
                num_win_d * num_win_h * num_win_w,
                num_patch_d_per_win * num_patch_h_per_win * num_patch_w_per_win,
                self.n_head,
                hc,
            )
        ).transpose(2, 3)
        q = q.reshape(
            (
                B,
                num_win_d * num_win_h * num_win_w,
                num_patch_d_per_win * num_patch_h_per_win * num_patch_w_per_win,
                self.n_head,
                hc,
            )
        ).transpose(2, 3)
        v = v.reshape(
            (
                B,
                num_win_d * num_win_h * num_win_w,
                num_patch_d_per_win * num_patch_h_per_win * num_patch_w_per_win,
                self.n_head,
                hc_v,
            )
        ).transpose(2, 3)
        end_timer(enable=self.with_timer, t=tm, msg="k, q, v - reshape")

        tm = start_timer(enable=self.with_timer)
        # [B, num_windows, num_heads, num_patches, hc] x [B, num_windows, num_heads, hc, num_patches] -> (B, num_windows, num_heads, num_patches, num_patches)
        if self.cosine_att:
            att = cosine_attention(q, k)
        else:
            if self.normalize_Q_K:
                q, k = normalize_qk(q, k)

            att = q @ k.transpose(-2, -1) * torch.sqrt(torch.tensor(1.0 / hc))
        end_timer(enable=self.with_timer, t=tm, msg="att")

        tm = start_timer(enable=self.with_timer)
        att = F.softmax(att, dim=-1)
        end_timer(enable=self.with_timer, t=tm, msg="softmax")

        tm = start_timer(enable=self.with_timer)
        if self.att_with_relative_position_bias:
            relative_position_bias = self.get_relative_position_bias_3D(
                num_patch_h_per_win, num_patch_w_per_win, num_patch_d_per_win
            )
            att = att + relative_position_bias
        end_timer(enable=self.with_timer, t=tm, msg="att_with_relative_position_bias")

        tm = start_timer(enable=self.with_timer)
        att = self.attn_drop(att)
        end_timer(enable=self.with_timer, t=tm, msg="attn_drop")

        tm = start_timer(enable=self.with_timer)
        # (B, num_windows, num_heads, num_patches, num_patches) * (B, num_windows, num_heads, num_patches, hc)
        y = att @ v  # (B, num_windows, num_heads, num_patches, hc)
        y = y.transpose(2, 3)  # (B, num_windows, num_patches, num_heads, hc)
        y = torch.reshape(
            y,
            (
                B,
                num_win_d,
                num_win_h,
                num_win_w,
                num_patch_d_per_win,
                num_patch_h_per_win,
                num_patch_w_per_win,
                pd_v,
                ph_v,
                pw_v,
                C,
            ),
        )
        end_timer(enable=self.with_timer, t=tm, msg="att @ v")

        tm = start_timer(enable=self.with_timer)
        y = self.grid2im(y)
        end_timer(enable=self.with_timer, t=tm, msg="grid2im")

        return y

    def forward(self, x):
        _B, C, _D, _H, _W = x.size()

        assert C == self.C_in, (
            f"Input channel {C} does not match expected input channel {self.C_in}"
        )

        if self.attention_type == "conv":
            tm = start_timer(enable=self.with_timer)
            k = self.key(x)
            q = self.query(x)
            v = self.value(x)
            end_timer(enable=self.with_timer, t=tm, msg="compute k, q, v")

            tm = start_timer(enable=self.with_timer)
            k = self.im2grid(k)
            q = self.im2grid(q)
            v = self.im2grid(v)
            end_timer(enable=self.with_timer, t=tm, msg="im2grid")
        else:
            tm = start_timer(enable=self.with_timer)
            x = self.im2grid(x)
            end_timer(enable=self.with_timer, t=tm, msg="im2grid")

            tm = start_timer(enable=self.with_timer)
            k = self.key(x)
            q = self.query(x)
            v = self.value(x)
            end_timer(enable=self.with_timer, t=tm, msg="compute k, q, v")

        y = self.attention(k, q, v)

        tm = start_timer(enable=self.with_timer)
        y = self.output_proj(y)
        end_timer(enable=self.with_timer, t=tm, msg="output_proj")

        return y

    def im2grid(self, x):
        """Reshape the input into windows of local areas."""
        _B, _C, D, H, W = x.shape

        if self.attention_type == "conv" and self.stride_qk[0] > 1:
            wind_view = rearrange(
                x,
                "b c (num_win_d num_patch_d patch_size_d) (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w) -> b num_win_d num_win_h num_win_w num_patch_d num_patch_h num_patch_w patch_size_d patch_size_h patch_size_w c",
                num_win_h=self.num_wind[0],
                num_patch_h=self.num_patch[0],
                patch_size_h=H // (self.num_patch[0] * self.num_wind[0]),
                num_win_w=self.num_wind[1],
                num_patch_w=self.num_patch[1],
                patch_size_w=W // (self.num_patch[1] * self.num_wind[1]),
                num_win_d=self.num_wind[2],
                num_patch_d=self.num_patch[2],
                patch_size_d=D // (self.num_patch[2] * self.num_wind[2]),
            )
        else:
            wind_view = rearrange(
                x,
                "b c (num_win_d num_patch_d patch_size_d) (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w) -> b num_win_d num_win_h num_win_w num_patch_d num_patch_h num_patch_w patch_size_d patch_size_h patch_size_w c",
                num_win_h=H // (self.num_patch[0] * self.patch_size[0]),
                num_patch_h=self.num_patch[0],
                patch_size_h=self.patch_size[0],
                num_win_w=W // (self.num_patch[1] * self.patch_size[1]),
                num_patch_w=self.num_patch[1],
                patch_size_w=self.patch_size[1],
                num_win_d=D // (self.num_patch[2] * self.patch_size[2]),
                num_patch_d=self.num_patch[2],
                patch_size_d=self.patch_size[2],
            )

        return wind_view

    def grid2im(self, x):
        """Reshape the windows back into the complete image."""
        (
            _B,
            num_win_d,
            num_win_h,
            num_win_w,
            num_patch_d,
            num_patch_h,
            num_patch_w,
            pd,
            ph,
            pw,
            _c,
        ) = x.shape

        im_view = rearrange(
            x,
            "b num_win_d num_win_h num_win_w num_patch_d num_patch_h num_patch_w patch_size_d patch_size_h patch_size_w c -> b c (num_win_d num_patch_d patch_size_d) (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w)",
            num_win_h=num_win_h,
            num_patch_h=num_patch_h,
            patch_size_h=ph,
            num_win_w=num_win_w,
            num_patch_w=num_patch_w,
            patch_size_w=pw,
            num_win_d=num_win_d,
            num_patch_d=num_patch_d,
            patch_size_d=pd,
        )
        return im_view

    def get_dimension_for_linear_mixer(self):
        return self.C_out * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]


# -------------------------------------------------------------------------------------------------
