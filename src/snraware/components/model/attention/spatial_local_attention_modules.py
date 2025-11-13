"""Spatial local attention module."""

from einops import rearrange

from .attention_modules import *


# -------------------------------------------------------------------------------------------------
class SpatialLocalAttention(CnnAttentionBase):
    """
    Multi-head attention model for the local patches. Every [H, W] frame is split into windows.
    Every window is split into patches. Attention is computed between all patches in a window.
    """

    def __init__(
        self,
        C_in,
        C_out=16,
        H=256,
        W=256,
        window_size=None,
        patch_size=None,
        num_wind=None,
        num_patch=None,
        attention_type="conv",
        n_head=8,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        stride_qk=(1, 1),
        att_dropout_p=0.0,
        cosine_att=False,
        normalize_Q_K=False,
        att_with_relative_position_bias=True,
        att_with_output_proj=True,
        with_timer=False,
    ):
        if num_patch is None:
            num_patch = [4, 4]
        if num_wind is None:
            num_wind = [8, 8]
        if patch_size is None:
            patch_size = [8, 8]
        if window_size is None:
            window_size = [64, 64]
        super().__init__(
            C_in=C_in,
            C_out=C_out,
            H=H,
            W=W,
            D=1,
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

        self.validate_window_patch()

        assert self.C_out * self.patch_size[0] * self.patch_size[1] % self.n_head == 0, (
            f"Number of pixels in a patch {self.C_out * self.patch_size[0] * self.patch_size[1]} should be divisible by number of heads {self.n_head}"
        )

        print(
            f"{Fore.GREEN}--> Spatial, local, win size {self.window_size}, patch size {self.patch_size}{Style.RESET_ALL}"
        )

        if attention_type == "conv":
            # key, query, value projections convolution
            # Wk, Wq, Wv
            self.key = Conv2DExt(
                C_in,
                C_out,
                kernel_size=kernel_size,
                stride=stride_qk,
                padding=padding,
                bias=True,
                channel_first=False,
            )
            self.query = Conv2DExt(
                C_in,
                C_out,
                kernel_size=kernel_size,
                stride=stride_qk,
                padding=padding,
                bias=True,
                channel_first=False,
            )
            self.value = Conv2DExt(
                C_in,
                C_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
                channel_first=False,
            )
        elif attention_type == "lin":
            # linear projections
            num_pixel_patch = self.patch_size[0] * self.patch_size[1]
            self.key = LinearGridExt(C_in * num_pixel_patch, C_out * num_pixel_patch, bias=True)
            self.query = LinearGridExt(C_in * num_pixel_patch, C_out * num_pixel_patch, bias=True)
            self.value = LinearGridExt(C_in * num_pixel_patch, C_out * num_pixel_patch, bias=True)
        else:
            raise NotImplementedError(f"Attention type not implemented: {attention_type}")

        if self.att_with_relative_position_bias:
            self.define_relative_position_bias_table(
                num_win_h=self.num_patch[0], num_win_w=self.num_patch[1]
            )
            self.define_relative_position_index(
                num_win_h=self.num_patch[0], num_win_w=self.num_patch[1]
            )

    def attention(self, k, q, v):
        B, T, num_win_h, num_win_w, num_patch_h_per_win, num_patch_w_per_win, ph, pw, C = k.shape
        _, _, _, _, _, _, ph_v, pw_v, _ = v.shape

        assert self.num_patch[0] == num_patch_h_per_win
        assert self.num_patch[1] == num_patch_w_per_win

        # format the window
        # use torch.sym_... to support onnx conversion
        hc = (C * ph * pw) // self.n_head
        hc_v = (C * ph_v * pw_v) // self.n_head

        # k, q, v will be [B, T, num_win_h*num_win_w, self.n_head, num_patch_h_per_win*num_patch_w_per_win, hc]
        tm = start_timer(enable=self.with_timer)
        k = k.reshape(
            (
                B,
                T,
                num_win_h * num_win_w,
                num_patch_h_per_win * num_patch_w_per_win,
                self.n_head,
                hc,
            )
        ).transpose(3, 4)
        q = q.reshape(
            (
                B,
                T,
                num_win_h * num_win_w,
                num_patch_h_per_win * num_patch_w_per_win,
                self.n_head,
                hc,
            )
        ).transpose(3, 4)
        v = v.reshape(
            (
                B,
                T,
                num_win_h * num_win_w,
                num_patch_h_per_win * num_patch_w_per_win,
                self.n_head,
                hc_v,
            )
        ).transpose(3, 4)
        end_timer(enable=self.with_timer, t=tm, msg="k, q, v - reshape")

        tm = start_timer(enable=self.with_timer)
        # [B, T, num_windows, num_heads, num_patches, hc] x [B, T, num_windows, num_heads, hc, num_patches] -> (B, T, num_windows, num_heads, num_patches, num_patches)
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
            relative_position_bias = self.get_relative_position_bias(
                num_patch_h_per_win, num_patch_w_per_win
            )
            att = att + relative_position_bias
        end_timer(enable=self.with_timer, t=tm, msg="att_with_relative_position_bias")

        tm = start_timer(enable=self.with_timer)
        att = self.attn_drop(att)
        end_timer(enable=self.with_timer, t=tm, msg="attn_drop")

        tm = start_timer(enable=self.with_timer)
        # (B, T, num_windows, num_heads, num_patches, num_patches) * (B, T, num_windows, num_heads, num_patches, hc)
        y = att @ v  # (B, T, num_windows, num_heads, num_patches, hc)
        y = y.transpose(3, 4)  # (B, T, num_windows, num_patches, num_heads, hc)
        y = torch.reshape(
            y,
            (B, T, num_win_h, num_win_w, num_patch_h_per_win, num_patch_w_per_win, ph_v, pw_v, C),
        )
        end_timer(enable=self.with_timer, t=tm, msg="att @ v")

        tm = start_timer(enable=self.with_timer)
        y = self.grid2im(y)
        end_timer(enable=self.with_timer, t=tm, msg="grid2im")

        return y

    def forward(self, x):
        _B, C, _T, _H, _W = x.size()

        assert C == self.C_in, (
            f"Input channel {C} does not match expected input channel {self.C_in}"
        )

        x = torch.permute(x, [0, 2, 1, 3, 4])

        if self.attention_type == "conv":
            tm = start_timer(enable=self.with_timer)
            k = self.key(x)  # (B, T, C, H_prime, W_prime)
            q = self.query(x)
            v = self.value(x)
            end_timer(enable=self.with_timer, t=tm, msg="compute k, q, v")

            tm = start_timer(enable=self.with_timer)
            k = self.im2grid(
                k
            )  # (B, T, num_win_h, num_win_w, num_patch_per_win, num_patch_per_win, Ps, Ps, C)
            q = self.im2grid(q)
            v = self.im2grid(v)
            end_timer(enable=self.with_timer, t=tm, msg="im2grid")
        else:
            tm = start_timer(enable=self.with_timer)
            x = self.im2grid(
                x
            )  # (B, T, num_win_h, num_win_w, num_patch_per_win, num_patch_per_win, Ps, Ps, C_in)
            end_timer(enable=self.with_timer, t=tm, msg="im2grid")

            tm = start_timer(enable=self.with_timer)
            k = self.key(
                x
            )  # (B, T, num_win_h, num_win_w, num_patch_per_win, num_patch_per_win, Ps, Ps, C)
            q = self.query(x)
            v = self.value(x)
            end_timer(enable=self.with_timer, t=tm, msg="compute k, q, v")

        y = self.attention(k, q, v)

        y = torch.permute(y, [0, 2, 1, 3, 4])

        tm = start_timer(enable=self.with_timer)
        y = self.output_proj(y)
        end_timer(enable=self.with_timer, t=tm, msg="output_proj")

        return y

    def im2grid(self, x):
        """Reshape the input into windows of local areas."""
        _, _, _, H, W = x.shape

        if self.attention_type == "conv" and self.stride_qk[0] > 1:
            wind_view = rearrange(
                x,
                "b t c (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w) -> b t num_win_h num_win_w num_patch_h num_patch_w patch_size_h patch_size_w c",
                num_win_h=self.num_wind[0],
                num_patch_h=self.num_patch[0],
                patch_size_h=H // (self.num_patch[0] * self.num_wind[0]),
                num_win_w=self.num_wind[1],
                num_patch_w=self.num_patch[1],
                patch_size_w=W // (self.num_patch[1] * self.num_wind[1]),
            )
        else:
            wind_view = rearrange(
                x,
                "b t c (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w) -> b t num_win_h num_win_w num_patch_h num_patch_w patch_size_h patch_size_w c",
                num_win_h=H // (self.num_patch[0] * self.patch_size[0]),
                num_patch_h=self.num_patch[0],
                patch_size_h=self.patch_size[0],
                num_win_w=W // (self.num_patch[1] * self.patch_size[1]),
                num_patch_w=self.num_patch[1],
                patch_size_w=self.patch_size[1],
            )

        return wind_view

    def grid2im(self, x):
        """Reshape the windows back into the complete image."""
        _b, _t, num_win_h, num_win_w, num_patch_h, num_patch_w, ph, pw, _c = x.shape

        im_view = rearrange(
            x,
            "b t num_win_h num_win_w num_patch_h num_patch_w patch_size_h patch_size_w c -> b t c (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w)",
            num_win_h=num_win_h,
            num_patch_h=num_patch_h,
            patch_size_h=ph,
            num_win_w=num_win_w,
            num_patch_w=num_patch_w,
            patch_size_w=pw,
        )
        return im_view

    def get_dimension_for_linear_mixer(self):
        return self.C_out * self.patch_size[0] * self.patch_size[1]


# -------------------------------------------------------------------------------------------------
