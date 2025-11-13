"""Implement the ViT style spatial attention on every [H, W] frame."""

from einops import rearrange

from .attention_modules import *


# -------------------------------------------------------------------------------------------------
class SpatialViTAttention(CnnAttentionBase):
    """
    Multi-head attention model for ViT style. An image is spatially splitted into windows.
    Attention matrix is computed between all windows. Number of pixels in a window are [window_size, window_size].
    """

    def __init__(
        self,
        C_in,
        C_out=16,
        H=128,
        W=128,
        window_size=None,
        num_wind=None,
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
        use_einsum=False,
        with_timer=False,
    ):
        """
        Input tensor is split into windows of size [window_size, window_size] for every [H, W] frame.
        Attention is computed between all windows.
        """
        if window_size is None:
            window_size = [32, 32]
        super().__init__(
            C_in=C_in,
            C_out=C_out,
            H=H,
            W=W,
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

        self.C_out = C_out
        self.attention_type = attention_type
        self.window_size = window_size
        self.num_wind = num_wind
        self.use_einsum = use_einsum

        self.set_and_check_wind()

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
            num_pixel_win = self.window_size[0] * self.window_size[1]
            self.key = LinearGridExt(C_in * num_pixel_win, C_out * num_pixel_win, bias=True)
            self.query = LinearGridExt(C_in * num_pixel_win, C_out * num_pixel_win, bias=True)
            self.value = LinearGridExt(C_in * num_pixel_win, C_out * num_pixel_win, bias=True)
        else:
            raise NotImplementedError(f"Attention type not implemented: {attention_type}")

        if self.att_with_relative_position_bias:
            self.define_relative_position_bias_table(
                num_win_h=self.num_wind[0], num_win_w=self.num_wind[1]
            )
            self.define_relative_position_index(
                num_win_h=self.num_wind[0], num_win_w=self.num_wind[1]
            )

    def attention(self, k, q, v):
        B, T, num_win_h, num_win_w, wh, ww, C = k.shape
        _, _, _, _, wh_v, ww_v, _ = v.shape

        assert self.num_wind[0] == num_win_h
        assert self.num_wind[1] == num_win_w

        # format the window
        hc = torch.div(C * wh * ww, self.n_head, rounding_mode="floor")
        hc_v = torch.div(C * wh_v * ww_v, self.n_head, rounding_mode="floor")

        tm = start_timer(enable=self.with_timer)
        q = q.reshape((B, T, num_win_h * num_win_w, self.n_head, hc)).transpose(2, 3)
        k = k.reshape((B, T, num_win_h * num_win_w, self.n_head, hc)).transpose(2, 3)
        v = v.reshape((B, T, num_win_h * num_win_w, self.n_head, hc_v)).transpose(2, 3)
        end_timer(enable=self.with_timer, t=tm, msg="k, q, v - reshape")

        tm = start_timer(enable=self.with_timer)
        # [B, T, num_heads, num_windows, hc] x [B, T, num_heads, hc, num_windows] -> (B, T, num_heads, num_windows, num_windows)
        if self.cosine_att:
            att = cosine_attention(q, k)
        else:
            if self.normalize_Q_K:
                q, k = normalize_qk(q, k)

            att = q @ k.transpose(-2, -1) * torch.sqrt(1.0 / hc)

        end_timer(enable=self.with_timer, t=tm, msg="att")

        tm = start_timer(enable=self.with_timer)
        att = F.softmax(att, dim=-1)
        end_timer(enable=self.with_timer, t=tm, msg="softmax")

        tm = start_timer(enable=self.with_timer)
        # add the relative positional bias
        if self.att_with_relative_position_bias:
            relative_position_bias = self.get_relative_position_bias(num_win_h, num_win_w)
            att = att + relative_position_bias
        end_timer(enable=self.with_timer, t=tm, msg="att_with_relative_position_bias")

        tm = start_timer(enable=self.with_timer)
        att = self.attn_drop(att)
        end_timer(enable=self.with_timer, t=tm, msg="attn_drop")

        tm = start_timer(enable=self.with_timer)
        # (B, T, num_heads, num_windows, num_windows) * (B, T, num_heads, num_windows, hc)
        y = att @ v
        y = y.transpose(2, 3)  # (B, T, num_windows, num_heads, hc)
        y = torch.reshape(y, (B, T, num_win_h, num_win_w, wh_v, ww_v, C))
        end_timer(enable=self.with_timer, t=tm, msg="att @ v")

        tm = start_timer(enable=self.with_timer)
        y = self.grid2im(y)
        end_timer(enable=self.with_timer, t=tm, msg="grid2im")

        return y

    def forward(self, x):
        _B, C, _T, H, W = x.size()

        if self.attention_type == "lin" and self.att_with_relative_position_bias:
            assert H == self.H and W == self.W, (
                "For lin attention_type with relative position bias, input H and W have to be the same as the class declaration."
            )

        assert C == self.C_in, (
            f"Input channel {C} does not match expected input channel {self.C_in}"
        )
        assert H % self.num_wind[0] == 0, (
            f"Height {H} should be divisible by window num {self.num_wind[0]}"
        )
        assert W % self.num_wind[1] == 0, (
            f"Width {W} should be divisible by window num {self.num_wind[1]}"
        )

        x = torch.permute(x, [0, 2, 1, 3, 4])

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
            x = self.im2grid(x)
            k = self.key(x)
            q = self.query(x)
            v = self.value(x)

        y = self.attention(k, q, v)

        y = torch.permute(y, [0, 2, 1, 3, 4])

        tm = start_timer(enable=self.with_timer)
        y = self.output_proj(y)
        end_timer(enable=self.with_timer, t=tm, msg="output_proj")

        return y

    def im2grid(self, x):
        """Reshape the input into windows of local areas."""
        _b, _t, _c, h, w = x.shape

        wind_view = rearrange(
            x,
            "b t c (num_win_h win_size_h) (num_win_w win_size_w) -> b t num_win_h num_win_w win_size_h win_size_w c",
            num_win_h=self.num_wind[0],
            win_size_h=h // self.num_wind[0],
            num_win_w=self.num_wind[1],
            win_size_w=w // self.num_wind[1],
        )

        return wind_view

    def grid2im(self, x):
        """Reshape the windows back into the complete image."""
        # b, t, num_win_h, num_win_w, c, wh, ww = x.shape

        _b, _t, num_win_h, num_win_w, wh, ww, _c = x.shape

        im_view = rearrange(
            x,
            "b t num_win_h num_win_w win_size_h win_size_w c -> b t c (num_win_h win_size_h) (num_win_w win_size_w)",
            num_win_h=num_win_h,
            win_size_h=wh,
            num_win_w=num_win_w,
            win_size_w=ww,
        )

        return im_view

    def get_dimension_for_linear_mixer(self):
        return self.C_out * self.window_size[0] * self.window_size[1]


# -------------------------------------------------------------------------------------------------
