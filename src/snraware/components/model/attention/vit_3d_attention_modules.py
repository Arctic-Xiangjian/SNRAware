"""Implement the ViT style spatial attention, for 3D patches."""

from einops import rearrange

from .attention_modules import *

# -------------------------------------------------------------------------------------------------
# CNN attention with the ViT style - an image is split into windows. Attention coefficients are computed among all windows.


class ViT3DAttention(CnnAttentionBase):
    """
    Multi-head cnn attention with the ViT style window split.
    Attention matrix is computed between all windows.
    Number of pixels in a window are [window_size, window_size, window_size].
    """

    def __init__(
        self,
        C_in,
        C_out=16,
        H=128,
        W=128,
        D=32,
        window_size=None,
        num_wind=None,
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
        Input to the attention layer has the size [B, C, T, H, W]
        Output has the size [B, output_channels, T, H, W].

        Either window_size or num_wind should be supplied.
        window_size is the number of pixels per window, along H and W and T.
        num_wind is the number of windows along H and W.
        if both are supplied, num_wind has the priority.

        @args:
            - window_size (int): number of pixels in a window
        """
        if num_wind is None:
            num_wind = [8, 8, 4]
        if window_size is None:
            window_size = [16, 16, 8]
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

        self.C_out = C_out
        self.attention_type = attention_type
        self.window_size = window_size
        self.num_wind = num_wind

        self.set_and_check_wind()

        if len(self.window_size) == 2:
            self.window_size.append(1)
        if len(self.num_wind) == 2:
            self.num_wind.append(1)

        print(
            f"{Fore.GREEN}--> Spatial, ViT3D, H {H}, W {W}, D {D}, win size {self.window_size}{Style.RESET_ALL}"
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
            num_pixel_win = self.window_size[0] * self.window_size[1] * self.window_size[2]
            self.key = LinearGrid3DExt(C_in * num_pixel_win, C_out * num_pixel_win, bias=True)
            self.query = LinearGrid3DExt(C_in * num_pixel_win, C_out * num_pixel_win, bias=True)
            self.value = LinearGrid3DExt(C_in * num_pixel_win, C_out * num_pixel_win, bias=True)
        else:
            raise NotImplementedError(f"Attention type not implemented: {attention_type}")

        if self.att_with_relative_position_bias:
            self.define_relative_position_bias_table_3D(
                num_win_h=self.num_wind[0], num_win_w=self.num_wind[1], num_win_d=self.num_wind[2]
            )
            self.define_relative_position_index_3D(
                num_win_h=self.num_wind[0], num_win_w=self.num_wind[1], num_win_d=self.num_wind[2]
            )

    def attention(self, k, q, v):
        B, num_win_d, num_win_h, num_win_w, wd, wh, ww, C = k.shape
        _, _, _, _, wd_v, wh_v, ww_v, _ = v.shape

        # format the window
        hc = torch.div(C * wd * wh * ww, self.n_head, rounding_mode="floor")
        hc_v = torch.div(C * wd_v * wh_v * ww_v, self.n_head, rounding_mode="floor")

        tm = start_timer(enable=self.with_timer)
        q = q.reshape((B, num_win_d * num_win_h * num_win_w, self.n_head, hc)).transpose(1, 2)
        k = k.reshape((B, num_win_d * num_win_h * num_win_w, self.n_head, hc)).transpose(1, 2)
        v = v.reshape((B, num_win_d * num_win_h * num_win_w, self.n_head, hc_v)).transpose(1, 2)
        end_timer(enable=self.with_timer, t=tm, msg="k, q, v - reshape")

        tm = start_timer(enable=self.with_timer)
        # [B, num_heads, num_windows, hc] x [B, num_heads, hc, num_windows] -> (B, num_heads, num_windows, num_windows)
        if self.cosine_att:
            att = cosine_attention(q, k)
        else:
            if self.normalize_Q_K:
                q, k = normalize_qk(q, k)

            att = q @ k.transpose(-2, -1) * torch.sqrt(1.0 / hc.clone().detach())

        end_timer(enable=self.with_timer, t=tm, msg="att")

        tm = start_timer(enable=self.with_timer)
        att = F.softmax(att, dim=-1)
        end_timer(enable=self.with_timer, t=tm, msg="softmax")

        tm = start_timer(enable=self.with_timer)
        # add the relative positional bias
        if self.att_with_relative_position_bias:
            relative_position_bias = self.get_relative_position_bias_3D(
                num_win_h, num_win_w, num_win_d
            )
            att = att + relative_position_bias
        end_timer(enable=self.with_timer, t=tm, msg="att_with_relative_position_bias_3D")

        tm = start_timer(enable=self.with_timer)
        att = self.attn_drop(att)
        end_timer(enable=self.with_timer, t=tm, msg="attn_drop")

        tm = start_timer(enable=self.with_timer)
        # (B, num_heads, num_windows, num_windows) * (B, num_heads, num_windows, hc)
        y = att @ v
        y = y.transpose(1, 2)  # (B, num_windows, num_heads, hc)
        y = torch.reshape(y, (B, num_win_d, num_win_h, num_win_w, wd_v, wh_v, ww_v, C))
        end_timer(enable=self.with_timer, t=tm, msg="att @ v")

        tm = start_timer(enable=self.with_timer)
        y = self.grid2im(y)
        end_timer(enable=self.with_timer, t=tm, msg="grid2im")

        return y

    def forward(self, x):
        """
        @args:
            x ([B, C, D, H, W]): Input of a batch of time series.

        @rets:
            y ([B, C_out, D, H', W']): output tensor
        """
        _B, C, D, H, W = x.size()

        if self.attention_type == "lin" and self.att_with_relative_position_bias:
            assert H == self.H and W == self.W and D == self.D, (
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
        assert D % self.num_wind[2] == 0, (
            f"Depth {D} should be divisible by window num {self.num_wind[2]}"
        )

        if self.attention_type == "conv":
            tm = start_timer(enable=self.with_timer)
            k = self.key(x)
            q = self.query(x)
            v = self.value(x)
            end_timer(enable=self.with_timer, t=tm, msg="compute k, q, v")

            tm = start_timer(enable=self.with_timer)
            k = self.im2grid(k)  # (B, num_win_d, num_win_h, num_win_w, wd, wh, ww, C)
            q = self.im2grid(q)
            v = self.im2grid(v)
            end_timer(enable=self.with_timer, t=tm, msg="im2grid")
        else:
            x = self.im2grid(x)  # (B, num_win_d, num_win_h, num_win_w, wd, wh, ww, C)
            k = self.key(x)
            q = self.query(x)
            v = self.value(x)

        y = self.attention(k, q, v)

        tm = start_timer(enable=self.with_timer)
        y = self.output_proj(y)
        end_timer(enable=self.with_timer, t=tm, msg="output_proj")

        return y

    def im2grid(self, x):
        """Reshape the input into windows of local areas."""
        _b, _c, d, h, w = x.shape

        wind_view = rearrange(
            x,
            "b c (num_win_d win_size_d) (num_win_h win_size_h) (num_win_w win_size_w) -> b num_win_d num_win_h num_win_w win_size_d win_size_h win_size_w c",
            num_win_d=self.num_wind[2],
            win_size_d=d // self.num_wind[2],
            num_win_h=self.num_wind[0],
            win_size_h=h // self.num_wind[0],
            num_win_w=self.num_wind[1],
            win_size_w=w // self.num_wind[1],
        )

        return wind_view

    def grid2im(self, x):
        """Reshape the windows back into the complete image."""
        _b, num_win_d, num_win_h, num_win_w, wd, wh, ww, _c = x.shape

        im_view = rearrange(
            x,
            "b num_win_d num_win_h num_win_w win_size_d win_size_h win_size_w c -> b c (num_win_d win_size_d) (num_win_h win_size_h) (num_win_w win_size_w)",
            num_win_d=num_win_d,
            win_size_d=wd,
            num_win_h=num_win_h,
            win_size_h=wh,
            num_win_w=num_win_w,
            win_size_w=ww,
        )

        return im_view

    def get_dimension_for_linear_mixer(self):
        return self.C_out * self.window_size[0] * self.window_size[1] * self.window_size[2]


# -------------------------------------------------------------------------------------------------
