"""Spatial global attention modules."""

from einops import rearrange

from .attention_modules import *


# -------------------------------------------------------------------------------------------------
class SpatialGlobalAttention(CnnAttentionBase):
    """
    Multi-head attention model for the global patching.
    Number of pixels in a window are [window_size, window_size].
    Number of pixels in a patch are [patch_size, patch_size].
    The tensor is split into windows of size [window_size, window_size] for every [H, W] frame.
    Every window is split into patches of size [patch_size, patch_size].
    Attention is computed between all corresponding patches among all windows.
    """

    def __init__(
        self,
        C_in,
        C_out=16,
        H=128,
        W=128,
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
        shuffle_in_window=False,
        with_timer=False,
    ):
        """
        @args:
            - shuffle_in_window (bool): If True, shuffle the order of patches in all windows; this will avoid the same set of patches are always inputted into the attention, but introducing randomness.
        """
        if num_patch is None:
            num_patch = [4, 4]
        if num_wind is None:
            num_wind = [4, 4]
        if patch_size is None:
            patch_size = [8, 8]
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

        self.attention_type = attention_type
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_wind = num_wind
        self.num_patch = num_patch
        self.shuffle_in_window = shuffle_in_window

        self.set_and_check_wind()
        self.set_and_check_patch()

        self.validate_window_patch()

        assert self.C_out * self.patch_size[0] * self.patch_size[1] % self.n_head == 0, (
            f"Number of pixels in a window {self.C_out * self.patch_size[0] * self.patch_size[1]} should be divisible by number of heads {self.n_head}"
        )

        print(
            f"{Fore.GREEN}--> Spatial, global, win size {self.window_size}, patch size {self.patch_size}{Style.RESET_ALL}"
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
                num_win_h=self.num_wind[0], num_win_w=self.num_wind[1]
            )
            self.define_relative_position_index(
                num_win_h=self.num_wind[0], num_win_w=self.num_wind[1]
            )

    def attention(self, k, q, v):
        B, T, num_patch_h_per_win, num_patch_w_per_win, num_win_h, num_win_w, ph, pw, C = k.shape
        ph_v, pw_v, _ = v.shape[-3:]

        # format the window
        hc = (C * ph * pw) // self.n_head
        hc_v = (C * ph_v * pw_v) // self.n_head

        tm = start_timer(enable=self.with_timer)
        # k, q, v will be [B, T, num_patch_h_per_win*num_patch_w_per_win, self.n_head, num_win_h*num_win_w, hc]
        k = k.reshape(
            (
                B,
                T,
                num_patch_h_per_win * num_patch_w_per_win,
                num_win_h * num_win_w,
                self.n_head,
                hc,
            )
        ).transpose(3, 4)
        q = q.reshape(
            (
                B,
                T,
                num_patch_h_per_win * num_patch_w_per_win,
                num_win_h * num_win_w,
                self.n_head,
                hc,
            )
        ).transpose(3, 4)
        v = v.reshape(
            (
                B,
                T,
                num_patch_h_per_win * num_patch_w_per_win,
                num_win_h * num_win_w,
                self.n_head,
                hc_v,
            )
        ).transpose(3, 4)
        end_timer(enable=self.with_timer, t=tm, msg="k, q, v - reshape")

        if self.shuffle_in_window:
            tm = start_timer(enable=self.with_timer)

            # random permute within a window
            patch_indexes = torch.zeros(
                [num_win_h * num_win_w, num_patch_h_per_win * num_patch_w_per_win],
                dtype=torch.long,
            )
            for w in range(num_win_h * num_win_w):
                patch_indexes[w, :] = torch.randperm(num_patch_h_per_win * num_patch_w_per_win)

            reverse_patch_indexes = num_patch_h_per_win * num_patch_w_per_win - 1 - patch_indexes
            reverse_patch_indexes = torch.flip(reverse_patch_indexes, dims=(1,))

            k_shuffled = torch.clone(k)
            q_shuffled = torch.clone(q)
            v_shuffled = torch.clone(v)

            for w in range(num_win_h * num_win_w):
                k_shuffled[:, :, :, :, w] = k[:, :, patch_indexes[w, :], :, w]
                q_shuffled[:, :, :, :, w] = q[:, :, patch_indexes[w, :], :, w]
                v_shuffled[:, :, :, :, w] = v[:, :, patch_indexes[w, :], :, w]

            k = k_shuffled
            q = q_shuffled
            v = v_shuffled

            end_timer(enable=self.with_timer, t=tm, msg="shuffle_in_window")

        tm = start_timer(enable=self.with_timer)
        # [B, T, num_patches, num_heads, num_windows, hc] x [B, T, num_patches, num_heads, hc, num_windows] -> (B, T, num_patches, num_heads, num_windows, num_windows)
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
            relative_position_bias = self.get_relative_position_bias(num_win_h, num_win_w)
            att = att + relative_position_bias
        end_timer(enable=self.with_timer, t=tm, msg="relative_position_bias")

        tm = start_timer(enable=self.with_timer)
        att = self.attn_drop(att)
        end_timer(enable=self.with_timer, t=tm, msg="attn_drop")

        tm = start_timer(enable=self.with_timer)
        # (B, T, num_patches, num_heads, num_windows, num_windows) * (B, T, num_patches, num_heads, num_windows, hc_v)
        y = att @ v  # (B, T, num_patches, num_heads, num_windows, hc_v)
        end_timer(enable=self.with_timer, t=tm, msg="att @ v")

        tm = start_timer(enable=self.with_timer)
        y = y.transpose(3, 4)  # (B, T, num_patches, num_windows, num_heads, hc_v)
        end_timer(enable=self.with_timer, t=tm, msg="y.transpose")

        tm = start_timer(enable=self.with_timer)
        if self.shuffle_in_window:
            y_restored = torch.clone(y)
            for w in range(num_win_h * num_win_w):
                y_restored[:, :, :, w] = y[:, :, reverse_patch_indexes[w, :], w]

            y = torch.reshape(
                y_restored,
                (
                    B,
                    T,
                    num_patch_h_per_win,
                    num_patch_w_per_win,
                    num_win_h,
                    num_win_w,
                    ph_v,
                    pw_v,
                    C,
                ),
            )
        else:
            y = torch.reshape(
                y,
                (
                    B,
                    T,
                    num_patch_h_per_win,
                    num_patch_w_per_win,
                    num_win_h,
                    num_win_w,
                    ph_v,
                    pw_v,
                    C,
                ),
            )
        end_timer(enable=self.with_timer, t=tm, msg="y reshape")

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
            )  # (B, T, num_patch_per_win, num_patch_per_win, num_win_h, num_win_w, Ps, Ps, C)
            q = self.im2grid(q)
            v = self.im2grid(v)
            end_timer(enable=self.with_timer, t=tm, msg="im2grid")
        else:
            tm = start_timer(enable=self.with_timer)
            x = self.im2grid(
                x
            )  # (B, T, num_patch_per_win, num_patch_per_win, num_win_h, num_win_w, Ps, Ps, C_in)
            end_timer(enable=self.with_timer, t=tm, msg="im2grid")

            tm = start_timer(enable=self.with_timer)
            k = self.key(
                x
            )  # (B, T, num_patch_per_win, num_patch_per_win, num_win_h, num_win_w, Ps, Ps, C)
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
        _b, _t, _c, h, w = x.shape

        if self.attention_type == "conv" and self.stride_qk[0] > 1:
            wind_view = rearrange(
                x,
                "b t c (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w) -> b t num_patch_h num_patch_w num_win_h num_win_w patch_size_h patch_size_w c",
                num_win_h=self.num_wind[0],
                num_patch_h=self.num_patch[0],
                patch_size_h=h // (self.num_wind[0] * self.num_patch[0]),
                num_win_w=self.num_wind[1],
                num_patch_w=self.num_patch[1],
                patch_size_w=w // (self.num_wind[0] * self.num_patch[1]),
            )
        else:
            wind_view = rearrange(
                x,
                "b t c (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w) -> b t num_patch_h num_patch_w num_win_h num_win_w patch_size_h patch_size_w c",
                num_win_h=self.num_wind[0],
                num_patch_h=h // (self.num_wind[0] * self.patch_size[0]),
                patch_size_h=self.patch_size[0],
                num_win_w=self.num_wind[1],
                num_patch_w=w // (self.num_wind[0] * self.patch_size[1]),
                patch_size_w=self.patch_size[1],
            )

        return wind_view

    def grid2im(self, x):
        """Reshape the windows back into the complete image."""
        _b, _t, num_patch_h, num_patch_w, num_win_h, num_win_w, ph, pw, _c = x.shape

        # im_view = torch.permute(x, (0, 1, 8, 4, 2, 6, 5, 3, 7))

        # im_view = rearrange(im_view, 'b t c num_win_h num_patch_h patch_size_h num_win_w num_patch_w patch_size_w -> b t c (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w)',
        #                       num_win_h=num_win_h, num_patch_h=num_patch_h, patch_size_h=ph,
        #                       num_win_w=num_win_w, num_patch_w=num_patch_w, patch_size_w=pw)

        im_view = rearrange(
            x,
            "b t num_patch_h num_patch_w num_win_h num_win_w patch_size_h patch_size_w c -> b t c (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w)",
            num_win_h=num_win_h,
            num_patch_h=num_patch_h,
            patch_size_h=ph,
            num_win_w=num_win_w,
            num_patch_w=num_patch_w,
            patch_size_w=pw,
        )

        # im_view = torch.permute(x, (0, 1, 4, 5, 2, 3, 6, 7, 8))

        # im_view = rearrange(x, 'b t num_patch_h num_patch_w num_win_h num_win_w patch_size_h patch_size_w c -> b t c (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w)',
        #                       num_win_h=num_win_h, num_patch_h=num_patch_h, patch_size_h=ph,
        #                       num_win_w=num_win_w, num_patch_w=num_patch_w, patch_size_w=pw)
        return im_view

    def get_dimension_for_linear_mixer(self):
        return self.C_out * self.patch_size[0] * self.patch_size[1]


# -------------------------------------------------------------------------------------------------
