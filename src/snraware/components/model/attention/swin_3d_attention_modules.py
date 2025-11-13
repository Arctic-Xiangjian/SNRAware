"""Implement the SWIN 3D attention."""

from functools import reduce
from operator import mul

import numpy as np
from einops import rearrange

from .attention_modules import *

# -------------------------------------------------------------------------------------------------
# SWIN attention includes two steps, one is the normal ViT type attention and another is the shifted window attention


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size.

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0],
        window_size[0],
        H // window_size[1],
        window_size[1],
        W // window_size[2],
        window_size[2],
        C,
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_partition_image(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size.

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image.

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


# -------------------------------------------------------------------------------------------------


def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for h in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for w in (
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(attn_mask == 0, 0.0)
    return attn_mask


# -------------------------------------------------------------------------------------------------


class Swin3DAttention(CnnAttentionBase):
    """Multi-head attention model for the swin. Number of pixels in a window are [window_size, window_size, window_size]."""

    def __init__(
        self,
        C_in,
        C_out=16,
        H=256,
        W=256,
        D=128,
        window_size=None,
        num_wind=None,
        attention_type="conv",
        n_head=8,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        stride_qk=(1, 1, 1),
        shifted=False,
        att_dropout_p=0.0,
        cosine_att=False,
        normalize_Q_K=False,
        att_with_relative_position_bias=True,
        att_with_output_proj=True,
        with_timer=False,
    ):
        """shifted: if Ture, run the shifted window swin; if False, run the normal window swin."""
        if num_wind is None:
            num_wind = [8, 8, 8]
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
        self.num_wind = num_wind

        self.shifted = shifted

        self.set_and_check_wind()

        if len(self.window_size) == 2:
            self.window_size.append(1)
        if len(self.num_wind) == 2:
            self.num_wind.append(1)

        num_pixel_win = self.window_size[0] * self.window_size[1] * self.window_size[2]
        assert self.C_out * num_pixel_win % self.n_head == 0, (
            f"Number of pixels in a patch {self.C_out * num_pixel_win} should be divisible by number of heads {self.n_head}"
        )

        self.win_size_h = self.window_size[0]
        self.win_size_w = self.window_size[1]
        self.win_size_d = self.window_size[2]

        self.shift_size_h = 0
        self.shift_size_w = 0
        self.shift_size_d = 0
        if self.shifted:
            self.shift_size_h = max(self.win_size_h // 2, 1)
            self.shift_size_w = max(self.win_size_w // 2, 1)
            self.shift_size_d = max(self.win_size_d // 2, 1)

        print(
            f"{Fore.YELLOW}--> Swin, H {H}, W {W}, D {D}, win size {self.window_size}, num_wind {self.num_wind}"
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
            self.key = LinearGrid3DExt(C_in * num_pixel_win, C_out * num_pixel_win, bias=True)
            self.query = LinearGrid3DExt(C_in * num_pixel_win, C_out * num_pixel_win, bias=True)
            self.value = LinearGrid3DExt(C_in * num_pixel_win, C_out * num_pixel_win, bias=True)
        else:
            raise NotImplementedError(f"Attention type not implemented: {attention_type}")

        if self.att_with_relative_position_bias:
            self.define_relative_position_bias_table_3D(
                num_win_h=self.win_size_h, num_win_w=self.win_size_w, num_win_d=self.win_size_d
            )
            self.define_relative_position_index_3D(
                num_win_h=self.win_size_h, num_win_w=self.win_size_w, num_win_d=self.win_size_d
            )

    def attention(self, k, q, v, mask_matrix):
        B, C, D, H, W = k.shape
        _, _, _D_v, _H_v, _W_v = v.shape

        k = torch.permute(k, [0, 2, 3, 4, 1])  # [B, D, H, W, C]
        q = torch.permute(q, [0, 2, 3, 4, 1])
        v = torch.permute(v, [0, 2, 3, 4, 1])

        window_size, shift_size = get_window_size(
            (D, H, W),
            [self.win_size_d, self.win_size_h, self.win_size_w],
            [self.shift_size_d, self.shift_size_h, self.shift_size_w],
        )

        # pad feature maps to multiples of window sizelspci | grep VGA
        tm = start_timer(enable=self.with_timer)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        if pad_d1 > 0 or pad_b > 0 or pad_r > 0:
            k = F.pad(k, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            q = F.pad(q, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            v = F.pad(v, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = k.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_k = torch.roll(
                k, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3)
            )
            shifted_q = torch.roll(
                q, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3)
            )
            shifted_v = torch.roll(
                v, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3)
            )
            attn_mask = mask_matrix
        else:
            shifted_k = k
            shifted_q = q
            shifted_v = v
            attn_mask = None
        end_timer(enable=self.with_timer, t=tm, msg="shifted, k, q, v")

        tm = start_timer(enable=self.with_timer)
        k = window_partition(shifted_k, window_size)  # B*num_win, wd*wh*ww, C
        q = window_partition(shifted_q, window_size)
        v = window_partition(shifted_v, window_size)
        end_timer(enable=self.with_timer, t=tm, msg="window_partition")

        # ---------------------------------------------------
        # compute attention matrix
        # attn_windows = self.attn(k_windows, q_windows, v_windows, mask=attn_mask)

        B_ = k.shape[0]
        N = k.shape[1]
        hc = C // self.n_head

        N_v = v.shape[1]

        k = torch.reshape(k, [B_, N, self.n_head, hc]).permute(0, 2, 1, 3)
        q = torch.reshape(q, [B_, N, self.n_head, hc]).permute(0, 2, 1, 3)
        v = torch.reshape(k, [B_, N_v, self.n_head, hc]).permute(0, 2, 1, 3)

        tm = start_timer(enable=self.with_timer)
        if self.cosine_att:
            attn = cosine_attention(q, k)
        else:
            if self.normalize_Q_K and k.shape[-1] > 0:
                q, k = normalize_qk(q, k)

            attn = q @ k.transpose(-2, -1) * torch.sqrt(torch.tensor(1.0 / hc))
        end_timer(enable=self.with_timer, t=tm, msg="attn")

        N = window_size[0] * window_size[1] * window_size[2]

        tm = start_timer(enable=self.with_timer)
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.n_head, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_head, N, N)
            attn = F.softmax(attn, dim=-1)
            end_timer(enable=self.with_timer, t=tm, msg="attn_mask and softmax")
        else:
            attn = F.softmax(attn, dim=-1)
            end_timer(enable=self.with_timer, t=tm, msg="softmax")

        tm = start_timer(enable=self.with_timer)
        if self.att_with_relative_position_bias:
            relative_position_bias = self.get_relative_position_bias_3D(
                self.win_size_h, self.win_size_w, self.win_size_d
            )
            attn = attn + relative_position_bias
        end_timer(enable=self.with_timer, t=tm, msg="att_with_relative_position_bias")

        tm = start_timer(enable=self.with_timer)
        attn = self.attn_drop(attn)
        end_timer(enable=self.with_timer, t=tm, msg="attn_drop")

        tm = start_timer(enable=self.with_timer)
        y = attn @ v
        y = y.transpose(1, 2).reshape(B_, N, C)
        end_timer(enable=self.with_timer, t=tm, msg="att @ v")

        # ---------------------------------------------------

        shifted_y = window_reverse(y, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            y = torch.roll(
                shifted_y,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            y = shifted_y

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            y = y[:, :D, :H, :W, :].contiguous()

        y = torch.permute(y, [0, 4, 1, 2, 3])  # from [B, D', H', W', C] to [B, C, D', H', W']

        return y

    def forward(self, x):
        _B, C, D, H, W = x.size()

        assert C == self.C_in, (
            f"Input channel {C} does not match expected input channel {self.C_in}"
        )

        if self.attention_type == "conv":
            tm = start_timer(enable=self.with_timer)
            k = self.key(x)
            q = self.query(x)
            v = self.value(x)
            end_timer(enable=self.with_timer, t=tm, msg="compute k, q, v")
        else:
            tm = start_timer(enable=self.with_timer)
            x = self.im2grid(x)
            end_timer(enable=self.with_timer, t=tm, msg="im2grid")

            tm = start_timer(enable=self.with_timer)
            k = self.key(x)
            q = self.query(x)
            v = self.value(x)
            end_timer(enable=self.with_timer, t=tm, msg="compute k, q, v")

            tm = start_timer(enable=self.with_timer)
            k = self.grid2im(k)
            q = self.grid2im(q)
            v = self.grid2im(v)
            end_timer(enable=self.with_timer, t=tm, msg="grid2im")

        tm = start_timer(enable=self.with_timer)
        if self.shifted:
            _, _, D, H, W = k.shape
            # compute mask_matrix
            window_size = [self.win_size_d, self.win_size_h, self.win_size_w]
            shift_size = [self.shift_size_d, self.shift_size_h, self.shift_size_w]
            window_size, shift_size = get_window_size((D, H, W), window_size, shift_size)
            Dp = int(np.ceil(D / window_size[0])) * window_size[0]
            Hp = int(np.ceil(H / window_size[1])) * window_size[1]
            Wp = int(np.ceil(W / window_size[2])) * window_size[2]
            mask_matrix = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        else:
            mask_matrix = None
        end_timer(enable=self.with_timer, t=tm, msg="compute_mask")

        # k, q, v are [B, C, D, H, W]
        tm = start_timer(enable=self.with_timer)
        y = self.attention(k, q, v, mask_matrix)
        end_timer(enable=self.with_timer, t=tm, msg="self.attention")

        tm = start_timer(enable=self.with_timer)
        y = self.output_proj(y)
        end_timer(enable=self.with_timer, t=tm, msg="output_proj")

        return y

    def im2grid(self, x):
        """Reshape the input into windows of local areas."""
        _B, _C, D, H, W = x.shape

        wind_view = rearrange(
            x,
            "b c (num_win_d win_size_d) (num_win_h win_size_h) (num_win_w win_size_w) -> b num_win_d num_win_h num_win_w win_size_d win_size_h win_size_w c",
            num_win_h=self.num_wind[0],
            win_size_h=H // self.num_wind[0],
            num_win_w=self.num_wind[1],
            win_size_w=W // self.num_wind[1],
            num_win_d=self.num_wind[2],
            win_size_d=D // self.num_wind[2],
        )

        return wind_view

    def grid2im(self, x):
        """Reshape the windows back into the complete image."""
        _b, num_win_d, num_win_h, num_win_w, win_size_d, win_size_h, win_size_w, _c = x.shape

        im_view = rearrange(
            x,
            "b num_win_d num_win_h num_win_w win_size_d win_size_h win_size_w c -> b c (num_win_d win_size_d) (num_win_h win_size_h) (num_win_w win_size_w)",
            num_win_h=num_win_h,
            win_size_h=win_size_h,
            num_win_w=num_win_w,
            win_size_w=win_size_w,
            num_win_d=num_win_d,
            win_size_d=win_size_d,
        )
        return im_view

    def get_dimension_for_linear_mixer(self):
        return self.C_out * self.window_size[0] * self.window_size[1] * self.window_size[2]


# -------------------------------------------------------------------------------------------------
