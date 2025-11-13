"""
Implement the temporal/frame cnn attention.
For a [B, C, T, H, W] input, the output is [B, C_out, T, H', W'] where usually H'=H and W'=W.
All frames along T dimension are used to calculate attention, i.e. all T frames are used to calculate attention for each frame.
"""

from .attention_modules import *

# -------------------------------------------------------------------------------------------------


class TemporalCnnAttentionBase(CnnAttentionBase):
    def __init__(
        self,
        C_in,
        C_out=16,
        H=128,
        W=128,
        is_causal=False,
        n_head=8,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        stride_qk=(1, 1),
        att_dropout_p=0.0,
        cosine_att=False,
        normalize_Q_K=False,
        att_with_output_proj=True,
        flash_att=False,
        with_timer=False,
    ):
        """
        @args:
            - is_causal (bool): whether to mask attention to imply causality.
        """
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
            att_with_output_proj=att_with_output_proj,
            flash_att=flash_att,
        )

        self.is_causal = is_causal
        self.stride_f = stride_qk[0]
        self.with_timer = with_timer

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

        self.register_buffer(
            "mask", torch.tril(torch.ones(1000, 1000, dtype=torch.bool)).view(1, 1, 1000, 1000)
        )

    def attention(self, k, q, v):
        raise NotImplementedError("This function is to be implemented by the subclass")

    def forward(self, x):
        """
        @args:
            x ([B, C, T, H, W]): Input of a batch of time series.

        @rets:
            y ([B, C_out, T, H', W']): logits
        """
        _B, C, _T, _H, _W = x.size()

        assert C == self.C_in, (
            f"Input channel {C} does not match expected input channel {self.C_in}"
        )

        x = torch.permute(x, [0, 2, 1, 3, 4])

        # apply the key, query and value matrix
        tm = start_timer(enable=self.with_timer)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        end_timer(enable=self.with_timer, t=tm, msg="compute k, q, v")

        y = self.attention(k, q, v)

        y = torch.permute(y, [0, 2, 1, 3, 4])

        tm = start_timer(enable=self.with_timer)
        y = self.output_proj(y)
        end_timer(enable=self.with_timer, t=tm, msg="output_proj")

        return y


# -------------------------------------------------------------------------------------------------


class TemporalChannelCnnAttention(TemporalCnnAttentionBase):
    """Multi-head attention is to split the channel dimension."""

    def __init__(
        self,
        C_in,
        C_out=16,
        H=128,
        W=128,
        is_causal=False,
        n_head=8,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        stride_qk=(1, 1),
        att_dropout_p=0.0,
        cosine_att=False,
        normalize_Q_K=False,
        att_with_output_proj=True,
        flash_att=False,
        with_timer=False,
    ):
        super().__init__(
            C_in=C_in,
            C_out=C_out,
            H=H,
            W=W,
            is_causal=is_causal,
            n_head=n_head,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            stride_qk=stride_qk,
            att_dropout_p=att_dropout_p,
            cosine_att=cosine_att,
            normalize_Q_K=normalize_Q_K,
            att_with_output_proj=att_with_output_proj,
            flash_att=flash_att,
            with_timer=with_timer,
        )

        assert self.C_out % self.n_head == 0, (
            f"Number of output channel {self.C_out} should be divisible by number of heads {self.n_head}"
        )

        print(
            f"{Fore.CYAN}--> Temporal, attention on channels only, C_in {C_in}, C_out {C_out}{Style.RESET_ALL}"
        )

    def attention(self, k, q, v):
        B, T, C_prime, H_prime, W_prime = k.shape

        hc = torch.div(C_prime, self.n_head, rounding_mode="floor")
        Hv, Wv = v.shape[-2:]

        tm = start_timer(enable=self.with_timer)
        k = k.view(B, T, self.n_head, hc, H_prime, W_prime).transpose(1, 2)
        q = q.view(B, T, self.n_head, hc, H_prime, W_prime).transpose(1, 2)
        v = v.view(B, T, self.n_head, hc, Hv, Wv).transpose(1, 2)
        end_timer(enable=self.with_timer, t=tm, msg="k, q, v - reshape")

        B, nh, T, hc, H_prime, W_prime = k.shape

        if (
            self.has_flash_attention
            and (not self.att_with_relative_position_bias)
            and (Hv == H_prime)
            and (Wv == W_prime)
        ):
            tm = start_timer(enable=self.with_timer)
            q = q.view(B, nh, T, hc * H_prime * W_prime)
            k = k.view(B, nh, T, hc * H_prime * W_prime)
            v = v.view(B, nh, T, hc * H_prime * W_prime)
            y = self.perform_flash_atten(q, k, v)
            end_timer(enable=self.with_timer, t=tm, msg="perform_flash_atten")
        else:
            q = q.view(B, nh, T, hc * H_prime * W_prime)
            k = k.view(B, nh, T, hc * H_prime * W_prime)

            tm = start_timer(enable=self.with_timer)
            # (B, nh, T, hc*H'*W') x (B, nh, hc*H'*W', T) -> (B, nh, T, T)
            if self.cosine_att:
                att = cosine_attention(q, k)
            else:
                if self.normalize_Q_K:
                    q, k = normalize_qk(q, k)

                att = (
                    q
                    @ k.transpose(-2, -1)
                    / torch.sqrt(torch.tensor(1.0 * hc * H_prime * W_prime))
                )
            end_timer(enable=self.with_timer, t=tm, msg="att")

            if self.is_causal:
                att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

            tm = start_timer(enable=self.with_timer)
            att = F.softmax(att, dim=-1)
            end_timer(enable=self.with_timer, t=tm, msg="softmax")

            tm = start_timer(enable=self.with_timer)
            att = self.attn_drop(att)
            end_timer(enable=self.with_timer, t=tm, msg="attn_drop")

            tm = start_timer(enable=self.with_timer)
            y = att @ v.view(B, nh, T, hc * Hv * Wv)
            end_timer(enable=self.with_timer, t=tm, msg="att @ v")

        tm = start_timer(enable=self.with_timer)
        y = y.transpose(1, 2).contiguous().view(B, T, self.C_out, Hv, Wv)
        end_timer(enable=self.with_timer, t=tm, msg="y.transpose")

        return y


# -------------------------------------------------------------------------------------------------


class TemporalCnnAttention(TemporalCnnAttentionBase):
    """Attention is on both channel and H and W dimensions."""

    def __init__(
        self,
        C_in,
        C_out=16,
        H=128,
        W=128,
        is_causal=False,
        n_head=8,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        stride_qk=(1, 1),
        att_dropout_p=0.0,
        cosine_att=False,
        normalize_Q_K=False,
        att_with_output_proj=True,
        flash_att=False,
        with_timer=False,
    ):
        super().__init__(
            C_in=C_in,
            C_out=C_out,
            H=H,
            W=W,
            is_causal=is_causal,
            n_head=n_head,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            stride_qk=stride_qk,
            att_dropout_p=att_dropout_p,
            cosine_att=cosine_att,
            normalize_Q_K=normalize_Q_K,
            att_with_output_proj=att_with_output_proj,
            flash_att=flash_att,
            with_timer=with_timer,
        )

        self.stride_f = stride_qk[0]

        assert self.C_out * H * W % self.n_head == 0, (
            f"Number of output {self.C_out * H * W} should be divisible by number of heads {self.n_head}"
        )

        print(
            f"{Fore.GREEN}--> Temporal, attention beyond channels, H {H}, W {W}, C_in {C_in}, C_out {C_out}{Style.RESET_ALL}"
        )

    def attention(self, k, q, v):
        B, T, C_prime, H_prime, W_prime = k.shape

        H = torch.div(C_prime * H_prime * W_prime, self.n_head, rounding_mode="floor")
        Hv, Wv = v.shape[-2:]

        tm = start_timer(enable=self.with_timer)
        k = k.view(B, T, self.n_head, H).transpose(1, 2)
        q = q.view(B, T, self.n_head, H).transpose(1, 2)
        v = v.view(B, T, self.n_head, H * self.stride_f * self.stride_f).transpose(1, 2)
        end_timer(enable=self.with_timer, t=tm, msg="k, q, v - reshape")

        # (B, nh, T, hc, H', W') x (B, nh, hc, H', W', T) -> (B, nh, T, T)
        if (
            self.has_flash_attention
            and (not self.att_with_relative_position_bias)
            and (Hv == H_prime)
            and (Wv == W_prime)
        ):
            tm = start_timer(enable=self.with_timer)
            y = self.perform_flash_atten(q, k, v)
            end_timer(enable=self.with_timer, t=tm, msg="perform_flash_atten")
        else:
            tm = start_timer(enable=self.with_timer)
            if self.cosine_att:
                att = cosine_attention(q, k)
            else:
                if self.normalize_Q_K:
                    q, k = normalize_qk(q, k)

                att = (q @ k.transpose(-2, -1)) * torch.sqrt(1.0 / H)
            end_timer(enable=self.with_timer, t=tm, msg="att")

            if self.is_causal:
                att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

            tm = start_timer(enable=self.with_timer)
            att = F.softmax(att, dim=-1)
            end_timer(enable=self.with_timer, t=tm, msg="softmax")

            tm = start_timer(enable=self.with_timer)
            att = self.attn_drop(att)
            end_timer(enable=self.with_timer, t=tm, msg="attn_drop")

            tm = start_timer(enable=self.with_timer)
            y = att @ v
            end_timer(enable=self.with_timer, t=tm, msg="att @ v")

        tm = start_timer(enable=self.with_timer)
        y = y.transpose(1, 2).contiguous().view(B, T, self.C_out, Hv, Wv)
        end_timer(enable=self.with_timer, t=tm, msg="y.transpose")

        return y


# -------------------------------------------------------------------------------------------------
