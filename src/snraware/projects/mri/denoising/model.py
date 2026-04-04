"""
Model for MR denoising. The input is [B, 3, T/F, H, W] tensor for real/imag and g-factor along the channel dimension.
Output is [B, 2, T/F, H, W] tensor with denoised results.
"""

import hydra
import torch

from snraware.components.model import Conv2DExt, HRnet, SOAnet, Unet

# -----------------------------------------------------------------


def create_backbone(config, component_type, C_in, H, W, D):
    """
    Create model components from configuration.
    Return value is the created component and C_out for number of output channels.
    """
    if component_type.lower() == "hrnet":
        model = HRnet(config=config, input_feature_channels=C_in, H=H, W=W, D=D)
        C_out = model.get_number_of_output_channels()
    elif component_type.lower() == "unet":
        model = Unet(config=config, input_feature_channels=C_in, H=H, W=W, D=D)
        C_out = model.get_number_of_output_channels()
    elif component_type.lower() == "soanet":
        model = SOAnet(config=config, input_feature_channels=C_in, H=H, W=W, D=D)
        C_out = model.get_number_of_output_channels()

    else:
        raise NotImplementedError(f"Not implemented: {component_type}")

    return model, C_out


# -----------------------------------------------------------------


class DenoisingModel(torch.nn.Module):
    def __init__(self, config, D, H, W, C_in=3, C_out=2):
        super().__init__()

        self.config = config
        self.D = D
        self.H = H
        self.W = W
        self.C_in = C_in
        self.C_out = C_out

        backbone_C = config.backbone.num_of_channels

        self.pre = Conv2DExt(
            in_channels=C_in,
            out_channels=backbone_C,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            padding_mode="reflect",
            bias=True,
            channel_first=True,
        )

        # create the backbone model
        backbone_config = hydra.utils.instantiate(config.backbone)
        self.bk, bk_C_out = create_backbone(
            backbone_config, component_type=config.backbone.name, C_in=backbone_C, H=H, W=W, D=D
        )

        self.post = Conv2DExt(
            in_channels=bk_C_out,
            out_channels=C_out,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            padding_mode="reflect",
            bias=True,
            channel_first=True,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        res_pre = self.pre(x)

        res_backbone = self.bk(res_pre)[-1]

        C = res_pre.shape[1]

        # establish the residual connection
        residual_head = res_pre + res_backbone[:, :C, :, :, :]
        if C == res_backbone.shape[1]:
            res_backbone = residual_head
        else:
            res_backbone = torch.cat([residual_head, res_backbone[:, C:, :, :, :]], dim=1)

        y_hat = self.post(res_backbone)
        return y_hat
