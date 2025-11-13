import torch
from hydra import compose, initialize

from snraware.components.heads import (
    PoolLinear,
    PreConv2D,
    SimpleConv2d,
)
from snraware.components.setup import set_seed

# -----------------------------------------------------------------


def create_heads(config, component_type, C_in, C_out, H, W, D):
    """
    Create model head components from configuration.
    Return value is the created component.

    C_in is the number of input channels.
    H, W, D is the tensor size.

    All components process 5D tensor [B, C, D, H, W]
    """
    # Pre heads
    if component_type == "PreConv2D":
        model = PreConv2D(C=C_in, C_out=C_out, bias=config.heads.PreConv2D.conv_with_bias)
    elif component_type == "PoolLinear":
        model = PoolLinear(
            config=config, C=C_in, num_classes=C_out, add_tanh=config.heads.PoolLinear.add_tanh
        )
    elif component_type == "SimpleConv2d":
        model = SimpleConv2d(config=config, C=C_in, num_classes=C_out)

    else:
        raise NotImplementedError(f"Model not implemented: {component_type}")

    return model


class TestModel:
    config = None

    def setup_class(self):
        set_seed(64861651)
        torch.set_printoptions(precision=10)

        with initialize(version_base=None, config_path="../../src/snraware/components/configs"):
            cfg = compose(config_name="config")
        self.config = cfg

    def teardown_class(self):
        pass

    def test_heads(self):
        B, C_in, D, H, W = 12, 2, 16, 64, 64
        backbone_C = 32

        PreConv2D = create_heads(
            config=self.config,
            component_type="PreConv2D",
            C_in=C_in,
            C_out=backbone_C,
            H=H,
            W=W,
            D=D,
        )

        num_classes = 10
        PoolLinear = create_heads(
            config=self.config,
            component_type="PoolLinear",
            C_in=backbone_C,
            C_out=num_classes,
            H=H,
            W=W,
            D=D,
        )

        num_seg_classes = 4
        SimpleConv2d = create_heads(
            config=self.config,
            component_type="SimpleConv2d",
            C_in=backbone_C,
            C_out=num_seg_classes,
            H=H,
            W=W,
            D=D,
        )

        input = torch.randn(B, C_in, D, H, W, dtype=torch.float32)
        backbone_output = torch.randn(B, backbone_C, D, H, W, dtype=torch.float32)

        with torch.inference_mode():
            y = PreConv2D(input)
            assert y.shape == (B, backbone_C, D, H, W)

            y = PoolLinear(backbone_output)
            assert y.shape == (B, num_classes)

            y = SimpleConv2d(backbone_output)
            assert y.shape == (B, num_seg_classes, D, H, W)


# ---------------------------------------------------------------
