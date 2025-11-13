
# Examples and tests

The modules and backbones in this package can be used to build models for different applications. The key design is to support 5D tensor [B, C, T/D/F/Z, H, W] for 2D, 2D+T, and 3D usecases.

## Attention modules

All attention modules are implemented as pytorch modules. They can be defined in the very straight-forward way as the following:

```python
import numpy as np
import torch
from snraware.components.setup import start_timer, end_timer, get_device
from snraware.components.model.attention import ViT3DAttention

# find the device, e.g. 'cuda:0'
device = get_device()

# define an input tensor
B, C, T, H, W = 1, 2, 16, 32, 32
C_out = 8
test_in = torch.rand(B, C, T, H, W).to(device=device)
print(test_in.shape)

# define a vit3d module

spacial_vit = ViT3DAttention(window_size=None, 
                            num_wind=[8, 8, 4], # here we choose to set number of windows; alternatively, we can set window_size to be [2, 2, 4]
                            attention_type="conv", # use convolution to compute Q/K/V in the attention
                            C_in=2, # input channel, 2
                            C_out=8, # output channel, 8
                            H=H, W=W, D=T, # tensor sizes
                            stride_qk=[1,1,1], # stride is 1 when computing Q and K
                            cosine_att=True, # use the cosine attention
                            normalize_Q_K=False, # normalize Q and K before computing attention matrix; if cosine_att is True, this option 
                            att_with_relative_position_bias=True,
                            att_with_output_proj=True)

spacial_vit.to(device=device)

test_out = spacial_vit(test_in)

print(f"ViT3DAttention, input tensor {test_in.shape}, output tensor {test_out.shape}")

```

## Backbones

The backbone models contain multiple Blocks and Cells. They are configured by supplying the block_str to define the attention structures. 

Unlike the attention modules where parameters are listed out, the model has many more parameters. We use the [hydra](https://hydra.cc/) to manage the parameters. The parameter configuration files are in the `src/ifm/configs` folder. To define the model parameters for a backbone model:

```
from hydra import initialize, compose
with initialize(version_base=None, config_path="../src/snraware/components/configs"):
    config = compose(config_name="config")
```

The backbone parameters are in `config.backbone`. The block and cell parameters are in `config.block` and `config.cell`.

```python

import torch
from snraware.components.setup import get_device
import hydra
from hydra import initialize, compose

# get the device, e.g. 'cuda:0'
device = get_device()

# define the input tensor
B, C, T, H, W = 1, 2, 32, 128, 128
test_in = torch.rand(B, C, T, H, W).to(device=device)

# get the mode parameter; note user can override the default configuration
# here we order a hrnet backbone
with initialize(version_base=None, config_path="../src/snraware/components/configs"):
    cfg = compose(config_name="config")

config = hydra.utils.instantiate(cfg.backbone)

# define the model
model = HRnet(config=config, input_feature_channels=C, H=H, W=W, D=T)
model = model.to(device=device)

# perform the forward pass
test_out = model(test_in)
print(f"{Fore.YELLOW}The output tensor shape is {test_out[-1].shape}.{Style.RESET_ALL}")
```