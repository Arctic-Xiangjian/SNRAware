
# Overview

Data formats in imaging and related domains can be flexible. A x-ray image is a 2D picture with `[C, H, W]` for channel (often being 1), and height and width. A MRI image to capture beating heart is a 2D+T imagery `[C, F, H, W]` with the frame dimension. A CT volume can be isotropic 3D data `[C, D, H, W]`. All these diversity can be captured by a 5D tensor `[B, C, F, H, W]`. For the 2D case, frame dimension is 1. For the dynamic imaging, frame dimension is the time. For the 3D case, frame dimension is the depth into the volume.

As an example, this is a `2D+T` case. Here we have F=30 frames to capture the beating heart. The image `H` and `W` is 192 and 256. So the tensor shape is `[B, 1, 30, 192, 256]`.

![ch4](images/ch4.gif)

For a 2D case, such as the chest x-ray image, the F is 1.

This package implemented building blocks to help develop AI models to process the 5D tensors. To standardize and simplify this process, all components take in a 5D tensor and output another 5D tensor. The number of channels can be increased or decreased,  but most components will keep the `F, H, W` unchanged. The explicit down/upsampling layers are inserted into the model to alter these dimensions if needed.

# What is in this package

- [Attention layers](./attention_layers.md) : the modified attention layers to capture local, global and inter-frame signal and noise information.
- [Cell and Block](./block_and_cell.md): the container objects to hold the attention layers.
- [Backbone models](./backbone.md): the backbone models consist of multiple blocks. It is the main body of a model.
- [Examples and tests](./examples_and_tests.md): examples to use these components and how to run the tests.
- [MR denoising](./mri_denoising.md): A MR denoising trianing using the imaging transformers.
