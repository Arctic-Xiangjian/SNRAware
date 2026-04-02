# MRI Denoising Model Architecture Info (`model_info.md`)

## 1. The Big Picture
The entire architecture is engineered to process 5D tensors $[Batch, Channels, Time/Depth, Height, Width]$. To achieve extreme configuration flexibility, the code is strictly partitioned into five nested levels:

* **Top-Level Wrapper (`DenoisingModel`):** Handles input/output channel alignment (e.g., mapping $C_{in}=3$ to hidden dimensions) and maintains the **Global Residual Connection**.
* **Backbones:** Define routing and feature fusion between different **Resolution Levels** (Options: UNet, HRNet, SOAnet).
* **Blocks:** Computational nodes at a specific resolution within the backbone. One Block contains multiple Cells.
* **Cells:** The smallest complete functional components of the Transformer/CNN, containing normalization, attention mechanisms, and mixers.
* **Core Operators:** Specific **Attention** layers (Local, Global, Temporal, etc.) and **Convolutions**.

---

## 2. Backbones
Backbones control spatial resolution ($H, W$) downsampling/upsampling and feature dimension scaling. All backbones inherit from `BackboneBase`.

### 2.1 HRNet (High-Resolution Network)
* **File:** `backbone_hrnet.py`
* **Core Logic:** Maintains high-resolution branches while computing low-resolution branches in parallel, performing continuous feature fusion across all resolutions.
* **Topology:**
    * Supports up to 5 resolution levels.
    * As depth increases, spatial resolution halves and channels double ($C \rightarrow 2C \rightarrow 4C \dots$).
    * **Fusion Phase:** Cross-resolution features are summed via parallel `DownSample` and `UpSample` modules.
    * **Output:** Upsamples all levels back to the original resolution and concatenates them along the channel dimension.

### 2.2 SOAnet (Stack of Attention Network)
* **File:** `backbone_soanet.py`
* **Core Logic:** A straightforward serial structure that stacks Attention Blocks like building blocks.
* **Topology:**
    * Linear forward propagation: Layer 0 $\rightarrow$ Layer 1 $\rightarrow$ Layer 2...
    * If `downsample=True`, a `DownSample` module follows each stage, halving resolution and doubling channels.
    * Returns a list of intermediate outputs from every layer.

### 2.3 UNet (U-Shape Network)
* **File:** `backbone_unet.py`
* **Core Logic:** Classic Encoder-Decoder structure.
* **Topology:** Gradual downsampling on the left (D0-D4) and upsampling on the right (U0-U4). Includes horizontal **Skip-connections** and uses `_unet_attention` to gate high-resolution Values with low-resolution Queries.

---

## 3. Blocks & Configuration
* **File:** `blocks.py`
* **Core Logic:** A Block is a container for multiple Cells.
* **Configuration (Block String):** Internal structure is controlled via strings like `"T1L1G1"`:
    * `T`: Temporal Cell | `L`: Local Cell | `G`: Global Cell
* **Dense Connection:** If `block_dense_connection=True`, internal cells use dense connections (the input to a cell is the sum of all preceding cells' outputs).

---

## 4. Cells
* **File:** `cells.py`
* **Core Logic:** Each Cell handles specific Attention + FFN (Mixer) computations. Supports 5 normalization modes via `norm_mode` (`layer`, `batch2d`, `instance2d`, `batch3d`, `instance3d`).

### 4.1 Sequential Cell (`Cell`)
Uses the classic **Pre-Norm Transformer** structure:
```text
x ──> Norm1 ──> Attention ──> + ──> Norm2 ──> Mixer(MLP/CNN) ──> + ──> Output
|                             ^ |                                ^
|_____________________________| |________________________________|
```

### 4.2 Parallel Cell (`Parallel_Cell`)
Parallelizes Attention and Mixer for accelerated gradient flow:
```text
          |──> Attention ──────────|
x ──> Norm1                        + ──> + ──> Output
          |──> Mixer (MLP/CNN) ────|     ^
|                                        |
|──────────── input_proj ────────────────|
```

---

## 5. Attention Modes
In `Cell_Base`, the model instantiates various receptive field mechanisms based on `att_mode`:

* **2D Modes:**
    * `local`: Spatial Local Window Self-Attention.
    * `global`: Spatial Global Attention (cross-window interaction via doubling windows and halving patches).
    * `temporal`: Temporal/Depth correlation attention along the $T$ dimension.
    * `vit_2d`: Standard Vision Transformer attention.
* **3D Modes:**
    * `local_3d`, `global_3d`, `vit_3d`, `swin_3d`: 3D self-attention across $(D, H, W)$ joint dimensions.
* **CNN Fallback:**
    * `conv2d`, `conv3d`: Degenerates to pure convolutional block operations.

---

## 6. Developer Guide (AI Agent & Extensibility)

* **Trace Tensor Shapes:** Always remember that input/outputs for any layer must be $[B, C, T, H, W]$. If adding a new Attention type that only handles 4D ($[B, C, H, W]$), you must use `permute` and `reshape` (e.g., `permute_to_B_T_C_H_W`) to fold $B$ and $T$.
* **Dynamic Window/Patch Scaling:** As the network downsamples, three `window_sizing_method` strategies are available:
    1.  `keep_num_window`: Constant number of windows (window size shrinks).
    2.  `keep_window_size`: Constant window pixel size (number of windows decreases).
    3.  `mixed`: A hybrid strategy (see `add_a_block` in Backbone).
* **Adding New Backbones:** To add a new backbone (e.g., VNet), inherit from `BackboneBase`, instantiate the required Blocks in `__init__`, and define the routing in `forward`. Ensure output channels are reported via `get_number_of_output_channels`.