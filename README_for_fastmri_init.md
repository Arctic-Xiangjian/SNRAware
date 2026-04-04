### 1. Project Overview & Agent Core Directives
**SNRAware** is a deep learning framework dedicated to Magnetic Resonance Imaging (MRI) denoising and reconstruction. This document provides AI Agents with global context, data flow logic, and the underlying principles of the LoRA fine-tuning mechanism.

> ⚠️ **CRITICAL DIRECTIVE (Core Isolation):**
> The `fastmri_data` folder and the `previous_train_just_For_agnet_reference` script provided in the project are for **historical background and architectural reference only**. This code originates from earlier, distinct projects and is **completely incompatible** with the current SNRAware training pipeline. Do not attempt any code-level fusion or adaptation. When generating code or providing suggestions, **never** attempt to replace the current Dataloader with these legacy FastMRI dataset classes.

---

### 2. SNRAware Core Architecture & LoRA Injection Mechanism
The repository implements a highly decoupled **Parameter-Efficient Fine-Tuning (PEFT)** mechanism specifically designed for MRI image restoration networks.

* **Injection Scope:** The system supports dynamic LoRA adapter injection for `nn.Linear`, `nn.Conv2d`, `nn.Conv3d`, and custom extended convolution modules (e.g., `Conv2DExt`, `Conv3DExt`). By default, target modules like `key`, `query`, `value`, `output_proj` in Transformer attention blocks and the feed-forward network (`mlp`) are matched via regular expressions.
* **Freezing & Training Strategy:** Once LoRA is enabled, the backbone weights are **completely frozen**. Only the low-rank and high-rank matrices (`lora_A`, `lora_B`) within the LoRA layers, along with the `pre` and `post` convolutional layers at the model's entry/exit points, undergo gradient updates.
* **Adapter Persistence:** The resulting `.pth` files do not contain full model weights but are exclusive **LoRA adapter checkpoints** (formatted as `snraware_lora_adapter_v1`). During inference, the pretrained Base model must be instantiated first, followed by mounting the LoRA adapter onto it.

---

### 3. Strict Dataset Contract
The SNRAware training engine enforces a rigorous specification for dataset outputs. Any new dataset implementation must return the following **four-tuple**:

1.  **`noisy` (Input Tensor):** Must contain **3 channels**, explicitly corresponding to **real**, **imag (imaginary)**, and **gmap (g-factor geometry map)**.
2.  **`clean` (Target Tensor):** Must contain **2 channels**, corresponding to **real** and **imag**.
3.  **`noise_sigma` (Scalar/Tensor):** Records noise levels for individual samples or repetitions; used for logging and validation scaling.
4.  **`placeholder`:** Currently inactive but required to satisfy internal batch unpacking logic.

**Data Requirements:**
* **Shape:** Must include a depth/temporal dimension $T$, formatted as $[C, T, H, W]$ or $[REP, C, T, H, W]$ for repeat sampling.
* **Type:** Strictly `float32`.

---

### 4. Background Reference Analysis (Conceptual Only)
Extracted concepts from legacy files to help Agents understand past research preferences and evaluation logic:

#### 4.1 Historical Training Script (`previous_train_just_For_agnet_reference`)
* **Hardware Optimization:** Enabled **TF32** matrix multiplication acceleration and utilized **ModelEmaV2** (Exponential Moving Average) from the `timm` library to stabilize the training trajectory and weight distribution.
* **Evaluation System:** Validation involves calculating slice-level losses and re-aggregating data into full 3D volumes to compute volume-level **PSNR, SSIM, and NMSE**. Validation is forced to **fp32** to ensure precision and fairness.

#### 4.2 Historical Data Pipeline (`fastmri_data`)
* **Metadata Extraction:** Dynamically parses **ISMRMRD XML** headers to extract `matrixSize`, encoding limits, and calculate required K-space zero-padding.
* **Contextual Awareness:** Uses `num_adj_slices` to extract adjacent slices, providing a 3D contextual field of view for 2D backbones.
* **Prior Loading:** Logic for loading pre-generated **Latent Features/Priors** from external directories, typically used for structural guidance in cascaded or autoregressive networks.

---

### 5. Project Architecture & Code Topology
SNRAware uses a modular "Base Component - Business Pipeline" decoupled design. Agents must follow this topology for code tracing or modifications.

#### 5.1 Two-Layer Decoupling: Components vs. Projects
The `src/snraware/` directory is split into two parallel worlds:

* **Foundation Component Library (`components/`):** Pure deep learning operators independent of specific medical tasks.
    * **Backbones:** Architectures like UNet, HRNet, and SoANet (`components/model/backbone/`).
    * **Building Blocks:** Attention mechanisms (Spatial, Temporal, ViT, Swin 3D), blocks, and cells.
    * **Ecosystem:** Optimizers (e.g., custom Sophia optimizer) and LR schedulers.
* **Business Logic Layer (`projects/`):** Instantiates components into specific task pipelines.
    * **MRI Denoising (`projects/mri/denoising/`):** Core task directory containing the Lightning wrapper, data logic, inference, and LoRA utilities.
    * **MRI Physics (`projects/mri/snr/`):** Physics-driven modules like Fourier transforms (`fftc.py`), noise modeling, and filtering.

#### 5.2 SNR-Aware Core Philosophy
The name "SNRAware" highlights the integration of **MRI physics** and **Spatial Signal-to-Noise Ratio (SNR)** distribution. By requiring the **gmap** channel in the input, the network explicitly perceives noise amplification penalties in parallel imaging, allowing for targeted structural recovery. Any data-level changes must ensure the lossless transmission of gmap information.

#### 5.3 Training & Configuration Engine
* **Configuration (Hydra):** Hyperparameters and module instantiations are managed via `configs/`. Use YAML overrides or CLI flags rather than modifying source code.
* **Lifecycle (PyTorch Lightning):** Training loops, AMP (Mixed Precision), distributed strategies, and logging are handled by Lightning. The entry point is `run.py`, which follows the flow: *Parse Config → Instantiate DataLoader → Instantiate Base Model → Mount LoRA (if enabled) → Wrap in LightningModule → Trainer.fit()*.