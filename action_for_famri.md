### 0. Project Context & Objectives
* **Goal:** Adapt the existing **SNRAware** MRI denoising pipeline to work with legacy FastMRI datasets that lack the 3rd `gmap` (g-factor) channel.
* **Method:** Inject a lightweight **NormUnet** at the head of the network to dynamically predict the `gmap` from the 2-channel (real, imag) input. Fine-tune this U-Net alongside the frozen SNRAware backbone via LoRA adapters. Create a unified custom training loop that respects legacy FastMRI evaluation standards (slice-to-volume metrics) while utilizing the new architecture.

---

### 1. Architecture Modification: The G-Factor Adopter
**Task:** Create a wrapper model, `SNRAwareWithGFactor`, that encapsulates the provided `NormUnet` and the base `SNRAware` model.

**Implementation Details:**
* Integrate the user-provided `NormUnet` class exactly as specified.
* **Forward Pass Logic:**
    * **Input Check:** If the input has 3 channels (real, imag, gmap), skip the `NormUnet` and pass directly to the base model (ensuring backward compatibility).
    * **2-Channel Branch:** If input has 2 channels (real, imag):
        1.  Reshape/format the tensor to match `NormUnet`’s expected complex shape requirement (last dimension = 2).
        2.  Pass through `NormUnet` to generate the 1-channel `gmap` prediction.
        3.  Concatenate the original 2-channel input with the 1-channel predicted `gmap` to form a 3-channel tensor.
        4.  Feed the 3-channel tensor into the base `SNRAware` model.

---

### 2. Data Pipeline: `fastmri_data/work_with_snraware.py`
**Task:** Create a bridge dataset class that applies legacy FastMRI transformations but interfaces correctly with the new training script.

**Implementation Details:**
* **Strict 2D Mode:** Only process and return 2D slices. Do not extrapolate to 3D depths.
* **Legacy Logic:** The `normalize` and mask generation methods **MUST** strictly follow the exact mathematical implementations found in historical `warp_fastmri_singlecoil.py` and `fastmri_data.py`.
* **Output Signature:** The `__getitem__` method must return:
    * `noisy_2ch`: 2-channel masked K-space/image (real, imag).
    * `clean_2ch`: 2-channel target image.
    * `metadata`: A dictionary containing `name` (for slice-to-volume grouping), `mean`, `std` (for un-normalization), and `mask`.

---

### 3. Training Engine: `trainer_fa.py` & `train.py`
**Task:** Build a custom training script (overriding PyTorch Lightning where necessary) to orchestrate 4-stage fine-tuning strategies and legacy evaluation.

**4 Fine-Tuning Modes:**
1.  **`unet_only`**: Train only the `NormUnet` (Backbone frozen, LoRA disabled).
2.  **`unet_and_lora`**: Train both `NormUnet` and LoRA adapters simultaneously.
3.  **`lora_only`**: Train only LoRA adapters (`NormUnet` frozen).
4.  **`warmup_then_both`**: Train `NormUnet` exclusively for $N$ epochs. At epoch $N+1$, unfreeze LoRA adapters and co-fine-tune both.

**Evaluation (Crucial):**
* Execute the test function adapted from `previous_train_just_For_agnet_reference` at every epoch end.
* Group 2D slices back into 3D volumes using the `name` key.
* Un-normalize tensors using saved `mean` and `std`.
* Compute volume-level **PSNR, SSIM, and NMSE** in **fp32** to avoid precision pollution.

---

### 4. Logging and Checkpointing
**Task:** Update WandB initialization and model saving paths.

* **Path Format:** `/working2/arctic/snrawre/SNRAware/checkpoints/fine_tune/[mode]_[timestamp]`
* **Artifact Saving:** The `.pth` logic must save **both** the `NormUnet` state dict and the LoRA adapter state dict (excluding frozen backbone weights).

---

### 5. 🚨 System-Level Missing Pieces & Sanity Checks
To ensure a closed-loop logic, the Agent must address the following:

* **Dimensionality Mismatch:** `NormUnet` requires complex input $[B, C, H, W, 2]$. The Agent must transform the dataset's $[B, 2, H, W]$ into $[B, 1, H, W, 2]$ via `view/unsqueeze` before prediction. The predicted $[B, 1, H, W]$ must then be concatenated with the original $[B, 2, H, W]$ into $[B, 3, H, W]$. If the base model requires 3D-style $[B, 3, 1, H, W]$, an additional `unsqueeze` is required.
* **Hardware Efficiency:** Explicitly enable TF32 support (`torch.backends.cuda.matmul.allow_tf32 = True`) at the start of `train.py`.
* **🚫 Absolute EMA Isolation:** Do **NOT** replicate or implement the **EMA (Exponential Moving Average)** logic found in legacy references. Keep the fine-tuning pipeline pure.
* **Dynamic Optimizer Control:** For `warmup_then_both`, use PyTorch `param_groups`. Manage `requires_grad` and learning rates dynamically at the epoch boundary rather than re-instantiating the optimizer.
* **Metadata Routing:** Ensure `mean` and `std` statistics penetrate from the `DataLoader` through the forward pass to reach the `fastmri.evaluate` calls in the test function.
* **Checkpoint Structure Conflict:** Extend the `save_lora_checkpoint` logic to include both modules: `{'lora_adapter': {...}, 'gfactor_unet': {...}}`. Ensure the inference engine can load both separately.