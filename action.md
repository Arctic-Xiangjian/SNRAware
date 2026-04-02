# Action Plan: LoRA Fine-tuning Integration (`action_need.md`)

## 1. Context & Objective
The current codebase features a modular MRI denoising model supporting 2D/3D inputs. With the introduction of proprietary datasets (native 3D data and 2D slice stacks), we need to implement **LoRA (Low-Rank Adaptation)** for Parameter-Efficient Fine-Tuning (PEFT).

**Core Goal:** Inject LoRA into core attention projection layers and Feature Mixers (FFN) without damaging the pre-trained backbone weights. Ensure forward/backward passes and weight-saving logic are mathematically sound.

## 2. Core Requirements

### 2.1 Target Modules for LoRA Injection
Utilize the `peft` library (or a clean PyTorch implementation) to inject LoRA adapters into the following:

* **Attention Projections:**
    * **Location:** Various modules under `snraware/components/model/attention/`.
    * **Targets:** Projection layers generating **Query, Key, and Value**, as well as the final **Output Projection**.
    * **Note:** Some projections use `nn.Linear`, while others use custom `Conv2DExt` or `Conv3DExt`.
* **Mixer / FFN:**
    * **Location:** `Cell` and `Parallel_Cell` classes in `snraware/components/model/backbone/cells.py`.
    * **Targets:** The two linear or convolutional transformation layers within the `self.mlp` module.

### 2.2 Crucial Engineering Watch-outs
The AI Agent must address these specific issues during implementation:

* **Custom Convolution Compatibility:**
    * The code heavily utilizes `Conv2DExt` and `Conv3DExt`. Standard PEFT usually only recognizes native `nn.Linear` or `nn.Conv2d`.
    * **Action:** Configure `peft.LoraConfig` using regex for `target_modules`. If PEFT does not natively support these custom extensions, implement a custom adapter wrapper or adjust class inheritance to ensure LoRA matrices attach successfully.
* **Unfreeze Pre/Post Layers:**
    * In the `DenoisingModel` class (`model.py`), `self.pre` and `self.post` are shallow convolutional layers.
    * **Action:** Do **not** apply LoRA to these layers. However, while the backbone is frozen (`requires_grad=False`), these specific layers **must be unfrozen** (`requires_grad=True`) to adapt to the physical scale and channel distribution of new datasets.
* **Independent Checkpointing:**
    * **Action:** Implement logic to ensure that after fine-tuning, only the LoRA weights (typically a few MBs) and the `pre/post` layer weights are saved, rather than the entire multi-hundred MB pre-trained model.

---

## 3. Dummy Data Test Plan
Upon completion, a standalone unit test script (e.g., `test_lora_integration.py`) must be written to verify LoRA mounting and backbone integrity using synthetic data.



### Test Case Logic:
```python
import torch

def test_lora_integration():
    # 1. Initialize base model and config
    config = get_dummy_config()
    base_model = DenoisingModel(config, D=1, H=64, W=64)
    
    # 2. Snapshot pre-trained weights
    base_weights_snapshot = {name: param.clone().detach() for name, param in base_model.named_parameters()}
    
    # 3. Apply LoRA injection
    lora_model = apply_lora_to_model(base_model)
    
    # 4. Verification I: Gradient State Check
    for name, param in lora_model.named_parameters():
        if "lora" in name or "pre" in name or "post" in name:
            assert param.requires_grad == True, f"{name} should be unfrozen"
        else:
            assert param.requires_grad == False, f"Backbone layer {name} should be frozen"
            
    # 5. Verification II: Forward/Backward Pass
    dummy_input = torch.randn(2, 3, 1, 64, 64) # [B, C, T, H, W]
    dummy_target = torch.randn(2, 2, 1, 64, 64)
    
    optimizer = torch.optim.Adam(lora_model.parameters(), lr=1e-3)
    output = lora_model(dummy_input)
    loss = torch.nn.functional.mse_loss(output, dummy_target)
    
    loss.backward()
    optimizer.step()
    
    # 6. Verification III: Backbone Integrity (Weight Contamination Check)
    for name, param in lora_model.named_parameters():
        # Adjust naming based on PEFT wrapping logic
        if "lora" not in name and "pre" not in name and "post" not in name:
            clean_name = name.replace("base_model.model.", "") # Example replacement
            original_param = base_weights_snapshot[clean_name]
            assert torch.equal(param, original_param), f"CRITICAL ERROR: Backbone weights in {name} changed after training!"
            
    print("✅ LoRA Integration Test Passed! Backbone is safe, and adapters are correctly mounted.")
```

---

## 4. Expected Deliverables
1.  `lora_utils.py`: Contains the function to transform `DenoisingModel` into a LoRA-enabled model.
2.  **Configuration Expansion:** Updated YAML support for LoRA hyperparameters (`r`, `lora_alpha`, `lora_dropout`, `target_modules`).
3.  `test_lora_integration.py`: The validation script containing the assertion logic described above.