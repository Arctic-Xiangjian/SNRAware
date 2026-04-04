"""Run the denoising model inference."""

import torch
from omegaconf import OmegaConf

from snraware.projects.mri.denoising.fastmri_compat import (
    build_fastmri_wrapped_model,
    is_fastmri_finetune_checkpoint,
    load_fastmri_finetune_checkpoint,
)
from snraware.projects.mri.denoising.lightning_denoising import LitDenoising
from snraware.projects.mri.denoising.lora_utils import (
    apply_lora_to_model,
    is_lora_checkpoint,
    load_lora_checkpoint,
)
from snraware.projects.mri.denoising.model import DenoisingModel

__all__ = [
    "load_fastmri_finetune_model",
    "load_lit_model",
    "load_model",
    "load_scripted_model",
]

# -------------------------------------------------------------------------------------------------


def load_scripted_model(saved_model_path):
    """
    Load a saved torch scripted ".pts" model
    @rets:
        - model (torch scripted model): the model ready for inference.
    """
    model = None
    try:
        model = torch.jit.load(saved_model_path)
    except Exception as e:
        print(f"Error happened in load_scripted_model for {saved_model_path}: {e}")

    return model


# -------------------------------------------------------------------------------------------------


def load_model(saved_model_path, saved_config_path):
    """
    Load a saved torch ".pth" model
    @rets:
        - model (torch model): the model ready for inference
        - config (omegaconf): the config used to create the model.
    """
    model = None
    config = None
    try:
        # load config
        config = OmegaConf.load(saved_config_path)

        # instantiate a model with this config
        model = DenoisingModel(
            config=config,
            D=config.dataset.cutout_shape[2],
            H=config.dataset.cutout_shape[0],
            W=config.dataset.cutout_shape[1],
        )

        # load the model weights
        status = torch.load(saved_model_path, map_location="cpu")
        if is_lora_checkpoint(status):
            # Adapter checkpoints require compatible frozen backbone weights in `model`.
            load_lora_checkpoint(model=model, checkpoint=status)
        elif "model_state_dict" in status:
            model.load_state_dict(status["model_state_dict"])
        else:
            model.load_state_dict(status)
    except Exception as e:
        print(f"Error happened in load_model for {saved_model_path}, {saved_config_path}: {e}")

    return model, config


# -------------------------------------------------------------------------------------------------


def load_lit_model(saved_model_path, saved_config_path):
    """
    Load a saved lightning model
    @rets:
        - model (torch model): the model ready for inference
        - config (omegaconf): the config used to create the model.
    """
    lit_model = None
    config = None
    try:
        # load config
        config = OmegaConf.load(saved_config_path)

        model = DenoisingModel(
            config=config,
            D=config.dataset.cutout_shape[2],
            H=config.dataset.cutout_shape[0],
            W=config.dataset.cutout_shape[1],
        )

        if config.get("lora") and bool(config.lora.get("enabled", False)):
            model = apply_lora_to_model(model=model)

        lit_model = LitDenoising.load_from_checkpoint(saved_model_path, model=model, config=config)
    except Exception as e:
        print(f"Error happened in load_lit_model for {saved_model_path}, {saved_config_path}: {e}")

    return lit_model, config


# ---------------------------------------------------------------------------------------


def load_fastmri_finetune_model(
    saved_model_path,
    base_model_path,
    base_config_path,
    crop_size=(320, 320),
    lora_config=None,
    gfactor_unet_kwargs=None,
):
    """
    Load a FastMRI fine-tune checkpoint consisting of a base model plus g-factor head.

    @rets:
        - model (SNRAwareWithGFactor): the wrapped model ready for inference
        - config (omegaconf): the base config used to create the model
    """
    model = None
    config = None
    try:
        model, config, _load_info = build_fastmri_wrapped_model(
            base_config_path=base_config_path,
            base_checkpoint_path=base_model_path,
            height=crop_size[0],
            width=crop_size[1],
            depth=1,
            lora_config=lora_config,
            gfactor_unet_kwargs=gfactor_unet_kwargs,
        )

        status = torch.load(saved_model_path, map_location="cpu")
        if not is_fastmri_finetune_checkpoint(status):
            raise ValueError("Provided checkpoint is not a FastMRI fine-tune checkpoint")

        load_fastmri_finetune_checkpoint(
            model=model,
            checkpoint=status,
            apply_lora_fn=apply_lora_to_model,
            lora_config=lora_config,
        )
    except Exception as e:
        print(
            f"Error happened in load_fastmri_finetune_model for "
            f"{saved_model_path}, {base_model_path}, {base_config_path}: {e}"
        )

    return model, config


# ---------------------------------------------------------------------------------------


if __name__ == "__main__":
    pass
