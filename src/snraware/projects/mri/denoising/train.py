"""Entry point for FastMRI fine-tuning with SNRAware."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import hydra
import torch
from colorama import Fore, Style
from omegaconf import DictConfig, OmegaConf

from snraware.projects.mri.denoising.base_model_resolver import resolve_base_model_paths
from snraware.projects.mri.denoising.fastmri_compat import build_fastmri_wrapped_model
from snraware.projects.mri.denoising.trainer_fa import (
    FastMRIFineTuneTrainer,
    build_fastmri_dataloaders,
    resolve_fastmri_precision,
    seed_everything,
)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not device_str.startswith("cuda"):
        return torch.device(device_str)
    if not torch.cuda.is_available():
        print(
            f"{Fore.YELLOW}CUDA requested but unavailable, falling back to CPU.{Style.RESET_ALL}",
            flush=True,
        )
        return torch.device("cpu")
    return torch.device(device_str)


def _normalized_wandb_entity(entity: object) -> str | None:
    if entity is None:
        return None
    text = str(entity).strip()
    if text == "" or text.lower() == "null":
        return None
    return text


@hydra.main(version_base=None, config_path="./configs", config_name="fastmri_finetune")
def run_fastmri_finetuning(config: DictConfig):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")

    try:
        from fastmri.evaluate import nmse as _nmse  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The FastMRI fine-tuning entrypoint requires the `fastmri` package for PSNR/SSIM/NMSE evaluation."
        ) from exc

    if config.seed is None:
        config.seed = torch.randint(0, 2**32, (1,)).item()
    seed_everything(int(config.seed))

    resolved_base_config_path, resolved_base_checkpoint_path = resolve_base_model_paths(
        variant=config.base_model.get("variant"),
        config_path=config.base_model.get("config_path"),
        checkpoint_path=config.base_model.get("checkpoint_path"),
    )
    config.base_model.config_path = resolved_base_config_path
    config.base_model.checkpoint_path = resolved_base_checkpoint_path

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = config.fastmri_finetune.run_name or f"{config.fastmri_finetune.mode}_{timestamp}"
    run_dir = Path(config.fastmri_finetune.save_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config.logging.output_dir = str(run_dir)
    OmegaConf.save(config=config, f=run_dir / "fastmri_finetune_config.yaml")

    print(f"{Fore.YELLOW}FastMRI Fine-Tune Configuration:{Style.RESET_ALL}")
    print(OmegaConf.to_yaml(config, resolve=True))
    print(f"{Fore.YELLOW}{'---' * 30}{Style.RESET_ALL}")
    print(f"Run directory: {run_dir}")
    print(f"Resolved base config: {resolved_base_config_path}")
    print(f"Resolved base checkpoint: {resolved_base_checkpoint_path}")
    print(f"{Fore.YELLOW}{'---' * 30}{Style.RESET_ALL}")

    wandb_run = None
    if config.logging.use_wandb:
        import wandb

        init_kwargs = dict(
            project=config.logging.project,
            name=run_name,
            dir=config.logging.wandb_dir,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=False),
        )
        wandb_entity = _normalized_wandb_entity(config.logging.get("wandb_entity"))
        if wandb_entity is not None:
            init_kwargs["entity"] = wandb_entity

        wandb_run = wandb.init(**init_kwargs)

    train_loader, val_loader, test_loader = build_fastmri_dataloaders(config)
    device = _resolve_device(str(config.fastmri_finetune.device))
    precision_state = resolve_fastmri_precision(
        device,
        use_bf16=bool(config.fastmri_finetune.use_bf16),
    )
    crop_h, crop_w = [int(dim) for dim in config.fastmri_finetune.crop_size]

    print(
        f"{Fore.GREEN}Training precision: {precision_state['mode']} "
        f"(validation/test remain fp32).{Style.RESET_ALL}",
        flush=True,
    )

    model, base_config, load_info = build_fastmri_wrapped_model(
        base_config_path=config.base_model.config_path,
        base_checkpoint_path=config.base_model.checkpoint_path,
        height=crop_h,
        width=crop_w,
        depth=1,
        lora_config=config.get("lora"),
        gfactor_unet_kwargs=OmegaConf.to_container(
            config.fastmri_finetune.gfactor_unet, resolve=True
        ),
    )

    print(
        f"{Fore.GREEN}Loaded base checkpoint ({load_info['weight_source']}) with "
        f"{load_info['matched_keys']} matched keys, "
        f"{load_info['mismatched_keys']} shape mismatches, "
        f"{load_info['total_model_keys']} total model keys.{Style.RESET_ALL}",
        flush=True,
    )
    if wandb_run is not None:
        wandb_run.summary["base_weight_source"] = load_info["weight_source"]
        wandb_run.summary["base_matched_keys"] = load_info["matched_keys"]
        wandb_run.summary["base_mismatched_keys"] = load_info["mismatched_keys"]
        wandb_run.summary["base_config_path"] = resolved_base_config_path
        wandb_run.summary["base_checkpoint_path"] = resolved_base_checkpoint_path
        wandb_run.summary["training_precision"] = precision_state["mode"]
        wandb_run.summary["evaluation_precision"] = "fp32"

    trainer = FastMRIFineTuneTrainer(
        model=model,
        config=config,
        device=device,
        run_dir=run_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        wandb_run=wandb_run,
        precision_state=precision_state,
    )

    results = trainer.train()

    if wandb_run is not None:
        wandb_run.summary["run_dir"] = str(run_dir)
        wandb_run.finish()

    print(f"{Fore.YELLOW}{'---' * 30}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}FastMRI fine-tuning finished. Outputs saved to {run_dir}{Style.RESET_ALL}")
    if "test" in results:
        test_metrics = results["test"]
        print(
            f"{Fore.GREEN}Test metrics: PSNR={test_metrics['psnr']:.4f}, "
            f"SSIM={test_metrics['ssim']:.4f}, NMSE={test_metrics['nmse']:.4f}{Style.RESET_ALL}"
        )
    print(f"{Fore.YELLOW}{'---' * 30}{Style.RESET_ALL}")

    return {"results": results, "base_config": OmegaConf.to_container(base_config, resolve=True)}


if __name__ == "__main__":
    run_fastmri_finetuning()
