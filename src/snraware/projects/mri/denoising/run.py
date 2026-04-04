"""Main training script for the MRI denoising model."""

import os

import hydra
import lightning as L
import torch
from colorama import Fore, Style
from lightning.fabric.utilities.throughput import measure_flops
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from omegaconf import DictConfig, OmegaConf

import snraware
from snraware.projects.mri.denoising.lightning_denoising import (
    DenoisingDataModule,
    LitDenoising,
    after_training,
)
from snraware.projects.mri.denoising.lora_utils import (
    apply_lora_to_model,
    count_trainable_parameters,
    is_lora_enabled,
)
from snraware.projects.mri.denoising.model import DenoisingModel


def _normalized_wandb_entity(entity: object) -> str | None:
    if entity is None:
        return None
    text = str(entity).strip()
    if text == "" or text.lower() == "null":
        return None
    return text


def _measure_flops(Lit_model, config):
    dummy_input = torch.randn(
        1,
        Lit_model.model.C_in,
        config.dataset.cutout_shape[2],
        config.dataset.cutout_shape[0],
        config.dataset.cutout_shape[1],
        device=Lit_model.device,
    )

    def model_fwd():
        return Lit_model(dummy_input)

    with torch.inference_mode():
        flops = measure_flops(Lit_model, model_fwd)
    return flops


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def run_training(config: DictConfig):
    torch.set_float32_matmul_precision("highest")

    # lightning needs NODE_RANK env variable for multi-node training
    if "NODE_RANK" not in os.environ and "RANK" in os.environ:
        os.environ["NODE_RANK"] = os.environ["RANK"]
        print(
            f"{Fore.YELLOW}NODE_RANK set to {os.environ['NODE_RANK']} from RANK{Style.RESET_ALL}",
            flush=True,
        )
        print(f"{Fore.YELLOW}RANK is {os.environ['RANK']}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.YELLOW}NODE_RANK is {os.environ['NODE_RANK']}{Style.RESET_ALL}", flush=True)
    if "MASTER_ADDR" in os.environ:
        print(
            f"{Fore.YELLOW}MASTER_ADDR is {os.environ['MASTER_ADDR']}{Style.RESET_ALL}", flush=True
        )
    if "MASTER_PORT" in os.environ:
        print(
            f"{Fore.YELLOW}MASTER_PORT is {os.environ['MASTER_PORT']}{Style.RESET_ALL}", flush=True
        )
    if "WORLD_SIZE" in os.environ:
        print(
            f"{Fore.YELLOW}WORLD_SIZE is {os.environ['WORLD_SIZE']}{Style.RESET_ALL}", flush=True
        )

    if config.seed is None:
        config.seed = torch.randint(0, 2**32, (1,)).item()
    L.seed_everything(config.seed)

    # Print configuration
    print(f"{Fore.YELLOW}Training Configuration:{Style.RESET_ALL}")
    OmegaConf.to_yaml(config, resolve=True)
    print(OmegaConf.to_yaml(config, resolve=True))
    print(f"{Fore.YELLOW}{'---' * 30}{Style.RESET_ALL}")
    print(f"Output directory: {config.logging.output_dir}")
    print(f"{Fore.YELLOW}{'---' * 30}{Style.RESET_ALL}")

    # init wandb
    wandb_logger = None
    if config.logging.use_wandb:
        logger_kwargs = dict(
            project=config.logging.project,
            name=config.logging.run_name,
            save_dir=config.logging.wandb_dir,
            log_model=True,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=False),
        )
        wandb_entity = _normalized_wandb_entity(config.logging.get("wandb_entity"))
        if wandb_entity is not None:
            logger_kwargs["entity"] = wandb_entity
        wandb_logger = WandbLogger(**logger_kwargs)

    # set up dataset
    data_module = DenoisingDataModule(config=config)

    # callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        monitor="val/loss",
        mode="min",
        save_last=True,
        every_n_epochs=min(config.logging.save_ckpt_every_n_epochs, config.trainer.max_epochs),
        dirpath=os.path.join(config.logging.output_dir, "checkpoints"),
        filename="checkpoint-{epoch:02d}",
    )

    # set up the trainer
    if config.trainer.strategy == "fsdp":
        policy = {
            snraware.components.model.backbone.Block,
            snraware.components.model.backbone.DownSample,
            snraware.components.model.backbone.UpSample,
        }
        strategy = FSDPStrategy(
            auto_wrap_policy=policy,
            sharding_strategy="HYBRID_SHARD",
            device_mesh=tuple(config.device_mesh),
            activation_checkpointing_policy={
                snraware.components.model.backbone.DownSample,
                snraware.components.model.backbone.UpSample,
            },
        )
    else:
        strategy = config.trainer.strategy

    progress_bar = TQDMProgressBar(refresh_rate=config.trainer.log_every_n_steps, leave=True)

    trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=[
            checkpoint_callback,
            progress_bar,
            ModelSummary(max_depth=4),
        ],
        logger=wandb_logger,
        strategy=strategy,
    )

    print(
        f"Using {trainer.num_devices} devices for training on {trainer.num_nodes} nodes, strategy {config.trainer.strategy}."
    )

    # set up model
    model = DenoisingModel(
        config=config,
        D=config.dataset.cutout_shape[2],
        H=config.dataset.cutout_shape[0],
        W=config.dataset.cutout_shape[1],
    )

    if is_lora_enabled(model_config=config):
        model = apply_lora_to_model(model=model)
        if trainer.global_rank == 0:
            trainable, total = count_trainable_parameters(model)
            pct = 100.0 * trainable / max(total, 1)
            print(
                f"{Fore.YELLOW}LoRA enabled: trainable parameters "
                f"{trainable / 1e6:.3f}M / {total / 1e6:.3f}M ({pct:.3f}%)"
                f"{Style.RESET_ALL}",
                flush=True,
            )

    # set up lightning components
    lit_model = LitDenoising(model=model, config=config)

    # report flops
    if trainer.global_rank == 0:
        flops = _measure_flops(lit_model, config)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{Fore.YELLOW}GFLOPs: {flops / 1e9:.2f} GFLOPs{Style.RESET_ALL}", flush=True)
        print(
            f"{Fore.YELLOW}Num_Parameters: {num_params / 1e6:.2f} M{Style.RESET_ALL}", flush=True
        )
        if wandb_logger is not None:
            wandb_logger.experiment.summary["GFLOPS"] = flops / 1e9
            wandb_logger.experiment.summary["Num_Parameters_m"] = num_params / 1e6

    # fit the model
    trainer.fit(model=lit_model, datamodule=data_module)
    trainer.print(torch.cuda.memory_summary())

    # test the model
    test_res = trainer.test(model=lit_model, datamodule=data_module)

    # save for use in production environment
    if trainer.global_rank == 0:
        model_scripted_fname, model_fname, config_fname = after_training(model, config)
        print(
            f"{Fore.GREEN}Trained model saved, scripted format : {model_scripted_fname}{Style.RESET_ALL}"
        )
        print(f"{Fore.GREEN}Trained model saved, torch format    : {model_fname}{Style.RESET_ALL}")
        print(
            f"{Fore.GREEN}Trained model saved, config file     : {config_fname}{Style.RESET_ALL}"
        )
    print(f"{Fore.YELLOW}{'---' * 30}{Style.RESET_ALL}")

    return test_res


if __name__ == "__main__":
    run_training()
