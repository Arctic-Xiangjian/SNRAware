from pathlib import Path

import hydra
import lightning as L
import pytest
import torch
from hydra import compose, initialize
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, TQDMProgressBar
from omegaconf import OmegaConf

from snraware.components.heads import PreConv2D, SimpleConv2d
from snraware.components.model import HRnet, SOAnet, Unet
from snraware.components.optim import OptimScheduler

# -----------------------------------------------------------------


def create_backbone(config, component_type, C_in, H, W, D):
    """
    Create model components from configuration.
    Return value is the created component and backbone_C_out for number of output channels from the backbone.
    backbone_C_out can vary depending on the backbone architecture.
    """
    if component_type.lower() == "hrnet":
        model = HRnet(config=config, input_feature_channels=C_in, H=H, W=W, D=D)

    elif component_type.lower() == "unet":
        model = Unet(config=config, input_feature_channels=C_in, H=H, W=W, D=D)

    elif component_type.lower() == "soanet":
        model = SOAnet(config=config, input_feature_channels=C_in, H=H, W=W, D=D)

    else:
        raise NotImplementedError(f"Not implemented: {component_type}")

    backbone_C_out = model.get_number_of_output_channels()
    return model, backbone_C_out


class SimpleRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, N, C, D, H, W):
        self.num = N
        self.shape = [C, D, H, W]

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        a = torch.randn(self.shape, dtype=torch.float32)
        b = 10 * a + 2.0
        return a, b


class SimpleRegressionDataModule(L.LightningDataModule):
    def __init__(self, config, C, D, H, W):
        super().__init__()
        self.config = config

        self.C = C
        self.D = D
        self.H = H
        self.W = W

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.train_set = SimpleRegressionDataset(
                N=1024, C=self.C, D=self.D, H=self.H, W=self.W
            )
            self.val_set = SimpleRegressionDataset(N=256, C=self.C, D=self.D, H=self.H, W=self.W)
        if stage == "test" or stage is None:
            self.test_set = SimpleRegressionDataset(N=256, C=self.C, D=self.D, H=self.H, W=self.W)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )


# -----------------------------------------------------------------


class SimpleRegressionModel(torch.nn.Module):
    def __init__(self, config, C_in, D, H, W, C_out):
        super().__init__()

        self.config = config

        backbone_C = config.backbone.num_of_channels

        # create a pre head
        self.pre_head = PreConv2D(
            C=C_in, C_out=backbone_C, bias=self.config.heads.PreConv2D.conv_with_bias
        )

        # create the backbone model
        backbone_config = hydra.utils.instantiate(config.backbone)
        self.bk, bk_C_out = create_backbone(
            backbone_config, component_type=config.backbone.name, C_in=backbone_C, H=H, W=W, D=D
        )

        # create the post head
        self.post_head = SimpleConv2d(config=config, C=bk_C_out, num_classes=C_out)

    def forward(self, x):
        x = self.pre_head(x)
        x = self.bk(x)
        x = self.post_head(x)
        return x


class LitSimpleRegression(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        self.batch_size = self.model.config.batch_size
        self.scheduler_type = self.model.config.scheduler.name

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, targets = batch
        outputs = self.model(images)

        loss = self.loss_fn(outputs, targets)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )

        curr_lrs = self.optim_sched.report_lr()
        self.log(
            "lr",
            curr_lrs[0][0],
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        mse = self.loss_fn(outputs, targets)
        self.log(
            "val_mse", mse, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.batch_size
        )

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        mse = self.loss_fn(outputs, targets)
        self.log("test_mse", mse, batch_size=self.batch_size)

    def configure_optimizers(self):
        total_num_steps = self.trainer.estimated_stepping_batches
        print(f"Total number of steps: {total_num_steps}")
        self.optim_sched = OptimScheduler(self.config, self.model, total_num_steps)
        return [self.optim_sched.optim], [
            {"scheduler": self.optim_sched.sched, "interval": "step", "frequency": 1}
        ]


# -----------------------------------------------------------------


class TestTrain:
    test_path = None

    def setup_class(self):
        torch.set_printoptions(precision=10)

        test_path = Path(__file__).parents[0].resolve()
        self.output_dir = str(test_path / "../.run/output")

    def teardown_class(self):
        pass

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "backbone",
        ["soanet", "hrnet", "unet"],
    )
    def test_trainer(self, backbone, request):
        selected_markers = request.config.getoption("-m")
        if "gpu" not in selected_markers:
            pytest.skip("Skipping because marker 'gpu' is not set")

        overrides = [
            f"backbone={backbone}",
            "backbone.block.cell.window_size=[4,4,1]",
            "backbone.block.cell.patch_size=[4,4,2]",
            "backbone.block.cell.n_head=16",
            "backbone.num_of_channels=16",
            "batch_size=2",
            "num_workers=2",
            "optim.lr=2e-4",
            "trainer.max_epochs=6",
            "trainer.check_val_every_n_epoch=1",
            "trainer.log_every_n_steps=50",
            "trainer.accelerator='auto'",
            "seed=54841631",
            "logging.project=simple-regression-demo",
            "logging.run_name=test_trainer",
            f"logging.output_dir={self.output_dir}",
        ]

        if backbone == "soanet":
            overrides.append("backbone.downsample=false")

        with initialize(
            version_base=None, config_path="../src/snraware/projects/mri/denoising/configs"
        ):
            cfg = compose(config_name="config", overrides=overrides)
        self.config = cfg

        C_in, D, H, W = 1, 8, 32, 32
        C_out = C_in

        torch.set_float32_matmul_precision("high")

        # Print configuration
        print("Training Configuration:")
        print(OmegaConf.to_yaml(self.config, resolve=True))

        if self.config.seed is not None:
            L.seed_everything(self.config.seed)

        # set up dataset
        data_module = SimpleRegressionDataModule(config=self.config, C=C_in, D=D, H=H, W=W)

        # callback to save the best model
        best_checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_mse",
            mode="min",
            dirpath=self.config.logging.output_dir,
            filename="best_checkpoint-{backbone}-{epoch:02d}-{val_mse:.2f}",
        )

        # set up the trainer
        trainer = hydra.utils.instantiate(
            self.config.trainer,
            callbacks=[
                best_checkpoint_callback,
                TQDMProgressBar(refresh_rate=20, leave=True),
                ModelSummary(max_depth=3),
            ],
            logger=None,  # No logger for this test
        )

        # set up model
        model = SimpleRegressionModel(config=self.config, C_in=C_in, D=D, H=H, W=W, C_out=C_out)

        # set up lightning components
        lit_model = LitSimpleRegression(model=model, config=self.config)

        # fit the model
        trainer.fit(model=lit_model, datamodule=data_module)

        # test the model
        test_res = trainer.test(model=lit_model, datamodule=data_module, ckpt_path="best")

        assert test_res[0]["test_mse"] < 0.5, (
            "Test MSE is higher than 0.5, indicating poor model performance."
        )
