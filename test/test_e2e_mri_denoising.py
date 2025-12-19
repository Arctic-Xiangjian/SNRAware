import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize

from snraware.components.setup import end_timer, get_device, start_timer
from snraware.projects.mri.denoising.inference_model import (
    load_lit_model,
    load_model,
    load_scripted_model,
)
from snraware.projects.mri.denoising.run import run_training
from snraware.projects.mri.denoising.run_inference import apply_model

# -----------------------------------------------------------------


class TestDenoising:
    data_root = None
    test_root = None
    output_dir = None

    def setup_class(self):
        torch.set_printoptions(precision=10)

        test_path = Path(__file__).parents[0].resolve()
        self.data_root = str(test_path / "data/mri/denoising/tra")
        self.test_root = str(test_path / "data/mri/denoising/test")
        self.output_dir = str(test_path / "../.run/output")

        shutil.rmtree(self.output_dir, ignore_errors=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def teardown_class(self):
        pass

    def load_and_test_saved_model(self, cfg):
        # load saved model and config
        model_scripted_fname = os.path.join(
            cfg.logging.output_dir, f"model_{cfg.logging.project}_{cfg.logging.run_name}.pts"
        )
        model_fname = os.path.join(
            cfg.logging.output_dir, f"model_{cfg.logging.project}_{cfg.logging.run_name}.pth"
        )
        config_fname = os.path.join(
            cfg.logging.output_dir, f"config_{cfg.logging.project}_{cfg.logging.run_name}.yaml"
        )
        model_lit_fname = os.path.join(cfg.logging.output_dir, "checkpoints", "last.ckpt")

        assert os.path.exists(model_scripted_fname), (
            f"Expected model file {model_scripted_fname} exists"
        )
        assert os.path.exists(model_fname), f"Expected model file {model_fname} exists"
        assert os.path.exists(config_fname), f"Expected config file {config_fname} exists"
        assert os.path.exists(model_lit_fname), f"Expected model file {model_lit_fname} exists"

        model, _config = load_model(model_fname, config_fname)
        scripted_model = load_scripted_model(model_scripted_fname)
        lit_model, _ = load_lit_model(model_lit_fname, config_fname)

        # Run inference on a sample image
        data = np.random.randn(128, 120, 30) + 1j * np.random.randn(128, 120, 30)
        gmap = 1.2 * np.ones([128, 120, 30])
        t0 = start_timer(enable=True)
        denoised_image = apply_model(
            model,
            data,
            gmap,
            scaling_factor=1.0,
            cutout=tuple(cfg.dataset.cutout_shape),
            overlap=(16, 16, 8),
            batch_size=1,
            device="cuda",
            verbose=False,
        )
        print(
            f"Torch model, inference time {end_timer(t=t0, enable=True, verbose=False) / 1e3:.2f} sec"
        )

        t0 = start_timer(enable=True)
        denoised_image_scripted = apply_model(
            scripted_model,
            data,
            gmap,
            scaling_factor=1.0,
            cutout=tuple(cfg.dataset.cutout_shape),
            overlap=(16, 16, 8),
            batch_size=1,
            device="cuda",
            verbose=False,
        )
        print(
            f"Torch scripted model, inference time {end_timer(t=t0, enable=True, verbose=False) / 1e3:.2f} sec"
        )

        t0 = start_timer(enable=True)
        denoised_image_lit = apply_model(
            lit_model,
            data,
            gmap,
            scaling_factor=1.0,
            cutout=tuple(cfg.dataset.cutout_shape),
            overlap=(16, 16, 8),
            batch_size=1,
            device="cuda",
            verbose=False,
        )
        print(
            f"Lit model, inference time {end_timer(t=t0, enable=True, verbose=False) / 1e3:.2f} sec"
        )

        diff = np.linalg.norm(denoised_image - denoised_image_scripted)
        assert diff / np.linalg.norm(denoised_image) < 1e-3

        diff = np.linalg.norm(denoised_image - denoised_image_lit)
        assert diff / np.linalg.norm(denoised_image) < 1e-3

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_training(self, request):
        selected_markers = request.config.getoption("-m")
        if "gpu" not in selected_markers or "slow" not in selected_markers:
            pytest.skip("Skipping because both markers 'gpu' and 'slow' are not set")

        device = get_device()
        if device != "cuda":
            pytest.skip("GPU only test")

        with initialize(
            version_base=None, config_path="../src/snraware/projects/mri/denoising/configs"
        ):
            cfg = compose(
                config_name="config",
                overrides=[
                    "backbone.block.cell.n_head=32",
                    "batch_size=1",
                    "optim.lr=1e-5",
                    "trainer.max_epochs=8",
                    "trainer.limit_train_batches=1.0",
                    "trainer.limit_val_batches=1.0",
                    "trainer.limit_test_batches=0.3",
                    "trainer.check_val_every_n_epoch=2",
                    "trainer.log_every_n_steps=2",
                    f"train_data_dir={self.data_root}",
                    f"test_data_dir={self.test_root}",
                    f"logging.output_dir={self.output_dir}",
                    "dataset.max_noise_level=32.0",
                    "seed=342080908",
                ],
            )

        test_res = run_training(cfg)
        test_res = test_res[0]
        assert test_res["test/PSNR"] > 46, f"Expected test PSNR > 46, got {test_res['test/PSNR']}"
        assert test_res["test/ssim"] < 0.2, (
            f"Expected test ssim loss < 0.2, got {test_res['test/ssim']}"
        )

        self.load_and_test_saved_model(cfg)

    @pytest.mark.gpu
    def test_training_single_epoch(self, request):
        selected_markers = request.config.getoption("-m")
        if "gpu" not in selected_markers:
            pytest.skip("Skipping because marker 'gpu' is not set")

        if os.path.exists(self.data_root) is False or os.path.exists(self.test_root) is False:
            pytest.skip("Skipping because test data not found")

        device = get_device()
        if device != "cuda":
            pytest.skip("GPU only test")

        with initialize(
            version_base=None, config_path="../src/snraware/projects/mri/denoising/configs"
        ):
            cfg = compose(
                config_name="config",
                overrides=[
                    "backbone.block.cell.n_head=16",
                    "batch_size=1",
                    "optim.lr=1e-5",
                    "trainer.max_epochs=1",
                    "trainer.limit_train_batches=0.2",
                    "trainer.limit_val_batches=0.1",
                    "trainer.limit_test_batches=0.1",
                    "trainer.check_val_every_n_epoch=1",
                    "trainer.log_every_n_steps=10",
                    "trainer.devices=1",
                    f"train_data_dir={self.data_root}",
                    f"test_data_dir={self.test_root}",
                    f"logging.output_dir={self.output_dir}",
                    "dataset.max_noise_level=32.0",
                    "seed=342080908",
                ],
            )

        test_res = run_training(cfg)
        assert test_res[0]["test/PSNR"] > 40, (
            f"Expected test PSNR > 40, got {test_res[0]['test/PSNR']}"
        )

        self.load_and_test_saved_model(cfg)
