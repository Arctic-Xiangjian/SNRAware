import os
import time

import cv2
import hydra
import lightning as L
import numpy as np
import torch
import wandb
from colorama import Fore, Style
from omegaconf import OmegaConf
from torch.utils.data import random_split

from snraware.components.optim import OptimScheduler
from snraware.projects.loss.imaging_loss import PSNR, Combined_Loss
from snraware.projects.mri.denoising.data import MRIDenoisingDatasetTest
from snraware.projects.mri.denoising.inference import running_inference

# -----------------------------------------------------------------


class LitDenoising(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.model = model
        self.batch_size = self.model.config.batch_size
        self.scheduler_type = self.model.config.scheduler.name
        self.optim_sched = None
        self.loss_fn = None
        self.metric_names = [
            "mse",
            "l1",
            "perpendicular",
            "perceptual",
            "psnr",
            "spec",
            "dwt",
            "ssim",
            "ssim3d",
        ]
        self.metrics = None
        self.durations = []

        self.train_saved_dir = os.path.join(self.config.logging.output_dir, "train_samples")
        self.val_saved_dir = os.path.join(self.config.logging.output_dir, "val_samples")
        self.test_saved_dir = os.path.join(self.config.logging.output_dir, "test_samples")

        if self.config.logging.save_batches_to_output_dir:
            os.makedirs(self.train_saved_dir, exist_ok=True)
            os.makedirs(self.val_saved_dir, exist_ok=True)
            os.makedirs(self.test_saved_dir, exist_ok=True)

        self._epoch_start_time = 0.0

    def forward(self, x):
        return self.model(x)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # create loss function
            if self.loss_fn is None:
                self.loss_fn = Combined_Loss(
                    losses=self.config.loss,
                    loss_weights=self.config.loss_weights,
                    complex_i=True,
                    device=self.device,
                )
            # create metrics
            if self.metrics is None:
                self.metrics = [
                    Combined_Loss(
                        losses=[n], loss_weights=[1.0], complex_i=True, device=self.device
                    )
                    for n in self.metric_names
                ]

                # add PSNR
                self.metric_names.insert(0, "PSNR")
                self.metrics.insert(0, PSNR(range=2048.0))

    def on_train_start(self):
        if self.trainer.global_rank == 0 and self.config.logging.log_train_batches > 0:
            # get some training samples
            tra_loader = torch.utils.data.DataLoader(
                self.trainer.datamodule.train_set,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                drop_last=True,
            )

            loader_iter = iter(tra_loader)

            for batch_idx in range(self.config.logging.log_train_batches):
                noisy, clean, noise_sigma = self._decompose_batch(next(loader_iter))
                self._save_and_log_batches(
                    stage="train",
                    noisy=noisy,
                    clean=clean,
                    noise_sigma=noise_sigma,
                    pred=None,
                    batch_idx=batch_idx,
                )

    def on_train_epoch_start(self):
        # mark epoch start using a stable wall clock
        self._epoch_start_time = time.perf_counter()

    def on_train_epoch_end(self):
        # compute elapsed time safely without relying on internal timers
        start = getattr(self, "_epoch_start_time", None)
        if start is None:
            return
        duration_in_seconds = time.perf_counter() - start
        self.durations.append(duration_in_seconds)

        # log as the max across ranks so the slowest worker defines the epoch time
        self.log(
            "time/epoch_seconds",
            torch.tensor(duration_in_seconds, device=self.device),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="max",
        )

    def _decompose_batch(self, batch):
        noisy, clean, noise_sigma, _ = batch
        # flatten batch and repetition
        if noisy.ndim == 6:
            noisy = noisy.view(-1, *noisy.shape[2:])
            clean = clean.view(-1, *clean.shape[2:])
            noise_sigma = noise_sigma.view(-1, *noise_sigma.shape[2:])
        return noisy, clean, noise_sigma

    def training_step(self, batch, batch_idx):
        try:
            noisy, clean, _ = self._decompose_batch(batch)
            pred = self.model(noisy)
            if torch.isnan(pred).any():
                return None
            loss = self.loss_fn(pred, clean)

            curr_lrs = self.optim_sched.report_lr()
            self.log(
                "lr",
                curr_lrs[0][0],
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                batch_size=self.batch_size,
            )

            self.log(
                "tra/loss",
                loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )

            for i, m in enumerate(self.metrics):
                v = m(pred, clean)
                self.log(
                    f"tra/{self.metric_names[i]}",
                    v,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=self.batch_size,
                )

            if torch.isnan(loss).any():
                return None

            return loss

        except Exception as e:
            print(f"Error computing loss: {e}")
            return None

    def validation_step(self, batch, batch_idx):
        noisy, clean, noise_sigma = self._decompose_batch(batch)
        pred = self.model(noisy)
        loss = self.loss_fn(pred, clean)

        if self.trainer.sanity_checking:
            print(f"Running validation sanity check for batch {batch_idx}")
            self.log(
                "val/loss_sanity_check",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
        else:
            self.log(
                "val/loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )

            # for the validation, to compare different sample more fairly
            B = noisy.shape[0]
            noise_sigmas = torch.reshape(noise_sigma, [B, 1, 1, 1, 1])

            noisy[:, 0] *= noise_sigmas[:, 0]
            noisy[:, 1] *= noise_sigmas[:, 0]
            clean *= noise_sigmas
            pred *= noise_sigmas

            for i, m in enumerate(self.metrics):
                v = m(pred, clean)
                self.log(
                    f"val/{self.metric_names[i]}",
                    v,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=self.batch_size,
                )

            if self.trainer.global_rank == 0:
                self._save_and_log_batches(
                    stage="val",
                    noisy=noisy,
                    clean=clean,
                    noise_sigma=noise_sigma,
                    pred=pred,
                    batch_idx=batch_idx,
                )

    def test_step(self, batch, batch_idx):
        noisy, clean, noise_sigma = self._decompose_batch(batch)
        B, C, T, H, W = noisy.shape
        pred = torch.zeros((B, C - 1, T, H, W), device=noisy.device, dtype=noisy.dtype)

        # since the whole images are used in testing, call the running_inference
        for b in range(B):
            input = noisy[b]
            output = running_inference(
                model=self.model,
                image=input.detach().cpu().numpy(),
                cutout=tuple(self.config.dataset.cutout_shape),
                overlap=tuple(self.config.overlap_for_inference),
                batch_size=self.config.batch_size,
                device=self.device,
            )
            pred[b] = torch.from_numpy(output).to(device=self.device, dtype=pred.dtype)

        noise_sigmas = torch.reshape(noise_sigma, [B, 1, 1, 1, 1])

        noisy[:, 0] *= noise_sigmas[:, 0]
        noisy[:, 1] *= noise_sigmas[:, 0]
        clean *= noise_sigmas
        pred *= noise_sigmas

        for i, m in enumerate(self.metrics):
            v = m(pred, clean)
            self.log(f"test/{self.metric_names[i]}", v, batch_size=B, sync_dist=True)

        if self.trainer.global_rank == 0:
            self._save_and_log_batches(
                stage="test",
                noisy=noisy,
                clean=clean,
                noise_sigma=noise_sigma,
                pred=pred,
                batch_idx=batch_idx,
            )

    def configure_optimizers(self):
        total_num_steps = self.trainer.estimated_stepping_batches
        print(f"Total number of steps: {total_num_steps}")
        self.optim_sched = OptimScheduler(self.config, self.model, total_num_steps)
        return [self.optim_sched.optim], [
            {"scheduler": self.optim_sched.sched, "interval": "step", "frequency": 1}
        ]

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val,
        gradient_clip_algorithm,
    ):
        # Clip the gradient norm using the FSDP-compatible method
        if gradient_clip_val is not None and gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=gradient_clip_val)

    def _save_and_log_batches(self, stage, noisy, clean, noise_sigma, pred, batch_idx):
        noisy_im, clean_im, pred_im, gmap, noise_sigma = _prepare_batch_samples(
            noisy=noisy, clean=clean, output=pred, noise_sigma=noise_sigma
        )

        if stage == "train":
            save_path = self.train_saved_dir
            fname = "train_batch"
            num_logged = self.config.logging.log_train_batches

        elif stage == "val":
            save_path = self.val_saved_dir
            fname = "val_batch"
            num_logged = self.config.logging.log_val_batches

        elif stage == "test":
            save_path = self.test_saved_dir
            fname = "test_batch"
            num_logged = self.config.logging.log_test_batches
        else:
            num_logged = 0

        if batch_idx < num_logged:
            if self.config.logging.save_batches_to_output_dir:
                _save_batch_samples(
                    saved_path=save_path,
                    fname=f"{fname}_{batch_idx}",
                    noisy_im=noisy_im,
                    clean_im=clean_im,
                    pred_im=pred_im,
                    gmap=gmap,
                    noise_sigma=noise_sigma,
                )

            # compute the metrics for logged cases
            B = noisy.shape[0]
            pred_metrics = dict()
            if pred is not None:
                for i, m in enumerate(self.metrics):
                    metric_res = []
                    for b in range(B):
                        v = m(pred[b].unsqueeze(0), clean[b].unsqueeze(0))
                        metric_res.append(v.item())
                    pred_metrics[self.metric_names[i]] = metric_res

            # log as videos
            vid, caption = _save_batch_samples_as_video(
                noisy_im=noisy_im,
                clean_im=clean_im,
                pred_im=pred_im,
                gmap=gmap,
                noise_sigma=noise_sigma,
                pred_metrics=pred_metrics,
            )
            if self.logger is not None and isinstance(
                self.logger, L.pytorch.loggers.wandb.WandbLogger
            ):
                self.logger.experiment.log(
                    {
                        f"{fname}_{batch_idx}": wandb.Video(
                            vid, caption=caption, fps=1, format="gif"
                        )
                    }
                )


# -----------------------------------------------------------------


class DenoisingDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rng = np.random.default_rng(self.config.seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(self.config.seed)

        self.train_data = None
        self.test_set = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if self.train_data is None:
            self.train_data = hydra.utils.instantiate(
                self.config.dataset,
                data_dir=self.config.train_data_dir,
                rng=self.rng,
            )
            val_size = int(len(self.train_data) * self.config.val_data_portion)
            train_size = len(self.train_data) - val_size
            self.train_set, self.val_set = random_split(
                self.train_data, [train_size, val_size], generator=self.generator
            )
            print(
                f"{Fore.GREEN}Train size: {train_size}, Validation size: {val_size}{Style.RESET_ALL}"
            )

        if self.test_set is None:
            self.test_set = MRIDenoisingDatasetTest(
                data_dir=self.config.test_data_dir,
                ignore_gmap=self.config.dataset.ignore_gmap,
                dicom_mode=self.config.dataset.dicom_mode,
            )
            print(f"{Fore.CYAN}Test size: {len(self.test_set)}{Style.RESET_ALL}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
        )


# -----------------------------------------------------------------


def _prepare_batch_samples(noisy, clean, output, noise_sigma):
    noisy_im = noisy.detach().cpu().numpy()
    clean_im = clean.detach().cpu().numpy()
    noise_sigma = noise_sigma.detach().cpu().numpy()

    noisy_im = np.transpose(noisy_im, [3, 4, 2, 1, 0])
    clean_im = np.transpose(clean_im, [3, 4, 2, 1, 0])

    pred_im = None
    if output is not None:
        pred_im = output.detach().to(torch.float32).cpu().numpy()
        pred_im = np.transpose(pred_im, [3, 4, 2, 1, 0])
        pred_im = pred_im[:, :, :, 0, :] + 1j * pred_im[:, :, :, 1, :]

    gmap = noisy_im[:, :, :, 2, :]
    noisy_im = noisy_im[:, :, :, 0, :] + 1j * noisy_im[:, :, :, 1, :]
    clean_im = clean_im[:, :, :, 0, :] + 1j * clean_im[:, :, :, 1, :]

    return noisy_im, clean_im, pred_im, gmap, noise_sigma


def _save_batch_samples(saved_path, fname, noisy_im, clean_im, pred_im, gmap, noise_sigma):
    np.save(os.path.join(saved_path, f"{fname}_noisy_im_real.npy"), np.real(noisy_im))
    np.save(os.path.join(saved_path, f"{fname}_noisy_im_imag.npy"), np.imag(noisy_im))
    np.save(os.path.join(saved_path, f"{fname}_gmap.npy"), gmap)
    np.save(os.path.join(saved_path, f"{fname}_clean_im_real.npy"), np.real(clean_im))
    np.save(os.path.join(saved_path, f"{fname}_clean_im_imag.npy"), np.imag(clean_im))
    np.save(os.path.join(saved_path, f"{fname}_noise_sigma.npy"), noise_sigma)

    if pred_im is not None:
        np.save(os.path.join(saved_path, f"{fname}_pred_im_real.npy"), np.real(pred_im))
        np.save(os.path.join(saved_path, f"{fname}_pred_im_imag.npy"), np.imag(pred_im))


def _save_batch_samples_as_video(noisy_im, clean_im, pred_im, gmap, noise_sigma, pred_metrics):
    """Save a batch and log it as a video if using wandb."""
    new_line = "\n"
    H, W, T, B = noisy_im.shape

    caption = f"{H}-{W}-{T}, "
    vid = np.zeros((T, B * H, 3 * W))  # for noisy, clean, pred
    a_vid = np.zeros((H, 3 * W, T))
    for b in range(B):
        a_vid[:, 0:W, :] = np.abs(noisy_im[:, :, :, b])

        a_clean_im = np.abs(clean_im[:, :, :, b])
        a_vid[:, W : 2 * W, :] = a_clean_im
        if pred_im is not None:
            a_vid[:, 2 * W : 3 * W, :] = np.abs(pred_im[:, :, :, b])
        else:
            a_vid[:, 2 * W : 3 * W, :] = a_clean_im

        gmap_range = np.percentile(gmap[:, :, :, b], [10, 90])
        caption += (
            f"gmap {gmap_range[0]:.4f}-{gmap_range[1]:.4f}, noise {noise_sigma[b].item():.2f}"
        )
        for k, v in pred_metrics.items():
            caption += f", {k} {v[b]:.2f}"
        caption += new_line

        a_vid = np.clip(
            a_vid, a_min=0.5 * np.median(a_clean_im), a_max=np.percentile(a_clean_im, 90)
        )
        temp = np.zeros_like(a_vid)
        vid[:, b * H : (b + 1) * H, :] = np.transpose(
            cv2.normalize(a_vid, temp, 0, 255, norm_type=cv2.NORM_MINMAX), [2, 0, 1]
        )

    return np.repeat(vid[:, np.newaxis, :, :].astype("uint8"), 3, axis=1), caption


# -----------------------------------------------------------------


def after_training(model, config):
    """Save the model for inference and check it is valid."""
    model.eval()
    model.to(device="cpu")

    model_input = torch.randn(
        1,
        3,
        config.dataset.cutout_shape[2],
        config.dataset.cutout_shape[0],
        config.dataset.cutout_shape[1],
        requires_grad=False,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        y = model(model_input)

    model_scripted = torch.jit.trace(model, model_input, strict=False)

    model_scripted_fname = os.path.join(
        config.logging.output_dir, f"model_{config.logging.project}_{config.logging.run_name}.pts"
    )
    model_scripted.save(model_scripted_fname)
    model_scripted_loaded = torch.jit.load(model_scripted_fname)

    with torch.inference_mode():
        y_scripted = model_scripted_loaded(model_input)

    diff = torch.linalg.norm(y - y_scripted)
    if diff / torch.linalg.norm(y) > 1e-3:
        print(
            f"{Fore.YELLOW}Saved model gives different inference results - {model_scripted_fname}{Style.RESET_ALL}"
        )
    else:
        print(
            f"{Fore.GREEN}Saved model is validated for inference - {model_scripted_fname}{Style.RESET_ALL}"
        )

    # further save model in pytorch format
    model_fname = os.path.join(
        config.logging.output_dir, f"model_{config.logging.project}_{config.logging.run_name}.pth"
    )
    torch.save(model.state_dict(), model_fname)

    config_yaml = OmegaConf.to_yaml(config, resolve=True)
    config_fname = os.path.join(
        config.logging.output_dir,
        f"config_{config.logging.project}_{config.logging.run_name}.yaml",
    )
    with open(config_fname, "w") as f:
        f.write(config_yaml)

    return model_scripted_fname, model_fname, config_fname


# -----------------------------------------------------------------

if __name__ == "__main__":
    pass
