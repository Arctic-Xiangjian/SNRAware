"""
Dataset class for MR denoising training. Data is loaded from the file system.

- All data files are loaded from `data_dir`.

- Each sample is organized as:
    /<Group>/<Case> containing both "image" and "gmap".

    - `im` has shape [H, W, T/F]
    - `gmap` has shape [H, W, 1 or T/F, N]
    - N is the number of gmaps; for 2D+T data, gmap.shape[2] is usually 1; for 3D datasets, gmap.shape[2] equals the number of frames F.

- Noise is generated and added to the clean image on the fly.

- The returned sample tensor has shape [repetition, 3, cx, cy, ct]:
    - `repetition` is the number of repetitions
    - `cx`, `cy`, `ct` are the cutout patch sizes
    - 3 channels correspond to real, imaginary, and gmap. If `dicom_mode` is enabled, the imaginary part is zero. Repetitions are created by sampling different noise levels and augmentations from the same input data.

"""

import os
import time

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from skimage.util import view_as_blocks

from snraware.projects.mri.denoising.utils import (
    deserialize_from_stream,
    find_files_with_extension,
)
from snraware.projects.mri.snr.fftc import fft2c, ifft2c
from snraware.projects.mri.snr.imaging import adjust_matrix_size, apply_resolution_reduction_2D
from snraware.projects.mri.snr.noise import NoiseGenerator

__all__ = ["MRIDenoisingDataset", "MRIDenoisingDatasetTest"]
# -------------------------------------------------------------------------------------------------
# train dataset class


class MRIDenoisingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        cutout_shape=(64, 64, 16),
        repetition=1,
        min_noise_level=1.0,
        max_noise_level=32.0,
        kspace_filter_sigma=(0.8, 1.0, 1.5, 2.0, 2.25),
        kspace_filter_sigma_T=(0.25, 0.5, 0.65, 0.85, 1.0, 1.5, 2.0, 2.25),
        prob_apply_filter_T=0.2,
        pf_filter_ratio=(1.0, 0.875, 0.75, 0.625),
        phase_resolution_ratio=(1.0, 0.75, 0.65, 0.55),
        readout_resolution_ratio=(1.0, 0.75, 0.65, 0.55),
        only_white_noise=False,
        ignore_gmap=False,
        add_noise=True,
        add_salt_pepper=False,
        salt_pepper_amount=0.4,
        salt_pepper_prob=0.4,
        add_poisson=False,
        poisson_prob=0.4,
        shuffle_along_3rd_dim=True,
        shuffle_along_3rd_dim_prob=0.5,
        matrix_size_adjust_ratio=(0.5, 0.75, 1.0, 1.25, 1.5),
        matrix_size_adjust_prob=0.25,
        resolution_reduction_prob=0.0,
        partial_fourier_prob=0.0,
        single_frame_mode=True,
        single_frame_mode_prob=0.05,
        dicom_mode=False,
        rng=None,
    ):
        """
        Initialize the denoising dataset.

        Args:
            data_dir (str): Path to the data directory.
            cutout_shape (list): Shape of the image cutouts, for [H, W, T/F]
            repetition (int): Number of repetitions for each sample. If set to K, every sampling
            will create K different noisy versions of the same input. It is useful to improve efficiency.

            Parameters for the NoiseGenerator:
                min_noise_level (float): Minimum noise level.
                max_noise_level (float): Maximum noise level.
                kspace_filter_sigma (list): List of k-space filter sigma values.
                kspace_filter_sigma_T (list): List of k-space filter sigma values for T dimension.
                prob_apply_filter_T (float): Probability of applying T dimension filter.
                pf_filter_ratio (list): List of phase filter ratios.
                phase_resolution_ratio (list): List of phase resolution ratios.
                readout_resolution_ratio (list): List of readout resolution ratios.
                only_white_noise (bool): Whether to use only white noise; if true, not create colored noise.

            ignore_gmap (bool): If ture, force to ignore the input gmap and supply an uniform R=1 array.
            add_noise (bool): Whether to add noise to the images; if false, no noise will be added.

            Parameters for more augmentation:
                add_salt_pepper (bool): Whether to add salt and pepper noise.
                salt_pepper_amount (float): Amount of salt and pepper noise.
                salt_pepper_prob (float): Probability of adding salt and pepper noise.
                add_poisson (bool): Whether to add Poisson noise.
                poisson_prob (float): Probability of adding Poisson noise.
                shuffle_along_3rd_dim (bool): Whether to shuffle cutout frames along the 3rd dimension, to introduce temporal jitter and break signal consistency.
                shuffle_along_3rd_dim_prob (float): Probability of shuffling along the 3rd dimension.
                matrix_size_adjust_ratio (list): List of adjustment ratios. Before adding noise, image is randomly resized according to these ratios.
                matrix_size_adjust_prob (float): Probability of applying matrix size adjustment.
                resolution_reduction_prob (float): Probability of applying resolution reduction on sample images.
                partial_fourier_prob (float): Probability of applying partial fourier on sample images.
                single_frame_mode (bool): To help model for a single 2D image denoising, a frame can be randomly picked from the 2D+T or 3D data and used as input to cut multiple patches. For the case where T/F is 1, single frame model is always applied.
                single_frame_mode_prob (float): Probability of applying single frame model.
                dicom_mode (bool): If true, the image will be scaled by the signal level and converted to integer, to mimic the dicom images.
        """
        assert os.path.exists(data_dir), f"data_dir {data_dir} does not exist."
        assert cutout_shape[0] >= 8 and cutout_shape[1] >= 8 and cutout_shape[2] >= 8, (
            f"cutout_shape {cutout_shape} is not valid."
        )
        assert repetition >= 1, f"repetition {repetition} is not valid."

        self.data_dir = data_dir

        self.cutout_shape = cutout_shape
        self.repetition = repetition

        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.kspace_filter_sigma = kspace_filter_sigma
        self.kspace_filter_sigma_T = kspace_filter_sigma_T
        self.prob_apply_filter_T = prob_apply_filter_T
        self.pf_filter_ratio = pf_filter_ratio
        self.phase_resolution_ratio = phase_resolution_ratio
        self.readout_resolution_ratio = readout_resolution_ratio

        self.ignore_gmap = ignore_gmap
        self.only_white_noise = only_white_noise
        self.add_noise = add_noise
        self.add_salt_pepper = add_salt_pepper
        self.salt_pepper_amount = salt_pepper_amount
        self.salt_pepper_prob = salt_pepper_prob
        self.add_poisson = add_poisson
        self.poisson_prob = poisson_prob

        self.shuffle_along_3rd_dim = shuffle_along_3rd_dim
        self.shuffle_along_3rd_dim_prob = shuffle_along_3rd_dim_prob
        self.matrix_size_adjust_ratio = matrix_size_adjust_ratio
        self.matrix_size_adjust_prob = matrix_size_adjust_prob
        self.resolution_reduction_prob = resolution_reduction_prob
        self.partial_fourier_prob = partial_fourier_prob
        self.single_frame_mode = single_frame_mode
        self.single_frame_mode_prob = single_frame_mode_prob
        self.dicom_mode = dicom_mode

        self.samples = find_files_with_extension(self.data_dir, ".dat")
        print(
            f"{Fore.YELLOW}Found {len(self.samples)} samples from data directory{Style.RESET_ALL}"
        )

        if rng is None:
            self.rng = np.random.Generator(np.random.PCG64(int(time.time())))
        else:
            self.rng = rng

        self.noise_gen = NoiseGenerator(
            min_noise_level=self.min_noise_level,
            max_noise_level=self.max_noise_level,
            kspace_filter_sigma=self.kspace_filter_sigma,
            kspace_filter_sigma_T=self.kspace_filter_sigma_T,
            pf_filter_ratio=self.pf_filter_ratio,
            prob_apply_filter_T=self.prob_apply_filter_T,
            phase_resolution_ratio=self.phase_resolution_ratio,
            readout_resolution_ratio=self.readout_resolution_ratio,
            only_white_noise=self.only_white_noise,
        )

    # -----------------------------------------------------------------------

    def _load_one_sample_from_file(self, fname):
        with open(fname, "rb") as f:
            data = deserialize_from_stream(f)
        return data

    def load_one_sample(self, i):
        """
        Load one sample and create noisy pairs.
        @args:
            - i (int): index of the sample to load
        @rets:
            - noisy_im (5D torch.Tensor): noisy data, in the shape of [repetition, 3, H, W] for real, imag and gmap
            - clean_im (5D torch.Tensor) : clean data, [repetition, 2, H, W] for real, imag
            - noise_sigma (1D torch.Tensor): noise levels sampled, [repetition].
        """
        # get the sample
        deserialized = self._load_one_sample_from_file(self.samples[i])

        data = deserialized[1]
        gmaps = deserialized[2]

        # load data and gmap
        RO, E1, T = data.shape

        # pick a gmap and check its consistency to the data
        num_gmaps = 1
        if self.ignore_gmap:
            gmaps = np.ones((RO, E1, T, num_gmaps), dtype=np.float32)
        else:
            assert gmaps.shape[0] == RO and gmaps.shape[1] == E1

            if gmaps.ndim == 2:
                gmaps = np.expand_dims(gmaps, axis=(2, 3))

            if gmaps.shape[2] == 1:
                gmaps = np.repeat(gmaps, T, axis=2)

        data = data.astype(np.complex64)
        gmaps = gmaps.astype(np.float32)

        # augmentation with the random flip
        data, gmaps = self._random_flip(data, gmaps)

        # augmentation with the matrix size adjustment
        if self.matrix_size_adjust_prob > 0 and self.rng.uniform() < self.matrix_size_adjust_prob:
            data, gmaps = self._random_adjust_matrix_size(data, gmaps)
            RO, E1, T = data.shape

        # ---------------------------------------------------------
        # for every repetition, create a pair of samples
        clean_data = None
        noisy_data = None
        noise_sigma = np.zeros((self.repetition,), dtype=np.float32)
        noise_sigma_generated = np.zeros((self.repetition,), dtype=np.float32)
        gmap_data = None
        single_frame_model = (T == 1) or (
            self.single_frame_mode and self.rng.uniform() < self.single_frame_mode_prob
        )
        for rep in range(self.repetition):
            # we can sample a different gmap every repetition
            gmap, _picked_ind_gmap = self._load_gmap(gmaps, index_picked=-1)

            if single_frame_model:
                gmap_rep = self._pad_data_2D_for_cutout(gmap)
                if rep == 0:  # only need to get the 2D frame once for all repetitions
                    data_rep, gmap_rep = self._process_single_frame_mode(
                        self._pad_data_2D_for_cutout(data), gmap_rep
                    )
                else:
                    _, gmap_rep = self._process_single_frame_mode(
                        self._pad_data_2D_for_cutout(data), gmap_rep
                    )

            else:
                # ensure sample size is still valid for cutout
                data_rep = self._pad_data_for_cutout(data)
                gmap_rep = self._pad_data_for_cutout(gmap)

            if rep == 0:
                clean_data = np.zeros((self.repetition, *data_rep.shape), dtype=np.complex64)
                noisy_data = np.zeros((self.repetition, *data_rep.shape), dtype=np.complex64)
                gmap_data = np.zeros((self.repetition, *gmap_rep.shape), dtype=np.float32)

            # create and add noise; different noise is sampled for every repetition
            clean_data[rep], noisy_data[rep], noise_sigma[rep], noise_sigma_generated[rep] = (
                self._generate_corrupted_data(data_rep, gmap_rep)
            )
            gmap_data[rep] = gmap_rep

        # ---------------------------------------------------------
        # augment with other noise types if needed
        if self.add_salt_pepper and self.rng.uniform() < self.salt_pepper_prob:
            noisy_data = self._add_salt_and_pepper_noise(noisy_data)

        if self.add_poisson and self.rng.uniform() < self.poisson_prob:
            noisy_data = self._add_poisson_noise(clean_data, noisy_data)

        if self.dicom_mode:
            clean_data, noisy_data, gmap_data = self._process_dicom_mode(
                clean_data, noisy_data, gmap_data
            )

        # ---------------------------------------------------------
        # cut out the patch
        s_x, s_y, s_t = self._get_cutout_range(clean_data[0])

        clean_data_patch = self._get_cutout(clean_data, s_x, s_y, s_t)
        noisy_data_patch = self._get_cutout(noisy_data, s_x, s_y, s_t)
        gmap_patch = self._get_cutout(gmap_data, s_x, s_y, s_t)

        # augment to introduce some inconsistency
        if (self.shuffle_along_3rd_dim) and (self.rng.uniform() < self.shuffle_along_3rd_dim_prob):
            # perform shuffle along the 3rd dimension
            t_indexes = np.arange(clean_data_patch.shape[-1])
            np.random.shuffle(t_indexes)

            clean_data_patch = clean_data_patch[:, :, :, t_indexes]
            noisy_data_patch = noisy_data_patch[:, :, :, t_indexes]

        # ---------------------------------------------------------
        # concatenate the real and imag parts
        clean_data_patch = np.expand_dims(clean_data_patch, axis=0)
        noisy_data_patch = np.expand_dims(noisy_data_patch, axis=0)
        gmap_patch = np.expand_dims(gmap_patch, axis=0)

        noisy_data = np.concatenate(
            [np.real(noisy_data_patch), np.imag(noisy_data_patch), gmap_patch], axis=0
        )
        clean_data = np.concatenate([np.real(clean_data_patch), np.imag(clean_data_patch)], axis=0)

        # ---------------------------------------------------------
        # from [C, REP, H, W, T] to [REP, C, T, H, W]
        noisy_data = np.transpose(noisy_data, [1, 0, 4, 2, 3])
        clean_data = np.transpose(clean_data, [1, 0, 4, 2, 3])

        return (
            torch.from_numpy(noisy_data.astype(np.float32)),
            torch.from_numpy(clean_data.astype(np.float32)),
            torch.from_numpy(noise_sigma.astype(np.float32)),
            torch.from_numpy(noise_sigma_generated.astype(np.float32)),
        )

    # ----------------------------------------------------------------

    def _pad_data_2D_for_cutout(self, data):
        if data.shape[0] < self.cutout_shape[0]:
            data = np.pad(
                data, ((0, self.cutout_shape[0] - data.shape[0]), (0, 0), (0, 0)), "symmetric"
            )
        if data.shape[1] < self.cutout_shape[1]:
            data = np.pad(
                data, ((0, 0), (0, self.cutout_shape[1] - data.shape[1]), (0, 0)), "symmetric"
            )
        return data

    def _pad_data_for_cutout(self, data):
        data = self._pad_data_2D_for_cutout(data)
        if data.shape[2] < self.cutout_shape[2]:
            data = np.pad(
                data, ((0, 0), (0, 0), (0, self.cutout_shape[2] - data.shape[2])), "symmetric"
            )
        return data

    # ----------------------------------------------------------------

    def _get_cutout_range(self, data):
        x, y, t = data.shape
        cx, cy, ct = self.cutout_shape

        s_x = self.rng.integers(0, x - cx + 1)
        s_y = self.rng.integers(0, y - cy + 1)
        s_t = self.rng.integers(0, t - ct + 1)

        return s_x, s_y, s_t

    # -----------------------------------------------------------

    def _get_cutout(self, data, s_x, s_y, s_t):
        """Given the start points, cut out a patch."""
        cx, cy, ct = self.cutout_shape
        if data.ndim == 4:
            return data[:, s_x : s_x + cx, s_y : s_y + cy, s_t : s_t + ct]
        else:
            return data[s_x : s_x + cx, s_y : s_y + cy, s_t : s_t + ct]

    # ----------------------------------------------------------------

    def _load_gmap(self, gmaps, index_picked=-1):
        """
        Loads a random gmap. Note it is a chance to get the R=1 gmap (aka 1.0 everywhere), no acceleration.

        Args:
            gmaps (np.array): gmap [H, W, 1 or T/F, N]
            index_picked (int, optional): if set, return gmap[:,:,:,index_picked]. Defaults to -1 for random picking.

        Returns:
            a_gmap (np.array): picked gmap [H, W, 1 or T/F]
        """
        N = gmaps.shape[-1]
        if index_picked >= 0 and index_picked < N:
            return gmaps[:, :, :, index_picked]

        random_factor = self.rng.integers(0, N + 1)
        if random_factor < N:
            return gmaps[:, :, :, random_factor], random_factor
        else:
            return np.ones(gmaps.shape[:3]), N

    # ----------------------------------------------------------------

    def _random_flip(self, data, gmap):
        """Randomly flips the input image and gmap."""
        flip1 = self.rng.integers(0, 2) > 0
        flip2 = self.rng.integers(0, 2) > 0

        def flip(image):
            if image.ndim == 4:
                if flip1:
                    image = image[::-1, :, :, :].copy()
                if flip2:
                    image = image[:, ::-1, :, :].copy()
            elif image.ndim == 2:
                if flip1:
                    image = image[::-1, :].copy()
                if flip2:
                    image = image[:, ::-1].copy()
            else:
                if flip1:
                    image = image[::-1, :, :].copy()
                if flip2:
                    image = image[:, ::-1, :].copy()
            return image

        return flip(data), flip(gmap)

    # ----------------------------------------------------------------

    def _random_adjust_matrix_size(self, data, gmaps):
        matrix_size_adjust_ratio = self.matrix_size_adjust_ratio[
            self.rng.integers(0, len(self.matrix_size_adjust_ratio)).item()
        ]
        data_adjusted = adjust_matrix_size(data, matrix_size_adjust_ratio)

        gmaps_adjusted = np.zeros([*data_adjusted.shape, gmaps.shape[-1]])
        for rep in range(gmaps.shape[-1]):
            for k in range(gmaps.shape[2]):
                gmaps_adjusted[:, :, k, rep] = cv2.resize(
                    gmaps[:, :, k, rep],
                    dsize=(data_adjusted.shape[1], data_adjusted.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

        return data_adjusted, gmaps_adjusted

    # ----------------------------------------------------------------

    def _random_reduce_resolution(self, data):
        ratio_RO = self.readout_resolution_ratio[
            self.rng.integers(0, len(self.readout_resolution_ratio)).item()
        ]
        ratio_E1 = self.phase_resolution_ratio[
            self.rng.integers(0, len(self.phase_resolution_ratio)).item()
        ]

        data_reduced_resolution, _, _ = apply_resolution_reduction_2D(
            im=data, ratio_RO=ratio_RO, ratio_E1=ratio_E1, snr_scaling=False, norm="backward"
        )

        return data_reduced_resolution

        # ro_filter_sigma = self.kspace_filter_sigma[self.rng.integers(0, len(self.kspace_filter_sigma)).item()]
        # e1_filter_sigma = self.kspace_filter_sigma[self.rng.integers(0, len(self.kspace_filter_sigma)).item()]
        # data_reduced_resolution_filtered, _, _ = apply_image_filter_2D(
        #     data_reduced_resolution, sigma_RO=ro_filter_sigma, sigma_E1=e1_filter_sigma
        # )

        # if self.rng.uniform() < self.prob_apply_filter_T:
        #     T_filter_sigma = self.kspace_filter_sigma_T[self.rng.integers(0, len(self.kspace_filter_sigma_T)).item()]
        #     data_degraded, _ = apply_image_filter_T(data_reduced_resolution_filtered, sigma_T=T_filter_sigma)
        #     return data_degraded
        # else:
        #     return data_reduced_resolution_filtered

    # ----------------------------------------------------------------

    def _random_partial_fourier(self, data):
        RO, E1 = data.shape[0], data.shape[1]
        pf_lottery = self.rng.integers(
            0, 3
        ).item()  # 0, only 1st dim; 1, only 2nd dim; 2, both dim
        pf_ratio_RO = self.pf_filter_ratio[self.rng.integers(0, len(self.pf_filter_ratio)).item()]
        pf_ratio_E1 = self.pf_filter_ratio[self.rng.integers(0, len(self.pf_filter_ratio)).item()]

        if self.rng.random() < 0.5:  # picke pre or post-zero PF mode
            start_RO = 0
            end_RO = int(pf_ratio_RO * RO)
        else:
            start_RO = RO - int(pf_ratio_RO * RO)
            end_RO = RO - 1

        if self.rng.random() < 0.5:
            start_E1 = 0
            end_E1 = int(pf_ratio_E1 * E1)
        else:
            start_E1 = E1 - int(pf_ratio_E1 * E1)
            end_E1 = E1 - 1

        if pf_lottery == 0:  # only apply PF on RO, not E1
            start_E1 = 0
            end_E1 = E1 - 1

        if pf_lottery == 1:  # only apply PF on E1, not RO
            start_RO = 0
            end_RO = RO - 1

        kspace = fft2c(data)
        kspace[:start_RO, :, :] = 0
        kspace[end_RO + 1 :, :, :] = 0
        kspace[:, :start_E1, :] = 0
        kspace[:, end_E1 + 1 :, :] = 0

        data_pf = ifft2c(kspace)
        return data_pf

    # ----------------------------------------------------------------

    def _process_single_frame_mode(self, data, gmap):
        """Pick a 2D frame and extract patches to add the 3rd dimension."""
        T = data.shape[2]

        picked_t = self.rng.integers(0, T)
        data_frame = data[:, :, picked_t]
        gmap_frame = gmap[:, :, picked_t]

        def prepare_patches(frame):
            pad_H = (-1 * frame.shape[0]) % self.cutout_shape[0]
            pad_W = (-1 * frame.shape[1]) % self.cutout_shape[1]
            frame = np.pad(frame, ((0, pad_H), (0, pad_W)), "symmetric")
            frame = view_as_blocks(frame, (self.cutout_shape[0], self.cutout_shape[1]))
            frame = np.reshape(frame, [-1, *frame.shape[-2:]])
            num_patches = frame.shape[0]
            if num_patches > self.cutout_shape[2]:
                start_t = self.rng.integers(0, max(num_patches - self.cutout_shape[2], 1))
                frame = frame[start_t : start_t + self.cutout_shape[2]]
            elif num_patches < self.cutout_shape[2]:
                pad_T = (self.cutout_shape[2] - num_patches) // 2
                pad_T_end = self.cutout_shape[2] - (num_patches + pad_T)
                frame = np.pad(frame, ((pad_T, pad_T_end), (0, 0), (0, 0)), "symmetric")

            return frame

        data_patches_2D = prepare_patches(data_frame)
        gmap_patches_2D = prepare_patches(gmap_frame)

        return np.transpose(data_patches_2D, [1, 2, 0]), np.transpose(gmap_patches_2D, [1, 2, 0])

    # ----------------------------------------------------------------

    def _apply_dicom_converion(self, im, scaling=1000.0):
        mag = np.abs(im)
        out = np.copy(im)
        out.real = np.round(scaling * mag)
        out.imag = 0
        return out

    def _process_dicom_mode(self, clean_data, noisy_data, gmap):
        signal_scaling = np.percentile(np.abs(clean_data), 95)
        clean_res = self._apply_dicom_converion(clean_data, scaling=1000.0 / signal_scaling)
        noisy_res = self._apply_dicom_converion(noisy_data, scaling=1000.0 / signal_scaling)
        return clean_res, noisy_res, np.full(gmap.shape, 1.0)

    # ----------------------------------------------------------------

    def _generate_corrupted_data(self, data, gmap):
        res = np.copy(data)

        if self.rng.uniform() < self.resolution_reduction_prob:
            res = self._random_reduce_resolution(res)

        if self.rng.uniform() < self.partial_fourier_prob:
            res = self._random_partial_fourier(res)

        if self.add_noise:
            # create noise
            nn, noise_sigma = self.noise_gen.generate(
                RO=data.shape[0], E1=data.shape[1], T=data.shape[2], REP=1
            )
            noise_sigma_generated = np.std(nn)

            # apply gmap
            nn_rep = np.squeeze(nn) * gmap

            # add noise to complex image and scale
            res += nn_rep

            # scale the data
            noise_scaling_ratio = 1.0
            if noise_sigma > 0:
                noise_scaling_ratio = np.sqrt(noise_sigma * noise_sigma + 1).item()

            clean_data = data / noise_scaling_ratio
            noisy_data = res / noise_scaling_ratio
            noise_sigmas = noise_sigma
        else:
            clean_data = data
            noisy_data = res
            noise_sigmas = 0.0
            noise_sigma_generated = 0.0

        return clean_data, noisy_data, noise_sigmas, noise_sigma_generated

    # ----------------------------------------------------------------

    def _add_salt_and_pepper_noise(self, noisy_data):
        s_vs_p = self.rng.uniform()  # a random number to decide how much salt and how much pepper
        amount = self.rng.uniform(0, self.salt_pepper_amount)
        out = np.copy(noisy_data)

        # Salt mode
        num_salt = np.ceil(amount * noisy_data.size * s_vs_p)
        coords = self.rng.integers(0, noisy_data.size, int(num_salt))
        cc = np.unravel_index(coords, noisy_data.shape)
        out[cc] *= self.rng.uniform(1.0, 10.0)

        # Pepper mode
        num_pepper = np.ceil(amount * noisy_data.size * (1.0 - s_vs_p))
        coords = self.rng.integers(0, noisy_data.size, int(num_pepper))
        cc = np.unravel_index(coords, noisy_data.shape)
        out[cc] *= self.rng.uniform(0, 1.0)

        return out

    # ----------------------------------------------------------------

    def _add_poisson_noise(self, clean_data, noisy_data):
        lam_ratio = self.rng.integers(1, 10)
        mag = np.abs(clean_data) / lam_ratio
        pn = np.random.poisson(mag, clean_data.shape) - mag
        return noisy_data + pn

    # ----------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    # ----------------------------------------------------------------

    def __getitem__(self, index):
        return self.load_one_sample(index)


# -------------------------------------------------------------------------------------------------
# test dataset class


class MRIDenoisingDatasetTest(MRIDenoisingDataset):
    """
    Dataset for MRI denoising testing.
    Returns full images. No cutouts.

    The test dataset has the entry for noisy_images as the input to the network.

    /<Group>/<Case> --> "image" for ground-truth
    /<Group>/<Case> --> "noisy" for test input
    /<Group>/<Case> --> "noise_sigma" for the noise level of this test case

    """

    def __init__(self, data_dir, ignore_gmap=False, dicom_mode=False):
        super().__init__(
            data_dir=data_dir,
            ignore_gmap=ignore_gmap,
            dicom_mode=dicom_mode,
        )

    def load_one_sample(self, i):
        """
        Loads one sample from the saved images
        @args:
            - i (int): index of retreive
        @rets:
            - noisy (5D torch.Tensor): noisy data, in the shape of [3, T, H, W]
            - clean (5D torch.Tensor) : clean data, [1, RO, E1] for magnitude and [2, T, H, W] for complex
            - noise_sigma (0D torch.Tensor): noise sigma added to the image.
        """
        deserialized = self._load_one_sample_from_file(self.samples[i])

        noisy_data = deserialized[0]
        clean_data = deserialized[1]
        gmap = deserialized[2]
        noise_sigma = deserialized[3]

        # load data and gmap
        RO, E1, T = clean_data.shape

        # pick a gmap and check its consistency to the data
        if self.ignore_gmap:
            gmap = np.ones((RO, E1, T), dtype=np.float32)

        if gmap.ndim == 2 or gmap.shape[2] == 1:
            gmap = np.repeat(gmap[:, :, np.newaxis], T, axis=2)

        clean_data = clean_data.astype(np.complex64)
        noisy_data = noisy_data.astype(np.complex64)
        gmap = gmap.astype(np.float32)

        if self.dicom_mode:
            clean_data, noisy_data, gmap = self._process_dicom_mode(clean_data, noisy_data, gmap)

        clean_data = np.expand_dims(clean_data, axis=0)
        noisy_data = np.expand_dims(noisy_data, axis=0)
        gmap = np.expand_dims(gmap, axis=0)

        noisy_data = np.concatenate([np.real(noisy_data), np.imag(noisy_data), gmap], axis=0)
        clean_data = np.concatenate([np.real(clean_data), np.imag(clean_data)], axis=0)

        # from [C, H, W, T] to [C, T, H, W]
        noisy_data = np.transpose(noisy_data, [0, 3, 1, 2])
        clean_data = np.transpose(clean_data, [0, 3, 1, 2])

        return (
            torch.from_numpy(noisy_data.astype(np.float32)),
            torch.from_numpy(clean_data.astype(np.float32)),
            torch.from_numpy(noise_sigma.astype(np.float32)),
            torch.from_numpy(noise_sigma.astype(np.float32)),
        )


# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# -----------------------------------------------------------------
