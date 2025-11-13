import math
import os
from pathlib import Path

import cv2
import numpy as np

from snraware.projects.mri.snr.fftc import fft2c, ifft2c
from snraware.projects.mri.snr.filter import (
    apply_kspace_filter_2D,
    generate_asymmetric_filter,
    generate_symmetric_filter,
)
from snraware.projects.mri.snr.imaging import (
    apply_matrix_size_reduction_2D,
    apply_resolution_reduction_2D,
    zero_padding_resize_2D,
)
from snraware.projects.mri.snr.noise import NoiseGenerator, sample_complex_noise


# -----------------------------------------------------------------
class TestSNRNoise:
    test_path = None
    data_root = None
    im = np.array(None, dtype=object)
    kspace = np.array(None, dtype=object)
    gmap = np.array(None, dtype=object)
    mask = np.array(None, dtype=object)

    def setup_class(self):
        test_path = Path(__file__).parents[0].resolve()
        data_root = str(test_path / "data/snr_unit_data")

        unwrappedIm_real = np.load(data_root + "/unwrappedIm_real.npy")
        unwrappedIm_imag = np.load(data_root + "/unwrappedIm_imag.npy")

        unwrappedIm = unwrappedIm_real + 1j * unwrappedIm_imag
        print("unwrappedIm is ", unwrappedIm.shape)

        unwrappedIm.astype(np.complex256)

        gmap = np.load(os.path.join(data_root, "gmap.npy"))
        print("gmap is ", gmap.shape)

        gmap.astype(np.float64)

        mask = np.load(os.path.join(data_root, "mask.npy"))
        print("mask is ", mask.shape)

        if gmap.ndim == 2:
            gmap = gmap[:, :, np.newaxis]

        kspace = fft2c(unwrappedIm)

        self.im = unwrappedIm
        self.kspace = kspace
        self.gmap = gmap
        self.mask = mask

    def teardown_class(self):
        pass

    # ---------------------------------------------------------------
    def test_snr_unit(self):
        """Test the data has SNR unit."""
        snr_im = self.im / self.gmap
        std_map = np.std(np.abs(snr_im), axis=2)

        noise_level = np.mean(std_map[self.mask > 0])
        print("noise level is ", noise_level)
        assert abs(noise_level - 1) < 0.03

    # ---------------------------------------------------------------
    def test_signal_level_after_reducing_matrix_size(self):
        """Test the resizing function preserves the signal level."""
        RO, E1 = self.im.shape[:2]

        # reduce the size of input image
        im_low_matrix = apply_matrix_size_reduction_2D(
            self.im, int(0.8 * RO), int(0.8 * E1), norm="ortho"
        )
        mask_low_matrix = cv2.resize(
            self.mask, dsize=[int(0.8 * RO), int(0.8 * E1)], interpolation=cv2.INTER_NEAREST
        )

        signal_level = np.abs(np.mean(self.im[self.mask > 0.1]))
        signal_level_low_matrix = np.abs(np.mean(im_low_matrix[mask_low_matrix > 0.1]))

        # the signal preservation resizing should not alter signal level, except the changes due to masking
        assert abs(signal_level - signal_level_low_matrix) / abs(signal_level) < 0.05

    # ---------------------------------------------------------------
    def test_signal_level_after_reducing_resolution(self):
        """Test the resizing function preserves the signal level."""
        im_low_matrix = apply_matrix_size_reduction_2D(
            self.im, int(0.8 * self.im.shape[1]), int(0.8 * self.im.shape[2]), norm="ortho"
        )
        mask_low_matrix = cv2.resize(
            self.mask, dsize=im_low_matrix.shape[1::-1], interpolation=cv2.INTER_NEAREST
        )
        signal_level_low_matrix = np.abs(np.mean(im_low_matrix[mask_low_matrix > 0.1]))

        ratio_RO = 0.57
        ratio_E1 = 0.65
        im_low_matrix_low_res, _fRO, _fE1 = apply_resolution_reduction_2D(
            im_low_matrix, ratio_RO, ratio_E1, snr_scaling=False, norm="backward"
        )

        signal_level_low_matrix_low_res = np.abs(
            np.mean(im_low_matrix_low_res[mask_low_matrix > 0.1])
        )
        assert (
            abs(signal_level_low_matrix - signal_level_low_matrix_low_res)
            / abs(signal_level_low_matrix)
            < 0.02
        )

    # ---------------------------------------------------------------
    def test_signal_level_after_partial_fourier(self):
        """Test the partial fourier preserves the signal level."""
        im_low_matrix = apply_matrix_size_reduction_2D(
            self.im, int(0.8 * self.im.shape[1]), int(0.8 * self.im.shape[2]), norm="ortho"
        )
        mask_low_matrix = cv2.resize(
            self.mask, dsize=im_low_matrix.shape[1::-1], interpolation=cv2.INTER_NEAREST
        )
        signal_level_low_matrix = np.abs(np.mean(im_low_matrix[mask_low_matrix > 0.1]))

        pf_ratio_RO = 0.57
        pf_ratio_E1 = 0.65

        RO, E1 = im_low_matrix.shape[:2]

        start_RO = RO - int(pf_ratio_RO * RO)
        end_RO = RO - 1

        start_E1 = 0
        end_E1 = int(pf_ratio_E1 * E1)

        kspace = fft2c(im_low_matrix)
        kspace[:start_RO, :, :] = 0
        kspace[end_RO + 1 :, :, :] = 0
        kspace[:, :start_E1, :] = 0
        kspace[:, end_E1 + 1 :, :] = 0

        im_low_matrix_pf = ifft2c(kspace)

        signal_level_low_matrix_pf = np.abs(np.mean(im_low_matrix_pf[mask_low_matrix > 0.1]))
        assert (
            abs(signal_level_low_matrix - signal_level_low_matrix_pf)
            / abs(signal_level_low_matrix)
            < 0.02
        )

    # ---------------------------------------------------------------
    def test_noise_creation(self):
        """Test the noise creation function."""
        RO = 192
        E1 = 144

        noise_sigma = 3.7
        nn = sample_complex_noise(noise_sigma, size=(RO, E1, 1024))

        std_r = np.mean(np.std(np.real(nn), axis=2))
        assert abs(std_r - noise_sigma) < 0.01

        std_i = np.mean(np.std(np.imag(nn), axis=2))
        assert abs(std_i - noise_sigma) < 0.01

    # ---------------------------------------------------------------
    def test_kspace_filter_on_RO(self):
        RO, E1 = self.kspace.shape[:2]

        fRO = generate_symmetric_filter(RO, filterType="Gaussian", sigma=1.5)
        fE1 = generate_symmetric_filter(E1, filterType="None", sigma=1.5)

        kspace_filtered = apply_kspace_filter_2D(self.kspace, fRO, fE1)
        im_filtered = ifft2c(kspace_filtered)

        snr_im = im_filtered / self.gmap
        std_map = np.std(np.abs(snr_im), axis=2)
        noise_level = np.mean(std_map[self.mask > 0])
        assert abs(noise_level - 1.0) < 0.02

    # ---------------------------------------------------------------
    def test_kspace_filter(self):
        RO, E1 = self.kspace.shape[:2]

        fRO = generate_symmetric_filter(RO, filterType="Gaussian", sigma=1.23)
        fE1 = generate_symmetric_filter(E1, filterType="Gaussian", sigma=1.45)

        kspace_filtered = apply_kspace_filter_2D(self.kspace, fRO, fE1)
        im_filtered = ifft2c(kspace_filtered)

        snr_im = im_filtered / self.gmap
        std_map = np.std(np.abs(snr_im), axis=2)
        noise_level = np.mean(std_map[self.mask > 0])
        assert abs(noise_level - 1.0) < 0.02

    # ---------------------------------------------------------------
    def test_partial_fourier_filter(self):
        RO, E1 = self.kspace.shape[:2]

        fRO = generate_asymmetric_filter(
            RO, 0, int(0.8 * RO), filterType="TapperedHanning", width=10
        )
        fE1 = generate_asymmetric_filter(E1, int(0.2 * E1), E1, filterType="None", width=20)

        kspace_filtered = apply_kspace_filter_2D(self.kspace, fRO, fE1)
        im_filtered = ifft2c(kspace_filtered)

        snr_im = im_filtered / self.gmap
        std_map = np.std(np.abs(snr_im), axis=2)
        noise_level = np.mean(std_map[self.mask > 0])
        assert abs(noise_level - 1.0) < 0.02

    # ---------------------------------------------------------------
    def test_kspace_and_partial_fourier_filter(self):
        RO, E1 = 512, 256

        # create some filters
        pf_fRO = generate_asymmetric_filter(
            RO, 0, int(0.8 * RO), filterType="TapperedHanning", width=10
        )
        pf_fE1 = generate_asymmetric_filter(E1, 0, E1 - 1, filterType="TapperedHanning", width=20)
        fRO = generate_symmetric_filter(RO, filterType="Gaussian", sigma=1.23)
        fE1 = generate_symmetric_filter(E1, filterType="Gaussian", sigma=1.45)

        # mix the pf and kspace filters
        fRO_used = fRO * pf_fRO
        fE1_used = fE1 * pf_fE1

        ratio_RO = 1 / math.sqrt(1 / RO * np.sum(fRO_used * fRO_used))
        ratio_E1 = 1 / math.sqrt(1 / E1 * np.sum(fE1_used * fE1_used))
        fRO_used *= ratio_RO
        fE1_used *= ratio_E1

        # create testing noise
        noise_sigma = 11.5
        nn = sample_complex_noise(noise_sigma, size=(RO, E1, 256))

        # apply the filter
        kspace_filtered = apply_kspace_filter_2D(fft2c(nn), fRO_used, fE1_used)
        nn_filtered = ifft2c(kspace_filtered)

        std_r_filtered = np.mean(np.std(np.real(nn_filtered), axis=2))
        std_i_filtered = np.mean(np.std(np.imag(nn_filtered), axis=2))

        # the noise level should be preserved
        assert abs(std_r_filtered - noise_sigma) < 0.04
        assert abs(std_i_filtered - noise_sigma) < 0.04

    # ---------------------------------------------------------------
    def test_zero_padding_resize(self):
        RO, E1 = self.im.shape[:2]

        mask_resized = zero_padding_resize_2D(self.mask, 2 * RO, 2 * E1)
        im_resized = zero_padding_resize_2D(self.im, 2 * RO, 2 * E1)

        std_map = np.std(np.abs(im_resized), axis=2)
        noise_level = np.mean(std_map[abs(mask_resized) > 0.1])
        assert abs(noise_level - 1.0) < 0.02

    # ---------------------------------------------------------------
    def test_reduce_resolution(self):
        """
        Test the resolution reduction function.
        The noise level should be preserved after reducing resolution.
        """
        RO, E1 = 192, 148

        ratio_RO = 0.85
        ratio_E1 = 0.65

        noise_sigma = 3.7
        nn = sample_complex_noise(noise_sigma, size=(RO, E1, 256))
        nn, _, _ = apply_resolution_reduction_2D(nn, ratio_RO, ratio_E1, snr_scaling=True)

        std_r = np.mean(np.std(np.real(nn), axis=2))
        std_i = np.mean(np.std(np.imag(nn), axis=2))

        assert abs(std_r - noise_sigma) < 0.02
        assert abs(std_i - noise_sigma) < 0.02

    # ---------------------------------------------------------------
    def test_reduce_matrix_size(self):
        RO, E1 = self.im.shape[:2]

        im_low_matrix = apply_matrix_size_reduction_2D(self.im, int(0.8 * RO), int(0.8 * E1))
        mask_low_matrix = cv2.resize(
            self.mask, dsize=im_low_matrix.shape[:2], interpolation=cv2.INTER_NEAREST
        )

        std_map = np.std(np.abs(im_low_matrix), axis=2)
        noise_level = np.mean(std_map[mask_low_matrix > 0.1])
        assert abs(noise_level - 1.0) < 0.02

    # ---------------------------------------------------------------
    def test_noise_generation(self):
        REP = 1024

        sigmas = np.linspace(1.0, 31.0, 30)
        for sigma in sigmas:
            print(f"Testing noise generation with sigma: {sigma}")
            nn_gen = NoiseGenerator(min_noise_level=sigma, max_noise_level=sigma)

            nns, noise_sigma = nn_gen.generate(RO=32, E1=32, T=8, REP=REP)

            assert nns.shape == (32, 32, 8, REP)
            assert abs(noise_sigma - sigma) < 0.01

            std_nns_real = np.mean(np.std(np.real(nns), axis=3))
            std_nns_imag = np.mean(np.std(np.imag(nns), axis=3))

            assert (
                abs(std_nns_real - nn_gen.actual_sigma_white_noise_real) / abs(std_nns_real) < 0.02
            )
            assert (
                abs(std_nns_imag - nn_gen.actual_sigma_white_noise_imag) / abs(std_nns_imag) < 0.02
            )

    # ---------------------------------------------------------------
