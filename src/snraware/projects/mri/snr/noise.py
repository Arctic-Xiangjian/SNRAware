"""Noise augmentation utilities."""

import time

import numpy as np

from .fftc import fft1c, fft2c, ifft1c, ifft2c
from .filter import (
    apply_kspace_filter_1D,
    apply_kspace_filter_2D,
    generate_asymmetric_filter,
    generate_symmetric_filter,
)
from .imaging import apply_resolution_reduction_2D

__all__ = [
    "NoiseGenerator",
    "generate_pf_filter_in_random",
    "sample_complex_noise",
]

# --------------------------------------------------------------


def sample_complex_noise(noise_sigma, size, rng=None):
    """
    Given the noise sigma, create white complex noise.

    Args:
        noise_sigma (float): noise sigma.
        size (tuple): size of the target tensor.
        rng (np.random.Generator): random number generator instance.

    Returns:
        np.ndarray: complex noise array.
    """
    if rng is None:
        rng = np.random.default_rng()
    nns = rng.standard_normal(size=size) + rng.standard_normal(size=size) * 1j
    return (noise_sigma * nns).astype(np.complex64)


def generate_pf_filter_in_random(RO, E1, rng, pf_filter_ratio):
    # compute pf filter
    pf_lottery = rng.integers(0, 3).item()  # 0, only 1st dim; 1, only 2nd dim; 2, both dim
    pf_ratio_RO = pf_filter_ratio[rng.integers(0, len(pf_filter_ratio)).item()]
    pf_ratio_E1 = pf_filter_ratio[rng.integers(0, len(pf_filter_ratio)).item()]

    if rng.random() < 0.5:  # picke pre or post-zero PF mode
        start = 0
        end = int(pf_ratio_RO * RO)
    else:
        start = RO - int(pf_ratio_RO * RO)
        end = RO - 1
    pf_fRO = generate_asymmetric_filter(RO, start, end, filterType="TapperedHanning", width=10)

    if rng.random() < 0.5:
        start = 0
        end = int(pf_ratio_E1 * E1)
    else:
        start = E1 - int(pf_ratio_E1 * E1)
        end = E1 - 1
    pf_fE1 = generate_asymmetric_filter(E1, start, end, filterType="TapperedHanning", width=10)

    if pf_lottery == 0:  # only apply PF on RO, not E1
        pf_ratio_E1 = 1.0
        pf_fE1 = np.ones(E1)

    if pf_lottery == 1:  # only apply PF on E1, not RO
        pf_ratio_RO = 1.0
        pf_fRO = np.ones(RO)

    return pf_fRO, pf_fE1, pf_ratio_RO, pf_ratio_E1


# --------------------------------------------------------------


class NoiseGenerator:
    def __init__(
        self,
        min_noise_level=1.0,
        max_noise_level=32.0,
        kspace_filter_sigma=None,
        pf_filter_ratio=None,
        kspace_filter_sigma_T=None,
        prob_apply_filter_T=0.5,
        phase_resolution_ratio=None,
        readout_resolution_ratio=None,
        only_white_noise=False,
    ):
        """
        The noise generator produces colored noise by randomly selecting filter strengths and resolution reduction parameters.
        The SNRUnit scaling is applied to ensure that each processing step maintains the fixed noise level.

        The generated noise array has shape [T, RO, E1, REP]. For each repetition, a unique combination of parameters is sampled to create colored noise.

        Args:
            RO (int, optional): readout dimension size. Defaults to 192.
            E1 (int, optional): phase encoding dimension size. Defaults to 144.
            T (int, optional): temporal or frame dimension size. Defaults to 30.
            REP (int, optional): repetitions. Defaults to 1.
            min_noise_level (float, optional): minimum noise level to create. Defaults to 1.0.
            max_noise_level (float, optional): maximum noise level. Defaults to 32.0. The noise level is sampled randomly in the range [min_noise_level, max_noise_level)
            kspace_filter_sigma (list, optional): k-space filter sigma values. Defaults to [0, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0].
            pf_filter_ratio (list, optional): phase filter ratio values. Defaults to [1.0, 0.875, 0.75, 0.625, 0.55].
            kspace_filter_sigma_T (list, optional): k-space filter sigma values for the T dimension. Defaults to [0, 0.5, 0.65, 0.85, 1.0, 1.5, 2.0, 2.25].
            prob_apply_filter_T (float, optional): probability of applying the T dimension filter. Defaults to 0.5.

            phase_resolution_ratio (list, optional): phase resolution ratio values. Defaults to [1.0, 0.85, 0.7, 0.65, 0.55].
            readout_resolution_ratio (list, optional): readout resolution ratio values. Defaults to [1.0, 0.85, 0.7, 0.65, 0.55].
            only_white_noise (bool, optional): whether to use only white noise. Defaults to False.
        """
        if readout_resolution_ratio is None:
            readout_resolution_ratio = [1.0, 0.85, 0.7, 0.65, 0.55]
        if phase_resolution_ratio is None:
            phase_resolution_ratio = [1.0, 0.85, 0.7, 0.65, 0.55]
        if kspace_filter_sigma_T is None:
            kspace_filter_sigma_T = [0, 0.5, 0.65, 0.85, 1.0, 1.5, 2.0, 2.25]
        if pf_filter_ratio is None:
            pf_filter_ratio = [1.0, 0.875, 0.75, 0.625, 0.55]
        if kspace_filter_sigma is None:
            kspace_filter_sigma = [0, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.kspace_filter_sigma = kspace_filter_sigma
        self.pf_filter_ratio = pf_filter_ratio
        self.kspace_filter_sigma_T = kspace_filter_sigma_T
        self.phase_resolution_ratio = phase_resolution_ratio
        self.readout_resolution_ratio = readout_resolution_ratio
        self.prob_apply_filter_T = prob_apply_filter_T

        self.only_white_noise = only_white_noise

        self.rng = np.random.Generator(np.random.PCG64(int(time.time())))
        self.actual_sigma_white_noise_real = 0.0
        self.actual_sigma_white_noise_imag = 0.0

    def generate(self, RO=192, E1=144, T=30, REP=1):
        """
        Generate colored noise for MRI simulations.

        Returns:
            np.ndarray: The generated noise array [RO, E1, T, REP].
            noise_sigma (float): the noise level.
        """
        # randomly sample the noise level
        noise_sigma = (
            self.max_noise_level - self.min_noise_level
        ) * np.random.random_sample() + self.min_noise_level

        # sample the noise at this level
        nns = sample_complex_noise(noise_sigma, (RO, E1, T, REP), self.rng)

        # if only white noise is needed, no further processing is required
        if self.only_white_noise:
            return nns, noise_sigma

        # due to the limitation of number of samples, compute the actual noise level in this particular array
        self.actual_sigma_white_noise_real = np.mean(np.std(np.real(nns), axis=3))
        self.actual_sigma_white_noise_imag = np.mean(np.std(np.imag(nns), axis=3))

        for i in range(REP):
            # ---------------------------------------------
            # apply resolution reduction
            ratio_RO = self.readout_resolution_ratio[
                self.rng.integers(0, len(self.readout_resolution_ratio)).item()
            ]
            ratio_E1 = self.phase_resolution_ratio[
                self.rng.integers(0, len(self.phase_resolution_ratio)).item()
            ]

            # no need to apply snr scaling here, since multiple filters may be applied
            # the precise snr unit will be performed once after applying all filters by counting the number of independent samples
            nns_rep, fdRO, fdE1 = apply_resolution_reduction_2D(
                nns[:, :, :, i], ratio_RO, ratio_E1, snr_scaling=False
            )

            # ---------------------------------------------
            # compute pf filter
            pf_fRO, pf_fE1, _pf_ratio_RO, _pf_ratio_E1 = generate_pf_filter_in_random(
                RO, E1, self.rng, self.pf_filter_ratio
            )

            # ---------------------------------------------
            # compute kspace filter
            ro_filter_sigma = self.kspace_filter_sigma[
                self.rng.integers(0, len(self.kspace_filter_sigma)).item()
            ]
            fRO = generate_symmetric_filter(RO, filterType="Gaussian", sigma=ro_filter_sigma)

            e1_filter_sigma = self.kspace_filter_sigma[
                self.rng.integers(0, len(self.kspace_filter_sigma)).item()
            ]
            fE1 = generate_symmetric_filter(E1, filterType="Gaussian", sigma=e1_filter_sigma)

            fT = None
            if np.random.uniform() < self.prob_apply_filter_T:
                T_filter_sigma = self.kspace_filter_sigma_T[
                    self.rng.integers(0, len(self.kspace_filter_sigma_T)).item()
                ]
                fT = generate_symmetric_filter(T, filterType="Gaussian", sigma=T_filter_sigma)

            # ---------------------------------------------
            # compute final filter
            fROs_used = fRO * pf_fRO * fdRO
            fE1s_used = fE1 * pf_fE1 * fdE1
            # make sure filter is noise preserving
            ratio_RO = 1 / np.sqrt(1 / RO * np.sum(fROs_used * fROs_used))
            ratio_E1 = 1 / np.sqrt(1 / E1 * np.sum(fE1s_used * fE1s_used))

            fROs_used *= ratio_RO
            fE1s_used *= ratio_E1

            # apply the filter
            # here we think the noise is sampled for kspace (to mimic the imaging process)
            # and we apply the filter in the image domain
            nns_rep = ifft2c(apply_kspace_filter_2D(fft2c(nns_rep), fROs_used, fE1s_used))

            if fT is not None:  # if still need to apply filter along temporal dimension
                nns_rep = ifft1c(
                    apply_kspace_filter_1D(fft1c(np.transpose(nns_rep, [2, 0, 1])), fT)
                )
                nns_rep = np.transpose(nns_rep, [1, 2, 0])

            nns[:, :, :, i] = nns_rep

        return nns, noise_sigma


# --------------------------------------------------------------

if __name__ == "__main__":
    pass
