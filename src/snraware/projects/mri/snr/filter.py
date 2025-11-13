"""K-space fiter and partial Fourier filter."""

import math

import numpy as np

from .fftc import fft1c, fft2c, ifft1c, ifft2c

__all__ = [
    "apply_image_filter_2D",
    "apply_image_filter_T",
    "apply_kspace_filter_1D",
    "apply_kspace_filter_2D",
    "apply_kspace_filter_3D",
    "generate_asymmetric_filter",
    "generate_symmetric_filter",
]

# --------------------------------------------------------------


def generate_symmetric_filter(len, filterType, sigma=1.5, snr_scaling=True):
    """
    Compute the SNR unit symmetric filter.

    Args:
        len (int): length of filter
        filterType (str): Gaussian or None
        sigma (float, optional): sigma for gaussian filter. Defaults to 1.5.
        snr_scaling (bool): if True, keep the noise level; if False, keep the signal level

    Returns:
        filter: len array
    """
    filter = np.ones(len, dtype=np.float32)

    if (filterType == "Gaussian") and sigma > 0:
        r = -1.0 * sigma * sigma / 2

        if len % 2 == 0:
            # to make sure the zero points match and boundary of filters are symmetric
            stepSize = 2.0 / (len - 2)
            x = -1 + np.arange(len - 1) * stepSize

            filter = np.zeros(len, dtype=np.float32)
            filter[1:] = np.exp(r * (x * x))
            filter[0] = 0
        else:
            stepSize = 2.0 / (len - 1)
            x = -1 + np.arange(len) * stepSize
            filter = np.exp(r * (x * x))

    if snr_scaling:
        sos = np.sum(filter * filter)
        filter /= math.sqrt(sos / len)
    else:
        filter /= np.max(filter)

    return filter


# --------------------------------------------------------------


def generate_asymmetric_filter(len, start, end, filterType="TapperedHanning", width=10):
    """
    Create the asymmetric kspace filter.

    Args:
        len (int): length of the filter
        start (int): start of filter
        end (int): end of the filter
        filterType (str): None or TapperedHanning
        width (int, optional): width of transition band. Defaults to 10.
    """
    if start > len - 1:
        start = 0

    if end > len - 1:
        end = len - 1

    if start > end:
        start = 0
        end = len - 1

    filter = np.zeros(len, dtype=np.float32)
    filter[start : end + 1] = 1.0

    if width == 0 or width >= len:
        width = 1

    w = np.ones(width)

    if filterType == "TapperedHanning":
        w = 0.5 * (1 - np.cos(2.0 * math.pi * np.arange(1, width + 1) / (2 * width + 1)))

    if start == 0 and end == len - 1:
        filter[:width] = w
        filter[-width:] = w[::-1]

    if start == 0 and end < len - 1:
        filter[end - width + 1 : end + 1] = w[::-1]

    if start > 0 and end == len - 1:
        filter[start : start + width] = w

    if start > 0 and end < len - 1:
        filter[start : start + width] = w
        filter[end - width + 1 : end + 1] = w[::-1]

    sos = np.sum(filter * filter)
    filter /= math.sqrt(sos / (len))

    return filter


# --------------------------------------------------------------


def apply_kspace_filter_1D(kspace, fRO):
    """
    Apply the 1D kspace filter.

    Args:
        kspace ([RO, E1, CHA, PHS]): kspace, can be 1D, 2D or 3D or 4D
        fRO ([RO]): kspace fitler along RO

    Returns:
        kspace_filtered: filtered ksapce
    """
    RO = kspace.shape[0]
    assert fRO.shape[0] == RO

    if kspace.ndim == 1:
        kspace_filtered = kspace * fRO
    if kspace.ndim == 2:
        kspace_filtered = kspace * fRO.reshape((RO, 1))
    if kspace.ndim == 3:
        kspace_filtered = kspace * fRO.reshape((RO, 1, 1))
    if kspace.ndim == 4:
        kspace_filtered = kspace * fRO.reshape((RO, 1, 1, 1))

    return kspace_filtered


# --------------------------------------------------------------


def apply_kspace_filter_2D(kspace, fRO, fE1):
    """
    Apply the 2D kspace filter.

    Args:
        kspace ([RO, E1, CHA, PHS]): kspace, can be 2D or 3D or 4D
        fRO ([RO]): kspace fitler along RO
        fE1 ([E1]): kspace filter along E1

    Returns:
        kspace_filtered: filtered ksapce
    """
    RO = kspace.shape[0]
    E1 = kspace.shape[1]

    assert kspace.ndim <= 4
    assert fRO.shape[0] == RO
    assert fE1.shape[0] == E1

    filter2D = np.outer(fRO, fE1)

    if kspace.ndim == 2:
        kspace_filtered = kspace * filter2D
    if kspace.ndim == 3:
        kspace_filtered = kspace * filter2D[:, :, np.newaxis]
    if kspace.ndim == 4:
        kspace_filtered = kspace * filter2D[:, :, np.newaxis, np.newaxis]

    return kspace_filtered


# --------------------------------------------------------------


def apply_kspace_filter_3D(kspace, fRO, fE1, fE2):
    """
    Apply the 3D kspace filter.

    Args:
        kspace ([RO, E1, E2, CHA]): kspace, can be 3D or 4D
        fRO ([RO]): kspace fitler along RO
        fE1 ([E1]): kspace filter along E1
        fE2 ([E2]): kspace filter along E2

    Returns:
        kspace_filtered: filtered ksapce
    """
    RO = kspace.shape[0]
    E1 = kspace.shape[1]
    E2 = kspace.shape[2]

    assert kspace.ndim <= 4
    assert fRO.shape[0] == RO
    assert fE1.shape[0] == E1
    assert fE2.shape[0] == E2

    filter3D = np.outer(fRO, fE1, fE2)

    if kspace.ndim == 3:
        kspace_filtered = kspace * filter3D
    if kspace.ndim == 4:
        kspace_filtered = kspace * filter3D[:, :, :, np.newaxis]

    return kspace_filtered


# --------------------------------------------------------------


def apply_image_filter_2D(data, sigma_RO=1.25, sigma_E1=1.25):
    """
    Apply a 2D Gaussian filter to the image data on the first two dimensions.

    Args:
        data (ndarray): [RO, E1, T, ...]
        sigma_RO (float, optional): The standard deviation for the Gaussian filter along the RO direction. Defaults to 1.25.
        sigma_E1 (float, optional): The standard deviation for the Gaussian filter along the E1 direction. Defaults to 1.25.

    Returns:
        ndarray: The filtered image data.
    """
    fRO = generate_symmetric_filter(
        data.shape[0], filterType="Gaussian", sigma=sigma_RO, snr_scaling=False
    )
    fE1 = generate_symmetric_filter(
        data.shape[1], filterType="Gaussian", sigma=sigma_E1, snr_scaling=False
    )
    data_filtered = ifft2c(apply_kspace_filter_2D(fft2c(data), fRO, fE1))
    return data_filtered, fRO, fE1


# --------------------------------------------------------------


def apply_image_filter_T(data, sigma_T=1.25):
    """
    Apply a 2D Gaussian filter to the image data along the 3rd dimension.

    Args:
        data (ndarray): [RO, E1, T, ...]
        sigma_T (float, optional): The standard deviation for the Gaussian filter along the T direction. Defaults to 1.25.

    Returns:
        ndarray: The filtered image data.
    """
    fT = generate_symmetric_filter(
        data.shape[2], filterType="Gaussian", sigma=sigma_T, snr_scaling=False
    )

    im = np.transpose(data, (2, 0, 1))
    im = ifft1c(apply_kspace_filter_1D(fft1c(im), fT))
    return np.transpose(im, (1, 2, 0)), fT


# --------------------------------------------------------------

if __name__ == "__main__":
    pass
