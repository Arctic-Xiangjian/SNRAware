"""SNR unit utility functions."""

import math

import numpy as np

from .fftc import fft2c, ifft2c

__all__ = [
    "adjust_matrix_size",
    "apply_matrix_size_reduction_2D",
    "apply_resolution_reduction_2D",
    "zero_padding_resize_2D",
]

# --------------------------------------------------------------


def apply_resolution_reduction_2D(im, ratio_RO, ratio_E1, snr_scaling=True, norm="ortho"):
    """
    Reduce image resolution by setting outer kspace as zeros. The image matrix size stays the same.

    Inputs:
        im: complex image [RO, E1, ...]
        ratio_RO, ratio_E1: ratio to reduce resolution, e.g. 0.75 for 75% resolution
        snr_scaling : if True, apply SNR scaling
        norm : backward or ortho

        snr_scaling should be False and norm should be backward to preserve signal level
    Returns:
        res: complex image with reduced phase resolution [RO, E1, ...]
        fRO, fE1 : equivalent kspace filter
    """
    kspace = fft2c(im, norm=norm)

    RO = kspace.shape[0]
    E1 = kspace.shape[1]

    assert ratio_RO <= 1.0 and ratio_RO > 0
    assert ratio_E1 <= 1.0 and ratio_E1 > 0

    num_masked_RO = int((RO - ratio_RO * RO) // 2)
    num_masked_E1 = int((E1 - ratio_E1 * E1) // 2)

    fRO = np.ones(RO)
    fE1 = np.ones(E1)

    if kspace.ndim == 2:
        if num_masked_RO > 0:
            kspace[0:num_masked_RO, :] = 0
            kspace[RO - num_masked_RO : RO, :] = 0

        if num_masked_E1 > 0:
            kspace[:, 0:num_masked_E1] = 0
            kspace[:, E1 - num_masked_E1 : E1] = 0

    if kspace.ndim == 3:
        if num_masked_RO > 0:
            kspace[0:num_masked_RO, :, :] = 0
            kspace[RO - num_masked_RO : RO, :, :] = 0

        if num_masked_E1 > 0:
            kspace[:, 0:num_masked_E1, :] = 0
            kspace[:, E1 - num_masked_E1 : E1, :] = 0

    if kspace.ndim == 4:
        if num_masked_RO > 0:
            kspace[0:num_masked_RO, :, :, :] = 0
            kspace[RO - num_masked_RO : RO, :, :, :] = 0

        if num_masked_E1 > 0:
            kspace[:, 0:num_masked_E1, :, :] = 0
            kspace[:, E1 - num_masked_E1 : E1, :, :] = 0

    fRO[0:num_masked_RO] = 0
    fRO[RO - num_masked_RO : RO] = 0

    fE1[0:num_masked_E1] = 0
    fE1[E1 - num_masked_E1 : E1] = 0

    if snr_scaling is True:
        ratio = math.sqrt(RO * E1) / math.sqrt((RO - 2 * num_masked_RO) * (E1 - 2 * num_masked_E1))
        im_low_res = ifft2c(kspace) * ratio
    else:
        im_low_res = ifft2c(kspace, norm=norm)

    return im_low_res, fRO, fE1


# --------------------------------------------------------------


def apply_matrix_size_reduction_2D(im, dst_RO, dst_E1, norm="ortho"):
    """
    Reduce the matrix size, keep the field-of-view unchanged.

    Inputs:
        im: complex image [RO, E1, ...]
        dst_RO, dst_E1: target matrix size
        norm : backward or ortho

    Returns:
        res: complex image with reduced matrix size [dst_RO, dst_E1, ...]
    """
    RO = im.shape[0]
    E1 = im.shape[1]

    assert dst_RO <= RO
    assert dst_E1 <= E1

    kspace = fft2c(im, norm=norm)

    num_ro = int((RO - dst_RO) // 2)
    num_e1 = int((E1 - dst_E1) // 2)

    if kspace.ndim == 2:
        kspace_dst = kspace[num_ro : num_ro + dst_RO, num_e1 : num_e1 + dst_E1]
    if kspace.ndim == 3:
        kspace_dst = kspace[num_ro : num_ro + dst_RO, num_e1 : num_e1 + dst_E1, :]
    if kspace.ndim == 4:
        kspace_dst = kspace[num_ro : num_ro + dst_RO, num_e1 : num_e1 + dst_E1, :, :]

    res = ifft2c(kspace_dst, norm=norm)

    return res


# --------------------------------------------------------------


def zero_padding_resize_2D(im, dst_RO, dst_E1, snr_scaling=True, norm="ortho"):
    """
    Zero padding resize up the image.

    Args:
        im ([RO, E1, ...]): complex image
        dst_RO (int): destination size
        dst_E1 (int): destination size
        norm : backward or ortho
    """
    RO = im.shape[0]
    E1 = im.shape[1]

    assert dst_RO >= RO and dst_E1 >= E1

    kspace = fft2c(im, norm=norm)

    new_shape = list(im.shape)
    new_shape[0] = dst_RO
    new_shape[1] = dst_E1
    padding = (np.array(new_shape) - np.array(im.shape)) // 2

    if im.ndim == 2:
        data_padded = np.pad(kspace, [(padding[0], padding[0]), (padding[1], padding[1])])

    if im.ndim == 3:
        data_padded = np.pad(kspace, [(padding[0], padding[0]), (padding[1], padding[1]), (0, 0)])

    if im.ndim == 4:
        data_padded = np.pad(
            kspace, [(padding[0], padding[0]), (padding[1], padding[1]), (0, 0), (0, 0)]
        )

    if snr_scaling is True:
        scaling = np.sqrt(dst_RO * dst_E1) / np.sqrt(RO * E1)
        data_padded *= scaling

    im_padded = ifft2c(data_padded, norm=norm)

    return im_padded


# --------------------------------------------------------------


def adjust_matrix_size(data, ratio):
    """
    Adjust image matrix size by either reducing the matrix size or zero-padding resizing up.

    Args:
        data ([RO, E1]): complex image
        ratio (float): <1.0, reduce matrix size; >1.0, increase matrix size; 1.0, do nothing
    """
    RO, E1 = data.shape[:2]
    dst_RO = round(ratio * RO)
    dst_E1 = round(ratio * E1)

    if RO == dst_RO and E1 == dst_E1:
        return data

    if ratio < 1.0:
        res_im = apply_matrix_size_reduction_2D(data, dst_RO, dst_E1)

    if ratio > 1.0:
        res_im = zero_padding_resize_2D(data, dst_RO, dst_E1)

    return res_im


# --------------------------------------------------------------

if __name__ == "__main__":
    pass
