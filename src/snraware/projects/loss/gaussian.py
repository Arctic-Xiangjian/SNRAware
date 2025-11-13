"""Gaussian kernel and its derivatives."""

import numpy as np

__all__ = ["create_gaussian_window_1d", "create_gaussian_window_2d", "create_gaussian_window_3d"]


# -------------------------------------------------------------------------------------------------


def gaussian_function(kernelSamplePoints, sigma):
    """
    Compute gaussian and its derivatives.

    Args:
        kernelSamplePoints (np array): sampled kernal points
        sigma (float): guassian sigma

    Returns:
        G, D, DD: guassian kernel, guassian 1st and 2nd order derivatives
    """
    N = 1 / np.sqrt(2 * np.pi * sigma * sigma)
    T = np.exp(-(kernelSamplePoints * kernelSamplePoints) / (2 * sigma * sigma))

    G = N * T
    G = G / np.sum(G)

    D = N * (-kernelSamplePoints / (sigma * sigma)) * T
    D = D / np.sum(np.abs(D))

    DD = N * (
        (-1 / (sigma * sigma) * T)
        + ((-kernelSamplePoints / (sigma * sigma)) * (-kernelSamplePoints / (sigma * sigma)) * T)
    )
    DD = DD / np.sum(np.abs(DD))

    return G, D, DD


# -------------------------------------------------------------------------------------------------


def compute_gaussian_and_derivatives_1D(sigma, halfwidth, voxelsize):
    """
    Compute gaussian kernels and their derivatives.

    Args:
        sigma (float): sigma in the unit of physical world
        halfwidth (float): sampled halfwidth
        voxelsize (float): voxel size, in the same unit of sigma

    Returns:
        kernelSamplePoints, Gx, Dx, Dxx: sampled locations, gaussian and its derivatives
    """
    s = np.arange(2 * round(halfwidth * sigma / voxelsize) + 1)
    kernelSamplePoints = (s - round(halfwidth * sigma / voxelsize)) * voxelsize
    Gx, Dx, Dxx = gaussian_function(kernelSamplePoints, sigma)

    return kernelSamplePoints, Gx, Dx, Dxx


# -------------------------------------------------------------------------------------------------


def create_gaussian_window_1d(sigma=1.25, halfwidth=3, voxelsize=1.0, order=1):
    """Creates a 1D gaussian kernel."""
    k_0 = compute_gaussian_and_derivatives_1D(sigma, halfwidth, voxelsize)
    window = k_0[order + 1]
    window /= np.sum(np.abs(window))

    return window


# -------------------------------------------------------------------------------------------------


def create_gaussian_window_2d(
    sigma=(1.25, 1.25), halfwidth=(3, 3), voxelsize=(1.0, 1.0), order=(1, 1)
):
    """Creates a 2D gaussian kernel."""
    k_0 = compute_gaussian_and_derivatives_1D(sigma[0], halfwidth[0], voxelsize[0])
    k_1 = compute_gaussian_and_derivatives_1D(sigma[1], halfwidth[1], voxelsize[1])
    window = k_0[order[0] + 1][:, np.newaxis] * k_1[order[1] + 1][:, np.newaxis].T
    window /= np.sum(np.abs(window))

    return window


# -------------------------------------------------------------------------------------------------


def create_gaussian_window_3d(
    sigma=(1.25, 1.25, 1.25), halfwidth=(3, 3, 3), voxelsize=(1.0, 1.0, 1.0), order=(1, 1, 1)
):
    """Creates a 3D gaussian kernel."""
    k_0 = compute_gaussian_and_derivatives_1D(sigma[0], halfwidth[0], voxelsize[0])
    k_1 = compute_gaussian_and_derivatives_1D(sigma[1], halfwidth[1], voxelsize[1])
    k_2 = compute_gaussian_and_derivatives_1D(sigma[2], halfwidth[2], voxelsize[2])
    window = k_0[order[0] + 1][:, np.newaxis] * k_1[order[1] + 1][:, np.newaxis].T
    window = window[:, :, np.newaxis] * np.expand_dims(k_2[order[2] + 1], axis=(0, 1))
    window /= np.sum(np.abs(window))

    return window


# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
