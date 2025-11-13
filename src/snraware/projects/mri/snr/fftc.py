"""Utility functions for fft and ifft, centered."""

from scipy.fft import fft, fft2, fftn, fftshift, ifft, ifft2, ifftn, ifftshift

__all__ = ["fft1c", "fft2c", "fft3c", "ifft1c", "ifft2c", "ifft3c"]
# --------------------------------------------------------------


def centered_fft(image, norm="ortho"):
    return fftshift(fft2(ifftshift(image), norm=norm))


def fft1c(image, norm="ortho"):
    """
    Perform centered 1D fft.

    Args:
        image ([RO, ...]): Perform fft2c on the first dimension
        norm : 'ortho' or 'backward'
    Returns:
        res: fft1c results
    """
    return fftshift(fft(ifftshift(image, axes=(0,)), axis=0, norm=norm), axes=(0,))


def fft2c(image, norm="ortho"):
    """
    Perform centered 2D fft.

    Args:
        image ([RO, E1, ...]): Perform fft2c on the first two dimensions
        norm : 'ortho' or 'backward'
    Returns:
        res: fft2c results
    """
    return fftshift(fft2(ifftshift(image, axes=(0, 1)), axes=(0, 1), norm=norm), axes=(0, 1))


def fft3c(image, norm="ortho"):
    """
    Perform centered 3D fft.

    Args:
        image ([RO, E1, E2, ...]): Perform fft3c on the first three dimensions
        norm : 'ortho' or 'backward'
    Returns:
        res: fft3c results
    """
    return fftshift(
        fftn(ifftshift(image, axes=(0, 1, 2)), axes=(0, 1, 2), norm=norm), axes=(0, 1, 2)
    )


# --------------------------------------------------------------


def centered_ifft(kspace, norm="ortho"):
    return fftshift(ifft2(ifftshift(kspace), norm=norm))


def ifft1c(kspace, norm="ortho"):
    """
    Perform centered 1D ifft.

    Args:
        image ([RO, ...]): Perform fft2c on the first dimension
        norm : 'ortho' or 'backward'
    Returns:
        res: fft1c results
    """
    return fftshift(ifft(ifftshift(kspace, axes=(0,)), axis=0, norm=norm), axes=(0,))


def ifft2c(kspace, norm="ortho"):
    """
    Perform centered 2D ifft.

    Args:
        image ([RO, E1, ...]): Perform fft2c on the first two dimensions
        norm : 'ortho' or 'backward'
    Returns:
        res: fft2c results
    """
    return fftshift(ifft2(ifftshift(kspace, axes=(0, 1)), axes=(0, 1), norm=norm), axes=(0, 1))


def ifft3c(kspace, norm="ortho"):
    """
    Perform centered 2D ifft.

    Args:
        image ([RO, E1, E2, ...]): Perform fft3c on the first two dimensions
        norm : 'ortho' or 'backward'
    Returns:
        res: fft3c results
    """
    return fftshift(
        ifftn(ifftshift(kspace, axes=(0, 1, 2)), axes=(0, 1, 2), norm=norm), axes=(0, 1, 2)
    )


# --------------------------------------------------------------

if __name__ == "__main__":
    pass
