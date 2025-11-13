"""Utility functions for MR denoising."""

import io
import os
import pickle

import numpy as np

__all__ = [
    "deserialize_from_bytes",
    "deserialize_from_stream",
    "find_files_with_extension",
    "find_samples_in_folder",
    "serialize_to_bytes",
    "serialize_to_stream",
]

# -------------------------------------------------------------------------------------------------


def serialize_to_stream(noisy, clean, gmap, noise_sigma, stream):
    """Serialize the sample to a stream."""
    pickle.dump((noisy, clean, gmap, noise_sigma), stream)


def deserialize_from_stream(stream):
    """Deserialize the sample from a stream."""
    return pickle.load(stream)


def serialize_to_bytes(
    noisy: np.ndarray,
    clean: np.ndarray,
    gamp: np.ndarray,
    noise_sigma: np.ndarray,
) -> bytes:
    """
    Serialize arrays to a bytes object.

    Uses the same pickle-based format as serialize_to_stream to keep consistency.
    """
    with io.BytesIO() as buf:
        serialize_to_stream(noisy, clean, gamp, noise_sigma, buf)
        return buf.getvalue()


def deserialize_from_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deserialize arrays from a bytes object."""
    with io.BytesIO(data) as buf:
        return deserialize_from_stream(buf)


# -------------------------------------------------------------------------------------------------


def find_files_with_extension(directory, extension):
    """
    Finds all files with a specified extension in a given directory
    and its subdirectories.

    Args:
        directory (str): The path to the starting directory.
        extension (str): The file extension to search for (e.g., ".txt", ".py").

    Returns:
        list: A list of full paths to the found files.
    """
    found_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                found_files.append(os.path.join(root, file))
    return found_files


def find_samples_in_folder(folder):
    """
    Find all samples in a folder and return a list of tuples.

    Args:
        folder (str): The path to the folder containing the samples.
    """
    samples = find_files_with_extension(folder, ".dat")
    return samples


# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import h5py

    h5_file = h5py.File("/data/raw_data/denoising/test_max_noise_level_32.0_1000.h5")
    keys = list(h5_file.keys())
    n = len(keys)

    dst_dir = "/data/raw_data/denoising/test/test_max_noise_level_32.0_1000"
    os.makedirs(dst_dir, exist_ok=True)

    for i in range(n):
        key = keys[i]

        case_dir = os.path.join(dst_dir, key)
        os.makedirs(case_dir, exist_ok=True)

        cases = h5_file[key]

        for case_key in cases.keys():
            data = cases[case_key]
            noisy = np.array(data["noisy"])
            clean = np.array(data["image"])
            gmap = np.array(data["gmap"])
            noise_sigma = np.array(data["noise_sigma"])

            dst_file = os.path.join(case_dir, f"{case_key}.dat")
            with open(dst_file, "wb") as f:
                serialize_to_stream(noisy, clean, gmap, noise_sigma, f)
