"""
image_processing.py
-------------------
Functions for loading, preprocessing, and skeletonizing microscopy images.
"""

import os
import numpy as np
import cv2
from skimage.morphology import skeletonize, remove_small_objects, dilation, disk
from scipy.ndimage import generic_filter


def _to_2d(img: np.ndarray) -> np.ndarray:
    """
    Collapse a loaded image to a single 2-D array.

    OpenCV sometimes returns 3-channel arrays even for grayscale TIFFs.
    If all channels are identical (true grayscale), we just take channel 0.
    If channels differ, we convert to grayscale via luminance weighting.

    Parameters
    ----------
    img : np.ndarray
        2-D or 3-D array as returned by cv2.imread.

    Returns
    -------
    np.ndarray
        2-D array of the same dtype.
    """
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] == 1:
            return img[:, :, 0]
        # Check if all channels are identical (common for grayscale TIFFs)
        if np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 0], img[:, :, 2]):
            return img[:, :, 0]
        # Fall back to proper grayscale conversion
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unexpected image shape: {img.shape}")


def detect_and_normalize(img: np.ndarray) -> np.ndarray:
    """
    Auto-detect pixel value convention and normalize to {0, 255}.

    Handles all known binary segmentation formats:

    ======================================  ================================
    Input unique values                     Convention assumed
    ======================================  ================================
    ``{0, 255}``                            Already correct — pass through.
    Any other two-value image               Signal = minority class.
    ======================================  ================================

    The minority-class heuristic works because microglia/neuron processes
    occupy a small fraction of the image frame, so the background is always
    the majority class regardless of whether it is encoded as 0, 1, or 2.

    Parameters
    ----------
    img : np.ndarray
        Raw 2-D image array (any dtype).

    Returns
    -------
    np.ndarray
        uint8 array with background = 0 and signal = 255.

    Raises
    ------
    ValueError
        If the image contains more than two unique pixel values.
    """
    if img.ndim != 2:
        raise ValueError(
            f"detect_and_normalize expects a 2-D array, got shape {img.shape}. "
            f"Call _to_2d() first."
        )

    unique_vals = np.unique(img)

    if len(unique_vals) > 2:
        raise ValueError(
            f"Expected a binary image with exactly 2 unique pixel values, "
            f"got {len(unique_vals)}: {unique_vals}. "
            f"Make sure the image is a binary segmentation."
        )

    # Already correct
    if set(unique_vals) == {0, 255}:
        return img.astype(np.uint8)

    # Any other two-value encoding ({0,1}, {1,2}, {1,255}, etc.)
    vals, counts = np.unique(img.flatten(), return_counts=True)
    signal_val = vals[np.argmin(counts)]
    normalized = np.where(img == signal_val, 255, 0).astype(np.uint8)

    detected_str = "{" + str(vals[0]) + ", " + str(vals[1]) + "}"
    print(
        f"  [normalize] Detected pixel values {detected_str} — "
        f"treating {signal_val} as signal (minority class)."
    )

    return normalized


def load_and_preprocess(filepath: str):
    """
    Load a TIFF image and normalize it to a 2-D binary {0, 255} array.

    Handles multi-channel TIFFs, grayscale TIFFs, and all known binary
    pixel-value conventions automatically.

    Parameters
    ----------
    filepath : str
        Full path to the ``.tiff`` image file.

    Returns
    -------
    img_normalized : np.ndarray
        2-D uint8 array with background = 0 and signal = 255.
    img_raw : np.ndarray
        The raw 2-D array as loaded (before normalization).

    Raises
    ------
    FileNotFoundError
        If *filepath* does not point to a readable image.
    ValueError
        If the image is not binary (more than 2 unique pixel values).
    """
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {filepath}")

    img_2d = _to_2d(img)
    img_normalized = detect_and_normalize(img_2d)

    print(f"  [load] Shape: {img.shape} → 2-D {img_2d.shape}  |  "
          f"unique values before norm: {np.unique(img_2d).tolist()}")

    return img_normalized, img_2d


def skeletonize_image(
    img_normalized: np.ndarray,
    min_size: int = 10,
    connectivity: int = 25,
) -> np.ndarray:
    """
    Skeletonize a binary image and remove small isolated objects.

    Parameters
    ----------
    img_normalized : np.ndarray
        2-D uint8 binary image (foreground = 255, background = 0).
    min_size : int, optional
        Minimum number of pixels for an object to be retained (default 10).
    connectivity : int, optional
        Connectivity used when removing small objects (default 25).

    Returns
    -------
    processed_skeleton : np.ndarray
        2-D float array of the cleaned skeleton (foreground = 255).
    """
    if img_normalized.ndim != 2:
        raise ValueError(
            f"skeletonize_image expects a 2-D array, got shape {img_normalized.shape}."
        )

    # skeletonize expects a bool array where True = foreground
    bool_img = img_normalized > 0
    skeleton = skeletonize(bool_img)

    processed = remove_small_objects(
        skeleton, min_size=min_size, connectivity=connectivity
    ).astype(np.uint8)

    processed_skeleton = np.where(processed > 0, 255, 0).astype(float)
    return processed_skeleton


def dilate_skeleton(skeleton: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Dilate the skeleton slightly so that circle intersections are not missed.

    Parameters
    ----------
    skeleton : np.ndarray
        2-D skeleton array.
    radius : int, optional
        Dilation disk radius (default 1).

    Returns
    -------
    np.ndarray
        Dilated skeleton array.
    """
    return dilation(skeleton, disk(radius=radius))


def find_endpoints(skeleton: np.ndarray):
    """
    Identify branch endpoints in the skeleton.

    A pixel is an endpoint if it is active (255) and has exactly one
    active neighbour in its 3x3 neighbourhood.

    Parameters
    ----------
    skeleton : np.ndarray
        2-D skeleton array (foreground = 255).

    Returns
    -------
    endpoint_arr : np.ndarray
        Array of same shape as *skeleton*; endpoint pixels are 255.
    n_endpoints : int
        Total number of endpoints found.
    """

    def _num_endpoints(p):
        return 255 * ((p[4] == 255) and np.sum(p) == 510)

    endpoint_arr = generic_filter(skeleton, _num_endpoints, (3, 3))

    counts = np.unique(endpoint_arr, return_counts=True)
    n_endpoints = counts[1][1] if len(counts[1]) > 1 else 0

    return endpoint_arr, n_endpoints


def smooth_binary(
    img_normalized: np.ndarray,
    gaussian_sigma: float = 1.0,
    threshold: int = 128,
) -> np.ndarray:
    """
    Smooth a binary image with a Gaussian blur then re-threshold.

    Useful for "chunky" segmentations where thick, blocky processes would
    produce a noisy skeleton. The blur softens the edges, and the
    re-threshold produces cleaner, thinner filled regions for skeletonizing.

    Pipeline::

        binary {0,255}  →  gaussian blur  →  re-threshold  →  binary {0,255}

    Parameters
    ----------
    img_normalized : np.ndarray
        2-D uint8 binary image with background = 0 and signal = 255,
        as returned by :func:`load_and_preprocess`.
    gaussian_sigma : float, optional
        Standard deviation of the Gaussian kernel in pixels (default 1.0).
        Larger values = more smoothing. A value of 0 disables smoothing
        and returns the image unchanged.
        Typical range: 0.5 (mild) to 3.0 (heavy).
    threshold : int, optional
        Pixel value cutoff after blurring (default 128). Pixels above this
        value become signal (255); at or below become background (0).
        Lower values preserve more of the blurred signal.

    Returns
    -------
    np.ndarray
        Smoothed binary uint8 image with background = 0 and signal = 255.

    Examples
    --------
    Mild smoothing before skeletonizing::

        img, _ = load_and_preprocess("cell.tiff")
        img_smooth = smooth_binary(img, gaussian_sigma=1.0)
        skeleton = skeletonize_image(img_smooth)

    No smoothing (passthrough)::

        img_smooth = smooth_binary(img, gaussian_sigma=0)
    """
    if gaussian_sigma == 0:
        return img_normalized.copy()

    if img_normalized.ndim != 2:
        raise ValueError(
            f"smooth_binary expects a 2-D array, got shape {img_normalized.shape}."
        )

    blurred = cv2.GaussianBlur(
        img_normalized,
        ksize=(0, 0),          # auto kernel size from sigma
        sigmaX=gaussian_sigma,
        sigmaY=gaussian_sigma,
    )

    smoothed = np.where(blurred > threshold, 255, 0).astype(np.uint8)
    signal_px = int(np.sum(smoothed == 255))
    print(f"  [smooth] sigma={gaussian_sigma}, threshold={threshold} → "
          f"{signal_px} signal pixels retained")

    return smoothed
