"""
test_images.py
--------------
Utilities for generating synthetic microglia-like binary TIFF images.

Useful for testing and validating the Sholl analysis pipeline without
needing real microscopy data.

Examples
--------
Generate a set of test images in a directory::

    from sholl_analysis.test_images import generate_test_dataset
    generate_test_dataset("/tmp/test_tiffs", n_images=3)

Generate a single image array (no file I/O)::

    from sholl_analysis.test_images import generate_microglia
    img = generate_microglia(size=512, n_primary=8, seed=42)
"""

import os
import numpy as np
import cv2


def _draw_branch(
    img: np.ndarray,
    start: tuple,
    angle: float,
    length: float,
    width: int = 2,
    n_segments: int = 8,
    spread: float = 0.3,
    rng: np.random.Generator = None,
) -> None:
    """
    Recursively draw a single branching process onto *img* in-place.

    Parameters
    ----------
    img : np.ndarray
        Target image array (modified in-place).
    start : tuple of int
        (x, y) starting pixel coordinate.
    angle : float
        Initial direction in radians.
    length : float
        Total length of this branch segment in pixels.
    width : int, optional
        Line width in pixels (default 2; tapers with recursion depth).
    n_segments : int, optional
        Number of sub-segments to draw with small random jitter (default 8).
    spread : float, optional
        Max angular jitter per segment in radians (default 0.3).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()

    if length < 10:
        return

    x, y = start
    for _ in range(n_segments):
        cur_angle = angle + rng.uniform(-spread, spread)
        end_x = int(x + np.cos(cur_angle) * (length / n_segments))
        end_y = int(y + np.sin(cur_angle) * (length / n_segments))

        # Clamp to image bounds
        end_x = np.clip(end_x, 0, img.shape[1] - 1)
        end_y = np.clip(end_y, 0, img.shape[0] - 1)

        cv2.line(img, (x, y), (end_x, end_y), 255, width)
        x, y = end_x, end_y

    # Randomly sprout sub-branches
    if length > 30 and rng.random() > 0.3:
        branch_angle = angle + rng.uniform(0.4, 0.9) * rng.choice([-1, 1])
        _draw_branch(img, (x, y), branch_angle, length * 0.6,
                     width=max(1, width - 1), rng=rng)
    if length > 50 and rng.random() > 0.5:
        branch_angle = angle + rng.uniform(0.4, 0.9) * rng.choice([-1, 1])
        _draw_branch(img, (x, y), branch_angle, length * 0.5,
                     width=max(1, width - 1), rng=rng)


def generate_microglia(
    size: int = 512,
    n_primary: int = 8,
    seed: int = 42,
    soma_radius: int = 10,
    branch_length_range: tuple = (100, 180),
    pixel_values: tuple = (0, 255),
) -> np.ndarray:
    """
    Generate a single synthetic microglia-like binary image.

    A filled circle represents the soma at the image centre; branching
    processes radiate outward at evenly spaced angles with random jitter.

    Parameters
    ----------
    size : int, optional
        Width and height of the square output image in pixels (default 512).
    n_primary : int, optional
        Number of primary processes (default 8).
    seed : int, optional
        Random seed for reproducibility (default 42).
    soma_radius : int, optional
        Radius of the soma circle in pixels (default 10).
    branch_length_range : tuple of int, optional
        (min, max) length of primary branches in pixels (default (100, 180)).
    pixel_values : tuple of int, optional
        (background, signal) pixel values in the output array. Use this to
        simulate different dataset conventions:

        - ``(0, 255)`` — standard (default)
        - ``(0, 1)``   — signal = 1
        - ``(1, 0)``   — inverted, signal = 0
        - ``(1, 2)``   — offset encoding

    Returns
    -------
    img : np.ndarray
        uint8 image array of shape ``(size, size)``.

    Examples
    --------
    >>> img = generate_microglia(size=256, n_primary=6, seed=7)
    >>> img.shape
    (256, 256)
    >>> set(np.unique(img)) == {0, 255}
    True

    Simulate a {1, 2} encoded dataset::

        img = generate_microglia(pixel_values=(1, 2))
    """
    rng = np.random.default_rng(seed)
    bg, signal = pixel_values

    # Start with background
    img = np.full((size, size), bg, dtype=np.uint8)

    # Use an intermediate {0, 255} canvas, then remap at the end
    canvas = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2

    # Soma
    cv2.circle(canvas, (cx, cy), soma_radius, 255, -1)

    # Primary processes
    for i in range(n_primary):
        base_angle = (2 * np.pi / n_primary) * i
        angle = base_angle + rng.uniform(-0.2, 0.2)
        length = rng.integers(*branch_length_range)
        _draw_branch(canvas, (cx, cy), angle, length, width=2, rng=rng)

    # Remap 255 → signal value, 0 → background value
    img = np.where(canvas == 255, signal, bg).astype(np.uint8)
    return img


def generate_test_dataset(
    output_dir: str,
    n_images: int = 3,
    size: int = 512,
    n_primary: int = 8,
    pixel_values: tuple = (0, 255),
    seeds: list = None,
) -> list:
    """
    Generate a set of synthetic microglia TIFFs and save them to disk.

    Parameters
    ----------
    output_dir : str
        Directory in which to save the images (created if it doesn't exist).
    n_images : int, optional
        Number of images to generate (default 3).
    size : int, optional
        Image size in pixels (default 512).
    n_primary : int, optional
        Number of primary processes per cell (default 8).
    pixel_values : tuple of int, optional
        (background, signal) encoding — see :func:`generate_microglia`.
        Defaults to ``(0, 255)``.
    seeds : list of int, optional
        Random seeds, one per image. Defaults to ``[42, 99, 7, ...]``.

    Returns
    -------
    filepaths : list of str
        Paths of the saved TIFF files.

    Examples
    --------
    Standard dataset::

        from sholl_analysis.test_images import generate_test_dataset
        paths = generate_test_dataset("/tmp/test_tiffs", n_images=5)

    Simulate a {1, 2} encoded dataset::

        paths = generate_test_dataset("/tmp/test_tiffs", pixel_values=(1, 2))
    """
    os.makedirs(output_dir, exist_ok=True)

    default_seeds = [42, 99, 7, 13, 21, 55, 77, 3, 17, 101]
    if seeds is None:
        seeds = default_seeds[:n_images]
    if len(seeds) < n_images:
        seeds = seeds + list(range(200, 200 + n_images - len(seeds)))

    bg, signal = pixel_values
    filepaths = []

    for i, seed in enumerate(seeds[:n_images]):
        img = generate_microglia(
            size=size,
            n_primary=n_primary,
            seed=seed,
            pixel_values=pixel_values,
        )
        fname = f"test_microglia_{i + 1:02d}.tiff"
        fpath = os.path.join(output_dir, fname)
        cv2.imwrite(fpath, img)
        filepaths.append(fpath)
        print(f"  Saved {fname}  (pixel values: {np.unique(img).tolist()})")

    print(f"\nGenerated {n_images} image(s) in: {output_dir}")
    print(f"Pixel encoding — background: {bg}, signal: {signal}")
    return filepaths


# All encodings we know about, with human-readable labels
KNOWN_ENCODINGS = {
    "standard":  (0, 255),  # background=0, signal=255
    "binary_01": (0, 1),    # background=0, signal=1
    "inverted":  (1, 0),    # background=1, signal=0
    "offset":    (1, 2),    # background=1, signal=2
    "fiji":      (1, 255),  # background=1, signal=255 (common Fiji output)
}


def generate_encoding_test_set(
    output_dir: str,
    size: int = 512,
    n_primary: int = 8,
    seed: int = 42,
    encodings: dict = None,
) -> dict:
    """
    Generate one TIFF per pixel-value encoding so you can verify that
    :func:`~sholl_analysis.image_processing.detect_and_normalize` handles
    each format correctly.

    By default the following encodings are generated:

    ============  ==========================
    Name          (background, signal)
    ============  ==========================
    standard      (0, 255)
    binary_01     (0, 1)
    inverted      (1, 0)
    offset        (1, 2)
    fiji          (1, 255)
    ============  ==========================

    Parameters
    ----------
    output_dir : str
        Directory in which to save the images (created if it doesn't exist).
    size : int, optional
        Image size in pixels (default 512).
    n_primary : int, optional
        Number of primary processes (default 8).
    seed : int, optional
        Random seed — same seed across all encodings so every image shows
        the identical cell, making visual comparison easy (default 42).
    encodings : dict or None, optional
        Custom ``{label: (background, signal)}`` dict. If provided, replaces
        the default set entirely. Example::

            encodings={"my_format": (3, 7)}

    Returns
    -------
    results : dict
        ``{label: filepath}`` mapping for every image that was saved.

    Examples
    --------
    Test all default encodings::

        from sholl_analysis.test_images import generate_encoding_test_set
        paths = generate_encoding_test_set("/tmp/encoding_test")

    Test a custom encoding only::

        paths = generate_encoding_test_set(
            "/tmp/encoding_test",
            encodings={"weird": (3, 7)},
        )

    Verify normalization works on every file::

        from sholl_analysis.image_processing import load_and_preprocess
        import numpy as np

        for label, path in paths.items():
            img, _ = load_and_preprocess(path)
            assert set(np.unique(img)) == {0, 255}, f"Failed for {label}"
            print(f"{label:12s} ✓")
    """
    os.makedirs(output_dir, exist_ok=True)

    if encodings is None:
        encodings = KNOWN_ENCODINGS

    results = {}
    print(f"Generating {len(encodings)} encoding variant(s) — same cell, different pixel values:\n")

    for label, pixel_values in encodings.items():
        bg, signal = pixel_values
        img = generate_microglia(
            size=size,
            n_primary=n_primary,
            seed=seed,
            pixel_values=pixel_values,
        )
        fname = f"test_encoding_{label}.tiff"
        fpath = os.path.join(output_dir, fname)
        cv2.imwrite(fpath, img)
        results[label] = fpath
        print(f"  {label:12s}  bg={bg}, signal={signal}  →  {fname}")

    print(f"\nSaved {len(results)} file(s) to: {output_dir}")
    print("\nTo verify normalization works on all of them:")
    print("  from sholl_analysis.image_processing import load_and_preprocess")
    print("  from sholl_analysis.test_images import generate_encoding_test_set")
    print("  import numpy as np")
    print("  paths = generate_encoding_test_set('/tmp/encoding_test')")
    print("  for label, path in paths.items():")
    print("      img, _ = load_and_preprocess(path)")
    print("      print(label, np.unique(img))  # should always be [0, 255]")

    return results
