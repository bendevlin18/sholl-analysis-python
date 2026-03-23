"""
tests/test_geometry.py
----------------------
Unit tests for the geometry module.
"""

import numpy as np
import pytest
from sholl_analysis.geometry import (
    dist_formula,
    x_y_separate,
    make_circles,
    clean_intersections,
)


def test_dist_formula_zero():
    assert dist_formula(0, 0, 0, 0) == 0.0


def test_dist_formula_pythagorean():
    assert np.isclose(dist_formula(0, 0, 3, 4), 5.0)


def test_x_y_separate_basic():
    arr = np.array([10, 20, 30, 40])
    x, y = x_y_separate(arr)
    assert list(x) == [10, 30]
    assert list(y) == [20, 40]


def test_x_y_separate_empty():
    x, y = x_y_separate(np.array([]))
    assert len(x) == 0 and len(y) == 0


def test_make_circles_count():
    center = (50, 50)
    radii = np.array([10, 20, 30])
    circles = make_circles(center, radii, (100, 100))
    assert len(circles) == 3


def test_make_circles_shape():
    center = (50, 50)
    radii = np.array([10])
    circles = make_circles(center, radii, (100, 100))
    assert circles[0].shape == (100, 100)


def test_make_circles_ring_pixels():
    """Ring pixels should be 255, background should be 1."""
    center = (50, 50)
    radii = np.array([20])
    circles = make_circles(center, radii, (100, 100))
    arr = circles[0]
    assert 255 in arr
    assert 1 in arr


# ---------------------------------------------------------------------------
# Tests for detect_and_normalize
# ---------------------------------------------------------------------------

from sholl_analysis.image_processing import detect_and_normalize
from sholl_analysis.test_images import generate_microglia, generate_test_dataset
import tempfile, os

def _make_binary(bg, signal, size=50, signal_frac=0.1):
    """Helper: small array with given bg/signal values."""
    arr = np.full((size, size), bg, dtype=np.uint8)
    n = int(size * size * signal_frac)
    idx = np.random.default_rng(0).choice(size * size, n, replace=False)
    arr.flat[idx] = signal
    return arr


def test_normalize_passthrough():
    arr = _make_binary(0, 255)
    out = detect_and_normalize(arr)
    assert set(np.unique(out)) == {0, 255}


def test_normalize_0_1():
    arr = _make_binary(0, 1)
    out = detect_and_normalize(arr)
    assert set(np.unique(out)) == {0, 255}


def test_normalize_1_0():
    arr = _make_binary(1, 0)
    out = detect_and_normalize(arr)
    assert set(np.unique(out)) == {0, 255}


def test_normalize_1_2():
    arr = _make_binary(1, 2)
    out = detect_and_normalize(arr)
    assert set(np.unique(out)) == {0, 255}


def test_normalize_1_255():
    arr = _make_binary(1, 255)
    out = detect_and_normalize(arr)
    assert set(np.unique(out)) == {0, 255}


def test_normalize_raises_on_non_binary():
    arr = np.array([0, 1, 2, 3], dtype=np.uint8)
    with pytest.raises(ValueError):
        detect_and_normalize(arr)


# ---------------------------------------------------------------------------
# Tests for generate_microglia / generate_test_dataset
# ---------------------------------------------------------------------------

def test_generate_microglia_shape():
    img = generate_microglia(size=128, seed=1)
    assert img.shape == (128, 128)


def test_generate_microglia_binary():
    img = generate_microglia(size=128, seed=1)
    assert set(np.unique(img)).issubset({0, 255})


def test_generate_microglia_pixel_values():
    img = generate_microglia(size=128, seed=1, pixel_values=(1, 2))
    assert set(np.unique(img)).issubset({1, 2})


def test_generate_microglia_reproducible():
    img1 = generate_microglia(size=128, seed=99)
    img2 = generate_microglia(size=128, seed=99)
    assert np.array_equal(img1, img2)


def test_generate_test_dataset_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_test_dataset(tmpdir, n_images=2, size=64)
        assert len(paths) == 2
        for p in paths:
            assert os.path.exists(p)
            assert p.endswith(".tiff")


# ---------------------------------------------------------------------------
# Tests for supported_formats
# ---------------------------------------------------------------------------

from sholl_analysis.supported_formats import is_supported, get_stem, SUPPORTED_EXTENSIONS

def test_is_supported_tiff():
    assert is_supported("cell.tiff")

def test_is_supported_tif():
    assert is_supported("cell.tif")

def test_is_supported_png():
    assert is_supported("cell.png")

def test_is_supported_jpg():
    assert is_supported("cell.jpg")

def test_is_supported_jpeg():
    assert is_supported("cell.jpeg")

def test_is_supported_case_insensitive():
    assert is_supported("cell.TIFF")
    assert is_supported("cell.PNG")
    assert is_supported("cell.JPG")

def test_is_supported_rejects_other():
    assert not is_supported("cell.bmp")
    assert not is_supported("cell.txt")
    assert not is_supported("cell")

def test_get_stem_tiff():
    assert get_stem("cell_01.tiff") == "cell_01"

def test_get_stem_tif():
    assert get_stem("cell_01.tif") == "cell_01"

def test_get_stem_png():
    assert get_stem("cell_01.png") == "cell_01"

def test_get_stem_jpg():
    assert get_stem("cell_01.jpg") == "cell_01"

def test_get_stem_uppercase():
    assert get_stem("cell_01.TIFF") == "cell_01"

def test_get_stem_dotted_name():
    # Names with dots in them should only strip the extension
    assert get_stem("cell.01.png") == "cell.01"
