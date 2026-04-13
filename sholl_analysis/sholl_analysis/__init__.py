"""
sholl_analysis
==============
A Python package for Sholl analysis of microscopy images.

Performs intersection counting of neuronal/microglial skeletons
across concentric rings to quantify dendritic arborization.

Usage
-----
    from sholl_analysis import ShollAnalyzer

    analyzer = ShollAnalyzer(start_radius=20, step_size=30, end_radius=1024)
    analyzer.run(input_dir="/path/to/tiffs", output_dir="/path/to/output")

Generating test images
----------------------
    from sholl_analysis.test_images import generate_test_dataset
    generate_test_dataset("/tmp/test_tiffs", n_images=3)
"""

from .analyzer import ShollAnalyzer
from .image_processing import (
    smooth_binary,
    load_and_preprocess,
    detect_and_normalize,
    skeletonize_image,
    dilate_skeleton,
    find_endpoints,
)
from .geometry import make_circles, calc_intersection
from .io import save_intersections, load_intersections
from .supported_formats import is_supported, get_stem, SUPPORTED_EXTENSIONS
from .test_images import generate_microglia, generate_test_dataset, generate_encoding_test_set, KNOWN_ENCODINGS

__version__ = "0.2.0"
__all__ = [
    "ShollAnalyzer",
    "load_and_preprocess",
    "detect_and_normalize",
    "skeletonize_image",
    "dilate_skeleton",
    "find_endpoints",
    "smooth_binary",
    "make_circles",
    "calc_intersection",
    "save_intersections",
    "load_intersections",
    "generate_microglia",
    "generate_test_dataset",
    "generate_encoding_test_set",
    "KNOWN_ENCODINGS",
]
