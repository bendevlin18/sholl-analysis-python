# sholl-analysis

A Python package for automated **Sholl analysis** of neuron and microglia microscopy images.

Sholl analysis quantifies the complexity of dendritic/microglial arbors by counting how many times a cell's branches intersect a series of concentric rings drawn around the soma. The result — a Sholl curve — is a standard measure of morphological complexity used widely in neuroscience.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Full Usage Guide](#full-usage-guide)
  - [Image Requirements](#image-requirements)
  - [Running an Analysis](#running-an-analysis)
  - [All Parameters](#all-parameters)
  - [Interactive Controls](#interactive-controls)
  - [Output Files](#output-files)
  - [Processing a Single Image](#processing-a-single-image)
  - [Using the CLI](#using-the-cli)
- [Gaussian Smoothing](#gaussian-smoothing)
- [Working with Test Images](#working-with-test-images)
- [Module Reference](#module-reference)
- [Troubleshooting](#troubleshooting)

---

## Installation

**Requirements:** Python 3.10 or higher. We recommend using a conda environment.

```bash
conda create -n sholl python=3.10
conda activate sholl
```

**Install from source:**

```bash
git clone https://github.com/bendevlin18/sholl-analysis-python.git
cd sholl-analysis-python/sholl_analysis_v6
pip install --upgrade pip
pip install -e .
```

---

## Quick Start

```python
from sholl_analysis import ShollAnalyzer

analyzer = ShollAnalyzer(start_radius=20, step_size=30, end_radius=600)
summary = analyzer.run("/path/to/folder/with/tiffs")
```

This will:
1. Load every `.tiff` file in the folder
2. Show each image and ask you to click the soma centre
3. Display a preview of the skeleton and Sholl rings for confirmation
4. Count intersections, save results, and move to the next image

---

## Full Usage Guide

### Image Requirements

- Images must be **binary segmentations** saved as `.tiff` files
- The package auto-detects pixel value conventions — no manual configuration needed:

| Pixel values | Convention |
|---|---|
| `{0, 255}` | Standard (background=0, signal=255) |
| `{0, 1}` | Signal=1 |
| `{1, 0}` | Inverted (signal=0) |
| `{1, 2}` | Offset encoding |
| `{1, 255}` | Common Fiji output |

- Images can be grayscale or multi-channel — the package collapses to 2-D automatically
- The cell should be reasonably centred in the frame with clear background

---

### Running an Analysis

```python
from sholl_analysis import ShollAnalyzer

analyzer = ShollAnalyzer(
    start_radius=20,    # innermost ring radius in pixels
    step_size=30,       # spacing between rings in pixels
    end_radius=600,     # outermost ring radius in pixels
)

summary = analyzer.run(
    input_dir="/path/to/tiff/images",
    output_dir="/path/to/results",   # optional; defaults to input_dir/sholl_output
)

print(summary)
```

`summary` is a `pandas.DataFrame` with one row per image and one column per radius, plus a `sholl_summary.csv` saved to the output directory.

---

### All Parameters

```python
analyzer = ShollAnalyzer(
    # Ring geometry
    start_radius=20,        # innermost ring (px)
    step_size=30,           # spacing between rings (px)
    end_radius=1024,        # outermost ring (px)

    # Skeleton processing
    min_object_size=10,     # min fragment size to keep (px); removes isolated noise
    dilation_radius=1,      # skeleton dilation before counting; helps catch edge intersections
    merge_dist=10.0,        # max distance (px) to merge nearby duplicate intersection points
    gaussian_sigma=0.0,     # Gaussian smoothing before skeletonizing (0 = off)

    # Display options
    show_sholl_curve=False, # pop up the Sholl curve plot per image (always saved to disk)
    show_results_plot=True, # pop up the annotated skeleton plot per image
    save_figures=True,      # save PNG figures to the output directory
    figure_dpi=150,         # resolution of saved figures
)
```

---

### Interactive Controls

The script is interactive — you will be prompted at two points for each image.

**Step 1 — Click the soma centre:**

A matplotlib window shows the image. Click once on the soma (cell body) centre, then press Enter. If you close the window without clicking, you will be asked whether to skip or quit.

**Step 2 — Preview confirmation:**

A side-by-side window shows the skeleton with Sholl rings overlaid. Check that:
- The skeleton looks correct (thin lines, no large filled blobs)
- The rings are centred on the soma
- The rings span the full extent of the processes

Then respond to the terminal prompt:

```
[Enter/y] continue  |  [s] skip  |  [q] quit  |  [r] redo centre
```

| Key | Action |
|---|---|
| Enter or `y` | Process this image and continue |
| `s` | Skip this image (no output written) |
| `q` | Stop the run; all completed images are saved |
| `r` | Re-do the soma centre click |

**At any time:** Press `Ctrl+C` to interrupt cleanly. All completed images are saved before exiting.

**After each image completes**, a one-line summary is printed:

```
✓  12 endpoints  |  peak 8 intersections @ radius 140px
```

---

### Output Files

All outputs are written to the output directory (default: `<input_dir>/sholl_output`).

| File | Description |
|---|---|
| `intersections/<stem>_raw_intersections.csv` | Per-ring intersection coordinates and endpoint count |
| `skeleton_plots/sholl_<stem>.png` | Annotated skeleton figure with rings, intersections, and endpoints |
| `sholl_curves/sholl_curve_<stem>.png` | Sholl curve plot (intersections vs. radius) |
| `sholl_summary.csv` | One row per image, one column per radius — the main results table |

The `sholl_summary.csv` is always written at the end of a run, even if cancelled early. Re-running on the same folder skips images whose CSV already exists, so you can safely resume a batch.

---

### Processing a Single Image

```python
from sholl_analysis import ShollAnalyzer

analyzer = ShollAnalyzer(start_radius=20, step_size=30, end_radius=600)
counts = analyzer.process_single(
    filepath="/path/to/image.tiff",
    output_dir="/path/to/results",
)
```

---

### Using the CLI

The package installs a `sholl-analysis` command:

```bash
sholl-analysis --input /path/to/tiffs --output /path/to/results
```

**All options:**

```
--input,  -i    Directory containing .tiff images (required)
--output, -o    Output directory (default: <input>/sholl_output)
--start         Start radius in pixels (default: 20)
--step          Step size in pixels (default: 30)
--end           End radius in pixels (default: 1024)
--min-size      Min skeleton fragment size in pixels (default: 10)
--dilation      Skeleton dilation radius (default: 1)
--merge-dist    Max distance to merge nearby intersections (default: 10.0)
--sigma         Gaussian smoothing sigma (default: 0 = off)
--show-curve    Show the Sholl curve plot for each image
--no-results    Hide the annotated skeleton plot
--no-save       Do not save PNG figures
--dpi           Figure resolution (default: 150)
```

**Example with options:**

```bash
sholl-analysis --input ./tiffs --start 20 --step 20 --end 400 --sigma 1.5 --show-curve
```

---

## Gaussian Smoothing

Some microglia segmentations are "chunky" — thick, blocky filled regions rather than thin lines. Skeletonizing these directly can produce noisy, spiky results. Gaussian smoothing softens the edges before skeletonizing, producing a cleaner skeleton.

```python
analyzer = ShollAnalyzer(gaussian_sigma=1.5)
```

**Choosing sigma:**

| Sigma | Effect |
|---|---|
| `0` | Disabled — no smoothing (default) |
| `0.5 – 1.0` | Mild — removes minor pixelation |
| `1.5 – 2.5` | Moderate — good for chunky segmentations |
| `3.0+` | Heavy — may merge nearby branches; use with care |

You can also apply smoothing manually:

```python
from sholl_analysis import load_and_preprocess, smooth_binary, skeletonize_image

img, _ = load_and_preprocess("cell.tiff")
img_smooth = smooth_binary(img, gaussian_sigma=1.5)
skeleton = skeletonize_image(img_smooth)
```

---

## Working with Test Images

### Generate a batch of TIFFs

```python
from sholl_analysis.test_images import generate_test_dataset

generate_test_dataset(
    output_dir="/path/to/save",
    n_images=5,
    size=512,
    n_primary=8,
    pixel_values=(0, 255),
)
```

### Generate a single image array

```python
from sholl_analysis.test_images import generate_microglia

img = generate_microglia(size=512, n_primary=8, seed=42)
```

Each unique `seed` produces a different cell morphology. The same seed always gives the same image.

### Test all pixel-value encodings

```python
from sholl_analysis.test_images import generate_encoding_test_set
from sholl_analysis import load_and_preprocess
import numpy as np

paths = generate_encoding_test_set("/path/to/output")

for label, path in paths.items():
    img, _ = load_and_preprocess(path)
    print(f"{label:12s}  {np.unique(img).tolist()}")  # should always be [0, 255]
```

Default encodings: `standard (0,255)`, `binary_01 (0,1)`, `inverted (1,0)`, `offset (1,2)`, `fiji (1,255)`.

---

## Module Reference

| Module | Key contents |
|---|---|
| `sholl_analysis.analyzer` | `ShollAnalyzer` — main pipeline class |
| `sholl_analysis.image_processing` | `load_and_preprocess`, `detect_and_normalize`, `smooth_binary`, `skeletonize_image`, `dilate_skeleton`, `find_endpoints` |
| `sholl_analysis.geometry` | `make_circles`, `calc_intersection`, `clean_intersections`, `x_y_separate`, `dist_formula` |
| `sholl_analysis.visualization` | `plot_preview`, `plot_results`, `plot_sholl_curve` |
| `sholl_analysis.io` | `save_intersections`, `load_intersections`, `ensure_output_dir` |
| `sholl_analysis.test_images` | `generate_microglia`, `generate_test_dataset`, `generate_encoding_test_set`, `KNOWN_ENCODINGS` |
| `sholl_analysis.cli` | CLI entry point (`sholl-analysis` command) |

---

## Troubleshooting

**Skeleton looks like filled polygons instead of thin lines**

The image is being read as a 3-channel array. This is handled automatically — if you still see this, check that your TIFF is a genuine binary segmentation and not a colour image.

**"Expected a binary image" ValueError**

The image has more than 2 unique pixel values. Make sure the image has been thresholded before running.

**Skeleton appears inverted**

The auto-detection uses the minority class as signal. If your cell fills more than 50% of the frame this can invert. Manually normalise before running:

```python
import numpy as np
img_fixed = np.where(img == YOUR_SIGNAL_VALUE, 255, 0).astype('uint8')
```

**Rings don't cover the full cell**

Increase `end_radius` to roughly the distance from soma to the furthest branch tip in pixels.

**Too many spurious intersections**

Try increasing `merge_dist` or enabling `gaussian_sigma` smoothing.

**pip install fails with "requires Python >=3.10"**

```bash
conda create -n sholl python=3.10
conda activate sholl
pip install -e .
```
