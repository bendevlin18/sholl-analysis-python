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
- [Physical Units (Pixel Size)](#physical-units-pixel-size)
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
cd sholl-analysis-python/sholl_analysis
pip install --upgrade pip
pip install -e .
```

---

## Quick Start

#### NOTE: This package does not inherently work with Jupyter notebooks. Because of the interactivity of the matplotlib objects it creates, Jupyter Notebooks can cause issues based on Matplotlib backends. Thus, the easiest way to run it is via the command line. I generally recommend folks just type 'python' while in the active sholl conda environment to enter interactive python in the terminal. From there, you can just copy and past the three lines below, replacing the path string with the actual path of your tiff files (note it also works with .tif, .png, and .jpgs).

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

    # Physical units
    pixel_size=1.0,         # µm per pixel — converts all radii and stats to µm (default 1.0 = stay in pixels)
    use_micron=False,       # if True, start/step/end are given in µm instead of pixels (requires pixel_size)

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
| `sholl_summary.csv` | One row per image — per-radius intersection counts plus derived statistics (critical radius, max intersections, mean intersections, AUC, Schoenen ramification index) |

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
--pixel-size    Pixel size in µm/px — converts radii and stats to µm (default: 1.0 = pixels)
--use-micron    Treat --start/--step/--end as µm instead of pixels (requires --pixel-size)
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

## Physical Units (Pixel Size)

By default, all radii are reported in **pixels**. If you know the pixel size of your microscope images (µm/px), you can work entirely in µm — both for specifying ring parameters and for all outputs.

### Option 1 — Output in µm, ring parameters still in pixels

Pass `pixel_size` alone. `start_radius`, `step_size`, and `end_radius` stay in pixels; all outputs are scaled to µm.

```python
analyzer = ShollAnalyzer(
    start_radius=20,
    step_size=30,
    end_radius=600,
    pixel_size=0.065,   # e.g. 0.065 µm/px for a 63× oil objective
)
```

### Option 2 — Everything in µm (recommended)

Pass both `pixel_size` and `use_micron=True`. Now `start_radius`, `step_size`, and `end_radius` are given in µm, and the package converts them to pixels internally for geometry. All outputs are in µm.

```python
analyzer = ShollAnalyzer(
    start_radius=2.0,   # µm
    step_size=2.0,      # µm
    end_radius=50.0,    # µm
    pixel_size=0.065,
    use_micron=True,
)
```

Or via the CLI:

```bash
sholl-analysis --input ./tiffs --pixel-size 0.065 --use-micron --start 2 --step 2 --end 50
```

> **Note:** `use_micron=True` requires `pixel_size` to be set. The package will raise an error immediately if you forget.

### Considerations when using `use_micron=True`

**Only ring geometry is in µm — processing parameters stay in pixels.**

`start_radius`, `step_size`, and `end_radius` are converted to pixels internally, but the following parameters are always in pixels regardless of `use_micron`:

| Parameter | Unit | Typical value |
|---|---|---|
| `merge_dist` | pixels | 10 |
| `min_object_size` | pixels | 10 |
| `dilation_radius` | pixels | 1 |

If you're working at a known pixel size and want to think about these in µm, just divide manually: e.g. for 0.5 µm merge distance at 0.065 µm/px → `merge_dist = 0.5 / 0.065 ≈ 8`.

---

**Make sure `end_radius` fits inside the image.**

With `use_micron=True`, the end radius in pixels is `end_radius / pixel_size`. This must be smaller than the shortest image dimension. For example, a 512 × 512 image at 0.065 µm/px has a maximum useful radius of `512 × 0.065 ≈ 33 µm`. Setting `end_radius=50` would silently produce no intersections beyond the image edge — rings are clipped to the image boundary.

A quick way to check in Python:
```python
max_radius_um = min(image_height, image_width) * pixel_size / 2
```

---

**`use_micron=True` is especially useful for cross-dataset consistency.**

If you image the same cell type across multiple sessions or microscopes with different pixel sizes, specifying ring parameters in µm means you're always sampling at the same biological scale — no manual conversion needed between datasets. With pixel-based parameters, a `step_size=30` means something different on a 20× vs. 63× objective.

---

**What changes when `pixel_size` is set:**

- Column headers in `sholl_summary.csv` are in µm (e.g. `1.3`, `3.25`, …) instead of pixels
- `critical_radius` in the summary is in µm
- `auc` is in intersections × µm instead of intersections × px
- The Sholl curve x-axis is labeled `Radius (µm)`
- The per-image completion message shows µm: `peak 8 intersections @ radius 9.1µm`

**Finding your pixel size:**

Your pixel size is determined by the objective and camera, and is usually recorded in the microscope metadata. Common sources:

- **Fiji / ImageJ**: `Image → Properties` shows pixel width in µm
- **TIFF metadata**: open in Fiji and check `Image → Show Info`
- **Microscope software**: check acquisition settings or the image properties panel

A typical value for a confocal with a 63× oil objective is ~0.065–0.13 µm/px depending on zoom and camera pixel pitch. Use the value from your actual acquisition.

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
