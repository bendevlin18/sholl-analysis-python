# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`sholl-analysis-python` is a Python package for automated Sholl analysis of neuron and microglia microscopy images. Sholl analysis quantifies dendritic/microglial arborization complexity by counting how many times a cell's branches intersect concentric rings drawn around the soma (cell body), producing a "Sholl curve" (intersections vs. radius).

## Installation & Setup

The package source lives in the `sholl_analysis/` subdirectory. Install from there:

```bash
cd sholl_analysis
pip install -e .
```

Python >= 3.10 is required (uses `|` union type hint syntax).

## Common Commands

```bash
# Run all tests
pytest sholl_analysis/tests/

# Run a single test file
pytest sholl_analysis/tests/test_geometry.py -v

# Run the CLI
sholl-analysis --input /path/to/images --output /path/to/results

# Key CLI flags
sholl-analysis --input ./images --start 5 --step 5 --end 100 --sigma 1.0 --show-curve
```

## Architecture

The pipeline follows this flow:

```
Image Load → Normalize → [Optional Smoothing] → Skeletonize → Interactive Centre Selection
  → Interactive Preview → Dilate → Count Ring Intersections → Find Endpoints → Save CSVs & Plots
```

### Module Responsibilities

- **`analyzer.py`** — `ShollAnalyzer` class; the pipeline orchestrator. `run()` handles batch processing; `process_single()` handles one image. This is the primary entry point for programmatic use.
- **`image_processing.py`** — Loads images, normalizes to `uint8 {0, 255}` with minority-class heuristic (smaller class = signal), skeletonizes, dilates, and finds endpoints.
- **`geometry.py`** — Creates concentric ring arrays (`make_circles`), computes pixel-level ring-skeleton overlaps (`calc_intersection`), and deduplicates nearby intersection points (`clean_intersections`).
- **`visualization.py`** — All matplotlib plotting (preview, results overlay, Sholl curve). Designed for interactive use via figure windows.
- **`io.py`** — File I/O: creates output directory tree, saves/loads intersection CSVs, checks for existing results (enables resumable batch runs).
- **`cli.py`** — `argparse`-based CLI; constructs `ShollAnalyzer` and calls `run()`.
- **`test_images.py`** — Generates synthetic branched morphology images for testing without real microscopy data.

### Output Directory Structure

```
<output>/
  sholl_summary.csv                         ← Master results (one row per image)
  intersections/<stem>_raw_intersections.csv
  skeleton_plots/sholl_<stem>.png
  sholl_curves/sholl_curve_<stem>.png
```

### Key Design Decisions

- **Interactive workflow**: The tool uses matplotlib `ginput` for soma-centre selection and keyboard prompts (`Enter`, `s`, `q`, `r`) for preview confirmation. This is intentional — the package is designed for lab use with visual QC at each step.
- **Resumable batch processing**: `io.csv_exists()` checks for prior output so re-running a batch skips already-processed images.
- **Pixel convention auto-detection**: `detect_and_normalize()` always treats the minority pixel class as signal, handling all common binary segmentation encoding styles automatically.
- **Intersection counting is pixel-based**: Ring-skeleton intersections are found by comparing pixel arrays, not by geometric calculation.

### Version Note

`pyproject.toml` has `version = "0.1.0"` but `__init__.py` exports `__version__ = "0.2.0"`. The `__init__.py` version is the authoritative one.
