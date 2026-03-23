"""
io.py
-----
Helper functions for saving and loading Sholl analysis results.

Output directory structure
--------------------------
sholl_output/
├── sholl_summary.csv          ← one row per image, all radii
├── intersections/             ← per-image raw intersection CSVs
│   └── <stem>_raw_intersections.csv
├── skeleton_plots/            ← annotated skeleton PNGs
│   └── sholl_<stem>.png
└── sholl_curves/              ← Sholl curve PNGs
    └── sholl_curve_<stem>.png
"""

import os
import pandas as pd


# Subfolder names — change here if you ever want to rename them
SUBDIR_CSVS   = "intersections"
SUBDIR_SKEL   = "skeleton_plots"
SUBDIR_CURVES = "sholl_curves"


def ensure_output_dirs(base_dir: str, subdir: str = "sholl_output") -> dict:
    """
    Create the full output directory tree and return paths for each subfolder.

    Structure created::

        <base_dir>/<subdir>/
            intersections/
            skeleton_plots/
            sholl_curves/

    Parameters
    ----------
    base_dir : str
        Parent directory (typically the input image folder).
    subdir : str, optional
        Top-level output folder name (default ``"sholl_output"``).

    Returns
    -------
    dirs : dict
        Dictionary with keys ``"root"``, ``"csvs"``, ``"skeletons"``,
        ``"curves"`` pointing to the respective absolute paths.
    """
    root = os.path.join(base_dir, subdir)
    dirs = {
        "root":      root,
        "csvs":      os.path.join(root, SUBDIR_CSVS),
        "skeletons": os.path.join(root, SUBDIR_SKEL),
        "curves":    os.path.join(root, SUBDIR_CURVES),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def ensure_output_dir(base_dir: str, subdir: str = "sholl_output") -> str:
    """
    Compatibility wrapper — creates the full tree and returns the root path.

    Use :func:`ensure_output_dirs` directly when you need subfolder paths.
    """
    return ensure_output_dirs(base_dir, subdir)["root"]


def save_intersections(df: pd.DataFrame, csvs_dir: str, filename_stem: str) -> str:
    """
    Save a cleaned-intersections DataFrame to the intersections subfolder.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by :func:`geometry.clean_intersections`.
    csvs_dir : str
        Path to the ``intersections/`` subfolder
        (i.e. ``dirs["csvs"]`` from :func:`ensure_output_dirs`).
    filename_stem : str
        Base name for the file (without extension), e.g. ``"cell_01"``.

    Returns
    -------
    out_path : str
        Full path of the written CSV file.
    """
    out_path = os.path.join(csvs_dir, f"{filename_stem}_raw_intersections.csv")
    df.to_csv(out_path)
    return out_path


def load_intersections(filepath: str) -> pd.DataFrame:
    """
    Load a previously saved intersections CSV back into a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV produced by :func:`save_intersections`.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(filepath, index_col="ring")


def csv_exists(output_root: str, filename_stem: str) -> bool:
    """
    Check whether a raw-intersections CSV has already been written for *stem*.

    Used by the analyzer to skip already-processed images.

    Parameters
    ----------
    output_root : str
        Root output directory (not the csvs subfolder).
    filename_stem : str
        Image stem to check.

    Returns
    -------
    bool
    """
    path = os.path.join(output_root, SUBDIR_CSVS,
                        f"{filename_stem}_raw_intersections.csv")
    return os.path.exists(path)
