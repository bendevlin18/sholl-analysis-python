"""
geometry.py
-----------
Functions for creating Sholl rings and computing their intersections
with a neuron/microglia skeleton.
"""

import numpy as np
from skimage import draw


def make_circles(
    center: tuple[float, float],
    radii: np.ndarray,
    image_shape: tuple[int, int],
) -> list[np.ndarray]:
    """
    Generate binary circle-perimeter arrays for each radius.

    Parameters
    ----------
    center : tuple of float
        (x, y) coordinates of the soma / analysis centre as returned by
        matplotlib ``ginput`` — i.e. (col, row).
    radii : array-like
        Sequence of radii (in pixels) at which to draw rings.
    image_shape : tuple of int
        (rows, cols) shape of the target image.

    Returns
    -------
    circles : list of np.ndarray
        One array per radius; ring pixels are set to 255, background to 1.
    """
    circles = []
    for rad in radii:
        arr = np.ones(image_shape)
        rr, cc = draw.circle_perimeter(
            int(center[1]),  # row  (y)
            int(center[0]),  # col  (x)
            radius=int(rad),
            shape=arr.shape,
        )
        arr[rr, cc] = 255
        circles.append(arr)
    return circles


def calc_intersection(
    circ_arr: np.ndarray,
    skeleton_arr: np.ndarray,
) -> np.ndarray:
    """
    Find pixels where a Sholl ring overlaps with the skeleton.

    Parameters
    ----------
    circ_arr : np.ndarray
        Circle-perimeter array (ring pixels = 255, background = 1).
    skeleton_arr : np.ndarray
        Skeleton array (foreground = 255).

    Returns
    -------
    intersects : np.ndarray
        Flat array of interleaved (row, col) pairs for every intersection.
    """
    rows, cols = np.where((circ_arr == 255) & (skeleton_arr == 255))
    if len(rows) == 0:
        return np.array([])
    return np.column_stack([rows, cols]).flatten()


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def x_y_separate(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a flat array of interleaved (row, col) pairs into two arrays.

    Parameters
    ----------
    arr : np.ndarray
        Flat array ``[r0, c0, r1, c1, ...]``.

    Returns
    -------
    x : np.ndarray   (rows)
    y : np.ndarray   (cols)
    """
    return arr[::2], arr[1::2]


def dist_formula(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def clean_intersections(raw_z: list, radii: np.ndarray, merge_dist: float = 10.0):
    """
    Deduplicate nearby intersection points for each Sholl ring.

    Points within *merge_dist* pixels of each other are collapsed to the
    same coordinate before duplicates are dropped.

    Parameters
    ----------
    raw_z : list of np.ndarray
        Per-ring flat intersection arrays from :func:`calc_intersection`.
    radii : np.ndarray
        Radii corresponding to each entry in *raw_z*.
    merge_dist : float, optional
        Distance threshold below which two points are considered the same
        intersection (default 10 pixels).

    Returns
    -------
    intersections_to_plot : pd.DataFrame
        DataFrame with columns [0 (row), 1 (col), 'ring'] indexed by ring radius.
    """
    import pandas as pd

    frames = []

    for idx, vals_raw in zip(radii, raw_z):
        vals = vals_raw if not hasattr(vals_raw, "dropna") else vals_raw.dropna().values
        if len(vals) == 0:
            continue

        x, y = x_y_separate(vals)

        # Snap each coordinate to the nearest merge_dist grid cell, then
        # drop duplicates.  This is O(n) and replaces the previous O(n²)
        # double-iterrows loop that compared every pair of points.
        snapped_x = np.round(x / merge_dist) * merge_dist
        snapped_y = np.round(y / merge_dist) * merge_dist

        frame = pd.DataFrame({0: snapped_x, 1: snapped_y}).drop_duplicates().copy()
        frame["ring"] = idx
        frame.set_index("ring", inplace=True)
        frames.append(frame)

    return pd.concat(frames) if frames else pd.DataFrame()
