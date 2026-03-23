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
    intersects = []
    for i in range(circ_arr.shape[0]):
        for j in range(circ_arr.shape[1]):
            if circ_arr[i][j] == skeleton_arr[i][j]:
                intersects = np.append(intersects, [i, j])
    return np.asarray(intersects)


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
    xs, ys = [], []
    for row, col in zip(arr[::2], arr[1::2]):
        xs = np.append(xs, row)
        ys = np.append(ys, col)
    return np.asarray(xs), np.asarray(ys)


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

    intersections_to_plot = pd.DataFrame()

    for idx, vals_raw in zip(radii, raw_z):
        vals = vals_raw if not hasattr(vals_raw, "dropna") else vals_raw.dropna().values
        if len(vals) == 0:
            continue

        x, y = x_y_separate(vals)
        cleaned = pd.DataFrame({0: x, 1: y})

        # Merge nearby points by snapping them to the first encountered neighbour
        for cur_idx, cur_row in cleaned.iterrows():
            for new_idx, new_row in cleaned.iterrows():
                if dist_formula(cur_row[0], cur_row[1], new_row[0], new_row[1]) < merge_dist:
                    cleaned.loc[new_idx] = cleaned.loc[cur_idx]

        final = cleaned.drop_duplicates().copy()
        final["ring"] = idx
        final.set_index("ring", inplace=True)
        intersections_to_plot = pd.concat([intersections_to_plot, final])

    return intersections_to_plot
