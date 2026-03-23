"""
visualization.py
----------------
Plotting helpers for Sholl analysis results.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter


def _make_ring_cmap():
    """Build a white-to-black colormap with increasing alpha for ring overlay."""
    color1 = colorConverter.to_rgba("white")
    color2 = colorConverter.to_rgba("black")
    cmap = mpl.colors.LinearSegmentedColormap.from_list("sholl_rings", [color1, color2], 256)
    cmap._init()
    cmap._lut[:, -1] = np.linspace(0, 0.8, cmap.N + 3)
    return cmap


def plot_preview(
    img_new: np.ndarray,
    processed_skeleton: np.ndarray,
    center: tuple,
    circles: list,
) -> plt.Figure:
    """Side-by-side preview: original image (left) and skeleton + rings (right)."""
    cmap2 = _make_ring_cmap()
    fig, (ax_orig, ax_skel) = plt.subplots(1, 2, figsize=(14, 5))

    ax_orig.imshow(img_new, cmap="gray")
    ax_orig.set_title("Original Image")

    ax_skel.scatter(center[0][0], center[0][1])
    ax_skel.imshow(processed_skeleton, origin="lower", cmap="gray")
    ax_skel.set_title("Look good??")
    for circ in circles:
        ax_skel.imshow(circ, interpolation="nearest", origin="lower", cmap=cmap2)
    ax_skel.invert_yaxis()

    return fig


def plot_results(
    processed_skeleton: np.ndarray,
    center: tuple,
    circles: list,
    intersections_to_plot,
    x_ep: np.ndarray,
    y_ep: np.ndarray,
    title: str = "",
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Final results figure: skeleton, rings, intersection scatter, and endpoints.

    Parameters
    ----------
    processed_skeleton : np.ndarray
        Cleaned skeleton array.
    center : tuple
        (x, y) centre from ginput.
    circles : list of np.ndarray
        Circle arrays.
    intersections_to_plot : pd.DataFrame
        Cleaned intersection DataFrame (columns 0=row, 1=col).
    x_ep, y_ep : np.ndarray
        Endpoint row and column coordinates.
    title : str, optional
        Figure title.
    save_path : str or None, optional
        If given, the figure is saved to this path.
    dpi : int, optional
        Resolution for saved figure (default 150).
    """
    cmap2 = _make_ring_cmap()
    fig, ax = plt.subplots(1, figsize=(10, 10))

    ax.scatter(center[0][0], center[0][1], zorder=5)
    ax.imshow(processed_skeleton, origin="lower", cmap="gray")
    ax.set_title(title)

    for circ in circles:
        ax.imshow(circ, interpolation="nearest", origin="lower", cmap=cmap2)

    ax.invert_yaxis()
    ax.scatter(intersections_to_plot[1].values, intersections_to_plot[0].values,
               label="intersections", zorder=4)
    ax.scatter(y_ep, x_ep, marker="^", label="endpoints", zorder=4)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_sholl_curve(
    radii: np.ndarray,
    intersection_counts: np.ndarray,
    title: str = "Sholl Analysis",
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot the classic Sholl curve (intersections vs. radius).

    Parameters
    ----------
    radii : np.ndarray
        Ring radii.
    intersection_counts : np.ndarray
        Number of intersections at each radius.
    title : str, optional
        Plot title.
    save_path : str or None, optional
        If given, save the figure to this path.
    dpi : int, optional
        Resolution for saved figure (default 150).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(radii, intersection_counts, marker="o", linewidth=1.5)
    ax.set_xlabel("Radius (px)")
    ax.set_ylabel("# Intersections")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
