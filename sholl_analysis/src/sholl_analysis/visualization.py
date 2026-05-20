"""
visualization.py
----------------
Plotting helpers for Sholl analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def _draw_rings(ax, center, circles, image_shape):
    """
    Draw Sholl rings as matplotlib Circle patches directly onto *ax*.

    This approach is backend- and OS-independent — it avoids the alpha
    colormap overlay that renders invisibly on some macOS backends.
    """
    cx, cy = center[0][0], center[0][1]
    for circ_arr in circles:
        rows, cols = np.where(circ_arr == 255)
        if len(rows) == 0:
            continue
        radius = np.mean(np.sqrt((cols - cx) ** 2 + (rows - cy) ** 2))
        patch = Circle(
            (cx, cy),
            radius=radius,
            fill=False,
            edgecolor="cyan",
            linewidth=0.8,
            alpha=0.6,
            zorder=3,
        )
        ax.add_patch(patch)


def plot_preview(
    img_new: np.ndarray,
    processed_skeleton: np.ndarray,
    center: tuple,
    circles: list,
) -> plt.Figure:
    """Side-by-side preview: original image (left) and skeleton + rings (right)."""
    fig, (ax_orig, ax_skel) = plt.subplots(1, 2, figsize=(14, 5))

    ax_orig.imshow(img_new, cmap="gray")
    ax_orig.set_title("Original Image")

    ax_skel.imshow(processed_skeleton, cmap="gray")
    ax_skel.set_title("Look good??")
    _draw_rings(ax_skel, center, circles, processed_skeleton.shape)
    ax_skel.scatter(center[0][0], center[0][1], c="blue", s=40, zorder=5,
                    label="centre")
    ax_skel.legend(fontsize=8)

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
    intersection_size: int = 12,
    show_endpoints: bool = False,
    endpoint_size: int = 15,
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
    intersection_size : int, optional
        Marker size for intersection dots (default 12).
    show_endpoints : bool, optional
        Whether to plot endpoint triangles (default False).
    endpoint_size : int, optional
        Marker size for endpoint triangles (default 15).
    """
    fig, ax = plt.subplots(1, figsize=(10, 10))

    ax.imshow(processed_skeleton, cmap="gray")
    ax.set_title(title)

    _draw_rings(ax, center, circles, processed_skeleton.shape)

    ax.scatter(center[0][0], center[0][1], c="blue", s=40, zorder=5,
               label="centre")
    ax.scatter(
        intersections_to_plot[1].values,
        intersections_to_plot[0].values,
        c="orange", s=intersection_size,
        label="intersections", zorder=4,
    )

    if show_endpoints:
        ax.scatter(
            y_ep, x_ep,
            c="lime", marker="^", s=endpoint_size,
            label="endpoints", zorder=4,
        )

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
    x_unit: str = "px",
) -> plt.Figure:
    """
    Plot the classic Sholl curve (intersections vs. radius).

    Parameters
    ----------
    radii : np.ndarray
        Ring radii (in whatever unit the caller provides — px or µm).
    intersection_counts : np.ndarray
        Number of intersections at each radius.
    title : str, optional
        Plot title.
    save_path : str or None, optional
        If given, save the figure to this path.
    dpi : int, optional
        Resolution for saved figure (default 150).
    x_unit : str, optional
        Unit label for the x-axis (default ``"px"``; pass ``"µm"`` when a
        pixel size has been applied to *radii*).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(radii, intersection_counts, marker="o", linewidth=1.5)
    ax.set_xlabel(f"Radius ({x_unit})")
    ax.set_ylabel("# Intersections")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
