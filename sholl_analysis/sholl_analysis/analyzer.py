"""
analyzer.py
-----------
High-level ShollAnalyzer class that orchestrates the full pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from .image_processing import (
    load_and_preprocess,
    skeletonize_image,
    dilate_skeleton,
    find_endpoints,
    smooth_binary,
)
from .geometry import make_circles, calc_intersection, x_y_separate, clean_intersections, sholl_stats
from .visualization import plot_preview, plot_results, plot_sholl_curve
from .io import save_intersections, ensure_output_dirs, csv_exists
from .supported_formats import is_supported, get_stem, extensions_str


class UserCancelledError(Exception):
    """Raised when the user explicitly cancels processing."""
    pass


def _wait_for_key(fig, prompt: str, valid_keys: dict) -> str:
    """
    Display *prompt* in the figure title and block until the user presses
    one of the keys in *valid_keys*.  Returns the corresponding action string.

    This keeps the matplotlib event loop alive on the main thread (required
    on macOS) while still waiting for user input — avoiding the freeze caused
    by mixing ``plt.show(block=False)`` with ``input()``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    prompt : str
        Shown in the figure suptitle so the user knows what to press.
    valid_keys : dict
        Mapping of key strings → action strings.
        Special key ``""`` is used as the default for Enter/Return.
        Example::

            {"enter": "ok", " ": "ok", "s": "skip", "q": "quit", "r": "redo"}

    Returns
    -------
    str
        One of the values from *valid_keys*.
    """
    result = [None]   # mutable container so the callback can write to it

    key_hint = "  |  ".join(f"[{k}] {v}" for k, v in valid_keys.items())
    fig.suptitle(f"{prompt}\n{key_hint}", fontsize=9, color="white",
                 bbox=dict(facecolor="black", alpha=0.6, pad=4))
    fig.canvas.draw_idle()

    def _on_key(event):
        k = (event.key or "").lower()
        # Treat enter / return / space as the first listed key if it maps to ok
        if k in ("enter", "return", " ", ""):
            k = "enter"
        if k in valid_keys:
            result[0] = valid_keys[k]
            plt.close(fig)

    cid = fig.canvas.mpl_connect("key_press_event", _on_key)

    # Also handle window-close as a cancel/skip
    def _on_close(event):
        if result[0] is None:
            result[0] = valid_keys.get("close", "skip")

    fig.canvas.mpl_connect("close_event", _on_close)

    # Block here — plt.show() runs the event loop until the figure is closed
    plt.show(block=True)
    fig.canvas.mpl_disconnect(cid)

    return result[0] or "skip"


class ShollAnalyzer:
    """
    Performs Sholl analysis on a directory of TIFF microscopy images.

    Parameters
    ----------
    start_radius : int, optional
        Innermost ring radius in pixels (default 20).
    step_size : int, optional
        Distance between successive rings in pixels (default 30).
    end_radius : int, optional
        Outermost ring radius in pixels (default 1024).
    min_object_size : int, optional
        Minimum skeleton fragment size to retain (default 10 pixels).
    dilation_radius : int, optional
        Dilation applied to skeleton before intersection counting (default 1).
    merge_dist : float, optional
        Pixel distance below which two intersection points are merged (default 10).
    show_sholl_curve : bool, optional
        Pop up the Sholl curve plot after each image (default False).
        The curve is always saved to disk regardless of this setting.
    show_results_plot : bool, optional
        Pop up the final annotated skeleton plot after each image (default True).
    save_figures : bool, optional
        Save PNG figures to the output directory (default True).
    figure_dpi : int, optional
        DPI for saved figures (default 150).
    gaussian_sigma : float, optional
        Gaussian smoothing applied before skeletonizing (default 0 = off).

    Output structure
    ----------------
    sholl_output/
        sholl_summary.csv
        intersections/
            <stem>_raw_intersections.csv
        skeleton_plots/
            sholl_<stem>.png
        sholl_curves/
            sholl_curve_<stem>.png

    Examples
    --------
    >>> from sholl_analysis import ShollAnalyzer
    >>> analyzer = ShollAnalyzer(start_radius=20, step_size=30, end_radius=600)
    >>> analyzer.run("/path/to/tiffs", "/path/to/output")
    """

    def __init__(
        self,
        start_radius: int = 20,
        step_size: int = 30,
        end_radius: int = 1024,
        min_object_size: int = 10,
        dilation_radius: int = 1,
        merge_dist: float = 10.0,
        show_sholl_curve: bool = False,
        show_results_plot: bool = True,
        save_figures: bool = True,
        figure_dpi: int = 150,
        gaussian_sigma: float = 0.0,
        intersection_size: int = 12,
        show_endpoints: bool = False,
        endpoint_size: int = 15,
        pixel_size: float = 1.0,
    ):
        self.start_radius = start_radius
        self.step_size = step_size
        self.end_radius = end_radius
        self.min_object_size = min_object_size
        self.dilation_radius = dilation_radius
        self.merge_dist = merge_dist
        self.show_sholl_curve = show_sholl_curve
        self.show_results_plot = show_results_plot
        self.save_figures = save_figures
        self.figure_dpi = figure_dpi
        self.gaussian_sigma = gaussian_sigma
        self.intersection_size = intersection_size
        self.show_endpoints = show_endpoints
        self.endpoint_size = endpoint_size
        self.pixel_size = pixel_size
        self.radii = np.arange(start_radius, end_radius, step_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, input_dir: str, output_dir: str | None = None) -> pd.DataFrame:
        """
        Process all TIFF images in *input_dir*.

        All prompts appear as keyboard shortcuts on the figure itself —
        no terminal input required. Ctrl+C in the terminal exits cleanly.

        Controls
        --------
        Centre-selection window:
          - Click soma, then close window or press Enter to confirm
          - Close without clicking to skip

        Preview window:
          - Enter / Space  → process and continue
          - s              → skip this image
          - r              → redo centre selection
          - q              → quit and save progress

        Results window (if shown):
          - Enter / Space / any key → next image
          - q                       → quit

        Parameters
        ----------
        input_dir : str
            Folder containing image files.
        output_dir : str or None, optional
            Root output folder. Defaults to ``<input_dir>/sholl_output``.

        Returns
        -------
        summary : pd.DataFrame
            One row per completed image; columns are per-radius intersection counts.
        """
        if output_dir is None:
            dirs = ensure_output_dirs(input_dir)
        else:
            dirs = ensure_output_dirs(os.path.dirname(output_dir),
                                      subdir=os.path.basename(output_dir))

        files = sorted(f for f in os.listdir(input_dir) if is_supported(f))
        if not files:
            raise FileNotFoundError(
                f"No supported images ({extensions_str()}) found in {input_dir!r}"
            )

        pending, already_done = [], []
        for f in files:
            stem = get_stem(f)
            if csv_exists(dirs["root"], stem):
                already_done.append(f)
            else:
                pending.append(f)

        print(f"Found {len(files)} image(s): {len(pending)} to process, "
              f"{len(already_done)} already done.")
        print(f"Output → {dirs['root']}\n")

        if not pending:
            print("Nothing to do — all images already processed.")
            return self._load_existing_summary(dirs["root"])

        print("Tip: all controls are keyboard shortcuts on the figure window.\n")

        existing_summary = self._load_existing_summary(dirs["root"])
        summary_rows = {}
        skipped = []

        try:
            for idx, filename in enumerate(pending, 1):
                filepath = os.path.join(input_dir, filename)
                stem = get_stem(filename)
                print(f"[{idx}/{len(pending)}] {filename}")

                try:
                    counts, action = self._process_image(filepath, stem, dirs)
                except UserCancelledError as e:
                    action = str(e)

                if action == "skip":
                    skipped.append(filename)
                    print("  Skipped.\n")
                elif action == "quit":
                    print("\nStopping early at user request.")
                    break
                else:
                    summary_rows[stem] = counts
                    print()

        except KeyboardInterrupt:
            print("\n\nInterrupted — saving progress so far …")

        return self._finalise(summary_rows, skipped, dirs, existing_summary)

    def process_single(
        self,
        filepath: str,
        output_dir: str,
        stem: str | None = None,
    ) -> np.ndarray:
        """
        Process a single TIFF image.

        Parameters
        ----------
        filepath : str
            Path to the TIFF file.
        output_dir : str
            Root output directory (subfolders are created automatically).
        stem : str or None, optional
            Base name for output files; inferred from *filepath* if not given.

        Returns
        -------
        counts : np.ndarray or None
            Per-radius intersection counts, or ``None`` if the user cancelled.
        """
        if stem is None:
            stem = get_stem(os.path.basename(filepath))
        dirs = ensure_output_dirs(os.path.dirname(output_dir),
                                  subdir=os.path.basename(output_dir))
        try:
            counts, _ = self._process_image(filepath, stem, dirs)
        except UserCancelledError:
            return None
        return counts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _x_unit(self) -> str:
        return "µm" if self.pixel_size != 1.0 else "px"

    def _process_image(self, filepath: str, stem: str, dirs: dict):
        """Full pipeline for one image. Returns (counts, action)."""

        # 1. Load + preprocess
        img_new, img_raw = load_and_preprocess(filepath)

        # 1b. Optional Gaussian smoothing
        if self.gaussian_sigma > 0:
            img_new = smooth_binary(img_new, gaussian_sigma=self.gaussian_sigma)

        # 2. Skeletonize
        skeleton = skeletonize_image(img_new, min_size=self.min_object_size)

        # 3. Centre selection — ginput handles its own event loop cleanly
        fig_centre, ax_centre = plt.subplots(figsize=(7, 7))
        ax_centre.imshow(img_new, cmap="gray")
        ax_centre.set_title(
            "Click the soma centre, then close the window\n"
            "(close without clicking to skip this image)",
            fontsize=9,
        )
        fig_centre.tight_layout()
        center = plt.ginput(1, timeout=0)   # blocks until click + close
        plt.close(fig_centre)

        if not center:
            print("  No centre selected.")
            # Ask via a tiny dialog figure rather than terminal input
            action = self._ask_figure("No centre selected — what next?",
                                      {"s": "skip", "q": "quit"})
            raise UserCancelledError(action)

        # 4. Build Sholl rings
        circles = make_circles(center[0], self.radii, skeleton.shape)

        # 5. Preview — response collected via keypress on the figure
        fig_preview = plot_preview(img_new, skeleton, center, circles)
        fig_preview.suptitle(stem, fontsize=10)
        resp = _wait_for_key(
            fig_preview,
            prompt="Preview — does this look right?",
            valid_keys={
                "enter": "ok", " ": "ok",
                "s": "skip",
                "q": "quit",
                "r": "redo",
                "close": "ok",   # closing the window = accept
            },
        )

        if resp == "quit":
            raise UserCancelledError("quit")
        if resp == "skip":
            raise UserCancelledError("skip")
        if resp == "redo":
            print("  Redoing centre selection …")
            return self._process_image(filepath, stem, dirs)

        # 6. Dilate skeleton + count intersections
        mgla_arr = dilate_skeleton(skeleton, radius=self.dilation_radius)

        raw_z = []
        intersection_counts = []
        print("  Counting intersections …")
        try:
            for circ in tqdm(circles, leave=False):
                hits = calc_intersection(circ, mgla_arr)
                raw_z.append(hits)
                intersection_counts.append(len(hits) // 2)
        except KeyboardInterrupt:
            print("\n  Counting interrupted.")
            raise UserCancelledError("skip")

        intersection_counts = np.array(intersection_counts)

        # 7. Clean duplicates
        intersections_to_plot = clean_intersections(
            raw_z, self.radii, merge_dist=self.merge_dist
        )

        # 8. Endpoints
        endpoint_arr, n_endpoints = find_endpoints(skeleton)
        intersections_to_plot["endpoints"] = n_endpoints

        # 9. Save CSV → intersections/
        save_intersections(intersections_to_plot, dirs["csvs"], stem)

        # 10. Endpoint coordinates
        ep = np.argwhere(endpoint_arr == 255.0)
        x_ep, y_ep = ep[:, 0], ep[:, 1]

        # 11. Results plot → skeleton_plots/
        results_path = (
            os.path.join(dirs["skeletons"], f"sholl_{stem}.png")
            if self.save_figures else None
        )
        fig_results = plot_results(
            skeleton, center, circles, intersections_to_plot,
            x_ep, y_ep, title=stem, save_path=results_path,
            dpi=self.figure_dpi,
            intersection_size=self.intersection_size,
            show_endpoints=self.show_endpoints,
            endpoint_size=self.endpoint_size,
        )
        if self.show_results_plot:
            resp = _wait_for_key(
                fig_results,
                prompt="Results",
                valid_keys={"enter": "ok", " ": "ok", "q": "quit", "close": "ok"},
            )
            if resp == "quit":
                raise UserCancelledError("quit")
        else:
            plt.close(fig_results)

        # 12. Sholl curve → sholl_curves/
        curve_path = (
            os.path.join(dirs["curves"], f"sholl_curve_{stem}.png")
            if self.save_figures else None
        )
        fig_curve = plot_sholl_curve(
            self.radii * self.pixel_size, intersection_counts,
            title=f"Sholl Curve – {stem}",
            save_path=curve_path,
            dpi=self.figure_dpi,
            x_unit=self._x_unit,
        )
        if self.show_sholl_curve:
            resp = _wait_for_key(
                fig_curve,
                prompt="Sholl curve",
                valid_keys={"enter": "ok", " ": "ok", "q": "quit", "close": "ok"},
            )
            if resp == "quit":
                raise UserCancelledError("quit")
        else:
            plt.close(fig_curve)

        # 13. Quick inline summary
        max_idx = np.argmax(intersection_counts)
        peak_radius = self.radii[max_idx] * self.pixel_size
        print(
            f"  ✓  {n_endpoints} endpoints  |  "
            f"peak {int(intersection_counts[max_idx])} intersections "
            f"@ radius {peak_radius:.1f}{self._x_unit}"
        )

        plt.close("all")
        return intersection_counts, "ok"

    def _ask_figure(self, message: str, valid_keys: dict) -> str:
        """Show a minimal figure with a message and wait for a keypress."""
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.axis("off")
        key_hint = "  |  ".join(f"[{k}] {v}" for k, v in valid_keys.items())
        ax.text(0.5, 0.5, f"{message}\n\n{key_hint}",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        result = _wait_for_key(fig, message, {**valid_keys, "close": "skip"})
        return result

    def _load_existing_summary(self, output_root: str) -> pd.DataFrame:
        """Load and return an existing summary CSV if present.

        Stat columns (string-named) are dropped so that only the per-radius
        counts (numeric column names) are returned — stats are always
        recomputed in ``_finalise``.
        """
        summary_path = os.path.join(output_root, "sholl_summary.csv")
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path, index_col="image")
            radius_cols = [c for c in df.columns if str(c).lstrip("-").isdigit()]
            return df[radius_cols]
        return pd.DataFrame()

    def _finalise(
        self,
        summary_rows: dict,
        skipped: list,
        dirs: dict,
        existing_summary: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Write the summary CSV to the root dir and print a completion report."""
        scaled_radii = self.radii * self.pixel_size
        new_summary = pd.DataFrame(summary_rows, index=scaled_radii).T
        new_summary.index.name = "image"

        if existing_summary is not None and not existing_summary.empty:
            # Drop any rows that were re-processed (new results take precedence)
            prior = existing_summary.drop(
                index=[s for s in summary_rows if s in existing_summary.index],
                errors="ignore",
            )
            summary = pd.concat([prior, new_summary])
        else:
            summary = new_summary

        # Compute derived stats for every row and append as columns
        scaled_radii = self.radii * self.pixel_size
        stat_rows = {
            stem: sholl_stats(scaled_radii, summary.loc[stem].values)
            for stem in summary.index
        }
        stats_df = pd.DataFrame(stat_rows).T
        stats_df.index.name = "image"
        summary = pd.concat([summary, stats_df], axis=1)

        summary.to_csv(os.path.join(dirs["root"], "sholl_summary.csv"))

        n_done = len(summary_rows)
        n_skip = len(skipped)
        print(f"\n{'─' * 40}")
        print(f"  Completed : {n_done} image(s)")
        if skipped:
            print(f"  Skipped   : {n_skip} — {', '.join(skipped)}")
        print(f"  Output    : {dirs['root']}")
        print(f"    ├── sholl_summary.csv")
        print(f"    ├── intersections/       ({n_done} CSV file(s))")
        print(f"    ├── skeleton_plots/      ({n_done} PNG file(s))")
        print(f"    └── sholl_curves/        ({n_done} PNG file(s))")
        print(f"{'─' * 40}")
        if n_done > 0:
            print("All done!")
        return summary

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ShollAnalyzer("
            f"start_radius={self.start_radius}, "
            f"step_size={self.step_size}, "
            f"end_radius={self.end_radius}, "
            f"pixel_size={self.pixel_size}, "
            f"show_sholl_curve={self.show_sholl_curve}, "
            f"gaussian_sigma={self.gaussian_sigma})"
        )
