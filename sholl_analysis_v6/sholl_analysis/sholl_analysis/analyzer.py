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
from .geometry import make_circles, calc_intersection, x_y_separate, clean_intersections
from .visualization import plot_preview, plot_results, plot_sholl_curve
from .io import save_intersections, ensure_output_dirs, csv_exists


class UserCancelledError(Exception):
    """Raised when the user explicitly cancels processing."""
    pass


def _ask_continue(prompt: str) -> str:
    """
    Print *prompt* and return the user's response (stripped, lower-cased).
    Handles EOF gracefully (e.g. non-interactive environments).
    """
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "q"


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
        self.radii = np.arange(start_radius, end_radius, step_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, input_dir: str, output_dir: str | None = None) -> pd.DataFrame:
        """
        Process all TIFF images in *input_dir*.

        Prompts after each image's preview:
          - Enter / y  → process and continue
          - s          → skip this image (no output written)
          - r          → redo centre selection
          - q          → stop and save progress

        Ctrl+C exits cleanly at any point, saving all completed images.

        Parameters
        ----------
        input_dir : str
            Folder containing ``.tiff`` images.
        output_dir : str or None, optional
            Root output folder. Defaults to ``<input_dir>/sholl_output``.

        Returns
        -------
        summary : pd.DataFrame
            One row per completed image; columns are per-radius intersection counts.
        """
        # Build the full subfolder tree up front
        if output_dir is None:
            dirs = ensure_output_dirs(input_dir)
        else:
            dirs = ensure_output_dirs(os.path.dirname(output_dir),
                                      subdir=os.path.basename(output_dir))

        files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".tiff"))
        if not files:
            raise FileNotFoundError(f"No .tiff files found in {input_dir!r}")

        # Separate already-done from pending
        pending, already_done = [], []
        for f in files:
            stem = f[:-5]
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

        print("Press Ctrl+C at any time to stop cleanly.\n")

        summary_rows = {}
        skipped = []

        try:
            for idx, filename in enumerate(pending, 1):
                filepath = os.path.join(input_dir, filename)
                stem = filename[:-5]
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

        return self._finalise(summary_rows, skipped, dirs)

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
        counts : np.ndarray
            Per-radius intersection counts.
        """
        if stem is None:
            stem = os.path.basename(filepath).replace(".tiff", "")
        dirs = ensure_output_dirs(os.path.dirname(output_dir),
                                  subdir=os.path.basename(output_dir))
        counts, _ = self._process_image(filepath, stem, dirs)
        return counts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_image(self, filepath: str, stem: str, dirs: dict):
        """Full pipeline for one image. Returns (counts, action)."""

        # 1. Load + preprocess
        img_new, img_raw = load_and_preprocess(filepath)

        # 1b. Optional Gaussian smoothing
        if self.gaussian_sigma > 0:
            img_new = smooth_binary(img_new, gaussian_sigma=self.gaussian_sigma)

        # 2. Skeletonize
        skeleton = skeletonize_image(img_new, min_size=self.min_object_size)

        # 3. Centre selection
        plt.imshow(img_new, cmap="gray")
        plt.title("Click the soma centre, then press Enter  |  Close to skip")
        try:
            center = plt.ginput(1, timeout=0)
        except Exception:
            center = []
        plt.close()

        if not center:
            resp = _ask_continue("  No centre selected. [s]kip / [q]uit? ")
            raise UserCancelledError("quit" if resp == "q" else "skip")

        # 4. Build Sholl rings
        circles = make_circles(center[0], self.radii, skeleton.shape)

        # 5. Preview
        fig_preview = plot_preview(img_new, skeleton, center, circles)
        plt.show(block=False)
        plt.pause(0.1)

        resp = _ask_continue(
            "  [Enter/y] continue  |  [s] skip  |  [q] quit  |  [r] redo centre: "
        )
        plt.close(fig_preview)

        if resp == "q":
            raise UserCancelledError("quit")
        if resp == "s":
            raise UserCancelledError("skip")
        if resp == "r":
            print("  Restarting centre selection …")
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
                intersection_counts.append(len(hits) / 2)
        except KeyboardInterrupt:
            print("\n  Counting interrupted.")
            resp = _ask_continue("  [s] skip  |  [q] quit: ")
            raise UserCancelledError("quit" if resp == "q" else "skip")

        intersection_counts = np.array(intersection_counts)

        # 7. Clean duplicates
        intersections_to_plot = clean_intersections(raw_z, self.radii, merge_dist=self.merge_dist)

        # 8. Endpoints
        endpoint_arr, n_endpoints = find_endpoints(skeleton)
        intersections_to_plot["endpoints"] = n_endpoints

        # 9. Save CSV → intersections/
        save_intersections(intersections_to_plot, dirs["csvs"], stem)

        # 10. Endpoint coordinates
        ep = []
        for i in range(endpoint_arr.shape[0]):
            for j in range(endpoint_arr.shape[1]):
                if endpoint_arr[i][j] == 255.0:
                    ep = np.append(ep, [i, j])
        x_ep, y_ep = x_y_separate(ep)

        # 11. Results plot → skeleton_plots/
        results_path = (
            os.path.join(dirs["skeletons"], f"sholl_{stem}.png")
            if self.save_figures else None
        )
        fig_results = plot_results(
            skeleton, center, circles, intersections_to_plot,
            x_ep, y_ep, title=stem, save_path=results_path,
            dpi=self.figure_dpi,
        )
        if self.show_results_plot:
            plt.show(block=False)
            plt.pause(0.1)
        plt.close(fig_results)

        # 12. Sholl curve → sholl_curves/
        curve_path = (
            os.path.join(dirs["curves"], f"sholl_curve_{stem}.png")
            if self.save_figures else None
        )
        fig_curve = plot_sholl_curve(
            self.radii, intersection_counts,
            title=f"Sholl Curve – {stem}",
            save_path=curve_path,
            dpi=self.figure_dpi,
        )
        if self.show_sholl_curve:
            plt.show(block=False)
            plt.pause(0.1)
        plt.close(fig_curve)

        # 13. Quick inline summary
        max_idx = np.argmax(intersection_counts)
        print(f"  ✓  {n_endpoints} endpoints  |  "
              f"peak {int(intersection_counts[max_idx])} intersections "
              f"@ radius {self.radii[max_idx]}px")

        resp = _ask_continue("  [Enter] next image  |  [q] quit: ")
        plt.close("all")

        if resp == "q":
            raise UserCancelledError("quit")

        return intersection_counts, "ok"

    def _load_existing_summary(self, output_root: str) -> pd.DataFrame:
        """Load and return an existing summary CSV if present."""
        summary_path = os.path.join(output_root, "sholl_summary.csv")
        if os.path.exists(summary_path):
            return pd.read_csv(summary_path, index_col="image")
        return pd.DataFrame()

    def _finalise(self, summary_rows: dict, skipped: list, dirs: dict) -> pd.DataFrame:
        """Write the summary CSV to the root dir and print a completion report."""
        summary = pd.DataFrame(summary_rows, index=self.radii).T
        summary.index.name = "image"
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
            f"show_sholl_curve={self.show_sholl_curve}, "
            f"gaussian_sigma={self.gaussian_sigma})"
        )
