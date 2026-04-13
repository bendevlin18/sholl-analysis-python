"""
cli.py
------
Command-line interface for sholl_analysis.

Usage
-----
    sholl-analysis --input /path/to/tiffs
    sholl-analysis --input /path/to/tiffs --output /path/to/results \\
                   --start 20 --step 30 --end 600 --sigma 1.5 --show-curve
"""

import argparse
import sys
from .analyzer import ShollAnalyzer


def main():
    parser = argparse.ArgumentParser(
        prog="sholl-analysis",
        description="Sholl analysis of microscopy images (.tiff, .tif, .png, .jpg).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument("--input",  "-i", required=True,
                        help="Directory containing images (.tiff, .tif, .png, .jpg).")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: <input>/sholl_output).")

    # Ring geometry
    parser.add_argument("--start",  type=int,   default=20,   help="Start radius in pixels.")
    parser.add_argument("--step",   type=int,   default=30,   help="Step size in pixels.")
    parser.add_argument("--end",    type=int,   default=1024, help="End radius in pixels.")

    # Skeleton processing
    parser.add_argument("--min-size",   type=int,   default=10,  help="Min skeleton fragment size (px).")
    parser.add_argument("--dilation",   type=int,   default=1,   help="Skeleton dilation radius.")
    parser.add_argument("--merge-dist", type=float, default=10.0,help="Max distance to merge nearby intersections (px).")
    parser.add_argument("--sigma",      type=float, default=0.0, help="Gaussian smoothing sigma (0 = off).")

    # Scale
    parser.add_argument("--pixel-size", type=float, default=1.0,
                        help="Pixel size in µm/px. Converts radii and curve x-axis to µm (default 1.0 = pixels).")
    parser.add_argument("--use-micron", action="store_true",
                        help="Interpret --start/--step/--end as µm instead of pixels. Requires --pixel-size.")

    # Display
    parser.add_argument("--show-curve",  action="store_true",  help="Show the Sholl curve plot for each image.")
    parser.add_argument("--no-results",  action="store_true",  help="Hide the annotated skeleton plot.")
    parser.add_argument("--no-save",     action="store_true",  help="Do not save PNG figures to disk.")
    parser.add_argument("--dpi",         type=int, default=150, help="Saved figure resolution (DPI).")

    args = parser.parse_args()

    analyzer = ShollAnalyzer(
        start_radius=args.start,
        step_size=args.step,
        end_radius=args.end,
        min_object_size=args.min_size,
        dilation_radius=args.dilation,
        merge_dist=args.merge_dist,
        gaussian_sigma=args.sigma,
        show_sholl_curve=args.show_curve,
        show_results_plot=not args.no_results,
        save_figures=not args.no_save,
        figure_dpi=args.dpi,
        pixel_size=args.pixel_size,
        use_micron=args.use_micron,
    )

    try:
        summary = analyzer.run(args.input, args.output)
        n = len(summary)
        if n:
            print(f"\nSummary written ({n} image(s) processed).")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
