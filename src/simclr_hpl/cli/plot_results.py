from __future__ import annotations

import argparse
from pathlib import Path

from simclr_hpl.visualization import create_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots from experiment metrics JSON files.")
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Path to a metrics JSON file produced by this project.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for plot outputs. Defaults to <metrics_dir>/plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    created = create_plots(args.metrics, args.output_dir)
    print(f"Generated {len(created)} plot(s):")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
