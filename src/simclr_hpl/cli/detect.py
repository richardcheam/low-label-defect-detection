from __future__ import annotations

import argparse
from pathlib import Path

from simclr_hpl.config import load_config
from simclr_hpl.detection.train import train_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RF-DETR detector on a COCO dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/sixray_detection.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = train_detector(config)
    print(f"Saved training summary to {summary['output_dir']}/train_summary.json")


if __name__ == "__main__":
    main()
