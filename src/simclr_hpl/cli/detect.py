from __future__ import annotations

import argparse
from pathlib import Path

from simclr_hpl.config import load_config
from simclr_hpl.detection.train import train_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RF-DETR detector on a COCO dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/sixray_detection.yaml"))
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Override train.devices from the config, e.g. '4' or 'auto'. "
            "For multi-GPU, launch this command with "
            "`torchrun --nproc_per_node=<N>` and pass --devices auto."
        ),
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help=(
            "Override train.strategy from the config (PTL Trainer(strategy=...)). "
            "For multi-GPU, use 'ddp_find_unused_parameters_true' to avoid "
            "'has parameters that were not used in producing loss' DDP errors."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.devices is not None:
        config.setdefault("train", {})["devices"] = args.devices
    if args.strategy is not None:
        config.setdefault("train", {})["strategy"] = args.strategy
    summary = train_detector(config)
    print(f"Saved training summary to {summary['output_dir']}/train_summary.json")


if __name__ == "__main__":
    main()
