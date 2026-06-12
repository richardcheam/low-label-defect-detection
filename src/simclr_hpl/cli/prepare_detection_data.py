from __future__ import annotations

import argparse
from pathlib import Path

from simclr_hpl.detection.data import DEFAULT_SPLITS, convert_roboflow_yolo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Roboflow YOLOv11 export to per-split COCO datasets for RF-DETR."
    )
    parser.add_argument(
        "--yolo-root",
        type=Path,
        default=Path("data/sixray_v3"),
        help="Root of the Roboflow YOLOv11 export (contains data.yaml, train/, valid/, test/).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/sixray_v3_coco"),
        help="Output root for the converted COCO datasets.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits to convert (default: train valid test). Missing splits are skipped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coco_paths = convert_roboflow_yolo_dataset(
        yolo_root=args.yolo_root,
        output_root=args.output_root,
        splits=tuple(args.splits),
    )
    print(f"Converted {len(coco_paths)} split(s) from {args.yolo_root} to {args.output_root}:")
    for split, path in coco_paths.items():
        print(f"  {split}: {path}")


if __name__ == "__main__":
    main()
