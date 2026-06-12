"""CLI: run a trained RF-DETR detector over threat/clean image sets.

Produces the ``predictions.json`` consumed by ``xray-roi``
(:mod:`simclr_hpl.cli.roi_report`): a list of
``{"score": float, "target": int}`` records, one per image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simclr_hpl.detection.predict import build_predictions
from simclr_hpl.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score threat/clean images with a trained RF-DETR detector "
        "and write a predictions.json for xray-roi."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", default="nano", choices=["nano", "small", "base"])
    parser.add_argument("--threat-dir", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("predictions.json"))
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()

    predictions = build_predictions(
        args.threat_dir,
        args.clean_dir,
        variant=args.variant,
        checkpoint=args.checkpoint,
        threshold=args.threshold,
    )

    save_json(args.out, predictions)

    threat_scores = [p["score"] for p in predictions if p["target"] == 1]
    clean_scores = [p["score"] for p in predictions if p["target"] == 0]

    print(f"Scored {len(predictions)} images")
    print(f"  threats: {len(threat_scores)} (mean score {_mean(threat_scores):.3f})")
    print(f"  clean:   {len(clean_scores)} (mean score {_mean(clean_scores):.3f})")
    print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
