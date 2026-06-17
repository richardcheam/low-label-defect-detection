"""CLI: Grounding DINO zero-shot threat detection (Phase 3).

Runs inference on a directory of X-ray images using Grounding DINO with text
prompts — no task-specific training required. The output ``predictions.json``
is compatible with ``xray-roi``, so the full ROI report can be generated
immediately afterward.

Usage (after ``uv sync --extra zeroshot``):

    uv run xray-zeroshot --config configs/grounding_dino_zeroshot.yaml
    uv run xray-roi --config configs/roi_zeroshot.yaml \\
        --predictions artifacts/grounding_dino_zeroshot/predictions.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simclr_hpl.config import load_config
from simclr_hpl.detection.zeroshot import build_zeroshot_predictions
from simclr_hpl.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grounding DINO zero-shot X-ray threat detection."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/grounding_dino_zeroshot.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = build_zeroshot_predictions(
        threat_dir=config["data"]["threat_dir"],
        clean_dir=config["data"].get("clean_dir"),
        model_id=config["model_id"],
        text_labels=config["text_labels"],
        threshold=config["threshold"],
    )

    out_path = output_dir / "predictions.json"
    save_json(out_path, predictions)

    n_threat = sum(1 for p in predictions if p["target"] == 1)
    n_clean = sum(1 for p in predictions if p["target"] == 0)
    print(f"Scored {n_threat} threat images and {n_clean} clean images.")
    print(f"Saved {len(predictions)} predictions → {out_path}")
    print(f"\nNext: uv run xray-roi --config configs/roi_zeroshot.yaml "
          f"--predictions {out_path}")


if __name__ == "__main__":
    main()
