"""CLI: full-split, single-process COCO evaluation of a trained detector.

Produces the number that is safe to report. The ``val/*`` columns written by
training are per-rank shards under DDP (see
:mod:`simclr_hpl.detection.evaluate` for why), so they describe convergence but
are not benchmark figures. This command runs one process over the whole split.

    uv run xray-eval --checkpoint artifacts/sixray_detection/checkpoint_best_total.pth \\
        --annotations data/sixray_v3_coco/test/_annotations.coco.json \\
        --images-dir data/sixray_v3_coco/test \\
        --out artifacts/sixray_detection/eval_test.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simclr_hpl.detection.evaluate import EVAL_SCORE_THRESHOLD, evaluate_split
from simclr_hpl.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RF-DETR checkpoint over a full COCO split "
        "in a single process, producing reportable mAP."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", default="nano", choices=["nano", "small", "base"])
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=EVAL_SCORE_THRESHOLD,
        help="Keep detections above this score. Deliberately low: mAP integrates "
        "the full precision/recall curve, so a 0.5 cut understates every AP.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N images (smoke test). Recorded in the output.",
    )
    parser.add_argument("--out", type=Path, default=Path("eval_results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = evaluate_split(
        args.annotations,
        args.images_dir,
        variant=args.variant,
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        limit=args.limit,
    )

    save_json(args.out, results)

    scope = (
        "full split" if results["full_split"] else f"FIRST {int(results['num_images'])} IMAGES ONLY"
    )
    print(f"Evaluated {int(results['num_images'])} images ({scope}), single process")
    for key in ("mAP_50_95", "mAP_50", "mAP_75", "mAR_100"):
        print(f"  {key:<10} {results[key]:.4f}")
    per_class = {k: v for k, v in results.items() if k.startswith("AP_50_95/")}
    if per_class:
        print("  per-class AP@[.50:.95]:")
        for name, value in sorted(per_class.items()):
            print(f"    {name.split('/', 1)[1]:<10} {value:.4f}")
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
