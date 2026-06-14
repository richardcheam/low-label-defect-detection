"""CLI: few-label + pseudo-box semi-supervised RF-DETR detection (Phase 2a).

Mirrors the classification-style iterative pseudo-labeling loop in
:mod:`simclr_hpl.cli.pseudo_labeling`, applied to object detection:

1. Split the COCO ``train`` split's images into a small "labeled" subset
   (``pseudo_labeling.label_fraction``, e.g. 1%) and a large "unlabeled" pool
   (:func:`simclr_hpl.detection.pseudo_box.split_labeled_unlabeled`).
2. Train a "round 0" baseline detector on the labeled subset only.
3. For ``pseudo_labeling.iterations`` rounds: run the current model on the
   unlabeled pool, keep detections above a (decaying) confidence threshold as
   pseudo-boxes, merge them into the training set, and retrain from scratch.

Each round's dataset lives under ``<output_dir>/round_<N>/`` (symlinked images
+ a merged ``_annotations.coco.json``, via
:func:`simclr_hpl.detection.pseudo_box.build_train_split` and
:func:`simclr_hpl.detection.pseudo_box.link_eval_splits`). Final stats are
written to ``<output_dir>/metrics.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from simclr_hpl.config import load_config
from simclr_hpl.detection.pseudo_box import (
    ANNOTATIONS_FILENAME,
    build_train_split,
    filter_annotations,
    generate_pseudo_annotations,
    link_eval_splits,
    load_coco,
    split_labeled_unlabeled,
)
from simclr_hpl.detection.train import resolve_model_class, run_training_round
from simclr_hpl.tracking import ExperimentTracker
from simclr_hpl.utils import save_json, seed_everything


def run_pseudo_box_detection(
    config: dict[str, Any], tracker: ExperimentTracker | None = None
) -> dict[str, Any]:
    """Run the few-label + pseudo-box semi-supervised training loop.

    Returns the metrics dict that is also written to
    ``<output_dir>/metrics.json``.
    """
    seed_everything(config["seed"])

    coco_root = Path(config["data"]["coco_root"])
    output_dir = Path(config["output_dir"])
    model_cls = resolve_model_class(config["model"]["variant"])
    train_cfg = config["train"]
    pseudo_cfg = config["pseudo_labeling"]

    if tracker is None:
        tracking_cfg = config.get("tracking", {})
        tracker = ExperimentTracker(
            enabled=tracking_cfg.get("enabled", True),
            tracking_uri=tracking_cfg.get("tracking_uri"),
            experiment=tracking_cfg.get("experiment"),
        )

    train_coco = load_coco(coco_root / "train" / ANNOTATIONS_FILENAME)
    categories = train_coco["categories"]
    all_annotations = train_coco["annotations"]

    labeled_images, unlabeled_images = split_labeled_unlabeled(
        train_coco, pseudo_cfg["label_fraction"], config["seed"]
    )
    labeled_ids = {img["id"] for img in labeled_images}
    real_annotations = filter_annotations(all_annotations, labeled_ids)
    next_ann_id = max((ann["id"] for ann in all_annotations), default=0) + 1

    metrics: dict[str, Any] = {
        "label_fraction": pseudo_cfg["label_fraction"],
        "n_labeled_images": len(labeled_images),
        "n_unlabeled_images": len(unlabeled_images),
        "iterations": [],
    }

    with tracker.run(run_name=f"pseudo-box-{config['model']['variant']}"):
        tracker.log_params(
            {
                "variant": config["model"]["variant"],
                "label_fraction": pseudo_cfg["label_fraction"],
                "confidence_threshold": pseudo_cfg["confidence_threshold"],
                "threshold_decay": pseudo_cfg["threshold_decay"],
                "iterations": pseudo_cfg["iterations"],
                "n_labeled_images": len(labeled_images),
                "n_unlabeled_images": len(unlabeled_images),
            }
        )

        # Round 0: few-label baseline.
        round_dir = output_dir / "round_0"
        build_train_split(coco_root, round_dir, labeled_images, real_annotations, categories)
        link_eval_splits(coco_root, round_dir)
        model, baseline_metrics = run_training_round(model_cls, round_dir, round_dir, train_cfg)
        metrics["baseline"] = {"n_images": len(labeled_images), **baseline_metrics}
        tracker.log_metrics({f"baseline_{k}": v for k, v in baseline_metrics.items()})

        accumulated_images = list(labeled_images)
        accumulated_annotations = list(real_annotations)
        remaining_unlabeled = list(unlabeled_images)
        current_threshold = pseudo_cfg["confidence_threshold"]

        for iteration in range(1, pseudo_cfg["iterations"] + 1):
            pseudo_annotations, pseudo_images, confidences = generate_pseudo_annotations(
                model,
                remaining_unlabeled,
                coco_root / "train",
                categories,
                threshold=current_threshold,
                next_id=next_ann_id,
            )
            next_ann_id += len(pseudo_annotations)

            pseudo_image_ids = {img["id"] for img in pseudo_images}
            remaining_unlabeled = [
                img for img in remaining_unlabeled if img["id"] not in pseudo_image_ids
            ]
            accumulated_images.extend(pseudo_images)
            accumulated_annotations.extend(pseudo_annotations)

            round_dir = output_dir / f"round_{iteration}"
            build_train_split(
                coco_root, round_dir, accumulated_images, accumulated_annotations, categories
            )
            link_eval_splits(coco_root, round_dir)
            model, round_metrics = run_training_round(model_cls, round_dir, round_dir, train_cfg)

            iteration_stats: dict[str, Any] = {
                "iteration": iteration,
                "confidence_threshold": current_threshold,
                "new_pseudo_labels": len(pseudo_annotations),
                "new_pseudo_images": len(pseudo_images),
                "avg_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
                "remaining_unlabeled": len(remaining_unlabeled),
                "n_train_images": len(accumulated_images),
                **round_metrics,
            }
            metrics["iterations"].append(iteration_stats)
            tracker.log_metrics(
                {
                    f"round{iteration}_{k}": v
                    for k, v in iteration_stats.items()
                    if isinstance(v, (int, float))
                }
            )

            current_threshold = max(0.5, current_threshold - pseudo_cfg["threshold_decay"])

    save_json(output_dir / "metrics.json", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Few-label + pseudo-box semi-supervised RF-DETR detection."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/pseudo_box_detection.yaml"))
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
    metrics = run_pseudo_box_detection(config)
    print(f"Saved metrics to {config['output_dir']}/metrics.json")
    print(f"Baseline: {metrics['baseline']}")
    for stats in metrics["iterations"]:
        print(f"Round {stats['iteration']}: {stats}")


if __name__ == "__main__":
    main()
