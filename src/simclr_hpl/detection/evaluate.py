"""Full-split, single-process COCO detection evaluation.

Why this module exists
----------------------
The ``val/*`` columns in ``artifacts/sixray_detection/metrics.csv`` are **not**
full-validation-set metrics, and must not be quoted as benchmark numbers.

:func:`simclr_hpl.detection.train._patch_map_metric_sync_on_compute` forces
``MeanAveragePrecision(sync_on_compute=False)`` to avoid a torchmetrics DDP
deadlock (Lightning-AI/torchmetrics#3199, #626). That patch is the right call
for *training* -- it keeps a 4-GPU run alive and the numbers are still usable
for monitoring convergence -- but it means each rank computes mAP over only the
shard of the validation set its ``DistributedSampler`` handed it. With
``--nproc_per_node=4`` on the 1,662-image validation split that is roughly 415
images and 794 boxes per rank.

Averaging or quoting such a value is wrong in a way that is easy to miss. mAP
is not a mean over samples: average precision is read off a precision/recall
curve built by ranking *all* detections in the split against *all* ground
truth. Computing it on a quarter of the data changes the ranking, changes the
recall denominator, and is especially unstable for rare classes -- ``Scissors``
has 206 boxes in the validation split, so about 51 per rank.

The fix here deliberately does not try to make the distributed path correct.
Model selection during training does not need a globally exact metric, and
gathering ragged detection tensors across ranks is exactly what deadlocks. The
reportable number instead comes from a separate evaluation pass that runs in
**one process** over the **whole** split, which is standard practice and is
trivially verifiable.

The same torchmetrics implementation is used as during training, so the only
variable that changes between the two numbers is the sharding.

``rfdetr`` is an optional, heavy CUDA/torch-dependent extra, so it is imported
lazily inside :func:`load_detector`. Importing this module stays safe in a
minimal environment, and :func:`compute_map` -- which holds all of the metric
logic -- is a pure function that can be tested without a model, a checkpoint,
or a GPU.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from simclr_hpl.detection.train import resolve_model_class

# mAP integrates over the full precision/recall curve, so inference must keep
# low-confidence detections. Thresholding at the 0.5 used for the ROI report
# would truncate the curve's tail and silently understate every AP.
EVAL_SCORE_THRESHOLD = 0.001

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    """Convert one COCO ``[x, y, w, h]`` box to ``[x1, y1, x2, y2]``.

    COCO stores width/height; torchmetrics expects corners. Getting this wrong
    does not raise -- it silently produces near-zero IoU and an mAP near zero.
    """
    x, y, width, height = (float(value) for value in box)
    return [x, y, x + width, y + height]


def load_coco_ground_truth(
    annotations_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, list]], dict[int, str]]:
    """Read a COCO annotation file into per-image targets.

    Returns ``(images, targets_by_image_id, category_names)``. Images with no
    annotations are kept with empty boxes: they are legitimate negatives and
    dropping them would inflate precision. Both SIXray splits contain two such
    images.
    """
    with Path(annotations_path).open() as handle:
        data = json.load(handle)

    category_names = {int(c["id"]): str(c["name"]) for c in data["categories"]}
    targets: dict[int, dict[str, list]] = {
        int(image["id"]): {"boxes": [], "labels": []} for image in data["images"]
    }

    for annotation in data["annotations"]:
        if annotation.get("iscrowd"):
            continue
        image_id = int(annotation["image_id"])
        targets[image_id]["boxes"].append(xywh_to_xyxy(annotation["bbox"]))
        targets[image_id]["labels"].append(int(annotation["category_id"]))

    return data["images"], targets, category_names


def _as_tensors(boxes: list, labels: list, scores: list | None = None) -> dict[str, torch.Tensor]:
    """Build a torchmetrics record, keeping empty entries correctly shaped.

    An empty ``(0, 4)`` box tensor is meaningful (a real negative); a bare
    ``(0,)`` tensor raises inside torchmetrics.
    """
    record = {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64).reshape(-1),
    }
    if scores is not None:
        record["scores"] = torch.tensor(scores, dtype=torch.float32).reshape(-1)
    return record


def compute_map(
    predictions: Iterable[dict[str, list]],
    targets: Iterable[dict[str, list]],
    category_names: dict[int, str] | None = None,
) -> dict[str, float]:
    """Compute COCO mAP over *all* supplied images in a single process.

    ``predictions`` records carry ``boxes``/``scores``/``labels``; ``targets``
    carry ``boxes``/``labels``. Boxes are ``[x1, y1, x2, y2]``.

    This is the whole metric surface of the module and it depends only on
    torchmetrics, so it is testable without a model or a checkpoint.
    ``sync_on_compute`` is left at its default: in one process there is nothing
    to synchronise, which is precisely the point of evaluating this way.
    """
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(
        [_as_tensors(p["boxes"], p["labels"], p["scores"]) for p in predictions],
        [_as_tensors(t["boxes"], t["labels"]) for t in targets],
    )
    raw = metric.compute()

    results = {
        "mAP_50_95": float(raw["map"]),
        "mAP_50": float(raw["map_50"]),
        "mAP_75": float(raw["map_75"]),
        "mAR_100": float(raw["mar_100"]),
    }

    per_class = raw.get("map_per_class")
    classes = raw.get("classes")
    if per_class is not None and classes is not None:
        # With a single class torchmetrics returns 0-dim tensors rather than
        # length-1 ones, so iterating naively drops per-class AP exactly when
        # the split has one category. atleast_1d normalises both shapes.
        names = category_names or {}
        class_ids = torch.atleast_1d(classes).tolist()
        values = torch.atleast_1d(per_class).tolist()
        for class_id, value in zip(class_ids, values, strict=False):
            label = names.get(int(class_id), str(class_id))
            results[f"AP_50_95/{label}"] = float(value)
    return results


def load_detector(variant: str, checkpoint: str | Path) -> Any:
    """Lazily import ``rfdetr`` and load a trained checkpoint."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        msg = (
            f"Checkpoint not found: {checkpoint_path}. Train a detector with "
            "`xray-detect` first, then pass its weights via --checkpoint."
        )
        raise FileNotFoundError(msg)
    return resolve_model_class(variant)(pretrain_weights=str(checkpoint_path))


def detections_to_record(detections: Any) -> dict[str, list]:
    """Convert a supervision ``Detections`` object to a prediction record."""
    boxes = detections.xyxy
    if boxes is None or len(boxes) == 0:
        return {"boxes": [], "scores": [], "labels": []}
    return {
        "boxes": [[float(v) for v in box] for box in boxes],
        "scores": [float(s) for s in detections.confidence],
        "labels": [int(c) for c in detections.class_id],
    }


def evaluate_split(
    annotations_path: str | Path,
    images_dir: str | Path,
    *,
    variant: str,
    checkpoint: str | Path,
    threshold: float = EVAL_SCORE_THRESHOLD,
    limit: int | None = None,
) -> dict[str, float]:
    """Evaluate a checkpoint over a whole COCO split in one process.

    ``limit`` truncates the image list for smoke tests. It is recorded in the
    output so a truncated run can never be mistaken for a full-split result.
    """
    images, targets_by_id, category_names = load_coco_ground_truth(annotations_path)
    if limit is not None:
        images = images[:limit]

    model = load_detector(variant, checkpoint)
    images_dir = Path(images_dir)

    predictions: list[dict[str, list]] = []
    targets: list[dict[str, list]] = []
    for image_info in images:
        image_path = images_dir / image_info["file_name"]
        with Image.open(image_path) as handle:
            detections = model.predict(handle.convert("RGB"), threshold=threshold)
        predictions.append(detections_to_record(detections))
        targets.append(targets_by_id[int(image_info["id"])])

    results = compute_map(predictions, targets, category_names)
    results["num_images"] = float(len(images))
    results["full_split"] = float(limit is None)
    return results
