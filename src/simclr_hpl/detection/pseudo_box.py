"""Data utilities for few-label + pseudo-box semi-supervised detection.

Used by ``xray-pseudo-box-detect`` (:mod:`simclr_hpl.cli.pseudo_box_detect`):

1. :func:`split_labeled_unlabeled` splits a COCO ``train`` split's images into a
   small "labeled" subset (the few-label budget) and a large "unlabeled" subset.
2. :func:`build_train_split` / :func:`link_eval_splits` assemble a standalone
   ``dataset_dir`` (``train``/``valid``/``test``) for each pseudo-labeling round,
   symlinking images from the original ``coco_root`` rather than copying them
   (the dataset is ~1GB; symlinks keep each round's directory small).
3. :func:`generate_pseudo_annotations` runs a trained detector's
   ``model.predict(image, threshold=...)`` on unlabeled images and converts the
   returned ``supervision.Detections`` into COCO annotation dicts via
   :func:`detections_to_coco_annotations`.

None of these functions import ``rfdetr`` -- they operate on plain dicts/paths
and a duck-typed ``model`` with a ``.predict()`` method, so they're fully
testable on CPU with fakes (see ``tests/test_detection_pseudo_box.py``).
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

from simclr_hpl.detection.data import COCO_ANNOTATIONS_FILENAME as ANNOTATIONS_FILENAME


def load_coco(path: str | Path) -> dict:
    """Read a COCO ``_annotations.coco.json`` file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_coco(data: dict, path: str | Path) -> None:
    """Write a COCO annotations dict, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def split_labeled_unlabeled(
    coco: dict, label_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split ``coco["images"]`` into ``(labeled_images, unlabeled_images)``.

    The split is by image: an image's annotations either fully belong to the
    labeled subset or are dropped (treated as unlabeled, to be pseudo-labeled
    later). At least one image is always labeled, even if
    ``round(len(images) * label_fraction)`` would be ``0``.
    """
    images = list(coco["images"])
    rng = random.Random(seed)
    rng.shuffle(images)
    n_labeled = max(1, round(len(images) * label_fraction))
    return images[:n_labeled], images[n_labeled:]


def filter_annotations(annotations: list[dict], image_ids: set[int]) -> list[dict]:
    """Keep only annotations whose ``image_id`` is in ``image_ids``."""
    return [ann for ann in annotations if ann["image_id"] in image_ids]


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink ``src`` to ``dst``, falling back to a copy if symlinks fail."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def build_train_split(
    coco_root: str | Path,
    output_dir: str | Path,
    images: list[dict],
    annotations: list[dict],
    categories: list[dict],
) -> Path:
    """Write a ``train/`` COCO split (``_annotations.coco.json`` + images).

    Images are symlinked (falling back to a copy) from ``<coco_root>/train/``
    so that each pseudo-labeling round's directory doesn't duplicate the
    underlying dataset. Returns the written ``train/`` directory.
    """
    src_dir = Path(coco_root) / "train"
    dst_dir = Path(output_dir) / "train"
    for image in images:
        _link_or_copy(src_dir / image["file_name"], dst_dir / image["file_name"])
    save_coco(
        {"images": images, "annotations": annotations, "categories": categories},
        dst_dir / ANNOTATIONS_FILENAME,
    )
    return dst_dir


def link_eval_splits(
    coco_root: str | Path,
    output_dir: str | Path,
    splits: tuple[str, ...] = ("valid", "test"),
) -> None:
    """Symlink ``valid``/``test`` split directories unchanged into ``output_dir``.

    rfdetr expects a ``dataset_dir`` containing ``train``/``valid``/``test``
    subdirectories, each with their own ``_annotations.coco.json``. The eval
    splits are identical across every pseudo-labeling round, so they're
    symlinked once instead of copied per round. Missing source splits or
    already-linked destinations are silently skipped.
    """
    for split in splits:
        src = Path(coco_root) / split
        dst = Path(output_dir) / split
        if not src.exists() or dst.exists() or dst.is_symlink():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src.resolve(), dst)


def detections_to_coco_annotations(
    detections: Any,
    image_id: int,
    categories: list[dict],
    next_id: int,
    threshold: float,
) -> tuple[list[dict], list[float]]:
    """Convert an ``sv.Detections``-like object into COCO annotation dicts.

    Only detections with ``confidence >= threshold`` are kept (a safety net;
    ``model.predict(image, threshold=...)`` is also expected to filter).
    ``class_id`` is assumed to index ``categories`` positionally, i.e.
    ``categories[class_id]`` gives the COCO category for that detection --
    see the "Known limitations" section of
    ``docs/explainers/pseudo_box_semi_supervised_detection.md`` for why this
    assumption needs verifying against a real rfdetr model.
    """
    annotations: list[dict] = []
    confidences: list[float] = []
    xyxy = detections.xyxy
    confidence = detections.confidence
    class_id = detections.class_id
    for i in range(len(confidence)):
        score = float(confidence[i])
        if score < threshold:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        width, height = x2 - x1, y2 - y1
        category = categories[int(class_id[i])]
        annotations.append(
            {
                "id": next_id + len(annotations),
                "image_id": image_id,
                "category_id": category["id"],
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
        )
        confidences.append(score)
    return annotations, confidences


def generate_pseudo_annotations(
    model: Any,
    images: list[dict],
    images_dir: str | Path,
    categories: list[dict],
    threshold: float,
    next_id: int,
) -> tuple[list[dict], list[dict], list[float]]:
    """Run ``model.predict`` on each unlabeled image and keep confident boxes.

    Returns ``(pseudo_annotations, pseudo_images, confidences)``:

    - ``pseudo_annotations``: COCO annotation dicts for accepted boxes
    - ``pseudo_images``: the subset of ``images`` that received >=1 accepted box
      (images with zero confident detections stay in the unlabeled pool)
    - ``confidences``: confidence of every accepted box, for reporting
    """
    from PIL import Image  # noqa: PLC0415 - keep PIL import local to this call

    images_dir = Path(images_dir)
    pseudo_annotations: list[dict] = []
    pseudo_images: list[dict] = []
    confidences: list[float] = []
    for image_meta in images:
        image = Image.open(images_dir / image_meta["file_name"]).convert("RGB")
        detections = model.predict(image, threshold=threshold)
        anns, confs = detections_to_coco_annotations(
            detections,
            image_id=image_meta["id"],
            categories=categories,
            next_id=next_id + len(pseudo_annotations),
            threshold=threshold,
        )
        if anns:
            pseudo_annotations.extend(anns)
            pseudo_images.append(image_meta)
            confidences.extend(confs)
    return pseudo_annotations, pseudo_images, confidences
