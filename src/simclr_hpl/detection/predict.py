"""RF-DETR detection inference -> ``predictions.json`` bridge.

This module produces the ``predictions.json`` consumed by the ``xray-roi``
CLI (:mod:`simclr_hpl.cli.roi_report`): a list of
``{"score": float, "target": int}`` records, where ``score`` is the
per-image *max* detection confidence (``0.0`` if nothing was detected) and
``target`` is the ground-truth label (``1`` = threat, ``0`` = clean).

As with :mod:`simclr_hpl.detection.train`, ``rfdetr`` is an optional,
heavy CUDA/torch-dependent extra. It is imported lazily *inside*
:func:`_load_model`, so importing this module is always safe in a minimal
environment, and tests can inject a fake ``rfdetr`` module via
``sys.modules`` before calling :func:`build_predictions`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from PIL import Image

from simclr_hpl.detection.train import resolve_model_class

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def max_confidence(detections: Any) -> float:
    """Return the max detection confidence, or ``0.0`` if none were found."""
    confidence = detections.confidence
    if confidence is None or len(confidence) == 0:
        return 0.0
    return float(confidence.max())


def _load_model(variant: str, checkpoint: str | Path) -> Any:
    """Lazily import ``rfdetr`` and construct the model for ``variant``.

    Uses the documented ``pretrain_weights=`` constructor kwarg to load a
    trained checkpoint.
    """
    model_cls = resolve_model_class(variant)
    return model_cls(pretrain_weights=str(checkpoint))


def _list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def score_images(model: Any, image_paths: list[Path], threshold: float = 0.5) -> list[float]:
    """Run ``model.predict`` on each image and return its max confidence."""
    scores = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        detections = model.predict(image, threshold=threshold)
        scores.append(max_confidence(detections))
    return scores


def build_predictions(
    threat_dir: str | Path,
    clean_dir: str | Path | None,
    *,
    variant: str,
    checkpoint: str | Path,
    threshold: float = 0.5,
) -> list[dict]:
    """Score images in ``threat_dir`` (target 1) and ``clean_dir`` (target 0).

    ``clean_dir`` may be ``None`` or missing, in which case only threat
    images are scored and a warning is emitted (the ROI loop needs negative
    examples to compute a meaningful review-queue rate).
    """
    threat_paths = _list_images(threat_dir)

    clean_paths: list[Path] = []
    if clean_dir is not None and Path(clean_dir).is_dir():
        clean_paths = _list_images(clean_dir)
    else:
        warnings.warn(
            "No clean_dir provided (or it does not exist); predictions will "
            "contain only threat images. The ROI loop needs negative "
            "(clean) examples to compute a meaningful review-queue rate.",
            stacklevel=2,
        )

    model = _load_model(variant, checkpoint)

    threat_scores = score_images(model, threat_paths, threshold=threshold)
    clean_scores = score_images(model, clean_paths, threshold=threshold)

    predictions = [{"score": score, "target": 1} for score in threat_scores]
    predictions += [{"score": score, "target": 0} for score in clean_scores]
    return predictions
