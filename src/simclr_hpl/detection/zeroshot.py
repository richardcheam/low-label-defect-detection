"""Zero-shot threat detection via Grounding DINO.

Lazily imports ``transformers`` (and ``torch``) inside functions so this module
is importable in CPU/CI environments without the 680MB model weights downloaded.
Tests inject fake ``transformers`` via ``sys.modules`` (same pattern as the
``fake_rfdetr`` fixture in ``tests/test_detection_predict.py``).

Grounding DINO uses open-vocabulary text prompts — e.g. ``"gun. knife. pliers."``
— to detect objects without task-specific training. The model is loaded from
HuggingFace via ``transformers.AutoModelForZeroShotObjectDetection``.

Install:
    uv sync --extra zeroshot
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any


def _list_images(directory: str | Path) -> list[Path]:
    """Return sorted image paths under ``directory``."""
    return sorted(
        p
        for p in Path(directory).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )


def _max_score(result: dict[str, Any]) -> float:
    """Return the max detection confidence from a Grounding DINO result dict."""
    scores = result.get("scores", [])
    if not hasattr(scores, "__len__") or len(scores) == 0:
        return 0.0
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    return float(max(scores))


def _load_model(model_id: str) -> tuple[Any, Any]:
    """Lazily import ``transformers`` and load the Grounding DINO processor+model."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: PLC0415

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.eval()
    return processor, model


def score_images(
    processor: Any,
    model: Any,
    image_paths: list[Path],
    text_labels: list[str],
    threshold: float,
) -> list[float]:
    """Run Grounding DINO on each image and return the max-confidence detection score.

    ``threshold`` is passed to
    ``processor.post_process_grounded_object_detection()`` — boxes below it are
    discarded. Images with no detections above threshold score ``0.0``.
    """
    from PIL import Image  # noqa: PLC0415

    try:
        import torch  # noqa: PLC0415
        _no_grad: Any = torch.no_grad
    except ImportError:
        import contextlib  # noqa: PLC0415
        _no_grad = contextlib.nullcontext

    text_prompt = ". ".join(text_labels) + "."
    scores: list[float] = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, text=text_prompt, return_tensors="pt")
        with _no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=[image.size[::-1]],
        )
        scores.append(_max_score(results[0]))
    return scores


def build_zeroshot_predictions(
    threat_dir: str | Path,
    clean_dir: str | Path | None,
    *,
    model_id: str,
    text_labels: list[str],
    threshold: float = 0.3,
) -> list[dict[str, float | int]]:
    """Run Grounding DINO zero-shot on ``threat_dir`` (and optionally ``clean_dir``).

    Returns ``[{"score": float, "target": int}]`` compatible with ``xray-roi``:
    threat images get ``target=1``, clean images ``target=0``.

    Args:
        threat_dir: Directory of threat/positive images.
        clean_dir: Directory of clean/negative images. ``None`` or a missing path
            emits a :class:`UserWarning` and skips clean images (matching the
            behaviour of ``detection.predict.build_predictions``).
        model_id: HuggingFace model ID, e.g. ``"IDEA-Research/grounding-dino-base"``.
        text_labels: List of threat class names. Joined as ``"gun. knife. pliers."``
        threshold: Minimum detection confidence to accept a box.
    """
    processor, model = _load_model(model_id)

    threat_paths = _list_images(threat_dir)
    threat_scores = score_images(processor, model, threat_paths, text_labels, threshold)
    predictions: list[dict[str, float | int]] = [
        {"score": s, "target": 1} for s in threat_scores
    ]

    if clean_dir is None:
        warnings.warn(
            "No clean_dir provided; predictions contain only threat images.",
            UserWarning,
            stacklevel=2,
        )
    else:
        clean_path = Path(clean_dir)
        if not clean_path.exists():
            warnings.warn(
                f"clean_dir {str(clean_dir)!r} does not exist; skipping clean images.",
                UserWarning,
                stacklevel=2,
            )
        else:
            clean_paths = _list_images(clean_path)
            clean_scores = score_images(processor, model, clean_paths, text_labels, threshold)
            predictions.extend({"score": s, "target": 0} for s in clean_scores)

    return predictions
