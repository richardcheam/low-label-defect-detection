"""RF-DETR detection training wrapper.

This module wraps the (heavy, CUDA/torch-dependent) ``rfdetr`` package so that
the rest of the codebase — and the test suite — can import and exercise the
training orchestration logic without ``rfdetr`` being installed.

``rfdetr`` is declared as an optional extra (``pip install '.[detection]'`` /
``uv sync --extra detection``) and is imported lazily *inside*
:func:`train_detector`, so importing this module is always safe in a minimal
environment, and tests can inject a fake ``rfdetr`` module via
``sys.modules`` before calling :func:`train_detector`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from simclr_hpl.tracking import ExperimentTracker
from simclr_hpl.utils import ensure_dir, save_json

# Maps the friendly `model.variant` config value to the rfdetr model class name.
_VARIANT_TO_CLASS_NAME = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "base": "RFDETRBase",
}


def resolve_model_class(variant: str) -> Any:
    """Lazily import ``rfdetr`` and return the model class for ``variant``.

    Shared between :mod:`simclr_hpl.detection.train` and
    :mod:`simclr_hpl.detection.predict` so both modules agree on the
    ``model.variant`` -> ``rfdetr`` class mapping.
    """
    class_name = _VARIANT_TO_CLASS_NAME.get(variant)
    if class_name is None:
        msg = (
            f"Unknown model.variant: {variant!r}. "
            f"Expected one of {sorted(_VARIANT_TO_CLASS_NAME)}."
        )
        raise ValueError(msg)

    import rfdetr  # noqa: PLC0415 - intentional lazy/optional import

    return getattr(rfdetr, class_name)


def _read_numeric_metrics(path: Path) -> dict[str, float]:
    """Read a JSON file and return its top-level ``int``/``float`` entries."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {
        key: float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _maybe_log_results_metrics(tracker: ExperimentTracker, output_dir: Path) -> None:
    """Best-effort: if rfdetr wrote a metrics/results JSON, log its scalars.

    Absence or any parsing error must not crash the run.
    """
    for candidate in ("results.json", "metrics.json"):
        results_path = output_dir / candidate
        if not results_path.exists():
            continue
        try:
            metrics = _read_numeric_metrics(results_path)
            if metrics:
                tracker.log_metrics(metrics)
        except Exception:  # noqa: BLE001 - best-effort, must never crash training
            pass


def train_detector(config: dict[str, Any], tracker: ExperimentTracker | None = None) -> dict:
    """Train an RF-DETR detector from a config dict.

    Args:
        config: Parsed config (see ``configs/sixray_detection.yaml``). Expects
            ``data.coco_root``, ``model.variant``, ``output_dir``,
            ``train.{epochs,batch_size,learning_rate}``, and ``tracking``.
        tracker: Optional pre-built :class:`ExperimentTracker`. If ``None``, a
            tracker is created from ``config["tracking"]``.

    Returns:
        A summary dict (also written to ``<output_dir>/train_summary.json``)
        containing the echoed ``config`` and the resolved ``output_dir``.
    """
    model_cls = resolve_model_class(config["model"]["variant"])

    if tracker is None:
        tracking_cfg = config.get("tracking", {})
        tracker = ExperimentTracker(
            enabled=tracking_cfg.get("enabled", True),
            tracking_uri=tracking_cfg.get("tracking_uri"),
            experiment=tracking_cfg.get("experiment"),
        )

    output_dir = ensure_dir(config["output_dir"])
    train_cfg = config["train"]
    coco_root = config["data"]["coco_root"]

    devices = train_cfg.get("devices", 1)
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)

    with tracker.run(run_name=f"rfdetr-{config['model']['variant']}"):
        tracker.log_params(
            {
                "variant": config["model"]["variant"],
                "epochs": train_cfg["epochs"],
                "batch_size": train_cfg["batch_size"],
                "learning_rate": train_cfg["learning_rate"],
                "devices": devices,
                "grad_accum_steps": grad_accum_steps,
                "coco_root": coco_root,
            }
        )

        model = model_cls()
        model.train(
            dataset_dir=coco_root,
            epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"],
            lr=train_cfg["learning_rate"],
            output_dir=str(output_dir),
            devices=devices,
            grad_accum_steps=grad_accum_steps,
        )

        tracker.log_artifact(output_dir)
        _maybe_log_results_metrics(tracker, output_dir)

    summary = {"config": config, "output_dir": str(output_dir)}
    save_json(output_dir / "train_summary.json", summary)
    return summary
