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


_METRICS_CSV_VAL_KEYS = frozenset(
    {"val/mAP_50", "val/mAP_50_95", "val/mAP_75", "val/mAR", "val/precision", "val/recall"}
)


def _read_metrics_csv(path: Path) -> dict[str, float]:
    """Read the last full-epoch val metrics from rfdetr's ``metrics.csv``.

    rfdetr writes one row per training step; val metrics appear only on rows
    where the val evaluation ran (end of epoch). This function scans all rows
    and keeps only the last non-empty value per val key, so the result reflects
    the final training epoch.
    """
    import csv

    result: dict[str, float] = {}
    try:
        with path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                update = {
                    k: float(v)
                    for k, v in row.items()
                    if k in _METRICS_CSV_VAL_KEYS and v.strip()
                }
                if update:
                    result.update(update)
    except Exception:  # noqa: BLE001 - best-effort, must never crash training
        pass
    return result


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


def _patch_map_metric_sync_on_compute() -> None:
    """Work around a torchmetrics DDP deadlock in rfdetr's validation callback.

    ``rfdetr``'s ``COCOEvalCallback`` builds ``torchmetrics.detection.MeanAveragePrecision``
    with the torchmetrics default ``sync_on_compute=True``. Under multi-GPU DDP,
    ``compute()`` then calls ``metric.sync()``, which does a cross-rank ``all_gather``
    that can deadlock and hang until the NCCL watchdog kills the job (see
    Lightning-AI/torchmetrics#3199 and #626). Patch the ``MeanAveragePrecision`` name
    in rfdetr's callback module so it always constructs with ``sync_on_compute=False``
    (each rank then reports mAP from its own validation shard, avoiding the hang).

    Best-effort: a no-op if ``rfdetr``/``torchmetrics`` aren't importable (e.g. in the
    CPU test environment, where ``rfdetr`` is a minimal stub).
    """
    try:
        from functools import partial

        import rfdetr.training.callbacks.coco_eval as coco_eval_module
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError:
        return

    coco_eval_module.MeanAveragePrecision = partial(MeanAveragePrecision, sync_on_compute=False)


def run_training_round(
    model_cls: Any,
    dataset_dir: str | Path,
    output_dir: str | Path,
    train_cfg: dict[str, Any],
) -> tuple[Any, dict[str, float]]:
    """Train a fresh model instance on ``dataset_dir`` for one round.

    Used by ``xray-pseudo-box-detect`` (:mod:`simclr_hpl.cli.pseudo_box_detect`)
    to retrain from scratch each pseudo-labeling round on a growing dataset.
    Unlike :func:`train_detector`, this has no tracking/seeding of its own —
    the caller owns the :class:`ExperimentTracker` run and seeding.

    Returns ``(model, metrics)``: ``model`` is the trained instance (reused
    immediately afterwards for pseudo-label generation via ``model.predict``),
    and ``metrics`` is whatever numeric scalars :func:`_read_numeric_metrics`
    finds in ``results.json``/``metrics.json`` under ``output_dir`` (``{}`` if
    neither file exists).
    """
    output_dir = ensure_dir(output_dir)
    devices = train_cfg.get("devices", 1)
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
    strategy = train_cfg.get("strategy", "auto")

    model = model_cls()
    model.train(
        dataset_dir=str(dataset_dir),
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        lr=train_cfg["learning_rate"],
        output_dir=str(output_dir),
        devices=devices,
        grad_accum_steps=grad_accum_steps,
        strategy=strategy,
    )

    metrics: dict[str, float] = {}
    for candidate in ("results.json", "metrics.json"):
        results_path = output_dir / candidate
        if results_path.exists():
            metrics = _read_numeric_metrics(results_path)
            break
    if not metrics:
        csv_path = output_dir / "metrics.csv"
        if csv_path.exists():
            metrics = _read_metrics_csv(csv_path)
    return model, metrics


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
    strategy = train_cfg.get("strategy", "auto")

    with tracker.run(run_name=f"rfdetr-{config['model']['variant']}"):
        tracker.log_params(
            {
                "variant": config["model"]["variant"],
                "epochs": train_cfg["epochs"],
                "batch_size": train_cfg["batch_size"],
                "learning_rate": train_cfg["learning_rate"],
                "devices": devices,
                "grad_accum_steps": grad_accum_steps,
                "strategy": strategy,
                "coco_root": coco_root,
            }
        )

        _patch_map_metric_sync_on_compute()

        model = model_cls()
        model.train(
            dataset_dir=coco_root,
            epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"],
            lr=train_cfg["learning_rate"],
            output_dir=str(output_dir),
            devices=devices,
            grad_accum_steps=grad_accum_steps,
            strategy=strategy,
        )

        tracker.log_artifact(output_dir)
        _maybe_log_results_metrics(tracker, output_dir)

    summary = {"config": config, "output_dir": str(output_dir)}
    save_json(output_dir / "train_summary.json", summary)
    return summary
