"""Business-impact (ROI) reporting CLI for the X-ray screening pipeline.

Consumes a ``predictions.json`` file (a list of
``{"score": float, "target": int}`` records — one per image, where ``score``
is the per-image max detection confidence and ``target`` is the
ground-truth label) and produces:

- ``report.json``: the full :func:`simclr_hpl.roi.evaluate_roi` output
  (review-queue metrics + ROI/safety figures).
- ``roi_summary.png``: a two-panel summary plot of the automation rate and
  the inspector-hours impact.
- An MLflow run logging the config thresholds/assumptions as params and the
  flattened metrics as scalars.

Until real detector inference results are available, ``--simulate-threat``
and ``--simulate-clean`` generate deterministic fake predictions via
:func:`simclr_hpl.roi.simulate_predictions`, so this CLI is a drop-in replacement
once a real ``predictions.json`` is produced by the detection step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from simclr_hpl.config import load_config
from simclr_hpl.roi import evaluate_roi, simulate_predictions
from simclr_hpl.tracking import ExperimentTracker
from simclr_hpl.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a business-impact (ROI) report from detection predictions."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/roi.yaml"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=(
            "Path to a predictions.json file (list of {'score': float, "
            "'target': int}). If omitted, predictions are simulated."
        ),
    )
    parser.add_argument(
        "--simulate-threat",
        type=int,
        default=None,
        help="Number of simulated threat images (used when --predictions is omitted).",
    )
    parser.add_argument(
        "--simulate-clean",
        type=int,
        default=None,
        help="Number of simulated clean images (used when --predictions is omitted).",
    )
    return parser.parse_args()


def _load_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _flatten_numeric(prefix: str, payload: dict) -> dict[str, float]:
    """Return the int/float (non-bool) entries of ``payload``, prefixed.

    Skips entries whose value is infinite, since MLflow cannot log ``inf``.
    """
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if value in (float("inf"), float("-inf")):
            continue
        metrics[f"{prefix}{key}"] = float(value)
    return metrics


def _plot_summary(report: dict, output_path: Path) -> None:
    review_metrics = report["review_metrics"]
    roi = report["roi"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    rates = {
        "Auto-decided": review_metrics.get("auto_decision_rate", 0.0) * 100,
        "Review queue": review_metrics.get("review_queue_rate", 0.0) * 100,
    }
    ax1.bar(rates.keys(), rates.values(), color=["#2ca02c", "#d62728"])
    ax1.set_ylabel("% of bags")
    ax1.set_ylim(0, 100)
    ax1.set_title("Automated decision vs. review queue")
    for index, value in enumerate(rates.values()):
        ax1.text(index, value + 1, f"{value:.1f}%", ha="center")

    hours = {
        "Baseline\n(manual review all)": roi.get("inspector_hours_baseline", 0.0),
        "Automated\n(review queue only)": roi.get("inspector_hours_automated", 0.0),
    }
    ax2.bar(hours.keys(), hours.values(), color=["#7f7f7f", "#1f77b4"])
    ax2.set_ylabel("Inspector hours")
    ax2.set_title("Inspector workload")
    for index, value in enumerate(hours.values()):
        ax2.text(index, value, f"{value:.2f}h", ha="center", va="bottom")

    fig.suptitle("ROI summary")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = ensure_dir(config["output_dir"])

    if args.predictions is not None:
        predictions = _load_predictions(args.predictions)
    else:
        n_threat = args.simulate_threat if args.simulate_threat is not None else 300
        n_clean = args.simulate_clean if args.simulate_clean is not None else 300
        predictions = simulate_predictions(n_threat=n_threat, n_clean=n_clean)

    scores = [item["score"] for item in predictions]
    targets = [item["target"] for item in predictions]

    detection_threshold = config["detection_threshold"]
    auto_decision_threshold = config["auto_decision_threshold"]
    assumptions = config["assumptions"]
    seconds_per_manual_review = assumptions["seconds_per_manual_review"]
    inspector_hourly_cost = assumptions["inspector_hourly_cost"]
    total_bags = assumptions.get("total_bags")

    report = evaluate_roi(
        scores,
        targets,
        detection_threshold=detection_threshold,
        auto_decision_threshold=auto_decision_threshold,
        total_bags=total_bags,
        seconds_per_manual_review=seconds_per_manual_review,
        inspector_hourly_cost=inspector_hourly_cost,
    )

    tracking_cfg = config.get("tracking", {})
    tracker = ExperimentTracker(
        enabled=tracking_cfg.get("enabled", True),
        tracking_uri=tracking_cfg.get("tracking_uri"),
        experiment=tracking_cfg.get("experiment"),
    )

    report_path = output_dir / "report.json"
    plot_path = output_dir / "roi_summary.png"

    with tracker.run(run_name="roi-report"):
        tracker.log_params(
            {
                "detection_threshold": detection_threshold,
                "auto_decision_threshold": auto_decision_threshold,
                "seconds_per_manual_review": seconds_per_manual_review,
                "inspector_hourly_cost": inspector_hourly_cost,
                "total_bags": total_bags,
            }
        )

        metrics = {
            **_flatten_numeric("review_", report["review_metrics"]),
            **_flatten_numeric("roi_", report["roi"]),
        }
        tracker.log_metrics(metrics)

        save_json(report_path, report)
        _plot_summary(report, plot_path)
        tracker.log_artifact(report_path)
        tracker.log_artifact(plot_path)

    review_metrics = report["review_metrics"]
    roi = report["roi"]
    print("ROI report")
    print(f"  total bags:           {roi['total_bags']}")
    print(f"  auto-decided:         {review_metrics['auto_decision_rate'] * 100:.1f}%")
    print(f"  review queue:         {review_metrics['review_queue_rate'] * 100:.1f}%")
    print(f"  inspector hours saved: {roi['inspector_hours_saved']:.2f} h")
    print(f"  cost saved:           ${roi['cost_saved']:.2f}")
    print(f"  missed-threat rate:   {roi['missed_threat_rate'] * 100:.2f}%")
    print(f"  auto defect recall:   {roi['auto_defect_recall'] * 100:.2f}%")
    print(f"Saved report to {report_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
