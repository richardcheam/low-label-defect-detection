"""ROI / business-impact reporting for the X-ray defect-detection pipeline.

This module translates per-image detection scores into image-level decisions
and quantifies the operational value of automating bag-screening decisions:
bags the model auto-clears (confidently clean) or auto-flags (confidently a
threat) no longer require a manual inspector review, freeing up inspector
time and reducing cost.

IMPORTANT: the ROI figures produced here (inspector hours saved, cost saved,
throughput multiplier) MUST always be reported alongside the safety
guardrail metrics (``missed_threat_rate`` / ``auto_defect_recall``). Time
savings are only acceptable if the rate of threats missed by the automated
decisions remains within an operationally tolerable bound. Reporting ROI
without the guardrail is misleading and must be avoided.

Typical usage combines :func:`image_level_decisions` with
``simclr_hpl.business.compute_review_queue_metrics`` to obtain the review
queue metrics, and then :func:`compute_roi` to translate those rates into
inspector-hours and cost figures.
"""

from __future__ import annotations

import random

from simclr_hpl.business import compute_review_queue_metrics


def image_level_decisions(
    image_scores: list[float],
    detection_threshold: float = 0.5,
) -> tuple[list[int], list[float]]:
    """Convert per-image max detection scores into decisions and confidences.

    Args:
        image_scores: Per-image maximum detection score (e.g. the highest
            confidence of any detected defect region in the image).
        detection_threshold: Score at or above which an image is predicted
            to contain a threat (label ``1``).

    Returns:
        A tuple ``(predictions, confidences)`` where ``predictions[i]`` is
        ``1`` if ``image_scores[i] >= detection_threshold`` else ``0``, and
        ``confidences[i]`` is the model's confidence in that decision:
        ``image_scores[i]`` for threat predictions, or
        ``1.0 - image_scores[i]`` for clean predictions. These outputs are
        suitable inputs to ``business.compute_review_queue_metrics``.
    """
    predictions: list[int] = []
    confidences: list[float] = []
    for score in image_scores:
        if score >= detection_threshold:
            predictions.append(1)
            confidences.append(score)
        else:
            predictions.append(0)
            confidences.append(1.0 - score)
    return predictions, confidences


def compute_roi(
    review_metrics: dict[str, float],
    *,
    total_bags: int,
    seconds_per_manual_review: float = 12.0,
    inspector_hourly_cost: float = 30.0,
) -> dict[str, float]:
    """Compute inspector-time and cost savings from automated bag decisions.

    Bags that the model auto-decides (auto-clears or auto-flags with high
    confidence) no longer need a manual inspector review. This function
    converts the ``auto_decision_rate`` / ``review_queue_rate`` produced by
    ``business.compute_review_queue_metrics`` into inspector-hours saved and
    the corresponding cost savings, while also surfacing the safety
    guardrail (``missed_threat_rate`` / ``auto_defect_recall``) that must be
    reported alongside any ROI figure.

    Args:
        review_metrics: Output of ``business.compute_review_queue_metrics``,
            expected to contain ``auto_decision_rate``, ``review_queue_rate``,
            ``overall_false_negative_rate``, and ``auto_defect_recall``.
        total_bags: Total number of bags screened. Must be non-negative.
        seconds_per_manual_review: Average time an inspector spends manually
            reviewing a single bag.
        inspector_hourly_cost: Fully-loaded hourly cost of an inspector.

    Returns:
        A flat dict of ROI and safety figures:

        - ``total_bags``, ``auto_decision_rate``, ``review_queue_rate``:
          echoed inputs.
        - ``auto_cleared_bags``, ``review_queue_bags``: bag counts.
        - ``inspector_hours_baseline``: hours required if every bag were
          manually reviewed.
        - ``inspector_hours_automated``: hours required to manually review
          only the review-queue bags.
        - ``inspector_hours_saved``: hours freed up by automation.
        - ``cost_saved``: dollar value of ``inspector_hours_saved``.
        - ``throughput_multiplier``: ``inspector_hours_baseline /
          inspector_hours_automated`` (``inf`` if the automated workload is
          zero, i.e. nothing reaches the review queue).
        - ``missed_threat_rate``, ``auto_defect_recall``: safety guardrail
          metrics that MUST be reported alongside the ROI figures above.

    Raises:
        ValueError: If ``total_bags`` is negative.
    """
    if total_bags < 0:
        msg = f"total_bags must be non-negative, got {total_bags}"
        raise ValueError(msg)

    auto_decision_rate = review_metrics.get("auto_decision_rate", 0.0)
    review_queue_rate = review_metrics.get("review_queue_rate", 0.0)

    auto_cleared_bags = round(auto_decision_rate * total_bags)
    review_queue_bags = total_bags - auto_cleared_bags

    inspector_hours_baseline = total_bags * seconds_per_manual_review / 3600
    inspector_hours_automated = review_queue_bags * seconds_per_manual_review / 3600
    inspector_hours_saved = inspector_hours_baseline - inspector_hours_automated
    cost_saved = inspector_hours_saved * inspector_hourly_cost

    if inspector_hours_automated == 0:
        throughput_multiplier = float("inf")
    else:
        throughput_multiplier = inspector_hours_baseline / inspector_hours_automated

    return {
        "total_bags": total_bags,
        "auto_decision_rate": auto_decision_rate,
        "review_queue_rate": review_queue_rate,
        "auto_cleared_bags": auto_cleared_bags,
        "review_queue_bags": review_queue_bags,
        "inspector_hours_baseline": inspector_hours_baseline,
        "inspector_hours_automated": inspector_hours_automated,
        "inspector_hours_saved": inspector_hours_saved,
        "cost_saved": cost_saved,
        "throughput_multiplier": throughput_multiplier,
        "missed_threat_rate": review_metrics.get("overall_false_negative_rate", 0.0),
        "auto_defect_recall": review_metrics.get("auto_defect_recall", 0.0),
    }


def evaluate_roi(
    scores: list[float],
    targets: list[int],
    *,
    detection_threshold: float = 0.5,
    auto_decision_threshold: float = 0.9,
    total_bags: int | None = None,
    seconds_per_manual_review: float = 12.0,
    inspector_hourly_cost: float = 30.0,
) -> dict[str, dict[str, float]]:
    """Run the full score -> decision -> review-queue -> ROI pipeline.

    This is the single entry point that chains :func:`image_level_decisions`,
    ``business.compute_review_queue_metrics``, and :func:`compute_roi`,
    so callers (e.g. the ``xray-roi`` CLI) only need a list of per-image
    scores and ground-truth targets to obtain a full business-impact report.

    Args:
        scores: Per-image maximum detection score (0..1).
        targets: Per-image ground-truth label (``1`` = threat, ``0`` = clean).
        detection_threshold: Score threshold used to derive predictions, see
            :func:`image_level_decisions`.
        auto_decision_threshold: Confidence threshold above which a decision
            is considered automatable (no manual review), see
            ``business.compute_review_queue_metrics``.
        total_bags: Total number of bags screened. Defaults to
            ``len(scores)`` when ``None``.
        seconds_per_manual_review: Average time an inspector spends manually
            reviewing a single bag.
        inspector_hourly_cost: Fully-loaded hourly cost of an inspector.

    Returns:
        A dict with two keys: ``"review_metrics"`` (the output of
        ``business.compute_review_queue_metrics``) and ``"roi"`` (the output
        of :func:`compute_roi`, including the safety guardrail metrics).
    """
    predictions, confidences = image_level_decisions(scores, detection_threshold)
    review_metrics = compute_review_queue_metrics(
        predictions, targets, confidences, auto_decision_threshold
    )
    roi = compute_roi(
        review_metrics,
        total_bags=total_bags if total_bags is not None else len(scores),
        seconds_per_manual_review=seconds_per_manual_review,
        inspector_hourly_cost=inspector_hourly_cost,
    )
    return {"review_metrics": review_metrics, "roi": roi}


def simulate_predictions(
    n_threat: int,
    n_clean: int,
    seed: int = 42,
    skill: float = 0.85,
) -> list[dict[str, float | int]]:
    """Generate deterministic, separable fake per-image detection scores.

    Useful for exercising the ROI reporting pipeline before real detector
    inference results are available. The output matches the predictions
    contract consumed by :func:`evaluate_roi`: a list of
    ``{"score": float, "target": int}`` dicts.

    Args:
        n_threat: Number of simulated threat (``target == 1``) images.
        n_clean: Number of simulated clean (``target == 0``) images.
        seed: Seed for the deterministic ``random.Random`` generator.
        skill: Detector "skill" in ``[0, 1]``. Higher values push threat
            scores closer to 1 and clean scores closer to 0, increasing the
            separation between the two distributions.

    Returns:
        A shuffled list of ``{"score": float, "target": int}`` dicts of
        length ``n_threat + n_clean``, with scores clipped to ``[0, 1]``.
    """
    rng = random.Random(seed)

    threat_mean = 0.5 + 0.4 * skill
    clean_mean = 0.5 - 0.4 * skill
    spread = 0.15

    predictions: list[dict[str, float | int]] = []
    for _ in range(n_threat):
        score = max(0.0, min(1.0, rng.gauss(threat_mean, spread)))
        predictions.append({"score": score, "target": 1})
    for _ in range(n_clean):
        score = max(0.0, min(1.0, rng.gauss(clean_mean, spread)))
        predictions.append({"score": score, "target": 0})

    rng.shuffle(predictions)
    return predictions
