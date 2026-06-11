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
