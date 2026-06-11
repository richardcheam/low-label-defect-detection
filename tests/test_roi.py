from __future__ import annotations

import math

from simclr_hpl.roi import compute_roi, evaluate_roi, image_level_decisions, simulate_predictions


def test_image_level_decisions_threshold_and_confidence():
    # per-image max detection score; threshold 0.5
    scores = [0.9, 0.4, 0.51, 0.0]
    preds, confs = image_level_decisions(scores, detection_threshold=0.5)
    assert preds == [1, 0, 1, 0]
    # threat -> confidence is the score; clean -> confidence is (1 - score)
    assert confs == [0.9, 0.6, 0.51, 1.0]


def test_compute_roi_basic():
    review_metrics = {
        "auto_decision_rate": 0.7,
        "review_queue_rate": 0.3,
        "overall_false_negative_rate": 0.01,
        "auto_defect_recall": 0.95,
    }
    roi = compute_roi(
        review_metrics,
        total_bags=1000,
        seconds_per_manual_review=12.0,
        inspector_hourly_cost=30.0,
    )
    assert roi["auto_cleared_bags"] == 700
    assert roi["review_queue_bags"] == 300
    # baseline = 1000*12/3600 h ; automated = 300*12/3600 h
    assert math.isclose(roi["inspector_hours_baseline"], 1000 * 12 / 3600)
    assert math.isclose(roi["inspector_hours_automated"], 300 * 12 / 3600)
    assert math.isclose(roi["inspector_hours_saved"], 700 * 12 / 3600)
    assert math.isclose(roi["cost_saved"], (700 * 12 / 3600) * 30.0)
    # safety guardrail surfaced
    assert roi["missed_threat_rate"] == 0.01
    assert roi["auto_defect_recall"] == 0.95


def test_evaluate_roi_chains_decision_to_roi():
    # 3 threats (high scores) + 2 clean (low scores)
    scores = [0.95, 0.92, 0.88, 0.10, 0.20]
    targets = [1, 1, 1, 0, 0]
    out = evaluate_roi(
        scores, targets,
        detection_threshold=0.5, auto_decision_threshold=0.9,
        seconds_per_manual_review=12.0, inspector_hourly_cost=30.0,
    )
    assert "review_metrics" in out and "roi" in out
    assert out["roi"]["total_bags"] == 5
    # missed-threat safety guardrail is always present
    assert "missed_threat_rate" in out["roi"]


def test_simulate_predictions_is_deterministic_and_separable():
    a = simulate_predictions(n_threat=50, n_clean=50, seed=7)
    b = simulate_predictions(n_threat=50, n_clean=50, seed=7)
    assert a == b  # deterministic
    threat_mean = sum(p["score"] for p in a if p["target"] == 1) / 50
    clean_mean = sum(p["score"] for p in a if p["target"] == 0) / 50
    assert threat_mean > clean_mean  # threats score higher on average
