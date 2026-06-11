from __future__ import annotations

import math

from simclr_hpl.roi import compute_roi, image_level_decisions


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
