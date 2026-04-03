from __future__ import annotations


def compute_review_queue_metrics(
    predictions: list[int],
    targets: list[int],
    confidences: list[float],
    auto_decision_threshold: float,
    defect_label: int = 1,
) -> dict[str, float]:
    total = len(targets)
    if total == 0:
        return {
            "auto_decision_rate": 0.0,
            "review_queue_rate": 0.0,
            "auto_decision_accuracy": 0.0,
            "auto_defect_precision": 0.0,
            "auto_defect_recall": 0.0,
            "overall_defect_recall": 0.0,
            "overall_false_negative_rate": 0.0,
        }

    auto_indices = [
        index for index, confidence in enumerate(confidences) if confidence >= auto_decision_threshold
    ]
    auto_index_set = set(auto_indices)
    review_indices = [index for index in range(total) if index not in auto_index_set]

    auto_correct = sum(predictions[index] == targets[index] for index in auto_indices)
    auto_defect_predictions = [index for index in auto_indices if predictions[index] == defect_label]
    auto_true_defects = [index for index in auto_defect_predictions if targets[index] == defect_label]
    all_defects = [index for index, target in enumerate(targets) if target == defect_label]
    overall_true_defects = [
        index for index, (prediction, target) in enumerate(zip(predictions, targets, strict=False))
        if prediction == defect_label and target == defect_label
    ]
    false_negatives = [
        index for index, (prediction, target) in enumerate(zip(predictions, targets, strict=False))
        if prediction != defect_label and target == defect_label
    ]

    return {
        "auto_decision_rate": len(auto_indices) / total,
        "review_queue_rate": len(review_indices) / total,
        "auto_decision_accuracy": 0.0 if not auto_indices else auto_correct / len(auto_indices),
        "auto_defect_precision": (
            0.0 if not auto_defect_predictions else len(auto_true_defects) / len(auto_defect_predictions)
        ),
        "auto_defect_recall": 0.0 if not all_defects else len(auto_true_defects) / len(all_defects),
        "overall_defect_recall": 0.0 if not all_defects else len(overall_true_defects) / len(all_defects),
        "overall_false_negative_rate": 0.0 if not all_defects else len(false_negatives) / len(all_defects),
    }
