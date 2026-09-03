import json

import pytest
from PIL import Image

from simclr_hpl.detection import evaluate as evaluate_module
from simclr_hpl.detection.evaluate import (
    compute_map,
    detections_to_record,
    evaluate_split,
    load_coco_ground_truth,
    xywh_to_xyxy,
)


class FakeDetections:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id


def _box(x, y, size=10.0):
    return [x, y, x + size, y + size]


def test_xywh_to_xyxy_converts_width_height_to_corners():
    assert xywh_to_xyxy([10, 20, 30, 40]) == [10.0, 20.0, 40.0, 60.0]


def test_load_ground_truth_keeps_images_without_annotations(tmp_path):
    """Unannotated images are real negatives; dropping them inflates precision."""
    annotations = tmp_path / "ann.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
                "categories": [{"id": 0, "name": "Gun"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10]}],
            }
        )
    )

    images, targets, names = load_coco_ground_truth(annotations)

    assert len(images) == 2
    assert names == {0: "Gun"}
    assert targets[1]["boxes"] == [[0.0, 0.0, 10.0, 10.0]]
    assert targets[2] == {"boxes": [], "labels": []}


def test_load_ground_truth_skips_crowd_annotations(tmp_path):
    annotations = tmp_path / "ann.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.jpg"}],
                "categories": [{"id": 0, "name": "Gun"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10]},
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 0,
                        "bbox": [5, 5, 10, 10],
                        "iscrowd": 1,
                    },
                ],
            }
        )
    )

    _, targets, _ = load_coco_ground_truth(annotations)

    assert len(targets[1]["boxes"]) == 1


def test_detections_to_record_handles_no_detections():
    empty = detections_to_record(FakeDetections(xyxy=[], confidence=[], class_id=[]))
    assert empty == {"boxes": [], "scores": [], "labels": []}


def test_detections_to_record_converts_arrays():
    record = detections_to_record(
        FakeDetections(xyxy=[[1, 2, 3, 4]], confidence=[0.9], class_id=[2])
    )
    assert record == {"boxes": [[1.0, 2.0, 3.0, 4.0]], "scores": [0.9], "labels": [2]}


def test_perfect_predictions_score_one():
    targets = [{"boxes": [_box(0, 0)], "labels": [0]}]
    predictions = [{"boxes": [_box(0, 0)], "scores": [0.9], "labels": [0]}]

    results = compute_map(predictions, targets, {0: "Gun"})

    assert results["mAP_50"] == pytest.approx(1.0)
    assert results["mAP_50_95"] == pytest.approx(1.0)
    assert results["AP_50_95/Gun"] == pytest.approx(1.0)


def test_missed_detections_lower_recall():
    targets = [{"boxes": [_box(0, 0)], "labels": [0]}]
    predictions = [{"boxes": [], "scores": [], "labels": []}]

    results = compute_map(predictions, targets, {0: "Gun"})

    assert results["mAP_50"] <= 0.0


def test_per_class_names_are_resolved():
    targets = [{"boxes": [_box(0, 0), _box(50, 50)], "labels": [0, 1]}]
    predictions = [{"boxes": [_box(0, 0), _box(50, 50)], "scores": [0.9, 0.8], "labels": [0, 1]}]

    results = compute_map(predictions, targets, {0: "Gun", 1: "Knife"})

    assert "AP_50_95/Gun" in results
    assert "AP_50_95/Knife" in results


def test_sharded_map_differs_from_full_split_map():
    """The bug this module exists to fix.

    Under DDP each rank computes mAP over its own shard. mAP is not a mean over
    samples -- AP is read off a precision/recall curve ranked across the whole
    split -- so a per-shard value is not an estimate of the full-split value and
    the two must not be used interchangeably.

    Here a false positive lands only in the second shard. A rank holding the
    first shard reports a perfect score; the full split does not.

    The false positive is scored *above* the true positives on purpose. COCO
    interpolated AP takes the maximum precision at each recall level, so a
    false positive ranked below every true positive leaves AP untouched -- only
    one that outranks a true positive moves the curve.
    """
    targets = [
        {"boxes": [_box(0, 0)], "labels": [0]},
        {"boxes": [_box(0, 0)], "labels": [0]},
    ]
    predictions = [
        {"boxes": [_box(0, 0)], "scores": [0.9], "labels": [0]},
        {"boxes": [_box(0, 0), _box(200, 200)], "scores": [0.9, 0.95], "labels": [0, 0]},
    ]

    shard_zero = compute_map(predictions[:1], targets[:1])
    full_split = compute_map(predictions, targets)

    assert shard_zero["mAP_50"] == pytest.approx(1.0)
    assert full_split["mAP_50"] < shard_zero["mAP_50"]


def _tiny_split(tmp_path, num_images=3):
    """Write a minimal COCO split with real image files on disk."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    images, annotations = [], []
    for index in range(num_images):
        name = f"img{index}.png"
        Image.new("RGB", (64, 64)).save(images_dir / name)
        images.append({"id": index, "file_name": name})
        annotations.append(
            {"id": index, "image_id": index, "category_id": 0, "bbox": [0, 0, 10, 10]}
        )
    annotations_path = tmp_path / "ann.json"
    annotations_path.write_text(
        json.dumps(
            {
                "images": images,
                "categories": [{"id": 0, "name": "Gun"}],
                "annotations": annotations,
            }
        )
    )
    return annotations_path, images_dir


class PerfectModel:
    """Returns exactly the ground-truth box for every image."""

    def predict(self, image, threshold=0.0):
        return FakeDetections(xyxy=[[0.0, 0.0, 10.0, 10.0]], confidence=[0.9], class_id=[0])


def test_evaluate_split_scores_a_perfect_model(tmp_path, monkeypatch):
    annotations_path, images_dir = _tiny_split(tmp_path)
    monkeypatch.setattr(evaluate_module, "load_detector", lambda *a, **k: PerfectModel())

    results = evaluate_split(annotations_path, images_dir, variant="nano", checkpoint="ignored.pth")

    assert results["mAP_50"] == pytest.approx(1.0)
    assert results["num_images"] == 3.0
    assert results["full_split"] == 1.0


def test_evaluate_split_flags_a_truncated_run(tmp_path, monkeypatch):
    """A --limit run must never be mistakable for a full-split result."""
    annotations_path, images_dir = _tiny_split(tmp_path)
    monkeypatch.setattr(evaluate_module, "load_detector", lambda *a, **k: PerfectModel())

    results = evaluate_split(
        annotations_path, images_dir, variant="nano", checkpoint="ignored.pth", limit=2
    )

    assert results["num_images"] == 2.0
    assert results["full_split"] == 0.0


def test_load_detector_reports_a_missing_checkpoint_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="xray-detect"):
        evaluate_module.load_detector("nano", tmp_path / "absent.pth")
