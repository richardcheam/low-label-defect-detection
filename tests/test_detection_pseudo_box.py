from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from simclr_hpl.detection.pseudo_box import (
    ANNOTATIONS_FILENAME,
    build_train_split,
    detections_to_coco_annotations,
    filter_annotations,
    generate_pseudo_annotations,
    link_eval_splits,
    load_coco,
    save_coco,
    split_labeled_unlabeled,
)


class _FakeDetections:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = np.array(xyxy, dtype=float)
        self.confidence = np.array(confidence, dtype=float)
        self.class_id = np.array(class_id, dtype=int)


def _coco_images(n):
    return [{"id": i, "file_name": f"img{i}.jpg", "width": 64, "height": 64} for i in range(n)]


def test_load_save_coco_roundtrip(tmp_path):
    data = {"images": [{"id": 1, "file_name": "a.jpg"}], "annotations": [], "categories": []}
    path = tmp_path / "_annotations.coco.json"
    save_coco(data, path)
    assert load_coco(path) == data


def test_split_labeled_unlabeled_fraction_and_determinism():
    coco = {"images": _coco_images(20)}

    labeled, unlabeled = split_labeled_unlabeled(coco, label_fraction=0.25, seed=42)
    assert len(labeled) == 5
    assert len(unlabeled) == 15
    assert {img["id"] for img in labeled} | {img["id"] for img in unlabeled} == set(range(20))

    # same seed -> same split
    labeled_again, unlabeled_again = split_labeled_unlabeled(coco, label_fraction=0.25, seed=42)
    assert [img["id"] for img in labeled] == [img["id"] for img in labeled_again]
    assert [img["id"] for img in unlabeled] == [img["id"] for img in unlabeled_again]

    # different seed -> different split
    labeled_other, _ = split_labeled_unlabeled(coco, label_fraction=0.25, seed=123)
    assert [img["id"] for img in labeled] != [img["id"] for img in labeled_other]


def test_split_labeled_unlabeled_min_one():
    coco = {"images": _coco_images(10)}
    labeled, unlabeled = split_labeled_unlabeled(coco, label_fraction=0.01, seed=0)
    assert len(labeled) == 1
    assert len(unlabeled) == 9


def test_filter_annotations():
    annotations = [
        {"id": 1, "image_id": 0},
        {"id": 2, "image_id": 1},
        {"id": 3, "image_id": 2},
    ]
    kept = filter_annotations(annotations, image_ids={0, 2})
    assert [ann["id"] for ann in kept] == [1, 3]


def test_build_train_split_symlinks_images_and_writes_json(tmp_path):
    coco_root = tmp_path / "coco_root"
    (coco_root / "train").mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (8, 8)).save(coco_root / "train" / f"img{i}.jpg")

    output_dir = tmp_path / "round_0"
    images = _coco_images(2)  # img0, img1 only
    annotations = [{"id": 1, "image_id": 0, "category_id": 1, "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0}]
    categories = [{"id": 1, "name": "Gun"}]

    train_dir = build_train_split(coco_root, output_dir, images, annotations, categories)

    assert train_dir == output_dir / "train"
    for i in range(2):
        linked = train_dir / f"img{i}.jpg"
        assert linked.exists()
        assert linked.is_symlink() or linked.is_file()
    assert not (train_dir / "img2.jpg").exists()

    written = load_coco(train_dir / ANNOTATIONS_FILENAME)
    assert written == {"images": images, "annotations": annotations, "categories": categories}


def test_link_eval_splits_creates_symlinks(tmp_path):
    coco_root = tmp_path / "coco_root"
    for split in ("valid", "test"):
        split_dir = coco_root / split
        split_dir.mkdir(parents=True)
        (split_dir / "marker.txt").write_text(split)

    output_dir = tmp_path / "round_0"
    output_dir.mkdir()

    link_eval_splits(coco_root, output_dir)

    for split in ("valid", "test"):
        linked = output_dir / split
        assert linked.exists()
        assert (linked / "marker.txt").read_text() == split


def test_link_eval_splits_skips_missing_source(tmp_path):
    coco_root = tmp_path / "coco_root"
    coco_root.mkdir()
    output_dir = tmp_path / "round_0"
    output_dir.mkdir()

    # neither valid/ nor test/ exist under coco_root -- should not raise
    link_eval_splits(coco_root, output_dir)

    assert not (output_dir / "valid").exists()
    assert not (output_dir / "test").exists()


def test_detections_to_coco_annotations_filters_and_maps_categories():
    categories = [{"id": 10, "name": "Gun"}, {"id": 20, "name": "Knife"}]
    detections = _FakeDetections(
        xyxy=[[0, 0, 10, 20], [5, 5, 15, 10], [1, 1, 2, 2]],
        confidence=[0.95, 0.4, 0.99],
        class_id=[0, 1, 1],
    )

    annotations, confidences = detections_to_coco_annotations(
        detections, image_id=7, categories=categories, next_id=100, threshold=0.5
    )

    assert confidences == pytest.approx([0.95, 0.99])
    assert [ann["id"] for ann in annotations] == [100, 101]
    assert all(ann["image_id"] == 7 for ann in annotations)
    # first detection: class_id 0 -> category id 10, bbox xyxy [0,0,10,20] -> xywh [0,0,10,20]
    assert annotations[0]["category_id"] == 10
    assert annotations[0]["bbox"] == [0.0, 0.0, 10.0, 20.0]
    assert annotations[0]["area"] == pytest.approx(200.0)
    # second kept detection: class_id 1 -> category id 20, bbox xyxy [1,1,2,2] -> xywh [1,1,1,1]
    assert annotations[1]["category_id"] == 20
    assert annotations[1]["bbox"] == [1.0, 1.0, 1.0, 1.0]


def test_generate_pseudo_annotations(tmp_path):
    images_dir = tmp_path / "unlabeled"
    images_dir.mkdir()
    for i in range(3):
        Image.new("RGB", (16, 16)).save(images_dir / f"img{i}.jpg")

    images = _coco_images(3)
    categories = [{"id": 1, "name": "Gun"}]

    # img0 -> one confident detection, img1 -> below threshold, img2 -> empty
    detections_by_image = {
        0: _FakeDetections(xyxy=[[0, 0, 4, 4]], confidence=[0.9], class_id=[0]),
        1: _FakeDetections(xyxy=[[0, 0, 4, 4]], confidence=[0.1], class_id=[0]),
        2: _FakeDetections(xyxy=[], confidence=[], class_id=[]),
    }

    class _FakeModel:
        def predict(self, image, threshold=0.5):
            # identify which image by re-reading call order
            idx = self.calls
            self.calls += 1
            return detections_by_image[idx]

        calls = 0

    model = _FakeModel()
    pseudo_annotations, pseudo_images, confidences = generate_pseudo_annotations(
        model, images, images_dir, categories, threshold=0.5, next_id=500
    )

    assert [img["id"] for img in pseudo_images] == [0]
    assert len(pseudo_annotations) == 1
    assert pseudo_annotations[0]["id"] == 500
    assert pseudo_annotations[0]["image_id"] == 0
    assert pseudo_annotations[0]["category_id"] == 1
    assert confidences == pytest.approx([0.9])


def test_build_train_split_creates_nested_filenames(tmp_path):
    # rfdetr/Roboflow filenames sometimes include subdirectory-like prefixes;
    # ensure build_train_split creates parent dirs for the symlink target.
    coco_root = tmp_path / "coco_root"
    (coco_root / "train" / "nested").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(coco_root / "train" / "nested" / "img0.jpg")

    output_dir = tmp_path / "round_0"
    images = [{"id": 0, "file_name": "nested/img0.jpg", "width": 8, "height": 8}]

    train_dir = build_train_split(coco_root, output_dir, images, [], [])

    assert (train_dir / "nested" / "img0.jpg").exists()


def test_save_coco_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / ANNOTATIONS_FILENAME
    save_coco({"images": [], "annotations": [], "categories": []}, path)
    assert json.loads(path.read_text()) == {"images": [], "annotations": [], "categories": []}
