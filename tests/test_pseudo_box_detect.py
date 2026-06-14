from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from simclr_hpl.cli.pseudo_box_detect import run_pseudo_box_detection
from simclr_hpl.detection.pseudo_box import ANNOTATIONS_FILENAME, load_coco, save_coco


class _FakeDetections:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = np.array(xyxy, dtype=float)
        self.confidence = np.array(confidence, dtype=float)
        self.class_id = np.array(class_id, dtype=int)


@pytest.fixture
def fake_rfdetr(monkeypatch):
    calls = {"init": [], "train": [], "predict": []}

    class _FakeModel:
        def __init__(self, **kw):
            calls["init"].append(kw)

        def train(self, **kw):
            calls["train"].append(kw)
            out = Path(kw["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            (out / "results.json").write_text(json.dumps({"map": 0.1 * len(calls["train"])}))

        def predict(self, image, threshold=0.5):
            idx = len(calls["predict"])
            calls["predict"].append(threshold)
            # alternate: even calls get a confident box, odd calls get a weak one
            if idx % 2 == 0:
                return _FakeDetections(xyxy=[[0, 0, 4, 4]], confidence=[0.95], class_id=[0])
            return _FakeDetections(xyxy=[[0, 0, 4, 4]], confidence=[0.3], class_id=[0])

    mod = types.ModuleType("rfdetr")
    mod.RFDETRNano = _FakeModel
    mod.RFDETRSmall = _FakeModel
    mod.RFDETRBase = _FakeModel
    monkeypatch.setitem(sys.modules, "rfdetr", mod)
    return calls


def _make_coco_root(tmp_path: Path, n_images: int = 10) -> Path:
    coco_root = tmp_path / "coco"
    categories = [{"id": 1, "name": "Gun", "supercategory": "none"}]

    train_dir = coco_root / "train"
    train_dir.mkdir(parents=True)
    images = []
    annotations = []
    for i in range(n_images):
        file_name = f"img{i}.jpg"
        Image.new("RGB", (32, 32)).save(train_dir / file_name)
        images.append({"id": i, "file_name": file_name, "width": 32, "height": 32})
        annotations.append(
            {
                "id": i + 1,
                "image_id": i,
                "category_id": 1,
                "bbox": [1, 1, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        )
    save_coco(
        {"images": images, "annotations": annotations, "categories": categories},
        train_dir / ANNOTATIONS_FILENAME,
    )

    for split in ("valid", "test"):
        split_dir = coco_root / split
        split_dir.mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(split_dir / "v0.jpg")
        save_coco(
            {
                "images": [{"id": 1000, "file_name": "v0.jpg", "width": 32, "height": 32}],
                "annotations": [],
                "categories": categories,
            },
            split_dir / ANNOTATIONS_FILENAME,
        )

    return coco_root


def _base_config(tmp_path: Path, coco_root: Path) -> dict:
    return {
        "seed": 42,
        "data": {"coco_root": str(coco_root)},
        "model": {"variant": "nano"},
        "output_dir": str(tmp_path / "out"),
        "pseudo_labeling": {
            "label_fraction": 0.2,
            "confidence_threshold": 0.9,
            "threshold_decay": 0.05,
            "iterations": 1,
        },
        "train": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-4},
        "tracking": {"enabled": False, "experiment": "t", "tracking_uri": None},
    }


def test_run_pseudo_box_detection_end_to_end(tmp_path, fake_rfdetr):
    coco_root = _make_coco_root(tmp_path, n_images=10)
    config = _base_config(tmp_path, coco_root)

    metrics = run_pseudo_box_detection(config)

    # 10 images, label_fraction=0.2 -> 2 labeled, 8 unlabeled
    assert metrics["n_labeled_images"] == 2
    assert metrics["n_unlabeled_images"] == 8

    # round 0: baseline trained on the labeled-only split
    out_dir = Path(config["output_dir"])
    round0_coco = load_coco(out_dir / "round_0" / "train" / ANNOTATIONS_FILENAME)
    assert len(round0_coco["images"]) == 2
    assert len(round0_coco["annotations"]) == 2
    assert (out_dir / "round_0" / "valid").exists()
    assert (out_dir / "round_0" / "test").exists()
    assert metrics["baseline"]["n_images"] == 2
    assert metrics["baseline"]["map"] == pytest.approx(0.1)

    # round 1: half of the 8 unlabeled images (4) get a confident pseudo-box
    assert len(metrics["iterations"]) == 1
    iteration = metrics["iterations"][0]
    assert iteration["new_pseudo_images"] == 4
    assert iteration["new_pseudo_labels"] == 4
    assert iteration["avg_confidence"] == pytest.approx(0.95)
    assert iteration["remaining_unlabeled"] == 4
    assert iteration["n_train_images"] == 6
    assert iteration["confidence_threshold"] == pytest.approx(0.9)
    assert iteration["map"] == pytest.approx(0.2)

    round1_coco = load_coco(out_dir / "round_1" / "train" / ANNOTATIONS_FILENAME)
    assert len(round1_coco["images"]) == 6
    assert len(round1_coco["annotations"]) == 6
    # pseudo annotation ids continue after the real annotations (ids 1..10)
    pseudo_anns = [ann for ann in round1_coco["annotations"] if ann["id"] > 10]
    assert len(pseudo_anns) == 4
    assert all(ann["category_id"] == 1 for ann in pseudo_anns)

    # metrics.json is written with the same content
    assert load_coco(out_dir / "metrics.json") == metrics

    # two training rounds total (round 0 + round 1)
    assert len(fake_rfdetr["train"]) == 2
    assert fake_rfdetr["train"][0]["dataset_dir"] == str(out_dir / "round_0")
    assert fake_rfdetr["train"][1]["dataset_dir"] == str(out_dir / "round_1")


def test_run_pseudo_box_detection_zero_iterations(tmp_path, fake_rfdetr):
    coco_root = _make_coco_root(tmp_path, n_images=10)
    config = _base_config(tmp_path, coco_root)
    config["pseudo_labeling"]["iterations"] = 0

    metrics = run_pseudo_box_detection(config)

    assert metrics["iterations"] == []
    assert len(fake_rfdetr["train"]) == 1
    assert fake_rfdetr["predict"] == []
