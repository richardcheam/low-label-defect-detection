from __future__ import annotations

import sys
import types

import pytest
from PIL import Image

from simclr_hpl.detection.zeroshot import build_zeroshot_predictions, _max_score


class _FakeProcessor:
    @classmethod
    def from_pretrained(cls, model_id):
        return cls()

    def __call__(self, *, images, text, return_tensors):
        return {"pixel_values": None, "input_ids": None}

    def post_process_grounded_object_detection(self, outputs, threshold, target_sizes):
        return self._detections

    _detections = [{"scores": [0.85], "labels": ["gun"]}]


class _EmptyProcessor(_FakeProcessor):
    _detections = [{"scores": [], "labels": []}]


class _FakeModel:
    @classmethod
    def from_pretrained(cls, model_id):
        return cls()

    def eval(self):
        return self

    def __call__(self, **kwargs):
        return {}


@pytest.fixture
def fake_transformers(monkeypatch):
    mod = types.ModuleType("transformers")
    mod.AutoProcessor = _FakeProcessor
    mod.AutoModelForZeroShotObjectDetection = _FakeModel
    monkeypatch.setitem(sys.modules, "transformers", mod)


@pytest.fixture
def fake_transformers_empty(monkeypatch):
    mod = types.ModuleType("transformers")
    mod.AutoProcessor = _EmptyProcessor
    mod.AutoModelForZeroShotObjectDetection = _FakeModel
    monkeypatch.setitem(sys.modules, "transformers", mod)


def _make_images(directory, n=2):
    for i in range(n):
        Image.new("RGB", (32, 32)).save(directory / f"img{i}.jpg")


def test_max_score_empty():
    assert _max_score({"scores": []}) == 0.0
    assert _max_score({}) == 0.0


def test_max_score_list():
    assert _max_score({"scores": [0.3, 0.85, 0.6]}) == pytest.approx(0.85)


def test_build_zeroshot_predictions_scores_threats(tmp_path, fake_transformers):
    threat_dir = tmp_path / "threats"
    threat_dir.mkdir()
    _make_images(threat_dir, 3)

    predictions = build_zeroshot_predictions(
        threat_dir, None,
        model_id="IDEA-Research/grounding-dino-base",
        text_labels=["gun", "knife"],
        threshold=0.3,
    )

    assert len(predictions) == 3
    assert all(p["target"] == 1 for p in predictions)
    assert all(p["score"] == pytest.approx(0.85) for p in predictions)


def test_build_zeroshot_predictions_none_clean_warns(tmp_path, fake_transformers):
    threat_dir = tmp_path / "threats"
    threat_dir.mkdir()
    _make_images(threat_dir, 1)

    with pytest.warns(UserWarning, match="clean"):
        predictions = build_zeroshot_predictions(
            threat_dir, None,
            model_id="x", text_labels=["gun"], threshold=0.3,
        )
    assert len(predictions) == 1


def test_build_zeroshot_predictions_missing_clean_warns(tmp_path, fake_transformers):
    threat_dir = tmp_path / "threats"
    threat_dir.mkdir()
    _make_images(threat_dir, 1)

    with pytest.warns(UserWarning, match="clean"):
        predictions = build_zeroshot_predictions(
            threat_dir, tmp_path / "nonexistent",
            model_id="x", text_labels=["gun"], threshold=0.3,
        )
    assert len(predictions) == 1
    assert predictions[0]["target"] == 1


def test_build_zeroshot_predictions_with_clean(tmp_path, fake_transformers):
    threat_dir = tmp_path / "threats"
    clean_dir = tmp_path / "clean"
    threat_dir.mkdir()
    clean_dir.mkdir()
    _make_images(threat_dir, 2)
    _make_images(clean_dir, 2)

    predictions = build_zeroshot_predictions(
        threat_dir, clean_dir,
        model_id="x", text_labels=["gun", "knife"], threshold=0.3,
    )

    assert len(predictions) == 4
    assert sum(p["target"] for p in predictions) == 2
    assert all(0.0 <= p["score"] <= 1.0 for p in predictions)


def test_build_zeroshot_predictions_zero_score_when_empty(tmp_path, fake_transformers_empty):
    threat_dir = tmp_path / "threats"
    threat_dir.mkdir()
    _make_images(threat_dir, 2)

    predictions = build_zeroshot_predictions(
        threat_dir, None,
        model_id="x", text_labels=["gun"], threshold=0.3,
    )

    assert all(p["score"] == pytest.approx(0.0) for p in predictions)
