from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from PIL import Image

from simclr_hpl.detection.predict import build_predictions, max_confidence


class _FakeDetections:
    def __init__(self, conf):
        self.confidence = np.array(conf, dtype=float)


def test_max_confidence_empty_and_nonempty():
    assert max_confidence(_FakeDetections([])) == 0.0
    assert max_confidence(_FakeDetections([0.3, 0.91, 0.5])) == pytest.approx(0.91)


@pytest.fixture
def fake_rfdetr(monkeypatch):
    calls = {}

    class _FakeModel:
        def __init__(self, **kw):
            calls["init"] = kw

        def predict(self, image, threshold=0.5):
            calls.setdefault("predict_calls", []).append(threshold)
            return _FakeDetections([0.8])

    mod = types.ModuleType("rfdetr")
    mod.RFDETRNano = _FakeModel
    mod.RFDETRSmall = _FakeModel
    mod.RFDETRBase = _FakeModel
    monkeypatch.setitem(sys.modules, "rfdetr", mod)
    return calls


@pytest.fixture
def checkpoint(tmp_path):
    ckpt = tmp_path / "ckpt.pth"
    ckpt.write_bytes(b"weights")
    return ckpt


def test_build_predictions_assigns_targets(tmp_path, fake_rfdetr, checkpoint):
    threat = tmp_path / "threat"
    clean = tmp_path / "clean"
    threat.mkdir()
    clean.mkdir()
    for i in range(3):
        Image.new("RGB", (16, 16)).save(threat / f"P{i}.jpg")
    for i in range(2):
        Image.new("RGB", (16, 16)).save(clean / f"N{i}.jpg")

    preds = build_predictions(threat, clean, variant="nano", checkpoint=checkpoint, threshold=0.5)

    assert len(preds) == 5
    assert sum(p["target"] for p in preds) == 3  # 3 threats labelled 1
    assert all(0.0 <= p["score"] <= 1.0 for p in preds)
    # model constructed with the documented pretrain_weights= kwarg
    assert fake_rfdetr["init"] == {"pretrain_weights": str(checkpoint)}
    # threshold is forwarded to predict()
    assert all(t == 0.5 for t in fake_rfdetr["predict_calls"])


def test_build_predictions_without_clean_dir_warns(tmp_path, fake_rfdetr, checkpoint):
    threat = tmp_path / "threat"
    threat.mkdir()
    Image.new("RGB", (16, 16)).save(threat / "P0.jpg")

    with pytest.warns(UserWarning, match="clean"):
        preds = build_predictions(threat, None, variant="nano", checkpoint=checkpoint)

    assert len(preds) == 1
    assert preds[0]["target"] == 1


def test_build_predictions_missing_clean_dir_warns(tmp_path, fake_rfdetr, checkpoint):
    threat = tmp_path / "threat"
    threat.mkdir()
    Image.new("RGB", (16, 16)).save(threat / "P0.jpg")
    missing_clean = tmp_path / "does_not_exist"

    with pytest.warns(UserWarning, match="clean"):
        preds = build_predictions(threat, missing_clean, variant="nano", checkpoint=checkpoint)

    assert len(preds) == 1
    assert preds[0]["target"] == 1


def test_build_predictions_missing_checkpoint_raises(tmp_path, fake_rfdetr):
    threat = tmp_path / "threat"
    clean = tmp_path / "clean"
    threat.mkdir()
    clean.mkdir()
    Image.new("RGB", (16, 16)).save(threat / "P0.jpg")

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        build_predictions(
            threat, clean, variant="nano", checkpoint=tmp_path / "missing.pth"
        )
