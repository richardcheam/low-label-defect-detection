from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from simclr_hpl.detection.train import train_detector


@pytest.fixture
def fake_rfdetr(monkeypatch):
    calls = {}

    class _FakeModel:
        def __init__(self, **kw):
            calls["init"] = kw

        def train(self, **kw):
            calls["train"] = kw
            Path(kw["output_dir"]).mkdir(parents=True, exist_ok=True)
            # simulate rfdetr writing a results file
            (Path(kw["output_dir"]) / "results.json").write_text(json.dumps({"map": 0.42}))

    mod = types.ModuleType("rfdetr")
    mod.RFDETRNano = _FakeModel
    mod.RFDETRSmall = _FakeModel
    mod.RFDETRBase = _FakeModel
    monkeypatch.setitem(sys.modules, "rfdetr", mod)
    return calls


def _base_config(tmp_path: Path, variant: str = "nano") -> dict:
    return {
        "seed": 0,
        "data": {"coco_root": str(tmp_path / "coco")},
        "model": {"variant": variant},
        "output_dir": str(tmp_path / "out"),
        "train": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-4},
        "tracking": {"enabled": False, "experiment": "t", "tracking_uri": None},
    }


def test_train_detector_invokes_rfdetr_and_logs(tmp_path, fake_rfdetr):
    config = _base_config(tmp_path)
    summary = train_detector(config)

    assert fake_rfdetr["train"]["epochs"] == 1
    assert fake_rfdetr["train"]["batch_size"] == 2
    assert fake_rfdetr["train"]["lr"] == 1e-4
    assert fake_rfdetr["train"]["dataset_dir"] == str(tmp_path / "coco")
    assert fake_rfdetr["train"]["output_dir"] == str(tmp_path / "out")
    assert fake_rfdetr["train"]["devices"] == 1
    assert fake_rfdetr["train"]["grad_accum_steps"] == 1
    assert fake_rfdetr["train"]["find_unused_parameters"] is False
    assert (tmp_path / "out").exists()
    assert summary["output_dir"] == str(tmp_path / "out")
    assert summary["config"] == config


def test_train_detector_passes_through_multi_gpu_settings(tmp_path, fake_rfdetr):
    config = _base_config(tmp_path)
    config["train"]["devices"] = "auto"
    config["train"]["grad_accum_steps"] = 4
    config["train"]["find_unused_parameters"] = True

    train_detector(config)

    assert fake_rfdetr["train"]["devices"] == "auto"
    assert fake_rfdetr["train"]["grad_accum_steps"] == 4
    assert fake_rfdetr["train"]["find_unused_parameters"] is True


def test_train_detector_writes_train_summary(tmp_path, fake_rfdetr):
    config = _base_config(tmp_path)
    train_detector(config)

    summary_path = Path(config["output_dir"]) / "train_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["output_dir"] == config["output_dir"]
    assert payload["config"] == config


@pytest.mark.parametrize(
    ("variant", "attr"),
    [("nano", "RFDETRNano"), ("small", "RFDETRSmall"), ("base", "RFDETRBase")],
)
def test_train_detector_resolves_model_variant(tmp_path, fake_rfdetr, variant, attr):
    config = _base_config(tmp_path, variant=variant)
    train_detector(config)
    # All variants are the same _FakeModel in this fixture, so just confirm
    # training was invoked successfully for each accepted variant name.
    assert "train" in fake_rfdetr


def test_train_detector_unknown_variant_raises(tmp_path, fake_rfdetr):
    config = _base_config(tmp_path, variant="huge")
    with pytest.raises(ValueError, match="huge"):
        train_detector(config)


def test_train_detector_logs_metrics_with_tracker(tmp_path, fake_rfdetr):
    from simclr_hpl.tracking import ExperimentTracker

    config = _base_config(tmp_path)
    config["tracking"]["enabled"] = True
    config["tracking"]["tracking_uri"] = (tmp_path / "mlruns").as_uri()

    tracker = ExperimentTracker(
        enabled=True,
        tracking_uri=config["tracking"]["tracking_uri"],
        experiment=config["tracking"]["experiment"],
    )
    summary = train_detector(config, tracker=tracker)

    import mlflow

    mlflow.set_tracking_uri(config["tracking"]["tracking_uri"])
    runs = mlflow.search_runs(experiment_names=[config["tracking"]["experiment"]])
    assert len(runs) == 1
    assert runs.iloc[0]["params.variant"] == "nano"
    assert runs.iloc[0]["params.epochs"] == "1"
    assert float(runs.iloc[0]["metrics.map"]) == pytest.approx(0.42)
    assert summary["output_dir"] == str(tmp_path / "out")


def test_train_detector_missing_results_file_does_not_crash(tmp_path, monkeypatch):
    """If rfdetr does not write a metrics file, train_detector must not crash."""

    class _NoResultsModel:
        def __init__(self, **kw):
            pass

        def train(self, **kw):
            Path(kw["output_dir"]).mkdir(parents=True, exist_ok=True)
            # intentionally do not write any results/metrics file

    mod = types.ModuleType("rfdetr")
    mod.RFDETRNano = _NoResultsModel
    mod.RFDETRSmall = _NoResultsModel
    mod.RFDETRBase = _NoResultsModel
    monkeypatch.setitem(sys.modules, "rfdetr", mod)

    config = _base_config(tmp_path)
    summary = train_detector(config)
    assert summary["output_dir"] == str(tmp_path / "out")
