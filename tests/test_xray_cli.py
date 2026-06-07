import json
from pathlib import Path

import yaml
from PIL import Image

from simclr_hpl.cli.xray_screening import main as xray_main


def _make_dataset(root: Path, n_per_class: int = 12) -> None:
    for cls, color in (("positive", (200, 50, 50)), ("negative", (50, 50, 200))):
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            Image.new("RGB", (32, 32), color=color).save(d / f"{i}.png")


def test_xray_cli_runs_and_writes_metrics(tmp_path, monkeypatch):
    data_root = tmp_path / "sixray"
    _make_dataset(data_root)
    out_dir = tmp_path / "out"
    config = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(out_dir),
        "data": {
            "root": str(data_root),
            "class_to_label": {"positive": 1, "negative": 0},
            "image_size": 32,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "pretrain_batch_size": 4,
            "eval_batch_size": 4,
            "num_workers": 0,
            "max_pretrain_images": 16,
            "max_labeled_per_class": 8,
        },
        "model": {
            "encoder": "resnet18",
            "input_channels": 3,
            "pretrained": False,
            "projection_dim": 32,
            "num_classes": 2,
        },
        "train": {"epochs": 1, "learning_rate": 0.001, "weight_decay": 0.0, "temperature": 0.5},
        "evaluation": {"validation_size": 0.25, "epochs": 1, "learning_rate": 0.001},
    }
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(config))

    monkeypatch.setattr("sys.argv", ["xray-screening", "--config", str(config_path)])
    xray_main()

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert "linear_probe" in metrics and "mlp_probe" in metrics
    assert (out_dir / "xray_encoder.pt").exists()
