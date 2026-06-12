from __future__ import annotations

import json

import yaml

from simclr_hpl.cli.roi_report import main as roi_main


def test_roi_cli_simulate_writes_report(tmp_path, monkeypatch):
    cfg = {
        "detection_threshold": 0.5, "auto_decision_threshold": 0.9,
        "assumptions": {"seconds_per_manual_review": 12.0, "inspector_hourly_cost": 30.0, "total_bags": None},
        "output_dir": str(tmp_path / "out"),
        "tracking": {"enabled": False, "experiment": "t", "tracking_uri": None},
    }
    cfg_path = tmp_path / "roi.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr("sys.argv", ["xray-roi", "--config", str(cfg_path),
                                     "--simulate-threat", "40", "--simulate-clean", "60"])
    roi_main()
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["roi"]["total_bags"] == 100
    assert (tmp_path / "out" / "roi_summary.png").exists()
