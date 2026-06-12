# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Reproducible research/portfolio project for label-efficient image learning and manufacturing
defect inspection (reference: SimCLR paper, https://arxiv.org/abs/2002.05709). It combines
SimCLR self-supervised pretraining, linear/MLP probing, semi-supervised training with hard
pseudo-labeling, and a confidence-aware MVTec AD inspection workflow.

The project has also expanded into an X-ray baggage threat-detection pipeline (RF-DETR
object detection) with MLflow experiment tracking, ROI/business-impact reporting, a DVC
pipeline, and a CPU Docker image — see the "Detection / X-ray pipeline" commands below.

## Commands

Environment is managed with `uv` (Python 3.11, pinned via `.python-version`).

- Install/sync deps: `uv sync --dev`
- Run all tests: `uv run pytest` (or `make test`)
- Run a single test: `uv run pytest tests/test_models.py::test_name`
- Lint: `uv run ruff check .` (or `make lint`)
- Format: `uv run ruff format .` (or `make format`)

Experiments (each is config-driven; configs live in `configs/*.yaml`, also runnable via
`make simclr` / `make pseudo` / `make benchmark` / `make mvtec` / `make plots`):

- `uv run simclr-train --config configs/simclr_mnist.yaml`
- `uv run pseudo-label-train --config configs/pseudo_label_mnist.yaml`
- `uv run transfer-benchmark --config configs/transfer_pseudo_label_mnist.yaml`
- `uv run mvtec-inspection --config configs/mvtec_bottle_inspection.yaml`
- `uv run plot-results --metrics artifacts/<experiment>/metrics.json` (writes to a sibling
  `plots/` dir by default)

CI (`.github/workflows/ci.yml`) runs `uv sync --dev`, `uv run ruff check .`, `uv run pytest` —
match this locally before pushing. CI also installs `libgl1 libglib2.0-0` (needed by
`opencv`/`supervision`).

### Detection / X-ray pipeline

- `uv run xray-screening --config configs/xray_screening.yaml` — Phase 0 binary
  threat/no-threat classifier (SimCLR + frozen-probe, on RGB X-ray crops)
- `uv run prepare-detection-data --yolo-root data/sixray_v3 --output-root data/sixray_v3_coco`
  — convert the Roboflow YOLO export to COCO via `supervision`
- `uv sync --extra detection` then `uv run xray-detect --config configs/sixray_detection.yaml`
  — train RF-DETR (`rfdetr` is an optional, heavy, GPU-oriented extra; lazily imported)
- `uv run xray-predict --checkpoint <ckpt> --variant nano --threat-dir <dir> [--clean-dir <dir>] --out predictions.json`
  — run inference, write `predictions.json` (`[{"score": float, "target": int}]`)
- `uv run xray-roi --config configs/roi.yaml [--predictions predictions.json | --simulate-threat N --simulate-clean M]`
  — business-impact (ROI) report (`artifacts/roi/report.json`, `roi_summary.png`)
- `bash scripts/run_results.sh` (or `make results`) — runs real predict+ROI if a checkpoint
  exists at `artifacts/sixray_detection/checkpoint_best_total.pth`, else falls back to a
  clearly-labeled simulated ROI report
- `bash scripts/run_detection.sh` — full chain (data prep → train → predict → ROI), intended
  for a GPU box
- `uv run dvc repro` (or `make repro`) — run the `dvc.yaml` pipeline (`prepare_data` →
  `train`[frozen] → `predict` → `roi`); `train` is frozen because real training happens on a
  separate CUDA box and the checkpoint is copied in and tracked via
  `dvc add artifacts/sixray_detection/checkpoint_best_total.pth`
- `docker build -t xray-screening . && docker run --rm xray-screening` — CPU image running
  the test suite; `docker compose up mlflow` serves the MLflow UI on `:5000`

CPU/GPU split: rfdetr training runs on a separate CUDA box; only the resulting checkpoint is
copied into `artifacts/sixray_detection/`. Everything else (data prep, predict, ROI, tests,
Docker) runs CPU-only. `MLFLOW_ALLOW_FILE_STORE=true` is set automatically by
`tracking.py` for mlflow>=3's local file store.

## Architecture

All experiment logic lives in `src/simclr_hpl/`. Every CLI entry point in
`src/simclr_hpl/cli/` (registered as console scripts in `pyproject.toml`) follows the same
pipeline, e.g. `cli/simclr.py`:

1. `config.load_config(path)` reads a YAML file into a plain dict (sections like `data`,
   `train`, `model`, `evaluation` — see `configs/*.yaml` for shape per experiment)
2. `utils.seed_everything(config["seed"])` and `utils.resolve_device(...)` set up
   reproducibility/device
3. Datasets, transforms and train/val splits come from `data.py` (MNIST + MVTec AD loaders,
   SimCLR contrastive-view augmentations, supervised augmentations)
4. Models come from `models.py`: `Encoder` (3-layer CNN backbone), `ProjectionHead`,
   `LinearProbe` / `MLPProbe`, `SemiSupervisedCNN`, `EncoderClassifier`
5. Training/eval loops and losses come from `training.py`: `NTXentLoss` (contrastive),
   `pretrain_simclr`, `train_classifier`, `evaluate_classifier`, `generate_pseudo_labels`
6. Results are persisted with `utils.save_checkpoint` / `utils.save_json` under
   `output_dir` (always under `artifacts/<experiment_name>/`)
7. `business.py` computes manufacturing-style review-queue metrics (`auto_decision_rate`,
   `review_queue_rate`, `auto_decision_accuracy`, `auto_defect_recall`) for the MVTec workflow
8. `visualization.py` + the `plot-results` CLI read an experiment's `metrics.json` and infer
   plot types to generate publication-style figures

The four experiment CLIs compose these pieces differently:
- `simclr-train`: pretrain encoder contrastively, then evaluate with linear/MLP probes
- `pseudo-label-train`: train on a small labeled set, generate confident hard pseudo-labels
  on unlabeled data, retrain iteratively under a decaying confidence threshold
- `transfer-benchmark`: compares random vs. SimCLR-pretrained initialization of the same
  `EncoderClassifier` across label budgets (configurable, e.g. 100/250/500 labels); will
  pretrain a SimCLR encoder automatically if `artifacts/simclr_mnist/simclr_encoder.pt`
  is missing
- `mvtec-inspection`: binary good-vs-defect classification on one MVTec AD category
  (expects data under `data/mvtec_ad/<category>/{train,test,ground_truth}/`), reporting
  both accuracy and review-queue metrics via `business.py`

`archive/` holds the original exploratory notebooks (`SimCLR.ipynb`,
`CNN_semi_supervised.ipynb`) that this package was refactored out of — treat them as
historical reference, not as code to extend.

### Detection / MLOps modules

- `tracking.py`: `ExperimentTracker` — thin MLflow wrapper (`run()` context manager,
  `log_params`/`log_metrics`/`log_artifact`) that degrades to a no-op if `mlflow` is
  unavailable or `enabled=False`. Sets `MLFLOW_ALLOW_FILE_STORE=true` on import.
- `detection/data.py`: `convert_roboflow_yolo_dataset` — YOLO → COCO conversion via
  `supervision.DetectionDataset`, used by `prepare-detection-data`.
- `detection/train.py`: `train_detector(config, tracker)` — lazily imports `rfdetr`,
  resolves `model.variant` (`nano`/`small`/`base`) to an `RFDETR*` class via
  `resolve_model_class`, writes `train_summary.json`, and best-effort logs any
  `results.json`/`metrics.json` numeric scalars to MLflow.
- `detection/predict.py`: `build_predictions(threat_dir, clean_dir, variant=, checkpoint=, threshold=)`
  — loads the checkpoint via `pretrain_weights=`, scores images by max detection
  confidence, returns `[{"score": float, "target": int}]`. Warns (doesn't fail) if
  `clean_dir` is missing.
- `roi.py` + `cli/roi_report.py`: `evaluate_roi(scores, targets, ...)` turns
  predictions into review-queue metrics (reusing `business.compute_review_queue_metrics`)
  plus ROI figures (inspector-hours/cost saved, missed-threat rate); `xray-roi` writes
  `report.json` + `roi_summary.png` and logs to MLflow.
- `dvc.yaml`: pipeline graph `prepare_data → train (frozen) → predict → roi`, with
  `configs/sixray_detection.yaml` / `configs/roi.yaml` declared as `params:` so config
  changes invalidate the right stage. `.gitignore` has explicit `!...` exceptions so the
  `*.dvc` pointer files and `dvc.yaml`/`dvc.lock` are tracked even though `data/` and
  `artifacts/` are otherwise gitignored.
