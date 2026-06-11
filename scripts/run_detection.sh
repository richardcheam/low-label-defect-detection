#!/usr/bin/env bash
set -euo pipefail

# RF-DETR detection training pipeline.
#
# Run from the repository root on the user's Linux/CUDA box. `rfdetr` is a
# heavy, CUDA/torch-dependent optional extra and is intentionally NOT part of
# the core environment used for the rest of this repo.

# 1. install with the detection extra (CUDA box)
uv sync --extra detection

# 2. convert the Roboflow YOLO export to COCO
uv run prepare-detection-data --yolo-root data/sixray_v3 --output-root data/sixray_v3_coco

# 3. train RF-DETR (logs to local ./mlruns)
uv run xray-detect --config configs/sixray_detection.yaml

# 4. browse results:  uv run mlflow ui
