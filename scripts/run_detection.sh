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
#    Multi-GPU: set NUM_GPUS to the number of GPUs to use (default 1). Uses
#    torchrun for DDP and passes --devices auto so rfdetr uses all GPUs
#    visible to the process group (per-GPU batch size stays as configured,
#    so effective batch size scales with NUM_GPUS - adjust
#    train.grad_accum_steps in the config if you need a fixed effective
#    batch size).
NUM_GPUS="${NUM_GPUS:-1}"
if [ "$NUM_GPUS" -gt 1 ]; then
    uv run torchrun --nproc_per_node="$NUM_GPUS" -m simclr_hpl.cli.detect \
        --config configs/sixray_detection.yaml --devices auto
else
    uv run xray-detect --config configs/sixray_detection.yaml
fi

# 4. browse results:  uv run mlflow ui

# 5. produce predictions.json from the trained model on threat + clean test images
#    (checkpoint name depends on what rfdetr wrote to output_dir, e.g. checkpoint_best_total.pth)
uv run xray-predict --checkpoint artifacts/sixray_detection/checkpoint_best_total.pth --variant nano \
    --threat-dir data/sixray_v3/test/images --clean-dir data/clean_bags --out predictions.json

# 6. ROI report from the real predictions
uv run xray-roi --config configs/roi.yaml --predictions predictions.json

# (preview without a model: ROI report on simulated detections)
uv run xray-roi --config configs/roi.yaml --simulate-threat 300 --simulate-clean 300
