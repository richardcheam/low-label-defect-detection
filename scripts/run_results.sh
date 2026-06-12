#!/usr/bin/env bash
set -euo pipefail

# Produce a results report from the X-ray detection pipeline.
#
# - If a trained RF-DETR checkpoint is present, runs real inference
#   (xray-predict) and a real ROI report (xray-roi --predictions ...).
# - If no checkpoint is present, falls back to a simulated ROI report
#   (xray-roi --simulate-*) and labels the output as such.
#
# Override any of these via environment variables, e.g.:
#   CKPT=artifacts/sixray_detection/checkpoint_best_ema.pth bash scripts/run_results.sh

CKPT="${CKPT:-artifacts/sixray_detection/checkpoint_best_total.pth}"
VARIANT="${VARIANT:-nano}"
THREAT_DIR="${THREAT_DIR:-data/sixray_v3/test/images}"
CLEAN_DIR="${CLEAN_DIR:-data/clean_bags}"

SUMMARY="artifacts/sixray_detection/train_summary.json"
if [ -f "$SUMMARY" ]; then
    echo "== Detection training summary ($SUMMARY) =="
    cat "$SUMMARY"
    echo
fi

if [ -f "$CKPT" ]; then
    echo "== REAL results: checkpoint found at $CKPT =="

    CLEAN_ARGS=()
    if [ -d "$CLEAN_DIR" ]; then
        CLEAN_ARGS=(--clean-dir "$CLEAN_DIR")
    else
        echo "NOTE: no clean-bag directory at $CLEAN_DIR -- scoring threat images only."
        echo "      See docs/explainers/dataset_decision.md for why clean-bag data is missing."
    fi

    uv run xray-predict \
        --checkpoint "$CKPT" \
        --variant "$VARIANT" \
        --threat-dir "$THREAT_DIR" \
        "${CLEAN_ARGS[@]}" \
        --out predictions.json

    uv run xray-roi --config configs/roi.yaml --predictions predictions.json

    echo
    echo "These ROI numbers are REAL, measured on the test split."
    if [ ! -d "$CLEAN_DIR" ]; then
        echo "Threat-only: the auto-clear (clean-bag) rate is not yet measured."
    fi
else
    echo "== SIMULATED/PROJECTED results: no checkpoint at $CKPT =="

    uv run xray-roi --config configs/roi.yaml --simulate-threat 300 --simulate-clean 300

    echo
    echo "WARNING: the ROI numbers above are SIMULATED, not measured."
    echo "Train a detector (scripts/run_detection.sh on a GPU box) and copy the"
    echo "resulting checkpoint to $CKPT to produce real numbers."
fi
