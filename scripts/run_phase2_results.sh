#!/usr/bin/env bash
set -euo pipefail

# Produce Phase 2a (few-label pseudo-box semi-supervised detection) results.
#
# - If artifacts/pseudo_box_detection/metrics.json is present, generates plots
#   via plot-results and prints a human-readable summary.
# - If the metrics file is absent, prints the torchrun command needed to run
#   Phase 2a training on the GPU box.
#
# Once the GPU run completes, copy the full artifacts/pseudo_box_detection/
# directory from the GPU box and re-run this script.

METRICS="${METRICS:-artifacts/pseudo_box_detection/metrics.json}"

if [ -f "$METRICS" ]; then
    echo "== Phase 2a results: $METRICS found =="
    echo

    uv run plot-results --metrics "$METRICS"

    echo
    echo "=== Summary ==="
    python3 - "$METRICS" <<'PYEOF'
import json, pathlib, sys

path = pathlib.Path(sys.argv[1])
try:
    data = json.loads(path.read_text())
except Exception as e:
    print(f"Could not parse metrics: {e}", file=sys.stderr)
    sys.exit(1)

lf = data.get("label_fraction", "?")
n_labeled = data.get("n_labeled_images", "?")
print(f"  label_fraction     : {lf}")
print(f"  n_labeled_images   : {n_labeled}")

baseline = data.get("baseline", {})
if baseline:
    # find a mAP key
    map_key = next((k for k in baseline if "map" in k.lower()), None)
    if map_key:
        print(f"  baseline {map_key:15s}: {baseline[map_key]:.4f}")

iterations = data.get("iterations", [])
print(f"  pseudo-label rounds: {len(iterations)}")
for i, it in enumerate(iterations, 1):
    new_ps = it.get("new_pseudo_images", "?")
    conf   = it.get("avg_confidence", None)
    map_key = next((k for k in it if "map" in k.lower()), None)
    map_val = f"{it[map_key]:.4f}" if map_key else "?"
    conf_str = f"{conf:.3f}" if conf is not None else "?"
    print(f"  round {i}: new_pseudo={new_ps:>5}  avg_conf={conf_str}  mAP={map_val}")

if iterations:
    last = iterations[-1]
    map_key = next((k for k in last if "map" in k.lower()), None)
    phase1_ref = 0.624
    if map_key:
        delta = last[map_key] - phase1_ref
        sign  = "+" if delta >= 0 else ""
        print()
        print(f"  Phase 1 reference  : {phase1_ref:.4f} (full supervision, all labels)")
        print(f"  Phase 2a best      : {last[map_key]:.4f}  ({sign}{delta:.4f} vs Phase 1)")
PYEOF

    echo
    echo "Plots written to artifacts/pseudo_box_detection/plots/"
else
    echo "== Phase 2a results not yet available: $METRICS not found =="
    echo
    echo "Run Phase 2a training on the GPU box, then copy the full"
    echo "artifacts/pseudo_box_detection/ directory back here."
    echo
    echo "GPU box command (4×GPU DDP):"
    echo
    echo "  uv sync --extra detection"
    echo "  uv run torchrun --nproc_per_node=4 -m simclr_hpl.cli.pseudo_box_detect \\"
    echo "    --config configs/pseudo_box_detection.yaml \\"
    echo "    --devices auto \\"
    echo "    --strategy ddp_find_unused_parameters_true"
    echo
    echo "Single-GPU fallback:"
    echo
    echo "  uv sync --extra detection"
    echo "  uv run xray-pseudo-box-detect --config configs/pseudo_box_detection.yaml"
    echo
    echo "Then copy results and re-run:"
    echo "  make results-phase2"
fi
