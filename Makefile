UV ?= uv

.PHONY: sync test lint format simclr pseudo benchmark mvtec plots

sync:
	$(UV) sync --dev

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

simclr:
	$(UV) run simclr-train --config configs/simclr_mnist.yaml

pseudo:
	$(UV) run pseudo-label-train --config configs/pseudo_label_mnist.yaml

benchmark:
	$(UV) run transfer-benchmark --config configs/transfer_pseudo_label_mnist.yaml

mvtec:
	$(UV) run mvtec-inspection --config configs/mvtec_bottle_inspection.yaml

plots:
	$(UV) run plot-results --metrics artifacts/transfer_benchmark_mnist/transfer_benchmark_metrics.json
