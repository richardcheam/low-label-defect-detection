UV ?= uv

.PHONY: sync test lint format simclr pseudo benchmark

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
