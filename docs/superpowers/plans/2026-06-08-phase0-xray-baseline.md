# Phase 0: Image-Level X-ray Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the existing SimCLR + probe-evaluation pipeline to RGB X-ray baggage images (binary threat / no-threat) using a ResNet18 encoder, producing the image-level low-label baseline for the threat-screening project.

**Architecture:** Add a ResNet18 feature encoder alongside the existing small `Encoder` (same `output_dim` interface so `EncoderClassifier`/`ProjectionHead` work unchanged), parametrize the probe heads' class count, add a folder-based binary X-ray loader that reuses the MVTec records pattern, and add a focused `xray-screening` CLI that reuses the existing library functions (`pretrain_simclr`, `train_classifier`, `evaluate_classifier`). All code is unit-tested against tiny **synthetic image fixtures**, so the whole phase is developable on a Mac with no GPU and no dataset download; the real SIXray subset is only needed for the actual training run (Colab).

**Tech Stack:** PyTorch, torchvision (ResNet18 + existing RGB transforms), scikit-learn (stratified split), uv, pytest, ruff. Dataset: SIXray via Kaggle (`khanhbtq99/sixray`), subsampled.

**Learning-track note (non-negotiable — user must understand the code, not a blackbox):** Task 8 produces a plain-language EXPLAINER + a runnable notebook, and after the phase we do a guided walkthrough. Keep commits small and explain what/why at each checkpoint.

---

## File structure

- `src/simclr_hpl/models.py` — add `ResNet18Encoder` + `build_encoder` factory; parametrize `LinearProbe`/`MLPProbe` with `num_classes`.
- `src/simclr_hpl/data.py` — add `load_binary_image_records` (folder-name → 0/1 mapping) reusing the MVTec/`ImagePathDataset` pattern.
- `configs/xray_screening.yaml` — new experiment config (RGB/224, resnet18, 2 classes, subsample caps).
- `src/simclr_hpl/cli/xray_screening.py` — new CLI: SimCLR pretrain on unlabeled X-ray → linear/MLP probe eval. Console script `xray-screening`.
- `pyproject.toml` — register the `xray-screening` console script.
- `tests/test_models.py`, `tests/test_data_xray.py`, `tests/test_xray_cli.py` — tests (synthetic fixtures).
- `scripts/get_sixray.md` — documented, verified data-acquisition steps (user runs with their own Kaggle token).
- `docs/explainers/phase0-xray-baseline.md` + `notebooks/phase0_xray_walkthrough.ipynb` — learning track.

---

## Task 1: ResNet18 encoder + encoder factory

**Files:**
- Modify: `src/simclr_hpl/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py  (add these)
import torch
from simclr_hpl.models import ResNet18Encoder, build_encoder, Encoder


def test_resnet18_encoder_output_shape():
    encoder = ResNet18Encoder(input_channels=3, pretrained=False)
    assert encoder.output_dim == 512
    out = encoder(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 512)


def test_build_encoder_selects_variant():
    assert isinstance(build_encoder("small", input_channels=1), Encoder)
    assert isinstance(build_encoder("resnet18", input_channels=3), ResNet18Encoder)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py::test_resnet18_encoder_output_shape -v`
Expected: FAIL with `ImportError: cannot import name 'ResNet18Encoder'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/simclr_hpl/models.py  — add near the top imports
from torchvision import models as tv_models

# ... add after the Encoder class ...

class ResNet18Encoder(nn.Module):
    def __init__(self, input_channels: int = 3, pretrained: bool = False) -> None:
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        if input_channels != 3:
            backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.output_dim = 512

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)


def build_encoder(
    name: str = "small",
    input_channels: int = 1,
    pretrained: bool = False,
) -> nn.Module:
    if name == "small":
        return Encoder(input_channels=input_channels)
    if name == "resnet18":
        return ResNet18Encoder(input_channels=input_channels, pretrained=pretrained)
    msg = f"Unknown encoder name: {name!r} (expected 'small' or 'resnet18')"
    raise ValueError(msg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (both new tests + existing tests).

- [ ] **Step 5: Commit**

```bash
git add src/simclr_hpl/models.py tests/test_models.py
git commit -m "feat(models): add ResNet18Encoder and build_encoder factory"
```

---

## Task 2: Parametrize probe class count

**Files:**
- Modify: `src/simclr_hpl/models.py` (`LinearProbe`, `MLPProbe`)
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py  (add)
from simclr_hpl.models import LinearProbe, MLPProbe, ResNet18Encoder


def test_probes_support_binary_num_classes():
    encoder = ResNet18Encoder(input_channels=3)
    linear = LinearProbe(encoder, num_classes=2)
    mlp = MLPProbe(encoder, num_classes=2)
    x = torch.randn(2, 3, 224, 224)
    assert linear(x).shape == (2, 2)
    assert mlp(x).shape == (2, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py::test_probes_support_binary_num_classes -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'num_classes'`.

- [ ] **Step 3: Write minimal implementation**

Replace the two probe classes' `__init__`/`classifier` to accept `num_classes` (default 10 keeps MNIST behavior):

```python
class LinearProbe(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int = 10) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.output_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(inputs))


class MLPProbe(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int = 10) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(inputs))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (new test + existing probe usage still defaults to 10).

- [ ] **Step 5: Commit**

```bash
git add src/simclr_hpl/models.py tests/test_models.py
git commit -m "feat(models): parametrize probe heads with num_classes"
```

---

## Task 3: Binary X-ray folder loader

**Files:**
- Modify: `src/simclr_hpl/data.py`
- Test: `tests/test_data_xray.py` (create)

Reuses `ImagePathDataset`, `split_records_stratified` already in `data.py`. Loader maps each immediate subfolder name to a 0/1 label via a provided mapping (label 1 = threat). Structure-tolerant so it adapts to the real SIXray mirror layout once inspected.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_xray.py  (create)
from pathlib import Path

from PIL import Image

from simclr_hpl.data import load_binary_image_records


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(120, 120, 120)).save(path)


def test_load_binary_image_records_assigns_labels(tmp_path):
    _make_image(tmp_path / "positive" / "a.png")
    _make_image(tmp_path / "positive" / "b.png")
    _make_image(tmp_path / "negative" / "c.png")

    records = load_binary_image_records(
        tmp_path,
        class_to_label={"positive": 1, "negative": 0},
    )

    labels = sorted(int(r["label"]) for r in records)
    assert labels == [0, 1, 1]
    assert all(Path(str(r["path"])).exists() for r in records)


def test_load_binary_image_records_errors_when_empty(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_binary_image_records(tmp_path, class_to_label={"positive": 1})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_xray.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_binary_image_records'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/simclr_hpl/data.py  (add near load_mvtec_records)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_binary_image_records(
    root: str | Path,
    class_to_label: dict[str, int],
) -> list[dict[str, object]]:
    """Scan immediate subfolders of ``root`` and label images by folder name.

    ``class_to_label`` maps a subfolder name to 0 (no-threat) or 1 (threat).
    Folders not present in the mapping are skipped.
    """
    root_path = Path(root)
    if not root_path.exists():
        msg = f"X-ray data root does not exist: {root_path}"
        raise FileNotFoundError(msg)

    records: list[dict[str, object]] = []
    for class_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        if class_dir.name not in class_to_label:
            continue
        label = int(class_to_label[class_dir.name])
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            records.append(
                {"path": image_path, "label": label, "class_name": class_dir.name}
            )
    if not records:
        msg = f"No labeled images found under {root_path} for classes {list(class_to_label)}"
        raise FileNotFoundError(msg)
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data_xray.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simclr_hpl/data.py tests/test_data_xray.py
git commit -m "feat(data): add binary folder-based X-ray image loader"
```

---

## Task 4: X-ray experiment config

**Files:**
- Create: `configs/xray_screening.yaml`

- [ ] **Step 1: Create the config**

```yaml
# configs/xray_screening.yaml
seed: 42
device: auto
output_dir: artifacts/xray_screening

data:
  root: data/sixray            # contains subfolders mapped below
  class_to_label:              # folder name -> 0 (no-threat) / 1 (threat)
    positive: 1
    negative: 0
  image_size: 224
  mean: [0.485, 0.456, 0.406]  # ImageNet (RGB X-ray pseudo-color)
  std: [0.229, 0.224, 0.225]
  pretrain_batch_size: 64
  eval_batch_size: 64
  num_workers: 2
  max_pretrain_images: 4000    # subsample cap for SSL pool (keeps Colab fast)
  max_labeled_per_class: 500   # small labeled budget for the probe

model:
  encoder: resnet18
  input_channels: 3
  pretrained: false
  projection_dim: 128
  num_classes: 2

train:
  epochs: 20
  learning_rate: 0.001
  weight_decay: 0.0001
  temperature: 0.5

evaluation:
  validation_size: 0.2
  epochs: 15
  learning_rate: 0.001
```

- [ ] **Step 2: Verify it loads**

Run: `uv run python -c "from simclr_hpl.config import load_config; print(sorted(load_config('configs/xray_screening.yaml')))"`
Expected: prints `['data', 'device', 'evaluation', 'model', 'output_dir', 'seed', 'train']`.

- [ ] **Step 3: Commit**

```bash
git add configs/xray_screening.yaml
git commit -m "feat(config): add xray_screening experiment config"
```

---

## Task 5: `xray-screening` CLI (SimCLR pretrain + probe eval)

**Files:**
- Create: `src/simclr_hpl/cli/xray_screening.py`
- Modify: `pyproject.toml` (console script)

Reuses: `data.load_binary_image_records`, `data.ImagePathDataset`, `data.ContrastiveViewDataset`, `data.build_rgb_simclr_transform`, `data.build_rgb_normalize_transform`, `data.split_records_stratified`, `data.build_train_val_subsets`, `models.build_encoder`, `models.ProjectionHead`, `models.LinearProbe`, `models.MLPProbe`, `training.{NTXentLoss,pretrain_simclr,train_classifier,evaluate_classifier,freeze_module,trainable_parameters}`, `utils.{seed_everything,resolve_device,ensure_dir,save_checkpoint,save_json}`.

> Note on `ProjectionHead`: its `input_dim` defaults to `Encoder.output_dim` (4096). For ResNet18 we MUST pass `input_dim=encoder.output_dim` (512) explicitly — handled below.

- [ ] **Step 1: Write the CLI**

```python
# src/simclr_hpl/cli/xray_screening.py
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from simclr_hpl.config import load_config
from simclr_hpl.data import (
    ContrastiveViewDataset,
    ImagePathDataset,
    build_rgb_normalize_transform,
    build_rgb_simclr_transform,
    build_train_val_subsets,
    load_binary_image_records,
    split_records_stratified,
)
from simclr_hpl.models import LinearProbe, MLPProbe, ProjectionHead, build_encoder
from simclr_hpl.training import (
    NTXentLoss,
    evaluate_classifier,
    freeze_module,
    pretrain_simclr,
    train_classifier,
    trainable_parameters,
)
from simclr_hpl.utils import (
    ensure_dir,
    resolve_device,
    save_checkpoint,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimCLR + probe eval on binary X-ray data.")
    parser.add_argument("--config", type=Path, default=Path("configs/xray_screening.yaml"))
    return parser.parse_args()


def _subsample(records, cap, seed):
    if cap is None or cap >= len(records):
        return records
    return random.Random(seed).sample(records, cap)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    device = resolve_device(config.get("device", "auto"))
    output_dir = ensure_dir(config["output_dir"])

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    eval_cfg = config["evaluation"]

    records = load_binary_image_records(data_cfg["root"], data_cfg["class_to_label"])
    train_records, test_records = split_records_stratified(
        records, test_size=0.2, seed=config["seed"]
    )

    image_size = int(data_cfg["image_size"])
    mean = tuple(data_cfg["mean"])
    std = tuple(data_cfg["std"])
    simclr_tf = build_rgb_simclr_transform(image_size, mean, std)
    eval_tf = build_rgb_normalize_transform(image_size, mean, std)

    # --- SimCLR pretraining on the (unlabeled) training pool ---
    pretrain_records = _subsample(
        train_records, data_cfg.get("max_pretrain_images"), config["seed"]
    )
    contrastive = ContrastiveViewDataset(
        ImagePathDataset(pretrain_records, transform=None), simclr_tf
    )
    contrastive_loader = DataLoader(
        contrastive,
        batch_size=data_cfg["pretrain_batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=data_cfg["num_workers"],
    )

    encoder = build_encoder(
        model_cfg["encoder"],
        input_channels=model_cfg["input_channels"],
        pretrained=model_cfg.get("pretrained", False),
    ).to(device)
    projection_head = ProjectionHead(
        input_dim=encoder.output_dim, output_dim=model_cfg["projection_dim"]
    ).to(device)
    criterion = NTXentLoss(temperature=train_cfg["temperature"])
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projection_head.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    simclr_history = pretrain_simclr(
        encoder=encoder,
        projection_head=projection_head,
        data_loader=contrastive_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=train_cfg["epochs"],
    )
    save_checkpoint(output_dir / "xray_encoder.pt", {"encoder": encoder.state_dict()})

    # --- Frozen-encoder probe evaluation (linear + MLP) ---
    eval_train = ImagePathDataset(train_records, transform=eval_tf)
    eval_test = ImagePathDataset(test_records, transform=eval_tf)
    probe_train, probe_val = build_train_val_subsets(
        eval_train, validation_size=eval_cfg["validation_size"], seed=config["seed"]
    )
    train_loader = DataLoader(probe_train, batch_size=data_cfg["eval_batch_size"], shuffle=True)
    val_loader = DataLoader(probe_val, batch_size=data_cfg["eval_batch_size"])
    test_loader = DataLoader(eval_test, batch_size=data_cfg["eval_batch_size"])

    num_classes = int(model_cfg["num_classes"])
    probe_criterion = nn.CrossEntropyLoss()
    results: dict[str, object] = {"simclr_final_loss": simclr_history["loss"][-1]}
    for name, probe_cls in (("linear_probe", LinearProbe), ("mlp_probe", MLPProbe)):
        freeze_module(encoder)
        probe = probe_cls(encoder, num_classes=num_classes).to(device)
        probe_optimizer = torch.optim.Adam(
            trainable_parameters(probe), lr=eval_cfg["learning_rate"]
        )
        train_classifier(
            model=probe,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=probe_optimizer,
            criterion=probe_criterion,
            device=device,
            epochs=eval_cfg["epochs"],
        )
        metrics = evaluate_classifier(probe, test_loader, probe_criterion, device)
        results[name] = metrics

    save_json(output_dir / "metrics.json", results)
    print(f"Saved metrics to {output_dir / 'metrics.json'}: {results}")


if __name__ == "__main__":
    main()
```

> **Signatures verified against `src/simclr_hpl/training.py`:** `pretrain_simclr(encoder, projection_head, data_loader, optimizer, criterion, device, epochs)` returns `{"loss": [...]}`; `train_classifier(model, train_loader, val_loader, optimizer, criterion, device, epochs)` and `evaluate_classifier(model, data_loader, criterion, device)` both **require a `criterion`** (`nn.CrossEntropyLoss()`, wired above).

- [ ] **Step 2: Register the console script**

```toml
# pyproject.toml  [project.scripts]  (add line)
xray-screening = "simclr_hpl.cli.xray_screening:main"
```

- [ ] **Step 3: Re-sync so the script is installed**

Run: `uv sync --dev`
Expected: no errors; `xray-screening` becomes available.

- [ ] **Step 4: Commit**

```bash
git add src/simclr_hpl/cli/xray_screening.py pyproject.toml
git commit -m "feat(cli): add xray-screening SimCLR + probe pipeline"
```

---

## Task 6: CLI smoke test on a synthetic dataset (no real data needed)

**Files:**
- Test: `tests/test_xray_cli.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_xray_cli.py  (create)
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
```

- [ ] **Step 2: Run test to verify it fails (then passes after Task 5 wiring is correct)**

Run: `uv run pytest tests/test_xray_cli.py -v`
Expected: initially FAIL if any signature mismatch from Task 5; fix the call sites, then PASS. This is the integration check that the whole Phase 0 pipeline runs end-to-end on CPU with no real data.

- [ ] **Step 3: Run the full suite + lint**

Run: `uv run pytest && uv run ruff check .`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_xray_cli.py
git commit -m "test(cli): end-to-end smoke test for xray-screening on synthetic data"
```

---

## Task 7: Data acquisition (user-run, with safeguards) — REAL training run

**Files:**
- Create: `scripts/get_sixray.md`

> This task downloads real data and is run by the user with their **own (rotated) Kaggle token**. The assistant must not handle the token. `kaggle` is not installed locally; run it via `uv`.

- [ ] **Step 1: Document and verify, size-check BEFORE downloading**

Create `scripts/get_sixray.md` with:

```bash
# 1. (One time) put a freshly-rotated token at ~/.kaggle/kaggle.json, chmod 600.
#    {"username":"<you>","key":"<new-key>"}

# 2. Inspect size FIRST — do not blind-download tens of GB.
uv run --with kaggle kaggle datasets metadata khanhbtq99/sixray
uv run --with kaggle kaggle datasets files khanhbtq99/sixray | head -50

# 3. Download + unzip into data/sixray (only if size is acceptable).
mkdir -p data/sixray
uv run --with kaggle kaggle datasets download khanhbtq99/sixray -p data/sixray --unzip
```

- [ ] **Step 2: Inspect the real folder structure (CHECKPOINT — report back)**

Run: `find data/sixray -maxdepth 2 -type d | head -40`
Then update `configs/xray_screening.yaml`'s `data.root` and `data.class_to_label` to match the real folder names (e.g. the mirror may use `P`/`N` or `positive`/`negative` or class-named folders). The loader from Task 3 is structure-tolerant; only the mapping changes.

- [ ] **Step 3: Real run (Colab GPU recommended; local CPU works on a small subsample)**

Run: `uv run xray-screening --config configs/xray_screening.yaml`
Expected: writes `artifacts/xray_screening/metrics.json` with linear/MLP probe accuracy, and `xray_encoder.pt`.

- [ ] **Step 4: Commit the config adjustment (data/ is gitignored, so only the config)**

```bash
git add configs/xray_screening.yaml
git commit -m "chore(config): align xray_screening to real SIXray folder layout"
```

---

## Task 8: Learning track — EXPLAINER + walkthrough notebook

**Files:**
- Create: `docs/explainers/phase0-xray-baseline.md`
- Create: `notebooks/phase0_xray_walkthrough.ipynb`

- [ ] **Step 1: Write the EXPLAINER**

`docs/explainers/phase0-xray-baseline.md` covers, in plain language tied to code lines: (a) why ResNet18 replaces the 28px `Encoder` for X-ray; (b) how SimCLR's `NTXentLoss` pulls two views of the same bag together; (c) why we freeze the encoder and train only a probe; (d) what the binary threat/no-threat metric means and how it connects to the later review-queue/confidence story.

- [ ] **Step 2: Build a runnable notebook**

`notebooks/phase0_xray_walkthrough.ipynb` reproduces the pipeline on the synthetic fixture (so it runs with no download): build encoder → one SimCLR epoch → probe → print metrics, with a markdown cell explaining each step. Mirrors the `archive/` notebook style.

- [ ] **Step 3: Commit**

```bash
git add docs/explainers/phase0-xray-baseline.md notebooks/phase0_xray_walkthrough.ipynb
git commit -m "docs(phase0): explainer + walkthrough notebook (learning track)"
```

- [ ] **Step 4: Guided walkthrough**

Walk the user through the changed files and the notebook; the user re-runs `uv run pytest` and the notebook themselves.

---

## Self-review notes
- **Spec coverage:** Phase 0 of the spec (ResNet18 encoder, X-ray loader, `configs/xray_screening.yaml`, run existing SSL/probe pipeline on X-ray, learning-track deliverable) — all covered (Tasks 1–8). Detection/RF-DETR/temporal demo are explicitly out of scope (Phases 1–3, separate plans).
- **Placeholders:** none — every code step has concrete code; the one genuine unknown (real SIXray folder names) is handled by a structure-tolerant loader + an explicit inspect-and-adjust checkpoint (Task 7 Step 2).
- **Type consistency:** `build_encoder`/`ResNet18Encoder.output_dim` (512) flows into `ProjectionHead(input_dim=...)` and `EncoderClassifier`/probes via `encoder.output_dim`; probes' `num_classes` defined in Task 2 and used in Task 5. `load_binary_image_records(root, class_to_label)` signature consistent across Tasks 3, 5, 6.
- **Signatures:** verified against `src/simclr_hpl/training.py` — `pretrain_simclr` uses `data_loader=` and returns `{"loss": [...]}`; `train_classifier`/`evaluate_classifier` both require a `criterion`. The Task 5 code reflects this (no outstanding guesses).
```
