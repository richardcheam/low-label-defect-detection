"""Generate notebooks/phase0_sixray_kaggle.ipynb.

Run: uv run --with nbformat python scripts/build_kaggle_notebook.py
The notebook is meant to be uploaded to / pasted into a Kaggle Notebook with the
`khanhbtq99/sixray` dataset attached and GPU enabled. It reuses the simclr_hpl
library to run the real Phase 0 pipeline (SimCLR pretrain -> frozen probe eval)
on actual X-ray data.
"""
from __future__ import annotations

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(
    """
# Phase 0 (real run): Low-Label X-ray Threat Screening on SIXray

This notebook runs the **real** Phase 0 pipeline on actual X-ray baggage images:
self-supervised **SimCLR** pretraining on a large *unlabeled* pool, then a
**frozen-encoder linear + MLP probe** trained on only a *few labeled* examples
(the low-label thesis). It reuses the `simclr_hpl` package from the repo.

## Before you run
1. **Add data:** in the right sidebar → *Add Input* → search **`khanhbtq99/sixray`** → add it.
2. **Enable GPU:** Settings (⚙️) → *Accelerator* → **GPU T4 / P100**.
3. **Internet on:** Settings → *Internet* → **On** (needed to clone the repo).
4. Run cells top to bottom. **Read the output of the "Inspect" cell** — if the
   detected label counts look wrong, tell me and we'll adjust the scheme.
"""
)

md("## 1. Clone the repo and install the package")
code(
    """
import os
REPO_URL = "https://github.com/richardcheam/low-label-defect-detection.git"
if not os.path.isdir("low-label-defect-detection"):
    !git clone --depth 1 {REPO_URL}
%cd low-label-defect-detection
# torch/torchvision are already present on the Kaggle GPU image; this installs
# the simclr_hpl package + any missing light deps.
!pip -q install -e .
"""
)

md(
    """
## 2. Locate & **inspect** the SIXray data — CHECK THIS OUTPUT
SIXray commonly labels images by **filename prefix** (`P…` = positive/contains a
prohibited item, `N…` = negative/clean) rather than by folder. This cell figures
out which scheme the mirror uses and prints histograms so we can confirm.
"""
)
code(
    """
import pathlib, collections, itertools

INPUT = pathlib.Path("/kaggle/input")
print("Attached inputs:", [p.name for p in INPUT.iterdir()])

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
img_paths = [p for p in INPUT.rglob("*") if p.suffix.lower() in IMG_SUFFIXES]
print("Total image files found:", len(img_paths))
print("\\nSample paths:")
for p in itertools.islice(img_paths, 8):
    print("  ", p)

first_char = collections.Counter(p.name[:1].upper() for p in img_paths)
parents = collections.Counter(p.parent.name for p in img_paths)
print("\\nFilename first-char histogram:", dict(first_char))
print("Parent-folder histogram (top 10):", parents.most_common(10))
"""
)

md(
    """
## 3. Build labeled records (adaptive: prefix **or** folder)
`LABEL_SCHEME = "auto"` picks based on the inspection above. Override to
`"prefix"` or `"folder"` if needed. We then **balance** the two classes and
**cap** the pool size so a first run is fast on a free GPU.
"""
)
code(
    """
import random

LABEL_SCHEME = "auto"      # "auto" | "prefix" | "folder"
MAX_PER_CLASS = 3000       # cap the (unlabeled) pool per class for a tractable first run
SEED = 42

POSITIVE_FOLDERS = {"positive", "p", "threat", "prohibited", "pos"}
NEGATIVE_FOLDERS = {"negative", "n", "normal", "clean", "good", "background", "neg"}


def _label_prefix(p):
    c = p.name[:1].upper()
    return 1 if c == "P" else 0 if c == "N" else None


def _label_folder(p):
    name = p.parent.name.lower()
    return 1 if name in POSITIVE_FOLDERS else 0 if name in NEGATIVE_FOLDERS else None


def _choose_scheme():
    if LABEL_SCHEME != "auto":
        return LABEL_SCHEME
    prefix_hits = sum(_label_prefix(p) is not None for p in img_paths)
    folder_hits = sum(_label_folder(p) is not None for p in img_paths)
    print(f"auto-detect: prefix matches={prefix_hits}, folder matches={folder_hits}")
    return "prefix" if prefix_hits >= folder_hits else "folder"


scheme = _choose_scheme()
labeler = _label_prefix if scheme == "prefix" else _label_folder
print("Using label scheme:", scheme)

records = []
for p in img_paths:
    lbl = labeler(p)
    if lbl is not None:
        records.append({"path": str(p), "label": int(lbl)})

by_label = {0: [], 1: []}
for r in records:
    by_label[r["label"]].append(r)
print("raw label counts:", {k: len(v) for k, v in by_label.items()})

rng = random.Random(SEED)
balanced = []
for lbl, rs in by_label.items():
    rng.shuffle(rs)
    balanced.extend(rs[:MAX_PER_CLASS])
rng.shuffle(balanced)
print("balanced+capped counts:",
      {lbl: sum(r["label"] == lbl for r in balanced) for lbl in (0, 1)})
assert balanced, "No labeled records built — check the inspection output / scheme."
"""
)

md(
    """
## 4. Split, then SimCLR-pretrain the ResNet18 encoder (unlabeled)
We reuse the library exactly as the `xray-screening` CLI does. The encoder trains
contrastively on the *unlabeled* pool — labels are not used here.
"""
)
code(
    """
import torch
from torch import nn
from torch.utils.data import DataLoader

from simclr_hpl.data import (
    ImagePathDataset, ContrastiveViewDataset,
    build_rgb_simclr_transform, build_rgb_normalize_transform,
    split_records_stratified, build_train_val_subsets, subsample_per_class,
)
from simclr_hpl.models import build_encoder, ProjectionHead, LinearProbe, MLPProbe
from simclr_hpl.training import (
    NTXentLoss, pretrain_simclr, train_classifier, evaluate_classifier,
    freeze_module, trainable_parameters,
)
from simclr_hpl.utils import seed_everything, resolve_device, save_json

# ---- knobs (tuned for a free Kaggle GPU first run) ----
IMAGE_SIZE = 224
PRETRAIN_EPOCHS = 15
PROBE_EPOCHS = 20
BATCH = 128
MAX_LABELED_PER_CLASS = 500   # the low-label budget for the probe
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

seed_everything(SEED)
device = resolve_device("auto")
print("device:", device)

train_records, test_records = split_records_stratified(balanced, test_size=0.2, seed=SEED)
simclr_tf = build_rgb_simclr_transform(IMAGE_SIZE, MEAN, STD)
eval_tf = build_rgb_normalize_transform(IMAGE_SIZE, MEAN, STD)

contrastive = ContrastiveViewDataset(ImagePathDataset(train_records, transform=None), simclr_tf)
contrastive_loader = DataLoader(contrastive, batch_size=BATCH, shuffle=True,
                                drop_last=True, num_workers=2, pin_memory=True)

encoder = build_encoder("resnet18", input_channels=3, pretrained=False).to(device)
projection_head = ProjectionHead(input_dim=encoder.output_dim, output_dim=128).to(device)
criterion = NTXentLoss(temperature=0.5)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(projection_head.parameters()),
    lr=1e-3, weight_decay=1e-4,
)
history = pretrain_simclr(
    encoder=encoder, projection_head=projection_head, data_loader=contrastive_loader,
    optimizer=optimizer, criterion=criterion, device=device, epochs=PRETRAIN_EPOCHS,
)
print("SimCLR final loss:", history["loss"][-1])
"""
)

md(
    """
## 5. Frozen-encoder probe evaluation (few labels)
The encoder is **frozen** (and kept in eval mode by the probe so its BatchNorm
stats don't drift). Only `MAX_LABELED_PER_CLASS` labels per class are used to
train each probe — that's the point: good unsupervised features → few labels needed.
"""
)
code(
    """
labeled_records = subsample_per_class(train_records, MAX_LABELED_PER_CLASS, SEED)
print("labeled budget used:",
      {lbl: sum(r["label"] == lbl for r in labeled_records) for lbl in (0, 1)})

eval_train = ImagePathDataset(labeled_records, transform=eval_tf)
eval_test = ImagePathDataset(test_records, transform=eval_tf)
probe_train, probe_val = build_train_val_subsets(eval_train, validation_size=0.2, seed=SEED)

train_loader = DataLoader(probe_train, batch_size=BATCH, shuffle=True,
                          drop_last=len(probe_train) > BATCH, num_workers=2)
val_loader = DataLoader(probe_val, batch_size=BATCH, num_workers=2)
test_loader = DataLoader(eval_test, batch_size=BATCH, num_workers=2)

probe_criterion = nn.CrossEntropyLoss()
results = {"simclr_final_loss": history["loss"][-1],
           "n_labeled": len(labeled_records), "n_test": len(test_records)}
for name, probe_cls in (("linear_probe", LinearProbe), ("mlp_probe", MLPProbe)):
    freeze_module(encoder)
    probe = probe_cls(encoder, num_classes=2).to(device)
    opt = torch.optim.Adam(trainable_parameters(probe), lr=1e-3)
    train_classifier(model=probe, train_loader=train_loader, val_loader=val_loader,
                     optimizer=opt, criterion=probe_criterion, device=device, epochs=PROBE_EPOCHS)
    metrics = evaluate_classifier(probe, test_loader, probe_criterion, device)
    results[name] = metrics
    print(name, metrics)

import pathlib
out = pathlib.Path("/kaggle/working")
save_json(out / "metrics.json", results)
torch.save({"encoder": encoder.state_dict()}, out / "xray_encoder.pt")
print("\\nSaved /kaggle/working/metrics.json and xray_encoder.pt")
results
"""
)

md(
    """
## 6. Quick visual + takeaways
A simple bar chart of probe accuracy. Both files in `/kaggle/working` are
downloadable from the *Output* tab (the `xray_encoder.pt` checkpoint feeds the
later demo phase).
"""
)
code(
    """
import matplotlib.pyplot as plt

names = ["linear_probe", "mlp_probe"]
accs = [results[n]["accuracy"] for n in names]
plt.figure(figsize=(4, 3))
plt.bar(names, accs, color=["#4C72B0", "#55A868"])
plt.ylim(0, 1)
plt.ylabel("test accuracy")
plt.title(f"Frozen-probe acc · {results['n_labeled']} labels")
for i, a in enumerate(accs):
    plt.text(i, a + 0.02, f"{a:.3f}", ha="center")
plt.tight_layout()
plt.savefig("/kaggle/working/probe_accuracy.png", dpi=120)
plt.show()
"""
)

md(
    """
## What to do with the results
- **Download** `/kaggle/working/metrics.json` + `xray_encoder.pt` (Output tab).
- Paste the printed `metrics` back to me — we'll interpret the numbers (accuracy
  vs label budget, linear vs MLP) and decide whether to scale up (`MAX_PER_CLASS`,
  epochs) or move to **Phase 1 (RF-DETR detection)**.
- The `xray_encoder.pt` checkpoint is the SimCLR-pretrained backbone the temporal
  demo phase will reuse.
"""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

with open("notebooks/phase0_sixray_kaggle.ipynb", "w", encoding="utf-8") as fh:
    nbf.write(nb, fh)
print("wrote notebooks/phase0_sixray_kaggle.ipynb with", len(cells), "cells")
