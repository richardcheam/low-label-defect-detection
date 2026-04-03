# SimCLR Hard Pseudo-Labeling

This project studies label-efficient image learning on MNIST by combining:

- SimCLR-style self-supervised representation learning
- linear probing and frozen-encoder MLP evaluation
- semi-supervised CNN training with hard pseudo-labeling
- iterative pseudo-label expansion under confidence thresholds

The original work lived in notebooks and already produced strong results. This repo now includes a more professional baseline structure with a Python package, YAML configs, CLI entry points, tests, and CI so it can evolve into a stronger portfolio project for Data Scientist and ML/AI Engineer roles.

Reference paper: https://arxiv.org/abs/2002.05709

## Original Results

These results came from the notebook-based version of the project.

### SimCLR pretraining

| Evaluation head | Test accuracy |
|---|---:|
| Linear probe | 98.55% |
| Frozen-encoder MLP | 98.44% |

### Semi-supervised CNN

| Setup | Test accuracy |
|---|---:|
| CNN + augmentation | 93.66% |
| CNN + augmentation + hard pseudo-labeling | 94.12% |
| Iterative pseudo-labeling | 97.18% |

## Repo Layout

```text
.
├── configs/                  # experiment configs
├── docs/                     # roadmap and project notes
├── src/simclr_hpl/           # reusable package code
├── tests/                    # smoke tests
├── SimCLR.ipynb              # legacy research notebook
├── CNN_semi_supervised.ipynb # legacy research notebook
└── pyproject.toml            # uv-compatible project metadata
```

## Workflow Diagram

```mermaid
flowchart TD
    A[Raw MNIST dataset] --> B[Balanced labeled subset]
    A --> C[Unlabeled subset]

    A --> D[SimCLR augmentations]
    D --> E[SimCLR pretraining]
    E --> F[Encoder checkpoint]
    E --> G[Linear probe and MLP probe evaluation]

    B --> H[Supervised augmentations]
    H --> I[Initial classifier training]

    F --> J[Transfer initialization benchmark]
    I --> K[Hard pseudo-label generation]
    C --> K
    K --> L[Combined labeled plus pseudo-labeled dataset]
    L --> M[Retraining and iterative pseudo-labeling]

    J --> M
    M --> N[Test evaluation]
    G --> O[Artifacts and metrics JSON]
    N --> O
```

## Current Workflow

The project now supports three main experiment paths:

1. `simclr-train`
   Pretrain an encoder with SimCLR and evaluate feature quality with a linear probe and an MLP probe.
2. `pseudo-label-train`
   Train a low-label classifier, generate hard pseudo-labels on unlabeled data, and iterate.
3. `transfer-benchmark`
   Compare pseudo-labeling performance from random initialization vs SimCLR initialization at multiple label budgets.

## Professional Setup With `uv`

### 1. Install and pin Python

```bash
uv python install 3.11
uv python pin 3.11
```

### 2. Create the environment and install dependencies

```bash
uv sync --dev
```

### 3. Run tests and lint checks

```bash
uv run pytest
uv run ruff check .
```

### 4. Run the experiments

```bash
uv run simclr-train --config configs/simclr_mnist.yaml
uv run pseudo-label-train --config configs/pseudo_label_mnist.yaml
uv run transfer-benchmark --config configs/transfer_pseudo_label_mnist.yaml
```

Artifacts and metrics are written under `artifacts/`.

## New Extension: SimCLR Transfer Into Pseudo-Labeling

The most valuable new benchmark in this repo compares:

- random initialization
- SimCLR-pretrained initialization

across balanced label budgets of:

- 100 labels
- 250 labels
- 500 labels

The benchmark uses the same `EncoderClassifier` architecture for both conditions, so the comparison is about representation initialization rather than model size.

```bash
uv run transfer-benchmark --config configs/transfer_pseudo_label_mnist.yaml
```

If `artifacts/simclr_mnist/simclr_encoder.pt` does not exist yet, the benchmark can pretrain a SimCLR encoder automatically and reuse it for the comparison.

## What Was Reworked

- Notebook logic has been lifted into `src/simclr_hpl/`
- experiments are config-driven through YAML files
- `uv` is now the expected environment workflow
- tests and GitHub Actions CI were added
- the original notebooks are still kept as historical references

## Why This Is A Better Portfolio Project Now

- It shows both research experimentation and engineering discipline.
- It is reproducible enough for reviewers to run locally.
- It gives you a clear story around self-supervision, pseudo-label confidence thresholds, and working under extreme label scarcity.

## Visualization

Visualization is worth investing in here. For a portfolio project, good figures make the difference between "interesting code" and "clear ML story."

The strongest visualizations for this repo are:

- a label-budget comparison plot: `100 vs 250 vs 500` labels for `random` and `simclr` initialization
- a training-curve plot: train and validation accuracy or loss over epochs
- a pseudo-label growth plot: how many confident pseudo-labels are added per iteration
- a confidence-threshold plot: threshold vs pseudo-label count vs final accuracy
- a feature-space plot: t-SNE or UMAP of encoder features before and after transfer

My take: t-SNE can look nice, but it should not be the main evidence. For hiring and project credibility, the most important figures are the benchmark comparison plots and pseudo-labeling dynamics. Those show decision-making, not just pretty embeddings.

If we keep extending this repo, I’d recommend adding a small plotting module that reads `artifacts/*/metrics.json` and automatically generates publication-style figures into `artifacts/plots/`.

## Strong Portfolio Angle

This project now lets you tell a stronger story than "I trained a model in notebooks":

- you designed low-label learning experiments
- you refactored them into a reproducible Python project
- you benchmarked representation transfer from self-supervised pretraining into semi-supervised training

That is much closer to how modern ML engineering work is presented.

More ideas are in [docs/portfolio_roadmap.md](/Users/macbookpro/Desktop/git/SimCLR-HardPseudoLabeling/docs/portfolio_roadmap.md).
