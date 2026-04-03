# Portfolio Rework Roadmap

This repo already has a good story: you explored label-efficient learning, got strong linear-probe results from self-supervision, and improved a low-label classifier with hard pseudo-labeling. The main upgrade now is presentation and reproducibility.

## What Makes This Strong In 2026

- It shows self-supervised learning, semi-supervised learning, and experiment design in one project.
- It creates a bridge between research thinking and engineering execution.
- It gives you a clean path to talk about tradeoffs: label scarcity, confidence thresholds, representation quality, and reproducibility.

## Best Next Extensions

### 1. Add a stronger benchmark than MNIST

Move the exact same pipeline to `FashionMNIST` and `CIFAR10`.

- Why it helps: MNIST alone looks academic; multi-dataset evidence looks professional.
- What to measure: baseline supervised, SimCLR + linear probe, single-round pseudo-labeling, iterative pseudo-labeling.

### 2. Add ablations and reliability metrics

Turn the project into a small experimental study.

- Compare thresholds like `0.95`, `0.97`, `0.99`.
- Compare pseudo-label schedule strategies.
- Track precision of pseudo-labels, not just downstream accuracy.
- Add calibration metrics such as expected calibration error.

### 3. Add experiment tracking

Use MLflow or Weights & Biases to log runs, configs, metrics, and checkpoints.

- Why it helps: this is one of the fastest ways to make the repo feel industry-ready.
- What to log: config, seed, dataset split, epoch metrics, pseudo-label counts, final test accuracy.

### 4. Reuse the SimCLR encoder inside the pseudo-labeling pipeline

This has now been started in the refactored codebase through a transfer benchmark:

- initialize the pseudo-labeling classifier from the pretrained encoder
- compare random initialization vs SimCLR initialization
- report sample efficiency across 100, 250, and 500 labels

The best follow-up is to move from MNIST-only results to a multi-dataset benchmark.

### 5. Add production-style tooling

- CI for tests and linting
- structured configs
- saved artifacts and checkpoints
- clear training entry points
- optional Dockerfile for fully reproducible runs

## Resume / Portfolio Positioning

Use language like this:

> Built a reproducible self-supervised and semi-supervised computer vision pipeline in PyTorch, converting exploratory notebook experiments into a config-driven Python package with CI, tests, and experiment scripts.

> Benchmarked SimCLR representation learning and hard pseudo-labeling under extreme label scarcity, reaching strong MNIST performance with only 100 labeled examples.
