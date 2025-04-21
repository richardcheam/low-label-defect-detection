# SimCLR-Augmented-PseudoLabeling-MNIST

This project explores the combination of **self-supervised learning (SimCLR: A Simple Framework for Contrastive Learning of Visual Representations)** and **semi-supervised learning with pseudo-labeling** for image classification on the MNIST dataset, but only having access to 100 labels. The aim is to train performant models with very limited labeled data. 

SimCLR: https://arxiv.org/abs/2002.05709


---

## Overview

I tackle the problem of learning from limited labels by combining:
- **SimCLR**: A contrastive self-supervised learning framework to pretrain a feature encoder.
- **Linear probing / MLP classification**: To evaluate SimCLR representation quality.
- **Semi-supervised CNN with Pseudo-labeling**: A supervised baseline that bootstraps predictions on unlabeled data with high-confidence thresholds.
- **Iterative Pseudo-labeling**: Refining the model further by adding confidently predicted labels over multiple rounds.

---

## Experimental Setup

### Dataset
- **MNIST** (60,000 training samples, 10,000 test samples)
- Only **100 labeled samples** were used initially (10 per digit)
- The rest were used as unlabeled data

---

## SimCLR Pretraining (Unsupervised)

| Model           | Test Accuracy |
|----------------|----------------|
| Linear Probing | 98.55%         |
| MLP Classifier | 98.44%         |

- SimCLR was trained without using any labels
- Strong linear separability confirms good feature learning

---

## Semi-Supervised CNN Training

### 1. CNN with Data Augmentation
- Labeled data: 100 samples
- Augmented dataset size: 5100
- Accuracy: 93.66%

### 2. CNN + Pseudo-Labeling (One-Shot)
- Hard threshold: 0.99
- Pseudo-labeled: 44,161 samples
- Combined training set size: 49,261
- Accuracy: 94.12%

---

## Iterative Pseudo-labeling (5 Iterations)

Each iteration:
1. Train on labeled + previously pseudo-labeled data
2. Predict on remaining unlabeled data
3. Add only high-confidence predictions (soften threshold at each iteration)

### Final Iteration (Iteration 5)
- Remaining unlabeled: 905
- New pseudo-labeled: 363
- Total training samples: 64,458
- Final Accuracy: 97.18%

---
## Future work

Use SimCLR encoder in the pseudo-labeling pipeline
Extend experiments to more complex datasets

-- 

## Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- matplotlib
- seaborn

Install dependencies:
```bash
pip install torch torchvision scikit-learn matplotlib seaborn
