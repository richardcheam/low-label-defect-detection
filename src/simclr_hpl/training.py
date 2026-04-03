from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)
        representations = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)
        similarity = representations @ representations.T
        similarity = similarity / self.temperature

        self_mask = torch.eye(2 * batch_size, device=similarity.device, dtype=torch.bool)
        positive_mask = torch.zeros_like(self_mask)
        positive_mask[:batch_size, batch_size:] = torch.eye(
            batch_size, device=similarity.device, dtype=torch.bool
        )
        positive_mask[batch_size:, :batch_size] = torch.eye(
            batch_size, device=similarity.device, dtype=torch.bool
        )

        positives = similarity[positive_mask].view(2 * batch_size, 1)
        negatives = similarity[~(self_mask | positive_mask)].view(2 * batch_size, -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=similarity.device)
        return F.cross_entropy(logits, labels)


def pretrain_simclr(
    encoder: nn.Module,
    projection_head: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
) -> dict[str, list[float]]:
    history = {"loss": []}
    for _ in tqdm(range(epochs), desc="SimCLR pretraining"):
        encoder.train()
        projection_head.train()
        running_loss = 0.0
        seen_samples = 0
        for view_i, view_j in data_loader:
            view_i = view_i.to(device)
            view_j = view_j.to(device)

            features_i = encoder(view_i)
            features_j = encoder(view_j)
            projections_i = projection_head(features_i)
            projections_j = projection_head(features_j)

            loss = criterion(projections_i, projections_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = view_i.size(0)
            running_loss += loss.item() * batch_size
            seen_samples += batch_size
        history["loss"].append(running_loss / max(seen_samples, 1))
    return history


def evaluate_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += inputs.size(0)
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def collect_prediction_outputs(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, list[float] | list[int]]:
    model.eval()
    probabilities: list[list[float]] = []
    predictions: list[int] = []
    targets_list: list[int] = []
    confidences: list[float] = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = softmax(logits).cpu()
            batch_confidences, batch_predictions = probs.max(dim=1)
            probabilities.extend(probs.tolist())
            predictions.extend(batch_predictions.tolist())
            confidences.extend(batch_confidences.tolist())
            targets_list.extend(targets.tolist())
    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "targets": targets_list,
        "confidences": confidences,
    }


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
) -> dict[str, list[float]]:
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    for _ in tqdm(range(epochs), desc="Supervised training"):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_correct += (logits.argmax(dim=1) == targets).sum().item()
            running_total += inputs.size(0)

        validation_metrics = evaluate_classifier(model, val_loader, criterion, device)
        history["train_loss"].append(running_loss / max(running_total, 1))
        history["train_accuracy"].append(running_correct / max(running_total, 1))
        history["val_loss"].append(validation_metrics["loss"])
        history["val_accuracy"].append(validation_metrics["accuracy"])
    return history


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def trainable_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def generate_pseudo_labels(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float,
    dataset_indices: list[int] | None = None,
) -> tuple[TensorDataset | None, list[int], list[float]]:
    model.eval()
    softmax = nn.Softmax(dim=1)
    pseudo_images: list[torch.Tensor] = []
    pseudo_labels: list[int] = []
    pseudo_indices: list[int] = []
    confidences: list[float] = []
    offset = 0

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            probabilities = softmax(model(inputs))
            batch_confidences, predictions = probabilities.max(dim=1)
            keep_mask = batch_confidences >= threshold

            if keep_mask.any():
                pseudo_images.extend(inputs[keep_mask].cpu())
                pseudo_labels.extend(predictions[keep_mask].cpu().tolist())
                confidences.extend(batch_confidences[keep_mask].cpu().tolist())
                if dataset_indices is not None:
                    batch_indices = dataset_indices[offset : offset + inputs.size(0)]
                    keep_lookup = keep_mask.cpu().tolist()
                    pseudo_indices.extend(
                        index for index, keep in zip(batch_indices, keep_lookup, strict=False) if keep
                    )
            offset += inputs.size(0)

    if not pseudo_images:
        return None, [], []

    dataset = TensorDataset(
        torch.stack(pseudo_images),
        torch.tensor(pseudo_labels, dtype=torch.long),
    )
    return dataset, pseudo_indices, confidences
