from __future__ import annotations

import random
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return self.transform(image), int(label)


class ContrastiveViewDataset(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, _ = self.dataset[index]
        return self.transform(image), self.transform(image)


def load_mnist(root: str | Path, train: bool) -> datasets.MNIST:
    return datasets.MNIST(root=str(root), train=train, transform=None, download=True)


def build_normalize_transform(
    mean: float = MNIST_MEAN,
    std: float = MNIST_STD,
):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )


def build_simclr_transform(
    mean: float = MNIST_MEAN,
    std: float = MNIST_STD,
):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=28, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomApply([transforms.RandomRotation(25)], p=0.7),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10)],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )


def build_supervised_augmentation_transforms(
    mean: float = MNIST_MEAN,
    std: float = MNIST_STD,
):
    return [
        transforms.Compose(
            [
                transforms.RandomRotation(45),
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,)),
            ]
        ),
        transforms.Compose(
            [
                transforms.RandomResizedCrop(28, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,)),
            ]
        ),
        transforms.Compose(
            [
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,)),
            ]
        ),
        transforms.Compose(
            [
                transforms.RandomAffine(degrees=0, shear=30),
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,)),
            ]
        ),
        transforms.Compose(
            [
                transforms.RandomRotation(45),
                transforms.RandomResizedCrop(28, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomAffine(degrees=0, shear=30),
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,)),
            ]
        ),
    ]


def sample_balanced_indices(dataset: datasets.MNIST, per_class: int, seed: int) -> tuple[list[int], list[int]]:
    generator = random.Random(seed)
    labeled_indices: list[int] = []
    targets = dataset.targets.tolist()
    for label in sorted(set(targets)):
        class_indices = [index for index, target in enumerate(targets) if target == label]
        labeled_indices.extend(generator.sample(class_indices, per_class))
    labeled_lookup = set(labeled_indices)
    unlabeled_indices = [index for index in range(len(dataset)) if index not in labeled_lookup]
    return sorted(labeled_indices), unlabeled_indices


def targets_from_indices(dataset: datasets.MNIST, indices: list[int]) -> list[int]:
    return [int(dataset.targets[index]) for index in indices]


def build_tensor_dataset(dataset: Dataset, indices: list[int], transform) -> TensorDataset:
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for index in indices:
        image, label = dataset[index]
        images.append(transform(image))
        labels.append(int(label))
    return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))


def build_augmented_tensor_dataset(
    dataset: Dataset,
    indices: list[int],
    base_transform,
    augmentation_transforms,
    copies_per_transform: int,
    seed: int,
) -> TensorDataset:
    random.seed(seed)
    torch.manual_seed(seed)
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for index in indices:
        image, label = dataset[index]
        label = int(label)
        images.append(base_transform(image))
        labels.append(label)
        for augmentation in augmentation_transforms:
            for _ in range(copies_per_transform):
                images.append(augmentation(image))
                labels.append(label)
    return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))


def collect_labels(dataset: Dataset) -> list[int]:
    if isinstance(dataset, TensorDataset):
        label_tensor = dataset.tensors[1]
        return [int(value) for value in label_tensor.tolist()]
    if isinstance(dataset, Subset):
        parent_labels = collect_labels(dataset.dataset)
        return [parent_labels[index] for index in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        labels: list[int] = []
        for child in dataset.datasets:
            labels.extend(collect_labels(child))
        return labels
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if isinstance(targets, torch.Tensor):
            return [int(value) for value in targets.tolist()]
        return [int(value) for value in targets]
    msg = f"Unsupported dataset type for label extraction: {type(dataset)!r}"
    raise TypeError(msg)


def build_train_val_subsets(
    dataset: Dataset,
    validation_size: float,
    seed: int,
) -> tuple[Subset, Subset]:
    labels = collect_labels(dataset)
    indices = list(range(len(labels)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=validation_size,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )
    return Subset(dataset, train_indices), Subset(dataset, val_indices)
