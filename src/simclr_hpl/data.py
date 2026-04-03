from __future__ import annotations

import random
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset
from torchvision.datasets.folder import default_loader
from torchvision import datasets, transforms

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


class ImagePathDataset(Dataset):
    def __init__(self, records: list[dict[str, object]], transform=None):
        self.records = records
        self.transform = transform
        self.targets = [int(record["label"]) for record in records]
        self.labels = self.targets

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = default_loader(str(record["path"]))
        if self.transform is not None:
            image = self.transform(image)
        return image, int(record["label"])


def load_mnist(root: str | Path, train: bool) -> datasets.MNIST:
    return datasets.MNIST(root=str(root), train=train, transform=None, download=True)


def load_mvtec_records(root: str | Path, category: str) -> list[dict[str, object]]:
    category_root = Path(root) / category
    if not category_root.exists():
        msg = f"MVTec category path does not exist: {category_root}"
        raise FileNotFoundError(msg)

    records: list[dict[str, object]] = []
    for split in ["train", "test"]:
        split_root = category_root / split
        if not split_root.exists():
            continue
        for defect_type_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            defect_type = defect_type_dir.name
            label = 0 if defect_type == "good" else 1
            for image_path in sorted(defect_type_dir.glob("*")):
                if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                    continue
                records.append(
                    {
                        "path": image_path,
                        "label": label,
                        "defect_type": defect_type,
                        "split": split,
                    }
                )
    if not records:
        msg = f"No image files found under {category_root}"
        raise FileNotFoundError(msg)
    return records


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


def build_rgb_normalize_transform(
    image_size: int = 224,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_rgb_simclr_transform(
    image_size: int = 224,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(size=image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)],
                p=0.5,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
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


def build_rgb_supervised_augmentation_transforms(
    image_size: int = 224,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
):
    return [
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
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


def sample_by_class_counts(
    labels: list[int],
    class_counts: dict[int, int],
    seed: int,
) -> tuple[list[int], list[int]]:
    generator = random.Random(seed)
    labeled_indices: list[int] = []
    for label, count in class_counts.items():
        class_indices = [index for index, value in enumerate(labels) if value == label]
        if len(class_indices) < count:
            msg = f"Not enough samples for class {label}: requested {count}, found {len(class_indices)}"
            raise ValueError(msg)
        labeled_indices.extend(generator.sample(class_indices, count))
    labeled_lookup = set(labeled_indices)
    unlabeled_indices = [index for index in range(len(labels)) if index not in labeled_lookup]
    return sorted(labeled_indices), unlabeled_indices


def split_records_stratified(
    records: list[dict[str, object]],
    test_size: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    labels = [int(record["label"]) for record in records]
    train_indices, test_indices = train_test_split(
        list(range(len(records))),
        test_size=test_size,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )
    train_records = [records[index] for index in train_indices]
    test_records = [records[index] for index in test_indices]
    return train_records, test_records


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
    if hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")
        return [int(value) for value in labels]
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
