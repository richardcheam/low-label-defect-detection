from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from simclr_hpl.config import load_config
from simclr_hpl.data import (
    TransformDataset,
    build_augmented_tensor_dataset,
    build_normalize_transform,
    build_supervised_augmentation_transforms,
    build_tensor_dataset,
    build_train_val_subsets,
    load_mnist,
    sample_balanced_indices,
)
from simclr_hpl.models import SemiSupervisedCNN
from simclr_hpl.training import evaluate_classifier, generate_pseudo_labels, train_classifier
from simclr_hpl.utils import ensure_dir, resolve_device, save_checkpoint, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hard pseudo-labeling pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pseudo_label_mnist.yaml"),
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def build_loaders(
    dataset,
    test_dataset,
    seed: int,
    validation_size: float,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
):
    train_subset, val_subset = build_train_val_subsets(
        dataset,
        validation_size=validation_size,
        seed=seed,
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed_everything(config["seed"])
    device = resolve_device(config.get("device", "auto"))
    output_dir = ensure_dir(config["output_dir"])

    data_config = config["data"]
    experiment_config = config["experiment"]
    train_config = config["train"]
    pseudo_config = config["pseudo_labeling"]

    raw_train = load_mnist(data_config["root"], train=True)
    raw_test = load_mnist(data_config["root"], train=False)

    labeled_indices, unlabeled_indices = sample_balanced_indices(
        raw_train,
        per_class=experiment_config["labeled_per_class"],
        seed=config["seed"],
    )

    base_transform = build_normalize_transform(data_config["mean"], data_config["std"])
    augmentation_transforms = build_supervised_augmentation_transforms(
        data_config["mean"],
        data_config["std"],
    )
    augmented_labeled_dataset = build_augmented_tensor_dataset(
        raw_train,
        labeled_indices,
        base_transform=base_transform,
        augmentation_transforms=augmentation_transforms,
        copies_per_transform=experiment_config["augmentation_copies_per_transform"],
        seed=config["seed"],
    )
    test_dataset = TransformDataset(raw_test, base_transform)

    train_loader, val_loader, test_loader = build_loaders(
        dataset=augmented_labeled_dataset,
        test_dataset=test_dataset,
        seed=config["seed"],
        validation_size=experiment_config["validation_size"],
        batch_size=data_config["batch_size"],
        eval_batch_size=data_config["eval_batch_size"],
        num_workers=data_config["num_workers"],
    )

    criterion = nn.CrossEntropyLoss()

    baseline_model = SemiSupervisedCNN().to(device)
    baseline_optimizer = torch.optim.Adam(
        baseline_model.parameters(),
        lr=train_config["learning_rate"],
    )
    baseline_history = train_classifier(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=baseline_optimizer,
        criterion=criterion,
        device=device,
        epochs=train_config["epochs"],
    )
    baseline_test = evaluate_classifier(baseline_model, test_loader, criterion, device)

    unlabeled_dataset = build_tensor_dataset(raw_train, unlabeled_indices, base_transform)
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=data_config["eval_batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
    )
    pseudo_dataset, pseudo_indices, _ = generate_pseudo_labels(
        model=baseline_model,
        data_loader=unlabeled_loader,
        device=device,
        threshold=pseudo_config["threshold"],
        dataset_indices=unlabeled_indices,
    )

    single_round_history = None
    single_round_test = None
    combined_dataset = augmented_labeled_dataset
    pseudo_model = None

    if pseudo_dataset is not None:
        combined_dataset = ConcatDataset([augmented_labeled_dataset, pseudo_dataset])
        train_loader, val_loader, test_loader = build_loaders(
            dataset=combined_dataset,
            test_dataset=test_dataset,
            seed=config["seed"],
            validation_size=experiment_config["validation_size"],
            batch_size=data_config["batch_size"],
            eval_batch_size=data_config["eval_batch_size"],
            num_workers=data_config["num_workers"],
        )
        pseudo_model = SemiSupervisedCNN().to(device)
        pseudo_optimizer = torch.optim.Adam(
            pseudo_model.parameters(),
            lr=train_config["learning_rate"],
        )
        single_round_history = train_classifier(
            model=pseudo_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=pseudo_optimizer,
            criterion=criterion,
            device=device,
            epochs=train_config["epochs"],
        )
        single_round_test = evaluate_classifier(pseudo_model, test_loader, criterion, device)

    iterative_results: list[dict[str, float | int]] = []
    if pseudo_model is not None:
        current_threshold = pseudo_config["threshold"]
        all_pseudo_indices = set(pseudo_indices)
        for iteration in range(pseudo_config["iterations"]):
            remaining_indices = [
                index for index in unlabeled_indices if index not in all_pseudo_indices
            ]
            if not remaining_indices:
                break

            remaining_dataset = build_tensor_dataset(raw_train, remaining_indices, base_transform)
            remaining_loader = DataLoader(
                remaining_dataset,
                batch_size=data_config["eval_batch_size"],
                shuffle=False,
                num_workers=data_config["num_workers"],
            )
            new_pseudo_dataset, new_indices, new_confidences = generate_pseudo_labels(
                model=pseudo_model,
                data_loader=remaining_loader,
                device=device,
                threshold=current_threshold,
                dataset_indices=remaining_indices,
            )
            if new_pseudo_dataset is None:
                break

            all_pseudo_indices.update(new_indices)
            combined_dataset = ConcatDataset([combined_dataset, new_pseudo_dataset])
            train_loader, val_loader, test_loader = build_loaders(
                dataset=combined_dataset,
                test_dataset=test_dataset,
                seed=config["seed"],
                validation_size=experiment_config["validation_size"],
                batch_size=data_config["batch_size"],
                eval_batch_size=data_config["eval_batch_size"],
                num_workers=data_config["num_workers"],
            )
            iterative_optimizer = torch.optim.Adam(
                pseudo_model.parameters(),
                lr=train_config["learning_rate"],
            )
            train_classifier(
                model=pseudo_model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=iterative_optimizer,
                criterion=criterion,
                device=device,
                epochs=train_config["iterative_epochs"],
            )
            iteration_test = evaluate_classifier(pseudo_model, test_loader, criterion, device)
            iterative_results.append(
                {
                    "iteration": iteration + 1,
                    "threshold": current_threshold,
                    "new_pseudo_labels": len(new_indices),
                    "avg_confidence": sum(new_confidences) / len(new_confidences),
                    "remaining_unlabeled": len(remaining_indices) - len(new_indices),
                    "test_accuracy": iteration_test["accuracy"],
                }
            )
            current_threshold = max(
                0.5,
                current_threshold - pseudo_config["threshold_decay"],
            )

    checkpoint_model = pseudo_model if pseudo_model is not None else baseline_model
    save_checkpoint(
        output_dir / "pseudo_label_model.pt",
        {
            "config": config,
            "state_dict": checkpoint_model.state_dict(),
        },
    )
    save_json(
        output_dir / "metrics.json",
        {
            "device": str(device),
            "labeled_examples": len(labeled_indices),
            "augmented_labeled_examples": len(augmented_labeled_dataset),
            "initial_pseudo_labels": len(pseudo_indices),
            "baseline": {
                "history": baseline_history,
                "test": baseline_test,
            },
            "single_round_pseudo_labeling": {
                "history": single_round_history,
                "test": single_round_test,
            },
            "iterative_pseudo_labeling": iterative_results,
        },
    )

    print(f"Saved pseudo-labeling artifacts to {output_dir}")
    print(f"Baseline test accuracy: {baseline_test['accuracy']:.4f}")
    if single_round_test is not None:
        print(f"Single-round pseudo-labeling accuracy: {single_round_test['accuracy']:.4f}")
    if iterative_results:
        print(f"Final iterative accuracy: {iterative_results[-1]['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
