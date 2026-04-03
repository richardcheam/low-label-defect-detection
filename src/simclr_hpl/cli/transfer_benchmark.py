from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from simclr_hpl.config import load_config
from simclr_hpl.data import (
    ContrastiveViewDataset,
    TransformDataset,
    build_augmented_tensor_dataset,
    build_normalize_transform,
    build_simclr_transform,
    build_supervised_augmentation_transforms,
    build_tensor_dataset,
    build_train_val_subsets,
    load_mnist,
    sample_balanced_indices,
)
from simclr_hpl.models import Encoder, EncoderClassifier, ProjectionHead
from simclr_hpl.training import (
    NTXentLoss,
    evaluate_classifier,
    generate_pseudo_labels,
    pretrain_simclr,
    train_classifier,
)
from simclr_hpl.utils import (
    ensure_dir,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark random vs SimCLR initialization for pseudo-labeling."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/transfer_pseudo_label_mnist.yaml"),
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


def pretrain_or_load_encoder(
    config: dict[str, Any],
    raw_train,
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, list[float]] | None, Path]:
    transfer_config = config["transfer_benchmark"]
    checkpoint_path = Path(transfer_config["simclr_checkpoint"])
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        return checkpoint["encoder_state_dict"], None, checkpoint_path

    if not transfer_config.get("train_simclr_if_missing", False):
        msg = (
            f"SimCLR checkpoint not found at {checkpoint_path}. "
            "Run simclr-train first or enable train_simclr_if_missing."
        )
        raise FileNotFoundError(msg)

    data_config = config["data"]
    simclr_config = config["simclr_pretraining"]
    seed_everything(seed)

    contrastive_dataset = ContrastiveViewDataset(
        raw_train,
        build_simclr_transform(data_config["mean"], data_config["std"]),
    )
    contrastive_loader = DataLoader(
        contrastive_dataset,
        batch_size=data_config["pretrain_batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=data_config["num_workers"],
    )
    encoder = Encoder().to(device)
    projection_head = ProjectionHead(output_dim=simclr_config["projection_dim"]).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projection_head.parameters()),
        lr=simclr_config["learning_rate"],
        weight_decay=simclr_config["weight_decay"],
    )
    history = pretrain_simclr(
        encoder=encoder,
        projection_head=projection_head,
        data_loader=contrastive_loader,
        optimizer=optimizer,
        criterion=NTXentLoss(temperature=simclr_config["temperature"]),
        device=device,
        epochs=simclr_config["epochs"],
    )
    trained_checkpoint_path = output_dir / "simclr_encoder_transfer.pt"
    save_checkpoint(
        trained_checkpoint_path,
        {
            "config": config,
            "encoder_state_dict": encoder.state_dict(),
            "projection_head_state_dict": projection_head.state_dict(),
        },
    )
    return copy.deepcopy(encoder.state_dict()), history, trained_checkpoint_path


def build_model(init_mode: str, encoder_state_dict: dict[str, torch.Tensor] | None):
    model = EncoderClassifier()
    if init_mode == "simclr":
        if encoder_state_dict is None:
            msg = "SimCLR initialization requested but no encoder state dict was provided."
            raise ValueError(msg)
        model.encoder.load_state_dict(copy.deepcopy(encoder_state_dict))
    return model


def run_pseudo_labeling_experiment(
    model: nn.Module,
    raw_train,
    raw_test,
    labeled_indices: list[int],
    unlabeled_indices: list[int],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    data_config = config["data"]
    experiment_config = config["experiment"]
    train_config = config["train"]
    pseudo_config = config["pseudo_labeling"]

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

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    baseline_history = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=train_config["epochs"],
    )
    baseline_test = evaluate_classifier(model, test_loader, criterion, device)

    unlabeled_dataset = build_tensor_dataset(raw_train, unlabeled_indices, base_transform)
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=data_config["eval_batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
    )
    pseudo_dataset, pseudo_indices, confidences = generate_pseudo_labels(
        model=model,
        data_loader=unlabeled_loader,
        device=device,
        threshold=pseudo_config["threshold"],
        dataset_indices=unlabeled_indices,
    )

    single_round_history = None
    single_round_test = None
    iterative_results: list[dict[str, float | int]] = []
    combined_dataset = augmented_labeled_dataset

    if pseudo_dataset is not None:
        combined_dataset = ConcatDataset([combined_dataset, pseudo_dataset])
        train_loader, val_loader, test_loader = build_loaders(
            dataset=combined_dataset,
            test_dataset=test_dataset,
            seed=config["seed"],
            validation_size=experiment_config["validation_size"],
            batch_size=data_config["batch_size"],
            eval_batch_size=data_config["eval_batch_size"],
            num_workers=data_config["num_workers"],
        )
        single_round_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_config["learning_rate"],
        )
        single_round_history = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=single_round_optimizer,
            criterion=criterion,
            device=device,
            epochs=train_config["epochs"],
        )
        single_round_test = evaluate_classifier(model, test_loader, criterion, device)

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
                model=model,
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
                model.parameters(),
                lr=train_config["learning_rate"],
            )
            train_classifier(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=iterative_optimizer,
                criterion=criterion,
                device=device,
                epochs=train_config["iterative_epochs"],
            )
            iteration_test = evaluate_classifier(model, test_loader, criterion, device)
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
            current_threshold = max(0.5, current_threshold - pseudo_config["threshold_decay"])

    return {
        "labeled_examples": len(labeled_indices),
        "augmented_labeled_examples": len(augmented_labeled_dataset),
        "initial_pseudo_labels": len(pseudo_indices),
        "initial_pseudo_confidence_mean": (
            sum(confidences) / len(confidences) if confidences else None
        ),
        "baseline": {
            "history": baseline_history,
            "test": baseline_test,
        },
        "single_round_pseudo_labeling": {
            "history": single_round_history,
            "test": single_round_test,
        },
        "iterative_pseudo_labeling": iterative_results,
    }


def summarize_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget_key, budget_runs in results.items():
        for init_mode, metrics in budget_runs.items():
            final_iterative = metrics["iterative_pseudo_labeling"]
            rows.append(
                {
                    "label_budget": int(budget_key),
                    "initialization": init_mode,
                    "baseline_accuracy": metrics["baseline"]["test"]["accuracy"],
                    "single_round_accuracy": (
                        None
                        if metrics["single_round_pseudo_labeling"]["test"] is None
                        else metrics["single_round_pseudo_labeling"]["test"]["accuracy"]
                    ),
                    "final_iterative_accuracy": (
                        None if not final_iterative else final_iterative[-1]["test_accuracy"]
                    ),
                    "initial_pseudo_labels": metrics["initial_pseudo_labels"],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed_everything(config["seed"])
    device = resolve_device(config.get("device", "auto"))
    output_dir = ensure_dir(config["output_dir"])

    raw_train = load_mnist(config["data"]["root"], train=True)
    raw_test = load_mnist(config["data"]["root"], train=False)

    encoder_state_dict = None
    simclr_history = None
    simclr_checkpoint_path = None
    if "simclr" in config["transfer_benchmark"]["initializations"]:
        encoder_state_dict, simclr_history, simclr_checkpoint_path = pretrain_or_load_encoder(
            config=config,
            raw_train=raw_train,
            device=device,
            seed=config["seed"],
            output_dir=output_dir,
        )

    benchmark_results: dict[str, dict[str, Any]] = {}
    for label_budget in config["transfer_benchmark"]["label_budgets"]:
        if label_budget % 10 != 0:
            msg = f"Label budget must be divisible by 10 for balanced sampling: {label_budget}"
            raise ValueError(msg)
        labeled_indices, unlabeled_indices = sample_balanced_indices(
            raw_train,
            per_class=label_budget // 10,
            seed=config["seed"],
        )
        budget_key = str(label_budget)
        benchmark_results[budget_key] = {}

        for init_mode in config["transfer_benchmark"]["initializations"]:
            seed_everything(config["seed"])
            model = build_model(init_mode, encoder_state_dict)
            benchmark_results[budget_key][init_mode] = run_pseudo_labeling_experiment(
                model=model,
                raw_train=raw_train,
                raw_test=raw_test,
                labeled_indices=labeled_indices,
                unlabeled_indices=unlabeled_indices,
                config=config,
                device=device,
            )

    summary_rows = summarize_results(benchmark_results)
    save_json(
        output_dir / "transfer_benchmark_metrics.json",
        {
            "device": str(device),
            "simclr_checkpoint": None if simclr_checkpoint_path is None else str(simclr_checkpoint_path),
            "simclr_pretraining_history": simclr_history,
            "benchmark_results": benchmark_results,
            "summary": summary_rows,
        },
    )

    print(f"Saved transfer benchmark artifacts to {output_dir}")
    for row in summary_rows:
        single_round = row["single_round_accuracy"]
        final_iterative = row["final_iterative_accuracy"]
        print(
            " | ".join(
                [
                    f"labels={row['label_budget']}",
                    f"init={row['initialization']}",
                    f"baseline={row['baseline_accuracy']:.4f}",
                    f"single_round={'n/a' if single_round is None else f'{single_round:.4f}'}",
                    f"iterative={'n/a' if final_iterative is None else f'{final_iterative:.4f}'}",
                ]
            )
        )


if __name__ == "__main__":
    main()
