from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from simclr_hpl.business import compute_review_queue_metrics
from simclr_hpl.config import load_config
from simclr_hpl.data import (
    ContrastiveViewDataset,
    ImagePathDataset,
    build_augmented_tensor_dataset,
    build_rgb_normalize_transform,
    build_rgb_simclr_transform,
    build_rgb_supervised_augmentation_transforms,
    build_tensor_dataset,
    build_train_val_subsets,
    load_mvtec_records,
    sample_by_class_counts,
    split_records_stratified,
)
from simclr_hpl.models import Encoder, EncoderClassifier, ProjectionHead
from simclr_hpl.training import (
    NTXentLoss,
    collect_prediction_outputs,
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
        description="Run a low-label manufacturing inspection workflow on MVTec AD."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mvtec_bottle_inspection.yaml"),
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
    pool_records: list[dict[str, object]],
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, list[float]] | None, Path]:
    simclr_config = config["simclr_pretraining"]
    checkpoint_path = Path(simclr_config["checkpoint_path"])
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        return checkpoint["encoder_state_dict"], None, checkpoint_path

    if not simclr_config.get("enabled", False):
        msg = f"SimCLR checkpoint not found at {checkpoint_path} and pretraining is disabled."
        raise FileNotFoundError(msg)

    data_config = config["data"]
    raw_dataset = ImagePathDataset(pool_records, transform=None)
    contrastive_dataset = ContrastiveViewDataset(
        raw_dataset,
        build_rgb_simclr_transform(
            image_size=data_config["image_size"],
            mean=tuple(data_config["mean"]),
            std=tuple(data_config["std"]),
        ),
    )
    contrastive_loader = DataLoader(
        contrastive_dataset,
        batch_size=data_config["pretrain_batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=data_config["num_workers"],
    )
    encoder = Encoder(input_channels=3).to(device)
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
    trained_checkpoint_path = output_dir / f"{config['data']['category']}_simclr_encoder.pt"
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
    model = EncoderClassifier(num_classes=2, input_channels=3)
    if init_mode == "simclr":
        if encoder_state_dict is None:
            msg = "SimCLR initialization requested but no encoder weights were provided."
            raise ValueError(msg)
        model.encoder.load_state_dict(copy.deepcopy(encoder_state_dict))
    return model


def run_inspection_experiment(
    model: nn.Module,
    train_records: list[dict[str, object]],
    test_records: list[dict[str, object]],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    data_config = config["data"]
    simulation_config = config["simulation"]
    train_config = config["train"]
    pseudo_config = config["pseudo_labeling"]
    business_config = config["business"]

    base_transform = build_rgb_normalize_transform(
        image_size=data_config["image_size"],
        mean=tuple(data_config["mean"]),
        std=tuple(data_config["std"]),
    )
    augmentation_transforms = build_rgb_supervised_augmentation_transforms(
        image_size=data_config["image_size"],
        mean=tuple(data_config["mean"]),
        std=tuple(data_config["std"]),
    )

    raw_pool_dataset = ImagePathDataset(train_records, transform=None)
    raw_pool_labels = raw_pool_dataset.targets
    labeled_indices, unlabeled_indices = sample_by_class_counts(
        raw_pool_labels,
        class_counts={0: simulation_config["labeled_normal"], 1: simulation_config["labeled_defect"]},
        seed=config["seed"],
    )

    labeled_dataset = build_augmented_tensor_dataset(
        raw_pool_dataset,
        labeled_indices,
        base_transform=base_transform,
        augmentation_transforms=augmentation_transforms,
        copies_per_transform=simulation_config["augmentation_copies_per_transform"],
        seed=config["seed"],
    )
    test_dataset = ImagePathDataset(test_records, transform=base_transform)

    train_loader, val_loader, test_loader = build_loaders(
        dataset=labeled_dataset,
        test_dataset=test_dataset,
        seed=config["seed"],
        validation_size=simulation_config["validation_size"],
        batch_size=data_config["batch_size"],
        eval_batch_size=data_config["eval_batch_size"],
        num_workers=data_config["num_workers"],
    )

    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    baseline_optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    baseline_history = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=baseline_optimizer,
        criterion=criterion,
        device=device,
        epochs=train_config["epochs"],
    )
    baseline_test = evaluate_classifier(model, test_loader, criterion, device)

    unlabeled_dataset = build_tensor_dataset(raw_pool_dataset, unlabeled_indices, base_transform)
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
    combined_dataset = labeled_dataset
    iterative_results: list[dict[str, float | int]] = []

    if pseudo_dataset is not None:
        combined_dataset = ConcatDataset([combined_dataset, pseudo_dataset])
        train_loader, val_loader, test_loader = build_loaders(
            dataset=combined_dataset,
            test_dataset=test_dataset,
            seed=config["seed"],
            validation_size=simulation_config["validation_size"],
            batch_size=data_config["batch_size"],
            eval_batch_size=data_config["eval_batch_size"],
            num_workers=data_config["num_workers"],
        )
        single_round_optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])
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

            remaining_dataset = build_tensor_dataset(raw_pool_dataset, remaining_indices, base_transform)
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
                validation_size=simulation_config["validation_size"],
                batch_size=data_config["batch_size"],
                eval_batch_size=data_config["eval_batch_size"],
                num_workers=data_config["num_workers"],
            )
            iterative_optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])
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

    prediction_outputs = collect_prediction_outputs(model, test_loader, device)
    review_metrics = compute_review_queue_metrics(
        predictions=prediction_outputs["predictions"],
        targets=prediction_outputs["targets"],
        confidences=prediction_outputs["confidences"],
        auto_decision_threshold=business_config["auto_decision_threshold"],
        defect_label=1,
    )

    return {
        "labeled_normal": simulation_config["labeled_normal"],
        "labeled_defect": simulation_config["labeled_defect"],
        "augmented_labeled_examples": len(labeled_dataset),
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
        "review_queue_metrics": review_metrics,
    }


def summarize_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for init_mode, metrics in results.items():
        final_iterative = metrics["iterative_pseudo_labeling"]
        review_metrics = metrics["review_queue_metrics"]
        summary.append(
            {
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
                "auto_decision_rate": review_metrics["auto_decision_rate"],
                "review_queue_rate": review_metrics["review_queue_rate"],
                "auto_decision_accuracy": review_metrics["auto_decision_accuracy"],
                "auto_defect_recall": review_metrics["auto_defect_recall"],
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed_everything(config["seed"])
    device = resolve_device(config.get("device", "auto"))
    output_dir = ensure_dir(config["output_dir"])

    all_records = load_mvtec_records(
        root=config["data"]["root"],
        category=config["data"]["category"],
    )
    train_records, test_records = split_records_stratified(
        all_records,
        test_size=config["simulation"]["test_size"],
        seed=config["seed"],
    )

    encoder_state_dict = None
    simclr_history = None
    simclr_checkpoint_path = None
    if "simclr" in config["experiment"]["initializations"]:
        encoder_state_dict, simclr_history, simclr_checkpoint_path = pretrain_or_load_encoder(
            config=config,
            pool_records=train_records,
            device=device,
            output_dir=output_dir,
        )

    results: dict[str, Any] = {}
    for init_mode in config["experiment"]["initializations"]:
        seed_everything(config["seed"])
        model = build_model(init_mode, encoder_state_dict)
        results[init_mode] = run_inspection_experiment(
            model=model,
            train_records=train_records,
            test_records=test_records,
            config=config,
            device=device,
        )
        save_checkpoint(
            output_dir / f"{config['data']['category']}_{init_mode}_inspection_model.pt",
            {"config": config, "state_dict": model.state_dict()},
        )

    summary = summarize_results(results)
    save_json(
        output_dir / "metrics.json",
        {
            "use_case": "manufacturing_defect_detection",
            "dataset": "mvtec_ad",
            "category": config["data"]["category"],
            "device": str(device),
            "simclr_checkpoint": None if simclr_checkpoint_path is None else str(simclr_checkpoint_path),
            "simclr_pretraining_history": simclr_history,
            "results": results,
            "summary": summary,
        },
    )

    print(f"Saved MVTec inspection artifacts to {output_dir}")
    for row in summary:
        print(
            " | ".join(
                [
                    f"init={row['initialization']}",
                    f"baseline={row['baseline_accuracy']:.4f}",
                    "iterative="
                    + (
                        "n/a"
                        if row["final_iterative_accuracy"] is None
                        else f"{row['final_iterative_accuracy']:.4f}"
                    ),
                    f"auto_decision_rate={row['auto_decision_rate']:.4f}",
                    f"review_queue_rate={row['review_queue_rate']:.4f}",
                    f"auto_defect_recall={row['auto_defect_recall']:.4f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
