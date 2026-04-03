from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from simclr_hpl.config import load_config
from simclr_hpl.data import (
    ContrastiveViewDataset,
    TransformDataset,
    build_normalize_transform,
    build_simclr_transform,
    build_train_val_subsets,
    load_mnist,
)
from simclr_hpl.models import Encoder, LinearProbe, MLPProbe, ProjectionHead
from simclr_hpl.training import (
    NTXentLoss,
    evaluate_classifier,
    freeze_module,
    pretrain_simclr,
    train_classifier,
    trainable_parameters,
)
from simclr_hpl.utils import ensure_dir, resolve_device, save_checkpoint, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SimCLR on MNIST and evaluate probes.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/simclr_mnist.yaml"),
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed_everything(config["seed"])
    device = resolve_device(config.get("device", "auto"))
    output_dir = ensure_dir(config["output_dir"])

    data_config = config["data"]
    train_config = config["train"]
    eval_config = config["evaluation"]
    model_config = config["model"]

    raw_train = load_mnist(data_config["root"], train=True)
    raw_test = load_mnist(data_config["root"], train=False)

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
    projection_head = ProjectionHead(output_dim=model_config["projection_dim"]).to(device)
    criterion = NTXentLoss(temperature=train_config["temperature"])
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projection_head.parameters()),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    simclr_history = pretrain_simclr(
        encoder=encoder,
        projection_head=projection_head,
        data_loader=contrastive_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=train_config["epochs"],
    )

    supervised_transform = build_normalize_transform(data_config["mean"], data_config["std"])
    full_train = TransformDataset(raw_train, supervised_transform)
    test_dataset = TransformDataset(raw_test, supervised_transform)
    train_subset, val_subset = build_train_val_subsets(
        full_train,
        validation_size=eval_config["validation_size"],
        seed=config["seed"],
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=data_config["eval_batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=data_config["eval_batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["eval_batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
    )

    probe_criterion = nn.CrossEntropyLoss()

    linear_probe = LinearProbe(encoder).to(device)
    freeze_module(linear_probe.encoder)
    linear_optimizer = torch.optim.Adam(
        trainable_parameters(linear_probe),
        lr=eval_config["learning_rate"],
    )
    linear_history = train_classifier(
        model=linear_probe,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=linear_optimizer,
        criterion=probe_criterion,
        device=device,
        epochs=eval_config["epochs"],
    )
    linear_test = evaluate_classifier(linear_probe, test_loader, probe_criterion, device)

    mlp_probe = MLPProbe(encoder).to(device)
    freeze_module(mlp_probe.encoder)
    mlp_optimizer = torch.optim.Adam(
        trainable_parameters(mlp_probe),
        lr=eval_config["learning_rate"],
    )
    mlp_history = train_classifier(
        model=mlp_probe,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=mlp_optimizer,
        criterion=probe_criterion,
        device=device,
        epochs=eval_config["epochs"],
    )
    mlp_test = evaluate_classifier(mlp_probe, test_loader, probe_criterion, device)

    save_checkpoint(
        output_dir / "simclr_encoder.pt",
        {
            "config": config,
            "encoder_state_dict": encoder.state_dict(),
            "projection_head_state_dict": projection_head.state_dict(),
        },
    )
    save_checkpoint(
        output_dir / "linear_probe.pt",
        {"state_dict": linear_probe.state_dict()},
    )
    save_checkpoint(
        output_dir / "mlp_probe.pt",
        {"state_dict": mlp_probe.state_dict()},
    )
    save_json(
        output_dir / "metrics.json",
        {
            "device": str(device),
            "simclr": simclr_history,
            "linear_probe": {
                "history": linear_history,
                "test": linear_test,
            },
            "mlp_probe": {
                "history": mlp_history,
                "test": mlp_test,
            },
        },
    )

    print(f"Saved SimCLR artifacts to {output_dir}")
    print(f"Linear probe test accuracy: {linear_test['accuracy']:.4f}")
    print(f"MLP probe test accuracy: {mlp_test['accuracy']:.4f}")


if __name__ == "__main__":
    main()
