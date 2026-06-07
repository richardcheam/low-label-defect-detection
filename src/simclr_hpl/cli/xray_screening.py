from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from simclr_hpl.config import load_config
from simclr_hpl.data import (
    ContrastiveViewDataset,
    ImagePathDataset,
    build_rgb_normalize_transform,
    build_rgb_simclr_transform,
    build_train_val_subsets,
    load_binary_image_records,
    split_records_stratified,
)
from simclr_hpl.models import LinearProbe, MLPProbe, ProjectionHead, build_encoder
from simclr_hpl.training import (
    NTXentLoss,
    evaluate_classifier,
    freeze_module,
    pretrain_simclr,
    train_classifier,
    trainable_parameters,
)
from simclr_hpl.utils import (
    ensure_dir,
    resolve_device,
    save_checkpoint,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimCLR + probe eval on binary X-ray data.")
    parser.add_argument("--config", type=Path, default=Path("configs/xray_screening.yaml"))
    return parser.parse_args()


def _subsample(records, cap, seed):
    if cap is None or cap >= len(records):
        return records
    return random.Random(seed).sample(records, cap)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    device = resolve_device(config.get("device", "auto"))
    output_dir = ensure_dir(config["output_dir"])

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    eval_cfg = config["evaluation"]

    records = load_binary_image_records(data_cfg["root"], data_cfg["class_to_label"])
    train_records, test_records = split_records_stratified(
        records, test_size=0.2, seed=config["seed"]
    )

    image_size = int(data_cfg["image_size"])
    mean = tuple(data_cfg["mean"])
    std = tuple(data_cfg["std"])
    simclr_tf = build_rgb_simclr_transform(image_size, mean, std)
    eval_tf = build_rgb_normalize_transform(image_size, mean, std)

    pretrain_records = _subsample(
        train_records, data_cfg.get("max_pretrain_images"), config["seed"]
    )
    contrastive = ContrastiveViewDataset(
        ImagePathDataset(pretrain_records, transform=None), simclr_tf
    )
    contrastive_loader = DataLoader(
        contrastive,
        batch_size=data_cfg["pretrain_batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=data_cfg["num_workers"],
    )

    encoder = build_encoder(
        model_cfg["encoder"],
        input_channels=model_cfg["input_channels"],
        pretrained=model_cfg.get("pretrained", False),
    ).to(device)
    projection_head = ProjectionHead(
        input_dim=encoder.output_dim, output_dim=model_cfg["projection_dim"]
    ).to(device)
    criterion = NTXentLoss(temperature=train_cfg["temperature"])
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projection_head.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    simclr_history = pretrain_simclr(
        encoder=encoder,
        projection_head=projection_head,
        data_loader=contrastive_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=train_cfg["epochs"],
    )
    save_checkpoint(output_dir / "xray_encoder.pt", {"encoder": encoder.state_dict()})

    eval_train = ImagePathDataset(train_records, transform=eval_tf)
    eval_test = ImagePathDataset(test_records, transform=eval_tf)
    probe_train, probe_val = build_train_val_subsets(
        eval_train, validation_size=eval_cfg["validation_size"], seed=config["seed"]
    )
    train_loader = DataLoader(probe_train, batch_size=data_cfg["eval_batch_size"], shuffle=True)
    val_loader = DataLoader(probe_val, batch_size=data_cfg["eval_batch_size"])
    test_loader = DataLoader(eval_test, batch_size=data_cfg["eval_batch_size"])

    num_classes = int(model_cfg["num_classes"])
    probe_criterion = nn.CrossEntropyLoss()
    results: dict[str, object] = {"simclr_final_loss": simclr_history["loss"][-1]}
    for name, probe_cls in (("linear_probe", LinearProbe), ("mlp_probe", MLPProbe)):
        freeze_module(encoder)
        probe = probe_cls(encoder, num_classes=num_classes).to(device)
        probe_optimizer = torch.optim.Adam(
            trainable_parameters(probe), lr=eval_cfg["learning_rate"]
        )
        train_classifier(
            model=probe,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=probe_optimizer,
            criterion=probe_criterion,
            device=device,
            epochs=eval_cfg["epochs"],
        )
        metrics = evaluate_classifier(probe, test_loader, probe_criterion, device)
        results[name] = metrics

    save_json(output_dir / "metrics.json", results)
    print(f"Saved metrics to {output_dir / 'metrics.json'}: {results}")


if __name__ == "__main__":
    main()
