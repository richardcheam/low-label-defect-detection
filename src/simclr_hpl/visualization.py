from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


def load_metrics(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_metrics_type(payload: dict[str, Any]) -> str:
    if "benchmark_results" in payload and "summary" in payload:
        return "transfer"
    if "simclr" in payload and "linear_probe" in payload and "mlp_probe" in payload:
        return "simclr"
    if "baseline" in payload and "single_round_pseudo_labeling" in payload:
        return "pseudo_label"
    msg = "Unsupported metrics file format."
    raise ValueError(msg)


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_simclr_metrics(payload: dict[str, Any], output_dir: Path) -> list[Path]:
    created: list[Path] = []

    simclr_loss = payload["simclr"]["loss"]
    fig, ax = plt.subplots()
    epochs = list(range(1, len(simclr_loss) + 1))
    ax.plot(epochs, simclr_loss, marker="o", linewidth=2, color="#1f77b4")
    ax.set_title("SimCLR Pretraining Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent Loss")
    loss_path = output_dir / "simclr_pretraining_loss.png"
    save_figure(fig, loss_path)
    created.append(loss_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    linear_history = payload["linear_probe"]["history"]
    mlp_history = payload["mlp_probe"]["history"]
    for history, label, color in [
        (linear_history, "Linear Probe", "#1f77b4"),
        (mlp_history, "Frozen MLP", "#ff7f0e"),
    ]:
        epochs = list(range(1, len(history["train_accuracy"]) + 1))
        axes[0].plot(epochs, history["train_accuracy"], label=f"{label} Train", color=color)
        axes[0].plot(
            epochs,
            history["val_accuracy"],
            label=f"{label} Val",
            color=color,
            linestyle="--",
        )
        axes[1].plot(epochs, history["train_loss"], label=f"{label} Train", color=color)
        axes[1].plot(
            epochs,
            history["val_loss"],
            label=f"{label} Val",
            color=color,
            linestyle="--",
        )
    axes[0].set_title("Probe Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[1].set_title("Probe Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].legend()
    probe_curves_path = output_dir / "simclr_probe_curves.png"
    save_figure(fig, probe_curves_path)
    created.append(probe_curves_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Linear Probe", "Frozen MLP"]
    accuracies = [
        payload["linear_probe"]["test"]["accuracy"],
        payload["mlp_probe"]["test"]["accuracy"],
    ]
    sns.barplot(x=labels, y=accuracies, ax=ax, palette=["#1f77b4", "#ff7f0e"], hue=labels)
    ax.set_title("SimCLR Downstream Test Accuracy")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    bar_path = output_dir / "simclr_test_accuracy.png"
    save_figure(fig, bar_path)
    created.append(bar_path)

    return created


def plot_pseudo_label_metrics(payload: dict[str, Any], output_dir: Path) -> list[Path]:
    created: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    baseline_history = payload["baseline"]["history"]
    single_round_history = payload["single_round_pseudo_labeling"]["history"]

    for history, label, color in [
        (baseline_history, "Baseline", "#2a9d8f"),
        (single_round_history, "Single-Round", "#e76f51"),
    ]:
        if history is None:
            continue
        epochs = list(range(1, len(history["train_accuracy"]) + 1))
        axes[0].plot(epochs, history["train_accuracy"], label=f"{label} Train", color=color)
        axes[0].plot(
            epochs,
            history["val_accuracy"],
            label=f"{label} Val",
            color=color,
            linestyle="--",
        )
        axes[1].plot(epochs, history["train_loss"], label=f"{label} Train", color=color)
        axes[1].plot(
            epochs,
            history["val_loss"],
            label=f"{label} Val",
            color=color,
            linestyle="--",
        )
    axes[0].set_title("Pseudo-Labeling Accuracy Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[1].set_title("Pseudo-Labeling Loss Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].legend()
    curves_path = output_dir / "pseudo_label_training_curves.png"
    save_figure(fig, curves_path)
    created.append(curves_path)

    iterative_results = payload["iterative_pseudo_labeling"]
    if iterative_results:
        iterations = [entry["iteration"] for entry in iterative_results]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(
            iterations,
            [entry["new_pseudo_labels"] for entry in iterative_results],
            marker="o",
            linewidth=2,
            color="#264653",
        )
        axes[0].set_title("New Pseudo-Labels Per Iteration")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Count")

        axes[1].plot(
            iterations,
            [entry["test_accuracy"] for entry in iterative_results],
            marker="o",
            linewidth=2,
            color="#f4a261",
        )
        axes[1].set_title("Iterative Test Accuracy")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Accuracy")

        iterative_path = output_dir / "pseudo_label_iteration_dynamics.png"
        save_figure(fig, iterative_path)
        created.append(iterative_path)

    return created


def plot_transfer_metrics(payload: dict[str, Any], output_dir: Path) -> list[Path]:
    created: list[Path] = []
    summary = payload["summary"]

    init_modes = sorted({row["initialization"] for row in summary})
    budgets = sorted({int(row["label_budget"]) for row in summary})
    stage_specs = [
        ("baseline_accuracy", "Baseline Accuracy"),
        ("single_round_accuracy", "Single-Round Pseudo-Labeling Accuracy"),
        ("final_iterative_accuracy", "Final Iterative Accuracy"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = {"random": "#6c757d", "simclr": "#1d3557"}
    for ax, (key, title) in zip(axes, stage_specs, strict=False):
        for init_mode in init_modes:
            rows = sorted(
                [row for row in summary if row["initialization"] == init_mode],
                key=lambda row: int(row["label_budget"]),
            )
            x_values = [int(row["label_budget"]) for row in rows]
            y_values = [row[key] for row in rows]
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2,
                label=init_mode,
                color=colors.get(init_mode, None),
            )
        ax.set_title(title)
        ax.set_xlabel("Label Budget")
        ax.set_xticks(budgets)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(title="Initialization")
    accuracy_path = output_dir / "transfer_accuracy_by_budget.png"
    save_figure(fig, accuracy_path)
    created.append(accuracy_path)

    fig, ax = plt.subplots(figsize=(9, 5))
    for init_mode in init_modes:
        rows = sorted(
            [row for row in summary if row["initialization"] == init_mode],
            key=lambda row: int(row["label_budget"]),
        )
        ax.plot(
            [int(row["label_budget"]) for row in rows],
            [row["initial_pseudo_labels"] for row in rows],
            marker="o",
            linewidth=2,
            label=init_mode,
            color=colors.get(init_mode, None),
        )
    ax.set_title("Initial Pseudo-Label Yield By Label Budget")
    ax.set_xlabel("Label Budget")
    ax.set_ylabel("Pseudo-Labels Above Threshold")
    ax.set_xticks(budgets)
    ax.legend(title="Initialization")
    pseudo_path = output_dir / "transfer_initial_pseudo_labels.png"
    save_figure(fig, pseudo_path)
    created.append(pseudo_path)

    return created


def create_plots(metrics_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    payload = load_metrics(metrics_path)
    metrics_type = infer_metrics_type(payload)
    metrics_path = Path(metrics_path)
    resolved_output_dir = (
        metrics_path.parent / "plots" if output_dir is None else Path(output_dir)
    )
    setup_plot_style()

    if metrics_type == "simclr":
        return plot_simclr_metrics(payload, resolved_output_dir)
    if metrics_type == "pseudo_label":
        return plot_pseudo_label_metrics(payload, resolved_output_dir)
    if metrics_type == "transfer":
        return plot_transfer_metrics(payload, resolved_output_dir)
    msg = f"Unexpected metrics type: {metrics_type}"
    raise ValueError(msg)
