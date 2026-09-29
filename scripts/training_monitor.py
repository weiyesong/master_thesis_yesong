from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCH_FIELDS = [
    "epoch",
    "seconds",
    "train_loss",
    "val_nll",
    "train_accuracy",
    "val_accuracy",
    "val_ece",
    "val_brier",
    "val_mean_confidence",
    "val_predictive_entropy",
    "learning_rate",
    "backbone_learning_rate",
    "head_learning_rate",
    "warmup_factor",
    "gradient_norm",
    "backbone_gradient_norm",
    "head_gradient_norm",
    "train_val_accuracy_gap",
]

BATCH_FIELDS = [
    "epoch",
    "batch",
    "global_step",
    "loss",
    "accuracy",
    "learning_rate",
    "backbone_learning_rate",
    "head_learning_rate",
    "gradient_norm",
    "backbone_gradient_norm",
    "head_gradient_norm",
]


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def flatten_epoch_history(history: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    rows = []
    for item in history:
        train = item.get("train", {})
        val = item.get("val", {})
        rows.append(
            {
                "epoch": item["epoch"],
                "seconds": item.get("seconds", math.nan),
                "train_loss": train.get("loss", math.nan),
                "val_nll": val.get("nll", math.nan),
                "train_accuracy": train.get("accuracy", math.nan),
                "val_accuracy": val.get("accuracy", math.nan),
                "val_ece": val.get("ece", math.nan),
                "val_brier": val.get("brier", math.nan),
                "val_mean_confidence": val.get("mean_confidence", math.nan),
                "val_predictive_entropy": val.get("predictive_entropy", math.nan),
                "learning_rate": item.get("learning_rate", math.nan),
                "backbone_learning_rate": item.get("backbone_learning_rate", math.nan),
                "head_learning_rate": item.get("head_learning_rate", math.nan),
                "warmup_factor": item.get("warmup_factor", 1.0),
                "gradient_norm": train.get("gradient_norm", math.nan),
                "backbone_gradient_norm": train.get("backbone_gradient_norm", math.nan),
                "head_gradient_norm": train.get("head_gradient_norm", math.nan),
                "train_val_accuracy_gap": item.get("train_val_accuracy_gap", math.nan),
            }
        )
    return rows


def _plot_lines(ax, x, series, title: str, ylabel: str, log_scale: bool = False) -> None:
    for label, values, color in series:
        ax.plot(x, values, marker="o", markersize=3, linewidth=1.7, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if log_scale and all(value > 0 for _, values, _ in series for value in values if math.isfinite(value)):
        ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)


def save_training_dashboard(
    history: List[Dict[str, Any]],
    batch_history: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Persist machine-readable logs and refresh the training dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_rows = flatten_epoch_history(history)
    _write_csv(output_dir / "training_metrics.csv", epoch_rows, EPOCH_FIELDS)
    _write_csv(output_dir / "batch_metrics.csv", batch_history, BATCH_FIELDS)

    if not epoch_rows:
        return output_dir / "training_dashboard.png"

    epochs = [row["epoch"] for row in epoch_rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    fig.suptitle("Training and uncertainty diagnostics", fontsize=15)

    _plot_lines(
        axes[0, 0],
        epochs,
        [
            ("Train CE", [row["train_loss"] for row in epoch_rows], "#1f77b4"),
            ("Validation NLL", [row["val_nll"] for row in epoch_rows], "#d62728"),
        ],
        "Loss / negative log-likelihood",
        "Loss",
    )
    _plot_lines(
        axes[0, 1],
        epochs,
        [
            ("Train", [row["train_accuracy"] for row in epoch_rows], "#1f77b4"),
            ("Validation", [row["val_accuracy"] for row in epoch_rows], "#2ca02c"),
        ],
        "Classification accuracy",
        "Accuracy",
    )
    axes[0, 1].set_ylim(0, 1.02)
    _plot_lines(
        axes[0, 2],
        epochs,
        [
            ("ECE", [row["val_ece"] for row in epoch_rows], "#ff7f0e"),
            ("Brier", [row["val_brier"] for row in epoch_rows], "#9467bd"),
        ],
        "Calibration quality (lower is better)",
        "Score",
    )
    _plot_lines(
        axes[1, 0],
        epochs,
        [
            ("Mean confidence", [row["val_mean_confidence"] for row in epoch_rows], "#8c564b"),
            ("Predictive entropy", [row["val_predictive_entropy"] for row in epoch_rows], "#17becf"),
        ],
        "Predictive uncertainty",
        "Value",
    )
    _plot_lines(
        axes[1, 1],
        epochs,
        [
            ("Backbone LR", [row["backbone_learning_rate"] for row in epoch_rows], "#7f7f7f"),
            ("Head LR", [row["head_learning_rate"] for row in epoch_rows], "#bcbd22"),
            ("Backbone grad norm", [row["backbone_gradient_norm"] for row in epoch_rows], "#e377c2"),
            ("Head grad norm", [row["head_gradient_norm"] for row in epoch_rows], "#ff9896"),
        ],
        "Optimization dynamics",
        "Value",
        log_scale=True,
    )

    batch_ax = axes[1, 2]
    if batch_history:
        steps = [row["global_step"] for row in batch_history]
        losses = [row["loss"] for row in batch_history]
        window = max(1, min(25, len(losses) // 10))
        smoothed = [
            sum(losses[max(0, index - window + 1) : index + 1])
            / len(losses[max(0, index - window + 1) : index + 1])
            for index in range(len(losses))
        ]
        batch_ax.plot(steps, losses, color="#9ecae1", alpha=0.35, linewidth=0.8, label="Batch loss")
        batch_ax.plot(steps, smoothed, color="#08519c", linewidth=1.6, label=f"Moving avg ({window})")
    batch_ax.set_title("Within-epoch loss dynamics")
    batch_ax.set_xlabel("Optimizer step")
    batch_ax.set_ylabel("Cross-entropy")
    batch_ax.grid(alpha=0.25)
    batch_ax.legend(frameon=False, fontsize=8)

    dashboard_path = output_dir / "training_dashboard.png"
    fig.savefig(dashboard_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return dashboard_path
