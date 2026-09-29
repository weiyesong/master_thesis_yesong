from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.evaluate import evaluate
from src.models import trainable_parameters
from src.utils import append_results_csv


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    limit_batches: int | None = None,
) -> float:
    model.train()
    if getattr(model, "freeze_backbone", False) and hasattr(model, "backbone"):
        model.backbone.eval()
    total_loss = 0.0
    total_examples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.numel()
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
):
    training_config = config["training"]
    output_config = config["outputs"]

    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    checkpoint_dir = Path(output_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{config['experiment']['name']}_best.pt"
    epoch_log_path = Path(output_config["log_dir"]) / f"{config['experiment']['name']}_epochs.csv"
    if epoch_log_path.exists():
        epoch_log_path.unlink()

    best_val_nll = float("inf")
    best_metrics = None
    history = []

    for epoch in range(1, int(training_config["epochs"]) + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            limit_batches=training_config.get("limit_train_batches"),
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            ece_bins=int(config["metrics"]["ece_bins"]),
            limit_batches=training_config.get("limit_val_batches"),
        )
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = train_loss
        history.append(val_metrics)
        append_results_csv(
            epoch_log_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_nll": val_metrics["nll"],
                "val_ece": val_metrics["ece"],
                "val_brier": val_metrics["brier"],
            },
        )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_nll={val_metrics['nll']:.4f} "
            f"val_ece={val_metrics['ece']:.4f} "
            f"val_brier={val_metrics['brier']:.4f}"
        )

        if val_metrics["nll"] < best_val_nll:
            best_val_nll = val_metrics["nll"]
            best_metrics = dict(val_metrics)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                },
                checkpoint_path,
            )

    return {
        "best_checkpoint": str(checkpoint_path),
        "best_metrics": best_metrics,
        "history": history,
    }
