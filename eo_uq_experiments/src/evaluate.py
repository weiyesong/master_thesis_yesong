from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.metrics import compute_classification_metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    ece_bins: int,
    limit_batches: int | None = None,
):
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    total_examples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        batch_size = labels.numel()
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = compute_classification_metrics(logits, labels, ece_bins=ece_bins).to_dict()
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics
