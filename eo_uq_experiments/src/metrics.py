from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class ClassificationMetrics:
    accuracy: float
    nll: float
    ece: float
    brier: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float(predictions.eq(labels).float().mean().item())


def nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, labels).item())


def brier_score_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = logits.softmax(dim=1)
    targets = F.one_hot(labels, num_classes=probs.shape[1]).float()
    return float(((probs - targets) ** 2).sum(dim=1).mean().item())


def ece_from_logits(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    probs = logits.softmax(dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()

    ece = torch.zeros((), device=logits.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)

    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() == 0:
            continue

        bin_accuracy = accuracies[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        ece += (bin_confidence - bin_accuracy).abs() * prop_in_bin

    return float(ece.item())


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ece_bins: int = 15,
) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=accuracy_from_logits(logits, labels),
        nll=nll_from_logits(logits, labels),
        ece=ece_from_logits(logits, labels, n_bins=ece_bins),
        brier=brier_score_from_logits(logits, labels),
    )
