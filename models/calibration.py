from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
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
    return float(compute_ece(probs, labels, n_bins=n_bins).item())


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=accuracy_from_logits(logits, labels),
        nll=nll_from_logits(logits, labels),
        ece=ece_from_logits(logits, labels, n_bins=n_bins),
        brier=brier_score_from_logits(logits, labels),
    )


def compute_ece(probs, labels, n_bins=15):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()

    ece = torch.zeros((), device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

    for index, (lower, upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
        if index == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() == 0:
            continue

        bin_accuracy = accuracies[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        ece += (bin_confidence - bin_accuracy).abs() * prop_in_bin

    return ece


def calibration_bins(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels).float()

    bin_edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    bin_acc = torch.zeros(n_bins, device=probs.device)
    bin_conf = torch.zeros(n_bins, device=probs.device)
    bin_counts = torch.zeros(n_bins, device=probs.device)

    for index, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if index == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)
        if not in_bin.any():
            continue
        bin_acc[index] = correct[in_bin].mean()
        bin_conf[index] = confidences[in_bin].mean()
        bin_counts[index] = in_bin.sum()

    return bin_edges, bin_acc, bin_conf, bin_counts, confidences


def plot_reliability_diagram(probs, labels, save_path, n_bins: int = 15, show_counts: bool = True):
    probs = probs.detach().cpu()
    labels = labels.detach().cpu()
    bin_edges, bin_acc, bin_conf, bin_counts, _ = calibration_bins(probs, labels, n_bins=n_bins)
    ece = compute_ece(probs, labels, n_bins=n_bins).item()

    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = 1.0 / n_bins
    counts = bin_counts.numpy()
    proportions = counts / max(float(counts.sum()), 1.0)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    if show_counts:
        fig, (ax, count_ax) = plt.subplots(
            2,
            1,
            figsize=(6.2, 5.2),
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
            sharex=True,
            constrained_layout=True,
        )
    else:
        fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
        count_ax = None

    ax.bar(
        centers.numpy(),
        bin_acc.numpy(),
        width=width * 0.92,
        align="center",
        color="#9ecae1",
        edgecolor="#2b6c8a",
        linewidth=0.8,
        label="Accuracy",
        zorder=2,
    )

    gap_bottom = torch.minimum(bin_acc, bin_conf)
    gap_height = torch.abs(bin_conf - bin_acc)
    nonempty = bin_counts > 0
    ax.bar(
        centers[nonempty].numpy(),
        gap_height[nonempty].numpy(),
        bottom=gap_bottom[nonempty].numpy(),
        width=width * 0.92,
        align="center",
        color="#fddbc7",
        edgecolor="#b2182b",
        linewidth=0.8,
        hatch="///",
        alpha=0.65,
        label="Calibration gap",
        zorder=3,
    )
    ax.plot([0, 1], [0, 1], color="0.2", linestyle="--", linewidth=1.2, label="Perfect calibration", zorder=4)
    ax.scatter(
        bin_conf[nonempty].numpy(),
        bin_acc[nonempty].numpy(),
        color="#08306b",
        s=18,
        zorder=5,
        label="Bin mean",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy / confidence")
    ax.set_title(f"Reliability Diagram (ECE = {ece:.4f})", pad=8)
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    if count_ax is not None:
        count_ax.bar(
            centers.numpy(),
            proportions,
            width=width * 0.92,
            align="center",
            color="0.72",
            edgecolor="0.35",
            linewidth=0.6,
        )
        count_ax.set_ylabel("Prop.")
        count_ax.set_xlabel("Confidence")
        count_ax.set_ylim(0, max(float(proportions.max()) * 1.25, 0.05))
        count_ax.grid(axis="y", color="0.9", linewidth=0.7)
    else:
        ax.set_xlabel("Confidence")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


class TemperatureScaling(nn.Module):
    """Simple temperature scaling post-hoc calibration.

    Usage:
        ts = TemperatureScaling()
        ts.fit(logits, labels)
        probs = ts.predict_proba(logits)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        scaled = self.forward(logits)
        return F.softmax(scaled, dim=1)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200):
        """Fit temperature on validation logits/labels by minimizing NLL."""
        device = logits.device
        self.to(device)

        logits = logits.clone().detach().to(device)
        labels = labels.clone().detach().to(device)

        nll_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())


TemperatureScaler = TemperatureScaling
