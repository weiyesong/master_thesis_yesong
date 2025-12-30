import torch
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassCalibrationError

def compute_ece(probs, labels, n_bins=15):
    ece = MulticlassCalibrationError(
        num_classes=probs.shape[1],
        n_bins=n_bins,
        norm="l1",
    )
    return ece(probs, labels)

def plot_reliability_diagram(probs, labels, save_path):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    bins = torch.linspace(0, 1, 11)
    bin_acc = []
    bin_conf = []

    for i in range(len(bins) - 1):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.any():
            bin_acc.append(accuracies[mask].float().mean().item())
            bin_conf.append(confidences[mask].mean().item())

    plt.figure()
    plt.plot(bin_conf, bin_acc, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.savefig(save_path)
    plt.close()
