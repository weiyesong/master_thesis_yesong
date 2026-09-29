from __future__ import annotations

import argparse
import ssl
import warnings

warnings.filterwarnings("ignore", message="Failed to load image Python extension.*")

from src.datasets import build_dataloaders
from src.models import build_model
from src.train import fit
from src.utils import append_results_csv, ensure_output_dirs, get_device, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="EuroSAT RGB calibration baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eurosat_resnet18_rgb.yaml",
        help="Path to a YAML experiment config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if bool(config["data"].get("allow_insecure_ssl", False)):
        ssl._create_default_https_context = ssl._create_unverified_context

    set_seed(int(config["experiment"]["seed"]))
    ensure_output_dirs(config)
    device = get_device(config)

    print(f"Running experiment: {config['experiment']['name']}")
    print(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)

    result = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    best_metrics = result["best_metrics"]
    row = {
        "experiment": config["experiment"]["name"],
        "dataset": config["data"]["name"],
        "model": config["model"]["name"],
        "epoch": best_metrics["epoch"],
        "train_loss": best_metrics["train_loss"],
        "val_accuracy": best_metrics["accuracy"],
        "val_nll": best_metrics["nll"],
        "val_ece": best_metrics["ece"],
        "val_brier": best_metrics["brier"],
        "best_checkpoint": result["best_checkpoint"],
    }
    append_results_csv(config["outputs"]["results_csv"], row)

    print(f"Best checkpoint: {result['best_checkpoint']}")
    print(f"Results appended to: {config['outputs']['results_csv']}")


if __name__ == "__main__":
    main()
