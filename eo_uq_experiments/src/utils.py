from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict

import torch
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_dir"] = str(path.resolve().parent)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(config: dict) -> torch.device:
    requested_device = config["experiment"].get("device", "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested_device)


def ensure_output_dirs(config: dict) -> None:
    for key in ["root", "checkpoint_dir", "log_dir"]:
        Path(config["outputs"][key]).mkdir(parents=True, exist_ok=True)


def append_results_csv(path: str | Path, row: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
