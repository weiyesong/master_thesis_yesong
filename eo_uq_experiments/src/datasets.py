from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def build_dataloaders(config: dict):
    """Build EuroSAT RGB train and validation dataloaders.

    torchvision.datasets.EuroSAT exposes the RGB dataset without official splits,
    so we create a deterministic train/validation split from the full dataset.
    """
    data_config = config["data"]
    training_config = config["training"]

    transform = transforms.Compose(
        [
            transforms.Resize((int(data_config["image_size"]), int(data_config["image_size"]))),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    dataset = datasets.EuroSAT(
        root=data_config["root"],
        transform=transform,
        download=bool(data_config.get("download", True)),
    )

    val_fraction = float(data_config.get("val_fraction", 0.2))
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(int(config["experiment"].get("seed", 42)))
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    common_loader_args = {
        "batch_size": int(training_config["batch_size"]),
        "num_workers": int(data_config.get("num_workers", 4)),
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_args)

    return train_loader, val_loader
