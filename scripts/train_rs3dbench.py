from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


@dataclass(frozen=True)
class DepthPair:
    image: Path
    depth: Path
    region: str


def _tile_name(line: str) -> Optional[str]:
    fields = line.strip().replace("\\", "/").split()
    if not fields:
        return None
    for field in fields:
        name = Path(field).name
        if name.lower().endswith((".png", ".tif", ".tiff")):
            return Path(name).stem
    return Path(fields[0]).stem


def _read_split(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        name
        for line in path.read_text(encoding="utf-8").splitlines()
        if (name := _tile_name(line)) is not None
    }


def discover_rs3dbench_pairs(root: Path) -> Tuple[List[DepthPair], List[DepthPair]]:
    if not root.exists():
        raise FileNotFoundError(
            f"RS3DBench root does not exist: {root}\n"
            "Extract the official dataset so each region contains "
            "'png-stretched-unique/' and 'DEM-unique/' directories."
        )

    explicit_train: List[DepthPair] = []
    explicit_test: List[DepthPair] = []
    unsplit: List[DepthPair] = []

    image_dirs = sorted(path for path in root.rglob("png-stretched-unique") if path.is_dir())
    if not image_dirs:
        raise FileNotFoundError(
            f"No 'png-stretched-unique' directory found under {root}. "
            "The ZIP archives must be extracted before training."
        )

    for image_dir in image_dirs:
        region_dir = image_dir.parent
        depth_dir = region_dir / "DEM-unique"
        if not depth_dir.is_dir():
            raise FileNotFoundError(f"Missing paired DEM directory: {depth_dir}")

        depths = {
            path.stem: path
            for path in depth_dir.iterdir()
            if path.suffix.lower() in {".tif", ".tiff"}
        }
        pairs = [
            DepthPair(image=image, depth=depths[image.stem], region=str(region_dir.relative_to(root)))
            for image in sorted(image_dir.glob("*.png"))
            if image.stem in depths
        ]
        if not pairs:
            continue

        train_names = _read_split(region_dir / "train.txt")
        test_names = _read_split(region_dir / "test.txt")
        if train_names or test_names:
            explicit_train.extend(pair for pair in pairs if pair.image.stem in train_names)
            explicit_test.extend(pair for pair in pairs if pair.image.stem in test_names)
            listed = train_names | test_names
            unsplit.extend(pair for pair in pairs if pair.image.stem not in listed)
        else:
            unsplit.extend(pairs)

    all_pairs = explicit_train + explicit_test + unsplit
    if not all_pairs:
        raise RuntimeError(f"No aligned PNG/TIF pairs were found under {root}")
    return explicit_train + unsplit, explicit_test


class RS3DBenchDataset(Dataset):
    def __init__(self, pairs: Sequence[DepthPair], image_size: int) -> None:
        self.pairs = list(pairs)
        self.image_size = image_size
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        pair = self.pairs[index]
        with Image.open(pair.image) as source:
            image = source.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        with rasterio.open(pair.depth) as source:
            depth_array = source.read(1).astype(np.float32)
            nodata = source.nodata

        depth = torch.from_numpy(depth_array).unsqueeze(0)
        valid = torch.isfinite(depth)
        if nodata is not None and math.isfinite(float(nodata)):
            valid &= depth.ne(float(nodata))
        depth = torch.nan_to_num(depth)
        depth = F.interpolate(
            depth.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        valid = F.interpolate(
            valid.float().unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0).bool()
        return {
            "image": self.image_transform(image),
            "depth": depth,
            "valid": valid,
            "name": pair.image.stem,
            "region": pair.region,
        }


def make_rs3dbench_dataloaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    data_cfg = config["data"]
    root = Path(data_cfg.get("root", "./data/RS3DBench"))
    train_candidates, explicit_test = discover_rs3dbench_pairs(root)
    seed = int(config.get("seed", 42))
    generator = torch.Generator().manual_seed(seed)

    if explicit_test:
        train_pairs = train_candidates
        val_pairs = explicit_test
    else:
        val_fraction = float(data_cfg.get("val_fraction", 0.1))
        order = torch.randperm(len(train_candidates), generator=generator).tolist()
        val_size = max(1, int(len(order) * val_fraction))
        val_indices = set(order[:val_size])
        train_pairs = [pair for idx, pair in enumerate(train_candidates) if idx not in val_indices]
        val_pairs = [pair for idx, pair in enumerate(train_candidates) if idx in val_indices]

    max_train = data_cfg.get("max_train_samples")
    max_val = data_cfg.get("max_val_samples")
    if max_train is not None:
        train_pairs = train_pairs[: int(max_train)]
    if max_val is not None:
        val_pairs = val_pairs[: int(max_val)]
    if not train_pairs or not val_pairs:
        raise RuntimeError(
            f"RS3DBench split is empty: train={len(train_pairs)}, val={len(val_pairs)}"
        )

    loader_args = {
        "batch_size": int(config["training"].get("batch_size", 8)),
        "num_workers": int(data_cfg.get("num_workers", 4)),
        "pin_memory": torch.cuda.is_available(),
    }
    print(f"RS3DBench pairs: train={len(train_pairs)}, val={len(val_pairs)}")
    return {
        "train": DataLoader(
            RS3DBenchDataset(train_pairs, int(data_cfg.get("image_size", 224))),
            shuffle=True,
            **loader_args,
        ),
        "val": DataLoader(
            RS3DBenchDataset(val_pairs, int(data_cfg.get("image_size", 224))),
            shuffle=False,
            **loader_args,
        ),
    }


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ResNetDepthModel(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = models.resnet18(weights=weights)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 32)
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        input_size = image.shape[-2:]
        stem = self.stem(image)
        x1 = self.layer1(self.pool(stem))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, stem)
        return F.interpolate(self.head(x), size=input_size, mode="bilinear", align_corners=False)


def _normalize_depth(depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    normalized = torch.zeros_like(depth)
    for idx in range(depth.shape[0]):
        values = depth[idx][valid[idx]]
        if values.numel() == 0:
            continue
        median = values.median()
        scale = (values - median).abs().mean().clamp_min(1e-6)
        normalized[idx] = (depth[idx] - median) / scale
    return normalized


def relative_depth_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    pred_normalized = _normalize_depth(prediction, valid)
    target_normalized = _normalize_depth(target, valid)
    return F.smooth_l1_loss(pred_normalized[valid], target_normalized[valid])


def _align_prediction(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    aligned = torch.zeros_like(prediction)
    for idx in range(prediction.shape[0]):
        mask = valid[idx]
        x = prediction[idx][mask].float()
        y = target[idx][mask].float()
        if x.numel() < 2:
            continue
        x_mean, y_mean = x.mean(), y.mean()
        scale = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean).square().sum() + 1e-6)
        shift = y_mean - scale * x_mean
        aligned[idx] = prediction[idx] * scale + shift
    return aligned


@torch.no_grad()
def evaluate_depth(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit_batches: Optional[int],
) -> Dict[str, float]:
    model.eval()
    abs_error = 0.0
    squared_error = 0.0
    abs_rel = 0.0
    count = 0
    losses = 0.0
    batches = 0
    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        image = batch["image"].to(device)
        target = batch["depth"].to(device)
        valid = batch["valid"].to(device)
        prediction = model(image)
        losses += float(relative_depth_loss(prediction, target, valid).item())
        aligned = _align_prediction(prediction, target, valid)
        error = aligned[valid] - target[valid]
        abs_error += float(error.abs().sum().item())
        squared_error += float(error.square().sum().item())
        abs_rel += float((error.abs() / target[valid].abs().clamp_min(1.0)).sum().item())
        count += int(valid.sum().item())
        batches += 1
    return {
        "loss": losses / max(batches, 1),
        "mae": abs_error / max(count, 1),
        "rmse": math.sqrt(squared_error / max(count, 1)),
        "abs_rel": abs_rel / max(count, 1),
    }


def _append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_rs3dbench_experiment(config: Dict[str, Any], experiment: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configured_device = config.get("device", "auto")
    device = torch.device(
        "cuda" if configured_device == "auto" and torch.cuda.is_available() else
        "cpu" if configured_device == "auto" else configured_device
    )

    loaders = make_rs3dbench_dataloaders(config)
    model_cfg = experiment.get("model", {})
    if model_cfg.get("name", "resnet18_depth") != "resnet18_depth":
        raise ValueError("RS3DBench currently supports model.name: resnet18_depth")
    model = ResNetDepthModel(pretrained=bool(model_cfg.get("pretrained", True))).to(device)
    training_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )

    output_dir = Path(config.get("output_dir", "./results/rs3dbench")) / experiment["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history: List[Dict[str, Any]] = []
    best_mae = math.inf
    start_epoch = 1

    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint.get("history", [])
        best_mae = float(checkpoint.get("best_mae", best_mae))
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[{experiment['name']}] resumed at epoch {start_epoch}")

    epochs = int(training_cfg.get("epochs", 20))
    limit_train = training_cfg.get("limit_train_batches")
    limit_val = training_cfg.get("limit_val_batches")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        started = time.time()
        loss_sum = 0.0
        batch_count = 0
        for batch_idx, batch in enumerate(loaders["train"]):
            if limit_train is not None and batch_idx >= int(limit_train):
                break
            image = batch["image"].to(device)
            target = batch["depth"].to(device)
            valid = batch["valid"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image)
            loss = relative_depth_loss(prediction, target, valid)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            batch_count += 1

        validation = evaluate_depth(model, loaders["val"], device, limit_val)
        epoch_row = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 3),
            "train_loss": loss_sum / max(batch_count, 1),
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(epoch_row)
        print(
            f"[{experiment['name']}] epoch {epoch:03d}/{epochs:03d} "
            f"train_loss={epoch_row['train_loss']:.4f} "
            f"val_mae={validation['mae']:.4f} "
            f"val_rmse={validation['rmse']:.4f} "
            f"val_abs_rel={validation['abs_rel']:.4f}"
        )
        if validation["mae"] < best_mae:
            best_mae = validation["mae"]
            torch.save(
                {"model": model.state_dict(), "config": config, "experiment": experiment, "epoch": epoch},
                best_path,
            )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "experiment": experiment,
                "epoch": epoch,
                "best_mae": best_mae,
                "history": history,
            },
            last_path,
        )
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    result = {
        "name": experiment["name"],
        "task": "depth_estimation",
        "dataset": "rs3dbench",
        "model": "resnet18_depth",
        "checkpoint_path": str(best_path),
        "best_val_mae": best_mae,
        "history": history,
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    _append_csv(
        Path(config.get("results_csv", "./results/rs3dbench/results.csv")),
        {
            "experiment": experiment["name"],
            "dataset": "rs3dbench",
            "model": "resnet18_depth",
            "best_val_mae": best_mae,
            "best_checkpoint": str(best_path),
        },
    )
    return result
