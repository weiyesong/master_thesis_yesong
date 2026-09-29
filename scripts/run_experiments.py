from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore", message="Failed to load image Python extension.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rasterio
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

try:
    from torchgeo.datasets import EuroSAT
except ImportError:
    EuroSAT = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.calibration import (
    TemperatureScaler,
    compute_classification_metrics,
    compute_ece,
    plot_reliability_diagram,
)
from scripts.training_monitor import save_training_dashboard
from scripts.train_rs3dbench import run_rs3dbench_experiment
from scripts.prediction_export import (
    checkpoint_sha256,
    classification_embedding_extractor,
    collect_deterministic_predictions,
    export_predictions,
    save_evaluation_artifacts,
)
from scripts.mc_dropout import (
    activate_downstream_mc_dropout,
    designate_mc_dropout,
    designated_mc_dropout_modules,
)
from scripts.experiment_manager import (
    collect_code_snapshot,
    collect_environment_metadata,
    make_invocation_summary_path,
    make_run_id,
    resolve_config,
    seed_dataloader_worker,
    set_global_seed,
    write_code_snapshot_manifest,
    write_json,
    write_resolved_config,
)
from scripts.geobench_datasets import GeoBenchTreeSatAITemporalDataset
from scripts.segmentation_pipeline import (
    audit_segmentation_model,
    build_segmentation_model,
    collect_segmentation_predictions,
    evaluate_segmentation,
    export_segmentation_predictions,
    make_segmentation_dataloaders,
    train_segmentation_epoch,
)


SENTINEL2_MEAN = torch.tensor(
    [
        1370.19,
        1184.39,
        1120.77,
        1136.89,
        1263.73,
        1645.40,
        1846.87,
        1762.59,
        1972.62,
        2197.07,
        2383.72,
        2093.36,
        1517.16,
    ]
)
SENTINEL2_STD = torch.tensor(
    [
        633.15,
        650.20,
        712.12,
        965.23,
        948.99,
        1108.06,
        1258.36,
        1233.18,
        1364.38,
        1497.52,
        1602.03,
        1505.44,
        1084.63,
    ]
)

# EuroSAT RGB images are rendered from Sentinel-2 visible bands in RGB order:
# B04 = red   = 0.665 micrometers
# B03 = green = 0.560 micrometers
# B02 = blue  = 0.490 micrometers
SENTINEL2_RGB_WAVELENGTHS = [0.665, 0.560, 0.490]

SO2SAT_S2_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
SO2SAT_CLASSES = (
    "Compact high-rise",
    "Compact middle-rise",
    "Compact low-rise",
    "Open high-rise",
    "Open middle-rise",
    "Open low-rise",
    "Lightweight low-rise",
    "Large low-rise",
    "Sparsely built",
    "Heavy industry",
    "Dense Trees",
    "Scattered trees",
    "Bush, scrub",
    "Low plants",
    "Bare rock or paved",
    "Bare soil or sand",
    "Water",
)
SO2SAT_OFFICIAL_SHA256 = "2f9aa3a0cbf7f5071d2fafee24156a6041a9c732f0f979881b8db582201aa7bc"
SO2SAT_WAVELENGTHS_NM = (
    492.9971095687347,
    559.5987534818435,
    664.6300422881802,
    704.0059319834206,
    740.5521320760564,
    782.4190761493182,
    827.5394062383036,
    864.7801257644385,
    1613.8624163477282,
    2203.6182057820033,
)
SO2SAT_NORMALIZATION_MEAN = (
    0.12951050698757172,
    0.11724399030208588,
    0.11381018161773682,
    0.12716519832611084,
    0.17067235708236694,
    0.19281364977359772,
    0.185484379529953,
    0.20729145407676697,
    0.1768450140953064,
    0.12849585711956024,
)
SO2SAT_NORMALIZATION_STD = (
    0.041423603892326355,
    0.05196256563067436,
    0.0733252465724945,
    0.06936437636613846,
    0.07505552470684052,
    0.0855887159705162,
    0.0865049883723259,
    0.09397122263908386,
    0.10238894075155258,
    0.09227467328310013,
)


def build_classification_head(
    embed_dim: int,
    num_classes: int,
    head_config: Optional[Dict[str, Any]] = None,
) -> nn.Sequential:
    """Build the shared downstream head from configuration."""
    head_config = head_config or {"architecture": "linear", "dropout": 0.0}
    architecture = head_config.get("architecture", "linear")
    dropout = float(head_config.get("dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("classification head dropout must satisfy 0 <= p < 1")
    dropout_module = nn.Dropout(dropout)
    if dropout > 0.0:
        designate_mc_dropout(dropout_module, "downstream_classification_head_before_final_linear")
    if architecture == "linear":
        return nn.Sequential(dropout_module, nn.Linear(embed_dim, num_classes))
    if architecture == "batchnorm_linear":
        modules: List[nn.Module] = [
            nn.BatchNorm1d(
                embed_dim,
                eps=float(head_config.get("batchnorm_eps", 1.0e-6)),
                affine=bool(head_config.get("batchnorm_affine", False)),
            )
        ]
        if dropout > 0.0:
            modules.append(dropout_module)
        modules.append(nn.Linear(embed_dim, num_classes))
        return nn.Sequential(*modules)
    if architecture == "mlp":
        hidden_dim = int(head_config.get("hidden_dim", embed_dim))
        return nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    raise ValueError(f"Unsupported head architecture: {architecture}")


class DOFARGBLinearProbe(nn.Module):
    """Pretrained DOFA backbone with a frozen RGB feature extractor and linear head."""

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_classes: int,
        wavelengths: List[float],
        freeze_backbone: bool = True,
        head_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.wavelengths = wavelengths
        self.freeze_backbone = freeze_backbone
        self.head = build_classification_head(embed_dim, num_classes, head_config)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def _encode_single_timestamp(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] != len(self.wavelengths):
            raise ValueError(
                f"DOFA received {image.shape[1]} channels but {len(self.wavelengths)} wavelengths were configured"
            )
        return self.backbone.forward_features(image, wave_list=self.wavelengths)

    def extract_features(self, image: torch.Tensor, temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return mean_temporal_encoder_features(image, self._encode_single_timestamp, temporal_mask)

    def forward(self, image: torch.Tensor, temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(self.extract_features(image, temporal_mask))


class PanopticonClassifier(nn.Module):
    """Panopticon backbone with the same downstream contract as the DOFA wrapper."""

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_classes: int,
        channel_ids_nm: List[float],
        freeze_backbone: bool = True,
        head_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if not channel_ids_nm:
            raise ValueError("Panopticon requires at least one optical channel ID")
        self.backbone = backbone
        self.channel_ids_nm = tuple(float(value) for value in channel_ids_nm)
        self.freeze_backbone = freeze_backbone
        self.head = build_classification_head(embed_dim, num_classes, head_config)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def _encode_single_timestamp(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] != len(self.channel_ids_nm):
            raise ValueError(
                f"Panopticon received {image.shape[1]} channels but {len(self.channel_ids_nm)} channel IDs were configured"
            )
        channel_ids = image.new_tensor(self.channel_ids_nm).unsqueeze(0).expand(image.shape[0], -1).clone()
        return self.backbone({"imgs": image, "chn_ids": channel_ids})

    def extract_features(self, image: torch.Tensor, temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return mean_temporal_encoder_features(image, self._encode_single_timestamp, temporal_mask)

    def forward(self, image: torch.Tensor, temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(self.extract_features(image, temporal_mask))


def mean_temporal_encoder_features(
    image: torch.Tensor,
    encoder,
    temporal_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply one shared spatial encoder per timestamp, then mean-pool features."""
    if image.ndim == 4:
        if temporal_mask is not None:
            raise ValueError("temporal_mask is only valid for [B,T,C,H,W] input")
        return encoder(image)
    if image.ndim != 5:
        raise ValueError(f"Expected [B,C,H,W] or [B,T,C,H,W], got {tuple(image.shape)}")
    batch_size, time_steps = image.shape[:2]
    if time_steps < 1:
        raise ValueError("Temporal input must contain at least one timestamp")
    encoded = encoder(image.flatten(0, 1)).reshape(batch_size, time_steps, -1)
    if temporal_mask is None:
        return encoded.mean(dim=1)
    if temporal_mask.shape != (batch_size, time_steps):
        raise ValueError(
            f"temporal_mask must have shape {(batch_size, time_steps)}, got {tuple(temporal_mask.shape)}"
        )
    weights = temporal_mask.to(device=encoded.device, dtype=encoded.dtype)
    counts = weights.sum(dim=1, keepdim=True)
    if bool((counts == 0).any()):
        raise ValueError("Every sample must have at least one valid timestamp")
    return (encoded * weights.unsqueeze(-1)).sum(dim=1) / counts


class EuroSATClassificationTransform:
    def __init__(
        self,
        image_size: int = 224,
        normalize: bool = True,
        input_bands: str = "all",
        normalization_mean: Optional[List[float]] = None,
        normalization_std: Optional[List[float]] = None,
    ):
        self.image_size = image_size
        self.normalize = normalize
        self.input_bands = input_bands.lower()
        if (normalization_mean is None) != (normalization_std is None):
            raise ValueError("normalization_mean and normalization_std must be configured together")
        self.normalization_mean = (
            torch.tensor(normalization_mean, dtype=torch.float32) if normalization_mean is not None else None
        )
        self.normalization_std = (
            torch.tensor(normalization_std, dtype=torch.float32) if normalization_std is not None else None
        )
        if self.normalization_std is not None and torch.any(self.normalization_std <= 0):
            raise ValueError("Every configured normalization standard deviation must be positive")

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"].float()
        if image.ndim == 4:
            image = image.squeeze(0)
        if self.input_bands == "rgb":
            # TorchGeo EuroSAT all-band order is Sentinel-2 B01, B02, B03, B04, ...
            # For RGB we keep only visible bands in image channel order:
            # B04 = red, B03 = green, B02 = blue.
            image = image[[3, 2, 1], ...]
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if self.normalize:
            if self.normalization_mean is not None and self.normalization_std is not None:
                if len(self.normalization_mean) != image.shape[0]:
                    raise ValueError(
                        f"Configured normalization has {len(self.normalization_mean)} channels, "
                        f"but transformed image has {image.shape[0]}"
                    )
                mean = self.normalization_mean.to(image.device).view(-1, 1, 1)
                std = self.normalization_std.to(image.device).view(-1, 1, 1)
            elif self.input_bands == "rgb":
                mean = SENTINEL2_MEAN[[3, 2, 1]].to(image.device).view(-1, 1, 1)
                std = SENTINEL2_STD[[3, 2, 1]].to(image.device).view(-1, 1, 1)
            else:
                mean = SENTINEL2_MEAN.to(image.device).view(-1, 1, 1)
                std = SENTINEL2_STD.to(image.device).view(-1, 1, 1)
            image = (image - mean) / std
        return {**sample, "image": image, "label": torch.as_tensor(sample["label"]).long()}


class ManifestEuroSATDataset(torch.utils.data.Dataset):
    """EuroSAT subset backed only by the persisted CSV split manifest."""

    def __init__(self, data_root: Path, records: List[Dict[str, str]], transform) -> None:
        self.data_root = data_root
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        path = self.data_root / record["file_path"]
        with rasterio.open(path) as dataset:
            image = torch.from_numpy(dataset.read()).float()
        sample = {
            "image": image,
            "label": int(record["class_index"]),
            "sample_id": record["sample_id"],
            "file_path": record["file_path"],
            "dataset_index": int(record["dataset_index"]),
            "split": record["split"],

        }
        return self.transform(sample)


@lru_cache(maxsize=8)
def _sha256_for_unchanged_file(path: str, size_bytes: int, modified_ns: int) -> str:
    """Hash a dataset file once per process, invalidating the cache if it changes."""
    del size_bytes, modified_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_artifact_record(path_value: str, expected_sha256: str, label: str) -> Dict[str, Any]:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Declared {label} artifact does not exist: {path}")
    stat = path.stat()
    actual_sha256 = _sha256_for_unchanged_file(str(path), stat.st_size, stat.st_mtime_ns)
    if actual_sha256 != str(expected_sha256).lower():
        raise RuntimeError(
            f"Declared {label} SHA256 mismatch for {path}: "
            f"expected {expected_sha256}, observed {actual_sha256}"
        )
    return {
        "label": label,
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "sha256": actual_sha256,
        "expected_sha256": str(expected_sha256).lower(),
        "verified": True,
    }


def collect_input_artifact_provenance(config: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Hash every dataset, manifest, and pretrained artifact declared for a final run."""
    provenance_cfg = config.get("provenance", {})
    required = bool(provenance_cfg.get("require_input_hashes", False))
    data_cfg = config["data"]
    records: Dict[str, Any] = {}
    declarations = (
        (data_cfg, "artifact_path", "artifact_sha256", "dataset"),
        (data_cfg, "split_manifest", "split_manifest_sha256", "split_manifest"),
        (data_cfg, "extracted_manifest_path", "extracted_manifest_sha256", "extracted_manifest"),
        (provenance_cfg, "protocol_path", "protocol_sha256", "training_protocol"),
    )
    for owner, path_key, hash_key, label in declarations:
        path_value = owner.get(path_key)
        expected = owner.get(hash_key)
        if path_value is None and expected is None:
            continue
        if not path_value or not expected:
            raise ValueError(f"{path_key} and {hash_key} must be declared together")
        records[label] = _input_artifact_record(str(path_value), str(expected), label)

    model_name = str(model_cfg.get("name"))
    if required and bool(model_cfg.get("pretrained", True)) and model_name in {"dofa", "panopticon"}:
        path_key = "weights_path" if model_name == "dofa" else "weights_cache_path"
        path_value = model_cfg.get(path_key)
        expected = model_cfg.get("weights_sha256")
        if not path_value or not expected:
            raise ValueError(
                f"Final pretrained {model_name} runs require model.{path_key} and model.weights_sha256"
            )
        records["pretrained_weights"] = _input_artifact_record(
            str(path_value), str(expected), f"{model_name}_pretrained_weights"
        )
    if required and not records:
        raise ValueError("provenance.require_input_hashes=true but no input artifacts were declared")
    return {
        "required": required,
        "all_verified": bool(records) and all(record["verified"] for record in records.values()),
        "records": records,
    }


class GeoBenchSo2SatClassificationDataset(torch.utils.data.Dataset):
    """Shared-pipeline adapter for the official GEO-Bench-2 m-So2Sat task."""

    def __init__(
        self,
        root: Path,
        split: str,
        image_size: int = 224,
        normalize: bool = True,
        download: bool = False,
        verify_checksum: bool = True,
    ) -> None:
        try:
            from geobench_v2.datasets import GeoBenchSo2Sat
            from geobench_v2.datasets.normalization import ZScoreNormalizer
        except ImportError as exc:
            raise RuntimeError(
                "So2Sat requires the pinned official GEO-Bench-2 package. Install the project environment lock."
            ) from exc

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported So2Sat split: {split}")
        if image_size <= 0:
            raise ValueError("So2Sat image_size must be positive")
        official_bands = tuple(GeoBenchSo2Sat.band_default_order["s2"])
        if official_bands != SO2SAT_S2_BANDS:
            raise RuntimeError(
                "Installed GEO-Bench-2 So2Sat optical band order differs from the validated protocol: "
                f"{official_bands} != {SO2SAT_S2_BANDS}"
            )
        if tuple(GeoBenchSo2Sat.classes) != SO2SAT_CLASSES:
            raise RuntimeError("Installed GEO-Bench-2 So2Sat class order differs from the validated protocol")
        installed_means = tuple(
            float(GeoBenchSo2Sat.normalization_stats["means"][band]) for band in SO2SAT_S2_BANDS
        )
        installed_stds = tuple(
            float(GeoBenchSo2Sat.normalization_stats["stds"][band]) for band in SO2SAT_S2_BANDS
        )
        if installed_means != SO2SAT_NORMALIZATION_MEAN or installed_stds != SO2SAT_NORMALIZATION_STD:
            raise RuntimeError("Installed GEO-Bench-2 So2Sat normalization differs from the validated protocol")

        data_normalizer: Any = ZScoreNormalizer if normalize else nn.Identity()
        self.dataset = GeoBenchSo2Sat(
            root=root,
            split=split,
            band_order={"s2": SO2SAT_S2_BANDS},
            data_normalizer=data_normalizer,
            download=download,
        )
        self.split = split
        self.image_size = int(image_size)
        self.class_names = SO2SAT_CLASSES

        tortilla_path = root / GeoBenchSo2Sat.paths[0]
        if verify_checksum:
            stat = tortilla_path.stat()
            actual_sha256 = _sha256_for_unchanged_file(
                str(tortilla_path.resolve()), stat.st_size, stat.st_mtime_ns
            )
            if actual_sha256 != SO2SAT_OFFICIAL_SHA256:
                raise ValueError(
                    f"So2Sat tortilla checksum mismatch: expected {SO2SAT_OFFICIAL_SHA256}, "
                    f"got {actual_sha256} for {tortilla_path}"
                )
        metadata = self.dataset.data_df
        required_metadata = {"patch_id", "labels", "tortilla:data_split"}
        missing_metadata = sorted(required_metadata - set(metadata.columns))
        if missing_metadata:
            raise ValueError(f"So2Sat metadata is missing required columns: {missing_metadata}")
        sample_ids = metadata["patch_id"].astype(str).tolist()
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"Duplicate So2Sat patch_id values detected in {split}")
        expected_embedded_split = "validation" if split == "val" else split
        embedded_splits = set(metadata["tortilla:data_split"].astype(str))
        if embedded_splits != {expected_embedded_split}:
            raise ValueError(
                f"So2Sat {split} loader selected unexpected embedded splits: {sorted(embedded_splits)}"
            )
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        official_sample = self.dataset[index]
        image = official_sample["image_s2"].float()
        if image.shape != (len(SO2SAT_S2_BANDS), 32, 32):
            raise ValueError(
                "Official So2Sat samples must be ten-band 32x32 tensors before model resizing; "
                f"got {tuple(image.shape)}"
            )
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if not torch.isfinite(image).all():
            raise FloatingPointError(f"Non-finite So2Sat image values for sample {self.sample_ids[index]}")

        label = int(official_sample["label"])
        if not 0 <= label < len(self.class_names):
            raise ValueError(f"Invalid So2Sat label index {label} for sample {self.sample_ids[index]}")
        metadata_label = str(self.dataset.data_df.iloc[index]["labels"])
        if metadata_label != self.class_names[label]:
            raise ValueError(
                f"So2Sat label mapping mismatch for {self.sample_ids[index]}: "
                f"index {label} maps to {self.class_names[label]!r}, metadata contains {metadata_label!r}"
            )
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": self.sample_ids[index],
            "dataset_index": index,
            "split": self.split,
        }

class TupleToDictDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image, label = self.dataset[index]
        return {"image": image, "label": torch.as_tensor(label).long()}


@dataclass
class ExperimentResult:
    name: str
    task: str
    dataset: str
    model: str
    checkpoint_path: str
    metrics: Dict[str, Any]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_dataloaders(config: Dict[str, Any], generator: Optional[torch.Generator] = None) -> Dict[str, DataLoader]:
    data_cfg = config["data"]
    dataset_name = str(data_cfg.get("name", "eurosat")).lower()
    input_bands = data_cfg.get("input_bands", "all").lower()
    normalization_cfg = data_cfg.get("normalization", {})
    if dataset_name == "treesatai":
        root = Path(data_cfg.get("root", "./datasets/treesatai"))
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        datasets_by_split = {
            split: GeoBenchTreeSatAITemporalDataset(
                root=root,
                split=split,
                image_size=int(data_cfg.get("image_size", 224)),
                download=bool(data_cfg.get("download", False)) and split == "train",
            )
            for split in ("train", "val", "test")
        }
        split_id_sets = {split: set(dataset.sample_ids) for split, dataset in datasets_by_split.items()}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = split_id_sets[left] & split_id_sets[right]
            if overlap:
                raise ValueError(f"TreeSatAI sample identity leakage between {left} and {right}: {len(overlap)}")
        loaders = {}
        for split, dataset in datasets_by_split.items():
            loaders[split] = DataLoader(
                dataset,
                batch_size=int(config["training"].get("batch_size", 32)),
                shuffle=split == "train",
                num_workers=int(data_cfg.get("num_workers", 0)),
                worker_init_fn=seed_dataloader_worker,
                generator=generator if split == "train" else None,
            )
        return loaders
    if dataset_name == "so2sat":
        if input_bands not in {"all", "s2"}:
            raise ValueError("The GEO-Bench-2 m-So2Sat protocol requires input_bands: s2")
        channels = tuple(data_cfg.get("channels", SO2SAT_S2_BANDS))
        if channels != SO2SAT_S2_BANDS:
            raise ValueError(
                f"The GEO-Bench-2 m-So2Sat protocol requires optical bands {SO2SAT_S2_BANDS}, got {channels}"
            )
        class_names = tuple(data_cfg.get("class_names", SO2SAT_CLASSES))
        if class_names != SO2SAT_CLASSES or int(data_cfg.get("num_classes", 17)) != len(SO2SAT_CLASSES):
            raise ValueError("So2Sat class names/order and num_classes must match the official 17-class protocol")
        wavelengths_nm = tuple(configured_wavelengths(config, "nanometers"))
        if wavelengths_nm != SO2SAT_WAVELENGTHS_NM:
            raise ValueError(
                "So2Sat wavelengths/order must match the validated Sentinel-2 spectral metadata; "
                f"got {wavelengths_nm}"
            )
        if bool(data_cfg.get("normalize", True)):
            means = tuple(float(value) for value in normalization_cfg.get("mean", ()))
            stds = tuple(float(value) for value in normalization_cfg.get("std", ()))
            if means != SO2SAT_NORMALIZATION_MEAN or stds != SO2SAT_NORMALIZATION_STD:
                raise ValueError("So2Sat normalization must match the official GEO-Bench-2 train statistics")

        root = Path(data_cfg.get("root", "./data/so2sat"))
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        datasets_by_split = {
            split: GeoBenchSo2SatClassificationDataset(
                root=root,
                split=split,
                image_size=int(data_cfg.get("image_size", 224)),
                normalize=bool(data_cfg.get("normalize", True)),
                download=bool(data_cfg.get("download", False)) and split == "train",
                verify_checksum=bool(data_cfg.get("verify_checksum", True)),
            )
            for split in ("train", "val", "test")
        }
        split_id_sets = {split: set(dataset.sample_ids) for split, dataset in datasets_by_split.items()}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = split_id_sets[left] & split_id_sets[right]
            if overlap:
                raise ValueError(
                    f"Official So2Sat split leakage: {len(overlap)} patch_id values overlap between {left} and {right}"
                )

        loaders = {}
        for offset, split in enumerate(("train", "val", "test")):
            split_generator = torch.Generator().manual_seed((int(config["seed"]) + offset) % (2**63))
            loaders[split] = DataLoader(
                datasets_by_split[split],
                batch_size=int(config["training"].get("batch_size", 32)),
                shuffle=split == "train",
                num_workers=int(data_cfg.get("num_workers", 4)),
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=seed_dataloader_worker,
                generator=split_generator,
            )
        return loaders

    if data_cfg.get("split_manifest"):
        root = Path(data_cfg.get("root", "./data"))
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        manifest_path = Path(data_cfg["split_manifest"])
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Configured EuroSAT split manifest not found: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        required_splits = {"train", "val", "calibration", "test"}
        available_splits = {record["split"] for record in records}
        if available_splits != required_splits:
            raise ValueError(f"Manifest splits must be {sorted(required_splits)}, got {sorted(available_splits)}")
        transform = EuroSATClassificationTransform(
            image_size=int(data_cfg.get("image_size", 224)),
            normalize=bool(data_cfg.get("normalize", True)),
            input_bands=input_bands,
            normalization_mean=normalization_cfg.get("mean"),
            normalization_std=normalization_cfg.get("std"),
        )
        datasets_by_split = {
            split: ManifestEuroSATDataset(root, [record for record in records if record["split"] == split], transform)
            for split in ("train", "val", "calibration", "test")
        }
        seed = config["seed"]
        loaders = {}
        for offset, split in enumerate(("train", "val", "calibration", "test")):
            split_generator = torch.Generator().manual_seed((seed + offset) % (2**63))
            loaders[split] = DataLoader(
                datasets_by_split[split], batch_size=int(config["training"].get("batch_size", 32)),
                shuffle=split == "train", num_workers=int(data_cfg.get("num_workers", 4)),
                pin_memory=torch.cuda.is_available(), worker_init_fn=seed_dataloader_worker,
                generator=split_generator,
            )
        return loaders
    if input_bands == "rgb" and data_cfg.get("source", "torchgeo_all_bands") == "torchvision_rgb":
        transform = transforms.Compose(
            [
                transforms.Resize((int(data_cfg.get("image_size", 224)), int(data_cfg.get("image_size", 224)))),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        dataset = datasets.EuroSAT(
            root=data_cfg.get("root", "./data"),
            transform=transform,
            download=bool(data_cfg.get("download", True)),
        )
        val_fraction = float(data_cfg.get("val_fraction", 0.2))
        val_size = int(len(dataset) * val_fraction)
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(config["seed"] % (2**63))
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

        loader_kwargs = {
            "batch_size": int(config["training"].get("batch_size", 32)),
            "num_workers": int(data_cfg.get("num_workers", 4)),
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": seed_dataloader_worker,
            "generator": generator,
        }
        return {
            "train": DataLoader(TupleToDictDataset(train_dataset), shuffle=True, **loader_kwargs),
            "val": DataLoader(TupleToDictDataset(val_dataset), shuffle=False, **loader_kwargs),
        }

    if EuroSAT is None:
        raise RuntimeError(
            "TorchGeo is required for the 13-band EuroSAT path. Install torchgeo "
            "or set data.input_bands: rgb to use the RGB first-stage experiment."
        )

    transform = EuroSATClassificationTransform(
        image_size=int(data_cfg.get("image_size", 224)),
        normalize=bool(data_cfg.get("normalize", True)),
        input_bands=input_bands,
        normalization_mean=normalization_cfg.get("mean"),
        normalization_std=normalization_cfg.get("std"),
    )
    root = Path(data_cfg.get("root", "./data/eurosat"))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    split_files = data_cfg["split_files"]
    normalized_split_files = {}
    for split, filename in split_files.items():
        split_path = Path(filename)
        if split_path.is_absolute():
            try:
                split_path = split_path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"Split file must be inside data.root: {filename}") from exc
        if not (root / split_path).is_file():
            raise FileNotFoundError(f"Configured {split} split file not found: {root / split_path}")
        normalized_split_files[split] = str(split_path)

    class ConfiguredEuroSAT(EuroSAT):
        split_filenames = {**EuroSAT.split_filenames, **normalized_split_files}

    train_dataset = ConfiguredEuroSAT(root=root, split="train", transforms=transform, download=bool(data_cfg.get("download", True)))
    val_dataset = ConfiguredEuroSAT(root=root, split="val", transforms=transform, download=False)
    test_dataset = None
    if "test" in normalized_split_files:
        test_dataset = ConfiguredEuroSAT(root=root, split="test", transforms=transform, download=False)

    loader_kwargs = {
        "batch_size": int(config["training"].get("batch_size", 32)),
        "num_workers": int(data_cfg.get("num_workers", 4)),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_dataloader_worker,
        "generator": generator,
    }
    loaders = {
        "train": DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_dataset, shuffle=False, **loader_kwargs),
    }
    if test_dataset is not None:
        loaders["test"] = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return loaders


def resolve_weights_path(model_cfg: Dict[str, Any]) -> Optional[str]:
    weights_path = model_cfg.get("weights_path")
    if weights_path:
        path = Path(weights_path)
        return str(path) if path.exists() else None

    model_size = model_cfg.get("size", "base")
    if model_size == "base":
        default_name = "DOFA_ViT_base_e100.pth"
    else:
        default_name = f"DOFA_ViT_{model_size}_e100.pth"
    default_path = PROJECT_ROOT / "DOFA" / "checkpoints" / default_name
    return str(default_path) if default_path.exists() else None


def configured_wavelengths(config: Dict[str, Any], target_units: str) -> List[float]:
    """Return canonical dataset wavelengths in the units required by a backbone."""
    data_cfg = config["data"]
    values = data_cfg.get("wavelengths_nm")
    if not isinstance(values, list) or not values:
        raise ValueError("Canonical data.wavelengths_nm must be configured explicitly")
    wavelengths_nm = [float(value) for value in values]
    channels = data_cfg.get("channels")
    if channels is not None and len(wavelengths_nm) != len(channels):
        raise ValueError("data.wavelengths_nm and data.channels must have the same length")
    if any(not 300.0 <= value <= 2500.0 for value in wavelengths_nm):
        raise ValueError(
            "data.wavelengths_nm is not on the expected optical nanometer scale [300, 2500]"
        )

    aliases = {
        "um": "micrometers",
        "µm": "micrometers",
        "micrometer": "micrometers",
        "micrometers": "micrometers",
        "nm": "nanometers",
        "nanometer": "nanometers",
        "nanometers": "nanometers",
    }
    requested_units = aliases.get(target_units.lower())
    if requested_units is None:
        raise ValueError(f"Unsupported target wavelength units: {target_units}")
    if requested_units == "nanometers":
        return wavelengths_nm
    wavelengths_um = [value / 1000.0 for value in wavelengths_nm]
    if any(not 0.3 <= value <= 2.5 for value in wavelengths_um):
        raise ValueError("DOFA wavelengths are not on the expected optical micrometer scale [0.3, 2.5]")
    return wavelengths_um


def model_display_name(model_cfg: Dict[str, Any]) -> str:
    """Stable model label for result and prediction metadata."""
    name = str(model_cfg.get("name", "unknown"))
    if name in {"dofa", "panopticon"}:
        return f"{name}-{model_cfg.get('size', 'base')}"
    return name


def classification_type_from_config(config: Dict[str, Any]) -> str:
    """Resolve the target contract shared by training, evaluation, and export."""
    task = str(config.get("task", "classification"))
    multilabel = bool(config.get("data", {}).get("multilabel", False))
    if task == "classification_multilabel" or multilabel:
        if task != "classification_multilabel" or not multilabel:
            raise ValueError("Multi-label classification requires task=classification_multilabel and data.multilabel=true")
        return "multilabel"
    if task != "classification":
        raise ValueError(f"Unsupported classification task: {task}")
    return "multiclass"


def build_model(config: Dict[str, Any], model_cfg: Dict[str, Any], num_classes: int) -> torch.nn.Module:
    model_name = model_cfg.get("name")

    if model_name in {"dofa", "panopticon"}:
        expected_units = "micrometers" if model_name == "dofa" else "nanometers"
        expected_wavelengths = configured_wavelengths(config, expected_units)
        saved_wavelengths = model_cfg.get("actual_wavelengths")
        saved_units = model_cfg.get("actual_wavelength_units")
        if saved_wavelengths is not None and [float(value) for value in saved_wavelengths] != expected_wavelengths:
            raise ValueError(f"{model_name} model.actual_wavelengths disagrees with canonical data.wavelengths_nm")
        if saved_units is not None and saved_units != expected_units:
            raise ValueError(f"{model_name} model.actual_wavelength_units must be {expected_units}")

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if model_cfg.get("pretrained", True) else None
        model = models.resnet18(weights=weights)
        if bool(model_cfg.get("freeze_backbone", True)):
            for parameter in model.parameters():
                parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "panopticon":
        model_size = str(model_cfg.get("size", "base"))
        if model_size != "base":
            raise ValueError("Panopticon currently supports model.size: base only")
        try:
            from torchgeo.models import Panopticon_Weights, panopticon_vitb14
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("Panopticon requires TorchGeo >= 0.7 with panopticon_vitb14 support") from exc

        weights = None
        pretrained_audit = None
        if bool(model_cfg.get("pretrained", True)):
            weights_name = str(model_cfg.get("weights", "VIT_BASE14"))
            try:
                weights = Panopticon_Weights[weights_name]
            except KeyError as exc:
                supported = ", ".join(item.name for item in Panopticon_Weights)
                raise ValueError(f"Unsupported Panopticon weights: {weights_name}. Expected one of: {supported}") from exc
            pretrained_audit = {
                "weights": weights.name,
                "source": weights.url,
                "missing_keys": [],
                "unexpected_keys": [],
            }
        backbone = panopticon_vitb14(
            weights=weights,
            img_size=int(config["data"].get("image_size", 224)),
        )
        if not bool(model_cfg.get("freeze_backbone", True)):
            named_backbone_parameters = dict(backbone.named_parameters())
            expected_frozen = list(model_cfg.get("expected_frozen_backbone_parameters", []))
            unknown_frozen = sorted(set(expected_frozen) - set(named_backbone_parameters))
            if unknown_frozen:
                raise ValueError(f"Unknown Panopticon expected_frozen_backbone_parameters: {unknown_frozen}")
            for name in expected_frozen:
                named_backbone_parameters[name].requires_grad = False
        backbone._pretrained_load_audit = pretrained_audit
        return PanopticonClassifier(
            backbone=backbone,
            embed_dim=int(backbone.model.num_features),
            num_classes=num_classes,
            channel_ids_nm=expected_wavelengths,
            freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
            head_config=model_cfg.get("head"),
        )

    if model_name != "dofa":
        raise ValueError(f"Unsupported model: {model_name}. Expected one of: resnet18, dofa, panopticon")

    weights_path = resolve_weights_path(model_cfg)
    if model_cfg.get("pretrained", True) and weights_path is None:
        size = model_cfg.get("size", "base")
        raise FileNotFoundError(
            f"Pretrained DOFA {size} weights were requested but no checkpoint was found. "
            "Place DOFA_ViT_base_e100.pth under DOFA/checkpoints/ or set model.weights_path."
        )

    dofa_root = Path(model_cfg.get("dofa_root", PROJECT_ROOT / "DOFA"))
    if not dofa_root.is_absolute():
        dofa_root = PROJECT_ROOT / dofa_root
    if not dofa_root.exists():
        raise RuntimeError(
            "DOFA source directory was not found. Clone/download DOFA under the project root "
            f"or set model.dofa_root. Configured path: {dofa_root}"
        )
    if str(dofa_root) not in sys.path:
        sys.path.insert(0, str(dofa_root))

    try:
        from dofa_v1 import vit_base_patch16, vit_large_patch16, vit_small_patch16
    except ImportError as exc:
        missing_name = getattr(exc, "name", None) or "a DOFA dependency"
        raise RuntimeError(
            "Could not import DOFA. Make sure the DOFA checkout is present and install "
            "its dependencies, especially timm. Example:\n"
            "  /opt/conda/bin/python -m pip install timm\n"
            f"Original missing module: {missing_name}"
        ) from exc

    factories = {
        "small": (vit_small_patch16, 384),
        "base": (vit_base_patch16, 768),
        "large": (vit_large_patch16, 1024),
    }
    model_size = model_cfg.get("size", "base")
    if model_size not in factories:
        raise ValueError(f"Unsupported DOFA size: {model_size}. Expected one of: small, base, large")

    factory, embed_dim = factories[model_size]
    backbone = factory(
        img_size=int(config["data"].get("image_size", 224)),
        num_classes=0,
        drop_rate=float(model_cfg.get("drop_rate", 0.0)),
        global_pool=bool(model_cfg.get("global_pool", False)),
    )

    if model_cfg.get("pretrained", True):
        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        msg = backbone.load_state_dict(state_dict, strict=False)
        backbone._pretrained_load_audit = {
            "checkpoint": str(weights_path),
            "missing_keys": list(msg.missing_keys),
            "unexpected_keys": list(msg.unexpected_keys),
        }
        print(f"Loaded DOFA checkpoint from {weights_path}")
        if msg.missing_keys:
            print(f"DOFA missing keys ignored for linear probing: {msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"DOFA unexpected keys ignored for linear probing: {msg.unexpected_keys}")

    return DOFARGBLinearProbe(
        backbone=backbone,
        embed_dim=embed_dim,
        num_classes=num_classes,
        wavelengths=expected_wavelengths,
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        head_config=model_cfg.get("head"),
    )


def make_model(model_cfg: Dict[str, Any], num_classes: int) -> torch.nn.Module:
    return build_model(
        {
            "data": {
                "image_size": 224,
                "input_bands": "rgb",
                "channels": ["B04", "B03", "B02"],
                "wavelengths_nm": [665.0, 560.0, 490.0],
            }
        },
        model_cfg,
        num_classes,
    )


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {"image": batch["image"].to(device), "label": batch["label"].to(device)}
    if "temporal_mask" in batch:
        moved["temporal_mask"] = batch["temporal_mask"].to(device)
    return moved


def forward_classification_batch(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "temporal_mask" in batch:
        return model(batch["image"], temporal_mask=batch["temporal_mask"])
    return model(batch["image"])


def classification_loss(logits: torch.Tensor, labels: torch.Tensor, classification_type: str) -> torch.Tensor:
    if classification_type == "multiclass":
        return F.cross_entropy(logits, labels.long())
    if classification_type == "multilabel":
        if labels.shape != logits.shape:
            raise ValueError(f"Multilabel targets must match logits shape, got {labels.shape} and {logits.shape}")
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    raise ValueError("classification_type must be multiclass or multilabel")


def classification_correctness(
    logits: torch.Tensor,
    labels: torch.Tensor,
    classification_type: str,
    threshold: float = 0.5,
) -> torch.Tensor:
    if classification_type == "multiclass":
        return logits.argmax(dim=1).eq(labels.long())
    if classification_type == "multilabel":
        predictions = logits.sigmoid().ge(threshold)
        return predictions.eq(labels.bool()).all(dim=1)
    raise ValueError("classification_type must be multiclass or multilabel")


def compute_task_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    classification_type: str,
    n_bins: int,
    threshold: float = 0.5,
) -> Dict[str, float]:
    if classification_type == "multiclass":
        return compute_classification_metrics(logits, labels.long(), n_bins=n_bins).to_dict()
    if classification_type != "multilabel":
        raise ValueError("classification_type must be multiclass or multilabel")
    if labels.shape != logits.shape:
        raise ValueError(f"Multilabel targets must match logits shape, got {labels.shape} and {logits.shape}")
    targets = labels.float()
    probabilities = logits.sigmoid()
    predictions = probabilities.ge(threshold)
    target_binary = targets.bool()
    exact_match = predictions.eq(target_binary).all(dim=1).float().mean()
    labelwise_accuracy = predictions.eq(target_binary).float().mean()
    true_positive = (predictions & target_binary).sum(dim=0).float()
    false_positive = (predictions & ~target_binary).sum(dim=0).float()
    false_negative = (~predictions & target_binary).sum(dim=0).float()
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    macro_f1 = (2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)).mean()
    binary_probabilities = torch.stack((1.0 - probabilities, probabilities), dim=-1).reshape(-1, 2)
    ece = compute_ece(binary_probabilities, target_binary.long().reshape(-1), n_bins=n_bins)
    return {
        "accuracy": float(exact_match.item()),
        "labelwise_accuracy": float(labelwise_accuracy.item()),
        "macro_f1": float(macro_f1.item()),
        "nll": float(F.binary_cross_entropy_with_logits(logits, targets).item()),
        "ece": float(ece.item()),
        "brier": float(torch.square(probabilities - targets).mean().item()),
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    limit_batches: Optional[int] = None,
    epoch: int = 1,
    global_step: int = 0,
    track_gradient_norm: bool = True,
    audit_gradients: bool = False,
    classification_type: str = "multiclass",
    multilabel_threshold: float = 0.5,
) -> Dict[str, float]:
    model.train()
    if getattr(model, "freeze_backbone", False) and hasattr(model, "backbone"):
        model.backbone.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    gradient_norm_sum = 0.0
    gradient_norm_count = 0
    backbone_gradient_norm_sum = 0.0
    head_gradient_norm_sum = 0.0
    gradient_audit: Dict[str, Any] = {
        "performed": False,
        "all_expected_parameters_received_gradient": None,
        "missing_gradient_parameters": [],
        "zero_gradient_parameters": [],
        "nonfinite_gradient_parameters": [],
    }
    backbone_named_parameters = list(getattr(model, "backbone", nn.Module()).named_parameters())
    head_named_parameters = list(getattr(model, "head", nn.Module()).named_parameters())
    batch_history: List[Dict[str, float]] = []

    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_classification_batch(model, batch)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite logits detected at epoch={epoch}, batch={batch_idx + 1}")
        loss = classification_loss(logits, batch["label"], classification_type)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss detected at epoch={epoch}, batch={batch_idx + 1}")
        loss.backward()
        gradient_norm = math.nan
        backbone_gradient_norm = math.nan
        head_gradient_norm = math.nan
        if track_gradient_norm:
            def group_gradient_norm(named_parameters):
                return math.sqrt(
                    sum(
                        float(parameter.grad.detach().float().norm(2).item()) ** 2
                        for _, parameter in named_parameters
                        if parameter.requires_grad and parameter.grad is not None
                    )
                )

            backbone_gradient_norm = group_gradient_norm(backbone_named_parameters)
            head_gradient_norm = group_gradient_norm(head_named_parameters)
            squared_norm = backbone_gradient_norm**2 + head_gradient_norm**2
            gradient_norm = math.sqrt(squared_norm)
            gradient_norm_sum += gradient_norm
            backbone_gradient_norm_sum += backbone_gradient_norm
            head_gradient_norm_sum += head_gradient_norm
            gradient_norm_count += 1
        if audit_gradients and batch_idx == 0:
            expected = [
                (f"backbone.{name}", parameter) for name, parameter in backbone_named_parameters
                if parameter.requires_grad
            ] + [
                (f"head.{name}", parameter) for name, parameter in head_named_parameters
                if parameter.requires_grad
            ]
            missing = [name for name, parameter in expected if parameter.grad is None]
            nonfinite = [
                name for name, parameter in expected
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            zero = [
                name for name, parameter in expected
                if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item() == 0)
            ]
            gradient_audit = {
                "performed": True,
                "expected_parameter_tensors": len(expected),
                "all_expected_parameters_received_gradient": not missing,
                "missing_gradient_parameters": missing,
                "zero_gradient_parameters": zero,
                "nonfinite_gradient_parameters": nonfinite,
            }
            if missing:
                raise RuntimeError(f"Expected trainable parameters did not receive gradients: {missing}")
            if nonfinite:
                raise FloatingPointError(f"Non-finite parameter gradients detected: {nonfinite}")
        else:
            nonfinite = [
                f"{group}.{name}"
                for group, named_parameters in (("backbone", backbone_named_parameters), ("head", head_named_parameters))
                for name, parameter in named_parameters
                if parameter.requires_grad
                and parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ]
            if nonfinite:
                raise FloatingPointError(
                    f"Non-finite parameter gradients detected at epoch={epoch}, batch={batch_idx + 1}: {nonfinite}"
                )
        optimizer.step()

        batch_size = int(batch["label"].shape[0])
        batch_correct = int(
            classification_correctness(logits, batch["label"], classification_type, multilabel_threshold).sum().item()
        )
        total_loss += float(loss.item()) * batch_size
        correct += batch_correct
        count += batch_size
        batch_history.append(
            {
                "epoch": epoch,
                "batch": batch_idx + 1,
                "global_step": global_step + len(batch_history) + 1,
                "loss": float(loss.item()),
                "accuracy": batch_correct / max(batch_size, 1),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "backbone_learning_rate": next(
                    (float(group["lr"]) for group in optimizer.param_groups if group.get("name") == "backbone"),
                    math.nan,
                ),
                "head_learning_rate": next(
                    (float(group["lr"]) for group in optimizer.param_groups if group.get("name") == "head"),
                    math.nan,
                ),
                "gradient_norm": gradient_norm,
                "backbone_gradient_norm": backbone_gradient_norm,
                "head_gradient_norm": head_gradient_norm,
            }
        )

    return {
        "loss": total_loss / max(count, 1),
        "accuracy": correct / max(count, 1),
        "gradient_norm": gradient_norm_sum / max(gradient_norm_count, 1),
        "backbone_gradient_norm": backbone_gradient_norm_sum / max(gradient_norm_count, 1),
        "head_gradient_norm": head_gradient_norm_sum / max(gradient_norm_count, 1),
        "gradient_audit": gradient_audit,
        "batch_history": batch_history,
    }


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    model.eval()
    logits: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        batch = move_batch(batch, device)
        logits.append(forward_classification_batch(model, batch).cpu())
        labels.append(batch["label"].cpu())
    return {"logits": torch.cat(logits), "labels": torch.cat(labels)}


def add_uncertainty_summaries(
    metrics: Dict[str, float], logits: torch.Tensor, classification_type: str = "multiclass"
) -> Dict[str, float]:
    if classification_type == "multiclass":
        probabilities = logits.softmax(dim=1)
        confidence = probabilities.max(dim=1).values
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    elif classification_type == "multilabel":
        probabilities = logits.sigmoid()
        confidence = torch.maximum(probabilities, 1.0 - probabilities).mean(dim=1)
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
            + (1.0 - probabilities) * (1.0 - probabilities).clamp_min(1e-12).log()
        ).sum(dim=1)
    else:
        raise ValueError("classification_type must be multiclass or multilabel")
    metrics["mean_confidence"] = float(confidence.mean().item())
    metrics["predictive_entropy"] = float(entropy.mean().item())
    return metrics


def enable_dropout(model: torch.nn.Module) -> None:
    activate_downstream_mc_dropout(model)


@torch.no_grad()
def collect_mc_dropout_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    samples: int,
    classification_type: str = "multiclass",
) -> Dict[str, torch.Tensor]:
    model.eval()
    enable_dropout(model)
    probs_per_sample: List[torch.Tensor] = []
    labels: Optional[torch.Tensor] = None

    for _ in range(samples):
        probs: List[torch.Tensor] = []
        labels_this_pass: List[torch.Tensor] = []
        for batch in loader:
            batch = move_batch(batch, device)
            pass_logits = forward_classification_batch(model, batch)
            probabilities = pass_logits.softmax(dim=1) if classification_type == "multiclass" else pass_logits.sigmoid()
            probs.append(probabilities.cpu())
            labels_this_pass.append(batch["label"].cpu())
        probs_per_sample.append(torch.cat(probs))
        if labels is None:
            labels = torch.cat(labels_this_pass)

    mean_probs = torch.stack(probs_per_sample).mean(dim=0)
    if classification_type == "multiclass":
        aggregate_logits = mean_probs.clamp_min(1e-12).log()
    elif classification_type == "multilabel":
        aggregate_logits = torch.logit(mean_probs.clamp(1e-7, 1.0 - 1e-7))
    else:
        raise ValueError("classification_type must be multiclass or multilabel")
    return {"logits": aggregate_logits, "labels": labels}


def evaluate_uq_methods(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    methods: Iterable[Dict[str, Any]],
    n_bins: int,
    limit_batches: Optional[int] = None,
    classification_type: str = "multiclass",
    multilabel_threshold: float = 0.5,
) -> Dict[str, Any]:
    raw = collect_logits(model, val_loader, device, limit_batches=limit_batches)
    results: Dict[str, Any] = {
        "none": compute_task_classification_metrics(
            raw["logits"], raw["labels"], classification_type, n_bins, multilabel_threshold
        )
    }

    for method in methods:
        name = method.get("name", "none")
        if name == "none":
            continue
        if name == "temperature_scaling":
            if classification_type == "multilabel":
                raise ValueError("Temperature Scaling for multi-label classification is not implemented or enabled yet")
            scaler = TemperatureScaler()
            temperature = scaler.fit(raw["logits"], raw["labels"], max_iter=int(method.get("max_iter", 50)))
            scaled_logits = scaler(raw["logits"]).detach()
            metrics = compute_classification_metrics(scaled_logits, raw["labels"], n_bins=n_bins).to_dict()
            metrics["temperature"] = temperature
            metrics["note"] = "Temperature fitted and evaluated on the validation split; use a held-out calibration split for final claims."
            results[name] = metrics
        elif name == "mc_dropout":
            mc = collect_mc_dropout_probs(
                model,
                val_loader,
                device,
                samples=int(method.get("samples", 10)),
                classification_type=classification_type,
            )
            metrics = compute_task_classification_metrics(
                mc["logits"], mc["labels"], classification_type, n_bins, multilabel_threshold
            )
            metrics["samples"] = int(method.get("samples", 10))
            results[name] = metrics
        elif name == "downstream_mc_dropout":
            # This configuration label identifies a model trained with the
            # prospectively frozen downstream dropout layer.  Final stochastic
            # evaluation is deliberately deferred to the complete-test C5
            # inference stage (fixed T=30); checkpoint selection remains the
            # deterministic validation metric already stored under ``none``.
            results[name] = {
                "status": "deferred_to_final_c5_inference",
                "dropout_probability": float(method.get("dropout_probability", 0.10)),
                "stochastic_passes": int(method.get("stochastic_passes", 30)),
                "aggregation": str(method.get("aggregation", "arithmetic_mean_probabilities")),
                "validation_stochastic_inference_performed": False,
                "test_access": False,
            }
        else:
            raise ValueError(f"Unsupported UQ method: {name}")

    return results


def append_results_csv(path: Path, row: Dict[str, Any]) -> None:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists and row.get("run_id") is not None:
        with path.open(newline="", encoding="utf-8") as handle:
            duplicate = next(
                (record for record in csv.DictReader(handle) if record.get("run_id") == str(row["run_id"])),
                None,
            )
        if duplicate is not None:
            raise ValueError(f"results CSV already contains run_id={row['run_id']}")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_optimizer(model: nn.Module, training_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_cfg = training_cfg.get("optimizer", {})
    optimizer_name = str(optimizer_cfg.get("name", "adamw")).lower()
    if optimizer_name not in {"adamw", "adam", "sgd"}:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    head_parameters = list(model.head.parameters()) if hasattr(model, "head") else []
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in head_ids
    ]
    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": float(training_cfg["backbone_learning_rate"]),
                "base_lr": float(training_cfg["backbone_learning_rate"]),
                "name": "backbone",
            }
        )
    trainable_head = [parameter for parameter in head_parameters if parameter.requires_grad]
    if trainable_head:
        parameter_groups.append(
            {
                "params": trainable_head,
                "lr": float(training_cfg["head_learning_rate"]),
                "base_lr": float(training_cfg["head_learning_rate"]),
                "name": "head",
            }
        )
    if not parameter_groups:
        parameter_groups = [{"params": [parameter for parameter in model.parameters() if parameter.requires_grad]}]

    common = {"weight_decay": float(training_cfg.get("weight_decay", 0.0))}
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameter_groups, momentum=float(optimizer_cfg.get("momentum", 0.9)), **common)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameter_groups, **common)
    return torch.optim.AdamW(parameter_groups, **common)


def set_epoch_learning_rates(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    warmup_cfg: Dict[str, Any],
) -> Dict[str, float]:
    """Apply linear epoch-level warm-up followed by constant group learning rates."""
    warmup_enabled = bool(warmup_cfg.get("enabled", False))
    warmup_epochs = int(warmup_cfg.get("epochs", 0)) if warmup_enabled else 0
    factor = min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0
    learning_rates = {}
    for index, group in enumerate(optimizer.param_groups):
        base_lr = float(group.get("base_lr", group["lr"]))
        group["base_lr"] = base_lr
        group["lr"] = base_lr * factor
        learning_rates[str(group.get("name", f"group_{index}"))] = float(group["lr"])
    return {"warmup_factor": factor, **learning_rates}


def capture_rng_state() -> Dict[str, Any]:
    """Capture process RNG state in a weights-only-checkpoint-safe form."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": [int(value) for value in numpy_state[1]],
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""
    random.setstate(tuple(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def capture_dataloader_generator_states(dataloaders: Dict[str, DataLoader]) -> Dict[str, torch.Tensor]:
    """Capture explicitly configured DataLoader generator states."""
    return {
        split: loader.generator.get_state()
        for split, loader in dataloaders.items()
        if loader.generator is not None
    }


def restore_dataloader_generator_states(
    dataloaders: Dict[str, DataLoader],
    states: Dict[str, torch.Tensor],
) -> None:
    for split, state in states.items():
        if split not in dataloaders or dataloaders[split].generator is None:
            raise ValueError(f"Checkpoint contains an unavailable DataLoader generator state: {split}")
        dataloaders[split].generator.set_state(state.cpu())


def reconstruct_classification_loader_states(
    dataloaders: Dict[str, DataLoader],
    completed_epochs: int,
) -> None:
    """Reproduce generator consumption for legacy checkpoints without saved states.

    Every completed classification epoch iterates train and then validation. A
    DataLoader first draws its worker base seed and then consumes its index
    sampler. Exhausting only the index sampler reproduces the generator state
    without loading samples or performing model inference.
    """
    if completed_epochs < 0:
        raise ValueError("completed_epochs must be non-negative")
    for _ in range(completed_epochs):
        for split in ("train", "val"):
            loader = dataloaders[split]
            if loader.generator is None:
                continue
            torch.empty((), dtype=torch.int64).random_(generator=loader.generator)
            for _indices in loader._index_sampler:  # mirrors DataLoader iterator consumption
                pass


def stochastic_training_modules(model: nn.Module) -> List[str]:
    stochastic_names = {"Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "DropPath", "StochasticDepth"}
    return [
        name or "<root>"
        for name, module in model.named_modules()
        if module.training and module.__class__.__name__ in stochastic_names
    ]


def _load_torch_checkpoint(path: Path, map_location: Any) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(value: Dict[str, Any], path: Path) -> None:
    """Write a checkpoint completely before atomically replacing its target."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _snapshot_changes(original: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, List[str]]:
    original_files = {item["path"]: item["sha256"] for item in original["files"]}
    current_files = {item["path"]: item["sha256"] for item in current["files"]}
    return {
        "changed": sorted(
            path for path in original_files.keys() & current_files.keys()
            if original_files[path] != current_files[path]
        ),
        "added": sorted(current_files.keys() - original_files.keys()),
        "removed": sorted(original_files.keys() - current_files.keys()),
    }


def prepare_classification_run_directory(
    config: Dict[str, Any],
    experiment: Dict[str, Any],
    resume_run_dir: Optional[Path],
) -> tuple[str, Dict[str, Any], Path, Optional[Dict[str, Any]]]:
    """Create a new run location or validate an explicit in-place resume target."""
    seed = int(config["seed"])
    output_root = Path(config.get("output_dir", "./results/experiments"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    if resume_run_dir is None:
        run_id = make_run_id(experiment["name"], seed)
        output_dir = output_root / "runs" / run_id
        run_config = copy.deepcopy(config)
        run_config["run_id"] = run_id
        run_config["active_experiment"] = experiment["name"]
        write_resolved_config(run_config, output_dir)
        code_snapshot = write_code_snapshot_manifest(output_dir)
        write_json(output_dir / "environment.json", collect_environment_metadata(code_snapshot=code_snapshot))
        return run_id, run_config, output_dir, None

    output_dir = resume_run_dir.resolve()
    expected_runs_root = (output_root / "runs").resolve()
    if output_dir.parent != expected_runs_root:
        raise ValueError(f"Resume directory must be an immediate child of {expected_runs_root}: {output_dir}")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Resume run directory does not exist: {output_dir}")
    required = ("resolved_config.yaml", "code_snapshot.json", "environment.json", "best.pt", "last.pt")
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Resume run is missing required artifacts: {missing}")
    if (output_dir / "run_summary.json").exists():
        raise ValueError("Refusing to resume a run that already has run_summary.json")

    original_config = yaml.safe_load((output_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if not isinstance(original_config, dict):
        raise ValueError("Resume resolved_config.yaml is not a mapping")
    run_id = str(original_config.get("run_id", ""))
    if not run_id or output_dir.name != run_id:
        raise ValueError("Resume directory name and recorded run_id do not match")
    expected_config = copy.deepcopy(config)
    expected_config["run_id"] = run_id
    expected_config["active_experiment"] = experiment["name"]
    if original_config != expected_config:
        raise ValueError("Current resolved configuration does not exactly match the interrupted run")

    checkpoint = _load_torch_checkpoint(output_dir / "last.pt", "cpu")
    if checkpoint.get("run_id") != run_id:
        raise ValueError("last.pt run_id does not match the resume directory")
    if checkpoint.get("experiment") != experiment:
        raise ValueError("last.pt experiment does not exactly match the selected experiment")
    if checkpoint.get("config") != config:
        raise ValueError("last.pt configuration does not exactly match the selected configuration")
    if int(checkpoint.get("epoch", 0)) >= int(config["training"]["epochs"]):
        raise ValueError("Resume checkpoint has already reached the configured epoch limit")
    history = checkpoint.get("history", [])
    expected_epochs = list(range(1, int(checkpoint["epoch"]) + 1))
    if [int(item.get("epoch", -1)) for item in history] != expected_epochs:
        raise ValueError("last.pt history is not contiguous through its recorded epoch")

    original_snapshot = json.loads((output_dir / "code_snapshot.json").read_text(encoding="utf-8"))
    original_environment = json.loads((output_dir / "environment.json").read_text(encoding="utf-8"))
    if original_environment.get("code_version") != f"sha256:{original_snapshot['code_sha256']}":
        raise ValueError("Original environment and code snapshot hashes do not agree")
    current_snapshot = collect_code_snapshot()
    code_changes = _snapshot_changes(original_snapshot, current_snapshot)
    allowed_resume_changes = {
        "scripts/run_experiments.py",
        "tests/test_experiment_configuration.py",
    }
    unexpected_changes = sorted(
        (set(code_changes["changed"]) | set(code_changes["added"]) | set(code_changes["removed"]))
        - allowed_resume_changes
    )
    if unexpected_changes:
        raise ValueError(f"Unexpected code/config changes since the interrupted run: {unexpected_changes}")

    return run_id, original_config, output_dir, {
        "checkpoint": checkpoint,
        "original_code_sha256": original_snapshot["code_sha256"],
        "original_snapshot": original_snapshot,
        "current_snapshot": current_snapshot,
        "code_changes": code_changes,
    }


def metric_improved(value: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    return value < best - min_delta if mode == "min" else value > best + min_delta


def audit_model_for_training(model: nn.Module, model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate adaptation invariants and describe train/eval-sensitive modules."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    backbone = getattr(model, "backbone", None)
    head = getattr(model, "head", None)
    if backbone is None or head is None:
        raise ValueError("Classification model must expose backbone and head for parameter auditing")

    backbone_total = sum(parameter.numel() for parameter in backbone.parameters())
    backbone_trainable = sum(parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad)
    head_total = sum(parameter.numel() for parameter in head.parameters())
    head_trainable = sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
    frozen = model_cfg.get("adaptation_mode") == "frozen"
    structurally_frozen_backbone = {
        name: parameter.numel() for name, parameter in backbone.named_parameters() if not parameter.requires_grad
    }
    if frozen and backbone_trainable != 0:
        raise RuntimeError(
            f"Frozen-backbone invariant failed: {backbone_trainable} backbone parameters require gradients"
        )
    if frozen and head_trainable == 0:
        raise RuntimeError("Frozen-backbone invariant failed: classification head has no trainable parameters")
    if not frozen:
        expected_frozen = set(model_cfg.get("expected_frozen_backbone_parameters", []))
        actual_frozen = set(structurally_frozen_backbone)
        if actual_frozen != expected_frozen:
            raise RuntimeError(
                "Full-finetune structural-freeze invariant failed: "
                f"expected frozen={sorted(expected_frozen)}, actual frozen={sorted(actual_frozen)}"
            )
    if head_trainable != head_total:
        raise RuntimeError(
            f"Classification head invariant failed: only {head_trainable}/{head_total} head parameters are trainable"
        )

    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)
    backbone_norms = [module for module in backbone.modules() if isinstance(module, norm_types)]
    backbone_dropouts = [module for module in backbone.modules() if isinstance(module, nn.Dropout)]
    backbone_drop_paths = [
        module for module in backbone.modules() if module.__class__.__name__.lower() in {"droppath", "stochasticdepth"}
    ]
    model.train()
    if frozen:
        backbone.eval()
    if frozen and any(module.training for module in backbone.modules()):
        raise RuntimeError("Frozen-backbone mode invariant failed: at least one backbone module remains in train mode")
    if not frozen and not backbone.training:
        raise RuntimeError("Full-finetune mode invariant failed: backbone is not in train mode")
    if not head.training:
        raise RuntimeError("Classification head must remain in train mode during gradient updates")

    head_config = model_cfg.get("head", {})
    if head_config.get("architecture") == "batchnorm_linear":
        modules = list(head.children())
        dropout_probability = float(head_config.get("dropout", 0.0))
        expected_types = [nn.BatchNorm1d, nn.Linear] if dropout_probability == 0.0 else [
            nn.BatchNorm1d, nn.Dropout, nn.Linear
        ]
        if len(modules) != len(expected_types) or any(
            not isinstance(module, expected_type) for module, expected_type in zip(modules, expected_types)
        ):
            raise RuntimeError(
                "batchnorm_linear head must be BatchNorm1d -> Linear for p=0 or "
                "BatchNorm1d -> Dropout -> Linear for p>0"
            )
        expected_affine = bool(head_config.get("batchnorm_affine", False))
        expected_eps = float(head_config.get("batchnorm_eps", 1.0e-6))
        if modules[0].affine != expected_affine or not math.isclose(
            float(modules[0].eps), expected_eps, rel_tol=0.0, abs_tol=0.0
        ):
            raise RuntimeError(
                "batchnorm_linear head does not match configured batchnorm_affine/batchnorm_eps"
            )
        if dropout_probability == 0.0 and any(isinstance(module, nn.Dropout) for module in head.modules()):
            raise RuntimeError("Baseline classification head unexpectedly contains dropout")
        if dropout_probability > 0.0:
            designated = designated_mc_dropout_modules(model)
            if len(designated) != 1 or not math.isclose(
                float(designated[0][1].p), dropout_probability, rel_tol=0.0, abs_tol=0.0
            ):
                raise RuntimeError("MC classification head does not expose its one configured dropout")

    return {
        "adaptation_mode": model_cfg.get("adaptation_mode"),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "backbone_total_parameters": backbone_total,
        "backbone_trainable_parameters": backbone_trainable,
        "backbone_structurally_frozen_parameters": structurally_frozen_backbone,
        "backbone_expected_trainable_parameters": backbone_total - sum(structurally_frozen_backbone.values()),
        "all_expected_backbone_parameters_trainable": (
            frozen or backbone_trainable == backbone_total - sum(structurally_frozen_backbone.values())
        ),
        "head_total_parameters": head_total,
        "head_trainable_parameters": head_trainable,
        "backbone_requires_grad_all_false": backbone_trainable == 0,
        "training_mode_policy": {
            "wrapper_training": model.training,
            "backbone_training": backbone.training,
            "head_training": head.training,
            "backbone_batchnorm": "eval; running statistics are not updated" if frozen else "train",
            "backbone_layernorm": (
                "eval with frozen affine parameters; LayerNorm has no running statistics" if frozen else "train"
            ),
            "backbone_dropout": "disabled because backbone is held in eval mode" if frozen else "enabled in train mode",
            "head_batchnorm": "train during updates; eval for validation/test",
            "head_dropout": "absent" if not any(isinstance(module, nn.Dropout) for module in head.modules()) else "configured",
        },
        "backbone_module_counts": {
            "normalization_total": len(backbone_norms),
            "batchnorm": sum(isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for module in backbone_norms),
            "layernorm": sum(isinstance(module, nn.LayerNorm) for module in backbone_norms),
            "groupnorm": sum(isinstance(module, nn.GroupNorm) for module in backbone_norms),
            "dropout": len(backbone_dropouts),
            "drop_path_or_stochastic_depth": len(backbone_drop_paths),
        },
        "head": repr(head),
        "pretrained_load": getattr(backbone, "_pretrained_load_audit", None),
    }


def run_classification_experiment(
    config: Dict[str, Any],
    experiment: Dict[str, Any],
    resume_run_dir: Optional[Path] = None,
) -> ExperimentResult:
    run_started = time.time()
    classification_type = classification_type_from_config(config)
    multilabel_threshold = float(config.get("metrics", {}).get("multilabel_threshold", 0.5))
    seed = config["seed"]
    reproducibility_cfg = config.get("reproducibility", {})
    generator = set_global_seed(
        seed,
        deterministic=bool(reproducibility_cfg.get("deterministic", True)),
        warn_only=bool(reproducibility_cfg.get("warn_only", False)),
    )
    run_id, run_config, output_dir, resume_context = prepare_classification_run_directory(
        config,
        experiment,
        resume_run_dir,
    )

    configured_device = config.get("device", "auto")
    if configured_device == "auto":
        configured_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(configured_device)
    dataloaders = make_dataloaders(run_config, generator=generator)
    model = build_model(config, experiment["model"], num_classes=int(config["data"].get("num_classes", 10))).to(device)
    model_audit = audit_model_for_training(model, experiment["model"])
    input_artifacts = collect_input_artifact_provenance(config, experiment["model"])
    if resume_context is None:
        write_json(output_dir / "model_audit.json", model_audit)
        write_json(output_dir / "input_artifacts.json", input_artifacts)
    else:
        if json.loads((output_dir / "model_audit.json").read_text(encoding="utf-8")) != model_audit:
            raise ValueError("Current model audit does not exactly match the interrupted run")
        if json.loads((output_dir / "input_artifacts.json").read_text(encoding="utf-8")) != input_artifacts:
            raise ValueError("Current input artifact hashes do not exactly match the interrupted run")
    print(
        f"[{experiment['name']}] parameters: "
        f"trainable={model_audit['trainable_parameters']:,} / total={model_audit['total_parameters']:,}; "
        f"backbone trainable={model_audit['backbone_trainable_parameters']:,}"
    )
    print(
        f"[{experiment['name']}] train-mode policy: backbone.training="
        f"{model_audit['training_mode_policy']['backbone_training']}, "
        f"head.training={model_audit['training_mode_policy']['head_training']}"
    )

    training_cfg = config["training"]
    optimizer = build_optimizer(model, training_cfg)
    optimizer_group_audit = {
        str(group.get("name", f"group_{index}")): {
            "parameter_tensors": len(group["params"]),
            "trainable_parameters": sum(parameter.numel() for parameter in group["params"]),
            "base_learning_rate": float(group.get("base_lr", group["lr"])),
            "weight_decay": float(group.get("weight_decay", optimizer.defaults.get("weight_decay", 0.0))),
        }
        for index, group in enumerate(optimizer.param_groups)
    }
    if experiment["model"].get("adaptation_mode") == "full_finetune":
        if set(optimizer_group_audit) != {"backbone", "head"}:
            raise RuntimeError(
                f"Full fine-tuning requires exactly backbone/head optimizer groups, got {optimizer_group_audit}"
            )
        if optimizer_group_audit["backbone"]["trainable_parameters"] != model_audit["backbone_trainable_parameters"]:
            raise RuntimeError("Backbone optimizer group does not cover every trainable backbone parameter")
        if optimizer_group_audit["head"]["trainable_parameters"] != model_audit["head_trainable_parameters"]:
            raise RuntimeError("Head optimizer group does not cover every trainable head parameter")
    if resume_context is None:
        write_json(output_dir / "optimizer_group_audit.json", optimizer_group_audit)
    elif json.loads((output_dir / "optimizer_group_audit.json").read_text(encoding="utf-8")) != optimizer_group_audit:
        raise ValueError("Current optimizer groups do not exactly match the interrupted run")
    print(
        f"[{experiment['name']}] trainable groups: "
        f"backbone={model_audit['backbone_trainable_parameters']:,}, "
        f"head={model_audit['head_trainable_parameters']:,}; "
        f"optimizer={optimizer_group_audit}"
    )

    checkpoint_path = output_dir / "best.pt"
    last_checkpoint_path = output_dir / "last.pt"

    checkpoint_cfg = training_cfg["checkpoint"]
    selection_metric = str(checkpoint_cfg.get("metric", "nll"))
    selection_mode = str(checkpoint_cfg.get("mode", "min"))
    best_metric = math.inf if selection_mode == "min" else -math.inf
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    batch_history: List[Dict[str, Any]] = []
    epochs = int(training_cfg.get("epochs", 5))
    n_bins = int(config.get("metrics", {}).get("n_bins", 15))
    limit_train_batches = training_cfg.get("limit_train_batches")
    limit_val_batches = training_cfg.get("limit_val_batches")
    start_epoch = 1
    visualization_cfg = config.get("visualization", {})
    visualization_enabled = bool(visualization_cfg.get("enabled", True))
    visualization_every = max(1, int(visualization_cfg.get("update_every_epochs", 1)))
    track_gradient_norm = bool(visualization_cfg.get("track_gradient_norm", True))
    early_cfg = training_cfg.get("early_stopping", {})
    warmup_cfg = training_cfg.get("warmup", {"enabled": False, "epochs": 0})
    early_enabled = bool(early_cfg.get("enabled", False))
    patience = int(early_cfg.get("patience", 0))
    min_delta = float(early_cfg.get("min_delta", 0.0))
    epochs_without_improvement = 0
    first_gradient_audit: Optional[Dict[str, Any]] = None
    resume_event_dir: Optional[Path] = None
    resume_training_started_epoch: Optional[int] = None
    generator_restore_method: Optional[str] = None

    if last_checkpoint_path.exists():
        checkpoint = _load_torch_checkpoint(last_checkpoint_path, device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        best_metric = float(checkpoint.get("best_metric", best_metric))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        history = list(checkpoint.get("history", history))
        batch_history = list(checkpoint.get("batch_history", batch_history))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", epochs_without_improvement)
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if resume_context is not None:
            first_gradient_audit = json.loads(
                (output_dir / "gradient_audit.json").read_text(encoding="utf-8")
            )
            saved_loader_states = checkpoint.get("dataloader_generator_states")
            if saved_loader_states:
                restore_dataloader_generator_states(dataloaders, saved_loader_states)
                generator_restore_method = "checkpoint_state"
            else:
                augmentations = config.get("data", {}).get("augmentations", {})
                train_augmentation = augmentations.get("train", "none") if isinstance(augmentations, dict) else None
                active_stochastic_modules = stochastic_training_modules(model)
                if (
                    experiment["model"].get("adaptation_mode") != "frozen"
                    or train_augmentation not in {None, "none"}
                    or active_stochastic_modules
                ):
                    raise ValueError(
                        "Legacy checkpoint lacks RNG state and is not safe to reconstruct: "
                        f"augmentation={train_augmentation!r}, stochastic_modules={active_stochastic_modules}"
                    )
                reconstruct_classification_loader_states(dataloaders, int(checkpoint["epoch"]))
                generator_restore_method = "legacy_exact_sampler_fast_forward"
            if checkpoint.get("rng_state") is not None:
                restore_rng_state(checkpoint["rng_state"])
                rng_restore_method = "checkpoint_state"
            else:
                rng_restore_method = "legacy_not_required_frozen_no_stochastic_training"

            event_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            resume_event_dir = output_dir / "resume_events" / event_timestamp
            resume_event_dir.mkdir(parents=True, exist_ok=False)
            archived_checkpoint = resume_event_dir / f"source_last_epoch{checkpoint['epoch']}.pt"
            shutil.copy2(last_checkpoint_path, archived_checkpoint)
            current_snapshot = copy.deepcopy(resume_context["current_snapshot"])
            current_snapshot["created_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(resume_event_dir / "code_snapshot.json", current_snapshot)
            current_snapshot_reference = {
                "code_sha256": current_snapshot["code_sha256"],
                "file_count": current_snapshot["file_count"],
                "manifest_path": str((resume_event_dir / "code_snapshot.json").resolve()),
            }
            write_json(
                resume_event_dir / "environment.json",
                collect_environment_metadata(code_snapshot=current_snapshot_reference),
            )
            original_files = {
                item["path"]: item["sha256"] for item in resume_context["original_snapshot"]["files"]
            }
            current_files = {item["path"]: item["sha256"] for item in current_snapshot["files"]}
            changed_paths = sorted(
                set(resume_context["code_changes"]["changed"])
                | set(resume_context["code_changes"]["added"])
                | set(resume_context["code_changes"]["removed"])
            )
            train_generator = dataloaders["train"].generator
            generator_state_sha256 = None
            if train_generator is not None:
                generator_state_sha256 = hashlib.sha256(
                    train_generator.get_state().cpu().numpy().tobytes()
                ).hexdigest()
            resume_request = {
                "schema_version": 1,
                "status": "started",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "seed": seed,
                "experiment": experiment["name"],
                "dataset": config["data"].get("name"),
                "model": experiment["model"].get("name"),
                "adaptation": experiment["model"].get("adaptation_mode"),
                "source_checkpoint": {
                    "path": str(last_checkpoint_path.resolve()),
                    "archived_path": str(archived_checkpoint.resolve()),
                    "sha256": checkpoint_sha256(archived_checkpoint),
                    "size_bytes": archived_checkpoint.stat().st_size,
                    "epoch": int(checkpoint["epoch"]),
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                "next_epoch": start_epoch,
                "original_code_sha256": resume_context["original_code_sha256"],
                "resume_code_sha256": current_snapshot["code_sha256"],
                "code_changes": [
                    {
                        "path": path,
                        "before_sha256": original_files.get(path),
                        "after_sha256": current_files.get(path),
                    }
                    for path in changed_paths
                ],
                "protected_provenance_sha256": {
                    name: checkpoint_sha256(output_dir / name)
                    for name in (
                        "resolved_config.yaml",
                        "code_snapshot.json",
                        "environment.json",
                        "input_artifacts.json",
                        "model_audit.json",
                        "optimizer_group_audit.json",
                        "gradient_audit.json",
                    )
                },
                "config_verified_exact": True,
                "input_artifacts_verified_exact": True,
                "model_audit_verified_exact": True,
                "optimizer_groups_verified_exact": True,
                "gradient_audit_preserved": True,
                "generator_restore_method": generator_restore_method,
                "train_generator_state_sha256": generator_state_sha256,
                "train_dataset_length": len(dataloaders["train"].dataset),
                "rng_restore_method": rng_restore_method,
                "pytorch_version": torch.__version__,
                "partial_epoch_policy": "discard_uncheckpointed_work_and_restart_next_epoch",
                "test_access": False,
                "command": sys.argv,
            }
            with (resume_event_dir / "resume_request.json").open("x", encoding="utf-8") as handle:
                json.dump(resume_request, handle, indent=2, ensure_ascii=False)
            resume_training_started_epoch = start_epoch
        if start_epoch > epochs:
            print(
                f"[{experiment['name']}] existing last checkpoint already reached "
                f"epoch {start_epoch - 1}/{epochs}; continuing to final evaluation."
            )
        else:
            print(f"[{experiment['name']}] resumed from {last_checkpoint_path} at epoch {start_epoch}/{epochs}")

    training_started = time.perf_counter()
    training_range = range(start_epoch, epochs + 1)
    if early_enabled and epochs_without_improvement >= patience:
        print(
            f"[{experiment['name']}] restored patience already meets stopping rule; "
            "skipping further training."
        )
        training_range = range(0)
    for epoch in training_range:
        start = time.time()
        learning_rates = set_epoch_learning_rates(optimizer, epoch, warmup_cfg)
        train_metrics = train_one_epoch(
            model,
            dataloaders["train"],
            optimizer,
            device,
            limit_batches=limit_train_batches,
            epoch=epoch,
            global_step=len(batch_history),
            track_gradient_norm=track_gradient_norm,
            audit_gradients=first_gradient_audit is None,
            classification_type=classification_type,
            multilabel_threshold=multilabel_threshold,
        )
        if first_gradient_audit is None:
            first_gradient_audit = dict(train_metrics["gradient_audit"])
            first_gradient_audit["epoch"] = epoch
            write_json(output_dir / "gradient_audit.json", first_gradient_audit)
        epoch_batch_history = train_metrics.pop("batch_history")
        batch_history.extend(epoch_batch_history)
        val = collect_logits(model, dataloaders["val"], device, limit_batches=limit_val_batches)
        val_metrics = add_uncertainty_summaries(
            compute_task_classification_metrics(
                val["logits"], val["labels"], classification_type, n_bins, multilabel_threshold
            ),
            val["logits"],
            classification_type,
        )
        history.append(
            {
                "epoch": epoch,
                "seconds": round(time.time() - start, 3),
                "train": train_metrics,
                "val": val_metrics,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "backbone_learning_rate": learning_rates.get("backbone", math.nan),
                "head_learning_rate": learning_rates.get("head", math.nan),
                "warmup_factor": learning_rates["warmup_factor"],
                "train_val_accuracy_gap": train_metrics["accuracy"] - val_metrics["accuracy"],
            }
        )
        write_json(output_dir / "training_history.json", history)
        print(
            f"[{experiment['name']}] epoch {epoch:03d}/{epochs:03d} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_nll={val_metrics['nll']:.4f} "
            f"val_ece={val_metrics['ece']:.4f} "
            f"val_brier={val_metrics['brier']:.4f} "
            f"confidence={val_metrics['mean_confidence']:.4f} "
            f"entropy={val_metrics['predictive_entropy']:.4f} "
            f"backbone_grad={train_metrics['backbone_gradient_norm']:.4f} "
            f"head_grad={train_metrics['head_gradient_norm']:.4f} "
            f"backbone_lr={learning_rates.get('backbone', math.nan):.2e} "
            f"head_lr={learning_rates.get('head', math.nan):.2e}"
        )
        if visualization_enabled and (epoch % visualization_every == 0 or epoch == epochs):
            dashboard_path = save_training_dashboard(history, batch_history, output_dir)
            print(f"[{experiment['name']}] refreshed training dashboard: {dashboard_path}")
        current_metric = float(val_metrics[selection_metric])
        if metric_improved(current_metric, best_metric, selection_mode, min_delta=min_delta):
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "experiment": experiment,
                    "config": config,
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                    "run_id": run_id,
                    "selection_metric": selection_metric,
                    "selection_value": current_metric,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "experiment": experiment,
                "config": config,
                "epoch": epoch,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "batch_history": batch_history,
                "run_id": run_id,
                "selection_metric": selection_metric,
                "rng_state": capture_rng_state(),
                "dataloader_generator_states": capture_dataloader_generator_states(dataloaders),
            },
            last_checkpoint_path,
        )
        if early_enabled and epochs_without_improvement >= patience:
            print(f"[{experiment['name']}] early stopping at epoch {epoch}; best epoch={best_epoch}")
            break
    training_segment_seconds = time.perf_counter() - training_started
    prior_training_seconds = 0.0
    if resume_training_started_epoch is not None:
        prior_training_seconds = sum(
            float(item.get("seconds", 0.0))
            for item in history
            if int(item["epoch"]) < resume_training_started_epoch
        )
    training_seconds = prior_training_seconds + training_segment_seconds

    best_history = next(item for item in history if item["epoch"] == best_epoch)
    last_history = history[-1]
    best_gap = float(best_history["train"]["accuracy"] - best_history["val"]["accuracy"])
    last_gap = float(last_history["train"]["accuracy"] - last_history["val"]["accuracy"])
    post_best_history = [item for item in history if item["epoch"] >= best_epoch]
    max_gap_item = max(
        post_best_history,
        key=lambda item: float(item["train"]["accuracy"] - item["val"]["accuracy"]),
    )
    max_post_best_gap = float(max_gap_item["train"]["accuracy"] - max_gap_item["val"]["accuracy"])
    stability = {
        "nan_or_inf_detected": False,
        "best_epoch": best_epoch,
        "last_epoch": int(last_history["epoch"]),
        "best_epoch_train_val_accuracy_gap": best_gap,
        "last_epoch_train_val_accuracy_gap": last_gap,
        "max_post_best_train_val_accuracy_gap": max_post_best_gap,
        "max_post_best_gap_epoch": int(max_gap_item["epoch"]),
        "best_val_nll": float(best_history["val"]["nll"]),
        "last_val_nll": float(last_history["val"]["nll"]),
        "overfitting_heuristic": {
            "detected": bool(
                max_post_best_gap > 0.02
                and any(item["val"]["nll"] > best_history["val"]["nll"] for item in post_best_history)
            ),
            "definition": (
                "at least one epoch at or after the best checkpoint has validation NLL above the best value, "
                "and the maximum post-best train-minus-validation accuracy gap exceeds 0.02"
            ),
        },
        "gradient_audit": first_gradient_audit,
        "backbone_gradient_norm_range": [
            min(float(item["train"]["backbone_gradient_norm"]) for item in history),
            max(float(item["train"]["backbone_gradient_norm"]) for item in history),
        ],
        "head_gradient_norm_range": [
            min(float(item["train"]["head_gradient_norm"]) for item in history),
            max(float(item["train"]["head_gradient_norm"]) for item in history),
        ],
    }
    write_json(output_dir / "training_stability.json", stability)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    uq_results = evaluate_uq_methods(
        model,
        dataloaders["val"],
        device,
        config.get("uq_methods", [{"name": "none"}]),
        n_bins=n_bins,
        limit_batches=limit_val_batches,
        classification_type=classification_type,
        multilabel_threshold=multilabel_threshold,
    )

    raw_eval = collect_logits(model, dataloaders["val"], device, limit_batches=limit_val_batches)
    if classification_type == "multilabel":
        positive_probabilities = raw_eval["logits"].sigmoid()
        reliability_probabilities = torch.stack(
            (1.0 - positive_probabilities, positive_probabilities), dim=-1
        ).reshape(-1, 2)
        reliability_labels = raw_eval["labels"].long().reshape(-1)
    else:
        reliability_probabilities = raw_eval["logits"].softmax(dim=1)
        reliability_labels = raw_eval["labels"]
    plot_reliability_diagram(
        reliability_probabilities,
        reliability_labels,
        output_dir / "reliability.png",
        n_bins=n_bins,
    )

    prediction_exports: Dict[str, str] = {}
    evaluation_metrics: Dict[str, Any] = {}
    inference_seconds: Dict[str, float] = {}
    export_cfg = config.get("prediction_export", {})
    if bool(export_cfg.get("enabled", False)):
        class_names = config["data"].get("class_names")
        if not class_names:
            raise ValueError("prediction_export requires data.class_names")
        embedding_extractor = classification_embedding_extractor(model)
        checkpoint_digest = checkpoint_sha256(checkpoint_path)
        for split in export_cfg.get("splits", ["val"]):
            if split not in dataloaders:
                raise ValueError(f"Prediction export requested unavailable split: {split}")
            export_limit = limit_val_batches if bool(config.get("dry_run", False)) else None
            inference_started = time.perf_counter()
            collected = collect_deterministic_predictions(
                model,
                dataloaders[split],
                device,
                limit_batches=export_limit,
                embedding_extractor=embedding_extractor,
            )
            split_inference_seconds = time.perf_counter() - inference_started
            expected_export_count = (
                len(collected["sample_ids"]) if export_limit is not None else len(dataloaders[split].dataset)
            )
            export_dir = output_dir / "predictions" / split / "deterministic"
            export_predictions(
                export_dir,
                sample_ids=collected["sample_ids"],
                true_labels=collected["labels"],
                logits=collected["logits"],
                class_names=class_names,
                model_name=model_display_name(experiment["model"]),
                dataset=config["data"].get("name", "eurosat"),
                adaptation_mode=experiment["model"].get("adaptation_mode", "unknown"),
                seed=seed,
                checkpoint=str(checkpoint_path),
                split=split,
                uq_method="deterministic",
                corruption_type=str(export_cfg.get("corruption_type", "none")),
                corruption_severity=int(export_cfg.get("corruption_severity", 0)),
                expected_count=expected_export_count,
                embeddings=collected["embeddings"],
                classification_type=classification_type,
                multilabel_threshold=multilabel_threshold,
                source={
                    "run_id": run_id,
                    "checkpoint_sha256": checkpoint_digest,
                    "split_manifest": str(config["data"].get("split_manifest", "legacy_split_files")),
                    "split_total_count": len(dataloaders[split].dataset),
                    "limit_batches": export_limit,
                    "inference_seconds": split_inference_seconds,
                    "samples_per_second": len(collected["sample_ids"]) / max(split_inference_seconds, 1.0e-12),
                },
            )
            evaluation_metrics[split] = save_evaluation_artifacts(export_dir, n_bins=n_bins)
            inference_seconds[split] = split_inference_seconds
            prediction_exports[split] = str(export_dir)

    runtime_seconds = time.time() - run_started
    metrics = {
        "history": history,
        "validation": uq_results,
        "evaluation": evaluation_metrics,
        "training_seconds": training_seconds,
        "training_segments": {
            "prior_epoch_seconds": prior_training_seconds,
            "current_process_seconds": training_segment_seconds,
            "resumed_from_epoch": resume_training_started_epoch,
        },
        "inference_seconds": inference_seconds,
        "training_stability": stability,
        "runtime_seconds": runtime_seconds,
        "prediction_exports": prediction_exports,
    }
    result = ExperimentResult(
        name=experiment["name"],
        task=str(config.get("task", "classification")),
        dataset=config["data"].get("name", "eurosat"),
        model=model_display_name(experiment["model"]),
        checkpoint_path=str(checkpoint_path),
        metrics=metrics,
    )
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result.__dict__, handle, indent=2)
    write_json(output_dir / "validation_metrics.json", uq_results)
    write_json(
        output_dir / "run_summary.json",
        {
            "run_id": run_id,
            "seed": seed,
            "status": "completed",
            "dry_run": bool(config.get("dry_run", False)),
            "runtime_seconds": runtime_seconds,
            "training_seconds": training_seconds,
            "training_segments": {
                "prior_epoch_seconds": prior_training_seconds,
                "current_process_seconds": training_segment_seconds,
                "resumed_from_epoch": resume_training_started_epoch,
            },
            "inference_seconds": inference_seconds,
            "best_epoch": best_epoch,
            "checkpoint_selection": {"split": "val", "metric": selection_metric, "mode": selection_mode},
            "best_checkpoint": str(checkpoint_path),
            "last_checkpoint": str(last_checkpoint_path),
            "prediction_exports": prediction_exports,
            "evaluation_metrics": evaluation_metrics,
            "model_audit": model_audit,
            "input_artifacts": input_artifacts,
            "training_stability": stability,
            "resume_event": str(resume_event_dir) if resume_event_dir is not None else None,
        },
    )

    best_validation = uq_results["none"]
    append_results_csv(
        Path(config.get("results_csv", "./results/experiments/results.csv")),
        {
            "experiment": experiment["name"],
            "run_id": run_id,
            "seed": seed,
            "dataset": config["data"].get("name", "eurosat"),
            "input_bands": config["data"].get("input_bands", "all"),
            "model": experiment["model"].get("name", "dofa"),
            "epoch": best_epoch,
            "val_accuracy": best_validation["accuracy"],
            "val_nll": best_validation["nll"],
            "val_ece": best_validation["ece"],
            "val_brier": best_validation["brier"],
            "best_checkpoint": str(checkpoint_path),
            "runtime_seconds": runtime_seconds,
            "training_seconds": training_seconds,
        },
    )
    if resume_event_dir is not None:
        with (resume_event_dir / "completion.json").open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "status": "completed",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "run_id": run_id,
                    "resumed_from_epoch": resume_training_started_epoch,
                    "last_epoch": int(last_history["epoch"]),
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "best_checkpoint_sha256": checkpoint_sha256(checkpoint_path),
                    "last_checkpoint_sha256": checkpoint_sha256(last_checkpoint_path),
                    "training_segment_seconds": training_segment_seconds,
                    "final_test_export": prediction_exports.get("test"),
                    "test_access_during_promotion": False,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
    return result


def run_segmentation_experiment(
    config: Dict[str, Any],
    experiment: Dict[str, Any],
    resume_run_dir: Optional[Path] = None,
) -> ExperimentResult:
    """Run an auditable, exactly resumable deterministic segmentation experiment."""
    run_started = time.time()
    seed = int(config["seed"])
    reproducibility_cfg = config.get("reproducibility", {})
    generator = set_global_seed(
        seed,
        deterministic=bool(reproducibility_cfg.get("deterministic", True)),
        warn_only=bool(reproducibility_cfg.get("warn_only", False)),
    )
    output_root = Path(config.get("output_dir", "./results/final_thesis/segmentation"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    resume_checkpoint: Optional[Dict[str, Any]] = None
    if resume_run_dir is None:
        run_id = make_run_id(experiment["name"], seed)
        output_dir = output_root / "runs" / run_id
        run_config = copy.deepcopy(config)
        run_config["run_id"] = run_id
        run_config["active_experiment"] = experiment["name"]
        write_resolved_config(run_config, output_dir)
        code_snapshot = write_code_snapshot_manifest(output_dir)
        write_json(output_dir / "environment.json", collect_environment_metadata(code_snapshot=code_snapshot))
    else:
        output_dir = resume_run_dir.resolve()
        if output_dir.parent != (output_root / "runs").resolve() or not output_dir.is_dir():
            raise ValueError(f"Invalid segmentation resume directory: {output_dir}")
        if (output_dir / "run_summary.json").exists():
            raise ValueError("Refusing to resume an already completed segmentation run")
        original_config = yaml.safe_load((output_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        run_id = str(original_config.get("run_id", ""))
        expected = copy.deepcopy(config)
        expected["run_id"] = run_id
        expected["active_experiment"] = experiment["name"]
        if output_dir.name != run_id or original_config != expected:
            raise ValueError("Segmentation resume configuration/run identity mismatch")
        original_snapshot = json.loads((output_dir / "code_snapshot.json").read_text(encoding="utf-8"))
        current_snapshot = collect_code_snapshot()
        code_changes = _snapshot_changes(original_snapshot, current_snapshot)
        allowed_resume_changes = {
            ".devcontainer/devcontainer.json",
            "scripts/run_experiments.py",
            "scripts/audit_c2_segmentation_runs.py",
        }
        changed_paths = set(code_changes["changed"]) | set(code_changes["added"]) | set(code_changes["removed"])
        unexpected_changes = sorted(changed_paths - allowed_resume_changes)
        if unexpected_changes:
            raise ValueError(
                "Scientifically relevant code/config changed since segmentation run started; "
                f"resume refused: {unexpected_changes}"
            )
        resume_checkpoint = _load_torch_checkpoint(output_dir / "last.pt", "cpu")
        required_resume_keys = {
            "model", "optimizer", "history", "rng_state", "dataloader_generator_states",
            "epochs_without_improvement", "training_seconds", "gradient_audit",
        }
        if required_resume_keys - set(resume_checkpoint):
            raise ValueError(f"Segmentation checkpoint is not exactly resumable: {sorted(required_resume_keys - set(resume_checkpoint))}")
        if resume_checkpoint.get("run_id") != run_id or resume_checkpoint.get("experiment") != experiment:
            raise ValueError("Segmentation resume checkpoint identity mismatch")
        if resume_checkpoint.get("config") != config:
            raise ValueError("Segmentation resume checkpoint configuration mismatch")
        event_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        resume_event_dir = output_dir / "resume_events" / event_timestamp
        resume_event_dir.mkdir(parents=True, exist_ok=False)
        archived_checkpoint = resume_event_dir / "source_last.pt"
        shutil.copy2(output_dir / "last.pt", archived_checkpoint)
        current_snapshot_record = copy.deepcopy(current_snapshot)
        current_snapshot_record["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(resume_event_dir / "code_snapshot.json", current_snapshot_record)
        write_json(
            resume_event_dir / "event.json",
            {
                "schema_version": 1,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "checkpoint_epoch": int(resume_checkpoint["epoch"]),
                "checkpoint_path": str((output_dir / "last.pt").resolve()),
                "archived_checkpoint_path": str(archived_checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha256(archived_checkpoint),
                "original_code_sha256": original_snapshot["code_sha256"],
                "resume_code_sha256": current_snapshot["code_sha256"],
                "code_changes": code_changes,
                "allowed_change_paths": sorted(allowed_resume_changes),
                "scientific_training_path_changed": False,
                "test_access": False,
            },
        )

    device_name = config.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    loaders = make_segmentation_dataloaders(config, generator=generator)
    model = build_segmentation_model(config, experiment["model"]).to(device)
    model_audit = audit_segmentation_model(model, experiment["model"])
    input_artifacts = collect_input_artifact_provenance(config, experiment["model"])

    training_cfg = config["training"]
    optimizer = build_optimizer(model, training_cfg)
    optimizer_group_audit = {
        str(group.get("name", f"group_{index}")): {
            "parameter_tensors": len(group["params"]),
            "trainable_parameters": sum(parameter.numel() for parameter in group["params"]),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        }
        for index, group in enumerate(optimizer.param_groups)
    }
    expected_groups = {"head"} if experiment["model"]["adaptation_mode"] == "frozen" else {"backbone", "head"}
    if set(optimizer_group_audit) != expected_groups:
        raise RuntimeError(f"Segmentation optimizer groups disagree with adaptation mode: {optimizer_group_audit}")
    if resume_checkpoint is None:
        write_json(output_dir / "model_audit.json", model_audit)
        write_json(output_dir / "optimizer_group_audit.json", optimizer_group_audit)
        write_json(output_dir / "input_artifacts.json", input_artifacts)
    else:
        for name, current in (
            ("model_audit.json", model_audit),
            ("optimizer_group_audit.json", optimizer_group_audit),
            ("input_artifacts.json", input_artifacts),
        ):
            if json.loads((output_dir / name).read_text(encoding="utf-8")) != current:
                raise ValueError(f"Segmentation resume invariant changed: {name}")

    data_cfg = config["data"]
    class_names = list(data_cfg["class_names"])
    ignore_index = data_cfg.get("ignore_index")
    metrics_cfg = config.get("metrics", {})
    n_bins = int(metrics_cfg.get("n_bins", 15))
    boundary_radius = int(metrics_cfg.get("boundary_radius", 1))
    foreground_index = int(data_cfg["foreground_class_index"]) if "foreground_class_index" in data_cfg else None
    limit_train_batches = training_cfg.get("limit_train_batches")
    limit_val_batches = training_cfg.get("limit_val_batches")
    checkpoint_cfg = training_cfg["checkpoint"]
    selection_metric = str(checkpoint_cfg.get("metric", "miou"))
    selection_mode = str(checkpoint_cfg.get("mode", "max"))
    early_cfg = training_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    patience = int(early_cfg.get("patience", 0))
    min_delta = float(early_cfg.get("min_delta", 0.0))
    if not early_enabled or patience <= 0:
        raise ValueError("Final segmentation runs require enabled early stopping with positive patience")
    checkpoint_path = output_dir / "best.pt"
    last_checkpoint_path = output_dir / "last.pt"

    if resume_checkpoint is None:
        start_epoch = 1
        history: List[Dict[str, Any]] = []
        best_value = -math.inf if selection_mode == "max" else math.inf
        best_epoch = 0
        epochs_without_improvement = 0
        prior_training_seconds = 0.0
        gradient_audit: Optional[Dict[str, Any]] = None
    else:
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        restore_rng_state(resume_checkpoint["rng_state"])
        restore_dataloader_generator_states(loaders, resume_checkpoint["dataloader_generator_states"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        history = list(resume_checkpoint["history"])
        if [int(item["epoch"]) for item in history] != list(range(1, start_epoch)):
            raise ValueError("Segmentation resume history is not a contiguous epoch prefix")
        best_value = float(resume_checkpoint["best_value"])
        best_epoch = int(resume_checkpoint["best_epoch"])
        epochs_without_improvement = int(resume_checkpoint["epochs_without_improvement"])
        prior_training_seconds = float(resume_checkpoint["training_seconds"])
        gradient_audit = dict(resume_checkpoint["gradient_audit"])

    segment_started = time.perf_counter()
    stopped_early = epochs_without_improvement >= patience
    for epoch in range(start_epoch, int(training_cfg.get("epochs", 1)) + 1):
        if stopped_early:
            break
        epoch_started = time.perf_counter()
        learning_rates = set_epoch_learning_rates(
            optimizer, epoch, training_cfg.get("warmup", {"enabled": False, "epochs": 0})
        )
        train_metrics = train_segmentation_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            ignore_index,
            limit_batches=limit_train_batches,
            audit_gradients=gradient_audit is None,
        )
        if gradient_audit is None:
            gradient_audit = dict(train_metrics.pop("gradient_audit"))
            gradient_audit["epoch"] = epoch
            groups = gradient_audit["groups"]
            if not groups["head"]["all_gradients_finite"] or not groups["head"]["any_nonzero_gradient"]:
                raise RuntimeError("Segmentation decoder gradient audit failed")
            if experiment["model"]["adaptation_mode"] == "frozen":
                if groups["backbone"]["gradient_parameter_tensors"] != 0:
                    raise RuntimeError("Frozen segmentation backbone unexpectedly received gradients")
            elif not groups["backbone"]["all_gradients_finite"] or not groups["backbone"]["any_nonzero_gradient"]:
                raise RuntimeError("Full-finetune segmentation backbone gradient audit failed")
            write_json(output_dir / "gradient_audit.json", gradient_audit)
        validation_metrics = evaluate_segmentation(
            model,
            loaders["val"],
            device,
            class_names,
            ignore_index,
            n_bins=n_bins,
            foreground_class_index=foreground_index,
            boundary_radius=boundary_radius,
            limit_batches=limit_val_batches,
        )
        current = float(validation_metrics[selection_metric])
        improved = metric_improved(current, best_value, selection_mode, min_delta)
        if improved:
            best_value = current
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_torch_save({
                "model": model.state_dict(),
                "experiment": experiment,
                "config": config,
                "epoch": epoch,
                "run_id": run_id,
                "selection_metric": selection_metric,
                "selection_value": current,
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": validation_metrics,
            "learning_rates": learning_rates,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "epoch_seconds": time.perf_counter() - epoch_started,
        })
        write_json(output_dir / "training_history.json", history)
        total_training_seconds = prior_training_seconds + (time.perf_counter() - segment_started)
        atomic_torch_save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "experiment": experiment,
            "config": config,
            "epoch": epoch,
            "run_id": run_id,
            "history": history,
            "best_epoch": best_epoch,
            "best_value": best_value,
            "epochs_without_improvement": epochs_without_improvement,
            "training_seconds": total_training_seconds,
            "gradient_audit": gradient_audit,
            "rng_state": capture_rng_state(),
            "dataloader_generator_states": capture_dataloader_generator_states(loaders),
        }, last_checkpoint_path)
        print(
            f"[{experiment['name']} seed={seed}] epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"val_miou={validation_metrics['miou']:.6f} best_epoch={best_epoch} "
            f"patience={epochs_without_improvement}/{patience}",
            flush=True,
        )
        stopped_early = epochs_without_improvement >= patience

    training_seconds = prior_training_seconds + (time.perf_counter() - segment_started)
    if not checkpoint_path.is_file() or not last_checkpoint_path.is_file() or best_epoch <= 0:
        raise RuntimeError("Segmentation training produced no valid selected checkpoint")
    checkpoint = _load_torch_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"], strict=True)

    evaluation: Dict[str, Any] = {
        "val": evaluate_segmentation(
            model, loaders["val"], device, class_names, ignore_index, n_bins=n_bins,
            foreground_class_index=foreground_index, boundary_radius=boundary_radius,
            limit_batches=limit_val_batches if bool(config.get("dry_run", False)) else None,
        )
    }
    export_cfg = config.get("prediction_export", {})
    test_evaluation_enabled = bool(config.get("evaluation", {}).get("test_enabled", True))
    prediction_exports: Dict[str, str] = {}
    inference_seconds: Dict[str, float] = {}
    if test_evaluation_enabled and bool(export_cfg.get("enabled", False)):
        test_export_dir = output_dir / "predictions" / "test" / "deterministic"
        if test_export_dir.exists():
            validation = json.loads((test_export_dir / "validation_report.json").read_text(encoding="utf-8"))
            if not validation.get("valid"):
                raise ValueError("Existing segmentation test export is invalid")
            evaluation["test"] = json.loads((output_dir / "test_metrics.json").read_text(encoding="utf-8"))
        else:
            inference_started = time.perf_counter()
            collected = collect_segmentation_predictions(
                model, loaders["test"], device, class_names, ignore_index, n_bins=n_bins,
                foreground_class_index=foreground_index, boundary_radius=boundary_radius,
                limit_batches=limit_val_batches if bool(config.get("dry_run", False)) else None,
            )
            inference_seconds["test"] = time.perf_counter() - inference_started
            evaluation["test"] = collected["metrics"]
            write_json(output_dir / "test_metrics.json", evaluation["test"])
            temporary_export = test_export_dir.with_name(f".deterministic.tmp-{os.getpid()}")
            export_segmentation_predictions(
                temporary_export,
                sample_ids=collected["sample_ids"], masks=collected["masks"], logits=collected["logits"],
                class_names=class_names, ignore_index=ignore_index,
                model_name=model_display_name(experiment["model"]), dataset=data_cfg["name"],
                adaptation_mode=experiment["model"]["adaptation_mode"], split="test",
                checkpoint=str(checkpoint_path), representations=collected["representations"],
                per_image_results=collected["per_image_results"],
                source={
                    "run_id": run_id, "seed": seed,
                    "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
                    "split_total_count": len(loaders["test"].dataset),
                    "representation_dimension": int(collected["representations"].shape[1]),
                },
            )
            test_export_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_export, test_export_dir)
        prediction_exports["test"] = str(test_export_dir)
    elif test_evaluation_enabled:
        evaluation["test"] = evaluate_segmentation(
            model, loaders["test"], device, class_names, ignore_index, n_bins=n_bins,
            foreground_class_index=foreground_index, boundary_radius=boundary_radius,
            limit_batches=limit_val_batches if bool(config.get("dry_run", False)) else None,
        )

    write_json(output_dir / "segmentation_metrics.json", evaluation)
    runtime_seconds = time.time() - run_started
    result = ExperimentResult(
        name=experiment["name"], task="segmentation", dataset=data_cfg["name"],
        model=model_display_name(experiment["model"]), checkpoint_path=str(checkpoint_path),
        metrics={
            "history": history, "evaluation": evaluation, "training_seconds": training_seconds,
            "runtime_seconds": runtime_seconds, "inference_seconds": inference_seconds,
            "prediction_exports": prediction_exports,
        },
    )
    write_json(output_dir / "results.json", result.__dict__)
    write_json(output_dir / "run_summary.json", {
        "run_id": run_id, "seed": seed, "status": "completed", "dry_run": bool(config.get("dry_run", False)),
        "best_epoch": best_epoch, "best_value": best_value,
        "last_epoch": int(history[-1]["epoch"]), "stopped_early": stopped_early,
        "checkpoint_selection": {"split": "val", "metric": selection_metric, "mode": selection_mode},
        "best_checkpoint": str(checkpoint_path), "best_checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path), "last_checkpoint_sha256": checkpoint_sha256(last_checkpoint_path),
        "training_seconds": training_seconds, "runtime_seconds": runtime_seconds,
        "prediction_exports": prediction_exports, "evaluation_metrics": evaluation,
        "input_artifacts": input_artifacts, "gradient_audit": gradient_audit,
        "resume_events": [str(path) for path in sorted((output_dir / "resume_events").glob("*/event.json"))]
        if (output_dir / "resume_events").is_dir() else [],
        "test_access_during_training_or_selection": False,
        "test_evaluation_performed": test_evaluation_enabled,
    })
    result_row = {
        "experiment": experiment["name"], "run_id": run_id, "seed": seed,
        "dataset": data_cfg["name"], "model": experiment["model"]["name"],
        "adaptation": experiment["model"]["adaptation_mode"], "best_epoch": best_epoch,
        "val_miou": evaluation["val"]["miou"],
        "best_checkpoint": str(checkpoint_path), "training_seconds": training_seconds,
        "test_evaluation_performed": test_evaluation_enabled,
    }
    if test_evaluation_enabled:
        test_metrics = evaluation["test"]
        result_row.update({
            "test_miou": test_metrics["miou"],
            "test_pixel_accuracy": test_metrics["pixel_accuracy"],
            "test_nll": test_metrics["nll"],
            "test_brier": test_metrics["brier"],
            f"test_ece_{n_bins}": test_metrics[f"ece_{n_bins}"],
        })
    append_results_csv(Path(config["results_csv"]), result_row)
    return result


def run_experiment(
    config: Dict[str, Any],
    experiment: Dict[str, Any],
    resume_run_dir: Optional[Path] = None,
) -> ExperimentResult:
    task = experiment.get("task", config.get("task", "classification"))
    dataset = config.get("data", {}).get("name")
    if task in {"classification", "classification_multilabel"} and dataset in {
        "eurosat",
        "treesatai",
        "so2sat",
    }:
        return run_classification_experiment(config, experiment, resume_run_dir=resume_run_dir)
    if task == "depth_estimation" and dataset == "rs3dbench":
        return run_rs3dbench_experiment(config, experiment)
    if task == "segmentation" and dataset in {"cloudsen12", "spacenet7"}:
        return run_segmentation_experiment(config, experiment, resume_run_dir=resume_run_dir)
    if resume_run_dir is not None:
        raise ValueError("In-place resume is unsupported for this task")
    raise ValueError(f"Unsupported task/dataset combination: task={task}, dataset={dataset}")


def select_experiments(config: Dict[str, Any], names: Optional[List[str]]) -> List[Dict[str, Any]]:
    experiments = config.get("experiments", [])
    if names is None:
        return experiments
    selected = [exp for exp in experiments if exp["name"] in names]
    missing = sorted(set(names) - {exp["name"] for exp in selected})
    if missing:
        raise ValueError(f"Unknown experiment(s): {', '.join(missing)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configurable UQ experiments.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--experiment", action="append", help="Run only this experiment name. Can be repeated.")
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Override configured seeds. Repeat to run multiple explicitly selected seeds.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run one epoch with very few batches and write all artifacts.")
    parser.add_argument("--validate-only", action="store_true", help="Validate configuration without loading data or a model.")
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        help="Resume exactly one interrupted classification or segmentation run from its committed last.pt.",
    )
    args = parser.parse_args()

    if args.resume_run_dir is not None:
        if args.config is not None or args.experiment is not None or args.seed is not None or args.dry_run:
            raise ValueError("--resume-run-dir cannot be combined with --config, --experiment, --seed, or --dry-run")
        resume_dir = args.resume_run_dir.resolve()
        resolved_path = resume_dir / "resolved_config.yaml"
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Resume run lacks resolved_config.yaml: {resume_dir}")
        config = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Resume resolved_config.yaml is not a mapping")
        active_experiment = str(config.get("active_experiment", ""))
        run_id = config.pop("run_id", None)
        config.pop("active_experiment", None)
        if not run_id or resume_dir.name != run_id:
            raise ValueError("Resume directory and resolved run_id do not match")
        experiments = select_experiments(config, [active_experiment])
        seeds = [int(config["seed"])]
    else:
        config_path = args.config or (PROJECT_ROOT / "configs" / "experiments.yaml")
        raw_config = load_config(config_path)
        config = resolve_config(raw_config, config_path, dry_run=args.dry_run)
        experiments = select_experiments(config, args.experiment)
        seeds = args.seed if args.seed is not None else config.get("seeds", [config["seed"]])
    configured_seeds = set(config.get("seeds", [config["seed"]]))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Selected seeds must be a non-empty list without duplicates")
    if not set(seeds).issubset(configured_seeds):
        raise ValueError(
            f"Selected seeds {seeds} must be a subset of configured seeds {sorted(configured_seeds)}"
        )
    immutable_experiments = {
        "eurosat_dofa_frozen_bnlinear",
        "eurosat_dofa_full_finetune",
    }
    protected = sorted(immutable_experiments & {experiment["name"] for experiment in experiments})
    if protected and not args.validate_only:
        raise RuntimeError(
            "Immutable DOFA-EuroSAT final experiments cannot be launched: " + ", ".join(protected)
        )
    if args.validate_only:
        validation = {
            "config": str(args.config) if args.config is not None else str(config.get("config_path")),
            "seeds": seeds,
            "experiments": experiments,
            "resume_run_dir": str(args.resume_run_dir.resolve()) if args.resume_run_dir is not None else None,
        }
        if args.resume_run_dir is not None:
            task = experiments[0].get("task", config.get("task"))
            if task in {"classification", "classification_multilabel"}:
                _, _, _, resume_context = prepare_classification_run_directory(
                    config, experiments[0], args.resume_run_dir,
                )
                validation["resume_checkpoint_epoch"] = int(resume_context["checkpoint"]["epoch"])
                validation["resume_next_epoch"] = int(resume_context["checkpoint"]["epoch"]) + 1
                validation["resume_code_changes"] = resume_context["code_changes"]
            else:
                checkpoint = _load_torch_checkpoint(args.resume_run_dir.resolve() / "last.pt", "cpu")
                validation["resume_checkpoint_epoch"] = int(checkpoint["epoch"])
                validation["resume_next_epoch"] = int(checkpoint["epoch"]) + 1
        print(json.dumps(validation, indent=2))
        return

    results = []
    if args.resume_run_dir is not None:
        lock_path = args.resume_run_dir.resolve() / ".resume.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Resume run is already locked by another process: {args.resume_run_dir}") from exc
            results.append(run_experiment(config, experiments[0], resume_run_dir=args.resume_run_dir))
    else:
        for seed in seeds:
            seed_config = copy.deepcopy(config)
            seed_config["seed"] = seed
            for experiment in experiments:
                results.append(run_experiment(seed_config, experiment))
    summary_root = Path(config.get("output_dir", "./results/experiments"))
    if not summary_root.is_absolute():
        summary_root = PROJECT_ROOT / summary_root
    if args.resume_run_dir is not None:
        summary_path = summary_root / "summaries" / f"{make_run_id('resume-summary', seeds[0])}.json"
    else:
        summary_path = make_invocation_summary_path(summary_root, seeds)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            [result.__dict__ if hasattr(result, "__dict__") else result for result in results],
            handle,
            indent=2,
        )
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
