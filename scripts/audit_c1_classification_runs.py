from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.experiment_manager import PROJECT_ROOT
from scripts.prediction_export import (
    checkpoint_sha256,
    load_prediction_export,
    recompute_metrics,
    validate_prediction_export,
)


SCHEMA_VERSION = 1
EXPECTED_CODE_HASH = re.compile(r"^[0-9a-f]{64}$")
FINAL_SEEDS = (42, 43, 44)
REPRESENTATION_DIM = 768
RESUME_CODE_CHANGE_ALLOWLIST = frozenset(
    {
        "scripts/run_experiments.py",
        "tests/test_experiment_configuration.py",
    }
)
RESUME_PROTECTED_PROVENANCE = (
    "resolved_config.yaml",
    "code_snapshot.json",
    "environment.json",
    "input_artifacts.json",
    "model_audit.json",
    "optimizer_group_audit.json",
    "gradient_audit.json",
)
LEGACY_RESUME_SOURCE_EPOCH = 5
LEGACY_RESUME_NEXT_EPOCH = 6

EUROSAT_SPLIT_MANIFEST = Path("splits/eurosat_70_10_10_10_spatial20m/eurosat_splits.csv")
EUROSAT_ARTIFACT = Path("data/EuroSATallBands.zip")
TREESATAI_ARTIFACT = Path("datasets/treesatai/geobench_treesatai.tortilla")
TREESATAI_MANIFEST = Path("reports/dataset_manifests/treesatai_actual_manifest.csv")
DOFA_WEIGHTS = Path("DOFA/checkpoints/DOFA_ViT_base_e100.pth")
FINAL_TRAINING_PROTOCOL = Path("reports/final_training_protocol.md")

EXPECTED_HASHES = {
    "eurosat_artifact": "751f070f9bffa2eed48b24ca2dd0b02959280c08837e8c9a5532a67ba611df59",
    "eurosat_split_manifest": "c5cadc7936394f0678307f7db25c55a2dc19f73ddb93891b772c16221e8abf22",
    "treesatai_artifact": "0ddb8068720242ad4f5931ea91f3459ed695ad490bbaa48905afe72dd9623aee",
    "treesatai_manifest": "b958ada672e7aeaa29652731af814834e9dae4fbe1c53d2edc92802ff18eae54",
    "dofa_weights": "4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1",
    "panopticon_weights": "55024f411a7f383ed1a646d9b833b65683a4443846603fc7d37591d5afe9d26e",
    "training_protocol": "cd569a938552f5cb85573d335d4f1a4c9b0815445a999853226b8052f099d711",
}

EUROSAT_CHANNELS = ("B04", "B03", "B02")
EUROSAT_WAVELENGTHS_NM = (664.6300422881802, 559.5987534818435, 492.9971095687347)
EUROSAT_MEAN = (936.085209866211, 1031.3388562784282, 1111.4795678522003)
EUROSAT_STD = (589.387623763833, 388.3087839027959, 327.14241012761033)
EUROSAT_CLASSES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)

TREESATAI_CHANNELS = (
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
)
TREESATAI_WAVELENGTHS_NM = (490.0, 560.0, 665.0, 842.0, 705.0, 740.0, 783.0, 865.0, 1610.0, 2190.0, 443.0, 945.0)
TREESATAI_MEAN = (
    245.31068420410156,
    387.63568115234375,
    248.4667205810547,
    2825.93603515625,
    625.9300537109375,
    2118.83740234375,
    2709.37890625,
    2982.208740234375,
    1316.7186279296875,
    594.203369140625,
    265.8070068359375,
    2962.182373046875,
)
TREESATAI_STD = (
    117.73491668701172,
    130.0995635986328,
    129.66375732421875,
    756.8175659179688,
    191.35238647460938,
    517.2822265625,
    691.1488037109375,
    754.9419555664062,
    411.339111328125,
    234.48863220214844,
    125.9928207397461,
    674.169189453125,
)
TREESATAI_CLASSES = (
    "Abies",
    "Acer",
    "Alnus",
    "Betula",
    "Cleared",
    "Fagus",
    "Fraxinus",
    "Larix",
    "Picea",
    "Pinus",
    "Populus",
    "Prunus",
    "Pseudotsuga",
    "Quercus",
    "Tilia",
)


@dataclass(frozen=True)
class Target:
    dataset: str
    model: str
    adaptation: str
    experiment: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.dataset}|{self.model}|{self.adaptation}|seed{self.seed}"

    @property
    def task(self) -> str:
        return "classification_multilabel" if self.dataset == "treesatai" else "classification"

    @property
    def expected_test_count(self) -> int:
        return 2000 if self.dataset == "treesatai" else 2714

    @property
    def expected_class_count(self) -> int:
        return 15 if self.dataset == "treesatai" else 10

    @property
    def expected_train_count(self) -> int:
        return 4000 if self.dataset == "treesatai" else 18866


CELL_DEFINITIONS = (
    ("eurosat", "panopticon", "frozen", "eurosat_panopticon_frozen_bnlinear"),
    ("eurosat", "panopticon", "full_finetune", "eurosat_panopticon_full_finetune"),
    ("treesatai", "dofa", "frozen", "treesatai_dofa_frozen"),
    ("treesatai", "dofa", "full_finetune", "treesatai_dofa_full_finetune"),
    ("treesatai", "panopticon", "frozen", "treesatai_panopticon_frozen"),
    ("treesatai", "panopticon", "full_finetune", "treesatai_panopticon_full_finetune"),
)


def expected_targets(seeds: Sequence[int]) -> list[Target]:
    invalid = sorted(set(seeds) - set(FINAL_SEEDS))
    if invalid:
        raise ValueError(f"C1 final seeds must be drawn from {FINAL_SEEDS}; got {invalid}")
    return [Target(*cell, int(seed)) for seed in seeds for cell in CELL_DEFINITIONS]


@lru_cache(maxsize=None)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def values_equal(actual: Any, expected: Any, tolerance: float = 1.0e-12) -> bool:
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            return False
        return all(values_equal(left, right, tolerance) for left, right in zip(actual, expected))
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance)
        except (TypeError, ValueError):
            return False
    return actual == expected


def require_equal(errors: list[str], label: str, actual: Any, expected: Any) -> None:
    if not values_equal(actual, expected):
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def active_experiment(config: Mapping[str, Any]) -> Mapping[str, Any]:
    name = config.get("active_experiment")
    matches = [item for item in config.get("experiments", []) if item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"Resolved config must identify exactly one active experiment; got {name!r}")
    return matches[0]


def identify_target(config: Mapping[str, Any]) -> Target | None:
    try:
        experiment = active_experiment(config)
    except (TypeError, ValueError):
        return None
    name = str(experiment.get("name"))
    seed = config.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        return None
    for dataset, model, adaptation, expected_name in CELL_DEFINITIONS:
        if name == expected_name:
            return Target(dataset, model, adaptation, expected_name, seed)
    return None


def protocol_errors(config: Mapping[str, Any], target: Target) -> list[str]:
    errors: list[str] = []
    data = config.get("data", {})
    training = config.get("training", {})
    metrics = config.get("metrics", {})
    reproducibility = config.get("reproducibility", {})
    export = config.get("prediction_export", {})
    provenance = config.get("provenance", {})

    try:
        experiment = active_experiment(config)
    except (TypeError, ValueError) as exc:
        return [str(exc)]
    model = experiment.get("model", {})

    require_equal(errors, "seed", config.get("seed"), target.seed)
    require_equal(errors, "task", config.get("task"), target.task)
    require_equal(errors, "data.name", data.get("name"), target.dataset)
    require_equal(errors, "active experiment", experiment.get("name"), target.experiment)
    require_equal(errors, "model.name", model.get("name"), target.model)
    require_equal(errors, "model.adaptation_mode", model.get("adaptation_mode"), target.adaptation)
    require_equal(errors, "model.pretrained", model.get("pretrained"), True)
    require_equal(errors, "model.size", model.get("size", "base"), "base")
    require_equal(errors, "dry_run", config.get("dry_run", False), False)
    require_equal(errors, "image size", data.get("image_size"), 224)
    require_equal(errors, "normalization enabled", data.get("normalize"), True)
    require_equal(errors, "training.batch_size", training.get("batch_size"), 64)
    require_equal(errors, "optimizer", training.get("optimizer", {}).get("name"), "adamw")
    require_equal(errors, "checkpoint split", training.get("checkpoint", {}).get("split"), "val")
    require_equal(errors, "checkpoint metric", training.get("checkpoint", {}).get("metric"), "nll")
    require_equal(errors, "checkpoint mode", training.get("checkpoint", {}).get("mode"), "min")
    require_equal(errors, "limit_train_batches", training.get("limit_train_batches"), None)
    require_equal(errors, "limit_val_batches", training.get("limit_val_batches"), None)
    require_equal(errors, "deterministic execution", reproducibility.get("deterministic"), True)
    require_equal(errors, "determinism warn_only", reproducibility.get("warn_only", False), False)
    require_equal(errors, "ECE bins", metrics.get("n_bins"), 15)
    require_equal(errors, "prediction export enabled", export.get("enabled"), True)
    require_equal(errors, "prediction export splits", export.get("splits"), ["test"])
    require_equal(errors, "representation export", export.get("save_backbone_representation"), True)
    require_equal(errors, "embedding sidecar export", export.get("save_embeddings"), True)
    require_equal(errors, "UQ methods during deterministic training", config.get("uq_methods"), [{"name": "none"}])
    require_equal(errors, "gradient clipping disabled", training.get("gradient_clipping", {}).get("enabled", False), False)
    require_equal(errors, "layerwise LR decay disabled", training.get("layerwise_lr_decay", {}).get("enabled", False), False)
    require_equal(errors, "input provenance required", provenance.get("require_input_hashes"), True)
    require_equal(errors, "training protocol ID", provenance.get("protocol_id"), "B5-2026-08-15-v1")
    require_equal(errors, "training protocol SHA", provenance.get("protocol_sha256"), EXPECTED_HASHES["training_protocol"])
    for split in ("train", "val", "calibration", "test"):
        require_equal(errors, f"augmentation.{split}", data.get("augmentations", {}).get(split, "none"), "none")

    normalization = data.get("normalization", {})
    require_equal(errors, "normalization method", normalization.get("method"), "channelwise_standardization")
    require_equal(errors, "normalization split", normalization.get("statistics_split"), "train")

    if target.dataset == "eurosat":
        require_equal(errors, "EuroSAT artifact SHA", data.get("artifact_sha256"), EXPECTED_HASHES["eurosat_artifact"])
        require_equal(
            errors,
            "EuroSAT split manifest SHA",
            data.get("split_manifest_sha256"),
            EXPECTED_HASHES["eurosat_split_manifest"],
        )
        require_equal(errors, "EuroSAT input bands", data.get("input_bands"), "rgb")
        require_equal(errors, "EuroSAT channels", data.get("channels"), list(EUROSAT_CHANNELS))
        require_equal(errors, "EuroSAT wavelengths_nm", data.get("wavelengths_nm"), list(EUROSAT_WAVELENGTHS_NM))
        require_equal(errors, "EuroSAT normalization mean", normalization.get("mean"), list(EUROSAT_MEAN))
        require_equal(errors, "EuroSAT normalization std", normalization.get("std"), list(EUROSAT_STD))
        require_equal(errors, "EuroSAT classes", data.get("class_names"), list(EUROSAT_CLASSES))
        require_equal(errors, "EuroSAT num_classes", data.get("num_classes"), 10)
        require_equal(errors, "EuroSAT head architecture", model.get("head", {}).get("architecture"), "batchnorm_linear")
        require_equal(errors, "EuroSAT head dropout", model.get("head", {}).get("dropout"), 0.0)
        require_equal(errors, "EuroSAT head BatchNorm affine", model.get("head", {}).get("batchnorm_affine"), False)
        require_equal(errors, "EuroSAT head BatchNorm eps", model.get("head", {}).get("batchnorm_eps"), 1.0e-6)
    else:
        require_equal(errors, "TreeSatAI artifact SHA", data.get("artifact_sha256"), EXPECTED_HASHES["treesatai_artifact"])
        require_equal(
            errors,
            "TreeSatAI extracted manifest SHA",
            data.get("extracted_manifest_sha256"),
            EXPECTED_HASHES["treesatai_manifest"],
        )
        require_equal(errors, "TreeSatAI protocol", data.get("protocol"), "geobench2_treesatai")
        if "fd9d0b664e6fb0faba54636bdff4906634debd4b" not in str(data.get("source", "")):
            errors.append("TreeSatAI source does not contain the pinned GEO-Bench-2 commit")
        require_equal(errors, "TreeSatAI input bands", data.get("input_bands"), "s2")
        require_equal(errors, "TreeSatAI channels", data.get("channels"), list(TREESATAI_CHANNELS))
        require_equal(errors, "TreeSatAI wavelengths_nm", data.get("wavelengths_nm"), list(TREESATAI_WAVELENGTHS_NM))
        require_equal(errors, "TreeSatAI normalization mean", normalization.get("mean"), list(TREESATAI_MEAN))
        require_equal(errors, "TreeSatAI normalization std", normalization.get("std"), list(TREESATAI_STD))
        require_equal(errors, "TreeSatAI classes", data.get("class_names"), list(TREESATAI_CLASSES))
        require_equal(errors, "TreeSatAI num_classes", data.get("num_classes"), 15)
        require_equal(errors, "TreeSatAI multilabel", data.get("multilabel"), True)
        require_equal(errors, "TreeSatAI threshold", metrics.get("multilabel_threshold"), 0.5)
        require_equal(
            errors,
            "TreeSatAI temporal protocol",
            data.get("temporal_protocol"),
            "shared_encoder_per_timestamp_then_mean_features",
        )
        require_equal(errors, "TreeSatAI head architecture", model.get("head", {}).get("architecture"), "linear")
        require_equal(errors, "TreeSatAI head dropout", model.get("head", {}).get("dropout"), 0.0)

    early = training.get("early_stopping", {})
    warmup = training.get("warmup", {})
    require_equal(errors, "early stopping enabled", early.get("enabled"), True)
    require_equal(errors, "early stopping min_delta", early.get("min_delta"), 0.0)

    if target.dataset == "eurosat" and target.adaptation == "frozen":
        expected = {"epochs": 50, "patience": 10, "weight_decay": 0.01, "head_lr": 1.0e-3}
    elif target.dataset == "eurosat":
        expected = {
            "epochs": 100,
            "patience": 15,
            "weight_decay": 0.0,
            "backbone_lr": 4.0e-4,
            "head_lr": 4.0e-3,
        }
    elif target.adaptation == "frozen":
        expected = {"epochs": 50, "patience": 10, "weight_decay": 0.01, "head_lr": 1.0e-3}
    else:
        expected = {
            "epochs": 100,
            "patience": 15,
            "weight_decay": 0.01,
            "backbone_lr": 1.0e-4,
            "head_lr": 1.0e-3,
        }
    require_equal(errors, "epoch cap", training.get("epochs"), expected["epochs"])
    require_equal(errors, "early stopping patience", early.get("patience"), expected["patience"])
    require_equal(errors, "weight decay", training.get("weight_decay"), expected["weight_decay"])
    require_equal(errors, "head LR", training.get("head_learning_rate"), expected["head_lr"])
    require_equal(
        errors,
        "default learning rate",
        training.get("learning_rate"),
        expected.get("backbone_lr", expected["head_lr"]),
    )
    if target.adaptation == "full_finetune":
        require_equal(errors, "backbone LR", training.get("backbone_learning_rate"), expected["backbone_lr"])
        require_equal(errors, "warmup enabled", warmup.get("enabled"), True)
        require_equal(errors, "warmup epochs", warmup.get("epochs"), 5)
        require_equal(errors, "warmup type", warmup.get("type"), "linear")
    else:
        require_equal(errors, "warmup enabled", warmup.get("enabled", False), False)

    expected_units = "micrometers" if target.model == "dofa" else "nanometers"
    expected_wavelengths = (
        [value / 1000.0 for value in data.get("wavelengths_nm", [])]
        if target.model == "dofa"
        else data.get("wavelengths_nm")
    )
    require_equal(errors, "actual model wavelength units", model.get("actual_wavelength_units"), expected_units)
    require_equal(errors, "actual model wavelengths", model.get("actual_wavelengths"), expected_wavelengths)
    require_equal(
        errors,
        "pretrained weights SHA",
        model.get("weights_sha256"),
        EXPECTED_HASHES["dofa_weights" if target.model == "dofa" else "panopticon_weights"],
    )
    return errors


def _validate_code_snapshot_bundle(
    snapshot_path: Path,
    environment_path: Path,
    errors: list[str],
    context: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    if not snapshot_path.is_file() or not environment_path.is_file():
        errors.append(f"{context} code_snapshot.json and/or environment.json is missing")
        return {}, {}
    try:
        snapshot = read_json(snapshot_path)
        environment = read_json(environment_path)
    except Exception as exc:
        errors.append(f"{context} code snapshot bundle could not be read: {type(exc).__name__}: {exc}")
        return {}, {}
    entries = snapshot.get("files", [])
    if not isinstance(entries, list):
        errors.append(f"{context} code snapshot files is not a list")
        return {}, {}
    aggregate = hashlib.sha256()
    seen: set[str] = set()
    entry_map: dict[str, str] = {}
    for item in entries:
        if not isinstance(item, Mapping):
            errors.append(f"{context} code snapshot contains a non-mapping file entry")
            continue
        relative = item.get("path")
        digest = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str) or not EXPECTED_CODE_HASH.fullmatch(digest):
            errors.append(f"{context} code snapshot contains an invalid path/hash entry")
            continue
        if relative in seen:
            errors.append(f"{context} code snapshot repeats path {relative}")
        seen.add(relative)
        entry_map[relative] = digest
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    actual = aggregate.hexdigest()
    recorded = snapshot.get("code_sha256")
    if not entries:
        errors.append(f"{context} code snapshot has no files")
    if actual != recorded:
        errors.append(f"{context} code snapshot aggregate mismatch: {actual} != {recorded}")
    if snapshot.get("file_count", len(entries)) != len(entries):
        errors.append(f"{context} code snapshot file_count does not match its entries")
    if environment.get("code_version") != f"sha256:{recorded}":
        errors.append(f"{context} environment code_version does not match code snapshot")
    environment_snapshot = environment.get("code_snapshot")
    if isinstance(environment_snapshot, Mapping):
        if environment_snapshot.get("code_sha256") != recorded:
            errors.append(f"{context} environment embedded code SHA does not match code snapshot")
        if environment_snapshot.get("file_count") != len(entries):
            errors.append(f"{context} environment embedded file count does not match code snapshot")
    return (
        {
            "code_sha256": recorded,
            "file_count": len(entries),
            "environment_code_version": environment.get("code_version"),
        },
        entry_map,
    )


def validate_code_snapshot(run_dir: Path, errors: list[str]) -> dict[str, Any]:
    evidence, _ = _validate_code_snapshot_bundle(
        run_dir / "code_snapshot.json",
        run_dir / "environment.json",
        errors,
        "original run",
    )
    return evidence


def expected_provenance_paths(project_root: Path, target: Target) -> dict[str, tuple[Path, str]]:
    common: dict[str, tuple[Path, str]] = {
        "training_protocol": (
            project_root / FINAL_TRAINING_PROTOCOL,
            EXPECTED_HASHES["training_protocol"],
        )
    }
    if target.dataset == "eurosat":
        common["dataset"] = (project_root / EUROSAT_ARTIFACT, EXPECTED_HASHES["eurosat_artifact"])
        common["split_manifest"] = (
            project_root / EUROSAT_SPLIT_MANIFEST,
            EXPECTED_HASHES["eurosat_split_manifest"],
        )
    else:
        common["dataset"] = (project_root / TREESATAI_ARTIFACT, EXPECTED_HASHES["treesatai_artifact"])
        common["extracted_manifest"] = (project_root / TREESATAI_MANIFEST, EXPECTED_HASHES["treesatai_manifest"])
    if target.model == "dofa":
        common["pretrained_weights"] = (project_root / DOFA_WEIGHTS, EXPECTED_HASHES["dofa_weights"])
    else:
        panopticon = Path(torch.hub.get_dir()) / "checkpoints" / "panopticon_vitb14_teacher.pth"
        common["pretrained_weights"] = (panopticon, EXPECTED_HASHES["panopticon_weights"])
    return common


def validate_provenance_files(project_root: Path, target: Target, errors: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, (path, expected) in expected_provenance_paths(project_root, target).items():
        if not path.is_file():
            errors.append(f"provenance artifact is missing: {path}")
            values[name] = {"path": str(path), "expected_sha256": expected, "actual_sha256": None}
            continue
        actual = sha256_file(path)
        if actual != expected:
            errors.append(f"provenance hash mismatch for {name}: {actual} != {expected}")
        values[name] = {"path": str(path), "expected_sha256": expected, "actual_sha256": actual}
    return values


def validate_recorded_input_artifacts(
    run_dir: Path,
    actual_provenance: Mapping[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    path = run_dir / "input_artifacts.json"
    if not path.is_file():
        errors.append("input_artifacts.json is missing")
        return {}
    recorded = read_json(path)
    if recorded.get("required") is not True or recorded.get("all_verified") is not True:
        errors.append("input artifact provenance was not required and fully verified at run start")
    records = recorded.get("records", {})
    if set(records) != set(actual_provenance):
        errors.append(
            f"recorded input artifact keys {sorted(records)} != expected {sorted(actual_provenance)}"
        )
    for name, actual in actual_provenance.items():
        item = records.get(name, {})
        if item.get("verified") is not True:
            errors.append(f"recorded provenance {name} is not verified")
        if item.get("path") != actual.get("path"):
            errors.append(f"recorded provenance path mismatch for {name}")
        if item.get("sha256") != actual.get("actual_sha256"):
            errors.append(f"recorded provenance SHA mismatch for {name}")
        if item.get("expected_sha256") != actual.get("expected_sha256"):
            errors.append(f"recorded expected provenance SHA mismatch for {name}")
    return recorded


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_history(
    history: Any,
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    target: Target,
    errors: list[str],
) -> dict[str, Any]:
    if not isinstance(history, list) or not history:
        errors.append("training history is empty or invalid")
        return {}
    epochs = [item.get("epoch") for item in history]
    expected_epoch_numbers = list(range(1, len(history) + 1))
    if epochs != expected_epoch_numbers:
        errors.append(f"training epochs are not contiguous from 1: {epochs}")

    required_train = ("loss", "accuracy", "gradient_norm", "backbone_gradient_norm", "head_gradient_norm")
    required_val = ("accuracy", "nll", "ece", "brier", "mean_confidence", "predictive_entropy")
    if target.dataset == "treesatai":
        required_val = (*required_val, "labelwise_accuracy", "macro_f1")
    for item in history:
        epoch = item.get("epoch")
        for group, keys in (("train", required_train), ("val", required_val)):
            metrics = item.get(group, {})
            for key in keys:
                if not finite_number(metrics.get(key)):
                    errors.append(f"epoch {epoch} {group}.{key} is missing or non-finite")
        if not finite_number(item.get("head_learning_rate")):
            errors.append(f"epoch {epoch} head_learning_rate is missing or non-finite")
        backbone_lr = item.get("backbone_learning_rate")
        if target.adaptation == "full_finetune" and not finite_number(backbone_lr):
            errors.append(f"epoch {epoch} backbone_learning_rate is missing or non-finite")

    val_nll = [float(item["val"]["nll"]) for item in history if finite_number(item.get("val", {}).get("nll"))]
    if len(val_nll) != len(history):
        return {}
    best_offset = min(range(len(val_nll)), key=val_nll.__getitem__)
    best_epoch = best_offset + 1
    if summary.get("best_epoch") != best_epoch:
        errors.append(f"run_summary best_epoch {summary.get('best_epoch')} != earliest minimum-val-NLL epoch {best_epoch}")

    training = config["training"]
    epoch_cap = int(training["epochs"])
    patience = int(training["early_stopping"]["patience"])
    last_epoch = len(history)
    if last_epoch > epoch_cap:
        errors.append(f"last epoch {last_epoch} exceeds cap {epoch_cap}")
    trailing_without_improvement = last_epoch - best_epoch
    if last_epoch < epoch_cap and trailing_without_improvement != patience:
        errors.append(
            f"early-stop counter mismatch: stopped at {last_epoch}, best={best_epoch}, "
            f"trailing={trailing_without_improvement}, patience={patience}"
        )
    if last_epoch == epoch_cap and trailing_without_improvement >= patience:
        errors.append("run reached epoch cap despite already exhausting early-stopping patience")

    for item in history:
        epoch = int(item["epoch"])
        expected_factor = min(1.0, epoch / 5.0) if target.adaptation == "full_finetune" else 1.0
        if not values_equal(item.get("warmup_factor"), expected_factor):
            errors.append(f"epoch {epoch} warmup_factor is incorrect")
        expected_head = float(training["head_learning_rate"]) * expected_factor
        if not values_equal(item.get("head_learning_rate"), expected_head):
            errors.append(f"epoch {epoch} head LR is incorrect")
        if target.adaptation == "full_finetune":
            expected_backbone = float(training["backbone_learning_rate"]) * expected_factor
            if not values_equal(item.get("backbone_learning_rate"), expected_backbone):
                errors.append(f"epoch {epoch} backbone LR is incorrect")

    return {
        "epochs_completed": last_epoch,
        "epoch_cap": epoch_cap,
        "best_epoch": best_epoch,
        "best_validation_nll": val_nll[best_offset],
        "first_validation_nll": val_nll[0],
        "last_validation_nll": val_nll[-1],
        "best_validation_accuracy": float(history[best_offset]["val"]["accuracy"]),
        "best_validation_macro_f1": (
            float(history[best_offset]["val"]["macro_f1"])
            if finite_number(history[best_offset]["val"].get("macro_f1"))
            else None
        ),
        "termination": "early_stopping" if last_epoch < epoch_cap else "epoch_cap",
    }


def validate_batch_metrics(path: Path, history_length: int, target: Target, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append("batch_metrics.csv is missing")
        return {}
    frame = pd.read_csv(path)
    expected_per_epoch = math.ceil(target.expected_train_count / 64)
    expected_total = history_length * expected_per_epoch
    if len(frame) != expected_total:
        errors.append(f"batch metric row count {len(frame)} != expected {expected_total}")
    if "global_step" not in frame or frame["global_step"].tolist() != list(range(1, len(frame) + 1)):
        errors.append("batch global_step is not contiguous from 1")
    for column in ("loss", "accuracy", "gradient_norm", "backbone_gradient_norm", "head_gradient_norm"):
        if column not in frame or not np.isfinite(frame[column].to_numpy(dtype=np.float64)).all():
            errors.append(f"batch metric {column} is missing or non-finite")
    counts = frame.groupby("epoch").size().to_dict() if "epoch" in frame else {}
    if counts != {epoch: expected_per_epoch for epoch in range(1, history_length + 1)}:
        errors.append(f"per-epoch batch counts are incorrect: {counts}")
    return {"rows": len(frame), "expected_rows": expected_total, "batches_per_epoch": expected_per_epoch}


def validate_epoch_metrics_csv(
    path: Path,
    history: Sequence[Mapping[str, Any]],
    target: Target,
    errors: list[str],
) -> dict[str, Any]:
    if not path.is_file():
        errors.append("training_metrics.csv is missing")
        return {}
    frame = pd.read_csv(path)
    if len(frame) != len(history):
        errors.append(f"training metric row count {len(frame)} != history length {len(history)}")
    if "epoch" not in frame or frame["epoch"].tolist() != list(range(1, len(frame) + 1)):
        errors.append("training_metrics.csv epochs are not contiguous from 1")
    finite_columns = (
        "train_loss",
        "val_nll",
        "train_accuracy",
        "val_accuracy",
        "val_ece",
        "val_brier",
        "val_mean_confidence",
        "val_predictive_entropy",
        "head_learning_rate",
        "gradient_norm",
        "backbone_gradient_norm",
        "head_gradient_norm",
    )
    for column in finite_columns:
        if column not in frame or not np.isfinite(frame[column].to_numpy(dtype=np.float64)).all():
            errors.append(f"training metric column {column} is missing or non-finite")
    if target.adaptation == "full_finetune":
        if "backbone_learning_rate" not in frame or not np.isfinite(
            frame["backbone_learning_rate"].to_numpy(dtype=np.float64)
        ).all():
            errors.append("full-finetune backbone_learning_rate CSV column is missing or non-finite")
    if len(frame) == len(history) and "val_nll" in frame:
        expected_nll = np.asarray([item["val"]["nll"] for item in history], dtype=np.float64)
        # pandas/CSV decimal round-tripping can move a binary64 value by one ULP.
        if not np.allclose(frame["val_nll"].to_numpy(dtype=np.float64), expected_nll, atol=1.0e-15, rtol=0.0):
            errors.append("training_metrics.csv validation NLL differs from training_history.json")
    return {"rows": len(frame), "expected_rows": len(history)}


def validate_model_and_gradients(run_dir: Path, target: Target, errors: list[str]) -> dict[str, Any]:
    required = {
        "model": run_dir / "model_audit.json",
        "optimizer": run_dir / "optimizer_group_audit.json",
        "gradient": run_dir / "gradient_audit.json",
        "stability": run_dir / "training_stability.json",
    }
    if any(not path.is_file() for path in required.values()):
        missing = [name for name, path in required.items() if not path.is_file()]
        errors.append(f"model/gradient audit artifacts are missing: {missing}")
        return {}
    model = read_json(required["model"])
    optimizer = read_json(required["optimizer"])
    gradient = read_json(required["gradient"])
    stability = read_json(required["stability"])

    if model.get("adaptation_mode") != target.adaptation:
        errors.append("model audit adaptation mode is incorrect")
    if model.get("head_trainable_parameters") != model.get("head_total_parameters"):
        errors.append("classification head is not fully trainable")
    if target.adaptation == "frozen":
        if model.get("backbone_trainable_parameters") != 0 or model.get("backbone_requires_grad_all_false") is not True:
            errors.append("frozen backbone contains trainable parameters")
        if set(optimizer) != {"head"}:
            errors.append(f"frozen optimizer groups must be head-only; got {sorted(optimizer)}")
    else:
        if model.get("all_expected_backbone_parameters_trainable") is not True:
            errors.append("full-finetune backbone trainability audit failed")
        if set(optimizer) != {"backbone", "head"}:
            errors.append(f"full-finetune optimizer groups must be backbone/head; got {sorted(optimizer)}")
        if optimizer.get("backbone", {}).get("trainable_parameters") != model.get("backbone_trainable_parameters"):
            errors.append("backbone optimizer coverage is incomplete")
    if optimizer.get("head", {}).get("trainable_parameters") != model.get("head_trainable_parameters"):
        errors.append("head optimizer coverage is incomplete")

    if gradient.get("performed") is not True:
        errors.append("first-batch gradient audit was not performed")
    for key in ("missing_gradient_parameters", "zero_gradient_parameters", "nonfinite_gradient_parameters"):
        if gradient.get(key) not in ([], None):
            errors.append(f"gradient audit {key} is nonempty: {gradient.get(key)}")
    if gradient.get("all_expected_parameters_received_gradient") is not True:
        errors.append("not all expected parameters received gradients")
    if stability.get("nan_or_inf_detected") is not False:
        errors.append("training stability reports NaN/Inf")

    backbone_range = stability.get("backbone_gradient_norm_range", [])
    head_range = stability.get("head_gradient_norm_range", [])
    if len(head_range) != 2 or not all(finite_number(value) and float(value) > 0 for value in head_range):
        errors.append("head gradient norm range is not finite and positive")
    if target.adaptation == "frozen":
        if backbone_range != [0.0, 0.0]:
            errors.append(f"frozen backbone gradient range must be [0,0], got {backbone_range}")
    elif len(backbone_range) != 2 or not all(finite_number(value) and float(value) > 0 for value in backbone_range):
        errors.append("full-finetune backbone gradient norm range is not finite and positive")

    pretrained = model.get("pretrained_load") or {}
    if pretrained.get("missing_keys") != []:
        errors.append(f"pretrained load has missing keys: {pretrained.get('missing_keys')}")
    if target.model == "panopticon":
        if pretrained.get("unexpected_keys") != []:
            errors.append(f"Panopticon pretrained load has unexpected keys: {pretrained.get('unexpected_keys')}")
        if pretrained.get("weights") != "VIT_BASE14":
            errors.append("Panopticon pretrained weight enum is not VIT_BASE14")
    else:
        # These are pretraining-only DOFA tensors and were already documented by
        # the validated backend; the downstream feature encoder has no matching
        # parameters for them.
        expected_pretraining_only = {"mask_token", "projector.weight", "projector.bias"}
        if set(pretrained.get("unexpected_keys", [])) != expected_pretraining_only:
            errors.append(
                "DOFA pretrained load unexpected-key set differs from the validated "
                f"pretraining-only set: {pretrained.get('unexpected_keys')}"
            )
    return {"model_audit": model, "optimizer_group_audit": optimizer, "gradient_audit": gradient}


def load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def legacy_exact_train_generator_state_sha256(
    seed: int,
    dataset_length: int,
    completed_epochs: int,
) -> str:
    """Reproduce the train-loader generator state used by the legacy C1 resume.

    The legacy loader uses a seeded ``RandomSampler`` without replacement. Each
    epoch consumes one worker-base-seed draw, one full ``randperm``, and the
    sampler's (empty, for ``num_samples == len(dataset)``) remainder randperm.
    This performs no dataset access, model inference, or training.
    """

    if dataset_length <= 0:
        raise ValueError("dataset_length must be positive")
    if completed_epochs < 0:
        raise ValueError("completed_epochs must be non-negative")
    generator = torch.Generator().manual_seed(int(seed) % (2**63))
    for _ in range(completed_epochs):
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        torch.randperm(dataset_length, generator=generator)
        # RandomSampler performs this second call even though the remainder is empty.
        torch.randperm(dataset_length, generator=generator)[:0]
    return hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    """Stable comparison for JSON-like checkpoint metadata, including NaN."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True)


def _records_match_frame(records: Sequence[Mapping[str, Any]], frame: pd.DataFrame) -> bool:
    """Compare checkpoint batch records with their CSV serialization."""

    expected = pd.DataFrame(records)
    if list(expected.columns) != list(frame.columns) or len(expected) != len(frame):
        return False
    for column in expected.columns:
        expected_column = expected[column]
        actual_column = frame[column]
        if pd.api.types.is_numeric_dtype(expected_column):
            try:
                if not np.allclose(
                    expected_column.to_numpy(dtype=np.float64),
                    actual_column.to_numpy(dtype=np.float64),
                    rtol=1.0e-12,
                    atol=1.0e-12,
                    equal_nan=True,
                ):
                    return False
            except (TypeError, ValueError):
                return False
        elif expected_column.astype(str).tolist() != actual_column.astype(str).tolist():
            return False
    return True


def validate_checkpoint(
    run_dir: Path,
    run_id: str,
    target: Target,
    config: Mapping[str, Any],
    curve: Mapping[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    if not best_path.is_file() or best_path.stat().st_size == 0:
        errors.append("best.pt is missing or empty")
        return {}
    if not last_path.is_file() or last_path.stat().st_size == 0:
        errors.append("last.pt is missing or empty")
    digest = checkpoint_sha256(best_path)
    try:
        checkpoint = load_checkpoint(best_path)
    except Exception as exc:  # corrupt checkpoints must become audit failures
        errors.append(f"best.pt could not be loaded: {type(exc).__name__}: {exc}")
        return {"path": str(best_path), "sha256": digest}
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        errors.append("best.pt does not contain a model state dictionary")
        return {"path": str(best_path), "sha256": digest}
    if checkpoint.get("run_id") != run_id:
        errors.append("best.pt run_id does not match run directory metadata")
    if checkpoint.get("selection_metric") != "nll":
        errors.append("best.pt was not selected by validation NLL")
    if checkpoint.get("epoch") != curve.get("best_epoch"):
        errors.append("best.pt epoch does not match the audited best epoch")
    if not values_equal(checkpoint.get("selection_value"), curve.get("best_validation_nll")):
        errors.append("best.pt selection value does not match training history")
    if not values_equal(checkpoint.get("val_metrics", {}).get("nll"), curve.get("best_validation_nll")):
        errors.append("best.pt validation metrics do not match training history")
    checkpoint_experiment = checkpoint.get("experiment", {})
    if checkpoint_experiment.get("name") != target.experiment:
        errors.append("best.pt experiment does not match target")
    nonfinite_tensors = [
        name
        for name, value in checkpoint["model"].items()
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all()
    ]
    if nonfinite_tensors:
        errors.append(f"best.pt contains non-finite model tensors: {nonfinite_tensors[:10]}")
    del checkpoint
    gc.collect()
    last_evidence: dict[str, Any] = {
        "path": str(last_path),
        "size_bytes": last_path.stat().st_size if last_path.is_file() else None,
        "sha256": checkpoint_sha256(last_path) if last_path.is_file() and last_path.stat().st_size > 0 else None,
    }
    if last_path.is_file() and last_path.stat().st_size > 0:
        try:
            last = load_checkpoint(last_path)
            if not isinstance(last, dict):
                raise ValueError("last checkpoint is not a mapping")
            require_equal(errors, "last.pt run_id", last.get("run_id"), run_id)
            require_equal(errors, "last.pt epoch", last.get("epoch"), curve.get("epochs_completed"))
            require_equal(errors, "last.pt best_epoch", last.get("best_epoch"), curve.get("best_epoch"))
            require_equal(errors, "last.pt best_metric", last.get("best_metric"), curve.get("best_validation_nll"))
            require_equal(errors, "last.pt selection_metric", last.get("selection_metric"), "nll")
            require_equal(
                errors,
                "last.pt early-stopping counter",
                last.get("epochs_without_improvement"),
                int(curve.get("epochs_completed", 0)) - int(curve.get("best_epoch", 0)),
            )
            if len(last.get("history", [])) != int(curve.get("epochs_completed", 0)):
                errors.append("last.pt history length does not match the completed epoch count")
            expected_batch_rows = int(curve.get("epochs_completed", 0)) * math.ceil(
                target.expected_train_count / 64
            )
            if len(last.get("batch_history", [])) != expected_batch_rows:
                errors.append("last.pt batch history length is incomplete")
            optimizer = last.get("optimizer", {})
            groups = optimizer.get("param_groups", []) if isinstance(optimizer, dict) else []
            expected_names = ["head"] if target.adaptation == "frozen" else ["backbone", "head"]
            if [group.get("name") for group in groups] != expected_names:
                errors.append(f"last.pt optimizer groups are incorrect: {[group.get('name') for group in groups]}")
            training = config["training"]
            for group in groups:
                require_equal(errors, "AdamW betas", group.get("betas"), (0.9, 0.999))
                require_equal(errors, "AdamW epsilon", group.get("eps"), 1.0e-8)
                require_equal(errors, "AdamW amsgrad", group.get("amsgrad"), False)
                require_equal(errors, "AdamW weight decay", group.get("weight_decay"), training.get("weight_decay"))
                expected_lr = training.get(f"{group.get('name')}_learning_rate")
                require_equal(errors, f"{group.get('name')} optimizer base LR", group.get("base_lr"), expected_lr)
            last_nonfinite = [
                name
                for name, value in last.get("model", {}).items()
                if torch.is_tensor(value)
                and (value.is_floating_point() or value.is_complex())
                and not torch.isfinite(value).all()
            ]
            if last_nonfinite:
                errors.append(f"last.pt contains non-finite model tensors: {last_nonfinite[:10]}")
            last_evidence.update(
                {
                    "load_valid": True,
                    "epoch": last.get("epoch"),
                    "best_epoch": last.get("best_epoch"),
                    "epochs_without_improvement": last.get("epochs_without_improvement"),
                    "optimizer_groups": [group.get("name") for group in groups],
                }
            )
            del last
            gc.collect()
        except Exception as exc:
            errors.append(f"last.pt could not be validated: {type(exc).__name__}: {exc}")
            last_evidence["load_valid"] = False
    return {
        "best": {"path": str(best_path), "sha256": digest, "size_bytes": best_path.stat().st_size},
        "last": last_evidence,
        # Compatibility aliases used by the prediction-manifest audit.
        "path": str(best_path),
        "sha256": digest,
    }


def validate_resume_provenance(
    run_dir: Path,
    run_id: str,
    target: Target,
    summary: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    curve: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    """Validate the exceptional, explicitly recorded legacy C1 resume event.

    Runs without a ``resume_events`` directory remain subject to the ordinary
    audit only. Once that directory exists, the resume becomes promotion-
    relevant provenance and must be a single, complete, internally consistent
    event.
    """

    resume_root = run_dir / "resume_events"
    if not resume_root.exists():
        return {"resumed": False, "event_count": 0}
    if not resume_root.is_dir():
        errors.append("resume_events exists but is not a directory")
        return {"resumed": True, "event_count": 0}

    event_dirs = sorted(path for path in resume_root.iterdir() if path.is_dir())
    stray_entries = sorted(path.name for path in resume_root.iterdir() if not path.is_dir())
    if stray_entries:
        errors.append(f"resume_events contains non-event entries: {stray_entries}")
    if len(event_dirs) != 1:
        errors.append(f"resumed run must contain exactly one resume event; found {len(event_dirs)}")
        return {"resumed": True, "event_count": len(event_dirs)}
    event_dir = event_dirs[0]

    required_event_files = (
        "resume_request.json",
        "code_snapshot.json",
        "environment.json",
        "completion.json",
        f"source_last_epoch{LEGACY_RESUME_SOURCE_EPOCH}.pt",
    )
    missing = [name for name in required_event_files if not (event_dir / name).is_file()]
    if missing:
        errors.append(f"resume event is incomplete; missing {missing}")

    def read_event_json(name: str) -> dict[str, Any]:
        path = event_dir / name
        if not path.is_file():
            return {}
        try:
            value = read_json(path)
        except Exception as exc:
            errors.append(f"resume {name} could not be read: {type(exc).__name__}: {exc}")
            return {}
        if not isinstance(value, dict):
            errors.append(f"resume {name} is not a JSON object")
            return {}
        return value

    request = read_event_json("resume_request.json")
    completion = read_event_json("completion.json")

    original_snapshot, original_files = _validate_code_snapshot_bundle(
        run_dir / "code_snapshot.json",
        run_dir / "environment.json",
        errors,
        "original run",
    )
    resume_snapshot, resume_files = _validate_code_snapshot_bundle(
        event_dir / "code_snapshot.json",
        event_dir / "environment.json",
        errors,
        "resume event",
    )
    require_equal(errors, "resume request status", request.get("status"), "started")
    require_equal(errors, "resume request run_id", request.get("run_id"), run_id)
    require_equal(errors, "resume request seed", request.get("seed"), target.seed)
    require_equal(errors, "resume request experiment", request.get("experiment"), target.experiment)
    require_equal(errors, "resume request dataset", request.get("dataset"), target.dataset)
    require_equal(errors, "resume request model", request.get("model"), target.model)
    require_equal(errors, "resume request adaptation", request.get("adaptation"), target.adaptation)
    require_equal(
        errors,
        "resume original code aggregate",
        request.get("original_code_sha256"),
        original_snapshot.get("code_sha256"),
    )
    require_equal(
        errors,
        "resume code aggregate",
        request.get("resume_code_sha256"),
        resume_snapshot.get("code_sha256"),
    )

    actual_changed_paths = {
        path for path in set(original_files) | set(resume_files) if original_files.get(path) != resume_files.get(path)
    }
    if actual_changed_paths != RESUME_CODE_CHANGE_ALLOWLIST:
        errors.append(
            "resume snapshot changes differ from the exact allowlist: "
            f"{sorted(actual_changed_paths)} != {sorted(RESUME_CODE_CHANGE_ALLOWLIST)}"
        )
    recorded_changes = request.get("code_changes")
    change_map: dict[str, Mapping[str, Any]] = {}
    if not isinstance(recorded_changes, list):
        errors.append("resume code_changes is not a list")
    else:
        for item in recorded_changes:
            if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                errors.append("resume code_changes contains an invalid entry")
                continue
            path = str(item["path"])
            if path in change_map:
                errors.append(f"resume code_changes repeats path {path}")
            change_map[path] = item
    if set(change_map) != RESUME_CODE_CHANGE_ALLOWLIST:
        errors.append(
            "resume code_changes paths differ from the exact allowlist: "
            f"{sorted(change_map)} != {sorted(RESUME_CODE_CHANGE_ALLOWLIST)}"
        )
    for path in RESUME_CODE_CHANGE_ALLOWLIST:
        item = change_map.get(path, {})
        before = item.get("before_sha256")
        after = item.get("after_sha256")
        if not isinstance(before, str) or not EXPECTED_CODE_HASH.fullmatch(before):
            errors.append(f"resume code change {path} has no valid before SHA256")
        if not isinstance(after, str) or not EXPECTED_CODE_HASH.fullmatch(after):
            errors.append(f"resume code change {path} has no valid after SHA256")
        if before != original_files.get(path) or after != resume_files.get(path):
            errors.append(f"resume code change hashes do not match both snapshots for {path}")

    protected = request.get("protected_provenance_sha256")
    if not isinstance(protected, Mapping):
        errors.append("resume protected_provenance_sha256 is missing or invalid")
        protected = {}
    if set(protected) != set(RESUME_PROTECTED_PROVENANCE):
        errors.append(
            "resume protected provenance keys differ from the required set: "
            f"{sorted(protected)} != {sorted(RESUME_PROTECTED_PROVENANCE)}"
        )
    protected_verified: dict[str, bool] = {}
    for name in RESUME_PROTECTED_PROVENANCE:
        path = run_dir / name
        recorded_sha = protected.get(name)
        valid_recorded_sha = isinstance(recorded_sha, str) and EXPECTED_CODE_HASH.fullmatch(recorded_sha)
        if not path.is_file():
            errors.append(f"resume-protected provenance file is missing: {name}")
            protected_verified[name] = False
            continue
        actual_sha = sha256_file(path)
        matched = bool(valid_recorded_sha and actual_sha == recorded_sha)
        protected_verified[name] = matched
        if not valid_recorded_sha:
            errors.append(f"resume-protected provenance SHA is invalid for {name}")
        elif not matched:
            errors.append(f"resume-protected provenance hash changed for {name}: {actual_sha} != {recorded_sha}")
    for flag in (
        "config_verified_exact",
        "input_artifacts_verified_exact",
        "model_audit_verified_exact",
        "optimizer_groups_verified_exact",
        "gradient_audit_preserved",
    ):
        require_equal(errors, f"resume {flag}", request.get(flag), True)

    require_equal(
        errors,
        "resume generator restore method",
        request.get("generator_restore_method"),
        "legacy_exact_sampler_fast_forward",
    )
    require_equal(errors, "resume train dataset length", request.get("train_dataset_length"), target.expected_train_count)
    generator_sha = request.get("train_generator_state_sha256")
    if not isinstance(generator_sha, str) or not EXPECTED_CODE_HASH.fullmatch(generator_sha):
        errors.append("resume train generator state SHA256 is missing or invalid")
    else:
        reconstructed_generator_sha = legacy_exact_train_generator_state_sha256(
            target.seed,
            target.expected_train_count,
            LEGACY_RESUME_SOURCE_EPOCH,
        )
        if generator_sha != reconstructed_generator_sha:
            errors.append(
                "resume train generator state SHA does not match exact legacy sampler fast-forward: "
                f"{generator_sha} != {reconstructed_generator_sha}"
            )
    require_equal(errors, "resume next epoch", request.get("next_epoch"), LEGACY_RESUME_NEXT_EPOCH)
    require_equal(
        errors,
        "resume partial epoch policy",
        request.get("partial_epoch_policy"),
        "discard_uncheckpointed_work_and_restart_next_epoch",
    )
    require_equal(errors, "resume request test_access", request.get("test_access"), False)

    source_record = request.get("source_checkpoint")
    if not isinstance(source_record, Mapping):
        errors.append("resume source_checkpoint record is missing or invalid")
        source_record = {}
    source_path = event_dir / f"source_last_epoch{LEGACY_RESUME_SOURCE_EPOCH}.pt"
    archived_recorded = source_record.get("archived_path")
    if isinstance(archived_recorded, str):
        recorded_path = Path(archived_recorded)
        if not recorded_path.is_absolute():
            recorded_path = event_dir / recorded_path
        if recorded_path.resolve() != source_path.resolve():
            errors.append("resume archived checkpoint path does not identify the event-local epoch-5 archive")
    else:
        errors.append("resume archived checkpoint path is missing")
    original_last_recorded = source_record.get("path")
    if not isinstance(original_last_recorded, str) or Path(original_last_recorded).resolve() != (run_dir / "last.pt").resolve():
        errors.append("resume source checkpoint path does not identify the run's canonical last.pt")
    require_equal(errors, "resume source epoch", source_record.get("epoch"), LEGACY_RESUME_SOURCE_EPOCH)

    source_history: list[Mapping[str, Any]] = []
    source_batches: list[Mapping[str, Any]] = []
    source_sha: str | None = None
    if source_path.is_file():
        source_sha = checkpoint_sha256(source_path)
        require_equal(errors, "resume archived checkpoint SHA256", source_record.get("sha256"), source_sha)
        require_equal(errors, "resume archived checkpoint size", source_record.get("size_bytes"), source_path.stat().st_size)
        try:
            source_checkpoint = load_checkpoint(source_path)
            if not isinstance(source_checkpoint, dict):
                raise ValueError("archived source checkpoint is not a mapping")
            require_equal(errors, "archived source checkpoint run_id", source_checkpoint.get("run_id"), run_id)
            require_equal(
                errors,
                "archived source checkpoint epoch",
                source_checkpoint.get("epoch"),
                LEGACY_RESUME_SOURCE_EPOCH,
            )
            require_equal(
                errors,
                "archived source checkpoint best_epoch",
                source_checkpoint.get("best_epoch"),
                source_record.get("best_epoch"),
            )
            require_equal(
                errors,
                "archived source checkpoint best_metric",
                source_checkpoint.get("best_metric"),
                source_record.get("best_metric"),
            )
            require_equal(
                errors,
                "archived source checkpoint early-stopping counter",
                source_checkpoint.get("epochs_without_improvement"),
                source_record.get("epochs_without_improvement"),
            )
            require_equal(errors, "archived source selection metric", source_checkpoint.get("selection_metric"), "nll")
            source_history = list(source_checkpoint.get("history", []))
            source_batches = list(source_checkpoint.get("batch_history", []))
            source_config = source_checkpoint.get("config")
            source_experiment = source_checkpoint.get("experiment")
            if isinstance(source_experiment, Mapping):
                require_equal(
                    errors,
                    "archived source checkpoint experiment",
                    source_experiment.get("name"),
                    target.experiment,
                )
            config_for_identity = dict(source_config) if isinstance(source_config, Mapping) else {}
            if "active_experiment" not in config_for_identity and isinstance(source_experiment, Mapping):
                config_for_identity["active_experiment"] = source_experiment.get("name")
            if identify_target(config_for_identity) != target:
                errors.append("archived source checkpoint config does not identify the audited target")
            del source_checkpoint
            gc.collect()
        except Exception as exc:
            errors.append(f"archived source checkpoint could not be validated: {type(exc).__name__}: {exc}")
    elif source_path.name not in missing:
        errors.append("resume archived epoch-5 source checkpoint is missing")

    expected_source_batches = LEGACY_RESUME_SOURCE_EPOCH * math.ceil(target.expected_train_count / 64)
    if len(source_history) != LEGACY_RESUME_SOURCE_EPOCH:
        errors.append(
            f"archived source history length {len(source_history)} != {LEGACY_RESUME_SOURCE_EPOCH}"
        )
    elif [item.get("epoch") for item in source_history] != list(range(1, LEGACY_RESUME_NEXT_EPOCH)):
        errors.append("archived source history epochs are not contiguous from 1 through 5")
    if len(source_batches) != expected_source_batches:
        errors.append(f"archived source batch count {len(source_batches)} != {expected_source_batches}")
    elif [item.get("global_step") for item in source_batches] != list(range(1, expected_source_batches + 1)):
        errors.append("archived source batch global steps are not contiguous")
    if source_history and _canonical_json(source_history) != _canonical_json(list(history[:LEGACY_RESUME_SOURCE_EPOCH])):
        errors.append("combined history does not preserve the archived epoch-1-through-5 prefix")

    gradient_path = run_dir / "gradient_audit.json"
    gradient: dict[str, Any] = {}
    if gradient_path.is_file():
        value = read_json(gradient_path)
        if isinstance(value, dict):
            gradient = value
    require_equal(errors, "preserved gradient audit epoch", gradient.get("epoch"), 1)
    require_equal(errors, "preserved gradient audit performed", gradient.get("performed"), True)
    if source_history:
        source_gradient = source_history[0].get("train", {}).get("gradient_audit", {})
        comparable_gradient = {key: value for key, value in gradient.items() if key != "epoch"}
        if _canonical_json(source_gradient) != _canonical_json(comparable_gradient):
            errors.append("preserved epoch-1 gradient audit differs from the archived source history")

    combined_history_valid = False
    combined_batches_valid = False
    final_last_path = run_dir / "last.pt"
    if final_last_path.is_file():
        try:
            final_last = load_checkpoint(final_last_path)
            if not isinstance(final_last, dict):
                raise ValueError("final last checkpoint is not a mapping")
            final_history = list(final_last.get("history", []))
            final_batches = list(final_last.get("batch_history", []))
            expected_final_epochs = list(range(1, len(history) + 1))
            combined_history_valid = (
                [item.get("epoch") for item in final_history] == expected_final_epochs
                and _canonical_json(final_history) == _canonical_json(list(history))
                and _canonical_json(final_history[:LEGACY_RESUME_SOURCE_EPOCH]) == _canonical_json(source_history)
            )
            if not combined_history_valid:
                errors.append("final checkpoint history is not a contiguous, prefix-preserving combined history")
            expected_final_batches = len(history) * math.ceil(target.expected_train_count / 64)
            combined_batches_valid = (
                len(final_batches) == expected_final_batches
                and [item.get("global_step") for item in final_batches] == list(range(1, expected_final_batches + 1))
                and _canonical_json(final_batches[:expected_source_batches]) == _canonical_json(source_batches)
            )
            if not combined_batches_valid:
                errors.append("final checkpoint batch history is not contiguous and prefix-preserving")
            batch_csv_path = run_dir / "batch_metrics.csv"
            if batch_csv_path.is_file() and not _records_match_frame(final_batches, pd.read_csv(batch_csv_path)):
                errors.append("batch_metrics.csv differs from the combined final checkpoint batch history")
            del final_last
            gc.collect()
        except Exception as exc:
            errors.append(f"final last.pt could not be resume-audited: {type(exc).__name__}: {exc}")

    require_equal(errors, "resume completion status", completion.get("status"), "completed")
    require_equal(errors, "resume completion run_id", completion.get("run_id"), run_id)
    require_equal(
        errors,
        "resume completion start epoch",
        completion.get("resumed_from_epoch"),
        LEGACY_RESUME_NEXT_EPOCH,
    )
    require_equal(errors, "resume completion last epoch", completion.get("last_epoch"), len(history))
    require_equal(errors, "resume completion best epoch", completion.get("best_epoch"), curve.get("best_epoch"))
    require_equal(
        errors,
        "resume completion best metric",
        completion.get("best_metric"),
        curve.get("best_validation_nll"),
    )
    require_equal(
        errors,
        "resume completion best checkpoint SHA256",
        completion.get("best_checkpoint_sha256"),
        checkpoint.get("best", {}).get("sha256"),
    )
    require_equal(
        errors,
        "resume completion last checkpoint SHA256",
        completion.get("last_checkpoint_sha256"),
        checkpoint.get("last", {}).get("sha256"),
    )
    require_equal(
        errors,
        "resume completion test_access_during_promotion",
        completion.get("test_access_during_promotion"),
        False,
    )
    expected_export = (run_dir / "predictions" / "test" / "deterministic").resolve()
    final_export = completion.get("final_test_export")
    if not isinstance(final_export, str) or Path(final_export).resolve() != expected_export:
        errors.append("resume completion final_test_export does not identify the canonical deterministic test export")
    require_equal(
        errors,
        "run summary resumed_from_epoch",
        summary.get("training_segments", {}).get("resumed_from_epoch"),
        LEGACY_RESUME_NEXT_EPOCH,
    )
    summary_event = summary.get("resume_event")
    if not isinstance(summary_event, str) or Path(summary_event).resolve() != event_dir.resolve():
        errors.append("run_summary resume_event does not identify the sole audited resume event")

    return {
        "resumed": True,
        "event_count": 1,
        "event_dir": str(event_dir),
        "source_checkpoint": {
            "path": str(source_path),
            "sha256": source_sha,
            "epoch": source_record.get("epoch"),
            "run_id_verified": bool(source_history),
        },
        "original_code_sha256": original_snapshot.get("code_sha256"),
        "resume_code_sha256": resume_snapshot.get("code_sha256"),
        "allowed_code_changes": sorted(actual_changed_paths),
        "protected_provenance_verified": protected_verified,
        "generator_restore_method": request.get("generator_restore_method"),
        "train_generator_state_sha256": generator_sha,
        "next_epoch": request.get("next_epoch"),
        "completion_status": completion.get("status"),
        "combined_history_valid": combined_history_valid,
        "combined_batches_valid": combined_batches_valid,
        "test_access": request.get("test_access"),
        "test_access_during_promotion": completion.get("test_access_during_promotion"),
    }


def expected_test_rows(project_root: Path, target: Target) -> tuple[list[str], np.ndarray]:
    if target.dataset == "eurosat":
        frame = pd.read_csv(project_root / EUROSAT_SPLIT_MANIFEST)
        frame = frame.loc[frame["split"] == "test"]
        return frame["sample_id"].astype(str).tolist(), frame["class_index"].to_numpy(dtype=np.int64)
    frame = pd.read_csv(project_root / TREESATAI_MANIFEST)
    frame = frame.loc[frame["split"] == "test"]
    class_index = {name: index for index, name in enumerate(TREESATAI_CLASSES)}
    labels = np.zeros((len(frame), len(TREESATAI_CLASSES)), dtype=np.int64)
    for row_index, encoded in enumerate(frame["labels"]):
        for name in json.loads(encoded):
            labels[row_index, class_index[name]] = 1
    return frame["sample_id"].astype(str).tolist(), labels


def compare_scalar_metrics(actual: Mapping[str, Any], expected: Mapping[str, Any], errors: list[str], context: str) -> None:
    keys = ("accuracy", "macro_f1", "nll", "brier", "ece")
    for key in keys:
        if not finite_number(actual.get(key)) or not finite_number(expected.get(key)):
            errors.append(f"{context} metric {key} is missing or non-finite")
        elif not math.isclose(float(actual[key]), float(expected[key]), rel_tol=1.0e-9, abs_tol=1.0e-12):
            errors.append(f"{context} metric {key} mismatch: {actual[key]} != {expected[key]}")


def validate_predictions(
    project_root: Path,
    run_dir: Path,
    run_id: str,
    target: Target,
    checkpoint: Mapping[str, Any],
    summary: Mapping[str, Any],
    errors: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    export_dir = run_dir / "predictions" / "test" / "deterministic"
    required = (
        "predictions.parquet",
        "embeddings.npz",
        "manifest.json",
        "validation_report.json",
        "metrics.json",
        "per_class_metrics.json",
        "confusion_matrix.csv",
        "confusion_matrix.npy",
    )
    missing = [name for name in required if not (export_dir / name).is_file()]
    if missing:
        errors.append(f"test prediction export is missing files: {missing}")
        return {}, {}
    try:
        validation = validate_prediction_export(export_dir, expected_count=target.expected_test_count)
        bundle = load_prediction_export(export_dir)
        recomputed = recompute_metrics(export_dir, n_bins=15)
    except Exception as exc:
        errors.append(f"prediction export could not be validated: {type(exc).__name__}: {exc}")
        return {}, {}
    if not validation.get("valid"):
        errors.append(f"prediction validator failed: {validation.get('errors')}")
    manifest = bundle.manifest
    source = manifest.get("source", {})
    require_equal(errors, "prediction schema version", manifest.get("schema_version"), 2)
    require_equal(errors, "prediction count", manifest.get("sample_count"), target.expected_test_count)
    require_equal(errors, "prediction expected count", manifest.get("expected_count"), target.expected_test_count)
    require_equal(errors, "prediction partial_export", manifest.get("partial_export"), False)
    require_equal(
        errors,
        "prediction classification type",
        manifest.get("classification_type"),
        "multilabel" if target.dataset == "treesatai" else "multiclass",
    )
    require_equal(errors, "prediction class count", len(manifest.get("class_names", [])), target.expected_class_count)
    require_equal(errors, "prediction source run_id", source.get("run_id"), run_id)
    require_equal(errors, "prediction source seed", source.get("seed"), target.seed)
    require_equal(errors, "prediction source split", source.get("split"), "test")
    require_equal(errors, "prediction UQ method", source.get("uq_method"), "deterministic")
    require_equal(errors, "prediction adaptation", source.get("adaptation_mode"), target.adaptation)
    require_equal(errors, "prediction checkpoint hash", source.get("checkpoint_sha256"), checkpoint.get("sha256"))
    require_equal(errors, "prediction split_total_count", source.get("split_total_count"), target.expected_test_count)
    require_equal(errors, "prediction limit_batches", source.get("limit_batches"), None)

    table = bundle.table
    logits = np.asarray(table["logits"].tolist(), dtype=np.float32)
    probabilities = np.asarray(table["probabilities"].tolist(), dtype=np.float32)
    representations = np.asarray(table["backbone_representation"].tolist(), dtype=np.float32)
    require_equal(errors, "logit shape", list(logits.shape), [target.expected_test_count, target.expected_class_count])
    require_equal(
        errors,
        "probability shape",
        list(probabilities.shape),
        [target.expected_test_count, target.expected_class_count],
    )
    require_equal(
        errors,
        "representation shape",
        list(representations.shape),
        [target.expected_test_count, REPRESENTATION_DIM],
    )
    expected_ids, expected_labels = expected_test_rows(project_root, target)
    actual_ids = table["sample_id"].astype(str).tolist()
    if actual_ids != expected_ids:
        errors.append("prediction sample IDs/order do not exactly match the official test manifest")
    actual_labels = (
        np.asarray(table["label"].tolist(), dtype=np.int64)
        if target.dataset == "treesatai"
        else table["label"].to_numpy(dtype=np.int64)
    )
    if not np.array_equal(actual_labels, expected_labels):
        errors.append("prediction labels do not exactly match the official test manifest")
    if bundle.embeddings is None or bundle.embeddings.get("embeddings", np.empty((0,))).shape != representations.shape:
        errors.append("embedding sidecar shape does not match the Parquet representations")

    saved_metrics = read_json(export_dir / "metrics.json")
    compare_scalar_metrics(saved_metrics, recomputed, errors, "saved/recomputed test")
    summary_test = summary.get("evaluation_metrics", {}).get("test", {})
    compare_scalar_metrics(summary_test, recomputed, errors, "run-summary/recomputed test")
    report_only_metrics = {
        key: recomputed.get(key)
        for key in ("accuracy", "labelwise_accuracy", "macro_f1", "nll", "brier", "ece")
        if key in recomputed
    }
    export_evidence = {
        "path": str(export_dir),
        "validation": validation,
        "sample_count": len(table),
        "logit_shape": list(logits.shape),
        "representation_shape": list(representations.shape),
        "exact_official_sample_order": actual_ids == expected_ids,
        "exact_official_labels": bool(np.array_equal(actual_labels, expected_labels)),
    }
    return export_evidence, report_only_metrics


def audit_run(project_root: Path, run_dir: Path, target: Target) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    required_top_level = (
        "resolved_config.yaml",
        "run_summary.json",
        "training_history.json",
        "training_metrics.csv",
        "batch_metrics.csv",
        "results.json",
        "validation_metrics.json",
        "reliability.png",
    )
    missing = [name for name in required_top_level if not (run_dir / name).is_file()]
    if missing:
        errors.append(f"required run artifacts are missing: {missing}")
    try:
        config = load_yaml(run_dir / "resolved_config.yaml")
        summary = read_json(run_dir / "run_summary.json")
        history = read_json(run_dir / "training_history.json")
    except Exception as exc:
        return {
            "target": asdict(target),
            "target_key": target.key,
            "run_dir": str(run_dir),
            "run_id": run_dir.name,
            "status": "BLOCKED",
            "promotion_eligible": False,
            "promotion_basis": "validation_and_artifact_integrity_only",
            "test_metrics_used_for_promotion": False,
            "errors": [*errors, f"core run metadata could not be read: {type(exc).__name__}: {exc}"],
            "warnings": warnings,
            "test_metrics_report_only": {},
        }

    run_id = str(summary.get("run_id", run_dir.name))
    if run_id != run_dir.name:
        errors.append(f"run_summary run_id {run_id!r} differs from directory name {run_dir.name!r}")
    if summary.get("status") != "completed" or summary.get("dry_run") is not False:
        errors.append("run is not a completed non-dry-run experiment")
    identified = identify_target(config)
    if identified != target:
        errors.append(f"resolved config identifies {identified}, expected {target}")
    errors.extend(protocol_errors(config, target))
    code = validate_code_snapshot(run_dir, errors)
    provenance = validate_provenance_files(project_root, target, errors)
    recorded_provenance = validate_recorded_input_artifacts(run_dir, provenance, errors)
    curve = validate_history(history, summary, config, target, errors)
    epoch_csv = validate_epoch_metrics_csv(run_dir / "training_metrics.csv", history, target, errors)
    batch = validate_batch_metrics(run_dir / "batch_metrics.csv", len(history), target, errors)
    model_gradient = validate_model_and_gradients(run_dir, target, errors)
    checkpoint = validate_checkpoint(run_dir, run_id, target, config, curve, errors)
    resume = validate_resume_provenance(
        run_dir,
        run_id,
        target,
        summary,
        history,
        curve,
        checkpoint,
        errors,
    )
    prediction, test_metrics = validate_predictions(
        project_root, run_dir, run_id, target, checkpoint, summary, errors
    )

    stability_path = run_dir / "training_stability.json"
    if stability_path.is_file():
        stability = read_json(stability_path)
        if stability.get("overfitting_heuristic", {}).get("detected"):
            warnings.append(
                "Post-best overfitting heuristic triggered; this does not invalidate the validation-selected checkpoint."
            )

    return {
        "target": asdict(target),
        "target_key": target.key,
        "run_dir": str(run_dir),
        "run_id": run_id,
        "status": "PROMOTABLE" if not errors else "BLOCKED",
        "promotion_eligible": not errors,
        "promotion_basis": "validation_and_artifact_integrity_only",
        "test_metrics_used_for_promotion": False,
        "errors": errors,
        "warnings": warnings,
        "code_provenance": code,
        "artifact_provenance": provenance,
        "recorded_input_artifacts": recorded_provenance,
        "validation_curve": curve,
        "epoch_metrics_audit": epoch_csv,
        "batch_audit": batch,
        "model_gradient_audit": model_gradient,
        "checkpoint": checkpoint,
        "resume_provenance": resume,
        "prediction_export": prediction,
        "test_metrics_report_only": test_metrics,
    }


def discover_candidates(search_roots: Sequence[Path], targets: Sequence[Target]) -> dict[str, list[Path]]:
    wanted = {target.key: target for target in targets}
    found = {key: [] for key in wanted}
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for config_path in root.rglob("resolved_config.yaml"):
            run_dir = config_path.parent.resolve()
            if run_dir in seen:
                continue
            seen.add(run_dir)
            try:
                config = load_yaml(config_path)
                target = identify_target(config)
            except Exception:
                continue
            if target is not None and target.key in wanted and config.get("dry_run", False) is False:
                found[target.key].append(run_dir)
    for paths in found.values():
        paths.sort()
    return found


def explicit_candidates(run_dirs: Sequence[Path], targets: Sequence[Target]) -> dict[str, list[Path]]:
    wanted = {target.key: target for target in targets}
    found = {key: [] for key in wanted}
    for raw_path in run_dirs:
        run_dir = raw_path.resolve()
        config_path = run_dir / "resolved_config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Explicit run directory lacks resolved_config.yaml: {run_dir}")
        target = identify_target(load_yaml(config_path))
        if target is None or target.key not in wanted:
            raise ValueError(f"Explicit run directory is not a requested C1 target: {run_dir}")
        found[target.key].append(run_dir)
    return found


def audit_targets(
    project_root: Path,
    targets: Sequence[Target],
    candidates: Mapping[str, Sequence[Path]],
) -> list[dict[str, Any]]:
    results = []
    for target in targets:
        paths = list(candidates.get(target.key, []))
        if not paths:
            results.append(
                {
                    "target": asdict(target),
                    "target_key": target.key,
                    "run_dir": None,
                    "run_id": None,
                    "status": "MISSING",
                    "promotion_eligible": False,
                    "promotion_basis": "validation_and_artifact_integrity_only",
                    "test_metrics_used_for_promotion": False,
                    "errors": ["No completed non-dry-run candidate was discovered or supplied."],
                    "warnings": [],
                    "test_metrics_report_only": {},
                }
            )
        elif len(paths) > 1:
            results.append(
                {
                    "target": asdict(target),
                    "target_key": target.key,
                    "run_dir": None,
                    "run_id": None,
                    "status": "AMBIGUOUS",
                    "promotion_eligible": False,
                    "promotion_basis": "validation_and_artifact_integrity_only",
                    "test_metrics_used_for_promotion": False,
                    "errors": ["Multiple candidates exist; supply exactly one with --run-dir."],
                    "warnings": [],
                    "candidate_run_dirs": [str(path) for path in paths],
                    "test_metrics_report_only": {},
                }
            )
        else:
            try:
                results.append(audit_run(project_root, paths[0], target))
            except Exception as exc:
                results.append(
                    {
                        "target": asdict(target),
                        "target_key": target.key,
                        "run_dir": str(paths[0]),
                        "run_id": paths[0].name,
                        "status": "BLOCKED",
                        "promotion_eligible": False,
                        "promotion_basis": "validation_and_artifact_integrity_only",
                        "test_metrics_used_for_promotion": False,
                        "errors": [f"audit utility caught {type(exc).__name__}: {exc}"],
                        "warnings": [],
                        "test_metrics_report_only": {},
                    }
                )
    return results


def csv_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in results:
        target = item["target"]
        curve = item.get("validation_curve", {})
        test = item.get("test_metrics_report_only", {})
        rows.append(
            {
                "task": "classification_multilabel" if target["dataset"] == "treesatai" else "classification",
                "dataset": target["dataset"],
                "model": target["model"],
                "adaptation": target["adaptation"],
                "seed": target["seed"],
                "run_id": item.get("run_id"),
                "run_dir": item.get("run_dir"),
                "status": item["status"],
                "promotion_eligible": item["promotion_eligible"],
                "best_epoch": curve.get("best_epoch"),
                "best_validation_nll": curve.get("best_validation_nll"),
                "best_validation_accuracy": curve.get("best_validation_accuracy"),
                "best_validation_macro_f1": curve.get("best_validation_macro_f1"),
                "reported_test_accuracy_excluded": test.get("accuracy"),
                "reported_test_macro_f1_excluded": test.get("macro_f1"),
                "reported_test_nll_excluded": test.get("nll"),
                "reported_test_brier_excluded": test.get("brier"),
                "reported_test_ece_excluded": test.get("ece"),
                "error_count": len(item.get("errors", [])),
                "errors": " | ".join(item.get("errors", [])),
            }
        )
    return rows


def markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# C1 deterministic classification run audit",
        "",
        f"Created: {payload['created_at_utc']}",
        "",
        "Promotion is based only on the frozen protocol, validation behavior, provenance, checkpoint integrity, "
        "and prediction-artifact integrity. Test metric values are reported for completeness and are explicitly "
        "excluded from every promotion decision.",
        "",
        f"Summary: **{payload['summary']['promotable']} promotable**, "
        f"**{payload['summary']['blocked']} blocked**, **{payload['summary']['missing']} missing**, "
        f"**{payload['summary']['ambiguous']} ambiguous**.",
        "",
        "| Dataset | Model | Adaptation | Seed | Run ID | Status | Best val NLL | Test accuracy (excluded) | Errors |",
        "|---|---|---|---:|---|---|---:|---:|---:|",
    ]
    for item in payload["runs"]:
        target = item["target"]
        curve = item.get("validation_curve", {})
        test = item.get("test_metrics_report_only", {})
        best_nll = curve.get("best_validation_nll")
        test_accuracy = test.get("accuracy")
        lines.append(
            "| {dataset} | {model} | {adaptation} | {seed} | `{run_id}` | **{status}** | {best} | {test} | {errors} |".format(
                dataset=target["dataset"],
                model=target["model"],
                adaptation=target["adaptation"],
                seed=target["seed"],
                run_id=item.get("run_id") or "—",
                status=item["status"],
                best=f"{best_nll:.8f}" if isinstance(best_nll, (int, float)) else "—",
                test=f"{test_accuracy:.8f}" if isinstance(test_accuracy, (int, float)) else "—",
                errors=len(item.get("errors", [])),
            )
        )
    blocked = [item for item in payload["runs"] if item.get("errors")]
    if blocked:
        lines.extend(["", "## Non-promotable targets", ""])
        for item in blocked:
            lines.append(f"### {item['target_key']}")
            lines.append("")
            for error in item["errors"]:
                lines.append(f"- {error}")
            lines.append("")
    lines.extend(
        [
            "## Decision rule",
            "",
            "A run is promotable only when every structural check passes. No minimum accuracy, Macro-F1, NLL, "
            "Brier, or ECE threshold is implemented for test data, and test values cannot select runs, seeds, "
            "checkpoints, or learning rates.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(output_dir: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "c1_run_audit.json"
    csv_path = output_dir / "c1_run_audit.csv"
    markdown_path = output_dir / "c1_run_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = csv_rows(payload["runs"])
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(markdown_report(payload), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(markdown_path)}


def default_output_dir(project_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = project_root / "reports" / "c1_classification_run_audits" / stamp
    counter = 1
    while candidate.exists():
        candidate = candidate.with_name(f"{stamp}_{counter}")
        counter += 1
    return candidate


def build_payload(project_root: Path, seeds: Sequence[int], candidates: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
    targets = expected_targets(seeds)
    runs = audit_targets(project_root, targets, candidates)
    counts = {name: sum(item["status"] == name.upper() for item in runs) for name in ("promotable", "blocked", "missing", "ambiguous")}
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "requested_seeds": list(seeds),
        "promotion_policy": {
            "basis": "validation_and_artifact_integrity_only",
            "test_metrics_used": False,
            "test_metrics_role": "report_only",
        },
        "summary": {**counts, "total": len(runs)},
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only audit of C1 deterministic classification final runs.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Explicit completed run directory. Repeat to avoid discovery ambiguity.",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Discovery root. Defaults to <project>/results/baselines and "
            "<project>/results/final_thesis/classification."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fail-on-nonpromotable", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    targets = expected_targets(args.seeds)
    if args.run_dir:
        candidates = explicit_candidates(args.run_dir, targets)
    else:
        search_roots = [resolve_project_path(project_root, path) for path in args.search_root]
        if not search_roots:
            search_roots = [
                project_root / "results" / "baselines",
                project_root / "results" / "final_thesis" / "classification",
            ]
        candidates = discover_candidates(search_roots, targets)
    payload = build_payload(project_root, args.seeds, candidates)
    output_dir = args.output_dir or default_output_dir(project_root)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    paths = write_outputs(output_dir, payload)
    print(json.dumps({"summary": payload["summary"], "outputs": paths}, indent=2))
    if args.fail_on_nonpromotable and payload["summary"]["promotable"] != payload["summary"]["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
