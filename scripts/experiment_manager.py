from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CODE_SNAPSHOT_SUFFIXES = frozenset({".cfg", ".ini", ".json", ".py", ".sh", ".toml", ".yaml", ".yml"})
CODE_SNAPSHOT_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".git.backup",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
        "checkpoints",
        "data",
        "datasets",
        "outputs",
        "reports",
        "results",
    }
)


def set_global_seed(seed: int, deterministic: bool = True, warn_only: bool = False) -> torch.Generator:
    """Seed all project RNGs and return a seeded DataLoader generator."""
    numpy_seed = seed % (2**32)
    torch_seed = seed % (2**63)
    os.environ["PYTHONHASHSEED"] = str(numpy_seed)
    random.seed(seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    generator = torch.Generator()
    generator.manual_seed(torch_seed)
    return generator


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python and NumPy from the worker seed assigned by PyTorch."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_run_id(experiment_name: str, seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    entropy = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
    safe_name = "".join(char if char.isalnum() or char in "-_" else "-" for char in experiment_name)
    return f"{timestamp}_{safe_name}_seed{seed}_{entropy}"


def make_invocation_summary_path(output_root: Path, seeds: list[int]) -> Path:
    """Return a unique summary path without creating or replacing any file."""
    if not seeds:
        raise ValueError("At least one seed is required for an invocation summary")
    return output_root / "summaries" / f"{make_run_id('summary', seeds[0])}.json"


def resolve_config(config: Dict[str, Any], config_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Copy, validate, and normalize a run configuration."""
    resolved = copy.deepcopy(config)
    resolved["config_path"] = str(config_path.resolve())
    resolved["project_root"] = str(PROJECT_ROOT)
    benchmark_lock_path = PROJECT_ROOT / "environment" / "geobench2.lock.yaml"
    if benchmark_lock_path.is_file():
        with benchmark_lock_path.open("r", encoding="utf-8") as handle:
            benchmark_lock = yaml.safe_load(handle)
        resolved["benchmark_lock"] = {
            "path": str(benchmark_lock_path.resolve()),
            **benchmark_lock["benchmark"],
        }

    required_sections = {"data", "training", "metrics", "uq_methods", "experiments"}
    missing = sorted(required_sections - resolved.keys())
    if missing:
        raise ValueError(f"Missing required configuration sections: {', '.join(missing)}")

    configured_seed = resolved.get("seed")
    configured_seeds = resolved.get("seeds")
    if configured_seed is None and configured_seeds is None:
        raise ValueError("Configure seed as an integer or seeds as a non-empty list of integers")
    if configured_seed is not None and (not isinstance(configured_seed, int) or isinstance(configured_seed, bool)):
        raise ValueError("seed must be an integer")
    if configured_seeds is not None:
        if not isinstance(configured_seeds, list) or not configured_seeds:
            raise ValueError("seeds must be a non-empty list")
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in configured_seeds):
            raise ValueError("every seeds entry must be an integer")
        if len(set(configured_seeds)) != len(configured_seeds):
            raise ValueError("seeds must not contain duplicates")
        if configured_seed is None:
            resolved["seed"] = configured_seeds[0]

    data = resolved["data"]
    wavelengths_nm = data.get("wavelengths_nm")
    if wavelengths_nm is not None:
        if not isinstance(wavelengths_nm, list) or not wavelengths_nm:
            raise ValueError("data.wavelengths_nm must be a non-empty list")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in wavelengths_nm):
            raise ValueError("every data.wavelengths_nm entry must be numeric")
        channels = data.get("channels")
        if channels is not None and len(wavelengths_nm) != len(channels):
            raise ValueError("data.wavelengths_nm and data.channels must have the same length")
        if any(not 300.0 <= float(value) <= 2500.0 for value in wavelengths_nm):
            raise ValueError(
                "data.wavelengths_nm must use canonical optical nanometers (expected values in [300, 2500]); "
                "micrometer-scale values indicate a unit error"
            )
        data["wavelengths_nm"] = [float(value) for value in wavelengths_nm]
    if "wavelengths" in data or "wavelength_units" in data:
        raise ValueError(
            "Use canonical data.wavelengths_nm only; legacy data.wavelengths/data.wavelength_units are forbidden "
            "for new experiments"
        )
    normalization = data.get("normalization")
    if normalization is not None:
        if not isinstance(normalization, dict):
            raise ValueError("data.normalization must be a mapping")
        if normalization.get("method", "channelwise_standardization") != "channelwise_standardization":
            raise ValueError("Only channelwise_standardization is supported for data.normalization.method")
        means = normalization.get("mean")
        stds = normalization.get("std")
        if not isinstance(means, list) or not isinstance(stds, list) or not means or len(means) != len(stds):
            raise ValueError("data.normalization.mean/std must be non-empty lists of equal length")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in means + stds):
            raise ValueError("data.normalization.mean/std values must be numeric")
        if any(float(value) <= 0 for value in stds):
            raise ValueError("data.normalization.std values must be positive")
        channels = data.get("channels")
        if channels is not None and len(means) != len(channels):
            raise ValueError("data.normalization mean/std and data.channels must have the same length")
        if normalization.get("statistics_split", "train") != "train":
            raise ValueError("Normalization statistics must be computed from the train split only")

    training = resolved["training"]
    training.setdefault("optimizer", {"name": "adamw"})
    training.setdefault("early_stopping", {"enabled": False, "patience": 0, "min_delta": 0.0})
    training.setdefault("checkpoint", {"metric": "nll", "mode": "min"})
    training.setdefault("learning_rate", 1e-4)
    training.setdefault("backbone_learning_rate", training["learning_rate"])
    training.setdefault("head_learning_rate", training["learning_rate"])
    training.setdefault("warmup", {"enabled": False, "epochs": 0, "type": "linear"})
    training.setdefault("gradient_clipping", {"enabled": False})
    training.setdefault("layerwise_lr_decay", {"enabled": False})
    resolved.setdefault("reproducibility", {"deterministic": True})

    dataset_name = str(resolved["data"].get("name", "eurosat")).lower()
    task_name = str(resolved.get("task", "classification"))
    if dataset_name == "treesatai":
        if task_name != "classification_multilabel" or not bool(resolved["data"].get("multilabel", False)):
            raise ValueError("TreeSatAI requires task=classification_multilabel and data.multilabel=true")
        if int(resolved["data"].get("num_classes", 0)) != 15:
            raise ValueError("TreeSatAI requires the official 15-class multi-label target contract")
    elif dataset_name == "eurosat" and task_name != "classification":
        raise ValueError("EuroSAT requires task=classification")
    elif dataset_name in {"cloudsen12", "spacenet7"}:
        if task_name != "segmentation":
            raise ValueError(f"{dataset_name} requires task=segmentation")
        expected_classes = 4 if dataset_name == "cloudsen12" else 2
        if int(resolved["data"].get("num_classes", 0)) != expected_classes:
            raise ValueError(f"{dataset_name} requires num_classes={expected_classes}")
        class_names = resolved["data"].get("class_names")
        if not isinstance(class_names, list) or len(class_names) != expected_classes:
            raise ValueError(f"{dataset_name} requires {expected_classes} ordered class_names")
        if dataset_name == "cloudsen12" and resolved["data"].get("ignore_index") is not None:
            raise ValueError("CloudSEN12 has no ignore index in the official artifact")
        if dataset_name == "spacenet7":
            if int(resolved["data"].get("ignore_index", -1)) != 255:
                raise ValueError("SpaceNet7 thesis mapping requires ignore_index=255")
            if int(resolved["data"].get("foreground_class_index", -1)) != 1:
                raise ValueError("SpaceNet7 building foreground_class_index must be 1")
    split_manifest = resolved["data"].get("split_manifest")
    split_files = resolved["data"].get("split_files", {})
    if dataset_name in {"so2sat", "treesatai", "cloudsen12", "spacenet7"}:
        if split_manifest or split_files:
            raise ValueError(
                f"{dataset_name} uses the official GEO-Bench-2 embedded train/validation/test partition; "
                "do not configure a local split manifest or split files"
            )
        forbidden_random_split_keys = sorted(
            key for key in ("val_fraction", "test_fraction", "split_seed", "random_split") if key in resolved["data"]
        )
        if forbidden_random_split_keys:
            raise ValueError(
                f"{dataset_name} random split configuration is forbidden when using the official GEO-Bench-2 protocol: "
                f"{forbidden_random_split_keys}"
            )
        expected_protocols = {
            "so2sat": "geobench2_m-so2sat",
            "treesatai": "geobench2_treesatai",
            "cloudsen12": "geobench2_cloudsen12",
            "spacenet7": "geobench2_spacenet7_static",
        }
        if resolved["data"].get("protocol") != expected_protocols[dataset_name]:
            raise ValueError(f"{dataset_name} data.protocol must be {expected_protocols[dataset_name]}")
        if "expected_split_counts" in resolved["data"]:
            raise ValueError(
                f"Do not hard-code {dataset_name} split counts; inspect and record the downloaded GEO-Bench-2 manifest"
            )
    elif not split_manifest:
        for split in ("train", "val"):
            if split not in split_files:
                raise ValueError(f"data.split_manifest or data.split_files.{split} must be configured")
    augmentations = resolved["data"].get("augmentations", {})
    for split in ("calibration", "test"):
        if augmentations.get(split, "none") not in (None, False, "none"):
            raise ValueError(f"Random augmentation is forbidden for the {split} split")
    if training["checkpoint"].get("split", "val") != "val":
        raise ValueError("Checkpoint selection is restricted to the validation split")
    if training["checkpoint"].get("metric") not in {
        "loss",
        "accuracy",
        "nll",
        "ece",
        "brier",
        "miou",
        "pixel_accuracy",
    }:
        raise ValueError("Unsupported checkpoint metric")
    if training["checkpoint"].get("mode") not in {"min", "max"}:
        raise ValueError("training.checkpoint.mode must be 'min' or 'max'")
    warmup = training.get("warmup", {})
    if warmup.get("enabled", False):
        if int(warmup.get("epochs", 0)) <= 0:
            raise ValueError("Enabled warm-up requires a positive epoch count")
        if warmup.get("type", "linear") != "linear":
            raise ValueError("Only linear warm-up is currently supported")
    if training.get("gradient_clipping", {}).get("enabled", False):
        raise ValueError("Gradient clipping is not implemented by this training loop; set enabled: false")
    if training.get("layerwise_lr_decay", {}).get("enabled", False):
        raise ValueError("Layer-wise LR decay is not implemented by this training loop; set enabled: false")

    prediction_export = resolved.setdefault("prediction_export", {"enabled": False})
    if bool(prediction_export.get("enabled", False)):
        # Schema v2 always materializes the representation in both Parquet and NPZ.
        prediction_export["save_backbone_representation"] = True
        prediction_export["save_embeddings"] = True
    metrics = resolved["metrics"]
    if task_name == "classification_multilabel":
        threshold = float(metrics.setdefault("multilabel_threshold", 0.5))
        if not 0.0 < threshold < 1.0:
            raise ValueError("metrics.multilabel_threshold must be strictly between zero and one")

    for experiment in resolved["experiments"]:
        model = experiment["model"]
        allowed_models = {"dofa", "panopticon"} if task_name == "segmentation" else {"dofa", "panopticon", "resnet18"}
        if model.get("name") not in allowed_models:
            raise ValueError(f"model.name must be one of {sorted(allowed_models)}")
        model.setdefault("adaptation_mode", "frozen" if model.get("freeze_backbone", True) else "full_finetune")
        if model["adaptation_mode"] not in {"frozen", "full_finetune"}:
            raise ValueError("model.adaptation_mode must be frozen or full_finetune")
        model["freeze_backbone"] = model["adaptation_mode"] == "frozen"
        if task_name == "segmentation":
            decoder = model.setdefault("decoder", {"architecture": "unet", "channels": [256, 128, 64, 32]})
            if decoder.get("architecture", "unet") != "unet":
                raise ValueError("The common segmentation protocol requires model.decoder.architecture=unet")
            channels = decoder.get("channels", [256, 128, 64, 32])
            if not isinstance(channels, list) or not channels or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in channels
            ):
                raise ValueError("model.decoder.channels must be a non-empty list of positive integers")
            decoder["architecture"] = "unet"
            decoder["channels"] = channels
            dropout = float(decoder.setdefault("dropout", 0.0))
            if not 0.0 <= dropout < 1.0:
                raise ValueError("model.decoder.dropout must satisfy 0 <= p < 1")
        else:
            model.setdefault("head", {"architecture": "linear", "dropout": 0.0})
            model["head"].setdefault("architecture", "linear")
            model["head"].setdefault("dropout", 0.0)
            dropout = float(model["head"]["dropout"])
            if not 0.0 <= dropout < 1.0:
                raise ValueError("model.head.dropout must satisfy 0 <= p < 1")
            if model["head"].get("architecture") not in {"linear", "mlp", "batchnorm_linear"}:
                raise ValueError("model.head.architecture must be linear, mlp, or batchnorm_linear")
        if model.get("name") == "panopticon":
            if model.get("size", "base") != "base":
                raise ValueError("Panopticon currently supports model.size: base only")
            if not wavelengths_nm:
                raise ValueError("Panopticon requires explicit canonical data.wavelengths_nm")
        if model.get("name") in {"dofa", "panopticon"}:
            if not wavelengths_nm:
                raise ValueError(f"{model['name']} requires explicit canonical data.wavelengths_nm")
            if model["name"] == "dofa":
                actual_wavelengths = [float(value) / 1000.0 for value in wavelengths_nm]
                actual_units = "micrometers"
                if any(not 0.3 <= value <= 2.5 for value in actual_wavelengths):
                    raise ValueError("DOFA wavelengths must be converted from nm to optical micrometers")
            else:
                actual_wavelengths = [float(value) for value in wavelengths_nm]
                actual_units = "nanometers"
                if any(not 300.0 <= value <= 2500.0 for value in actual_wavelengths):
                    raise ValueError("Panopticon channel IDs must remain in optical nanometers")
            model["actual_wavelengths"] = actual_wavelengths
            model["actual_wavelength_units"] = actual_units

    if task_name == "segmentation":
        foundation_decoders = {
            json.dumps(experiment["model"]["decoder"], sort_keys=True)
            for experiment in resolved["experiments"]
        }
        if len(foundation_decoders) > 1:
            raise ValueError("DOFA and Panopticon must use the same downstream segmentation decoder")
    else:
        foundation_heads = {
            json.dumps(experiment["model"]["head"], sort_keys=True)
            for experiment in resolved["experiments"]
            if experiment["model"].get("name") in {"dofa", "panopticon"}
        }
        if len(foundation_heads) > 1:
            raise ValueError("DOFA and Panopticon must use the same downstream classification-head configuration")

    resolved["dry_run"] = bool(dry_run)
    if dry_run:
        dry = resolved.get("dry_run_settings", {})
        training["epochs"] = int(dry.get("epochs", 1))
        training["limit_train_batches"] = int(dry.get("limit_train_batches", 1))
        training["limit_val_batches"] = int(dry.get("limit_val_batches", 1))
        resolved["data"]["num_workers"] = int(dry.get("num_workers", 0))
        resolved["visualization"] = {**resolved.get("visualization", {}), "enabled": False}
        normal_output = Path(resolved.get("output_dir", "./results/experiments"))
        resolved["output_dir"] = str(Path(dry.get("output_dir", normal_output / "dry_runs")))
        resolved["results_csv"] = str(Path(resolved["output_dir"]) / "results.csv")
    return resolved


def write_resolved_config(config: Dict[str, Any], run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=False)
    path = run_dir / "resolved_config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return path


def _command_result(command: list[str]) -> tuple[bool, str | None]:
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return True, result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return False, None


def _command_output(command: list[str]) -> str | None:
    return _command_result(command)[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_code_snapshot_files(project_root: Path) -> Iterable[Path]:
    for current_root, directory_names, file_names in os.walk(project_root):
        directory_names[:] = sorted(
            name for name in directory_names if name not in CODE_SNAPSHOT_EXCLUDED_DIRECTORIES
        )
        current_path = Path(current_root)
        for file_name in sorted(file_names):
            path = current_path / file_name
            if path.is_symlink():
                continue
            if path.suffix.lower() in CODE_SNAPSHOT_SUFFIXES or file_name in {"Dockerfile", "Makefile"}:
                yield path


def collect_code_snapshot(project_root: str | Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Create deterministic, content-addressed metadata for research code and configs."""
    root = Path(project_root).resolve()
    files = []
    aggregate = hashlib.sha256()
    for path in _iter_code_snapshot_files(root):
        relative_path = path.relative_to(root).as_posix()
        file_hash = _sha256_file(path)
        size_bytes = path.stat().st_size
        files.append({"path": relative_path, "sha256": file_hash, "size_bytes": size_bytes})
        aggregate.update(relative_path.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(file_hash.encode("ascii"))
        aggregate.update(b"\n")
    if not files:
        raise RuntimeError(f"No code/config files found for provenance snapshot under {root}")
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "code_sha256": aggregate.hexdigest(),
        "file_count": len(files),
        "files": files,
        "excluded_directory_names": sorted(CODE_SNAPSHOT_EXCLUDED_DIRECTORIES),
    }


def write_code_snapshot_manifest(run_dir: str | Path, project_root: str | Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Write the immutable code manifest for a newly created run directory."""
    run_dir = Path(run_dir)
    snapshot = collect_code_snapshot(project_root)
    snapshot["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = run_dir / "code_snapshot.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, ensure_ascii=False)
    return {
        "code_sha256": snapshot["code_sha256"],
        "file_count": snapshot["file_count"],
        "manifest_path": str(path.resolve()),
    }


def collect_environment_metadata(code_snapshot: Dict[str, Any] | None = None) -> Dict[str, Any]:
    packages = {}
    for package in (
        "numpy",
        "torch",
        "torchvision",
        "torchgeo",
        "timm",
        "PyYAML",
        "matplotlib",
        "GeoBenchV2",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "capability": [properties.major, properties.minor],
                }
            )
    git_commit_ok, git_commit = _command_result(["git", "rev-parse", "HEAD"])
    git_status_ok, git_status = _command_result(["git", "status", "--porcelain"])
    geobench_direct_url = None
    try:
        distribution = importlib.metadata.distribution("GeoBenchV2")
        direct_url_file = next(
            (file for file in (distribution.files or []) if str(file).endswith("direct_url.json")),
            None,
        )
        if direct_url_file is not None:
            direct_url_path = Path(distribution.locate_file(direct_url_file))
            geobench_direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError, json.JSONDecodeError):
        geobench_direct_url = None
    benchmark_lock_path = PROJECT_ROOT / "environment" / "geobench2.lock.yaml"
    benchmark_lock = None
    if benchmark_lock_path.is_file():
        with benchmark_lock_path.open("r", encoding="utf-8") as handle:
            benchmark_lock = yaml.safe_load(handle).get("benchmark")
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "gpus": gpus,
        "packages": packages,
        "git_available": bool(git_commit_ok and git_status_ok),
        "git_commit": git_commit if git_commit_ok else None,
        "git_dirty": bool(git_status) if git_status_ok else None,
        "geobench2": {
            "lock": benchmark_lock,
            "installed_direct_url": geobench_direct_url,
        },
    }
    if code_snapshot is not None:
        metadata["code_version"] = f"sha256:{code_snapshot['code_sha256']}"
        metadata["code_snapshot"] = dict(code_snapshot)
    return metadata


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
