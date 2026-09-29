from __future__ import annotations

"""Real one-batch preflight for the 16-cell deterministic thesis matrix."""

import argparse
import copy
import gc
import hashlib
import json
import math
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_manager import resolve_config, set_global_seed
from scripts.prediction_export import (
    classification_embedding_extractor,
    export_predictions,
    validate_prediction_export,
)
from scripts.run_experiments import (
    audit_model_for_training,
    build_model,
    build_optimizer,
    classification_loss,
    classification_type_from_config,
    compute_task_classification_metrics,
    forward_classification_batch,
    load_config,
    make_dataloaders,
    model_display_name,
    move_batch,
    select_experiments,
)
from scripts.segmentation_pipeline import (
    audit_segmentation_model,
    build_segmentation_model,
    export_segmentation_predictions,
    make_segmentation_dataloaders,
    move_segmentation_batch,
    segmentation_loss,
    segmentation_metrics,
    validate_segmentation_prediction_export,
)


CLASSIFICATION_CELLS = (
    ("EuroSAT", "DOFA", "frozen", "configs/eurosat_dofa_frozen_baseline.yaml", "eurosat_dofa_frozen_bnlinear"),
    ("EuroSAT", "DOFA", "full_finetune", "configs/eurosat_dofa_full_finetune.yaml", "eurosat_dofa_full_finetune"),
    ("EuroSAT", "Panopticon", "frozen", "configs/eurosat_panopticon_frozen_baseline.yaml", "eurosat_panopticon_frozen_bnlinear"),
    ("EuroSAT", "Panopticon", "full_finetune", "configs/eurosat_panopticon_full_finetune.yaml", "eurosat_panopticon_full_finetune"),
    ("TreeSatAI", "DOFA", "frozen", "configs/treesatai_classification.yaml", "treesatai_dofa_frozen"),
    ("TreeSatAI", "DOFA", "full_finetune", "configs/treesatai_classification.yaml", "treesatai_dofa_full_finetune"),
    ("TreeSatAI", "Panopticon", "frozen", "configs/treesatai_classification.yaml", "treesatai_panopticon_frozen"),
    ("TreeSatAI", "Panopticon", "full_finetune", "configs/treesatai_classification.yaml", "treesatai_panopticon_full_finetune"),
)

SEGMENTATION_CELLS = (
    ("CloudSEN12", "DOFA", "frozen", "configs/cloudsen12_segmentation.yaml", "cloudsen12_dofa_frozen"),
    ("CloudSEN12", "DOFA", "full_finetune", "configs/cloudsen12_segmentation.yaml", "cloudsen12_dofa_full_finetune"),
    ("CloudSEN12", "Panopticon", "frozen", "configs/cloudsen12_segmentation.yaml", "cloudsen12_panopticon_frozen"),
    ("CloudSEN12", "Panopticon", "full_finetune", "configs/cloudsen12_segmentation.yaml", "cloudsen12_panopticon_full_finetune"),
    ("SpaceNet7", "DOFA", "frozen", "configs/spacenet7_segmentation.yaml", "spacenet7_dofa_frozen"),
    ("SpaceNet7", "DOFA", "full_finetune", "configs/spacenet7_segmentation.yaml", "spacenet7_dofa_full_finetune"),
    ("SpaceNet7", "Panopticon", "frozen", "configs/spacenet7_segmentation.yaml", "spacenet7_panopticon_frozen"),
    ("SpaceNet7", "Panopticon", "full_finetune", "configs/spacenet7_segmentation.yaml", "spacenet7_panopticon_full_finetune"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _gradient_audit(model: torch.nn.Module, adaptation: str) -> Dict[str, Any]:
    groups = {
        "backbone": list(model.backbone.named_parameters()),
        "head": list(model.head.named_parameters()),
    }
    summary: Dict[str, Any] = {}
    for group_name, parameters in groups.items():
        trainable = [(name, parameter) for name, parameter in parameters if parameter.requires_grad]
        frozen = [(name, parameter) for name, parameter in parameters if not parameter.requires_grad]
        missing = [name for name, parameter in trainable if parameter.grad is None]
        nonfinite = [
            name
            for name, parameter in trainable
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        zero = [
            name
            for name, parameter in trainable
            if parameter.grad is not None and int(torch.count_nonzero(parameter.grad).item()) == 0
        ]
        frozen_with_gradient = [name for name, parameter in frozen if parameter.grad is not None]
        summary[group_name] = {
            "trainable_parameter_tensors": len(trainable),
            "frozen_parameter_tensors": len(frozen),
            "missing_gradient_parameters": missing,
            "nonfinite_gradient_parameters": nonfinite,
            "zero_gradient_parameters": zero,
            "frozen_parameters_with_gradient": frozen_with_gradient,
        }
    backbone = summary["backbone"]
    head = summary["head"]
    if adaptation == "frozen":
        valid = (
            backbone["trainable_parameter_tensors"] == 0
            and not backbone["frozen_parameters_with_gradient"]
            and head["trainable_parameter_tensors"] > 0
            and not head["missing_gradient_parameters"]
            and not head["nonfinite_gradient_parameters"]
            and not head["zero_gradient_parameters"]
        )
    else:
        valid = (
            backbone["trainable_parameter_tensors"] > 0
            and not backbone["missing_gradient_parameters"]
            and not backbone["nonfinite_gradient_parameters"]
            and not backbone["zero_gradient_parameters"]
            and not backbone["frozen_parameters_with_gradient"]
            and head["trainable_parameter_tensors"] > 0
            and not head["missing_gradient_parameters"]
            and not head["nonfinite_gradient_parameters"]
            and not head["zero_gradient_parameters"]
        )
    summary["valid"] = valid
    return summary


def _checkpoint_roundtrip(
    cell_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="checkpoint_roundtrip_", dir=cell_dir) as directory:
        checkpoint_path = Path(directory) / "smoke.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": 1,
                "preflight": metadata,
            },
            checkpoint_path,
        )
        size = checkpoint_path.stat().st_size
        digest = _sha256(checkpoint_path)
        try:
            loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            loaded = torch.load(checkpoint_path, map_location="cpu")
        incompatible = model.load_state_dict(loaded["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Checkpoint strict round-trip failed: {incompatible}")
        if int(loaded["epoch"]) != 1 or not digest or size <= 0:
            raise RuntimeError("Checkpoint round-trip metadata or file is invalid")
        return {
            "valid": True,
            "bytes": size,
            "sha256": digest,
            "strict_missing_keys": list(incompatible.missing_keys),
            "strict_unexpected_keys": list(incompatible.unexpected_keys),
            "retained": False,
        }


def _classification_metadata_check(config: Dict[str, Any], batch: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    data = config["data"]
    channels = list(data["channels"])
    wavelengths = list(data["wavelengths_nm"])
    images = batch["image"]
    ids = [str(value) for value in batch["sample_id"]]
    valid = len(channels) == len(wavelengths) and len(ids) == len(set(ids)) and torch.isfinite(images).all()
    if dataset == "TreeSatAI":
        valid = bool(
            valid
            and images.ndim == 5
            and images.shape[1] >= 1
            and images.shape[2] == len(channels)
            and batch["label"].shape == (images.shape[0], int(data["num_classes"]))
            and batch["temporal_mask"].shape == images.shape[:2]
            and batch["temporal_mask"].any(dim=1).all()
        )
    else:
        valid = bool(
            valid
            and images.ndim == 4
            and images.shape[1] == len(channels)
            and batch["label"].shape == (images.shape[0],)
        )
    if not valid:
        raise RuntimeError(f"Classification dataloader/metadata validation failed for {dataset}")
    return {
        "valid": True,
        "batch_image_shape": list(images.shape),
        "batch_label_shape": list(batch["label"].shape),
        "sample_ids": ids,
        "channels": channels,
        "wavelengths_nm": wavelengths,
        "temporal_mask_shape": list(batch["temporal_mask"].shape) if "temporal_mask" in batch else None,
    }


def _segmentation_metadata_check(config: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
    data = config["data"]
    images = batch["image"]
    masks = batch["mask"]
    channels = list(data["channels"])
    wavelengths = list(data["wavelengths_nm"])
    ids = [str(value) for value in batch["sample_id"]]
    ignore = data.get("ignore_index")
    valid_mask = torch.ones_like(masks, dtype=torch.bool) if ignore is None else masks.ne(int(ignore))
    invalid = valid_mask & ((masks < 0) | (masks >= int(data["num_classes"])))
    valid = bool(
        images.ndim == 4
        and images.shape[1] == len(channels) == len(wavelengths)
        and masks.shape == (images.shape[0], *images.shape[-2:])
        and len(ids) == len(set(ids))
        and torch.isfinite(images).all()
        and valid_mask.any()
        and not invalid.any()
    )
    if not valid:
        raise RuntimeError(f"Segmentation dataloader/metadata validation failed for {data['name']}")
    return {
        "valid": True,
        "batch_image_shape": list(images.shape),
        "batch_mask_shape": list(masks.shape),
        "sample_ids": ids,
        "channels": channels,
        "wavelengths_nm": wavelengths,
        "mask_values": sorted(int(value) for value in torch.unique(masks).tolist()),
        "valid_pixels": int(valid_mask.sum().item()),
        "ignored_pixels": int((~valid_mask).sum().item()),
    }


def _classification_cell(
    dataset: str,
    model_label: str,
    adaptation: str,
    config_path: Path,
    experiment_name: str,
    cell_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    config = resolve_config(load_config(config_path), config_path, dry_run=False)
    config = copy.deepcopy(config)
    config["seed"] = 20260812
    config["training"]["batch_size"] = 2
    config["data"]["num_workers"] = 0
    experiment = select_experiments(config, [experiment_name])[0]
    reproducibility = config.get("reproducibility", {})
    generator = set_global_seed(
        config["seed"],
        deterministic=bool(reproducibility.get("deterministic", True)),
        warn_only=bool(reproducibility.get("warn_only", False)),
    )
    loaders = make_dataloaders(config, generator=generator)
    train_batch_raw = next(iter(loaders["train"]))
    metadata = _classification_metadata_check(config, train_batch_raw, dataset)
    model = build_model(config, experiment["model"], int(config["data"]["num_classes"])).to(device)
    model_structure = audit_model_for_training(model, experiment["model"])
    optimizer = build_optimizer(model, config["training"])
    classification_type = classification_type_from_config(config)
    threshold = float(config.get("metrics", {}).get("multilabel_threshold", 0.5))

    model.train()
    if model.freeze_backbone:
        model.backbone.eval()
    train_batch = move_batch(train_batch_raw, device)
    optimizer.zero_grad(set_to_none=True)
    train_logits = forward_classification_batch(model, train_batch)
    if not torch.isfinite(train_logits).all():
        raise FloatingPointError("Classification forward produced non-finite logits")
    loss = classification_loss(train_logits, train_batch["label"], classification_type)
    if not torch.isfinite(loss):
        raise FloatingPointError("Classification loss is non-finite")
    loss.backward()
    gradients = _gradient_audit(model, adaptation)
    if not gradients["valid"]:
        raise RuntimeError(f"Classification gradient audit failed: {gradients}")
    optimizer.step()

    model.eval()
    validation_raw = next(iter(loaders["val"]))
    _classification_metadata_check(config, validation_raw, dataset)
    validation = move_batch(validation_raw, device)
    temporal_mask = validation.get("temporal_mask")
    extractor = classification_embedding_extractor(model)
    with torch.no_grad():
        if temporal_mask is not None:
            representations = extractor(validation["image"], temporal_mask=temporal_mask)
        else:
            representations = extractor(validation["image"])
        validation_logits = model.head(representations)
    metrics = compute_task_classification_metrics(
        validation_logits.cpu(),
        validation["label"].cpu(),
        classification_type,
        n_bins=15,
        threshold=threshold,
    )
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise RuntimeError(f"Classification validation metrics are non-finite: {metrics}")

    checkpoint = _checkpoint_roundtrip(
        cell_dir,
        model,
        optimizer,
        {"dataset": dataset, "model": model_label, "adaptation": adaptation},
    )
    export_dir = cell_dir / "prediction_export"
    export_predictions(
        export_dir,
        sample_ids=[str(value) for value in validation_raw["sample_id"]],
        true_labels=validation["label"].cpu(),
        logits=validation_logits.cpu(),
        class_names=config["data"]["class_names"],
        model_name=model_display_name(experiment["model"]),
        dataset=str(config["data"]["name"]),
        adaptation_mode=adaptation,
        seed=config["seed"],
        checkpoint="temporary_preflight_checkpoint_not_retained",
        split="val_tiny_preflight",
        expected_count=len(validation_raw["sample_id"]),
        embeddings=representations.cpu(),
        classification_type=classification_type,
        multilabel_threshold=threshold,
        source={"preflight": True, "split_total_count": len(validation_raw["sample_id"])},
    )
    export_validation = validate_prediction_export(export_dir, expected_count=len(validation_raw["sample_id"]))
    if not export_validation["valid"]:
        raise RuntimeError(f"Classification prediction export failed: {export_validation}")
    return {
        "dataloader": {"valid": True, "split_sizes": {name: len(loader.dataset) for name, loader in loaders.items()}},
        "metadata": metadata,
        "forward": {"valid": True, "logits_shape": list(train_logits.shape), "finite": True},
        "loss": {"valid": True, "name": "cross_entropy" if classification_type == "multiclass" else "bce_with_logits", "value": float(loss.item())},
        "gradients": gradients,
        "validation_metric": {"valid": True, **metrics},
        "checkpoint_writing": checkpoint,
        "prediction_export": {"valid": True, "path": str(export_dir), **export_validation},
        "model_audit": model_structure,
    }


def _segmentation_cell(
    dataset: str,
    model_label: str,
    adaptation: str,
    config_path: Path,
    experiment_name: str,
    cell_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    config = resolve_config(load_config(config_path), config_path, dry_run=False)
    config = copy.deepcopy(config)
    config["seed"] = 20260812
    config["training"]["batch_size"] = 1
    config["data"]["num_workers"] = 0
    experiment = select_experiments(config, [experiment_name])[0]
    reproducibility = config.get("reproducibility", {})
    generator = set_global_seed(
        config["seed"],
        deterministic=bool(reproducibility.get("deterministic", True)),
        warn_only=bool(reproducibility.get("warn_only", False)),
    )
    loaders = make_segmentation_dataloaders(config, generator=generator)
    train_batch_raw = next(iter(loaders["train"]))
    metadata = _segmentation_metadata_check(config, train_batch_raw)
    model = build_segmentation_model(config, experiment["model"]).to(device)
    model_structure = audit_segmentation_model(model, experiment["model"])
    optimizer = build_optimizer(model, config["training"])
    ignore_index = config["data"].get("ignore_index")

    model.train()
    train_batch = move_segmentation_batch(train_batch_raw, device)
    optimizer.zero_grad(set_to_none=True)
    train_logits = model(train_batch["image"])
    if not torch.isfinite(train_logits).all():
        raise FloatingPointError("Segmentation forward produced non-finite logits")
    loss = segmentation_loss(train_logits, train_batch["mask"], ignore_index)
    if not torch.isfinite(loss):
        raise FloatingPointError("Segmentation loss is non-finite")
    loss.backward()
    gradients = _gradient_audit(model, adaptation)
    if not gradients["valid"]:
        raise RuntimeError(f"Segmentation gradient audit failed: {gradients}")
    optimizer.step()

    model.eval()
    validation_raw = next(iter(loaders["val"]))
    _segmentation_metadata_check(config, validation_raw)
    validation = move_segmentation_batch(validation_raw, device)
    with torch.no_grad():
        validation_logits = model(validation["image"])
    foreground = config["data"].get("foreground_class_index")
    metrics = segmentation_metrics(
        validation_logits.cpu(),
        validation["mask"].cpu(),
        config["data"]["class_names"],
        ignore_index=ignore_index,
        n_bins=15,
        foreground_class_index=int(foreground) if foreground is not None else None,
        boundary_radius=int(config.get("metrics", {}).get("boundary_radius", 1)),
    )
    primary_values = [metrics["miou"], metrics["pixel_accuracy"], metrics["nll"], metrics["brier"], metrics["ece_15"]]
    if not all(math.isfinite(float(value)) for value in primary_values):
        raise RuntimeError(f"Segmentation validation metrics are non-finite: {metrics}")

    checkpoint = _checkpoint_roundtrip(
        cell_dir,
        model,
        optimizer,
        {"dataset": dataset, "model": model_label, "adaptation": adaptation},
    )
    export_dir = cell_dir / "prediction_export"
    export_segmentation_predictions(
        export_dir,
        sample_ids=[str(value) for value in validation_raw["sample_id"]],
        masks=validation["mask"].cpu(),
        logits=validation_logits.cpu(),
        class_names=config["data"]["class_names"],
        ignore_index=ignore_index,
        model_name=model_display_name(experiment["model"]),
        dataset=str(config["data"]["name"]),
        adaptation_mode=adaptation,
        split="val_tiny_preflight",
        checkpoint="temporary_preflight_checkpoint_not_retained",
    )
    export_validation = validate_segmentation_prediction_export(export_dir)
    if not export_validation["valid"]:
        raise RuntimeError(f"Segmentation prediction export failed: {export_validation}")
    return {
        "dataloader": {"valid": True, "split_sizes": {name: len(loader.dataset) for name, loader in loaders.items()}},
        "metadata": metadata,
        "forward": {"valid": True, "logits_shape": list(train_logits.shape), "finite": True},
        "loss": {"valid": True, "name": "pixelwise_cross_entropy", "value": float(loss.item())},
        "gradients": gradients,
        "validation_metric": {"valid": True, **metrics},
        "checkpoint_writing": checkpoint,
        "prediction_export": {"valid": True, "path": str(export_dir), **export_validation},
        "model_audit": model_structure,
    }


def _run_cell(
    task: str,
    cell: Sequence[str],
    output_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    dataset, model_label, adaptation, configured_path, experiment_name = cell
    cell_id = f"{task}_{dataset}_{model_label}_{adaptation}".lower().replace(" ", "_")
    cell_dir = output_root / cell_id
    cell_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    base = {
        "cell_id": cell_id,
        "task": task,
        "dataset": dataset,
        "model": model_label,
        "adaptation": adaptation,
        "config": configured_path,
        "experiment": experiment_name,
        "artifact_dir": str(cell_dir),
    }
    try:
        if task == "classification":
            checks = _classification_cell(
                dataset,
                model_label,
                adaptation,
                PROJECT_ROOT / configured_path,
                experiment_name,
                cell_dir,
                device,
            )
        else:
            checks = _segmentation_cell(
                dataset,
                model_label,
                adaptation,
                PROJECT_ROOT / configured_path,
                experiment_name,
                cell_dir,
                device,
            )
        result = {**base, "status": "READY", "blocker": None, "checks": checks}
    except Exception as exc:
        result = {
            **base,
            "status": "BLOCKED",
            "blocker": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "checks": {},
        }
    result["elapsed_seconds"] = time.perf_counter() - started
    result["peak_gpu_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    (cell_dir / "summary.json").write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root or PROJECT_ROOT / "results" / "preflight" / f"full_16_cell_{timestamp}"
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=False)
    results = []
    for task, cells in (("classification", CLASSIFICATION_CELLS), ("segmentation", SEGMENTATION_CELLS)):
        for cell in cells:
            print(f"[preflight] {task} / {cell[0]} / {cell[1]} / {cell[2]}", flush=True)
            result = _run_cell(task, cell, output_root, device)
            results.append(result)
            print(f"[preflight] -> {result['status']}: {result.get('blocker') or 'all checks passed'}", flush=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torch_version": torch.__version__,
        "cell_count": len(results),
        "ready_count": sum(result["status"] == "READY" for result in results),
        "blocked_count": sum(result["status"] == "BLOCKED" for result in results),
        "one_optimizer_step_per_cell": True,
        "final_training_performed": False,
        "results": results,
    }
    results_path = output_root / "preflight_results.json"
    results_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "results": str(results_path), "ready": payload["ready_count"], "blocked": payload["blocked_count"]}, indent=2))


if __name__ == "__main__":
    main()
