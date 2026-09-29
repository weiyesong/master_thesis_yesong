from __future__ import annotations

"""Final C5 downstream MC-Dropout inference, validation, and reporting."""

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import yaml

from scripts.c3_calibration_ensembles import (
    _load_classification_prediction,
    classification_metrics,
    classification_registry,
    segmentation_registry,
)
from scripts.experiment_manager import PROJECT_ROOT
from scripts.mc_dropout import activate_downstream_mc_dropout, designated_mc_dropout_modules
from scripts.prediction_export import (
    checkpoint_sha256,
    export_stochastic_predictions,
    validate_prediction_export,
)
from scripts.run_experiments import (
    build_model,
    classification_type_from_config,
    make_dataloaders,
    model_display_name,
    move_batch,
)
from scripts.segmentation_pipeline import (
    SegmentationMetricAccumulator,
    build_segmentation_model,
    make_segmentation_dataloaders,
    move_segmentation_batch,
    segmentation_metrics,
)


OUTPUT_ROOT = PROJECT_ROOT / "results/final_thesis/mc_dropout"
REGISTRY_PATH = OUTPUT_ROOT / "run_registry.json"
INFERENCE_ROOT = OUTPUT_ROOT / "inference"
SUBSET_PATH = PROJECT_ROOT / "reports/mc_dropout_segmentation_research_subset_ids.json"
RESULTS_CSV = PROJECT_ROOT / "reports/mc_dropout_results.csv"
SUMMARY_REPORT = PROJECT_ROOT / "reports/mc_dropout_summary.md"
ROBUSTNESS_REPORT = PROJECT_ROOT / "reports/mc_dropout_robustness.md"
PASSES = 30
DROPOUT_PROBABILITY = 0.10
N_BINS = 15


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.write("\n")


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_registry() -> dict[str, Any]:
    value = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if (
        value.get("training_run_count") != 24
        or value.get("primary_seed42_run_count") != 16
        or value.get("dropout_probability") != DROPOUT_PROBABILITY
        or value.get("stochastic_passes") != PASSES
    ):
        raise ValueError("C5 run registry no longer matches the frozen protocol")
    return value


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return row["task"], row["dataset"], row["model"], row["adaptation"], int(row["seed"])


def select_registry_rows(keys: Sequence[str] | None) -> list[dict[str, Any]]:
    rows = _load_registry()["runs"]
    if not keys:
        return rows
    wanted = set(keys)
    selected = [row for row in rows if compact_key(row) in wanted]
    missing = wanted - {compact_key(row) for row in selected}
    if missing:
        raise ValueError(f"Unknown C5 keys: {sorted(missing)}")
    return selected


def compact_key(row: dict[str, Any]) -> str:
    return f"{row['dataset']}:{row['model']}:{row['adaptation']}:{row['seed']}"


def _find_training_run(row: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = Path(row["output_root"])
    candidates = []
    for summary_path in (root / "runs").glob("*/run_summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        resolved_path = summary_path.parent / "resolved_config.yaml"
        if not resolved_path.is_file():
            continue
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") == "completed"
            and not summary.get("dry_run", False)
            and int(summary.get("seed", -1)) == int(row["seed"])
            and resolved.get("active_experiment") == row["experiment"]
        ):
            candidates.append((summary_path.parent, summary, resolved))
    if len(candidates) != 1:
        raise ValueError(f"Expected one completed C5 training run for {compact_key(row)}, found {len(candidates)}")
    return candidates[0]


def _training_audit(
    row: dict[str, Any], run_dir: Path, summary: dict[str, Any], resolved: dict[str, Any]
) -> dict[str, Any]:
    checkpoint_path = run_dir / "best.pt"
    checkpoint_hash = checkpoint_sha256(checkpoint_path)
    model_audit = json.loads((run_dir / "model_audit.json").read_text(encoding="utf-8"))
    gradient_audit = json.loads((run_dir / "gradient_audit.json").read_text(encoding="utf-8"))
    code_snapshot = json.loads((run_dir / "code_snapshot.json").read_text(encoding="utf-8"))
    experiment = next(item for item in resolved["experiments"] if item["name"] == row["experiment"])
    dropout = (
        experiment["model"]["head"]["dropout"]
        if row["task"] == "classification"
        else experiment["model"]["decoder"]["dropout"]
    )
    expected_selection = (
        {"split": "val", "metric": "nll", "mode": "min"}
        if row["task"] == "classification"
        else {"split": "val", "metric": "miou", "mode": "max"}
    )
    evaluation = summary.get("evaluation_metrics", {})
    no_test_metrics = "test" not in evaluation
    no_test_exports = not summary.get("prediction_exports")
    designated = model_audit.get("mc_dropout")
    if designated is None:
        designated = [
            {
                "p": dropout,
                "placement": model_audit.get("training_mode_policy", {}).get("head_dropout"),
            }
        ] if model_audit.get("training_mode_policy", {}).get("head_dropout") == "configured" else []
    if row["task"] == "classification":
        gradients_valid = bool(
            gradient_audit.get("performed")
            and gradient_audit.get("all_expected_parameters_received_gradient")
            and not gradient_audit.get("nonfinite_gradient_parameters")
        )
    else:
        head = gradient_audit.get("groups", {}).get("head", {})
        backbone = gradient_audit.get("groups", {}).get("backbone", {})
        gradients_valid = bool(
            head.get("all_gradients_finite")
            and head.get("all_trainable_parameters_have_gradients")
            and head.get("any_nonzero_gradient")
            and (
                backbone.get("gradient_parameter_tensors") == 0
                if row["adaptation"] == "frozen"
                else backbone.get("all_gradients_finite") and backbone.get("any_nonzero_gradient")
            )
        )
    checks = {
        "status_completed": summary.get("status") == "completed",
        "full_not_dry_run": not summary.get("dry_run", False),
        "seed_matches": int(summary["seed"]) == int(row["seed"]),
        "checkpoint_selection_matches": summary.get("checkpoint_selection") == expected_selection,
        "test_metrics_absent_during_training": no_test_metrics,
        "test_export_absent_during_training": no_test_exports,
        "test_evaluation_disabled": not bool(resolved.get("evaluation", {}).get("test_enabled", True)),
        "dropout_probability_0_10": math.isclose(float(dropout), DROPOUT_PROBABILITY, abs_tol=0.0),
        "one_designated_dropout_in_model_audit": len(designated) == 1,
        "gradient_audit_valid": gradients_valid,
        "input_artifacts_verified": summary.get("input_artifacts", {}).get("all_verified", False),
        "valid_code_snapshot": (
            code_snapshot.get("algorithm") == "sha256"
            and isinstance(code_snapshot.get("code_sha256"), str)
            and len(code_snapshot["code_sha256"]) == 64
            and int(code_snapshot.get("file_count", 0)) > 0
            and len(code_snapshot.get("files", [])) == int(code_snapshot.get("file_count", -1))
        ),
        "checkpoint_hash_matches_summary": checkpoint_hash
        == summary.get("best_checkpoint_sha256", checkpoint_hash),
        "source_config_hash_matches_registry": sha256_file(Path(row["config_path"])) == row["config_sha256"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "best_epoch": summary.get("best_epoch"),
        "last_epoch": summary.get("last_epoch", len(summary.get("history", [])) or None),
        "training_seconds": summary.get("training_seconds"),
        "code_sha256": code_snapshot.get("code_sha256"),
        "code_file_count": code_snapshot.get("file_count"),
    }


def _bn_state(model: nn.Module) -> dict[str, tuple[torch.Tensor | None, ...]]:
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            result[name] = tuple(
                None if value is None else value.detach().cpu().clone()
                for value in (module.running_mean, module.running_var, module.num_batches_tracked)
            )
    return result


def _bn_equal(left: dict[str, tuple[Any, ...]], right: dict[str, tuple[Any, ...]]) -> bool:
    if left.keys() != right.keys():
        return False
    return all(
        (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
        for key in left for a, b in zip(left[key], right[key])
    )


def _classification_uncertainty(probabilities: np.ndarray, classification_type: str) -> dict[str, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float64)
    mean = values.mean(axis=1)
    variance = values.var(axis=1)
    eps = np.finfo(np.float64).tiny
    if classification_type == "multiclass":
        predictive = -(mean * np.log(np.clip(mean, eps, 1.0))).sum(axis=1)
        per_pass = -(values * np.log(np.clip(values, eps, 1.0))).sum(axis=2)
        expected = per_pass.mean(axis=1)
        disagreement = predictive - expected
        return {
            "mean_probabilities": mean.astype(np.float32),
            "predictive_entropy": predictive.astype(np.float32),
            "expected_predictive_entropy": expected.astype(np.float32),
            "mi_style_disagreement": disagreement.astype(np.float32),
            "predictive_variance": variance.astype(np.float32),
        }
    if classification_type != "multilabel":
        raise ValueError(classification_type)
    clipped_mean = np.clip(mean, eps, 1.0 - 1.0e-15)
    clipped = np.clip(values, eps, 1.0 - 1.0e-15)
    predictive_per_label = -(
        clipped_mean * np.log(clipped_mean) + (1.0 - clipped_mean) * np.log(1.0 - clipped_mean)
    )
    per_pass = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    expected_per_label = per_pass.mean(axis=1)
    disagreement_per_label = predictive_per_label - expected_per_label
    return {
        "mean_probabilities": mean.astype(np.float32),
        "predictive_entropy_per_label": predictive_per_label.astype(np.float32),
        "expected_predictive_entropy_per_label": expected_per_label.astype(np.float32),
        "mi_style_disagreement_per_label": disagreement_per_label.astype(np.float32),
        # Per-label values are the unreduced scientific archive.  The scalar
        # sample summaries use the joint independent-Bernoulli entropy sum,
        # matching the existing final prediction-export convention.
        "predictive_entropy": predictive_per_label.sum(axis=1).astype(np.float32),
        "expected_predictive_entropy": expected_per_label.sum(axis=1).astype(np.float32),
        "mi_style_disagreement": disagreement_per_label.sum(axis=1).astype(np.float32),
        "predictive_variance": variance.astype(np.float32),
    }


def _write_classification_uncertainty(
    output_dir: Path,
    sample_ids: list[str],
    labels: np.ndarray,
    classification_type: str,
    uncertainty: dict[str, np.ndarray],
) -> None:
    probabilities = uncertainty["mean_probabilities"]
    if classification_type == "multiclass":
        prediction: Any = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
    else:
        prediction = (probabilities >= 0.5).astype(np.int8)
        confidence = np.maximum(probabilities, 1.0 - probabilities).mean(axis=1)
    table: dict[str, Any] = {
        "sample_id": sample_ids,
        "label": labels.tolist() if labels.ndim == 2 else labels,
        "mean_probability": [row.tolist() for row in probabilities],
        "prediction": prediction.tolist() if prediction.ndim == 2 else prediction,
        "confidence": confidence,
        "predictive_entropy": uncertainty["predictive_entropy"],
        "expected_predictive_entropy": uncertainty["expected_predictive_entropy"],
        "mi_style_disagreement": uncertainty["mi_style_disagreement"],
        "predictive_variance": [row.tolist() for row in uncertainty["predictive_variance"]],
    }
    for name in (
        "predictive_entropy_per_label",
        "expected_predictive_entropy_per_label",
        "mi_style_disagreement_per_label",
    ):
        if name in uncertainty:
            table[name] = [row.tolist() for row in uncertainty[name]]
    frame = pd.DataFrame(table)
    arrow = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(arrow, output_dir / "uncertainty_summaries.parquet", compression="zstd")
    np.savez_compressed(
        output_dir / "uncertainty_summaries.npz",
        sample_id=np.asarray(sample_ids),
        label=labels,
        prediction=prediction,
        confidence=confidence.astype(np.float32),
        **uncertainty,
    )


def infer_classification(row: dict[str, Any], device: torch.device) -> dict[str, Any]:
    run_dir, summary, config = _find_training_run(row)
    training_audit = _training_audit(row, run_dir, summary, config)
    if not training_audit["passed"]:
        raise RuntimeError(f"C5 training audit failed for {compact_key(row)}: {training_audit['checks']}")
    checkpoint_path = Path(training_audit["checkpoint_path"])
    checkpoint_before = checkpoint_sha256(checkpoint_path)
    deterministic = row["deterministic_comparator"]
    deterministic_checkpoint_before = sha256_file(Path(deterministic["checkpoint_path"]))
    deterministic_prediction_before = sha256_file(Path(deterministic["prediction_path"]))
    experiment = next(item for item in config["experiments"] if item["name"] == row["experiment"])
    classification_type = classification_type_from_config(config)
    model = build_model(config, experiment["model"], int(config["data"]["num_classes"])).to(device)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    loader = make_dataloaders(config)["test"]
    checkpoint_count = len(loader.dataset)
    batchnorm_before = _bn_state(model)
    activation = activate_downstream_mc_dropout(model)
    designated = designated_mc_dropout_modules(model)
    if len(designated) != 1 or not math.isclose(float(designated[0][1].p), DROPOUT_PROBABILITY, abs_tol=0.0):
        raise RuntimeError("Classification inference does not expose the frozen one-layer p=0.10 dropout")
    sample_ids: list[str] = []
    labels_parts: list[torch.Tensor] = []
    logits_parts: list[torch.Tensor] = []
    representation_parts: list[torch.Tensor] = []
    inference_seed = 5_000_000 + int(row["seed"])
    torch.manual_seed(inference_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(inference_seed)
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            # The foundation representation is deterministic in the frozen MC
            # inference mode.  Compute it once, then draw all 30 samples from
            # the designated downstream-head dropout using that same tensor.
            representations = model.extract_features(batch["image"], batch.get("temporal_mask"))
            passes = [model.head(representations).detach().float().cpu() for _ in range(PASSES)]
            stochastic_logits = torch.stack(passes, dim=1)
            if not torch.isfinite(stochastic_logits).all() or not torch.isfinite(representations).all():
                raise FloatingPointError("Non-finite C5 classification output")
            sample_ids.extend(str(value) for value in raw_batch["sample_id"])
            labels_parts.append(batch["label"].detach().cpu())
            logits_parts.append(stochastic_logits)
            representation_parts.append(representations.detach().float().cpu())
    labels = torch.cat(labels_parts).numpy()
    stochastic_logits = torch.cat(logits_parts).numpy()
    representations = torch.cat(representation_parts).numpy()
    if len(sample_ids) != checkpoint_count or len(set(sample_ids)) != checkpoint_count:
        raise ValueError("Classification inference did not cover the unique complete test set")
    if classification_type == "multiclass":
        stochastic_probabilities = torch.softmax(torch.from_numpy(stochastic_logits), dim=2).numpy()
    else:
        stochastic_probabilities = torch.sigmoid(torch.from_numpy(stochastic_logits)).numpy()
    uncertainty = _classification_uncertainty(stochastic_probabilities, classification_type)
    metrics = classification_metrics(
        uncertainty["mean_probabilities"], labels, classification_type, n_bins=N_BINS
    )
    output_dir = INFERENCE_ROOT / "classification" / row["dataset"] / row["model"] / row["adaptation"] / f"seed{row['seed']}"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    export_stochastic_predictions(
        output_dir,
        stochastic_logits=stochastic_logits,
        sample_ids=sample_ids,
        true_labels=labels,
        class_names=config["data"]["class_names"],
        model_name=model_display_name(experiment["model"]),
        dataset=row["dataset"],
        adaptation_mode=row["adaptation"],
        seed=int(row["seed"]),
        checkpoint=str(checkpoint_path),
        split="test",
        uq_method="downstream_head_mc_dropout_t30_probability_mean",
        expected_count=checkpoint_count,
        embeddings=representations,
        classification_type=classification_type,
        multilabel_threshold=float(config.get("metrics", {}).get("multilabel_threshold", 0.5)),
        source={
            "run_id": run_dir.name,
            "checkpoint_sha256": checkpoint_before,
            "passes": PASSES,
            "dropout_probability": DROPOUT_PROBABILITY,
            "aggregation": "arithmetic mean of probabilities, never logits",
            "inference_seed": inference_seed,
            "complete_official_test_split": True,
        },
    )
    _write_classification_uncertainty(output_dir, sample_ids, labels, classification_type, uncertainty)
    _write_json_exclusive(output_dir / "metrics.json", metrics)
    inference_checks = {
        "prediction_export_valid": validate_prediction_export(output_dir, expected_count=checkpoint_count)["valid"],
        "stochastic_shape_n_t_c": list(stochastic_logits.shape)
        == [checkpoint_count, PASSES, int(config["data"]["num_classes"])],
        "stochastic_outputs_nonidentical": bool(
            np.any(stochastic_probabilities[:, 1:] != stochastic_probabilities[:, :-1])
        ),
        "mean_probability_not_mean_logits": True,
        "one_designated_dropout": len(designated) == 1,
        "dropout_active": designated[0][1].training,
        "batchnorm_eval": not activation["batchnorm_training_modules"],
        "batchnorm_buffers_unchanged": _bn_equal(batchnorm_before, _bn_state(model)),
        "backbone_stochasticity_disabled": not activation["backbone_stochastic_training_modules"],
        "no_gradients": all(parameter.grad is None for parameter in model.parameters()),
        "mc_checkpoint_unchanged": checkpoint_before == checkpoint_sha256(checkpoint_path),
        "deterministic_checkpoint_unchanged": deterministic_checkpoint_before
        == sha256_file(Path(deterministic["checkpoint_path"])) == deterministic["checkpoint_sha256"],
        "deterministic_prediction_unchanged": deterministic_prediction_before
        == sha256_file(Path(deterministic["prediction_path"])) == deterministic["prediction_sha256"],
    }
    payload = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "cell": compact_key(row),
        "training_audit": training_audit,
        "activation_audit": activation,
        "inference_checks": inference_checks,
        "passed": all(inference_checks.values()),
        "metrics": metrics,
        "uncertainty_summary": {
            "mean_predictive_entropy": float(np.mean(uncertainty["predictive_entropy"])),
            "mean_expected_predictive_entropy": float(np.mean(uncertainty["expected_predictive_entropy"])),
            "mean_mi_style_disagreement": float(np.mean(uncertainty["mi_style_disagreement"])),
            "mean_predictive_variance": float(np.mean(uncertainty["predictive_variance"])),
        },
        "uncertainty_reduction": (
            "categorical entropy per sample"
            if classification_type == "multiclass"
            else "per-label Bernoulli quantities retained; scalar sample/report values sum across labels"
        ),
        "output_dir": str(output_dir.resolve()),
        "artifact_hashes": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "c5_audit.json"
        },
    }
    if not payload["passed"]:
        raise RuntimeError(f"C5 classification inference audit failed for {compact_key(row)}")
    _write_json_exclusive(output_dir / "c5_audit.json", payload)
    return payload


def _segmentation_uncertainty(probabilities: torch.Tensor) -> dict[str, torch.Tensor]:
    values = probabilities.double()
    mean = values.mean(dim=1)
    predictive = -(mean * mean.clamp_min(1.0e-12).log()).sum(dim=1)
    expected = -(values * values.clamp_min(1.0e-12).log()).sum(dim=2).mean(dim=1)
    return {
        "mean_probabilities": mean.float(),
        "predictive_entropy": predictive.float(),
        "expected_predictive_entropy": expected.float(),
        "mi_style_disagreement": (predictive - expected).float(),
        "predictive_variance": values.var(dim=1, unbiased=False).float(),
    }


def _per_image_segmentation_row(
    sample_id: str,
    mean_probabilities: torch.Tensor,
    target: torch.Tensor,
    config: dict[str, Any],
    uncertainty: dict[str, torch.Tensor],
) -> dict[str, Any]:
    data, metric_cfg = config["data"], config.get("metrics", {})
    foreground = data.get("foreground_class_index")
    metrics = segmentation_metrics(
        mean_probabilities.clamp_min(1.0e-12).log().unsqueeze(0),
        target.unsqueeze(0),
        data["class_names"],
        ignore_index=data.get("ignore_index"),
        n_bins=N_BINS,
        foreground_class_index=foreground,
        boundary_radius=int(metric_cfg.get("boundary_radius", 1)),
    )
    valid = torch.ones_like(target, dtype=torch.bool)
    if data.get("ignore_index") is not None:
        valid &= target.ne(int(data["ignore_index"]))
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "miou": metrics["miou"],
        "pixel_accuracy": metrics["pixel_accuracy"],
        "nll": metrics["nll"],
        "brier": metrics["brier"],
        "ece_15": metrics["ece_15"],
        "valid_pixels": metrics["valid_pixels"],
        "ignored_pixels": metrics["ignored_pixels"],
        "mean_predictive_entropy": float(uncertainty["predictive_entropy"][valid].mean()),
        "mean_expected_predictive_entropy": float(uncertainty["expected_predictive_entropy"][valid].mean()),
        "mean_mi_style_disagreement": float(uncertainty["mi_style_disagreement"][valid].mean()),
        "mean_predictive_variance": float(
            uncertainty["predictive_variance"].permute(1, 2, 0)[valid].mean()
        ),
    }
    row.update({f"iou_{name}": value for name, value in metrics["per_class_iou"].items()})
    if foreground is not None:
        row.update(
            {
                "foreground_ece_15": metrics["foreground_ece_15"],
                "foreground_nll": metrics["foreground_nll"],
                "foreground_brier": metrics["foreground_brier"],
                "boundary_ece_15": metrics["boundary_calibration"]["ece_15"],
                "boundary_foreground_ece_15": metrics["boundary_calibration"]["foreground_ece_15"],
                "boundary_pixel_count": metrics["boundary_calibration"]["pixel_count"],
            }
        )
    return row


def _write_csv_exclusive(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to create empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_segmentation(row: dict[str, Any], device: torch.device) -> dict[str, Any]:
    run_dir, summary, config = _find_training_run(row)
    training_audit = _training_audit(row, run_dir, summary, config)
    if not training_audit["passed"]:
        raise RuntimeError(f"C5 training audit failed for {compact_key(row)}: {training_audit['checks']}")
    checkpoint_path = Path(training_audit["checkpoint_path"])
    checkpoint_before = checkpoint_sha256(checkpoint_path)
    deterministic = row["deterministic_comparator"]
    deterministic_checkpoint_before = sha256_file(Path(deterministic["checkpoint_path"]))
    deterministic_prediction_before = sha256_file(Path(deterministic["prediction_path"]))
    experiment = next(item for item in config["experiments"] if item["name"] == row["experiment"])
    model = build_segmentation_model(config, experiment["model"]).to(device)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    loader = make_segmentation_dataloaders(config)["test"]
    expected_count = len(loader.dataset)
    subset_manifest = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
    fixed_subset_ids = subset_manifest["datasets"][row["dataset"]]["final_test_research_subset_ids"]
    fixed_subset_set = set(fixed_subset_ids)
    if len(fixed_subset_ids) != 32:
        raise ValueError("Frozen segmentation research subset no longer has 32 IDs")
    batchnorm_before = _bn_state(model)
    activation = activate_downstream_mc_dropout(model)
    designated = designated_mc_dropout_modules(model)
    if (
        len(designated) != 1
        or not isinstance(designated[0][1], nn.Dropout2d)
        or not math.isclose(float(designated[0][1].p), DROPOUT_PROBABILITY, abs_tol=0.0)
    ):
        raise RuntimeError("Segmentation inference does not expose the frozen one-layer p=0.10 Dropout2d")
    data, metric_cfg = config["data"], config.get("metrics", {})
    foreground = data.get("foreground_class_index")
    aggregate = SegmentationMetricAccumulator(
        int(data["num_classes"]), data["class_names"], data.get("ignore_index"), N_BINS,
        foreground, int(metric_cfg.get("boundary_radius", 1)),
    )
    sample_ids: list[str] = []
    labels_parts: list[np.ndarray] = []
    probabilities_parts: list[np.ndarray] = []
    predictions_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    predictive_parts: list[np.ndarray] = []
    expected_parts: list[np.ndarray] = []
    disagreement_parts: list[np.ndarray] = []
    variance_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    per_image_rows: list[dict[str, Any]] = []
    subset_probabilities: dict[str, np.ndarray] = {}
    subset_labels: dict[str, np.ndarray] = {}
    stochastic_outputs_nonidentical = False
    inference_seed = 7_000_000 + int(row["seed"])
    torch.manual_seed(inference_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(inference_seed)
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_segmentation_batch(raw_batch, device)
            dense_features = model.extract_dense_features(batch["image"])
            passes = []
            for _ in range(PASSES):
                logits = model.head(dense_features, output_size=batch["image"].shape[-2:])
                passes.append(logits.softmax(dim=1).detach().float().cpu())
            stochastic_probabilities = torch.stack(passes, dim=1)
            stochastic_outputs_nonidentical |= bool(
                torch.any(stochastic_probabilities[:, 1:] != stochastic_probabilities[:, :-1])
            )
            if not torch.isfinite(stochastic_probabilities).all() or not torch.isfinite(dense_features).all():
                raise FloatingPointError("Non-finite C5 segmentation output")
            uncertainty = _segmentation_uncertainty(stochastic_probabilities)
            mean_probabilities = uncertainty["mean_probabilities"]
            target = batch["mask"].detach().cpu()
            aggregate.update(mean_probabilities.clamp_min(1.0e-12).log(), target)
            batch_ids = [str(value) for value in raw_batch["sample_id"]]
            prediction = mean_probabilities.argmax(dim=1)
            confidence = mean_probabilities.max(dim=1).values
            valid = torch.ones_like(target, dtype=torch.bool)
            if data.get("ignore_index") is not None:
                valid &= target.ne(int(data["ignore_index"]))
            sample_ids.extend(batch_ids)
            labels_parts.append(target.numpy())
            probabilities_parts.append(mean_probabilities.numpy())
            predictions_parts.append(prediction.numpy())
            confidence_parts.append(confidence.numpy())
            predictive_parts.append(uncertainty["predictive_entropy"].numpy())
            expected_parts.append(uncertainty["expected_predictive_entropy"].numpy())
            disagreement_parts.append(uncertainty["mi_style_disagreement"].numpy())
            variance_parts.append(uncertainty["predictive_variance"].numpy())
            valid_parts.append(valid.numpy())
            for index, sample_id in enumerate(batch_ids):
                per_image_rows.append(
                    _per_image_segmentation_row(
                        sample_id,
                        mean_probabilities[index],
                        target[index],
                        config,
                        {name: value[index] for name, value in uncertainty.items() if name != "mean_probabilities"},
                    )
                )
                if sample_id in fixed_subset_set:
                    subset_probabilities[sample_id] = stochastic_probabilities[index].numpy()
                    subset_labels[sample_id] = target[index].numpy()
    if len(sample_ids) != expected_count or len(set(sample_ids)) != expected_count:
        raise ValueError("Segmentation inference did not cover the unique complete test set")
    if set(subset_probabilities) != fixed_subset_set:
        raise ValueError("Not every frozen research-subset ID was collected")
    arrays = {
        "sample_id": np.asarray(sample_ids),
        "label": np.concatenate(labels_parts),
        "probabilities": np.concatenate(probabilities_parts),
        "prediction": np.concatenate(predictions_parts),
        "confidence": np.concatenate(confidence_parts),
        "predictive_entropy": np.concatenate(predictive_parts),
        "expected_predictive_entropy": np.concatenate(expected_parts),
        "mi_style_disagreement": np.concatenate(disagreement_parts),
        "predictive_variance": np.concatenate(variance_parts),
        "valid_mask": np.concatenate(valid_parts),
    }
    metrics = aggregate.compute()
    valid = arrays["valid_mask"]
    uncertainty_summary = {
        "mean_predictive_entropy": float(arrays["predictive_entropy"][valid].mean()),
        "mean_expected_predictive_entropy": float(arrays["expected_predictive_entropy"][valid].mean()),
        "mean_mi_style_disagreement": float(arrays["mi_style_disagreement"][valid].mean()),
        "mean_predictive_variance": float(
            np.moveaxis(arrays["predictive_variance"], 1, -1)[valid].mean()
        ),
    }
    output_dir = INFERENCE_ROOT / "segmentation" / row["dataset"] / row["model"] / row["adaptation"] / f"seed{row['seed']}"
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "aggregate_uncertainty_maps.npz", **arrays)
    np.savez_compressed(
        output_dir / "research_subset_stochastic_probabilities.npz",
        sample_id=np.asarray(fixed_subset_ids),
        label=np.stack([subset_labels[value] for value in fixed_subset_ids]),
        probabilities=np.stack([subset_probabilities[value] for value in fixed_subset_ids]),
        axes=np.asarray(["sample", "pass", "class", "height", "width"]),
    )
    _write_csv_exclusive(output_dir / "per_image_metrics.csv", per_image_rows)
    _write_json_exclusive(output_dir / "metrics.json", metrics)
    boundary_undefined = 0
    if row["dataset"] == "spacenet7":
        boundary_undefined = sum(math.isnan(float(item["boundary_ece_15"])) for item in per_image_rows)
    inference_checks = {
        "complete_test_count": len(sample_ids) == expected_count,
        "sample_ids_unique": len(set(sample_ids)) == expected_count,
        "probabilities_finite": bool(np.isfinite(arrays["probabilities"]).all()),
        "probabilities_sum_to_one": bool(
            np.allclose(arrays["probabilities"].sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0)
        ),
        "research_subset_shape": list(np.stack([subset_probabilities[value] for value in fixed_subset_ids]).shape)
        == [32, PASSES, int(data["num_classes"]), int(data["image_size"]), int(data["image_size"])],
        "stochastic_outputs_nonidentical": stochastic_outputs_nonidentical,
        "mean_probability_not_mean_logits": True,
        "one_designated_dropout2d": len(designated) == 1 and isinstance(designated[0][1], nn.Dropout2d),
        "dropout_active": designated[0][1].training,
        "batchnorm_eval": not activation["batchnorm_training_modules"],
        "batchnorm_buffers_unchanged": _bn_equal(batchnorm_before, _bn_state(model)),
        "backbone_stochasticity_disabled": not activation["backbone_stochastic_training_modules"],
        "no_gradients": all(parameter.grad is None for parameter in model.parameters()),
        "mc_checkpoint_unchanged": checkpoint_before == checkpoint_sha256(checkpoint_path),
        "deterministic_checkpoint_unchanged": deterministic_checkpoint_before
        == sha256_file(Path(deterministic["checkpoint_path"])) == deterministic["checkpoint_sha256"],
        "deterministic_prediction_unchanged": deterministic_prediction_before
        == sha256_file(Path(deterministic["prediction_path"])) == deterministic["prediction_sha256"],
        "space_boundary_undefined_preserved": row["dataset"] != "spacenet7"
        or all(
            math.isnan(float(item["boundary_ece_15"])) == (int(item["boundary_pixel_count"]) == 0)
            for item in per_image_rows
        ),
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "cell": compact_key(row),
        "run_id": run_dir.name,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_before,
        "complete_official_test_split": True,
        "sample_count": expected_count,
        "passes": PASSES,
        "dropout_probability": DROPOUT_PROBABILITY,
        "aggregation": "arithmetic mean of per-pass probabilities; logits were never averaged",
        "uncertainty_terminology": {
            "expected_predictive_entropy": "expected entropy across stochastic dropout predictions",
            "mi_style_disagreement": "stochastic disagreement / epistemic proxy; not asserted to be true epistemic uncertainty",
        },
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
        "research_subset": {
            "source": str(SUBSET_PATH.resolve()),
            "source_sha256": sha256_file(SUBSET_PATH),
            "sample_ids": fixed_subset_ids,
            "shape": [32, PASSES, int(data["num_classes"]), int(data["image_size"]), int(data["image_size"])],
        },
        "space_boundary_undefined_image_count": boundary_undefined if row["dataset"] == "spacenet7" else None,
        "metrics": metrics,
        "uncertainty_summary": uncertainty_summary,
        "training_audit": training_audit,
        "activation_audit": activation,
        "inference_checks": inference_checks,
        "passed": all(inference_checks.values()),
    }
    if not manifest["passed"]:
        raise RuntimeError(f"C5 segmentation inference audit failed for {compact_key(row)}: {inference_checks}")
    manifest["artifact_hashes"] = {
        path.name: sha256_file(path) for path in sorted(output_dir.iterdir()) if path.is_file()
    }
    _write_json_exclusive(output_dir / "manifest.json", manifest)
    return manifest


def infer(rows: Sequence[dict[str, Any]], device_name: str) -> list[dict[str, Any]]:
    device = torch.device(device_name)
    values = []
    for row in rows:
        output_dir = INFERENCE_ROOT / row["task"] / row["dataset"] / row["model"] / row["adaptation"] / f"seed{row['seed']}"
        if output_dir.exists():
            audit_path = output_dir / ("c5_audit.json" if row["task"] == "classification" else "manifest.json")
            if not audit_path.is_file() or not json.loads(audit_path.read_text(encoding="utf-8")).get("passed"):
                raise FileExistsError(f"Incomplete prior C5 output requires manual audit: {output_dir}")
            values.append(json.loads(audit_path.read_text(encoding="utf-8")))
            continue
        print(f"C5 inference START {compact_key(row)}", flush=True)
        value = infer_classification(row, device) if row["task"] == "classification" else infer_segmentation(row, device)
        values.append(value)
        print(f"C5 inference DONE {compact_key(row)}", flush=True)
    return values


def _deterministic_metrics(row: dict[str, Any]) -> dict[str, Any]:
    if row["task"] == "classification":
        member = next(
            value for value in classification_registry()
            if (value.dataset, value.model, value.adaptation, value.seed)
            == (row["dataset"], row["model"], row["adaptation"], row["seed"])
        )
        data = _load_classification_prediction(member)
        return classification_metrics(data["probabilities"], data["labels"], data["classification_type"], N_BINS)
    run_dir = Path(row["deterministic_comparator"]["run_dir"])
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    metrics = summary["evaluation_metrics"]["test"]
    return metrics


def _flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("accuracy", "macro_f1", "miou", "pixel_accuracy", "nll", "brier", "ece_15"):
        if name in metrics:
            result[f"{prefix}_{name}"] = metrics[name]
    if "per_class_iou" in metrics:
        for name, value in metrics["per_class_iou"].items():
            result[f"{prefix}_iou_{name}"] = value
    for name in ("foreground_ece_15", "foreground_nll", "foreground_brier"):
        if name in metrics:
            result[f"{prefix}_{name}"] = metrics[name]
    if "classwise_calibration" in metrics:
        for name, value in metrics["classwise_calibration"].items():
            result[f"{prefix}_classwise_ece_{name}"] = value["ece_15"]
    if "boundary_calibration" in metrics:
        result[f"{prefix}_boundary_ece_15"] = metrics["boundary_calibration"]["ece_15"]
        result[f"{prefix}_boundary_foreground_ece_15"] = metrics["boundary_calibration"]["foreground_ece_15"]
    return result


def _read_mc_result(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    output_dir = INFERENCE_ROOT / row["task"] / row["dataset"] / row["model"] / row["adaptation"] / f"seed{row['seed']}"
    audit_path = output_dir / ("c5_audit.json" if row["task"] == "classification" else "manifest.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise ValueError(f"Failed C5 audit: {compact_key(row)}")
    return audit["metrics"], audit["uncertainty_summary"], str(output_dir.resolve()), audit


def finalize() -> list[dict[str, Any]]:
    registry = _load_registry()
    result_rows: list[dict[str, Any]] = []
    for row in registry["runs"]:
        mc, uncertainty, output_path, audit = _read_mc_result(row)
        if checkpoint_sha256(Path(audit["training_audit"]["checkpoint_path"])) != audit["training_audit"]["checkpoint_sha256"]:
            raise ValueError(f"MC checkpoint changed after inference: {compact_key(row)}")
        deterministic_source = row["deterministic_comparator"]
        if (
            sha256_file(Path(deterministic_source["checkpoint_path"]))
            != deterministic_source["checkpoint_sha256"]
            or sha256_file(Path(deterministic_source["prediction_path"]))
            != deterministic_source["prediction_sha256"]
        ):
            raise ValueError(f"Deterministic artifact changed during C5: {compact_key(row)}")
        deterministic = _deterministic_metrics(row)
        flat_mc = _flatten_metrics("mc", mc)
        flat_det = _flatten_metrics("deterministic", deterministic)
        result: dict[str, Any] = {
            "task": row["task"],
            "dataset": row["dataset"],
            "model": row["model"],
            "adaptation": row["adaptation"],
            "seed": row["seed"],
            "replication_role": (
                "primary_and_robustness"
                if row["robustness_subset"] and int(row["seed"]) == 42
                else "robustness"
                if row["robustness_subset"]
                else "primary"
            ),
            "method": "downstream_head_mc_dropout" if row["task"] == "classification" else "downstream_decoder_mc_dropout",
            "dropout_probability": DROPOUT_PROBABILITY,
            "stochastic_passes": PASSES,
            "aggregation": "mean_probabilities",
            "mc_run_id": audit["training_audit"]["run_id"],
            "mc_checkpoint_sha256": audit["training_audit"]["checkpoint_sha256"],
            "mc_code_sha256": audit["training_audit"]["code_sha256"],
            "deterministic_run_id": row["deterministic_comparator"]["run_id"],
            "deterministic_checkpoint_sha256": row["deterministic_comparator"]["checkpoint_sha256"],
            **flat_mc,
            **flat_det,
            "mean_predictive_entropy": uncertainty["mean_predictive_entropy"],
            "mean_expected_predictive_entropy": uncertainty["mean_expected_predictive_entropy"],
            "mean_mi_style_disagreement": uncertainty["mean_mi_style_disagreement"],
            "mean_predictive_variance": uncertainty["mean_predictive_variance"],
            "output_path": output_path,
        }
        for metric in ("accuracy", "macro_f1", "miou", "pixel_accuracy", "nll", "brier", "ece_15"):
            if f"mc_{metric}" in result and f"deterministic_{metric}" in result:
                result[f"delta_{metric}_mc_minus_deterministic"] = (
                    float(result[f"mc_{metric}"]) - float(result[f"deterministic_{metric}"])
                )
        result_rows.append(result)
    if RESULTS_CSV.exists() or SUMMARY_REPORT.exists() or ROBUSTNESS_REPORT.exists():
        raise FileExistsError("Final C5 report artifact already exists; refusing to overwrite")
    _write_csv_exclusive(RESULTS_CSV, result_rows)
    _write_final_reports(result_rows, registry)
    return result_rows


def _format(value: Any) -> str:
    if value is None or value == "":
        return "—"
    try:
        number = float(value)
        return "undefined" if math.isnan(number) else f"{number:.6f}"
    except (TypeError, ValueError):
        return str(value)


def _write_final_reports(rows: list[dict[str, Any]], registry: dict[str, Any]) -> None:
    primary = sorted((row for row in rows if int(row["seed"]) == 42), key=lambda x: (x["task"], x["dataset"], x["model"], x["adaptation"]))
    lines = [
        "# Final MC Dropout Results",
        "",
        f"Created: {utc_now()}",
        "",
        "All 24 models are newly trained downstream-head/decoder MC-Dropout models. No deterministic checkpoint was modified or retrained. The primary analysis is the predeclared seed-42 paired comparison in all 16 cells. Each predictive distribution is the arithmetic mean of 30 probability passes; logits are never averaged.",
        "",
        "## Primary paired matrix (seed 42)",
        "",
        "| Task | Dataset | Model | Adaptation | Accuracy or mIoU MC / deterministic | Macro-F1 or pixel accuracy MC / deterministic | NLL MC / deterministic | Brier MC / deterministic | ECE-15 MC / deterministic | Predictive entropy | Expected predictive entropy | MI-style disagreement |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        task_metric = "accuracy" if row["task"] == "classification" else "miou"
        secondary_metric = "macro_f1" if row["task"] == "classification" else "pixel_accuracy"
        lines.append(
            f"| {row['task']} | {row['dataset']} | {row['model']} | {row['adaptation']} | "
            f"{_format(row.get('mc_'+task_metric))} / {_format(row.get('deterministic_'+task_metric))} | "
            f"{_format(row.get('mc_'+secondary_metric))} / {_format(row.get('deterministic_'+secondary_metric))} | "
            f"{_format(row.get('mc_nll'))} / {_format(row.get('deterministic_nll'))} | "
            f"{_format(row.get('mc_brier'))} / {_format(row.get('deterministic_brier'))} | "
            f"{_format(row.get('mc_ece_15'))} / {_format(row.get('deterministic_ece_15'))} | "
            f"{_format(row['mean_predictive_entropy'])} | {_format(row['mean_expected_predictive_entropy'])} | "
            f"{_format(row['mean_mi_style_disagreement'])} |"
        )
    lines.extend(
        [
            "",
            "## Primary segmentation class and SpaceNet7 calibration detail",
            "",
            "| Dataset | Model | Adaptation | MC per-class IoU | Foreground ECE-15 | Foreground NLL | Foreground Brier | Background classwise ECE-15 | Building classwise ECE-15 | Boundary ECE-15 | Boundary foreground ECE-15 |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in (value for value in primary if value["task"] == "segmentation"):
        class_values = sorted(
            (name.removeprefix("mc_iou_"), value)
            for name, value in row.items() if name.startswith("mc_iou_")
        )
        formatted_classes = "; ".join(f"{name}={_format(value)}" for name, value in class_values)
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['adaptation']} | {formatted_classes} | "
            f"{_format(row.get('mc_foreground_ece_15'))} | {_format(row.get('mc_foreground_nll'))} | "
            f"{_format(row.get('mc_foreground_brier'))} | {_format(row.get('mc_classwise_ece_background'))} | "
            f"{_format(row.get('mc_classwise_ece_building'))} | {_format(row.get('mc_boundary_ece_15'))} | "
            f"{_format(row.get('mc_boundary_foreground_ece_15'))} |"
        )
    lines.extend(
        [
            "",
            "## Protocol and archive validation",
            "",
            f"- Frozen dropout probability: `{DROPOUT_PROBABILITY}`.",
            f"- Frozen stochastic pass count: `{PASSES}`.",
            "- Classification uses one designated downstream-head `Dropout`; segmentation uses one designated final-decoder-feature `Dropout2d`. Foundation-model native dropout, DropPath, and stochastic depth remain inactive at inference.",
            "- BatchNorm stayed in evaluation mode, inference used no gradients, and checkpoint hashes were verified before and after inference.",
            "- Classification archives contain the complete `[N,T,C]` logits and probabilities plus mean probabilities, labels/predictions, confidence, backbone representations, predictive entropy, expected predictive entropy, MI-style disagreement, and per-class probability variance.",
            "- TreeSatAI retains unreduced per-label Bernoulli entropy/disagreement arrays. Its scalar per-sample and reported entropy/disagreement summaries sum across labels, matching the existing multilabel prediction-export convention.",
            "- Segmentation archives contain complete-test mean-probability, prediction, confidence, predictive-entropy, expected-predictive-entropy, disagreement, and variance maps. Raw `[32,T,C,H,W]` probabilities are retained for each dataset's prospectively fixed research subset.",
            "- SpaceNet7 metrics include overall and building/foreground calibration, one-vs-rest classwise calibration, foreground NLL/Brier, and valid-boundary-only calibration. Images with no valid boundary keep undefined per-image boundary metrics.",
            "",
            "## Interpretation limits",
            "",
            "`Expected predictive entropy` is the entropy averaged across stochastic predictions. `MI-style disagreement` is reported as stochastic disagreement/an epistemic proxy; neither is claimed to identify true aleatoric or true epistemic uncertainty. The 30 passes are not independent training replicates. Seed-robust claims are limited to the four prospectively replicated cells in the separate robustness report.",
            "",
            "EuroSAT retains the frozen backend-specific preprocessing history: DOFA MC runs mirror the immutable DOFA comparator's historical RGB normalization, while Panopticon MC runs mirror the newer final-train-statistics protocol. Cross-model EuroSAT interpretation must disclose this confound. The C4 DOFA pilot used the newer statistics only for technical dropout validation and is not a final result.",
            "",
            f"Machine-readable results: `{RESULTS_CSV}`.",
        ]
    )
    _write_text_exclusive(SUMMARY_REPORT, "\n".join(lines) + "\n")

    robustness = [
        row
        for row in rows
        if row["replication_role"] in {"primary_and_robustness", "robustness"}
    ]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in robustness:
        grouped[(row["task"], row["dataset"], row["model"], row["adaptation"])].append(row)
    robust_lines = [
        "# MC Dropout Robustness Subset",
        "",
        "The four cells below were fixed before any final MC-Dropout result was inspected. Each contains independently trained MC models at seeds 42/43/44 and is paired seed-by-seed with the corresponding deterministic model. Thirty stochastic passes within one checkpoint are not treated as independent replicates.",
        "",
    ]
    for key in sorted(grouped):
        task, dataset, model, adaptation = key
        values = sorted(grouped[key], key=lambda x: int(x["seed"]))
        task_metric = "accuracy" if task == "classification" else "miou"
        secondary_metric = "macro_f1" if task == "classification" else "pixel_accuracy"
        robust_lines.extend(
            [
                f"## {dataset} / {model} / {adaptation}",
                "",
                f"| Seed | MC {task_metric} | Deterministic {task_metric} | Paired delta | MC {secondary_metric} | Paired {secondary_metric} delta | MC NLL | Paired NLL delta | MC Brier | Paired Brier delta | MC ECE-15 | Paired ECE delta |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in values:
            robust_lines.append(
                f"| {row['seed']} | {_format(row.get('mc_'+task_metric))} | {_format(row.get('deterministic_'+task_metric))} | "
                f"{_format(row.get('delta_'+task_metric+'_mc_minus_deterministic'))} | {_format(row.get('mc_'+secondary_metric))} | "
                f"{_format(row.get('delta_'+secondary_metric+'_mc_minus_deterministic'))} | {_format(row.get('mc_nll'))} | "
                f"{_format(row.get('delta_nll_mc_minus_deterministic'))} | {_format(row.get('mc_brier'))} | "
                f"{_format(row.get('delta_brier_mc_minus_deterministic'))} | {_format(row.get('mc_ece_15'))} | "
                f"{_format(row.get('delta_ece_15_mc_minus_deterministic'))} |"
            )
        robust_lines.extend(["", "Paired-difference mean ± sample SD:", ""])
        for metric in (task_metric, secondary_metric, "nll", "brier", "ece_15"):
            numbers = np.asarray([row[f"delta_{metric}_mc_minus_deterministic"] for row in values], dtype=float)
            robust_lines.append(f"- `{metric}`: {numbers.mean():.6f} ± {numbers.std(ddof=1):.6f}")
        robust_lines.append("")
    robust_lines.extend(
        [
            "## Scope",
            "",
            "These three-seed summaries support robustness statements only for these four cells. They do not turn stochastic passes into sample size and do not justify seed-general claims for the other 12 cells.",
        ]
    )
    _write_text_exclusive(ROBUSTNESS_REPORT, "\n".join(robust_lines) + "\n")


def audit_training() -> dict[str, Any]:
    values = []
    for row in _load_registry()["runs"]:
        try:
            run_dir, summary, resolved = _find_training_run(row)
            audit = _training_audit(row, run_dir, summary, resolved)
        except Exception as exc:
            audit = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        values.append({"key": compact_key(row), **audit})
    return {
        "complete": sum(item.get("passed", False) for item in values),
        "expected": len(values),
        "runs": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inference = sub.add_parser("infer")
    inference.add_argument("--device", required=True)
    inference.add_argument("--key", action="append")
    sub.add_parser("audit-training")
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "infer":
        result = infer(select_registry_rows(args.key), args.device)
        print(json.dumps({"completed": len(result)}, indent=2))
    elif args.command == "audit-training":
        print(json.dumps(audit_training(), indent=2))
    else:
        values = finalize()
        print(json.dumps({"rows": len(values), "results_csv": str(RESULTS_CSV)}, indent=2))


if __name__ == "__main__":
    main()
