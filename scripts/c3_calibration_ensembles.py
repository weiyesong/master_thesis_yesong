from __future__ import annotations

"""C3 post-hoc calibration and probability-mean ensemble construction.

This module never imports a training loop or performs model inference.  Calibration
logits are produced separately with ``scripts.export_checkpoint_predictions``.
"""

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp

from scripts.experiment_manager import PROJECT_ROOT, write_code_snapshot_manifest
from scripts.segmentation_pipeline import SegmentationMetricAccumulator


N_BINS = 15
SEEDS = (42, 43, 44)
OUTPUT_ROOT = PROJECT_ROOT / "results/final_thesis/c3_temperature_scaling_and_ensembles"
CALIBRATION_ROOT = OUTPUT_ROOT / "classification/calibration_exports"
TREESATAI_VALIDATION_ROOT = OUTPUT_ROOT / "classification/validation_exports/treesatai"
REPORT_PATH = PROJECT_ROOT / "reports/c3_temperature_scaling_and_ensembles.md"
CLASSIFICATION_CSV = PROJECT_ROOT / "reports/c3_classification_results.csv"
SEGMENTATION_CSV = PROJECT_ROOT / "reports/c3_segmentation_ensemble_results.csv"
TREESATAI_TEMPERATURE_REPORT = PROJECT_ROOT / "reports/c3_treesatai_temperature_scaling.md"
TREESATAI_TEMPERATURE_CSV = PROJECT_ROOT / "reports/c3_treesatai_temperature_scaling_results.csv"
DOFA_FINAL_MANIFEST = PROJECT_ROOT / "reports/dofa_eurosat_final_manifest.json"
C1_AUDIT = PROJECT_ROOT / "reports/c1_classification_run_audits/20260816T063820Z/c1_run_audit.json"
C2_AUDIT = PROJECT_ROOT / "reports/c2_segmentation_run_audits/20260824T162250Z/audit.json"
EXISTING_DOFA_ENSEMBLE_ROOT = PROJECT_ROOT / "results/final_thesis/dofa_eurosat_ensembles"


@dataclass(frozen=True)
class Member:
    task: str
    dataset: str
    model: str
    adaptation: str
    seed: int
    run_id: str
    run_dir: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    prediction_path: Path
    config_path: Path | None = None
    experiment: str | None = None

    @property
    def cell_key(self) -> tuple[str, str, str]:
        return self.dataset, self.model, self.adaptation


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _exclusive_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to create empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Sigmoid logits contain NaN or Inf")
    return expit(values)


def classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classification_type: str,
    n_bins: int = N_BINS,
    threshold: float = 0.5,
) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or not np.isfinite(probabilities).all():
        raise ValueError("Classification probabilities must be a finite [N,C] matrix")
    if classification_type == "multiclass":
        if labels.shape != (len(probabilities),):
            raise ValueError("Multiclass labels must have shape [N]")
        predictions = probabilities.argmax(axis=1)
        accuracy = float(np.mean(predictions == labels))
        selected = np.clip(
            probabilities[np.arange(len(labels)), labels], np.finfo(np.float64).tiny, 1.0
        )
        nll = float(-np.log(selected).mean())
        targets = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
        brier = float(np.square(probabilities - targets).sum(axis=1).mean())
        confidence = probabilities.max(axis=1)
        outcomes = predictions == labels
        f1_values = []
        for class_index in range(probabilities.shape[1]):
            true_positive = np.sum((labels == class_index) & (predictions == class_index))
            predicted_count = np.sum(predictions == class_index)
            support = np.sum(labels == class_index)
            precision = true_positive / predicted_count if predicted_count else 0.0
            recall = true_positive / support if support else 0.0
            f1_values.append(
                2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            )
        macro_f1 = float(np.mean(f1_values))
    elif classification_type == "multilabel":
        if labels.shape != probabilities.shape or not np.isin(labels, (0, 1)).all():
            raise ValueError("Multilabel labels must be a binary [N,C] matrix")
        predictions = probabilities >= threshold
        accuracy = float(np.mean(np.all(predictions == labels, axis=1)))
        clipped = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0 - 1.0e-15)
        nll = float(
            -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped))
        )
        brier = float(np.mean(np.square(probabilities - labels)))
        f1_values = []
        for class_index in range(probabilities.shape[1]):
            target = labels[:, class_index].astype(bool)
            prediction = predictions[:, class_index]
            true_positive = np.sum(target & prediction)
            predicted_count = np.sum(prediction)
            support = np.sum(target)
            precision = true_positive / predicted_count if predicted_count else 0.0
            recall = true_positive / support if support else 0.0
            f1_values.append(
                2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            )
        macro_f1 = float(np.mean(f1_values))
        confidence = np.maximum(probabilities, 1.0 - probabilities).reshape(-1)
        outcomes = (predictions == labels).reshape(-1)
    else:
        raise ValueError(f"Unsupported classification type: {classification_type}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= lower if index == 0 else confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(outcomes[mask].mean()) - float(confidence[mask].mean()))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "nll": nll,
        "brier": brier,
        "ece_15": float(ece),
    }


def fit_positive_temperature(logits: np.ndarray, labels: np.ndarray) -> dict[str, float | int | bool]:
    """Fit log(T) on multiclass calibration NLL, guaranteeing T > 0."""

    logits64 = np.asarray(logits, dtype=np.float64)
    labels64 = np.asarray(labels, dtype=np.int64)
    if logits64.ndim != 2 or labels64.shape != (len(logits64),):
        raise ValueError("Temperature fitting requires multiclass logits [N,C] and labels [N]")
    if not np.isfinite(logits64).all() or labels64.min() < 0 or labels64.max() >= logits64.shape[1]:
        raise ValueError("Temperature fitting received invalid logits or labels")

    def objective(log_temperature: float) -> float:
        inverse_temperature = math.exp(-float(log_temperature))
        scaled = logits64 * inverse_temperature
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels64)), labels64]))

    bounds = (-7.0, 7.0)
    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=bounds,
        options={"xatol": 1.0e-12, "maxiter": 1000},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError(f"Positive temperature optimization failed: {result.message}")
    log_temperature = float(result.x)
    if min(abs(log_temperature - bound) for bound in bounds) < 1.0e-5:
        raise RuntimeError("Positive temperature optimum reached the safety bound")
    temperature = math.exp(log_temperature)
    raw_nll = objective(0.0)
    scaled_nll = objective(log_temperature)
    if scaled_nll > raw_nll + 1.0e-10:
        raise RuntimeError("Temperature optimization increased calibration NLL")
    return {
        "temperature": temperature,
        "log_temperature": log_temperature,
        "calibration_nll_before": raw_nll,
        "calibration_nll_after": scaled_nll,
        "optimizer_success": bool(result.success),
        "optimizer_iterations": int(result.nfev),
    }


def fit_positive_multilabel_temperature(
    logits: np.ndarray, labels: np.ndarray
) -> dict[str, float | int | bool]:
    """Fit one positive scalar by mean BCE-with-logits over sample-label decisions."""

    logits64 = np.asarray(logits, dtype=np.float64)
    labels64 = np.asarray(labels, dtype=np.float64)
    if logits64.ndim != 2 or labels64.shape != logits64.shape:
        raise ValueError("Multilabel temperature fitting requires matching [N,C] logits and labels")
    if not np.isfinite(logits64).all() or not np.isin(labels64, (0.0, 1.0)).all():
        raise ValueError("Multilabel temperature fitting received invalid logits or labels")

    def objective(log_temperature: float) -> float:
        scaled = logits64 * math.exp(-float(log_temperature))
        return float(np.mean(np.logaddexp(0.0, scaled) - labels64 * scaled))

    bounds = (-7.0, 7.0)
    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=bounds,
        options={"xatol": 1.0e-12, "maxiter": 1000},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError(f"Positive multilabel temperature optimization failed: {result.message}")
    log_temperature = float(result.x)
    if min(abs(log_temperature - bound) for bound in bounds) < 1.0e-5:
        raise RuntimeError("Positive multilabel temperature optimum reached the safety bound")
    temperature = math.exp(log_temperature)
    fit_nll_before = objective(0.0)
    fit_nll_after = objective(log_temperature)
    if fit_nll_after > fit_nll_before + 1.0e-10:
        raise RuntimeError("Multilabel temperature optimization increased fit-split NLL")
    return {
        "temperature": temperature,
        "log_temperature": log_temperature,
        "fit_nll_before": fit_nll_before,
        "fit_nll_after": fit_nll_after,
        "optimizer_success": bool(result.success),
        "optimizer_iterations": int(result.nfev),
    }


def multilabel_reliability_bins(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS
) -> dict[str, np.ndarray | float]:
    """Reliability data for flattened binary decisions under the thesis ECE convention."""

    probabilities64 = np.asarray(probabilities, dtype=np.float64)
    labels64 = np.asarray(labels, dtype=np.int64)
    if probabilities64.ndim != 2 or labels64.shape != probabilities64.shape:
        raise ValueError("Multilabel reliability requires matching [N,C] arrays")
    if not np.isfinite(probabilities64).all() or not np.isin(labels64, (0, 1)).all():
        raise ValueError("Invalid multilabel reliability inputs")
    predictions = probabilities64 >= 0.5
    confidence = np.maximum(probabilities64, 1.0 - probabilities64).reshape(-1)
    outcomes = (predictions == labels64).reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    accuracy = np.full(n_bins, np.nan, dtype=np.float64)
    mean_confidence = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    ece = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= lower if index == 0 else confidence > lower) & (confidence <= upper)
        counts[index] = int(mask.sum())
        if mask.any():
            accuracy[index] = float(outcomes[mask].mean())
            mean_confidence[index] = float(confidence[mask].mean())
            ece += float(mask.mean()) * abs(accuracy[index] - mean_confidence[index])
    return {
        "edges": edges,
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "counts": counts,
        "ece_15": float(ece),
    }


def plot_paired_multilabel_reliability(
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    n_bins: int = N_BINS,
) -> None:
    raw = multilabel_reliability_bins(raw_probabilities, labels, n_bins)
    calibrated = multilabel_reliability_bins(calibrated_probabilities, labels, n_bins)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    centers = (np.asarray(raw["edges"])[:-1] + np.asarray(raw["edges"])[1:]) / 2.0
    width = 0.92 / n_bins
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True, sharey=True, constrained_layout=True)
    for axis, title, bins in (
        (axes[0], "Raw", raw),
        (axes[1], "Temperature-scaled", calibrated),
    ):
        nonempty = np.asarray(bins["counts"]) > 0
        accuracy = np.asarray(bins["accuracy"])
        mean_confidence = np.asarray(bins["mean_confidence"])
        axis.bar(
            centers[nonempty], accuracy[nonempty], width=width, color="#9ecae1",
            edgecolor="#2b6c8a", linewidth=0.8, label="Binary-decision accuracy",
        )
        axis.scatter(
            mean_confidence[nonempty], accuracy[nonempty], color="#08306b", s=18,
            zorder=3, label="Bin mean",
        )
        axis.plot([0, 1], [0, 1], "--", color="0.25", linewidth=1.1, label="Perfect calibration")
        axis.set_title(f"{title} (ECE-15={float(bins['ece_15']):.4f})")
        axis.set_xlim(0.5, 1.0)
        axis.set_ylim(0.5, 1.0)
        axis.grid(axis="y", color="0.88", linewidth=0.8)
        axis.set_xlabel("Binary-decision confidence")
    axes[0].set_ylabel("Binary-decision accuracy")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("TreeSatAI test reliability: raw vs validation-fitted temperature scaling")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def classification_registry() -> list[Member]:
    immutable = json.loads(DOFA_FINAL_MANIFEST.read_text(encoding="utf-8"))
    if immutable.get("status") != "FINAL_IMMUTABLE" or not immutable.get("immutability", {}).get("verified"):
        raise ValueError("DOFA-EuroSAT immutable manifest is not verified")
    members: list[Member] = []
    for row in immutable["runs"]:
        config_path = Path(row["config"])
        config = _read_yaml(config_path)
        run_dir = Path(row["checkpoint_path"]).parent
        members.append(
            Member(
                task="classification",
                dataset="eurosat",
                model="dofa",
                adaptation=row["adaptation"],
                seed=int(row["seed"]),
                run_id=row["run_id"],
                run_dir=run_dir,
                checkpoint_path=Path(row["checkpoint_path"]),
                checkpoint_sha256=row["checkpoint_sha256"],
                prediction_path=Path(row["test_prediction_path"]),
                config_path=config_path,
                experiment=config["active_experiment"],
            )
        )

    audit = json.loads(C1_AUDIT.read_text(encoding="utf-8"))
    if audit.get("summary", {}).get("promotable") != 18:
        raise ValueError("C1 audit does not contain exactly 18 promotable runs")
    for row in audit["runs"]:
        if row["status"] != "PROMOTABLE":
            raise ValueError(f"Non-promotable C1 member: {row['run_id']}")
        target = row["target"]
        run_dir = Path(row["run_dir"])
        members.append(
            Member(
                task="classification",
                dataset=target["dataset"],
                model=target["model"],
                adaptation=target["adaptation"],
                seed=int(target["seed"]),
                run_id=row["run_id"],
                run_dir=run_dir,
                checkpoint_path=Path(row["checkpoint"]["path"]),
                checkpoint_sha256=row["checkpoint"]["sha256"],
                prediction_path=Path(row["prediction_export"]["path"]) / "predictions.parquet",
                config_path=run_dir / "resolved_config.yaml",
                experiment=target["experiment"],
            )
        )
    _validate_registry(members, expected_cells=8, expected_members=24)
    return members


def segmentation_registry() -> list[Member]:
    audit = json.loads(C2_AUDIT.read_text(encoding="utf-8"))
    if not isinstance(audit, list) or len(audit) != 24:
        raise ValueError("C2 audit must contain exactly 24 runs")
    members = []
    for row in audit:
        if row["status"] != "PROMOTABLE":
            raise ValueError(f"Non-promotable C2 member: {row['run_id']}")
        run_dir = Path(row["run_dir"])
        prediction_path = run_dir / "predictions/test/deterministic/predictions.npz"
        manifest_path = prediction_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest["source"]
        members.append(
            Member(
                task="segmentation",
                dataset=row["dataset"],
                model=row["model"],
                adaptation=row["adaptation"],
                seed=int(row["seed"]),
                run_id=row["run_id"],
                run_dir=run_dir,
                checkpoint_path=run_dir / "best.pt",
                checkpoint_sha256=source["checkpoint_sha256"],
                prediction_path=prediction_path,
                config_path=run_dir / "resolved_config.yaml",
            )
        )
    _validate_registry(members, expected_cells=8, expected_members=24)
    return members


def _validate_registry(members: Sequence[Member], expected_cells: int, expected_members: int) -> None:
    if len(members) != expected_members:
        raise ValueError(f"Registry contains {len(members)} members, expected {expected_members}")
    cells: dict[tuple[str, str, str], list[int]] = {}
    for member in members:
        cells.setdefault(member.cell_key, []).append(member.seed)
        for path in (member.run_dir, member.checkpoint_path, member.prediction_path):
            if not path.exists():
                raise FileNotFoundError(path)
    if len(cells) != expected_cells:
        raise ValueError(f"Registry contains {len(cells)} cells, expected {expected_cells}")
    for key, seeds in cells.items():
        if sorted(seeds) != list(SEEDS):
            raise ValueError(f"Cell {key} has seeds {sorted(seeds)}, expected {SEEDS}")


def calibration_export_path(member: Member) -> Path:
    return CALIBRATION_ROOT / member.model / member.adaptation / f"seed{member.seed}"


def prepare_calibration_configs() -> list[dict[str, Any]]:
    """Create derived inference-only configs without touching immutable run metadata."""

    registry_dir = OUTPUT_ROOT / "classification/calibration_configs"
    if registry_dir.exists():
        raise FileExistsError(f"Refusing to overwrite derived calibration configs: {registry_dir}")
    registry_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for member in sorted(
        (value for value in classification_registry() if value.dataset == "eurosat"),
        key=lambda value: (value.model, value.adaptation, value.seed),
    ):
        assert member.config_path is not None
        source_hash = sha256_file(member.config_path)
        output_path = registry_dir / f"{member.model}_{member.adaptation}_seed{member.seed}.yaml"
        config = _read_yaml(member.config_path)
        changes: dict[str, Any] = {}
        if member.model == "dofa":
            if config["data"].get("wavelengths_nm") is not None:
                raise ValueError(f"Historical DOFA config unexpectedly already has wavelengths: {member.config_path}")
            config["data"]["wavelengths_nm"] = [665.0, 560.0, 490.0]
            config["data"]["wavelength_source"] = (
                "historical DOFA RGB wrapper constants: scripts.run_experiments.SENTINEL2_RGB_WAVELENGTHS"
            )
            changes = {
                "data.wavelengths_nm": [665.0, 560.0, 490.0],
                "model_actual_wavelengths_after_adapter_um": [0.665, 0.560, 0.490],
                "reason": (
                    "Current inference asserts explicit canonical nm metadata; the immutable training config predates "
                    "that field and the historical DOFA wrapper used these exact RGB micrometer constants."
                ),
            }
        config["c3_inference_derivation"] = {
            "source_config_path": str(member.config_path.resolve()),
            "source_config_sha256": source_hash,
            "historical_metadata_modified": False,
            "changes": changes,
        }
        with output_path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        rows.append(
            {
                "run_id": member.run_id,
                "model": member.model,
                "adaptation": member.adaptation,
                "seed": member.seed,
                "source_config": str(member.config_path.resolve()),
                "source_config_sha256": source_hash,
                "derived_config": str(output_path.resolve()),
                "derived_config_sha256": sha256_file(output_path),
                "changes": changes,
            }
        )
    _exclusive_json(registry_dir / "manifest.json", rows)
    return rows


def derived_calibration_config_path(member: Member) -> Path:
    return OUTPUT_ROOT / "classification/calibration_configs" / (
        f"{member.model}_{member.adaptation}_seed{member.seed}.yaml"
    )


def calibration_export_commands(shard: int, shards: int, device: str) -> list[str]:
    if not 0 <= shard < shards:
        raise ValueError("Invalid calibration shard")
    eurosat = sorted(
        (member for member in classification_registry() if member.dataset == "eurosat"),
        key=lambda item: (item.model, item.adaptation, item.seed),
    )
    commands = []
    for index, member in enumerate(eurosat):
        if index % shards != shard:
            continue
        output = calibration_export_path(member)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite calibration export: {output}")
        config_path = derived_calibration_config_path(member)
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Derived inference config is missing; run prepare-calibration-configs first: {config_path}"
            )
        commands.append(
            "python -m scripts.export_checkpoint_predictions "
            f"--config {config_path} --checkpoint {member.checkpoint_path} "
            f"--experiment {member.experiment} --split calibration --output-dir {output} --device {device}"
        )
    return commands


def treesatai_validation_export_path(member: Member) -> Path:
    return TREESATAI_VALIDATION_ROOT / member.model / member.adaptation / f"seed{member.seed}"


def treesatai_validation_export_commands(shard: int, shards: int, device: str) -> list[str]:
    """Build forward-only commands for the official TreeSatAI validation split."""

    if not 0 <= shard < shards:
        raise ValueError("Invalid TreeSatAI validation-export shard")
    members = sorted(
        (member for member in classification_registry() if member.dataset == "treesatai"),
        key=lambda item: (item.model, item.adaptation, item.seed),
    )
    commands = []
    for index, member in enumerate(members):
        if index % shards != shard:
            continue
        output = treesatai_validation_export_path(member)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite TreeSatAI validation export: {output}")
        if member.config_path is None or not member.config_path.is_file():
            raise FileNotFoundError(f"Missing validated TreeSatAI config: {member.config_path}")
        commands.append(
            "python -m scripts.export_checkpoint_predictions "
            f"--config {member.config_path} --checkpoint {member.checkpoint_path} "
            f"--experiment {member.experiment} --split val --output-dir {output} --device {device}"
        )
    return commands


def _load_classification_prediction(member: Member, path: Path | None = None) -> dict[str, Any]:
    prediction_path = path or member.prediction_path
    table = pd.read_parquet(prediction_path, engine="pyarrow")
    required = {"sample_id", "logits", "probabilities"}
    if not required <= set(table.columns):
        raise ValueError(f"Missing columns in {prediction_path}: {sorted(required - set(table.columns))}")
    label_column = "label" if "label" in table.columns else "true_label"
    labels = np.asarray(table[label_column].tolist(), dtype=np.int64)
    logits = np.asarray(table["logits"].tolist(), dtype=np.float64)
    probabilities = np.asarray(table["probabilities"].tolist(), dtype=np.float64)
    sample_ids = table["sample_id"].astype(str).to_numpy()
    classification_type = "multilabel" if labels.ndim == 2 else "multiclass"
    expected = stable_softmax(logits) if classification_type == "multiclass" else stable_sigmoid(logits)
    if not np.allclose(probabilities, expected, atol=2.0e-6, rtol=1.0e-6):
        raise ValueError(f"Saved probabilities do not match logits in {prediction_path}")
    if len(set(sample_ids.tolist())) != len(sample_ids) or not np.isfinite(logits).all():
        raise ValueError(f"Invalid samples/logits in {prediction_path}")
    return {
        "path": prediction_path,
        "table": table,
        "sample_ids": sample_ids,
        "labels": labels,
        "logits": logits,
        "probabilities": probabilities,
        "classification_type": classification_type,
    }


def _classification_summaries(probabilities: np.ndarray, classification_type: str) -> dict[str, np.ndarray]:
    safe = np.clip(probabilities.astype(np.float64), np.finfo(np.float64).tiny, 1.0)
    if classification_type == "multiclass":
        prediction = probabilities.argmax(axis=1).astype(np.int64)
        confidence = probabilities.max(axis=1)
        entropy = -(probabilities * np.log(safe)).sum(axis=1)
    else:
        prediction = (probabilities >= 0.5).astype(np.int8)
        confidence = np.maximum(probabilities, 1.0 - probabilities).mean(axis=1)
        entropy = -(
            probabilities * np.log(safe)
            + (1.0 - probabilities)
            * np.log(np.clip(1.0 - probabilities, np.finfo(np.float64).tiny, 1.0))
        ).sum(axis=1)
    return {"prediction": prediction, "confidence": confidence, "entropy": entropy}


def _write_classification_ensemble(
    members: Sequence[Member],
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite ensemble: {output_dir}")
    loaded = [_load_classification_prediction(member) for member in members]
    reference = loaded[0]
    for value in loaded[1:]:
        if not np.array_equal(value["sample_ids"], reference["sample_ids"]):
            raise ValueError("Classification ensemble sample IDs/order differ")
        if not np.array_equal(value["labels"], reference["labels"]):
            raise ValueError("Classification ensemble labels differ")
        if value["classification_type"] != reference["classification_type"]:
            raise ValueError("Classification ensemble task types differ")
    member_probabilities = np.stack([value["probabilities"] for value in loaded], axis=1).astype(np.float32)
    member_logits = np.stack([value["logits"] for value in loaded], axis=1).astype(np.float32)
    ensemble_probabilities = member_probabilities.astype(np.float64).mean(axis=1)
    if reference["classification_type"] == "multiclass":
        if not np.allclose(ensemble_probabilities.sum(axis=1), 1.0, atol=2.0e-6, rtol=0):
            raise ValueError("Classification ensemble probabilities are not normalized")
    elif not ((ensemble_probabilities >= 0).all() and (ensemble_probabilities <= 1).all()):
        raise ValueError("Multilabel ensemble probabilities leave [0,1]")
    metrics = classification_metrics(
        ensemble_probabilities, reference["labels"], reference["classification_type"]
    )
    summaries = _classification_summaries(ensemble_probabilities, reference["classification_type"])
    output_dir.mkdir(parents=True, exist_ok=False)
    member_path = output_dir / "member_predictions.npz"
    np.savez_compressed(
        member_path,
        sample_id=reference["sample_ids"],
        label=reference["labels"],
        seeds=np.asarray([member.seed for member in members], dtype=np.int64),
        run_ids=np.asarray([member.run_id for member in members]),
        logits=member_logits,
        probabilities=member_probabilities,
    )
    ensemble_npz = output_dir / "ensemble_probabilities.npz"
    np.savez_compressed(
        ensemble_npz,
        sample_id=reference["sample_ids"],
        label=reference["labels"],
        probabilities=ensemble_probabilities.astype(np.float32),
        prediction=summaries["prediction"],
        confidence=summaries["confidence"].astype(np.float32),
        predictive_entropy=summaries["entropy"].astype(np.float32),
    )
    labels = reference["labels"]
    prediction = summaries["prediction"]
    correctness = (
        prediction == labels
        if reference["classification_type"] == "multiclass"
        else np.all(prediction == labels, axis=1)
    )
    table = pd.DataFrame(
        {
            "sample_id": reference["sample_ids"],
            "label": labels.tolist() if labels.ndim == 2 else labels,
            "probabilities": [row.tolist() for row in ensemble_probabilities],
            "prediction": prediction.tolist() if prediction.ndim == 2 else prediction,
            "correctness": correctness,
            "confidence": summaries["confidence"].astype(np.float32),
            "predictive_entropy": summaries["entropy"].astype(np.float32),
            "aggregation": "arithmetic_mean_of_member_probabilities",
        }
    )
    table_path = output_dir / "ensemble_predictions.parquet"
    arrow = pa.Table.from_pandas(table, preserve_index=False)
    index = arrow.schema.get_field_index("probabilities")
    arrow = arrow.set_column(
        index,
        "probabilities",
        pa.array([row.tolist() for row in ensemble_probabilities], type=pa.list_(pa.float64())),
    )
    pq.write_table(arrow, table_path, compression="zstd")
    _exclusive_json(output_dir / "metrics.json", metrics)
    source_rows = []
    for member, value in zip(members, loaded):
        actual_checkpoint = sha256_file(member.checkpoint_path)
        if actual_checkpoint != member.checkpoint_sha256:
            raise ValueError(f"Checkpoint hash mismatch for {member.run_id}")
        source_rows.append(
            {
                "seed": member.seed,
                "run_id": member.run_id,
                "checkpoint_path": str(member.checkpoint_path.resolve()),
                "checkpoint_sha256": actual_checkpoint,
                "prediction_path": str(member.prediction_path.resolve()),
                "prediction_sha256": sha256_file(member.prediction_path),
            }
        )
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "task": "classification",
        "dataset": members[0].dataset,
        "model": members[0].model,
        "adaptation": members[0].adaptation,
        "classification_type": reference["classification_type"],
        "split": "test",
        "sample_count": int(len(labels)),
        "member_seeds": list(SEEDS),
        "members": source_rows,
        "aggregation": "arithmetic mean of member probabilities; member logits were never averaged",
        "arrays": {
            "member_logits": [*member_logits.shape],
            "member_probabilities": [*member_probabilities.shape],
            "ensemble_probabilities": [*ensemble_probabilities.shape],
        },
        "metrics": metrics,
    }
    _exclusive_json(output_dir / "manifest.json", manifest)
    return manifest


def build_classification_ensembles() -> list[dict[str, Any]]:
    members = classification_registry()
    rows: list[dict[str, Any]] = []
    for key in sorted({member.cell_key for member in members}):
        dataset, model, adaptation = key
        cell = sorted((member for member in members if member.cell_key == key), key=lambda value: value.seed)
        if dataset == "eurosat" and model == "dofa":
            existing = EXISTING_DOFA_ENSEMBLE_ROOT / adaptation
            manifest_path = existing / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if [row["seed"] for row in manifest["member_axis_order"]] != list(SEEDS):
                raise ValueError(f"Existing DOFA ensemble seed order is invalid: {adaptation}")
            # Recompute from the immutable Parquets to prove the existing artifact is direct aggregation.
            loaded = [_load_classification_prediction(member) for member in cell]
            probabilities = np.stack([item["probabilities"] for item in loaded]).mean(axis=0)
            stored = np.load(existing / "ensemble_probabilities.npz", allow_pickle=False)["probabilities"]
            if not np.allclose(probabilities, stored, atol=1.0e-7, rtol=0):
                raise ValueError(f"Existing DOFA ensemble does not match direct Parquet probability mean: {adaptation}")
            metrics = classification_metrics(probabilities, loaded[0]["labels"], "multiclass")
            for name, value in metrics.items():
                if not math.isclose(value, float(manifest["metrics"].get(name, value)), abs_tol=1.0e-10):
                    raise ValueError(f"Existing DOFA ensemble metric mismatch: {adaptation}/{name}")
            output_path = existing
            reuse = True
        else:
            output_path = OUTPUT_ROOT / "classification/deep_ensembles" / dataset / model / adaptation
            manifest = _write_classification_ensemble(cell, output_path)
            metrics = manifest["metrics"]
            reuse = False
        rows.append(
            {
                "task": "classification",
                "dataset": dataset,
                "model": model,
                "adaptation": adaptation,
                "method": "deep_ensemble_probability_mean",
                "seeds": "42;43;44",
                **metrics,
                "output_path": str(output_path.resolve()),
                "reused_existing_artifact": reuse,
            }
        )
    _exclusive_json(OUTPUT_ROOT / "classification/deep_ensemble_summary.json", rows)
    return rows


def build_temperature_scaling() -> list[dict[str, Any]]:
    members = sorted(
        (member for member in classification_registry() if member.dataset == "eurosat"),
        key=lambda item: (item.model, item.adaptation, item.seed),
    )
    rows = []
    by_cell_calibration: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for member in members:
        calibration_path = calibration_export_path(member) / "predictions.parquet"
        if not calibration_path.is_file():
            raise FileNotFoundError(f"Missing dedicated calibration export: {calibration_path}")
        calibration = _load_classification_prediction(member, calibration_path)
        test = _load_classification_prediction(member)
        if calibration["classification_type"] != "multiclass" or test["classification_type"] != "multiclass":
            raise ValueError("EuroSAT Temperature Scaling must be multiclass")
        if len(calibration["sample_ids"]) != 2713 or len(test["sample_ids"]) != 2714:
            raise ValueError("EuroSAT calibration/test sample count differs from the frozen manifest")
        if set(calibration["sample_ids"]) & set(test["sample_ids"]):
            raise ValueError("Calibration and test sample IDs overlap")
        reference = by_cell_calibration.setdefault(
            member.cell_key, (calibration["sample_ids"], calibration["labels"])
        )
        if not np.array_equal(calibration["sample_ids"], reference[0]) or not np.array_equal(
            calibration["labels"], reference[1]
        ):
            raise ValueError(f"Calibration samples/labels differ across seeds for {member.cell_key}")
        fit = fit_positive_temperature(calibration["logits"], calibration["labels"])
        temperature = float(fit["temperature"])
        raw_probabilities = stable_softmax(test["logits"])
        scaled_logits = test["logits"] / temperature
        scaled_probabilities = stable_softmax(scaled_logits)
        before = classification_metrics(raw_probabilities, test["labels"], "multiclass")
        after = classification_metrics(scaled_probabilities, test["labels"], "multiclass")
        if before["accuracy"] != after["accuracy"]:
            raise ValueError("Positive scalar temperature unexpectedly changed accuracy")
        output_dir = OUTPUT_ROOT / "classification/temperature_scaling/eurosat" / member.model / member.adaptation / f"seed{member.seed}"
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite Temperature Scaling output: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=False)
        summaries = _classification_summaries(scaled_probabilities, "multiclass")
        table = pd.DataFrame(
            {
                "sample_id": test["sample_ids"],
                "label": test["labels"],
                "raw_logits": [row.tolist() for row in test["logits"]],
                "scaled_logits": [row.tolist() for row in scaled_logits],
                "probabilities": [row.tolist() for row in scaled_probabilities],
                "prediction": summaries["prediction"],
                "correctness": summaries["prediction"] == test["labels"],
                "confidence": summaries["confidence"].astype(np.float32),
                "predictive_entropy": summaries["entropy"].astype(np.float32),
                "temperature": temperature,
            }
        )
        table_path = output_dir / "test_predictions.parquet"
        arrow = pa.Table.from_pandas(table, preserve_index=False)
        for name, values in (
            ("raw_logits", test["logits"]),
            ("scaled_logits", scaled_logits),
            ("probabilities", scaled_probabilities),
        ):
            index = arrow.schema.get_field_index(name)
            arrow = arrow.set_column(
                index, name, pa.array([row.tolist() for row in values], type=pa.list_(pa.float64()))
            )
        pq.write_table(arrow, table_path, compression="zstd")
        actual_checkpoint = sha256_file(member.checkpoint_path)
        if actual_checkpoint != member.checkpoint_sha256:
            raise ValueError(f"Checkpoint hash mismatch for {member.run_id}")
        payload = {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "dataset": "eurosat",
            "model": member.model,
            "adaptation": member.adaptation,
            "seed": member.seed,
            "run_id": member.run_id,
            "fit_split": "calibration",
            "evaluation_split": "test",
            "fit_sample_count": int(len(calibration["labels"])),
            "test_sample_count": int(len(test["labels"])),
            "fit_objective": "multiclass NLL",
            "parameterization": "T=exp(log_temperature), guaranteeing T>0",
            "checkpoint_path": str(member.checkpoint_path.resolve()),
            "checkpoint_sha256": actual_checkpoint,
            "calibration_prediction_path": str(calibration_path.resolve()),
            "calibration_prediction_sha256": sha256_file(calibration_path),
            "test_prediction_path": str(member.prediction_path.resolve()),
            "test_prediction_sha256": sha256_file(member.prediction_path),
            "fit": fit,
            "test_metrics_before": before,
            "test_metrics_after": after,
            "output": table_path.name,
        }
        _exclusive_json(output_dir / "metrics_and_manifest.json", payload)
        rows.append(
            {
                "task": "classification",
                "dataset": "eurosat",
                "model": member.model,
                "adaptation": member.adaptation,
                "method": "temperature_scaling",
                "seed": member.seed,
                "temperature": temperature,
                "accuracy_before": before["accuracy"],
                "accuracy_after": after["accuracy"],
                "nll_before": before["nll"],
                "nll_after": after["nll"],
                "brier_before": before["brier"],
                "brier_after": after["brier"],
                "ece_15_before": before["ece_15"],
                "ece_15_after": after["ece_15"],
                "output_path": str(output_dir.resolve()),
            }
        )
    _exclusive_json(OUTPUT_ROOT / "classification/temperature_scaling_summary.json", rows)
    return rows


def build_treesatai_temperature_scaling() -> list[dict[str, Any]]:
    """Fit validation-BCE scalar temperatures and apply them to saved test logits."""

    members = sorted(
        (member for member in classification_registry() if member.dataset == "treesatai"),
        key=lambda item: (item.model, item.adaptation, item.seed),
    )
    rows: list[dict[str, Any]] = []
    reference_validation: tuple[np.ndarray, np.ndarray] | None = None
    for member in members:
        validation_dir = treesatai_validation_export_path(member)
        validation_path = validation_dir / "predictions.parquet"
        if not validation_path.is_file():
            raise FileNotFoundError(f"Missing official-validation logits: {validation_path}")
        validation_report = json.loads((validation_dir / "validation_report.json").read_text(encoding="utf-8"))
        validation_manifest = json.loads((validation_dir / "manifest.json").read_text(encoding="utf-8"))
        source = validation_manifest["source"]
        if not validation_report.get("valid") or validation_manifest.get("partial_export"):
            raise ValueError(f"Invalid or partial validation export: {member.run_id}")
        for name, expected in (
            ("dataset", "treesatai"),
            ("adaptation_mode", member.adaptation),
            ("seed", member.seed),
            ("split", "val"),
            ("checkpoint_run_id", member.run_id),
            ("checkpoint_sha256", member.checkpoint_sha256),
        ):
            if source.get(name) != expected:
                raise ValueError(
                    f"TreeSatAI validation provenance mismatch for {member.run_id}/{name}: "
                    f"{source.get(name)!r} != {expected!r}"
                )

        validation = _load_classification_prediction(member, validation_path)
        test = _load_classification_prediction(member)
        if validation["classification_type"] != "multilabel" or test["classification_type"] != "multilabel":
            raise ValueError("TreeSatAI Temperature Scaling must follow the multilabel protocol")
        if validation["labels"].shape != (1000, 15) or test["labels"].shape != (2000, 15):
            raise ValueError("TreeSatAI validation/test shapes differ from the downloaded official artifact")
        if set(validation["sample_ids"]) & set(test["sample_ids"]):
            raise ValueError("TreeSatAI official validation and test sample IDs overlap")
        if reference_validation is None:
            reference_validation = (validation["sample_ids"], validation["labels"])
        elif not np.array_equal(validation["sample_ids"], reference_validation[0]) or not np.array_equal(
            validation["labels"], reference_validation[1]
        ):
            raise ValueError("TreeSatAI validation sample order or labels differ across checkpoints")

        fit = fit_positive_multilabel_temperature(validation["logits"], validation["labels"])
        temperature = float(fit["temperature"])
        raw_probabilities = stable_sigmoid(test["logits"])
        scaled_logits = test["logits"] / temperature
        calibrated_probabilities = stable_sigmoid(scaled_logits)
        before = classification_metrics(raw_probabilities, test["labels"], "multilabel")
        after = classification_metrics(calibrated_probabilities, test["labels"], "multilabel")
        raw_prediction = raw_probabilities >= 0.5
        calibrated_prediction = calibrated_probabilities >= 0.5
        threshold_predictions_unchanged = bool(np.array_equal(raw_prediction, calibrated_prediction))
        argmax_predictions_unchanged = bool(
            np.array_equal(raw_probabilities.argmax(axis=1), calibrated_probabilities.argmax(axis=1))
        )
        if not threshold_predictions_unchanged or not argmax_predictions_unchanged:
            raise ValueError("Positive scalar temperature changed TreeSatAI predicted labels")
        if before["accuracy"] != after["accuracy"] or before["macro_f1"] != after["macro_f1"]:
            raise ValueError("Prediction-preserving temperature changed TreeSatAI discrete metrics")

        output_dir = (
            OUTPUT_ROOT
            / "classification/temperature_scaling/treesatai"
            / member.model
            / member.adaptation
            / f"seed{member.seed}"
        )
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite TreeSatAI Temperature Scaling output: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=False)
        raw_summaries = _classification_summaries(raw_probabilities, "multilabel")
        calibrated_summaries = _classification_summaries(calibrated_probabilities, "multilabel")
        labels = test["labels"]
        table = pd.DataFrame(
            {
                "sample_id": test["sample_ids"],
                "label": [row.tolist() for row in labels],
                "raw_logits": [row.tolist() for row in test["logits"]],
                "scaled_logits": [row.tolist() for row in scaled_logits],
                "raw_probabilities": [row.tolist() for row in raw_probabilities],
                "calibrated_probabilities": [row.tolist() for row in calibrated_probabilities],
                "raw_prediction": [row.tolist() for row in raw_prediction.astype(np.int8)],
                "calibrated_prediction": [row.tolist() for row in calibrated_prediction.astype(np.int8)],
                "raw_correctness": np.all(raw_prediction == labels, axis=1),
                "calibrated_correctness": np.all(calibrated_prediction == labels, axis=1),
                "raw_confidence": raw_summaries["confidence"].astype(np.float32),
                "calibrated_confidence": calibrated_summaries["confidence"].astype(np.float32),
                "raw_predictive_entropy": raw_summaries["entropy"].astype(np.float32),
                "calibrated_predictive_entropy": calibrated_summaries["entropy"].astype(np.float32),
                "temperature": temperature,
            }
        )
        table_path = output_dir / "test_predictions_raw_and_calibrated.parquet"
        arrow = pa.Table.from_pandas(table, preserve_index=False)
        for name, values in (
            ("raw_logits", test["logits"]),
            ("scaled_logits", scaled_logits),
            ("raw_probabilities", raw_probabilities),
            ("calibrated_probabilities", calibrated_probabilities),
        ):
            index = arrow.schema.get_field_index(name)
            arrow = arrow.set_column(
                index, name, pa.array([row.tolist() for row in values], type=pa.list_(pa.float64()))
            )
        pq.write_table(arrow, table_path, compression="zstd")
        probability_path = output_dir / "test_probabilities_raw_and_calibrated.npz"
        np.savez_compressed(
            probability_path,
            sample_id=test["sample_ids"],
            label=labels,
            raw_probabilities=raw_probabilities.astype(np.float32),
            calibrated_probabilities=calibrated_probabilities.astype(np.float32),
            raw_prediction=raw_prediction.astype(np.int8),
            calibrated_prediction=calibrated_prediction.astype(np.int8),
            temperature=np.asarray(temperature, dtype=np.float64),
        )
        reliability_path = output_dir / "reliability_raw_vs_calibrated.png"
        plot_paired_multilabel_reliability(
            raw_probabilities, calibrated_probabilities, labels, reliability_path, n_bins=N_BINS
        )

        actual_checkpoint_hash = sha256_file(member.checkpoint_path)
        if actual_checkpoint_hash != member.checkpoint_sha256:
            raise ValueError(f"Checkpoint hash mismatch for {member.run_id}")
        payload = {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "dataset": "treesatai",
            "classification_type": "multilabel",
            "model": member.model,
            "adaptation": member.adaptation,
            "seed": member.seed,
            "run_id": member.run_id,
            "fit_split": "official_validation",
            "fit_split_was_also_used_for_checkpoint_selection": True,
            "evaluation_split": "untouched_official_test",
            "protocol_difference": (
                "TreeSatAI has no independent official calibration split. Under the final thesis protocol, "
                "Temperature Scaling is therefore fitted on official validation logits; EuroSAT instead uses "
                "its dedicated calibration split. No temperature is selected or rejected from test results."
            ),
            "fit_sample_count": int(len(validation["labels"])),
            "fit_decision_count": int(validation["labels"].size),
            "test_sample_count": int(len(labels)),
            "test_decision_count": int(labels.size),
            "fit_objective": "mean binary cross-entropy with logits over sample-label decisions",
            "parameterization": "T=exp(log_temperature), guaranteeing T>0",
            "checkpoint_path": str(member.checkpoint_path.resolve()),
            "checkpoint_sha256": actual_checkpoint_hash,
            "validation_prediction_path": str(validation_path.resolve()),
            "validation_prediction_sha256": sha256_file(validation_path),
            "test_prediction_path": str(member.prediction_path.resolve()),
            "test_prediction_sha256": sha256_file(member.prediction_path),
            "fit": fit,
            "test_metrics_before": before,
            "test_metrics_after": after,
            "invariants": {
                "threshold_0_5_prediction_vectors_unchanged": threshold_predictions_unchanged,
                "per_sample_argmax_labels_unchanged": argmax_predictions_unchanged,
                "test_not_used_for_fit_or_selection": True,
                "model_weights_changed": False,
            },
            "outputs": {
                "parquet": table_path.name,
                "probability_archive": probability_path.name,
                "paired_reliability_diagram": reliability_path.name,
            },
        }
        _exclusive_json(output_dir / "metrics_and_manifest.json", payload)
        rows.append(
            {
                "task": "classification",
                "dataset": "treesatai",
                "model": member.model,
                "adaptation": member.adaptation,
                "method": "temperature_scaling_validation_fitted",
                "fit_split": "official_validation",
                "seed": member.seed,
                "temperature": temperature,
                "accuracy_before": before["accuracy"],
                "accuracy_after": after["accuracy"],
                "macro_f1_before": before["macro_f1"],
                "macro_f1_after": after["macro_f1"],
                "nll_before": before["nll"],
                "nll_after": after["nll"],
                "brier_before": before["brier"],
                "brier_after": after["brier"],
                "ece_15_before": before["ece_15"],
                "ece_15_after": after["ece_15"],
                "threshold_predictions_unchanged": threshold_predictions_unchanged,
                "argmax_predictions_unchanged": argmax_predictions_unchanged,
                "reliability_diagram": str(reliability_path.resolve()),
                "output_path": str(output_dir.resolve()),
            }
        )
    _exclusive_json(
        OUTPUT_ROOT / "classification/treesatai_temperature_scaling_summary.json", rows
    )
    return rows


def _write_replacement(path: Path, content: str) -> None:
    """Atomically replace a report while refusing to reuse a stale temporary file."""

    temporary = path.with_name(f".{path.name}.treesatai-ts.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale report-update temporary file: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
    temporary.replace(path)


def _replace_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_name(f".{path.name}.treesatai-ts.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale CSV-update temporary file: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _treesatai_temperature_table(rows: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        "| Model | Adaptation | Seed | T | Accuracy before/after | Macro-F1 before/after | NLL before/after | Brier before/after | ECE-15 before/after | Argmax unchanged |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['adaptation']} | {row['seed']} | {row['temperature']:.6f} | "
            f"{row['accuracy_before']:.6f} / {row['accuracy_after']:.6f} | "
            f"{row['macro_f1_before']:.6f} / {row['macro_f1_after']:.6f} | "
            f"{row['nll_before']:.6f} / {row['nll_after']:.6f} | "
            f"{row['brier_before']:.6f} / {row['brier_after']:.6f} | "
            f"{row['ece_15_before']:.6f} / {row['ece_15_after']:.6f} | "
            f"{str(row['argmax_predictions_unchanged']).lower()} |"
        )
    return lines


def finalize_treesatai_temperature_scaling() -> dict[str, Any]:
    """Publish TreeSatAI TS results and amend the completed C3 report in place."""

    summary_path = OUTPUT_ROOT / "classification/treesatai_temperature_scaling_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        (model, adaptation, seed)
        for model in ("dofa", "panopticon")
        for adaptation in ("frozen", "full_finetune")
        for seed in SEEDS
    }
    observed = {(row["model"], row["adaptation"], int(row["seed"])) for row in rows}
    if len(rows) != 12 or observed != expected:
        raise ValueError(f"TreeSatAI TS matrix is incomplete: {sorted(expected - observed)}")
    for row in rows:
        if not row["threshold_predictions_unchanged"] or not row["argmax_predictions_unchanged"]:
            raise ValueError("TreeSatAI TS result violates the prediction-invariance requirement")
        if row["fit_split"] != "official_validation" or float(row["temperature"]) <= 0.0:
            raise ValueError("TreeSatAI TS result has invalid fit provenance or temperature")

    _exclusive_csv(TREESATAI_TEMPERATURE_CSV, rows)
    tree_root = OUTPUT_ROOT / "classification/temperature_scaling/treesatai"
    code_snapshot = write_code_snapshot_manifest(tree_root, project_root=PROJECT_ROOT)
    report_lines = [
        "# C3 — TreeSatAI validation-fitted Temperature Scaling",
        "",
        f"Created: {utc_now()}",
        "",
        "TreeSatAI uses the downloaded official GEO-Bench-2 train/validation/test artifact. It has no independent calibration split. Under the final thesis protocol, one positive scalar temperature is therefore fitted to each checkpoint's official validation logits. This official validation split was also used for checkpoint selection; that dual use is an explicit protocol difference from EuroSAT, which uses a dedicated calibration split.",
        "",
        "The official test split remained untouched during fitting. No temperature was selected, rejected, or adjusted from test performance, and no neural-network weight was changed. All reported test results apply the validation-fitted temperature prospectively to the existing saved test logits.",
        "",
        "TreeSatAI is multilabel: Accuracy is strict exact match, Macro-F1 is label-macro positive-class F1, NLL and Brier average all sample-label decisions, and ECE-15 uses flattened binary-decision confidence. A positive scalar leaves both each 0.5-threshold prediction vector and the per-sample argmax label unchanged; both invariants were verified for every checkpoint.",
        "",
        "## Results",
        "",
        *_treesatai_temperature_table(rows),
        "",
        "## Saved artifacts",
        "",
        "Each checkpoint directory contains raw and calibrated test probabilities in Parquet and NPZ formats, a paired raw/calibrated reliability diagram, and a metrics/provenance manifest. The validation-logit exports and their checkpoint hashes are recorded in those manifests.",
        "",
        f"TreeSatAI Temperature Scaling code snapshot: `sha256:{code_snapshot['code_sha256']}` ({code_snapshot['file_count']} files).",
        "",
        "The optional EuroSAT validation-fit sensitivity analysis was not run; it is not needed to resolve the required TreeSatAI result and no protocol was chosen using test performance.",
        "",
    ]
    TREESATAI_TEMPERATURE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with TREESATAI_TEMPERATURE_REPORT.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines))

    with CLASSIFICATION_CSV.open(encoding="utf-8", newline="") as handle:
        existing_rows = list(csv.DictReader(handle))
    if any(
        row.get("dataset") == "treesatai" and row.get("method", "").startswith("temperature_scaling")
        for row in existing_rows
    ):
        raise ValueError("Consolidated classification CSV already contains TreeSatAI TS rows")
    _replace_csv(CLASSIFICATION_CSV, [*existing_rows, *rows])

    existing_report = REPORT_PATH.read_text(encoding="utf-8")
    if "## Temperature Scaling — TreeSatAI" in existing_report:
        raise ValueError("Consolidated C3 report already contains TreeSatAI TS results")
    old_intro = (
        "No model training was performed. Temperature Scaling was fitted only on EuroSAT's dedicated "
        "2,713-sample calibration split and evaluated on the untouched 2,714-sample test split. Deep "
        "Ensembles use seeds 42/43/44 and arithmetic means of probabilities, never logits."
    )
    new_intro = (
        "No model training was performed. EuroSAT Temperature Scaling uses its dedicated 2,713-sample "
        "calibration split; TreeSatAI uses its official validation split because the benchmark has no "
        "independent calibration split. Both are evaluated on untouched official test logits. Deep "
        "Ensembles use seeds 42/43/44 and arithmetic means of probabilities, never logits."
    )
    if old_intro not in existing_report:
        raise ValueError("Could not find the expected C3 report introduction")
    existing_report = existing_report.replace(old_intro, new_intro, 1)
    insertion = "\n".join(
        [
            "## Temperature Scaling — TreeSatAI",
            "",
            "One positive scalar temperature was fitted per checkpoint on official validation logits only. Validation was also used for checkpoint selection; this declared dual use is required because the official benchmark has no separate calibration split. Test logits were untouched during fitting and no test result informed temperature acceptance or selection.",
            "",
            *_treesatai_temperature_table(rows),
            "",
        ]
    )
    marker = "## Classification Deep Ensembles"
    if marker not in existing_report:
        raise ValueError("Could not find classification-ensemble insertion point")
    existing_report = existing_report.replace(marker, insertion + marker, 1)
    old_limitation = (
        "## Protocol limitation\n\n"
        "TreeSatAI Temperature Scaling is intentionally not reported. The pinned official GEO-Bench-2 "
        "artifact provides train/validation/test only; validation was already used for early stopping and "
        "checkpoint selection. It is therefore not a dedicated calibration split, and test labels were not "
        "used for fitting. Producing a defensible TreeSatAI Temperature Scaling result would require a "
        "prospectively reserved calibration split and consequently new training under that revised split protocol."
    )
    new_protocol = (
        "## Protocol differences\n\n"
        "- EuroSAT fits Temperature Scaling on its independent dedicated calibration split.\n"
        "- TreeSatAI has no independent official calibration split, so its temperature is fitted on official "
        "validation logits. That split was also used for checkpoint selection; this dual use is documented "
        "rather than concealed.\n"
        "- Every official test set remained untouched during fitting, and no temperature was selected or "
        "rejected using test performance."
    )
    if old_limitation not in existing_report:
        raise ValueError("Could not find the obsolete TreeSatAI protocol-limitation text")
    existing_report = existing_report.replace(old_limitation, new_protocol, 1)
    snapshot_marker = "## Artifact rules"
    existing_report = existing_report.replace(
        snapshot_marker,
        snapshot_marker
        + "\n\n- TreeSatAI Temperature Scaling artifacts have their own extension code snapshot: "
        + f"`sha256:{code_snapshot['code_sha256']}` ({code_snapshot['file_count']} files).",
        1,
    )
    _write_replacement(REPORT_PATH, existing_report)
    return {
        "report": str(TREESATAI_TEMPERATURE_REPORT),
        "csv": str(TREESATAI_TEMPERATURE_CSV),
        "consolidated_report": str(REPORT_PATH),
        "consolidated_classification_csv": str(CLASSIFICATION_CSV),
        "rows": len(rows),
        "code_snapshot": code_snapshot,
    }


def _segmentation_cell_members(dataset: str, model: str, adaptation: str) -> list[Member]:
    members = sorted(
        (
            member
            for member in segmentation_registry()
            if member.cell_key == (dataset, model, adaptation)
        ),
        key=lambda value: value.seed,
    )
    if [member.seed for member in members] != list(SEEDS):
        raise ValueError(f"Missing segmentation ensemble members: {dataset}/{model}/{adaptation}")
    return members


def _metrics_from_segmentation_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    ignore_index: int | None,
) -> dict[str, Any]:
    foreground = 1 if list(class_names) == ["background", "building"] else None
    accumulator = SegmentationMetricAccumulator(
        num_classes=probabilities.shape[1],
        class_names=class_names,
        ignore_index=ignore_index,
        n_bins=N_BINS,
        foreground_class_index=foreground,
        boundary_radius=1,
    )
    for start in range(0, len(labels), 8):
        chunk = torch.from_numpy(probabilities[start : start + 8]).float()
        pseudo_logits = chunk.clamp_min(1.0e-30).log()
        accumulator.update(pseudo_logits, torch.from_numpy(labels[start : start + 8]).long())
    return accumulator.compute()


def build_segmentation_cell(dataset: str, model: str, adaptation: str) -> dict[str, Any]:
    members = _segmentation_cell_members(dataset, model, adaptation)
    output_dir = OUTPUT_ROOT / "segmentation/deep_ensembles" / dataset / model / adaptation
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite segmentation ensemble: {output_dir}")
    reference_ids = reference_labels = reference_valid = None
    probability_sum: np.ndarray | None = None
    class_names: list[str] | None = None
    ignore_index: int | None = None
    member_rows = []
    for member in members:
        manifest_path = member.prediction_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest["source"]
        for key, expected in (
            ("run_id", member.run_id),
            ("seed", member.seed),
            ("dataset", dataset),
            ("adaptation_mode", adaptation),
        ):
            if source.get(key) != expected:
                raise ValueError(f"Segmentation member manifest mismatch: {member.run_id}/{key}")
        if model not in str(source.get("model_name", "")):
            raise ValueError(f"Segmentation member model mismatch: {member.run_id}")
        checkpoint_hash = sha256_file(member.checkpoint_path)
        if checkpoint_hash != member.checkpoint_sha256:
            raise ValueError(f"Segmentation checkpoint hash mismatch: {member.run_id}")
        with np.load(member.prediction_path, allow_pickle=False) as archive:
            ids = archive["sample_id"].astype(str)
            labels = archive["label"]
            valid = archive["valid_mask"].astype(bool)
            probabilities = archive["probabilities"].astype(np.float64)
        if not np.isfinite(probabilities).all() or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=2.0e-6, rtol=0
        ):
            raise ValueError(f"Invalid segmentation probabilities: {member.run_id}")
        if reference_ids is None:
            reference_ids, reference_labels, reference_valid = ids, labels, valid
            probability_sum = np.zeros_like(probabilities, dtype=np.float64)
            class_names = list(manifest["class_names"])
            ignore_index = manifest["ignore_index"]
        else:
            if not np.array_equal(ids, reference_ids):
                raise ValueError("Segmentation ensemble sample IDs/order differ")
            if not np.array_equal(labels, reference_labels) or not np.array_equal(valid, reference_valid):
                raise ValueError("Segmentation ensemble labels/valid masks differ")
            if list(manifest["class_names"]) != class_names or manifest["ignore_index"] != ignore_index:
                raise ValueError("Segmentation ensemble class/ignore metadata differ")
        assert probability_sum is not None
        probability_sum += probabilities
        member_rows.append(
            {
                "seed": member.seed,
                "run_id": member.run_id,
                "checkpoint_path": str(member.checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "prediction_path": str(member.prediction_path.resolve()),
                "prediction_sha256": sha256_file(member.prediction_path),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
        del probabilities
    assert probability_sum is not None and reference_labels is not None and reference_valid is not None
    assert reference_ids is not None and class_names is not None
    ensemble_probabilities = probability_sum / len(members)
    if not np.allclose(ensemble_probabilities.sum(axis=1), 1.0, atol=2.0e-6, rtol=0):
        raise ValueError("Averaged segmentation probabilities are not normalized")
    metrics = _metrics_from_segmentation_probabilities(
        ensemble_probabilities, reference_labels, class_names, ignore_index
    )
    prediction = ensemble_probabilities.argmax(axis=1).astype(np.int64)
    confidence = ensemble_probabilities.max(axis=1).astype(np.float32)
    entropy = -(
        ensemble_probabilities
        * np.log(np.clip(ensemble_probabilities, np.finfo(np.float64).tiny, 1.0))
    ).sum(axis=1).astype(np.float32)
    correctness = prediction == reference_labels
    correctness &= reference_valid
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "ensemble_predictions.npz"
    np.savez_compressed(
        output_path,
        sample_id=reference_ids,
        label=reference_labels,
        probabilities=ensemble_probabilities.astype(np.float32),
        prediction=prediction,
        correctness=correctness,
        confidence=confidence,
        predictive_entropy=entropy,
        valid_mask=reference_valid,
    )
    _exclusive_json(output_dir / "metrics.json", metrics)
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "task": "semantic_segmentation",
        "dataset": dataset,
        "model": model,
        "adaptation": adaptation,
        "split": "test",
        "sample_count": int(len(reference_ids)),
        "class_names": class_names,
        "ignore_index": ignore_index,
        "member_seeds": list(SEEDS),
        "aggregation": "arithmetic mean of member probabilities; member logits were never averaged",
        "member_predictions": member_rows,
        "member_storage": (
            "Existing complete deterministic member bundles are retained in place and SHA256-pinned here; "
            "they are not duplicated because each source bundle is approximately 1-1.7 GB."
        ),
        "ensemble_output": output_path.name,
        "arrays": {
            "label": list(reference_labels.shape),
            "probabilities": list(ensemble_probabilities.shape),
            "prediction": list(prediction.shape),
            "confidence": list(confidence.shape),
            "predictive_entropy": list(entropy.shape),
            "valid_mask": list(reference_valid.shape),
        },
        "metrics": metrics,
    }
    _exclusive_json(output_dir / "manifest.json", manifest)
    return manifest


def _flatten_segmentation_manifest(manifest: dict[str, Any], path: Path) -> dict[str, Any]:
    metrics = manifest["metrics"]
    row: dict[str, Any] = {
        "task": "segmentation",
        "dataset": manifest["dataset"],
        "model": manifest["model"],
        "adaptation": manifest["adaptation"],
        "method": "deep_ensemble_probability_mean",
        "seeds": "42;43;44",
        "miou": metrics["miou"],
        "pixel_accuracy": metrics["pixel_accuracy"],
        "nll": metrics["nll"],
        "brier": metrics["brier"],
        "ece_15": metrics["ece_15"],
        "output_path": str(path.parent.resolve()),
    }
    for name, value in metrics["per_class_iou"].items():
        row[f"iou_{name}"] = value
    for name in (
        "foreground_ece_15",
        "foreground_nll",
        "foreground_brier",
    ):
        if name in metrics:
            row[name] = metrics[name]
    if "boundary_calibration" in metrics:
        row["boundary_ece_15"] = metrics["boundary_calibration"]["ece_15"]
        row["boundary_foreground_ece_15"] = metrics["boundary_calibration"]["foreground_ece_15"]
    return row


def finalize() -> dict[str, Any]:
    classification_ensemble_path = OUTPUT_ROOT / "classification/deep_ensemble_summary.json"
    temperature_path = OUTPUT_ROOT / "classification/temperature_scaling_summary.json"
    if not classification_ensemble_path.is_file() or not temperature_path.is_file():
        raise FileNotFoundError("Classification C3 summaries are incomplete")
    ensemble_rows = json.loads(classification_ensemble_path.read_text(encoding="utf-8"))
    temperature_rows = json.loads(temperature_path.read_text(encoding="utf-8"))
    classification_rows = [*ensemble_rows, *temperature_rows]

    segmentation_rows = []
    for dataset in ("cloudsen12", "spacenet7"):
        for model in ("dofa", "panopticon"):
            for adaptation in ("frozen", "full_finetune"):
                path = OUTPUT_ROOT / "segmentation/deep_ensembles" / dataset / model / adaptation / "manifest.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                segmentation_rows.append(
                    _flatten_segmentation_manifest(json.loads(path.read_text(encoding="utf-8")), path)
                )
    _exclusive_csv(CLASSIFICATION_CSV, classification_rows)
    _exclusive_csv(SEGMENTATION_CSV, segmentation_rows)
    code_snapshot = write_code_snapshot_manifest(OUTPUT_ROOT, project_root=PROJECT_ROOT)

    temp_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in temperature_rows:
        temp_by_cell.setdefault((row["model"], row["adaptation"]), []).append(row)
    lines = [
        "# C3 — Temperature Scaling and Deep Ensembles",
        "",
        f"Created: {utc_now()}",
        "",
        "No model training was performed. Temperature Scaling was fitted only on EuroSAT's dedicated 2,713-sample calibration split and evaluated on the untouched 2,714-sample test split. Deep Ensembles use seeds 42/43/44 and arithmetic means of probabilities, never logits.",
        "",
        "## Temperature Scaling — EuroSAT",
        "",
        "| Model | Adaptation | Seed | T | Accuracy before/after | NLL before/after | Brier before/after | ECE-15 before/after |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in temperature_rows:
        lines.append(
            f"| {row['model']} | {row['adaptation']} | {row['seed']} | {row['temperature']:.6f} | "
            f"{row['accuracy_before']:.6f} / {row['accuracy_after']:.6f} | "
            f"{row['nll_before']:.6f} / {row['nll_after']:.6f} | "
            f"{row['brier_before']:.6f} / {row['brier_after']:.6f} | "
            f"{row['ece_15_before']:.6f} / {row['ece_15_after']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Classification Deep Ensembles",
            "",
            "| Dataset | Model | Adaptation | Accuracy | Macro-F1 | NLL | Brier | ECE-15 |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ensemble_rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['adaptation']} | {row['accuracy']:.6f} | "
            f"{row['macro_f1']:.6f} | {row['nll']:.6f} | {row['brier']:.6f} | {row['ece_15']:.6f} |"
        )
    lines.extend(
        [
            "",
            "TreeSatAI metrics follow its multilabel protocol: accuracy is strict exact match, Macro-F1 is label-macro positive-class F1, NLL/Brier average sample-label decisions, and ECE-15 flattens binary decisions. They are not directly comparable to EuroSAT multiclass metric definitions.",
            "",
            "## Segmentation Deep Ensembles",
            "",
            "| Dataset | Model | Adaptation | mIoU | Pixel accuracy | NLL | Brier | ECE-15 |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in segmentation_rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['adaptation']} | {row['miou']:.6f} | "
            f"{row['pixel_accuracy']:.6f} | {row['nll']:.6f} | {row['brier']:.6f} | {row['ece_15']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Protocol limitation",
            "",
            "TreeSatAI Temperature Scaling is intentionally not reported. The pinned official GEO-Bench-2 artifact provides train/validation/test only; validation was already used for early stopping and checkpoint selection. It is therefore not a dedicated calibration split, and test labels were not used for fitting. Producing a defensible TreeSatAI Temperature Scaling result would require a prospectively reserved calibration split and consequently new training under that revised split protocol.",
            "",
            "## Artifact rules",
            "",
            "- DOFA–EuroSAT ensembles are the previously frozen direct-Parquet artifacts and were revalidated against the immutable member Parquets without inference.",
            "- New classification ensemble bundles contain aligned member logits/probabilities and ensemble probabilities.",
            "- Segmentation manifests SHA256-pin the existing complete member prediction bundles; these 1–1.7 GB files are retained in place rather than duplicated. Ensemble probability/confidence/entropy maps are saved separately.",
            f"- C3 analysis code snapshot: `sha256:{code_snapshot['code_sha256']}` ({code_snapshot['file_count']} files).",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return {
        "report": str(REPORT_PATH),
        "classification_csv": str(CLASSIFICATION_CSV),
        "segmentation_csv": str(SEGMENTATION_CSV),
        "classification_rows": len(classification_rows),
        "segmentation_rows": len(segmentation_rows),
        "code_snapshot": code_snapshot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = subparsers.add_parser("calibration-commands")
    commands.add_argument("--shard", type=int, required=True)
    commands.add_argument("--shards", type=int, default=2)
    commands.add_argument("--device", required=True)
    subparsers.add_parser("prepare-calibration-configs")
    subparsers.add_parser("classification-ensembles")
    subparsers.add_parser("temperature-scaling")
    treesatai_commands = subparsers.add_parser("treesatai-validation-commands")
    treesatai_commands.add_argument("--shard", type=int, required=True)
    treesatai_commands.add_argument("--shards", type=int, default=2)
    treesatai_commands.add_argument("--device", required=True)
    subparsers.add_parser("treesatai-temperature-scaling")
    subparsers.add_parser("finalize-treesatai-temperature-scaling")
    segmentation = subparsers.add_parser("segmentation-cell")
    segmentation.add_argument("--dataset", choices=("cloudsen12", "spacenet7"), required=True)
    segmentation.add_argument("--model", choices=("dofa", "panopticon"), required=True)
    segmentation.add_argument("--adaptation", choices=("frozen", "full_finetune"), required=True)
    subparsers.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "prepare-calibration-configs":
        print(json.dumps(prepare_calibration_configs(), indent=2))
    elif args.command == "calibration-commands":
        print("\n".join(calibration_export_commands(args.shard, args.shards, args.device)))
    elif args.command == "classification-ensembles":
        print(json.dumps(build_classification_ensembles(), indent=2))
    elif args.command == "temperature-scaling":
        print(json.dumps(build_temperature_scaling(), indent=2))
    elif args.command == "treesatai-validation-commands":
        print("\n".join(treesatai_validation_export_commands(args.shard, args.shards, args.device)))
    elif args.command == "treesatai-temperature-scaling":
        print(json.dumps(build_treesatai_temperature_scaling(), indent=2))
    elif args.command == "finalize-treesatai-temperature-scaling":
        print(json.dumps(finalize_treesatai_temperature_scaling(), indent=2))
    elif args.command == "segmentation-cell":
        print(json.dumps(build_segmentation_cell(args.dataset, args.model, args.adaptation), indent=2))
    elif args.command == "finalize":
        print(json.dumps(finalize(), indent=2))


if __name__ == "__main__":
    main()
