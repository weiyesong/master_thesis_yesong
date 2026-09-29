#!/usr/bin/env python3
"""Build the final thesis result package from already validated artifacts only.

This script performs aggregation and visualization.  It never loads a model,
trains a model, or runs model inference.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RESULTS = ROOT / "results" / "final_thesis"
MASTER = REPORTS / "thesis_master_results.csv"
METHOD_KEYS = {"Deterministic": "deterministic", "Temperature Scaling": "temperature_scaling", "MC Dropout": "mc_dropout", "Deep Ensemble": "deep_ensemble"}

CLASS_METRICS = ("accuracy", "macro_f1", "nll", "brier", "ece_15")
SEG_METRICS = (
    "miou",
    "pixel_accuracy",
    "nll",
    "brier",
    "ece_15",
    "iou_clear",
    "iou_thick_cloud",
    "iou_thin_cloud",
    "iou_cloud_shadow",
    "iou_background",
    "iou_building",
    "foreground_ece_15",
    "foreground_nll",
    "foreground_brier",
    "classwise_ece_background",
    "classwise_ece_building",
    "boundary_ece_15",
    "boundary_foreground_ece_15",
)
DATASETS_CLASS = ("eurosat", "treesatai")
DATASETS_SEG = ("cloudsen12", "spacenet7")
MODELS = ("dofa", "panopticon")
ADAPTATIONS = ("frozen", "full_finetune")
DATASET_DISPLAY = {
    "eurosat": "EuroSAT",
    "treesatai": "TreeSatAI",
    "cloudsen12": "CloudSEN12",
    "spacenet7": "SpaceNet7",
}
METHOD_ORDER = ("Deterministic", "Temperature Scaling", "MC Dropout", "Deep Ensemble")
METHOD_COLORS = {
    "Deterministic": "#4c78a8",
    "Temperature Scaling": "#f58518",
    "MC Dropout": "#54a24b",
    "Deep Ensemble": "#e45756",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def slug(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def aggregate(records: list[dict[str, Any]], metrics: Iterable[str]) -> dict[str, Any]:
    require(records, "Cannot aggregate an empty record set")
    result: dict[str, Any] = {}
    for metric in metrics:
        values = [number(record.get(metric)) for record in records]
        finite = [value for value in values if value is not None]
        if not finite:
            result[metric] = None
            result[f"{metric}_std"] = None
            continue
        require(len(finite) == len(records), f"Partial metric {metric} in aggregate")
        result[metric] = float(np.mean(finite))
        result[f"{metric}_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else None
    return result


def flatten_seg_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        key: number(metrics.get(key)) for key in ("miou", "pixel_accuracy", "nll", "brier", "ece_15")
    }
    for name, value in metrics.get("per_class_iou", {}).items():
        result[f"iou_{slug(name)}"] = number(value)
    result["foreground_ece_15"] = number(metrics.get("foreground_ece_15"))
    result["foreground_nll"] = number(metrics.get("foreground_nll"))
    result["foreground_brier"] = number(metrics.get("foreground_brier"))
    classwise = metrics.get("classwise_calibration", {})
    for name in ("background", "building"):
        result[f"classwise_ece_{name}"] = number(classwise.get(name, {}).get("ece_15"))
    boundary = metrics.get("boundary_calibration", {})
    result["boundary_ece_15"] = number(boundary.get("ece_15"))
    result["boundary_foreground_ece_15"] = number(boundary.get("foreground_ece_15"))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def fmt(row: dict[str, Any], metric: str, digits: int = 4) -> str:
    value = number(row.get(metric))
    if value is None:
        return "N/A"
    std = number(row.get(f"{metric}_std"))
    if std is not None:
        return f"{value:.{digits}f} ± {std:.{digits}f}"
    return f"{value:.{digits}f}"


def temperature_scope(master_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    """Use the authoritative matrix, never the existence of historical TS runs."""
    scope = {}
    for row in master_rows:
        if row["uq_method"] != "temperature_scaling" or row["record_type"] not in {"MEAN_STD", "NOT_APPLICABLE"}:
            continue
        cell = (row["dataset"], row["model"], row["adaptation"])
        require(cell not in scope, f"Duplicate TS applicability row: {cell}")
        require(row["status"] in {"COMPLETE", "N/A"}, f"Unresolved TS scope: {cell}")
        scope[cell] = row
    expected = {(d, m, a) for d in (*DATASETS_CLASS, *DATASETS_SEG) for m in MODELS for a in ADAPTATIONS}
    require(set(scope) == expected, "Master TS applicability matrix is incomplete")
    return scope


def na_temperature_row(cell: tuple[str, str, str], scope: dict) -> dict[str, Any]:
    dataset, model, adaptation = cell
    entry = scope[cell]
    require(entry["status"] == "N/A", f"TS is applicable in master: {cell}")
    return {
        "task": entry["task"], "dataset": dataset, "model": model, "adaptation": adaptation,
        "uq_method": "Temperature Scaling", "status": "N/A", "seeds": "N/A", "n_runs": 0,
        "reporting_basis": entry["applicability_reason"], "source": str(MASTER),
    }


def build_classification_sources() -> tuple[
    dict[tuple[str, str, str, int], dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    deterministic: dict[tuple[str, str, str, int], dict[str, Any]] = {}

    dofa_manifest_path = REPORTS / "dofa_eurosat_final_manifest.json"
    dofa_manifest = read_json(dofa_manifest_path)
    require(
        dofa_manifest.get("status") == "FINAL_IMMUTABLE"
        and dofa_manifest.get("immutability", {}).get("verified") is True,
        "DOFA-EuroSAT manifest is not verified immutable final",
    )
    require(len(dofa_manifest.get("runs", [])) == 6, "Expected six frozen DOFA-EuroSAT runs")
    for run in dofa_manifest["runs"]:
        prediction = Path(run["test_prediction_path"])
        require(prediction.exists(), f"Missing frozen prediction {prediction}")
        require(sha256(prediction) == run["test_prediction_sha256"], f"Prediction hash mismatch: {prediction}")
        metrics = {metric: number(run["metrics"].get(metric)) for metric in CLASS_METRICS}
        key = ("eurosat", "dofa", run["adaptation"], int(run["seed"]))
        deterministic[key] = {**metrics, "prediction_path": str(prediction), "source": str(dofa_manifest_path)}

    audit_paths = sorted((REPORTS / "c1_classification_run_audits").glob("*/c1_run_audit.json"))
    require(audit_paths, "No C1 classification audit found")
    c1_path = audit_paths[-1]
    c1 = read_json(c1_path)
    require(c1["summary"] == {"promotable": 18, "blocked": 0, "missing": 0, "ambiguous": 0, "total": 18}, "C1 audit is not fully promotable")
    for run in c1["runs"]:
        require(run.get("status") == "PROMOTABLE" and run.get("promotion_eligible") is True, "Non-promotable C1 run")
        require(run.get("test_metrics_used_for_promotion") is False, "C1 test metric selection leakage")
        require(run["prediction_export"]["validation"].get("valid") is True, "Invalid C1 prediction export")
        target = run["target"]
        raw = run["test_metrics_report_only"]
        metrics = {
            "accuracy": number(raw.get("accuracy")),
            "macro_f1": number(raw.get("macro_f1")),
            "nll": number(raw.get("nll")),
            "brier": number(raw.get("brier")),
            "ece_15": number(raw.get("ece_15", raw.get("ece"))),
        }
        prediction = Path(run["prediction_export"]["path"]) / "predictions.parquet"
        require(prediction.exists(), f"Missing C1 prediction {prediction}")
        key = (target["dataset"], target["model"], target["adaptation"], int(target["seed"]))
        deterministic[key] = {**metrics, "prediction_path": str(prediction), "source": str(c1_path)}

    expected = {
        (dataset, model, adaptation, seed)
        for dataset in DATASETS_CLASS
        for model in MODELS
        for adaptation in ADAPTATIONS
        for seed in (42, 43, 44)
    }
    require(set(deterministic) == expected, "Classification deterministic matrix is incomplete or duplicated")

    c3_path = REPORTS / "c3_classification_results.csv"
    c3_rows = read_csv(c3_path)
    ts_rows = [row for row in c3_rows if row["method"].startswith("temperature_scaling")]
    ensemble_rows = [row for row in c3_rows if row["method"] == "deep_ensemble_probability_mean"]
    require(len(ts_rows) == 24 and len(ensemble_rows) == 8, "C3 classification row counts are invalid")
    for row in c3_rows:
        require(Path(row["output_path"]).is_dir(), f"Missing C3 output {row['output_path']}")
    return deterministic, ts_rows, ensemble_rows


def build_classification_table(
    deterministic: dict[tuple[str, str, str, int], dict[str, Any]],
    ts_rows: list[dict[str, str]],
    ensemble_rows: list[dict[str, str]],
    mc_rows: list[dict[str, str]],
    scope: dict,
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for dataset in DATASETS_CLASS:
        for model in MODELS:
            for adaptation in ADAPTATIONS:
                cell = (dataset, model, adaptation)
                det_records = [deterministic[(*cell, seed)] for seed in (42, 43, 44)]
                table.append({
                    "task": "classification", "dataset": dataset, "model": model, "adaptation": adaptation,
                    "uq_method": "Deterministic", "status": "COMPLETE", "seeds": "42;43;44", "n_runs": 3,
                    "reporting_basis": "mean ± sample std across deterministic seeds",
                    **aggregate(det_records, CLASS_METRICS),
                    "source": det_records[0]["source"],
                })

                selected_ts = sorted(
                    (row for row in ts_rows if (row["dataset"], row["model"], row["adaptation"]) == cell),
                    key=lambda row: int(row["seed"]),
                )
                if scope[cell]["status"] == "N/A":
                    table.append(na_temperature_row(cell, scope))
                    selected_ts = []
                else:
                    require([int(row["seed"]) for row in selected_ts] == [42, 43, 44], f"Incomplete TS cell {cell}")
                ts_records = []
                for row in selected_ts:
                    seed = int(row["seed"])
                    macro = number(row.get("macro_f1_after"))
                    if macro is None:
                        macro = deterministic[(*cell, seed)]["macro_f1"]
                    ts_records.append({
                        "accuracy": number(row["accuracy_after"]), "macro_f1": macro,
                        "nll": number(row["nll_after"]), "brier": number(row["brier_after"]),
                        "ece_15": number(row["ece_15_after"]),
                    })
                    if dataset == "eurosat":
                        require(
                            number(row.get("temperature")) is not None
                            and number(row.get("temperature")) > 0
                            and abs(float(row["accuracy_before"]) - float(row["accuracy_after"])) < 1e-12,
                            f"EuroSAT scalar-temperature invariance failed for {cell} seed {seed}",
                        )
                    else:
                        require(row.get("threshold_predictions_unchanged") in ("True", "true", "1"), f"TreeSatAI TS prediction changed for {cell} seed {seed}")
                if ts_records:
                    table.append({
                        "task": "classification", "dataset": dataset, "model": model, "adaptation": adaptation,
                        "uq_method": "Temperature Scaling", "status": "COMPLETE", "seeds": "42;43;44", "n_runs": 3,
                        "reporting_basis": "mean ± sample std; dedicated calibration split" if dataset == "eurosat" else "mean ± sample std; official validation-fitted (no calibration split)",
                        **aggregate(ts_records, CLASS_METRICS),
                        "source": str(REPORTS / "c3_classification_results.csv"),
                    })

                mc = next(row for row in mc_rows if row["task"] == "classification" and int(row["seed"]) == 42 and (row["dataset"], row["model"], row["adaptation"]) == cell)
                table.append({
                    "task": "classification", "dataset": dataset, "model": model, "adaptation": adaptation,
                    "uq_method": "MC Dropout", "status": "COMPLETE", "seeds": "42", "n_runs": 1,
                    "reporting_basis": "primary paired seed42; 30 probability passes",
                    "accuracy": number(mc["mc_accuracy"]), "macro_f1": number(mc["mc_macro_f1"]),
                    "nll": number(mc["mc_nll"]), "brier": number(mc["mc_brier"]), "ece_15": number(mc["mc_ece_15"]),
                    "source": str(REPORTS / "mc_dropout_results.csv"),
                })

                ensemble = next(row for row in ensemble_rows if (row["dataset"], row["model"], row["adaptation"]) == cell)
                table.append({
                    "task": "classification", "dataset": dataset, "model": model, "adaptation": adaptation,
                    "uq_method": "Deep Ensemble", "status": "COMPLETE", "seeds": ensemble["seeds"], "n_runs": 1,
                    "reporting_basis": "single 3-member probability-mean ensemble; not pooled with members",
                    **{metric: number(ensemble[metric]) for metric in CLASS_METRICS},
                    "source": str(REPORTS / "c3_classification_results.csv"),
                })
    require(len(table) == 32, "Classification final table must contain 32 rows")
    return table


def build_segmentation_sources() -> tuple[
    dict[tuple[str, str, str, int], dict[str, Any]], list[dict[str, str]]
]:
    audit_path = REPORTS / "c2_segmentation_run_audits" / "20260824T162250Z" / "audit.csv"
    audit_rows = read_csv(audit_path)
    require(len(audit_rows) == 24 and all(row["status"] == "PROMOTABLE" for row in audit_rows), "C2 segmentation audit is not fully promotable")
    deterministic: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in audit_rows:
        run_dir = Path(row["run_dir"])
        validation = read_json(run_dir / "predictions" / "test" / "deterministic" / "validation_report.json")
        require(validation.get("valid") is True, f"Invalid segmentation prediction bundle: {run_dir}")
        metrics = flatten_seg_metrics(read_json(run_dir / "test_metrics.json"))
        key = (row["dataset"], row["model"], row["adaptation"], int(row["seed"]))
        deterministic[key] = {
            **metrics,
            "prediction_path": str(run_dir / "predictions" / "test" / "deterministic" / "predictions.npz"),
            "source": str(audit_path),
        }
    expected = {
        (dataset, model, adaptation, seed)
        for dataset in DATASETS_SEG for model in MODELS for adaptation in ADAPTATIONS for seed in (42, 43, 44)
    }
    require(set(deterministic) == expected, "Segmentation deterministic matrix is incomplete")

    ensemble_path = REPORTS / "c3_segmentation_ensemble_results.csv"
    ensembles = read_csv(ensemble_path)
    require(len(ensembles) == 8, "Expected eight segmentation ensemble rows")
    for row in ensembles:
        output = Path(row["output_path"])
        require((output / "manifest.json").exists() and (output / "ensemble_predictions.npz").exists(), f"Incomplete segmentation ensemble {output}")
    return deterministic, ensembles


def seg_from_prefixed(row: dict[str, str], prefix: str) -> dict[str, float | None]:
    normalized = {slug(key): value for key, value in row.items()}
    result: dict[str, float | None] = {}
    for metric in SEG_METRICS:
        key = slug(f"{prefix}{metric}")
        if key in normalized:
            result[metric] = number(normalized[key])
    return result


def build_segmentation_table(
    deterministic: dict[tuple[str, str, str, int], dict[str, Any]],
    ensemble_rows: list[dict[str, str]],
    mc_rows: list[dict[str, str]],
    scope: dict,
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for dataset in DATASETS_SEG:
        for model in MODELS:
            for adaptation in ADAPTATIONS:
                cell = (dataset, model, adaptation)
                det_records = [deterministic[(*cell, seed)] for seed in (42, 43, 44)]
                table.append({
                    "task": "segmentation", "dataset": dataset, "model": model, "adaptation": adaptation,
                    "uq_method": "Deterministic", "status": "COMPLETE", "seeds": "42;43;44", "n_runs": 3,
                    "reporting_basis": "mean ± sample std across deterministic seeds",
                    **aggregate(det_records, SEG_METRICS), "source": det_records[0]["source"],
                })
                table.append(na_temperature_row(cell, scope))
                mc = next(row for row in mc_rows if row["task"] == "segmentation" and int(row["seed"]) == 42 and (row["dataset"], row["model"], row["adaptation"]) == cell)
                table.append({
                    "task": "segmentation", "dataset": dataset, "model": model, "adaptation": adaptation,
                    "uq_method": "MC Dropout", "status": "COMPLETE", "seeds": "42", "n_runs": 1,
                    "reporting_basis": "primary paired seed42; 30 probability passes",
                    **seg_from_prefixed(mc, "mc_"), "source": str(REPORTS / "mc_dropout_results.csv"),
                })
                ensemble = next(row for row in ensemble_rows if (row["dataset"], row["model"], row["adaptation"]) == cell)
                normalized_ensemble = {slug(key): value for key, value in ensemble.items()}
                ensemble_metrics = {metric: number(normalized_ensemble.get(metric)) for metric in SEG_METRICS}
                ensemble_manifest_path = Path(ensemble["output_path"]) / "manifest.json"
                manifest_metrics = flatten_seg_metrics(read_json(ensemble_manifest_path)["metrics"])
                # The historical C3 summary CSV omitted SpaceNet7's two
                # classwise-ECE columns even though the validated ensemble
                # manifests contain those already-computed metrics.  Fill only
                # absent summary fields; cross-check every value present in both
                # sources so this remains aggregation, not metric invention.
                for metric in SEG_METRICS:
                    summary_value = ensemble_metrics.get(metric)
                    manifest_value = manifest_metrics.get(metric)
                    if summary_value is None:
                        ensemble_metrics[metric] = manifest_value
                    elif manifest_value is not None:
                        require(
                            math.isclose(summary_value, manifest_value, rel_tol=1.0e-9, abs_tol=1.0e-12),
                            f"C3 segmentation summary/manifest mismatch for {cell} {metric}",
                        )
                table.append({
                    "task": "segmentation", "dataset": dataset, "model": model, "adaptation": adaptation,
                    "uq_method": "Deep Ensemble", "status": "COMPLETE", "seeds": ensemble["seeds"], "n_runs": 1,
                    "reporting_basis": "single 3-member probability-mean ensemble; not pooled with members",
                    **ensemble_metrics, "source": str(ensemble_manifest_path),
                })
    require(len(table) == 32, "Segmentation final table must contain 32 rows")
    return table


def build_robustness_table(mc_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = [row for row in mc_rows if row["replication_role"] in {"primary_and_robustness", "robustness"}]
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        groups[(row["task"], row["dataset"], row["model"], row["adaptation"])].append(row)
    require(len(groups) == 4 and all(sorted(int(row["seed"]) for row in values) == [42, 43, 44] for values in groups.values()), "Invalid MC robustness subset")
    output = []
    for (task, dataset, model, adaptation), rows in sorted(groups.items()):
        metrics = CLASS_METRICS if task == "classification" else SEG_METRICS
        prefix = "mc_"
        records = [{metric: number(row.get(prefix + metric)) for metric in metrics} for row in rows]
        output.append({
            "task": task, "dataset": dataset, "model": model, "adaptation": adaptation,
            "uq_method": "MC Dropout", "status": "COMPLETE", "seeds": "42;43;44", "n_runs": 3,
            "reporting_basis": "predeclared robustness subset mean ± sample std",
            **aggregate(records, metrics), "source": str(REPORTS / "mc_dropout_results.csv"),
        })
    return output


def stack_list_column(table: Any, name: str) -> np.ndarray:
    return np.asarray(table[name].to_pylist())


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def class_probability_bundle(
    dataset: str,
    cell: tuple[str, str, str],
    method: str,
    ts_rows: list[dict[str, str]],
    ensemble_rows: list[dict[str, str]],
    mc_rows: list[dict[str, str]],
    deterministic: dict,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "Deterministic":
        table = pq.read_table(deterministic[(*cell, 42)]["prediction_path"])
        label_column = "label" if "label" in table.column_names else "true_label"
        labels = table[label_column].to_numpy() if dataset == "eurosat" else stack_list_column(table, label_column)
        return labels, stack_list_column(table, "probabilities")
    if method == "Temperature Scaling":
        row = next(row for row in ts_rows if int(row["seed"]) == 42 and (row["dataset"], row["model"], row["adaptation"]) == cell)
        output = Path(row["output_path"])
        if dataset == "eurosat":
            table = pq.read_table(output / "test_predictions.parquet", columns=["label", "raw_logits", "probabilities"])
            labels = table["label"].to_numpy()
            probabilities = stack_list_column(table, "probabilities")
        else:
            table = pq.read_table(output / "test_predictions_raw_and_calibrated.parquet", columns=["label", "raw_probabilities", "calibrated_probabilities"])
            labels = stack_list_column(table, "label")
            probabilities = stack_list_column(table, "calibrated_probabilities")
        return labels, probabilities
    if method == "Deep Ensemble":
        row = next(row for row in ensemble_rows if (row["dataset"], row["model"], row["adaptation"]) == cell)
        table = pq.read_table(Path(row["output_path"]) / "ensemble_predictions.parquet")
        if dataset == "eurosat":
            label_column = "true_label" if "true_label" in table.column_names else "label"
            labels = table[label_column].to_numpy()
        else:
            labels = stack_list_column(table, "label")
        return labels, stack_list_column(table, "probabilities")
    row = next(row for row in mc_rows if row["task"] == "classification" and int(row["seed"]) == 42 and (row["dataset"], row["model"], row["adaptation"]) == cell)
    table = pq.read_table(Path(row["output_path"]) / "predictions.parquet", columns=["label", "probabilities"])
    labels = table["label"].to_numpy() if dataset == "eurosat" else stack_list_column(table, "label")
    return labels, stack_list_column(table, "probabilities")


def calibration_bins(confidence: np.ndarray, outcome: np.ndarray, bins: int = 15) -> dict[str, np.ndarray]:
    """Frozen ECE convention: [0,1/B], (1/B,2/B], ..., ((B-1)/B,1]."""
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    outcome = np.asarray(outcome, dtype=np.float64).reshape(-1)
    require(confidence.shape == outcome.shape and confidence.size > 0, "Invalid calibration arrays")
    require(np.isfinite(confidence).all() and np.isfinite(outcome).all(), "Nonfinite calibration inputs")
    require(((confidence >= 0) & (confidence <= 1)).all(), "Probability outside [0,1]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.maximum(0, np.searchsorted(edges, confidence, side="left") - 1)
    count = np.bincount(index, minlength=bins)
    conf_sum = np.bincount(index, weights=confidence, minlength=bins)
    outcome_sum = np.bincount(index, weights=outcome, minlength=bins)
    conf_mean = np.full(bins, np.nan)
    out_mean = np.full(bins, np.nan)
    np.divide(conf_sum, count, out=conf_mean, where=count > 0)
    np.divide(outcome_sum, count, out=out_mean, where=count > 0)
    return {"count": count, "mean_confidence": conf_mean, "observed_frequency": out_mean,
            "lower": edges[:-1], "upper": edges[1:]}


def classification_bins(labels: np.ndarray, probabilities: np.ndarray, multilabel: bool,
                        bins: int = 15, positive_label: bool = False) -> dict[str, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if multilabel:
        if positive_label:
            confidence, outcome = probabilities, labels
        else:
            confidence = np.maximum(probabilities, 1.0 - probabilities)
            outcome = ((probabilities >= 0.5) == labels).astype(float)
    else:
        confidence = probabilities.max(axis=1)
        outcome = (probabilities.argmax(axis=1) == labels.astype(int)).astype(float)
    return calibration_bins(confidence, outcome, bins)


def reliability_points(labels: np.ndarray, probabilities: np.ndarray, multilabel: bool, bins: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = classification_bins(labels, probabilities, multilabel, bins)
    valid = data["count"] > 0
    return data["mean_confidence"][valid], data["observed_frequency"][valid], data["count"][valid]


def bin_ece(data: dict[str, np.ndarray]) -> float:
    valid = data["count"] > 0
    return float(np.sum(data["count"][valid] * np.abs(data["mean_confidence"][valid] - data["observed_frequency"][valid])) / data["count"].sum())


def reliability_axes(cells: list[tuple[str, str, str]], title: str):
    columns = 4 if len(cells) == 8 else 2
    fig = plt.figure(figsize=(18 if columns == 4 else 12, 12))
    outer = fig.add_gridspec(2, columns, left=.07, right=.99, bottom=.08, top=.89, hspace=.45, wspace=.25)
    pairs = []
    for index, cell in enumerate(cells):
        inner = outer[index // columns, index % columns].subgridspec(2, 1, height_ratios=[3, 1], hspace=.08)
        ax = fig.add_subplot(inner[0])
        hist = fig.add_subplot(inner[1], sharex=ax)
        ax.plot([0, 1], [0, 1], "--", color="0.4", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), ylabel="Observed frequency (fraction)")
        ax.tick_params(labelbottom=False)
        dataset, model, adaptation = cell
        ax.set_title(f"{DATASET_DISPLAY[dataset]} · {model.upper()}\n{adaptation.replace('_', ' ')}", fontsize=10)
        hist.set(xlabel="Mean confidence / bin (fraction)", ylabel="Count")
        hist.set_yscale("symlog", linthresh=1)
        ax.grid(alpha=.2)
        hist.grid(alpha=.2)
        pairs.append((ax, hist))
    fig.suptitle(title, fontsize=13, y=.975)
    return fig, pairs


def add_curve(ax, hist, method: str, data: dict[str, np.ndarray]) -> None:
    valid = data["count"] > 0
    ax.plot(data["mean_confidence"][valid], data["observed_frequency"][valid], marker="o", markersize=4,
            linewidth=1.3, label=f"{method} ({bin_ece(data):.4f})", color=METHOD_COLORS[method])
    hist.stairs(data["count"], np.r_[data["lower"], data["upper"][-1]], linewidth=1.2, color=METHOD_COLORS[method])
    ax.legend(loc="upper left", fontsize=7, title="ECE-15", title_fontsize=8, framealpha=.7)


def bin_records(cell: tuple[str, str, str], method: str, data: dict[str, np.ndarray], kind: str) -> list[dict]:
    return [{"dataset": cell[0], "model": cell[1], "adaptation": cell[2], "uq_method": METHOD_KEYS[method],
             "seed": "42;43;44" if method == "Deep Ensemble" else "42", "kind": kind, "bin": index,
             "n_bins": len(data["count"]), "interval": "[lower,upper]" if index == 0 else "(lower,upper]",
             **{key: (int(values[index]) if key == "count" else number(values[index])) for key, values in data.items()}}
            for index in range(len(data["count"]))]


def finish_reliability(path: Path, fig, records: list[dict]) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    write_csv(path.with_suffix(".csv"), records, list(records[0]))


def plot_classification_reliability(path: Path, ts_rows: list[dict], ensemble_rows: list[dict],
                                    mc_rows: list[dict], deterministic: dict, scope: dict,
                                    positive_label: bool = False) -> None:
    datasets = ("treesatai",) if positive_label else DATASETS_CLASS
    cells = [(d, m, a) for d in datasets for m in MODELS for a in ADAPTATIONS]
    meaning = "TreeSatAI pooled positive-label diagnostic: p(label=1) versus binary label" if positive_label else "Classification decision reliability: EuroSAT top label; TreeSatAI flattened binary decisions"
    fig, pairs = reliability_axes(cells, meaning + "\n15 right-closed bins; seed 42 (ensemble members 42/43/44); counts include empty bins")
    records = []
    for (ax, hist), cell in zip(pairs, cells):
        for method in METHOD_ORDER:
            if method == "Temperature Scaling" and scope[cell]["status"] == "N/A":
                continue
            labels, probabilities = class_probability_bundle(cell[0], cell, method, ts_rows, ensemble_rows, mc_rows, deterministic)
            data = classification_bins(labels, probabilities, cell[0] == "treesatai", positive_label=positive_label)
            add_curve(ax, hist, method, data)
            records.extend(bin_records(cell, method, data, "pooled_positive_label" if positive_label else "decision"))
        if positive_label:
            hist.set_xlabel("Label probability / bin (fraction)")
    finish_reliability(path, fig, records)


def seg_reliability_from_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        confidence = archive["confidence"]
        label = archive["label"]
        prediction = archive["prediction"]
        valid = archive["valid_mask"].astype(bool)
        return calibration_bins(confidence[valid], (label[valid] == prediction[valid]).astype(float))


def plot_segmentation_reliability(path: Path, deterministic: dict, ensemble_rows: list[dict], mc_rows: list[dict]) -> None:
    cells = [(d, m, a) for d in DATASETS_SEG for m in MODELS for a in ADAPTATIONS]
    fig, pairs = reliability_axes(cells, "Segmentation top-label reliability over all valid test pixels (ignore pixels excluded)\n15 right-closed bins; seed 42 (ensemble members 42/43/44); counts include empty bins")
    records = []
    for (ax, hist), cell in zip(pairs, cells):
        paths = {
            "Deterministic": Path(deterministic[(*cell, 42)]["prediction_path"]),
            "Deep Ensemble": Path(next(row for row in ensemble_rows if (row["dataset"], row["model"], row["adaptation"]) == cell)["output_path"]) / "ensemble_predictions.npz",
            "MC Dropout": Path(next(row for row in mc_rows if row["task"] == "segmentation" and int(row["seed"]) == 42 and (row["dataset"], row["model"], row["adaptation"]) == cell)["output_path"]) / "aggregate_uncertainty_maps.npz",
        }
        for method, source in paths.items():
            data = seg_reliability_from_npz(source)
            add_curve(ax, hist, method, data)
            records.extend(bin_records(cell, method, data, "decision"))
        ax.set_ylabel("Pixel accuracy (fraction)")
    finish_reliability(path, fig, records)


def plot_performance_calibration(path: Path, rows: list[dict[str, Any]], task: str) -> None:
    datasets = DATASETS_CLASS if task == "classification" else DATASETS_SEG
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    model_markers = {"dofa": "o", "panopticon": "s"}
    for ax, (dataset, adaptation) in zip(axes.flat, [(d, a) for d in datasets for a in ADAPTATIONS]):
        performance = "miou" if task == "segmentation" else "macro_f1" if dataset == "treesatai" else "accuracy"
        selected = [row for row in rows if row["dataset"] == dataset and row["adaptation"] == adaptation and row["status"] == "COMPLETE"]
        for row in selected:
            ax.scatter(row["ece_15"], row[performance], s=70, marker=model_markers[row["model"]], color=METHOD_COLORS[row["uq_method"]], edgecolor="black", linewidth=0.4)
        ax.set_title(f"{DATASET_DISPLAY[dataset]} · {adaptation.replace('_', ' ')}")
        ax.set_xlabel("ECE-15 (fraction; lower is better)")
        ax.set_ylabel(performance + " (fraction; higher is better)")
        ax.grid(alpha=0.2)
    method_handles = [plt.Line2D([], [], marker="o", linestyle="", color=METHOD_COLORS[m], label=m) for m in METHOD_ORDER if m != "Temperature Scaling" or task == "classification"]
    model_handles = [plt.Line2D([], [], marker=model_markers[m], linestyle="", color="black", label=m.upper()) for m in MODELS]
    fig.legend(handles=method_handles + model_handles, loc="lower center", ncol=len(method_handles) + 2, frameon=False)
    fig.suptitle(f"{task.title()} performance–calibration trade-off", fontsize=14)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def decode_ids(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values.tolist()]


def plot_uncertainty_maps(path: Path, dataset: str, mc_rows: list[dict[str, str]], fixed_ids: dict[str, Any]) -> None:
    sample_id = fixed_ids["datasets"][dataset]["final_test_research_subset_ids"][0]
    maps = []
    for model in MODELS:
        for adaptation in ADAPTATIONS:
            row = next(row for row in mc_rows if row["task"] == "segmentation" and int(row["seed"]) == 42 and (row["dataset"], row["model"], row["adaptation"]) == (dataset, model, adaptation))
            archive_path = Path(row["output_path"]) / "aggregate_uncertainty_maps.npz"
            with np.load(archive_path, allow_pickle=False) as archive:
                ids = decode_ids(archive["sample_id"])
                require(sample_id in ids, f"Fixed sample {sample_id} absent from {archive_path}")
                index = ids.index(sample_id)
                valid = archive["valid_mask"][index].astype(bool)
                values = {
                    "label": archive["label"][index].astype(float),
                    "prediction": archive["prediction"][index].astype(float),
                    "confidence": archive["confidence"][index].astype(float),
                    "predictive_entropy": archive["predictive_entropy"][index].astype(float),
                    "mi_style_disagreement": archive["mi_style_disagreement"][index].astype(float),
                }
                for key in ("label", "prediction", "confidence", "predictive_entropy", "mi_style_disagreement"):
                    values[key][~valid] = np.nan
                maps.append((model, adaptation, values))
    classes = 4 if dataset == "cloudsen12" else 2
    disagreement_values = np.concatenate([item[2]["mi_style_disagreement"][np.isfinite(item[2]["mi_style_disagreement"])] for item in maps])
    disagreement_max = max(float(np.percentile(disagreement_values, 99.5)), 1e-8)
    fig = plt.figure(figsize=(16, 12.6))
    grid = fig.add_gridspec(
        5,
        5,
        height_ratios=(1, 1, 1, 1, 0.075),
        left=0.06,
        right=0.98,
        bottom=0.06,
        top=0.89,
        wspace=0.10,
        hspace=0.13,
    )
    axes = np.asarray([[fig.add_subplot(grid[row, column]) for column in range(5)] for row in range(4)])
    columns = ("label", "prediction", "confidence", "predictive_entropy", "mi_style_disagreement")
    titles = ("Ground truth", "MC mean prediction", "Confidence", "Predictive entropy", "MI-style disagreement")
    class_cmap = plt.get_cmap("tab10", classes)
    last_images = {}
    for row_index, (model, adaptation, values) in enumerate(maps):
        for column_index, key in enumerate(columns):
            if key in {"label", "prediction"}:
                image = axes[row_index, column_index].imshow(
                    values[key],
                    cmap=class_cmap,
                    vmin=-0.5,
                    vmax=classes - 0.5,
                    interpolation="nearest",
                )
            elif key == "confidence":
                image = axes[row_index, column_index].imshow(values[key], cmap="viridis", vmin=0, vmax=1)
            elif key == "predictive_entropy":
                image = axes[row_index, column_index].imshow(values[key], cmap="magma", vmin=0, vmax=math.log(classes))
            else:
                image = axes[row_index, column_index].imshow(values[key], cmap="cividis", vmin=0, vmax=disagreement_max)
            last_images[key] = image
            axes[row_index, column_index].set_xticks([])
            axes[row_index, column_index].set_yticks([])
            if row_index == 0:
                axes[row_index, column_index].set_title(titles[column_index])
        axes[row_index, 0].set_ylabel(f"{model.upper()}\n{adaptation.replace('_', ' ')}")
    class_names = (
        ("clear", "thick cloud", "thin cloud", "cloud shadow")
        if dataset == "cloudsen12"
        else ("background", "building")
    )
    categorical_axis = fig.add_subplot(grid[4, 0:2])
    categorical_bar = fig.colorbar(
        last_images["prediction"], cax=categorical_axis, orientation="horizontal", ticks=np.arange(classes)
    )
    categorical_bar.ax.set_xticklabels(class_names, fontsize=8)
    categorical_bar.set_label("Segmentation class", fontsize=9)
    for column_index, key in enumerate(columns[2:], start=2):
        color_axis = fig.add_subplot(grid[4, column_index])
        color_bar = fig.colorbar(last_images[key], cax=color_axis, orientation="horizontal")
        color_bar.ax.tick_params(labelsize=8)
        color_bar.set_label(titles[column_index], fontsize=9)
    fig.suptitle(
        f"{DATASET_DISPLAY[dataset]} downstream MC-Dropout uncertainty\nfixed test sample: {sample_id}",
        fontsize=13,
        y=0.975,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(rows: list[dict[str, Any]], task: str, dataset: str) -> list[str]:
    selected = [row for row in rows if row["dataset"] == dataset]
    if task == "classification":
        lines = [
            "| Model | Adaptation | UQ | Seeds/basis | Accuracy | Macro-F1 | NLL | Brier | ECE-15 | Status |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in selected:
            lines.append(f"| {row['model'].upper()} | {row['adaptation']} | {row['uq_method']} | {row['seeds']} | {fmt(row,'accuracy')} | {fmt(row,'macro_f1')} | {fmt(row,'nll')} | {fmt(row,'brier')} | {fmt(row,'ece_15')} | {row['status']} |")
        return lines
    if dataset == "cloudsen12":
        lines = [
            "| Model | Adaptation | UQ | Seeds/basis | mIoU | IoU clear | IoU thick cloud | IoU thin cloud | IoU shadow | Pixel acc. | NLL | Brier | ECE-15 | Status |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in selected:
            lines.append(f"| {row['model'].upper()} | {row['adaptation']} | {row['uq_method']} | {row['seeds']} | {fmt(row,'miou')} | {fmt(row,'iou_clear')} | {fmt(row,'iou_thick_cloud')} | {fmt(row,'iou_thin_cloud')} | {fmt(row,'iou_cloud_shadow')} | {fmt(row,'pixel_accuracy')} | {fmt(row,'nll')} | {fmt(row,'brier')} | {fmt(row,'ece_15')} | {row['status']} |")
        return lines
    lines = [
        "| Model | Adaptation | UQ | Seeds/basis | mIoU | IoU bg/building | Pixel acc. | NLL | Brier | ECE-15 | Building ECE | Classwise ECE bg/building | Foreground NLL/Brier | Boundary ECE | Status |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in selected:
        lines.append(f"| {row['model'].upper()} | {row['adaptation']} | {row['uq_method']} | {row['seeds']} | {fmt(row,'miou')} | {fmt(row,'iou_background')} / {fmt(row,'iou_building')} | {fmt(row,'pixel_accuracy')} | {fmt(row,'nll')} | {fmt(row,'brier')} | {fmt(row,'ece_15')} | {fmt(row,'foreground_ece_15')} | {fmt(row,'classwise_ece_background')} / {fmt(row,'classwise_ece_building')} | {fmt(row,'foreground_nll')} / {fmt(row,'foreground_brier')} | {fmt(row,'boundary_ece_15')} | {row['status']} |")
    return lines


def write_report(path: Path, classification: list[dict[str, Any]], segmentation: list[dict[str, Any]], robustness: list[dict[str, Any]]) -> None:
    lines = [
        "# Final Thesis Results Package",
        "",
        "Completion entry point: [audit closure](__COMPLETION_ROOT__/COMPLETION_REPORT.md), [RQ results section](__COMPLETION_ROOT__/RESULTS_SECTION.md), and [24 same-checkpoint deterministic/dropout-off/MC comparisons](__COMPLETION_ROOT__/mc_dropout_three_way.csv). The core tables below retain the frozen MC-versus-original-deterministic reporting matrix; the linked supplement isolates the fixed-weight inference effect.",
        "",
        "This package contains only results admitted by the frozen C1–C5 validation and promotion audits. No model was trained and no model inference was run for C6; C6 only aggregates saved metrics/predictions and renders figures.",
        "",
        "## Reporting rules",
        "",
        "- Deterministic and Temperature-Scaling rows are mean ± sample standard deviation over seeds 42/43/44.",
        "- Deep Ensemble is one separate three-member probability-mean result. It is never pooled into member mean ± std.",
        "- The MC-Dropout main matrix is the predeclared seed-42 paired result with 30 probability passes. Three-seed MC robustness is reported separately for only the four predeclared cells.",
        "- Core Temperature Scaling is applicable only to EuroSAT, fitted on its dedicated calibration split. TreeSatAI and segmentation are N/A according to the authoritative master matrix and dated freeze; all historical TreeSatAI TS results are preserved separately below.",
        "- Segmentation Temperature Scaling is outside the frozen protocol and is reported as N/A, not missing.",
        "- TreeSatAI accuracy is strict exact match across the 15-label vector; Macro-F1 is the primary task performance view. Its ECE flattens all binary decisions: confidence=max(p,1-p), outcome=correctness at threshold 0.5. Its NLL/Brier average sample-label decisions. EuroSAT uses top-label ECE, multiclass NLL and class-summed Brier.",
        "- Metrics are fractions, not percentages; NLL is in nats. Segmentation multiclass Brier sums classes before averaging valid pixels. Positive-label reliability is a separate diagnostic and is not the main decision ECE.",
        "",
        "## Classification results",
    ]
    for dataset in DATASETS_CLASS:
        lines.extend(["", f"### {DATASET_DISPLAY[dataset]}", "", *markdown_table(classification, "classification", dataset)])
    lines.extend(["", "## Segmentation results"])
    for dataset in DATASETS_SEG:
        lines.extend(["", f"### {DATASET_DISPLAY[dataset]}", "", *markdown_table(segmentation, "segmentation", dataset)])
    lines.extend([
        "",
        "## MC-Dropout robustness subset",
        "",
        "| Task | Dataset | Model | Adaptation | Seeds | Primary performance (accuracy / Macro-F1 / mIoU) | NLL | Brier | ECE-15 |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ])
    for row in robustness:
        perf = "miou" if row["task"] == "segmentation" else "macro_f1" if row["dataset"] == "treesatai" else "accuracy"
        lines.append(f"| {row['task']} | {row['dataset']} | {row['model'].upper()} | {row['adaptation']} | {row['seeds']} | {fmt(row,perf)} | {fmt(row,'nll')} | {fmt(row,'brier')} | {fmt(row,'ece_15')} |")
    lines.extend([
        "",
        "## Reliability diagrams",
        "",
        "![Classification reliability](figures/classification_reliability_grid.png)",
        "",
        "![Segmentation reliability](figures/segmentation_reliability_grid.png)",
        "",
        "Reliability curves use 15 equal-width right-closed bins (lo,hi], with 0 included in the first bin; empty bins retain count 0 and undefined means. Curves show seed 42, except the single ensemble of seeds 42/43/44. Histograms display every bin count (symlog count scale). Classification uses top-label confidence/correctness for EuroSAT and flattened binary-decision confidence/correctness for TreeSatAI. Segmentation pools all valid test pixels, excluding ignore pixels. Exact plotted bins and counts are saved alongside each PNG as CSV. Plot ECE is checked against the corresponding seed-level master metric.",
        "",
        "![TreeSatAI positive-label diagnostic](figures/treesatai_positive_label_diagnostic.png)",
        "",
        "The separate TreeSatAI figure pools p(label=1) against the binary label, including negative labels. This quantity differs from decision ECE and from the mean of 15 label-specific ECE values. Label-level diagnostics and 10/15/30-bin sensitivity are retained in the companion audit evidence linked below.",
        "",
        "## Performance–calibration views",
        "",
        "![Classification primary performance versus ECE: EuroSAT accuracy and TreeSatAI Macro-F1](figures/classification_accuracy_vs_ece.png)",
        "",
        "![Segmentation mIoU versus ECE](figures/segmentation_miou_vs_ece.png)",
        "",
        "These plots are descriptive. They do not imply statistical significance or a single preferred operating point.",
        "",
        "## Qualitative segmentation uncertainty",
        "",
        "![CloudSEN12 uncertainty maps](figures/cloudsen12_mc_uncertainty_maps.png)",
        "",
        "![SpaceNet7 uncertainty maps](figures/spacenet7_mc_uncertainty_maps.png)",
        "",
        "Each map uses the first ID from the prospectively fixed C4/C5 research subset, selected before stochastic results were inspected. Predictive entropy is descriptive; MI-style disagreement is an epistemic proxy and is not claimed to be true epistemic uncertainty. A common within-dataset 99.5th-percentile color cap is used only for visual comparability.",
        "",
        "## Applicability and caveats",
        "",
        "- All 8 classification cells have deterministic, MC Dropout and Deep Ensemble results. EuroSAT has 4 applicable TS cells; TreeSatAI has 4 N/A TS cells. All 12 N/A cells carry blank metrics rather than zeros.",
        "- All 8 segmentation cells have deterministic, MC Dropout, and Deep Ensemble results; Temperature Scaling is N/A by design.",
        "- SpaceNet7 boundary calibration uses valid boundary pixels only. Boundary-free per-image values remain undefined in the underlying archives.",
        "- The C3 segmentation ensemble summary CSV omitted SpaceNet7 classwise-ECE columns. This package restores the already-computed values from each validated ensemble manifest; no new inference or metric estimation was performed.",
        "- EuroSAT retains a frozen backend-specific preprocessing history: immutable DOFA comparators use their historical RGB normalization, whereas Panopticon uses the newer final train-statistics convention. Cross-model EuroSAT comparisons must retain this confound disclosure.",
        "- Some full-finetuning learning curves deteriorate after the validation-selected optimum; final rows use only the frozen validation-selected best checkpoints.",
        "",
        "## Machine-readable package",
        "",
        "- `tables/classification_results.csv`",
        "- `tables/segmentation_results.csv`",
        "- `tables/mc_dropout_robustness.csv`",
        "- `tables/method_applicability.csv`",
        "- `tables/source_provenance.csv`",
        "- `tables/package_manifest.json`",
        "",
        "The package manifest pins source summaries, raw predictions used in figures, generator code, and generated tables/figures by SHA256. The complete seed-level master is copied into tables/thesis_master_results.csv; the MC robustness summary for four selected cells is distinct from the seed-42 display.",
        "",
        "Further direct-prediction diagnostics from the completed audit: [classification class diagnostics](../../core_rq_audit_20260918/class_diagnostics.csv), [segmentation class diagnostics](../../core_rq_audit_20260918/segmentation_class_diagnostics.csv), [classification 10/15/30-bin tables](../../core_rq_audit_20260918/classification_reliability_bins.csv), [segmentation 10/15/30-bin tables](../../core_rq_audit_20260918/segmentation_reliability_bins.csv). SpaceNet7 building calibration evaluates building probability against the building indicator across all valid pixels, not only true-building pixels.",
    ])
    text = "\n".join(lines) + "\n"
    completion_relative = os.path.relpath(REPORTS / "core_rq_completion_20260921", path.parent)
    text = text.replace("__COMPLETION_ROOT__", completion_relative)
    audit_relative = os.path.relpath(REPORTS / "core_rq_audit_20260918", path.parent)
    text = text.replace("../../core_rq_audit_20260918", audit_relative)
    path.write_text(text, encoding="utf-8")


def validate_package_tables(rows: list[dict], master_rows: list[dict]) -> list[dict]:
    checks = []
    for row in rows:
        key = (row["dataset"], row["model"], row["adaptation"], METHOD_KEYS[row["uq_method"]])
        record_type = "INDIVIDUAL_SEED" if row["uq_method"] == "MC Dropout" else "ENSEMBLE" if row["uq_method"] == "Deep Ensemble" else "NOT_APPLICABLE" if row["status"] == "N/A" else "MEAN_STD"
        matches = [m for m in master_rows if (m["dataset"], m["model"], m["adaptation"], m["uq_method"]) == key and m["record_type"] == record_type and (record_type != "INDIVIDUAL_SEED" or m["seed"] == "42")]
        require(len(matches) == 1, f"Ambiguous master counterpart: {key} {record_type}")
        master = matches[0]
        require(row["status"] == master["status"], f"Status differs from master: {key}")
        differences = []
        for metric in CLASS_METRICS if row["task"] == "classification" else SEG_METRICS:
            for name in (metric, metric + "_std"):
                actual, expected = number(row.get(name)), number(master.get(name))
                require((actual is None) == (expected is None), f"Missing or spurious metric: {key} {name}")
                if actual is not None:
                    differences.append(abs(actual - expected))
                    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), f"Master metric mismatch: {key} {name}")
        checks.append({"cell": list(key), "status": "PASS", "metric_count": len(differences), "max_absolute_error": max(differences, default=0.0)})
    return checks


def validate_plotted_ece(figures: Path, master_rows: list[dict]) -> list[dict]:
    checks = []
    for filename in ("classification_reliability_grid.csv", "segmentation_reliability_grid.csv"):
        groups = defaultdict(list)
        for row in read_csv(figures / filename):
            groups[(row["dataset"], row["model"], row["adaptation"], row["uq_method"])].append(row)
        for cell, rows in groups.items():
            require(len(rows) == 15, f"Plot does not retain all bins: {cell}")
            count = sum(int(row["count"]) for row in rows)
            actual = sum(int(row["count"]) * abs(float(row["mean_confidence"]) - float(row["observed_frequency"])) for row in rows if int(row["count"])) / count
            matches = [row for row in master_rows if (row["dataset"], row["model"], row["adaptation"], row["uq_method"]) == cell and (row["record_type"] == "ENSEMBLE" if cell[3] == "deep_ensemble" else row["record_type"] == "INDIVIDUAL_SEED" and row["seed"] == "42")]
            require(len(matches) == 1, f"Missing plot counterpart: {cell}")
            expected = float(matches[0]["ece_15"])
            require(abs(actual - expected) < 1e-7, f"Plot ECE differs from metric: {cell}: {actual} vs {expected}")
            checks.append({"cell": list(cell), "count": count, "plot_ece_15": actual, "source_ece_15": expected, "absolute_error": abs(actual - expected), "status": "PASS"})
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPORTS / "final_thesis_package",
                        help="New directory for report, tables and figures; refuses existing destinations")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    require(not output_root.exists(), f"Refusing to overwrite {output_root}")
    mc_rows = read_csv(REPORTS / "mc_dropout_results.csv")
    require(len(mc_rows) == 24, "C5 results must contain 24 rows")
    master_rows = read_csv(MASTER)
    scope = temperature_scope(master_rows)
    class_det, ts_rows, class_ensembles = build_classification_sources()
    seg_det, seg_ensembles = build_segmentation_sources()
    classification = build_classification_table(class_det, ts_rows, class_ensembles, mc_rows, scope)
    segmentation = build_segmentation_table(seg_det, seg_ensembles, mc_rows, scope)
    robustness = build_robustness_table(mc_rows)
    table_checks = validate_package_tables(classification + segmentation, master_rows)
    historical_ts = [{**row, "status": "HISTORICAL_DIAGNOSTIC", "core_applicable": "False"} for row in ts_rows if row["dataset"] == "treesatai"]
    require(len(historical_ts) == 12, "Historical TreeSatAI TS record count changed")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=".c6_final_", dir=output_root.parent))
    temp_tables, temp_figures = temp_root / "tables", temp_root / "figures"
    temp_report = temp_root / "final_thesis_results.md"
    temp_tables.mkdir()
    temp_figures.mkdir()
    try:
        base_fields = ["task", "dataset", "model", "adaptation", "uq_method", "status", "seeds", "n_runs", "reporting_basis"]
        class_fields = base_fields + [value for metric in CLASS_METRICS for value in (metric, f"{metric}_std")] + ["source"]
        seg_fields = base_fields + [value for metric in SEG_METRICS for value in (metric, f"{metric}_std")] + ["source"]
        write_csv(temp_tables / "classification_results.csv", classification, class_fields)
        write_csv(temp_tables / "segmentation_results.csv", segmentation, seg_fields)
        robust_fields = list(dict.fromkeys(base_fields + [value for metric in (*CLASS_METRICS, *SEG_METRICS) for value in (metric, f"{metric}_std")] + ["source"]))
        write_csv(temp_tables / "mc_dropout_robustness.csv", robustness, robust_fields)
        applicability = [{key: row[key] for key in ("task", "dataset", "model", "adaptation", "uq_method", "status", "reporting_basis", "source")} for row in classification + segmentation]
        write_csv(temp_tables / "method_applicability.csv", applicability, list(applicability[0]))
        write_csv(temp_tables / "historical_treesatai_temperature_scaling.csv", historical_ts, list(historical_ts[0]))
        shutil.copyfile(MASTER, temp_tables / "thesis_master_results.csv")
        plot_classification_reliability(temp_figures / "classification_reliability_grid.png", ts_rows, class_ensembles, mc_rows, class_det, scope)
        plot_classification_reliability(temp_figures / "treesatai_positive_label_diagnostic.png", ts_rows, class_ensembles, mc_rows, class_det, scope, positive_label=True)
        plot_segmentation_reliability(temp_figures / "segmentation_reliability_grid.png", seg_det, seg_ensembles, mc_rows)
        plot_checks = validate_plotted_ece(temp_figures, master_rows)
        plot_performance_calibration(temp_figures / "classification_accuracy_vs_ece.png", classification, "classification")
        plot_performance_calibration(temp_figures / "segmentation_miou_vs_ece.png", segmentation, "segmentation")
        fixed_ids = read_json(REPORTS / "mc_dropout_segmentation_research_subset_ids.json")
        plot_uncertainty_maps(temp_figures / "cloudsen12_mc_uncertainty_maps.png", "cloudsen12", mc_rows, fixed_ids)
        plot_uncertainty_maps(temp_figures / "spacenet7_mc_uncertainty_maps.png", "spacenet7", mc_rows, fixed_ids)
        write_report(temp_report, classification, segmentation, robustness)
        with temp_report.open("a", encoding="utf-8") as handle:
            handle.write("\n## Historical TreeSatAI temperature scaling: excluded from the core comparison\n\n")
            handle.write("All 12 seed-level historical results, including adverse changes, are retained in [this table](tables/historical_treesatai_temperature_scaling.csv). Temperature was fitted on official validation, also used for checkpoint selection; test labels did not fit temperature. Such reuse does not by itself invalidate held-out evaluation. The exclusion follows the dated reporting policy, not a mathematical requirement for a separately named calibration split.\n\n")
            handle.write("| Model | Adaptation | Seed | Temperature | Δ exact-match accuracy | Δ Macro-F1 | Δ NLL | Δ Brier | Δ decision ECE-15 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in historical_ts:
                delta = [float(row[metric + "_after"]) - float(row[metric + "_before"]) for metric in CLASS_METRICS]
                handle.write(f"| {row['model']} | {row['adaptation']} | {row['seed']} | {float(row['temperature']):.4f} | " + " | ".join(f"{value:+.6f}" for value in delta) + " |\n")
            handle.write("\nThe sequence is explicit: **2026-08-24** historical TreeSatAI TS results were available (`reports/c3_treesatai_temperature_scaling.md`); **2026-08-25** the freeze excluded them from final core reporting (`reports/pre_uq_protocol_freeze.md`); **2026-08-26** the old C6 package still marked them COMPLETE (`reports/final_thesis_results.md`); **2026-08-27** the master/RQ3 tables followed the freeze and marked them N/A (`reports/thesis_master_results.csv`, `reports/rq3_uq_effects.csv`). Thus the TS exclusion was not registered before all TS outcomes were visible. This rebuilt package resolves the stale C6 conflict while preserving the old files. The prospective MC plan and historical TS reporting decision have different timelines. No motive for the exclusion is inferred.\n")
        sources = [MASTER, Path(__file__).resolve(), REPORTS / "dofa_eurosat_final_manifest.json",
                   sorted((REPORTS / "c1_classification_run_audits").glob("*/c1_run_audit.json"))[-1],
                   REPORTS / "c2_segmentation_run_audits" / "20260824T162250Z" / "audit.csv",
                   REPORTS / "c3_classification_results.csv", REPORTS / "c3_segmentation_ensemble_results.csv",
                   REPORTS / "c3_treesatai_temperature_scaling.md", REPORTS / "c3_treesatai_temperature_scaling_results.csv",
                   REPORTS / "mc_dropout_results.csv", REPORTS / "pre_uq_protocol_freeze.md",
                   REPORTS / "mc_dropout_segmentation_research_subset_ids.json",
                   REPORTS / "final_thesis_results.md", REPORTS / "rq3_uq_effects.csv"]
        sources += [Path(row["output_path"]) / "manifest.json" for row in seg_ensembles]
        # Pin the raw predictions used in each figure, not another agent's summary.
        sources += [Path(record["prediction_path"]) for key, record in {**class_det, **seg_det}.items() if key[-1] == 42]
        sources += [Path(row["output_path"]) / "ensemble_predictions.parquet" for row in class_ensembles]
        sources += [Path(row["output_path"]) / "ensemble_predictions.npz" for row in seg_ensembles]
        sources += [Path(row["output_path"]) / ("predictions.parquet" if row["task"] == "classification" else "aggregate_uncertainty_maps.npz") for row in mc_rows if row["seed"] == "42"]
        sources += [Path(row["output_path"]) / "test_predictions.parquet" for row in ts_rows if row["dataset"] == "eurosat" and row["seed"] == "42"]
        source_rows = [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in dict.fromkeys(sources)]
        write_csv(temp_tables / "source_provenance.csv", source_rows, ["path", "sha256", "size_bytes"])
        validation = {"scope_source": str(MASTER), "table_checks": table_checks, "plot_checks": plot_checks,
                      "na_cells": sum(row["status"] == "N/A" for row in classification + segmentation),
                      "historical_ts_rows": len(historical_ts), "all_passed": True}
        (temp_tables / "package_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        generated = [temp_report, *sorted(temp_tables.glob("*")), *sorted(temp_figures.glob("*"))]
        manifest = {"schema_version": 2, "scope": "C6 aggregation and visualization only; no model training or inference",
                    "output_root": str(output_root), "validated_source_count": len(source_rows),
                    "classification_rows": len(classification), "segmentation_rows": len(segmentation),
                    "mc_robustness_rows": len(robustness), "generated_artifacts": [
                        {"path": str(path.relative_to(temp_root)), "sha256": sha256(path), "size_bytes": path.stat().st_size}
                        for path in generated]}
        (temp_tables / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temp_root, output_root)
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)
    print(json.dumps({"report": str(output_root / temp_report.name), "classification_rows": len(classification),
                      "segmentation_rows": len(segmentation), "na_cells": 12, "historical_ts_rows": 12,
                      "plot_ece_checks": len(plot_checks)}, indent=2))


if __name__ == "__main__":
    main()
