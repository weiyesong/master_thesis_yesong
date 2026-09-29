#!/usr/bin/env python3
"""Build the final A–C thesis evidence matrix without model execution."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FINAL_CSV = REPORTS / "thesis_master_results.csv"
FINAL_MD = REPORTS / "thesis_evidence_matrix.md"

MODELS = ("dofa", "panopticon")
ADAPTATIONS = ("frozen", "full_finetune")
CLASS_DATASETS = ("eurosat", "treesatai")
SEG_DATASETS = ("cloudsen12", "spacenet7")
DISPLAY = {"eurosat": "EuroSAT", "treesatai": "TreeSatAI", "cloudsen12": "CloudSEN12", "spacenet7": "SpaceNet7"}

METRICS = (
    "accuracy", "macro_f1", "miou", "pixel_accuracy", "nll", "brier", "ece_15",
    "iou_clear", "iou_thick_cloud", "iou_thin_cloud", "iou_cloud_shadow",
    "iou_background", "iou_building", "foreground_ece_15", "foreground_nll",
    "foreground_brier", "classwise_ece_background", "classwise_ece_building",
    "boundary_ece_15", "boundary_foreground_ece_15",
)

IDENTITY_FIELDS = (
    "task", "dataset", "model", "adaptation", "uq_method", "record_type", "status", "seed",
    "contributing_seeds", "n_runs", "replication_role", "applicability_reason",
    "checkpoint_selection_criterion", "temperature_fit_split", "temperature",
    "dropout_probability", "stochastic_passes", "ensemble_members", "metric_semantics",
)
COST_FIELDS = (
    "training_seconds", "training_seconds_std", "training_cost_basis",
    "additional_uq_training_seconds", "additional_uq_training_cost_status",
    "inference_seconds", "inference_seconds_std", "inference_cost_basis",
    "fit_optimizer_iterations", "fit_optimizer_iterations_std", "compute_proxy",
)
PROVENANCE_FIELDS = ("run_id", "checkpoint_path", "checkpoint_sha256", "source_paths", "notes")
CSV_FIELDS = (*IDENTITY_FIELDS, *[field for metric in METRICS for field in (metric, f"{metric}_std")], *COST_FIELDS, *PROVENANCE_FIELDS)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    output = float(value)
    return output if math.isfinite(output) else None


def slug(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def base_row(**values: Any) -> dict[str, Any]:
    row = {field: None for field in CSV_FIELDS}
    row.update(values)
    return row


def flatten_classification(metrics: dict[str, Any]) -> dict[str, float | None]:
    return {
        "accuracy": number(metrics.get("accuracy")),
        "macro_f1": number(metrics.get("macro_f1")),
        "nll": number(metrics.get("nll")),
        "brier": number(metrics.get("brier")),
        "ece_15": number(metrics.get("ece_15", metrics.get("ece"))),
    }


def flatten_segmentation(metrics: dict[str, Any]) -> dict[str, float | None]:
    output = {key: number(metrics.get(key)) for key in ("miou", "pixel_accuracy", "nll", "brier", "ece_15")}
    for name, value in metrics.get("per_class_iou", {}).items():
        output[f"iou_{slug(name)}"] = number(value)
    for key in ("foreground_ece_15", "foreground_nll", "foreground_brier"):
        output[key] = number(metrics.get(key))
    classwise = metrics.get("classwise_calibration", {})
    for name in ("background", "building"):
        output[f"classwise_ece_{name}"] = number(classwise.get(name, {}).get("ece_15"))
    boundary = metrics.get("boundary_calibration", {})
    output["boundary_ece_15"] = number(boundary.get("ece_15"))
    output["boundary_foreground_ece_15"] = number(boundary.get("foreground_ece_15"))
    return output


def metric_semantics(task: str, dataset: str) -> str:
    if dataset == "treesatai":
        return "multilabel: accuracy=strict exact match; NLL/Brier average sample-label decisions; ECE-15 flattened binary decisions"
    if task == "classification":
        return "multiclass image-level metrics"
    if dataset == "spacenet7":
        return "valid-pixel multiclass metrics; building/foreground and valid-boundary calibration additionally reported"
    return "valid-pixel multiclass segmentation metrics"


def criterion(task: str) -> str:
    return "minimum validation NLL (earliest tie)" if task == "classification" else "maximum validation mIoU (earliest tie)"


def source_join(paths: Iterable[Path | str]) -> str:
    values = []
    for path in paths:
        text = rel(path) if isinstance(path, Path) else str(path)
        if text not in values:
            values.append(text)
    return ";".join(values)


def close_metrics(left: dict[str, Any], right: dict[str, Any], fields: Iterable[str], label: str) -> None:
    for field in fields:
        a, b = number(left.get(field)), number(right.get(field))
        if a is not None and b is not None:
            require(math.isclose(a, b, rel_tol=1.0e-7, abs_tol=1.0e-9), f"Metric mismatch {label} {field}: {a} vs {b}")


def aggregate_rows(records: list[dict[str, Any]], *, replication_role: str, notes: str = "") -> dict[str, Any]:
    require(len(records) >= 2, "Mean/std requires at least two rows")
    first = records[0]
    row = base_row(
        task=first["task"], dataset=first["dataset"], model=first["model"], adaptation=first["adaptation"],
        uq_method=first["uq_method"], record_type="MEAN_STD", status="COMPLETE", seed=None,
        contributing_seeds=";".join(str(record["seed"]) for record in sorted(records, key=lambda value: int(value["seed"]))),
        n_runs=len(records), replication_role=replication_role, checkpoint_selection_criterion=first["checkpoint_selection_criterion"],
        temperature_fit_split=first["temperature_fit_split"], dropout_probability=first["dropout_probability"],
        stochastic_passes=first["stochastic_passes"], ensemble_members=None, metric_semantics=first["metric_semantics"],
        training_cost_basis="MEAN_AND_SAMPLE_STD_OF_SEED_TRAINING_SECONDS",
        additional_uq_training_cost_status=first["additional_uq_training_cost_status"],
        inference_cost_basis="MEAN_AND_SAMPLE_STD_OF_RECORDED_SEED_INFERENCE_SECONDS",
        compute_proxy=first["compute_proxy"], source_paths="", notes=notes,
    )
    # Flatten source lists without relying on filesystem order.
    row["source_paths"] = ";".join(dict.fromkeys(path for record in records for path in record["source_paths"].split(";") if path))
    for field in METRICS:
        values = [number(record.get(field)) for record in records]
        finite = [value for value in values if value is not None]
        if finite:
            require(len(finite) == len(records), f"Partial aggregate metric {field}")
            row[field] = float(np.mean(finite))
            row[f"{field}_std"] = float(np.std(finite, ddof=1))
    for field in ("training_seconds", "inference_seconds", "fit_optimizer_iterations"):
        values = [number(record.get(field)) for record in records]
        finite = [value for value in values if value is not None]
        if finite:
            require(len(finite) == len(records), f"Partial aggregate cost {field}")
            row[field] = float(np.mean(finite))
            row[f"{field}_std"] = float(np.std(finite, ddof=1))
        elif field == "inference_seconds":
            row["inference_cost_basis"] = "UNAVAILABLE_NOT_RECORDED"
    if first["uq_method"] == "temperature_scaling":
        row["training_cost_basis"] = "REUSED_DETERMINISTIC_CHECKPOINT: MEAN_AND_SAMPLE_STD_OF_MEMBER_TRAINING_SECONDS"
        row["inference_cost_basis"] = "UNAVAILABLE_POSTHOC_FIT_AND_APPLY_TIME_NOT_RECORDED"
        if number(row["fit_optimizer_iterations"]) is not None:
            row["compute_proxy"] = f"scalar fit: mean {float(row['fit_optimizer_iterations']):.1f} optimizer iterations across seeds; 1 probability transform"
    elif first["uq_method"] == "mc_dropout":
        row["training_cost_basis"] = "MEAN_AND_SAMPLE_STD_OF_MC_DROPOUT_SEED_TRAINING_SECONDS"
    elif first["uq_method"] == "deterministic":
        row["training_cost_basis"] = "MEAN_AND_SAMPLE_STD_OF_DETERMINISTIC_SEED_TRAINING_SECONDS"
    temperatures = [number(record.get("temperature")) for record in records]
    if all(value is not None for value in temperatures):
        row["temperature"] = float(np.mean(temperatures))
    return row


def build_rows() -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, int], dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    used_reports: set[str] = set()
    deterministic: dict[tuple[str, str, str, int], dict[str, Any]] = {}

    dofa_path = REPORTS / "dofa_eurosat_final_manifest.json"
    dofa = read_json(dofa_path)
    require(dofa["status"] == "FINAL_IMMUTABLE" and dofa["immutability"]["verified"], "DOFA-EuroSAT manifest not immutable")
    used_reports.add(rel(dofa_path))
    for item in dofa["runs"]:
        run_dir = Path(item["checkpoint_path"]).parent
        summary_path = run_dir / "run_summary.json"
        summary = read_json(summary_path)
        metrics = flatten_classification(summary["evaluation_metrics"]["test"])
        close_metrics(metrics, item["metrics"], ("accuracy", "macro_f1", "nll", "brier", "ece_15"), item["run_id"])
        key = ("eurosat", "dofa", item["adaptation"], int(item["seed"]))
        row = base_row(
            task="classification", dataset=key[0], model=key[1], adaptation=key[2], uq_method="deterministic",
            record_type="INDIVIDUAL_SEED", status="COMPLETE", seed=key[3], contributing_seeds=str(key[3]), n_runs=1,
            replication_role="deterministic_seed", checkpoint_selection_criterion=criterion("classification"),
            metric_semantics=metric_semantics("classification", key[0]), training_seconds=summary["training_seconds"],
            training_cost_basis="MEASURED_CUMULATIVE_TRAINING_WALL_SECONDS", inference_seconds=summary["inference_seconds"]["test"],
            inference_cost_basis="MEASURED_SINGLE_PASS_TEST_INFERENCE_WALL_SECONDS", compute_proxy="1 deterministic pass",
            run_id=item["run_id"], checkpoint_path=item["checkpoint_path"], checkpoint_sha256=item["checkpoint_sha256"],
            source_paths=source_join((dofa_path, summary_path)), **metrics,
        )
        deterministic[key] = row
        rows.append(row)

    c1_path = sorted((REPORTS / "c1_classification_run_audits").glob("*/c1_run_audit.json"))[-1]
    c1 = read_json(c1_path)
    require(c1["summary"]["promotable"] == 18, "Latest C1 audit is not 18/18 promotable")
    used_reports.add(rel(c1_path))
    for item in c1["runs"]:
        require(item["status"] == "PROMOTABLE", f"Unpromoted C1 run {item['run_id']}")
        target = item["target"]
        run_dir = Path(item["run_dir"])
        summary_path = run_dir / "run_summary.json"
        summary = read_json(summary_path)
        metrics = flatten_classification(summary["evaluation_metrics"]["test"])
        close_metrics(metrics, flatten_classification(item["test_metrics_report_only"]), ("accuracy", "macro_f1", "nll", "brier", "ece_15"), item["run_id"])
        key = (target["dataset"], target["model"], target["adaptation"], int(target["seed"]))
        row = base_row(
            task="classification", dataset=key[0], model=key[1], adaptation=key[2], uq_method="deterministic",
            record_type="INDIVIDUAL_SEED", status="COMPLETE", seed=key[3], contributing_seeds=str(key[3]), n_runs=1,
            replication_role="deterministic_seed", checkpoint_selection_criterion=criterion("classification"),
            metric_semantics=metric_semantics("classification", key[0]), training_seconds=summary["training_seconds"],
            training_cost_basis="MEASURED_CUMULATIVE_TRAINING_WALL_SECONDS", inference_seconds=summary["inference_seconds"]["test"],
            inference_cost_basis="MEASURED_SINGLE_PASS_TEST_INFERENCE_WALL_SECONDS", compute_proxy="1 deterministic pass",
            run_id=item["run_id"], checkpoint_path=item["checkpoint"]["path"], checkpoint_sha256=item["checkpoint"]["sha256"],
            source_paths=source_join((c1_path, summary_path)), **metrics,
        )
        deterministic[key] = row
        rows.append(row)
    require(len([key for key in deterministic if key[0] in CLASS_DATASETS]) == 24, "Classification deterministic coverage !=24")

    c2_csv = REPORTS / "c2_segmentation_run_audits/20260824T162250Z/audit.csv"
    c2_json = REPORTS / "c2_segmentation_run_audits/20260824T162250Z/audit.json"
    c2 = read_csv(c2_csv)
    require(len(c2) == 24 and all(item["status"] == "PROMOTABLE" for item in c2), "C2 final audit is not 24/24")
    used_reports.update((rel(c2_csv), rel(c2_json)))
    for item in c2:
        run_dir = Path(item["run_dir"])
        summary_path = run_dir / "run_summary.json"
        results_path = run_dir / "results.json"
        summary = read_json(summary_path)
        results = read_json(results_path)
        metrics = flatten_segmentation(summary["evaluation_metrics"]["test"])
        inference_seconds = number(results.get("metrics", {}).get("inference_seconds", {}).get("test"))
        require(inference_seconds is not None, f"Missing C2 test inference timing: {results_path}")
        key = (item["dataset"], item["model"], item["adaptation"], int(item["seed"]))
        row = base_row(
            task="segmentation", dataset=key[0], model=key[1], adaptation=key[2], uq_method="deterministic",
            record_type="INDIVIDUAL_SEED", status="COMPLETE", seed=key[3], contributing_seeds=str(key[3]), n_runs=1,
            replication_role="deterministic_seed", checkpoint_selection_criterion=criterion("segmentation"),
            metric_semantics=metric_semantics("segmentation", key[0]), training_seconds=summary["training_seconds"],
            training_cost_basis="MEASURED_CUMULATIVE_TRAINING_WALL_SECONDS", inference_seconds=inference_seconds,
            inference_cost_basis="MEASURED_SINGLE_PASS_TEST_INFERENCE_WALL_SECONDS",
            compute_proxy="1 deterministic pass", run_id=item["run_id"], checkpoint_path=summary["best_checkpoint"],
            checkpoint_sha256=summary["best_checkpoint_sha256"], source_paths=source_join((c2_csv, c2_json, summary_path, results_path)), **metrics,
        )
        deterministic[key] = row
        rows.append(row)
    require(len(deterministic) == 48, "Deterministic coverage !=48")

    # Deterministic mean ± sample standard deviation for all 16 cells.
    for task, datasets in (("classification", CLASS_DATASETS), ("segmentation", SEG_DATASETS)):
        for dataset in datasets:
            for model in MODELS:
                for adaptation in ADAPTATIONS:
                    members = [deterministic[(dataset, model, adaptation, seed)] for seed in (42, 43, 44)]
                    rows.append(aggregate_rows(members, replication_role="deterministic_three_seed_summary"))

    c3_class_path = REPORTS / "c3_classification_results.csv"
    c3_class = read_csv(c3_class_path)
    require(len(c3_class) == 32, "C3 classification summary row count !=32")
    used_reports.add(rel(c3_class_path))
    euro_ts: list[dict[str, Any]] = []
    for item in c3_class:
        if not item["method"].startswith("temperature_scaling") or item["dataset"] != "eurosat":
            continue
        dataset, model, adaptation, seed = item["dataset"], item["model"], item["adaptation"], int(item["seed"])
        output = Path(item["output_path"])
        manifest_path = output / "metrics_and_manifest.json"
        manifest = read_json(manifest_path)
        require(manifest["fit_split"] == "calibration", f"Invalid EuroSAT TS split {manifest_path}")
        metrics = flatten_classification(manifest["test_metrics_after"])
        # The historical summary CSV omitted EuroSAT Macro-F1, while each
        # validated per-seed manifest retained it.  Cross-check every summary
        # value that is actually present and use the complete manifest record.
        csv_metrics = {name: number(item[f"{name}_after"]) for name in ("accuracy", "macro_f1", "nll", "brier", "ece_15")}
        close_metrics(metrics, csv_metrics, metrics, str(manifest_path))
        base = deterministic[(dataset, model, adaptation, seed)]
        row = base_row(
            task="classification", dataset=dataset, model=model, adaptation=adaptation, uq_method="temperature_scaling",
            record_type="INDIVIDUAL_SEED", status="COMPLETE", seed=seed, contributing_seeds=str(seed), n_runs=1,
            replication_role="temperature_seed", checkpoint_selection_criterion="minimum validation NLL; scalar T minimizes dedicated calibration-split multiclass NLL",
            temperature_fit_split="dedicated_calibration", temperature=number(item["temperature"]), metric_semantics=metric_semantics("classification", dataset),
            training_seconds=base["training_seconds"], training_cost_basis="REUSED_DETERMINISTIC_CHECKPOINT_TRAINING_SECONDS",
            additional_uq_training_cost_status="N/A_NO_ADDITIONAL_NEURAL_NETWORK_TRAINING", inference_cost_basis="UNAVAILABLE_POSTHOC_FIT_AND_APPLY_TIME_NOT_RECORDED",
            fit_optimizer_iterations=manifest["fit"]["optimizer_iterations"], compute_proxy=f"scalar fit: {manifest['fit']['optimizer_iterations']} optimizer iterations; 1 probability transform",
            run_id=manifest["run_id"], checkpoint_path=manifest["checkpoint_path"], checkpoint_sha256=manifest["checkpoint_sha256"],
            source_paths=source_join((c3_class_path, manifest_path)), notes="Test logits untouched during fitting.", **metrics,
        )
        rows.append(row)
        euro_ts.append(row)
    require(len(euro_ts) == 12, "Valid EuroSAT TS seed rows !=12")
    for model in MODELS:
        for adaptation in ADAPTATIONS:
            members = [row for row in euro_ts if row["model"] == model and row["adaptation"] == adaptation]
            rows.append(aggregate_rows(members, replication_role="temperature_three_seed_summary"))

    freeze_path = REPORTS / "pre_uq_protocol_freeze.md"
    freeze_text = freeze_path.read_text(encoding="utf-8")
    require("TreeSatAI Temperature Scaling is `N/A`" in freeze_text, "TreeSatAI TS freeze not found")
    used_reports.add(rel(freeze_path))
    for model in MODELS:
        for adaptation in ADAPTATIONS:
            rows.append(base_row(
                task="classification", dataset="treesatai", model=model, adaptation=adaptation, uq_method="temperature_scaling",
                record_type="NOT_APPLICABLE", status="N/A", n_runs=0, replication_role="not_applicable",
                applicability_reason="N/A — no independent calibration split under the completed benchmark protocol",
                checkpoint_selection_criterion="N/A — post-hoc method excluded by final freeze", temperature_fit_split="N/A",
                training_cost_basis="N/A", additional_uq_training_cost_status="N/A_METHOD_EXCLUDED", inference_cost_basis="N/A",
                metric_semantics=metric_semantics("classification", "treesatai"), source_paths=source_join((freeze_path,)),
                notes="Validation-fitted C3 artifacts are immutable historical/diagnostic provenance and excluded from final quantitative tables.",
            ))

    # Temperature Scaling is outside the frozen segmentation protocol.
    for dataset in SEG_DATASETS:
        for model in MODELS:
            for adaptation in ADAPTATIONS:
                rows.append(base_row(
                    task="segmentation", dataset=dataset, model=model, adaptation=adaptation, uq_method="temperature_scaling",
                    record_type="NOT_APPLICABLE", status="N/A", n_runs=0, replication_role="not_applicable",
                    applicability_reason="N/A — Temperature Scaling excluded from the frozen segmentation protocol",
                    checkpoint_selection_criterion="N/A", temperature_fit_split="N/A", training_cost_basis="N/A",
                    additional_uq_training_cost_status="N/A_METHOD_EXCLUDED", inference_cost_basis="N/A",
                    metric_semantics=metric_semantics("segmentation", dataset), source_paths=source_join((freeze_path,)),
                ))

    c5_path = REPORTS / "mc_dropout_results.csv"
    c5 = read_csv(c5_path)
    require(len(c5) == 24, "C5 final row count !=24")
    used_reports.add(rel(c5_path))
    mc_rows: list[dict[str, Any]] = []
    for item in c5:
        task, dataset, model, adaptation, seed = item["task"], item["dataset"], item["model"], item["adaptation"], int(item["seed"])
        output = Path(item["output_path"])
        audit_path = output / ("c5_audit.json" if task == "classification" else "manifest.json")
        audit = read_json(audit_path)
        require(audit["passed"], f"Invalid C5 artifact {audit_path}")
        prefix_metrics = {slug(key[3:]): value for key, value in item.items() if key.startswith("mc_")}
        if task == "classification":
            metrics = {key: number(prefix_metrics.get(key)) for key in ("accuracy", "macro_f1", "nll", "brier", "ece_15")}
            audit_metrics = flatten_classification(audit["metrics"])
        else:
            metrics = {metric: number(prefix_metrics.get(metric)) for metric in METRICS}
            audit_metrics = flatten_segmentation(audit["metrics"])
        close_metrics(metrics, audit_metrics, METRICS, str(audit_path))
        training = audit["training_audit"]
        row = base_row(
            task=task, dataset=dataset, model=model, adaptation=adaptation, uq_method="mc_dropout",
            record_type="INDIVIDUAL_SEED", status="COMPLETE", seed=seed, contributing_seeds=str(seed), n_runs=1,
            replication_role=item["replication_role"], checkpoint_selection_criterion=criterion(task),
            dropout_probability=number(item["dropout_probability"]), stochastic_passes=int(item["stochastic_passes"]),
            metric_semantics=metric_semantics(task, dataset), training_seconds=training["training_seconds"],
            training_cost_basis="MEASURED_CUMULATIVE_MC_DROPOUT_TRAINING_WALL_SECONDS", inference_cost_basis="UNAVAILABLE_NOT_RECORDED",
            compute_proxy=f"T={item['stochastic_passes']} stochastic probability passes", run_id=training["run_id"],
            checkpoint_path=training["checkpoint_path"], checkpoint_sha256=training["checkpoint_sha256"],
            source_paths=source_join((c5_path, audit_path)), notes="Primary comparison is paired to deterministic model with the same seed.", **metrics,
        )
        rows.append(row)
        mc_rows.append(row)
    require(sum(int(row["seed"]) == 42 for row in mc_rows) == 16, "MC primary seed42 coverage !=16")
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in mc_rows:
        if row["replication_role"] in {"primary_and_robustness", "robustness"}:
            groups[(row["task"], row["dataset"], row["model"], row["adaptation"])].append(row)
    require(len(groups) == 4 and all(sorted(row["seed"] for row in values) == [42, 43, 44] for values in groups.values()), "Invalid MC robustness subset")
    for values in groups.values():
        rows.append(aggregate_rows(values, replication_role="predeclared_robustness_three_seed_summary"))

    # Classification Deep Ensembles.
    ensemble_class = [item for item in c3_class if item["method"] == "deep_ensemble_probability_mean"]
    require(len(ensemble_class) == 8, "Classification ensemble coverage !=8")
    for item in ensemble_class:
        dataset, model, adaptation = item["dataset"], item["model"], item["adaptation"]
        output = Path(item["output_path"])
        manifest_path = output / "manifest.json"
        members = [deterministic[(dataset, model, adaptation, seed)] for seed in (42, 43, 44)]
        train_sum = sum(float(member["training_seconds"]) for member in members)
        infer_values = [number(member["inference_seconds"]) for member in members]
        infer_sum = sum(infer_values) if all(value is not None for value in infer_values) else None
        metrics = {key: number(item[key]) for key in ("accuracy", "macro_f1", "nll", "brier", "ece_15")}
        rows.append(base_row(
            task="classification", dataset=dataset, model=model, adaptation=adaptation, uq_method="deep_ensemble",
            record_type="ENSEMBLE", status="COMPLETE", contributing_seeds="42;43;44", n_runs=1,
            replication_role="three_member_probability_mean", checkpoint_selection_criterion="N/A — post-hoc probability mean; members use minimum validation NLL",
            ensemble_members=3, metric_semantics=metric_semantics("classification", dataset), training_seconds=train_sum,
            training_cost_basis="DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS", inference_seconds=infer_sum,
            inference_cost_basis="DERIVED_SUM_MEMBER_TEST_INFERENCE_SECONDS; AGGREGATION_OVERHEAD_NOT_RECORDED" if infer_sum is not None else "UNAVAILABLE_NOT_RECORDED",
            compute_proxy="M=3 member probability evaluations", source_paths=source_join((c3_class_path, manifest_path)),
            notes="Single ensemble result; never pooled into member mean ± std. Probabilities, not logits, were averaged.", **metrics,
        ))

    # Segmentation Deep Ensembles: manifests are canonical because the C3 CSV omitted SpaceNet7 classwise ECE.
    c3_seg_path = REPORTS / "c3_segmentation_ensemble_results.csv"
    c3_seg = read_csv(c3_seg_path)
    require(len(c3_seg) == 8, "Segmentation ensemble coverage !=8")
    used_reports.add(rel(c3_seg_path))
    for item in c3_seg:
        dataset, model, adaptation = item["dataset"], item["model"], item["adaptation"]
        manifest_path = Path(item["output_path"]) / "manifest.json"
        manifest = read_json(manifest_path)
        require(manifest["aggregation"].startswith("arithmetic mean"), f"Invalid segmentation ensemble {manifest_path}")
        members = [deterministic[(dataset, model, adaptation, seed)] for seed in (42, 43, 44)]
        infer_values = [number(member["inference_seconds"]) for member in members]
        require(all(value is not None for value in infer_values), f"Missing member inference timing for {dataset}/{model}/{adaptation}")
        infer_sum = sum(infer_values)
        metrics = flatten_segmentation(manifest["metrics"])
        rows.append(base_row(
            task="segmentation", dataset=dataset, model=model, adaptation=adaptation, uq_method="deep_ensemble",
            record_type="ENSEMBLE", status="COMPLETE", contributing_seeds="42;43;44", n_runs=1,
            replication_role="three_member_probability_mean", checkpoint_selection_criterion="N/A — post-hoc probability mean; members use maximum validation mIoU",
            ensemble_members=3, metric_semantics=metric_semantics("segmentation", dataset),
            training_seconds=sum(float(member["training_seconds"]) for member in members),
            training_cost_basis="DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS", inference_seconds=infer_sum,
            inference_cost_basis="DERIVED_SUM_MEMBER_TEST_INFERENCE_SECONDS; AGGREGATION_OVERHEAD_NOT_RECORDED",
            compute_proxy="M=3 member probability evaluations", source_paths=source_join((c3_seg_path, manifest_path)),
            notes="Single ensemble result; never pooled into member mean ± std. Probabilities, not logits, were averaged.", **metrics,
        ))

    require(len(rows) == 136, f"Master row count must be 136, got {len(rows)}")
    return rows, deterministic, used_reports


def fmt(value: Any, digits: int = 4) -> str:
    parsed = number(value)
    return "—" if parsed is None else f"{parsed:.{digits}f}"


def fmt_metric(row: dict[str, Any], metric: str) -> str:
    mean = number(row.get(metric))
    if mean is None:
        return "N/A" if row.get("status") == "N/A" else "—"
    std = number(row.get(f"{metric}_std"))
    return f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f}"


def select(rows: list[dict[str, Any]], **conditions: Any) -> list[dict[str, Any]]:
    return [row for row in rows if all(row.get(key) == value for key, value in conditions.items())]


def registry_role(path: Path, used_reports: set[str], latest_c1: str) -> tuple[str, str, str, str]:
    relative = rel(path)
    name = path.name
    protocol_names = {
        "pre_uq_protocol_freeze.md", "final_training_protocol.md", "global_experiment_conventions.md",
        "final_dataset_protocols.md", "mc_dropout_protocol.md", "mc_dropout_pilot_validation.md",
    }
    historical_names = {
        "code_audit.md", "existing_runs.csv", "experiment_registry.csv", "classification_gap_analysis.md",
        "pipeline_refactor_plan.md", "panopticon_real_model_validation.md", "so2sat_protocol.md",
        "c2_execution_state.md", "c3_treesatai_temperature_scaling.md", "c3_treesatai_temperature_scaling_results.csv",
    }
    if name in protocol_names:
        return "FROZEN_PROTOCOL", "AUTHORITATIVE", "NO", "Controls inclusion, selection, or interpretation"
    if relative in used_reports:
        return "FINAL_NUMERIC_SOURCE", "AUTHORITATIVE", "YES", "Used directly after frozen-protocol filters"
    if name in {"thesis_master_results.csv", "thesis_evidence_matrix.md"}:
        return "MASTER_OUTPUT", "DERIVED_CURRENT", "NO", "Current unified evidence output"
    if name in {"final_thesis_results.md", "classification_results.csv", "method_applicability.csv"}:
        return "DERIVED_SUMMARY", "REQUIRES_OVERRIDE", "NO", "Reusable except stale TreeSatAI TS COMPLETE entries"
    if "final_thesis_tables" in path.parts:
        return "DERIVED_SUMMARY", "SUPPORTING_ONLY", "NO", "Reusable C6 supporting output; no conflicting TreeSatAI TS result"
    if name.startswith("c3_treesatai_temperature_scaling"):
        return "HISTORICAL_DIAGNOSTIC", "HISTORICAL_ONLY", "NO", "Validation-fitted TreeSatAI TS excluded by later freeze"
    if "c1_classification_run_audits" in path.parts:
        if relative.startswith(str(Path(latest_c1).parent)):
            return "FINAL_VALIDATION", "AUTHORITATIVE_SUPPORT", "NO", "Latest 18/18 promotion audit"
        return "AUDIT_SNAPSHOT", "SUPERSEDED", "NO", "Retained chronology; later C1 audit controls"
    if "c2_segmentation_run_audits" in path.parts:
        if "20260824T162250Z" in relative:
            return "FINAL_VALIDATION", "AUTHORITATIVE_SUPPORT", "NO", "Final 24/24 promotion audit"
        return "AUDIT_SNAPSHOT", "SUPERSEDED", "NO", "Retained chronology; final C2 audit controls"
    if "c2_execution_logs" in path.parts:
        return "EXECUTION_LOG", "SUPPORTING_ONLY", "NO", "Operational history; not numeric authority"
    if "dataset_manifests" in path.parts:
        return "DATASET_PROVENANCE", "AUTHORITATIVE_SUPPORT", "NO", "Downloaded benchmark manifest evidence"
    if name in historical_names:
        return "HISTORICAL_OR_OUT_OF_SCOPE", "HISTORICAL_ONLY", "NO", "Retained but excluded from final numeric authority"
    if name in {"panopticon_validation.md", "full_16_cell_preflight.md", "checkpoint_selection_sensitivity.md", "mc_dropout_summary.md", "mc_dropout_robustness.md", "c1_classification_deterministic_completion.md", "c3_temperature_scaling_and_ensembles.md"}:
        return "SUPPORTING_VALIDATION", "AUTHORITATIVE_SUPPORT", "NO", "Supports final artifacts/protocol interpretation"
    return "SUPPORTING_REPORT", "SUPPORTING_ONLY", "NO", "Indexed for future analysis; not used as final numeric authority"


def report_registry(used_reports: set[str]) -> list[dict[str, str]]:
    latest_c1 = rel(sorted((REPORTS / "c1_classification_run_audits").glob("*/c1_run_audit.json"))[-1])
    paths = sorted(REPORTS.rglob("*"))
    files = [
        path for path in paths
        if path.is_file()
        and path not in {FINAL_CSV, FINAL_MD}
        and not any(part.startswith(".thesis_evidence_") for part in path.relative_to(REPORTS).parts)
    ]
    files.extend((FINAL_CSV, FINAL_MD))
    output = []
    for path in files:
        role, authority, numeric, note = registry_role(path, used_reports, latest_c1)
        generated = "current build" if path in {FINAL_CSV, FINAL_MD} else datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        output.append({"path": rel(path), "mtime_utc": generated, "role": role, "authority": authority, "numeric_use": numeric, "note": note})
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in CSV_FIELDS})


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    used_reports: set[str],
    deterministic: dict[tuple[str, str, str, int], dict[str, Any]],
) -> None:
    det_agg = select(rows, uq_method="deterministic", record_type="MEAN_STD")
    lines = [
        "# Thesis Evidence Matrix — A–C Final Experiments",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This document is the unified, time-independent evidence entry point for RQ1–RQ3. Report age does not determine authority: frozen protocol decisions control inclusion; validated per-run artifacts control numbers; later derived summaries are reusable only where they agree with those authorities. No training, model loading, checkpoint inference, or experiment-artifact mutation was performed.",
        "",
        "## Evidence precedence and protocol resolution",
        "",
        "1. `pre_uq_protocol_freeze.md`, `final_training_protocol.md`, global conventions, and final dataset protocols control eligibility and interpretation.",
        "2. Immutable/promoted C1 and C2 runs control deterministic seed-level evidence.",
        "3. C3 controls valid EuroSAT Temperature Scaling and Deep Ensembles; C5 controls MC Dropout.",
        "4. Derived C6 tables are supporting summaries, not higher authority than the frozen protocol.",
        "",
        "**TreeSatAI Temperature Scaling resolution:** the validation-fitted C3 artifacts exist and remain immutable historical/diagnostic evidence, but the later final pre-UQ freeze explicitly excludes them from thesis comparative results because no independent calibration split exists. The four TreeSatAI method cells are therefore `N/A`, not failed or missing. This intentionally overrides the stale `COMPLETE` entries in the C6 classification summary without modifying C3/C6 artifacts.",
        "",
        "Segmentation Temperature Scaling is likewise `N/A` by the frozen protocol. SpaceNet7 classwise-ECE values for ensembles come from validated ensemble manifests because the C3 summary CSV omitted those columns.",
        "",
        "## Master table coverage",
        "",
        "| Record type | Rows | Meaning |",
        "|---|---:|---|",
    ]
    counts = Counter(row["record_type"] for row in rows)
    meanings = {
        "INDIVIDUAL_SEED": "One independently trained deterministic/MC model or one valid EuroSAT temperature applied to that seed",
        "MEAN_STD": "Arithmetic mean ± sample std across the explicitly listed seeds",
        "ENSEMBLE": "One separate three-member probability-mean result; never pooled with member statistics",
        "NOT_APPLICABLE": "Method intentionally excluded by frozen protocol; contains no synthetic metric values",
    }
    for record_type in ("INDIVIDUAL_SEED", "MEAN_STD", "ENSEMBLE", "NOT_APPLICABLE"):
        lines.append(f"| {record_type} | {counts[record_type]} | {meanings[record_type]} |")
    lines.extend([
        "",
        "Machine-readable source: `reports/thesis_master_results.csv`. Blank timing fields mean unavailable/not recorded unless the status explicitly says `N/A`; they are never interpreted as zero.",
        "",
        "## RQ1 — Calibration after downstream adaptation",
        "",
        "### Deterministic classification baselines (three-seed mean ± sample std)",
        "",
        "| Dataset | Model | Adaptation | Accuracy | Macro-F1 | NLL | Brier | ECE-15 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in sorted((value for value in det_agg if value["task"] == "classification"), key=lambda value: (CLASS_DATASETS.index(value["dataset"]), MODELS.index(value["model"]), ADAPTATIONS.index(value["adaptation"]))):
        lines.append(f"| {DISPLAY[row['dataset']]} | {row['model'].upper()} | {row['adaptation']} | {fmt_metric(row,'accuracy')} | {fmt_metric(row,'macro_f1')} | {fmt_metric(row,'nll')} | {fmt_metric(row,'brier')} | {fmt_metric(row,'ece_15')} |")
    lines.extend([
        "",
        "TreeSatAI is multilabel: Accuracy is strict exact match, and NLL/Brier/ECE operate over binary label decisions. Its values must not be treated as multiclass-simplex metrics.",
        "",
        "### Deterministic segmentation baselines (three-seed mean ± sample std)",
        "",
        "| Dataset | Model | Adaptation | mIoU | Pixel accuracy | NLL | Brier | ECE-15 | Per-class IoU |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in sorted((value for value in det_agg if value["task"] == "segmentation"), key=lambda value: (SEG_DATASETS.index(value["dataset"]), MODELS.index(value["model"]), ADAPTATIONS.index(value["adaptation"]))):
        per_class = (f"clear {fmt_metric(row,'iou_clear')}; thick {fmt_metric(row,'iou_thick_cloud')}; thin {fmt_metric(row,'iou_thin_cloud')}; shadow {fmt_metric(row,'iou_cloud_shadow')}" if row["dataset"] == "cloudsen12" else f"background {fmt_metric(row,'iou_background')}; building {fmt_metric(row,'iou_building')}")
        lines.append(f"| {DISPLAY[row['dataset']]} | {row['model'].upper()} | {row['adaptation']} | {fmt_metric(row,'miou')} | {fmt_metric(row,'pixel_accuracy')} | {fmt_metric(row,'nll')} | {fmt_metric(row,'brier')} | {fmt_metric(row,'ece_15')} | {per_class} |")
    lines.extend([
        "",
        "Cross-task raw calibration differences are descriptive only: classification uses minimum validation NLL, segmentation uses maximum validation mIoU, and their output units/label semantics differ.",
        "",
        "## RQ2 — Frozen versus full fine-tuning",
        "",
        "Deltas below are full fine-tuning minus frozen adaptation using deterministic three-seed means. Positive performance delta is favorable; negative NLL/Brier/ECE delta is favorable.",
        "",
        "| Task | Dataset | Model | Δ performance | Δ NLL | Δ Brier | Δ ECE-15 |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for task, datasets, performance in (("classification", CLASS_DATASETS, "accuracy"), ("segmentation", SEG_DATASETS, "miou")):
        for dataset in datasets:
            for model in MODELS:
                frozen = next(row for row in det_agg if (row["task"], row["dataset"], row["model"], row["adaptation"]) == (task, dataset, model, "frozen"))
                full = next(row for row in det_agg if (row["task"], row["dataset"], row["model"], row["adaptation"]) == (task, dataset, model, "full_finetune"))
                lines.append(f"| {task} | {DISPLAY[dataset]} | {model.upper()} | {float(full[performance])-float(frozen[performance]):+.4f} | {float(full['nll'])-float(frozen['nll']):+.4f} | {float(full['brier'])-float(frozen['brier']):+.4f} | {float(full['ece_15'])-float(frozen['ece_15']):+.4f} |")

    lines.extend([
        "",
        "These are descriptive paired-cell contrasts, not causal estimates. Architecture-specific preprocessing history and validation-selected checkpoint behavior remain relevant caveats.",
        "",
        "## RQ3 — UQ/calibration methods, predictive performance, and cost",
        "",
        "Comparison basis: valid Temperature Scaling uses three-seed means against deterministic three-seed means; replicated MC cells use their predeclared three-seed summaries; other MC cells use seed-42 paired comparisons; ensembles remain single M=3 results compared descriptively with deterministic member means.",
        "",
        "| Task | Dataset | Model | Adaptation | Method | Basis | Performance | Δ performance | Δ NLL | Δ Brier | Δ ECE-15 | Training cost | Inference proxy |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ])
    methods = ("temperature_scaling", "mc_dropout", "deep_ensemble")
    for task, datasets, performance in (("classification", CLASS_DATASETS, "accuracy"), ("segmentation", SEG_DATASETS, "miou")):
        for dataset in datasets:
            for model in MODELS:
                for adaptation in ADAPTATIONS:
                    det_mean = next(row for row in det_agg if (row["task"], row["dataset"], row["model"], row["adaptation"]) == (task, dataset, model, adaptation))
                    for method in methods:
                        candidates = select(rows, task=task, dataset=dataset, model=model, adaptation=adaptation, uq_method=method)
                        na = next((row for row in candidates if row["record_type"] == "NOT_APPLICABLE"), None)
                        if na is not None:
                            lines.append(f"| {task} | {DISPLAY[dataset]} | {model.upper()} | {adaptation} | {method} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
                            continue
                        if method == "mc_dropout":
                            summary = next((row for row in candidates if row["record_type"] == "MEAN_STD"), None)
                            if summary is not None:
                                method_row, comparator, basis = summary, det_mean, "3-seed robustness subset"
                            else:
                                method_row = next(row for row in candidates if row["record_type"] == "INDIVIDUAL_SEED" and row["seed"] == 42)
                                comparator = deterministic[(dataset, model, adaptation, 42)]
                                basis = "paired seed42"
                        elif method == "temperature_scaling":
                            method_row = next(row for row in candidates if row["record_type"] == "MEAN_STD")
                            comparator, basis = det_mean, "3-seed mean"
                        else:
                            method_row = next(row for row in candidates if row["record_type"] == "ENSEMBLE")
                            comparator, basis = det_mean, "single M=3 ensemble vs member mean"
                        train = f"{float(method_row['training_seconds']):.1f}s ({method_row['training_cost_basis']})" if number(method_row["training_seconds"]) is not None else method_row["training_cost_basis"]
                        lines.append(f"| {task} | {DISPLAY[dataset]} | {model.upper()} | {adaptation} | {method} | {basis} | {fmt(method_row[performance])} | {float(method_row[performance])-float(comparator[performance]):+.4f} | {float(method_row['nll'])-float(comparator['nll']):+.4f} | {float(method_row['brier'])-float(comparator['brier']):+.4f} | {float(method_row['ece_15'])-float(comparator['ece_15']):+.4f} | {train} | {method_row['compute_proxy'] or method_row['inference_cost_basis']} |")

    lines.extend([
        "",
        "No method is chosen or rejected from test performance. MC Dropout is downstream-head/decoder dropout with T=30; Deep Ensemble averages probabilities across M=3 independently trained seeds. Expected predictive entropy and MI-style disagreement are descriptive quantities/proxies, not literal aleatoric/epistemic truth.",
        "",
        "### Cost evidence availability",
        "",
        "| Method/result | Training timing | Inference timing | Valid compute proxy |",
        "|---|---|---|---|",
        "| Deterministic classification | 24/24 measured cumulative training seconds | 24/24 measured single-pass test seconds | 1 pass |",
        "| Deterministic segmentation | 24/24 measured cumulative training seconds | 24/24 measured single-pass test seconds | 1 pass |",
        "| EuroSAT Temperature Scaling | Reuses deterministic checkpoint; scalar-fit time not recorded | Apply time not recorded | Optimizer iterations retained per seed |",
        "| MC Dropout | 24/24 measured cumulative MC-training seconds | 0/24; not recorded | T=30 stochastic passes |",
        "| Deep Ensemble | Derived sum of three member-training seconds | Derived serial member-sum available for all 16 ensembles; aggregation overhead not recorded | M=3 members |",
        "",
        "`training_seconds` never substitutes process-local `runtime_seconds`, which can be misleading for resumed runs. Missing timings are `UNAVAILABLE_NOT_RECORDED`, never zero.",
        "",
        "## SpaceNet7 required calibration evidence",
        "",
        "The CSV retains overall ECE, foreground/building ECE, one-vs-rest background/building ECE, foreground NLL/Brier, boundary ECE, and boundary foreground ECE. Boundary-free per-image quantities remain undefined in their source archives and are never converted to zero. Low building IoU is a validated result, not an exclusion criterion.",
        "",
        "## Known evidence inconsistencies and preserved caveats",
        "",
        "- TreeSatAI validation-fitted TS artifacts/report: historical diagnostic only; final comparative entry is N/A under the later freeze.",
        "- C6 `final_thesis_results.md` and classification table: numerically reusable except TreeSatAI TS `COMPLETE` entries, which this matrix overrides.",
        "- C3 segmentation summary CSV omitted SpaceNet7 classwise ECE; validated ensemble manifests supply the already-computed values.",
        "- Earlier C1/C2 audit snapshots and execution logs remain indexed but are superseded by the final 18/18 and 24/24 promotion audits.",
        "- TreeSatAI uses the validated static T=1 artifact and is multilabel; it must not be described as genuinely multi-temporal here.",
        "- CloudSEN12 DOFA full-finetune seed44 has a one-pixel float32 CPU/GPU tie warning but remains promoted.",
        "- EuroSAT retains a backend-specific preprocessing-history caveat; cross-model comparisons must not conceal it.",
        "",
        "## Unified report registry",
        "",
        "Every current file under `reports/` is indexed below regardless of generation time. `mtime_utc` is discovery metadata only and never determines precedence. `numeric_use=YES` means the file directly contributed values after protocol filtering; all other reports remain available for provenance, validation, caveats, or historical analysis.",
        "",
        "| Path | mtime UTC | Registry role | Authority | Numeric use | Inclusion note |",
        "|---|---|---|---|---|---|",
    ])
    for item in report_registry(used_reports):
        lines.append(f"| `{item['path']}` | {item['mtime_utc']} | {item['role']} | {item['authority']} | {item['numeric_use']} | {item['note']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["record_type"] for row in rows)
    require(counts == {"INDIVIDUAL_SEED": 84, "MEAN_STD": 24, "ENSEMBLE": 16, "NOT_APPLICABLE": 12}, f"Unexpected record counts {counts}")
    require(not [row for row in rows if row["dataset"] == "treesatai" and row["uq_method"] == "temperature_scaling" and row["status"] != "N/A"], "Quantitative TreeSatAI TS leaked into final table")
    require(len(select(rows, task="segmentation", uq_method="temperature_scaling", status="N/A")) == 8, "Segmentation TS N/A coverage !=8")
    require(len(select(rows, uq_method="deep_ensemble", record_type="ENSEMBLE")) == 16, "Ensemble coverage !=16")
    require(not [row for row in rows if row["record_type"] == "ENSEMBLE" and any(number(row[f"{metric}_std"]) is not None for metric in METRICS)], "Ensemble std must be blank")
    require(len(select(rows, uq_method="mc_dropout", record_type="INDIVIDUAL_SEED")) == 24, "MC seed rows !=24")
    require(len(select(rows, uq_method="mc_dropout", record_type="MEAN_STD")) == 4, "MC robustness summaries !=4")
    for row in rows:
        for source in row["source_paths"].split(";") if row["source_paths"] else ():
            require((ROOT / source).exists(), f"Missing evidence source {source}")
        if row["record_type"] == "NOT_APPLICABLE":
            require(all(number(row[metric]) is None for metric in METRICS), "N/A row contains metrics")
    deterministic = select(rows, uq_method="deterministic", record_type="INDIVIDUAL_SEED")
    require(len(deterministic) == 48 and all(number(row["training_seconds"]) is not None for row in deterministic), "Deterministic training-cost coverage invalid")
    require(sum(number(row["inference_seconds"]) is not None for row in deterministic) == 48, "Deterministic inference timing coverage invalid")
    require(all(number(row["inference_seconds"]) is not None for row in select(rows, uq_method="deep_ensemble", record_type="ENSEMBLE")), "Ensemble member-sum inference timing coverage invalid")


def main() -> None:
    require(not FINAL_CSV.exists(), f"Refusing to overwrite {FINAL_CSV}")
    require(not FINAL_MD.exists(), f"Refusing to overwrite {FINAL_MD}")
    rows, deterministic, used_reports = build_rows()
    validate(rows)
    staging = Path(tempfile.mkdtemp(prefix=".thesis_evidence_", dir=REPORTS))
    temp_csv, temp_md = staging / FINAL_CSV.name, staging / FINAL_MD.name
    try:
        ordered = sorted(rows, key=lambda row: (
            ("classification", "segmentation").index(row["task"]),
            (*CLASS_DATASETS, *SEG_DATASETS).index(row["dataset"]), MODELS.index(row["model"]),
            ADAPTATIONS.index(row["adaptation"]), ("deterministic", "temperature_scaling", "mc_dropout", "deep_ensemble").index(row["uq_method"]),
            ("INDIVIDUAL_SEED", "MEAN_STD", "ENSEMBLE", "NOT_APPLICABLE").index(row["record_type"]),
            -1 if row["seed"] is None else int(row["seed"]),
        ))
        write_csv(temp_csv, ordered)
        # Registry sees final paths conceptually even while writes remain staged.
        write_markdown(temp_md, ordered, used_reports, deterministic)
        os.replace(temp_csv, FINAL_CSV)
        os.replace(temp_md, FINAL_MD)
    finally:
        if staging.exists():
            import shutil
            shutil.rmtree(staging)
    print(json.dumps({"csv": str(FINAL_CSV), "markdown": str(FINAL_MD), "rows": len(rows), "record_types": Counter(row["record_type"] for row in rows)}, default=dict, indent=2))


if __name__ == "__main__":
    main()
