#!/usr/bin/env python3
"""Build paired RQ2/RQ3 effect tables from the frozen thesis master CSV.

This script performs arithmetic on validated summary metrics only. It never loads a
model or recomputes predictions.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "reports/thesis_master_results.csv"
RQ2_OUT = ROOT / "reports/rq2_adaptation_effects.csv"
RQ3_OUT = ROOT / "reports/rq3_uq_effects.csv"
SUMMARY_OUT = ROOT / "reports/rq_effect_summary.md"

DATASET_ORDER = {"eurosat": 0, "treesatai": 1, "cloudsen12": 2, "spacenet7": 3}
MODEL_ORDER = {"dofa": 0, "panopticon": 1}
ADAPTATION_ORDER = {"frozen": 0, "full_finetune": 1}
METHOD_ORDER = {"temperature_scaling": 0, "mc_dropout": 1, "deep_ensemble": 2}
DISPLAY = {
    "eurosat": "EuroSAT",
    "treesatai": "TreeSatAI",
    "cloudsen12": "CloudSEN12",
    "spacenet7": "SpaceNet7",
}

# The requested paired effects. Accuracy/Macro-F1 apply to classification and
# mIoU to segmentation; inapplicable cells remain blank rather than zero.
METRICS = ("accuracy", "macro_f1", "miou", "nll", "brier", "ece_15")

IDENTITY_FIELDS = (
    "rq", "task", "dataset", "model", "adaptation", "baseline_adaptation",
    "comparison_adaptation", "baseline_method", "comparison_method",
    "comparison_scope", "record_type", "status", "seed", "contributing_seeds",
    "n_pairs", "n_members", "replication_role", "checkpoint_selection_criterion",
    "metric_semantics", "primary_performance_metric", "delta_definition",
    "applicability_reason",
)
TASK_PERFORMANCE_FIELDS = (
    "baseline_task_performance", "baseline_task_performance_std",
    "comparison_task_performance", "comparison_task_performance_std",
    "delta_task_performance", "delta_task_performance_std",
)
METRIC_FIELDS = tuple(
    field
    for metric in METRICS
    for field in (
        f"baseline_{metric}", f"baseline_{metric}_std",
        f"comparison_{metric}", f"comparison_{metric}_std",
        f"delta_{metric}", f"delta_{metric}_std",
    )
)
PROVENANCE_FIELDS = (
    "baseline_record_type", "comparison_record_type", "baseline_run_ids",
    "comparison_run_ids", "baseline_checkpoint_paths", "comparison_checkpoint_paths",
    "baseline_checkpoint_sha256s", "comparison_checkpoint_sha256s",
    "baseline_source_paths", "comparison_source_paths", "pairing_note",
)
FIELDS = (*IDENTITY_FIELDS, *TASK_PERFORMANCE_FIELDS, *METRIC_FIELDS, *PROVENANCE_FIELDS)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_master() -> list[dict[str, str]]:
    with MASTER.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 136, f"Unexpected master row count: {len(rows)}")
    return rows


def number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def seed_number(row: dict[str, str]) -> int | None:
    value = number(row.get("seed"))
    return None if value is None else int(value)


def primary_metric(task: str) -> str:
    return "accuracy" if task == "classification" else "miou"


def empty_effect(**values: Any) -> dict[str, Any]:
    row = {field: None for field in FIELDS}
    row.update(values)
    return row


def unique_join(values: Iterable[str], *, split_sources: bool = False) -> str:
    output: list[str] = []
    for raw in values:
        candidates = raw.split(";") if split_sources and raw else [raw]
        for value in candidates:
            if value and value not in output:
                output.append(value)
    return ";".join(output)


def provenance(rows: list[dict[str, str]], prefix: str) -> dict[str, str]:
    return {
        f"{prefix}_record_type": unique_join(row["record_type"] for row in rows),
        f"{prefix}_run_ids": unique_join(row["run_id"] for row in rows),
        f"{prefix}_checkpoint_paths": unique_join(row["checkpoint_path"] for row in rows),
        f"{prefix}_checkpoint_sha256s": unique_join(row["checkpoint_sha256"] for row in rows),
        f"{prefix}_source_paths": unique_join((row["source_paths"] for row in rows), split_sources=True),
    }


def set_task_performance(row: dict[str, Any]) -> None:
    metric = row["primary_performance_metric"]
    for stem in ("baseline", "comparison", "delta"):
        row[f"{stem}_task_performance"] = row.get(f"{stem}_{metric}")
        row[f"{stem}_task_performance_std"] = row.get(f"{stem}_{metric}_std")


def seed_effect(
    baseline: dict[str, str],
    comparison: dict[str, str],
    **identity: Any,
) -> dict[str, Any]:
    row = empty_effect(**identity)
    for metric in METRICS:
        before = number(baseline.get(metric))
        after = number(comparison.get(metric))
        require((before is None) == (after is None), f"Metric applicability mismatch: {metric}")
        row[f"baseline_{metric}"] = before
        row[f"comparison_{metric}"] = after
        row[f"delta_{metric}"] = None if before is None else after - before
    row.update(provenance([baseline], "baseline"))
    row.update(provenance([comparison], "comparison"))
    set_task_performance(row)
    return row


def mean_std(values: list[float]) -> tuple[float, float]:
    require(len(values) >= 2, "At least two values are required for sample std")
    return statistics.fmean(values), statistics.stdev(values)


def assert_close(actual: float | None, expected: float | None, label: str) -> None:
    require((actual is None) == (expected is None), f"Missing-value mismatch for {label}")
    if actual is not None:
        require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), f"Value mismatch for {label}: {actual} vs {expected}")


def paired_summary(
    seed_rows: list[dict[str, Any]],
    baseline_master_rows: list[dict[str, str]],
    comparison_master_rows: list[dict[str, str]],
    **identity: Any,
) -> dict[str, Any]:
    require(len(seed_rows) == len(baseline_master_rows) == len(comparison_master_rows) == 3, "Paired summary must contain three seeds")
    row = empty_effect(**identity)
    for metric in METRICS:
        baseline_values = [number(item[f"baseline_{metric}"]) for item in seed_rows]
        comparison_values = [number(item[f"comparison_{metric}"]) for item in seed_rows]
        delta_values = [number(item[f"delta_{metric}"]) for item in seed_rows]
        if all(value is None for value in baseline_values):
            require(all(value is None for value in comparison_values + delta_values), f"Partial applicability for {metric}")
            continue
        require(all(value is not None for value in baseline_values + comparison_values + delta_values), f"Partial paired values for {metric}")
        for stem, values in (
            ("baseline", baseline_values),
            ("comparison", comparison_values),
            ("delta", delta_values),
        ):
            mean, std = mean_std(values)  # type: ignore[arg-type]
            row[f"{stem}_{metric}"] = mean
            row[f"{stem}_{metric}_std"] = std
    row.update(provenance(baseline_master_rows, "baseline"))
    row.update(provenance(comparison_master_rows, "comparison"))
    set_task_performance(row)
    return row


def compatible_identity(left: dict[str, str], right: dict[str, str], fields: Iterable[str]) -> None:
    for field in fields:
        require(left[field] == right[field], f"Identity mismatch for {field}: {left[field]} vs {right[field]}")


def master_index(rows: list[dict[str, str]], *, method: str, record_type: str) -> dict[tuple[Any, ...], dict[str, str]]:
    selected = [row for row in rows if row["uq_method"] == method and row["record_type"] == record_type and row["status"] == "COMPLETE"]
    index: dict[tuple[Any, ...], dict[str, str]] = {}
    for row in selected:
        key = (row["task"], row["dataset"], row["model"], row["adaptation"], seed_number(row))
        require(key not in index, f"Duplicate master key: {key}")
        index[key] = row
    return index


def build_rq2(master: list[dict[str, str]]) -> list[dict[str, Any]]:
    deterministic = master_index(master, method="deterministic", record_type="INDIVIDUAL_SEED")
    groups: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, str], dict[str, str]]]] = defaultdict(list)
    output: list[dict[str, Any]] = []
    for task, dataset, model in sorted(
        {(key[0], key[1], key[2]) for key in deterministic},
        key=lambda key: (DATASET_ORDER[key[1]], MODEL_ORDER[key[2]]),
    ):
        for seed in (42, 43, 44):
            frozen = deterministic[(task, dataset, model, "frozen", seed)]
            full = deterministic[(task, dataset, model, "full_finetune", seed)]
            compatible_identity(frozen, full, ("task", "dataset", "model", "metric_semantics", "checkpoint_selection_criterion"))
            effect = seed_effect(
                frozen,
                full,
                rq="RQ2",
                task=task,
                dataset=dataset,
                model=model,
                baseline_adaptation="frozen",
                comparison_adaptation="full_finetune",
                baseline_method="deterministic",
                comparison_method="deterministic",
                comparison_scope="MATCHED_TRAINING_SEED",
                record_type="PAIRED_SEED",
                status="COMPLETE",
                seed=seed,
                contributing_seeds=str(seed),
                n_pairs=1,
                replication_role="deterministic_seed_pair",
                checkpoint_selection_criterion=frozen["checkpoint_selection_criterion"],
                metric_semantics=frozen["metric_semantics"],
                primary_performance_metric=primary_metric(task),
                delta_definition="full_finetune_minus_frozen",
                pairing_note="Independently trained adaptations paired by the same declared training seed; descriptive contrast.",
            )
            output.append(effect)
            groups[(task, dataset, model)].append((effect, frozen, full))

    require(len(output) == 24, f"RQ2 seed-pair count !=24: {len(output)}")
    for (task, dataset, model), members in groups.items():
        members.sort(key=lambda item: int(item[0]["seed"]))
        effects = [item[0] for item in members]
        frozen_rows = [item[1] for item in members]
        full_rows = [item[2] for item in members]
        output.append(paired_summary(
            effects,
            frozen_rows,
            full_rows,
            rq="RQ2",
            task=task,
            dataset=dataset,
            model=model,
            baseline_adaptation="frozen",
            comparison_adaptation="full_finetune",
            baseline_method="deterministic",
            comparison_method="deterministic",
            comparison_scope="THREE_MATCHED_TRAINING_SEEDS",
            record_type="PAIRED_MEAN_STD",
            status="COMPLETE",
            contributing_seeds="42;43;44",
            n_pairs=3,
            replication_role="paired_three_seed_summary",
            checkpoint_selection_criterion=frozen_rows[0]["checkpoint_selection_criterion"],
            metric_semantics=frozen_rows[0]["metric_semantics"],
            primary_performance_metric=primary_metric(task),
            delta_definition="mean_of_seedwise_full_finetune_minus_frozen",
            pairing_note="Mean and sample std (ddof=1) of three seedwise differences; effect std is not derived from arm stds.",
        ))
    require(Counter(row["record_type"] for row in output) == {"PAIRED_SEED": 24, "PAIRED_MEAN_STD": 8}, "Invalid RQ2 output structure")
    return sorted(output, key=lambda row: (
        DATASET_ORDER[row["dataset"]], MODEL_ORDER[row["model"]],
        0 if row["record_type"] == "PAIRED_SEED" else 1,
        -1 if row["seed"] is None else int(row["seed"]),
    ))


def build_rq3(master: list[dict[str, str]]) -> list[dict[str, Any]]:
    deterministic_seed = master_index(master, method="deterministic", record_type="INDIVIDUAL_SEED")
    deterministic_mean = master_index(master, method="deterministic", record_type="MEAN_STD")
    output: list[dict[str, Any]] = []
    seed_effects: dict[tuple[str, str, str, str, str, int], tuple[dict[str, Any], dict[str, str], dict[str, str]]] = {}

    # Seed-paired Temperature Scaling and MC Dropout effects.
    uq_seed_rows = [
        row for row in master
        if row["status"] == "COMPLETE"
        and row["record_type"] == "INDIVIDUAL_SEED"
        and row["uq_method"] in {"temperature_scaling", "mc_dropout"}
    ]
    for target in uq_seed_rows:
        seed = seed_number(target)
        require(seed is not None, "Valid seed-level UQ row has no seed")
        key = (target["task"], target["dataset"], target["model"], target["adaptation"], seed)
        baseline = deterministic_seed[key]
        compatible_identity(baseline, target, ("task", "dataset", "model", "adaptation", "metric_semantics"))
        if target["uq_method"] == "temperature_scaling":
            require(target["run_id"] == baseline["run_id"], "Temperature Scaling run ID mismatch")
            require(target["checkpoint_sha256"] == baseline["checkpoint_sha256"], "Temperature Scaling checkpoint mismatch")
            scope = "SAME_CHECKPOINT_POSTHOC_SEED_PAIR"
            note = "Same deterministic checkpoint and seed; scalar temperature changes probabilities only."
        else:
            scope = "MATCHED_TRAINING_SEED_DESIGN_PAIR"
            note = "Independently trained MC-Dropout and deterministic models paired by declared seed; not a same-weight transformation."
        effect = seed_effect(
            baseline,
            target,
            rq="RQ3",
            task=target["task"],
            dataset=target["dataset"],
            model=target["model"],
            adaptation=target["adaptation"],
            baseline_adaptation=target["adaptation"],
            comparison_adaptation=target["adaptation"],
            baseline_method="deterministic",
            comparison_method=target["uq_method"],
            comparison_scope=scope,
            record_type="PAIRED_SEED",
            status="COMPLETE",
            seed=seed,
            contributing_seeds=str(seed),
            n_pairs=1,
            replication_role=target["replication_role"],
            checkpoint_selection_criterion=target["checkpoint_selection_criterion"],
            metric_semantics=target["metric_semantics"],
            primary_performance_metric=primary_metric(target["task"]),
            delta_definition=f"{target['uq_method']}_minus_deterministic_same_seed",
            pairing_note=note,
        )
        output.append(effect)
        effect_key = (*key[:4], target["uq_method"], seed)
        require(effect_key not in seed_effects, f"Duplicate RQ3 seed effect {effect_key}")
        seed_effects[effect_key] = (effect, baseline, target)

    require(len(seed_effects) == 36, f"RQ3 seed pairs !=36: {len(seed_effects)}")

    # Predeclared three-seed summaries. Deltas and their std are recomputed from
    # paired seed effects, not by subtracting aggregate standard deviations.
    uq_mean_rows = [
        row for row in master
        if row["status"] == "COMPLETE"
        and row["record_type"] == "MEAN_STD"
        and row["uq_method"] in {"temperature_scaling", "mc_dropout"}
    ]
    for target_mean in uq_mean_rows:
        cell = (target_mean["task"], target_mean["dataset"], target_mean["model"], target_mean["adaptation"])
        triples = [seed_effects[(*cell, target_mean["uq_method"], seed)] for seed in (42, 43, 44)]
        effects = [item[0] for item in triples]
        baselines = [item[1] for item in triples]
        targets = [item[2] for item in triples]
        baseline_mean = deterministic_mean[(*cell, None)]
        # Verify that master aggregate rows equal the same three seed sets.
        for metric in METRICS:
            baseline_values = [number(item.get(metric)) for item in baselines]
            target_values = [number(item.get(metric)) for item in targets]
            if all(value is None for value in baseline_values):
                continue
            bmean, bstd = mean_std(baseline_values)  # type: ignore[arg-type]
            tmean, tstd = mean_std(target_values)  # type: ignore[arg-type]
            assert_close(number(baseline_mean.get(metric)), bmean, f"deterministic aggregate {cell}/{metric}")
            assert_close(number(baseline_mean.get(f"{metric}_std")), bstd, f"deterministic aggregate std {cell}/{metric}")
            assert_close(number(target_mean.get(metric)), tmean, f"UQ aggregate {cell}/{target_mean['uq_method']}/{metric}")
            assert_close(number(target_mean.get(f"{metric}_std")), tstd, f"UQ aggregate std {cell}/{target_mean['uq_method']}/{metric}")
        output.append(paired_summary(
            effects,
            baselines,
            targets,
            rq="RQ3",
            task=target_mean["task"],
            dataset=target_mean["dataset"],
            model=target_mean["model"],
            adaptation=target_mean["adaptation"],
            baseline_adaptation=target_mean["adaptation"],
            comparison_adaptation=target_mean["adaptation"],
            baseline_method="deterministic",
            comparison_method=target_mean["uq_method"],
            comparison_scope="THREE_PAIRED_SEEDS",
            record_type="PAIRED_MEAN_STD",
            status="COMPLETE",
            contributing_seeds="42;43;44",
            n_pairs=3,
            replication_role=target_mean["replication_role"],
            checkpoint_selection_criterion=target_mean["checkpoint_selection_criterion"],
            metric_semantics=target_mean["metric_semantics"],
            primary_performance_metric=primary_metric(target_mean["task"]),
            delta_definition=f"mean_of_seedwise_{target_mean['uq_method']}_minus_deterministic",
            pairing_note="Mean and sample std (ddof=1) of three seedwise effects; effect std is not derived from arm stds.",
        ))

    require(len(uq_mean_rows) == 8, f"RQ3 paired summaries !=8: {len(uq_mean_rows)}")

    # A Deep Ensemble is one aggregate result and is compared only with the mean
    # of its exact deterministic members. It is never treated as another seed.
    ensemble_rows = [row for row in master if row["uq_method"] == "deep_ensemble" and row["record_type"] == "ENSEMBLE" and row["status"] == "COMPLETE"]
    for target in ensemble_rows:
        cell = (target["task"], target["dataset"], target["model"], target["adaptation"])
        baseline = deterministic_mean[(*cell, None)]
        require(target["contributing_seeds"] == baseline["contributing_seeds"] == "42;43;44", f"Ensemble member-seed mismatch: {cell}")
        member_rows = [deterministic_seed[(*cell, seed)] for seed in (42, 43, 44)]
        row = empty_effect(
            rq="RQ3",
            task=target["task"],
            dataset=target["dataset"],
            model=target["model"],
            adaptation=target["adaptation"],
            baseline_adaptation=target["adaptation"],
            comparison_adaptation=target["adaptation"],
            baseline_method="deterministic",
            comparison_method="deep_ensemble",
            comparison_scope="ENSEMBLE_VS_DETERMINISTIC_MEMBER_MEAN",
            record_type="AGGREGATED_COMPARISON",
            status="COMPLETE",
            contributing_seeds="42;43;44",
            n_members=3,
            replication_role=target["replication_role"],
            checkpoint_selection_criterion=target["checkpoint_selection_criterion"],
            metric_semantics=target["metric_semantics"],
            primary_performance_metric=primary_metric(target["task"]),
            delta_definition="deep_ensemble_minus_mean_of_its_three_deterministic_members",
            pairing_note="Single M=3 probability-mean ensemble versus its member-model mean; not a seed-paired estimate and no effect std is assigned.",
        )
        for metric in METRICS:
            before = number(baseline.get(metric))
            after = number(target.get(metric))
            require((before is None) == (after is None), f"Ensemble metric applicability mismatch: {cell}/{metric}")
            row[f"baseline_{metric}"] = before
            row[f"baseline_{metric}_std"] = number(baseline.get(f"{metric}_std"))
            row[f"comparison_{metric}"] = after
            row[f"delta_{metric}"] = None if before is None else after - before
        row.update(provenance(member_rows, "baseline"))
        row.update(provenance([target], "comparison"))
        set_task_performance(row)
        output.append(row)

    require(len(ensemble_rows) == 16, f"RQ3 ensemble count !=16: {len(ensemble_rows)}")

    # Preserve explicit method applicability decisions without inventing effects.
    na_rows = [row for row in master if row["uq_method"] != "deterministic" and row["record_type"] == "NOT_APPLICABLE"]
    for source in na_rows:
        row = empty_effect(
            rq="RQ3",
            task=source["task"],
            dataset=source["dataset"],
            model=source["model"],
            adaptation=source["adaptation"],
            baseline_adaptation=source["adaptation"],
            comparison_adaptation=source["adaptation"],
            baseline_method="deterministic",
            comparison_method=source["uq_method"],
            comparison_scope="NOT_APPLICABLE",
            record_type="NOT_APPLICABLE",
            status="N/A",
            n_pairs=0,
            replication_role=source["replication_role"],
            checkpoint_selection_criterion=source["checkpoint_selection_criterion"],
            metric_semantics=source["metric_semantics"],
            primary_performance_metric=primary_metric(source["task"]),
            delta_definition=f"{source['uq_method']}_minus_deterministic",
            applicability_reason=source["applicability_reason"],
            comparison_record_type="NOT_APPLICABLE",
            comparison_source_paths=source["source_paths"],
            pairing_note="No effect calculated because the frozen final protocol marks this method/cell N/A.",
        )
        output.append(row)
    require(len(na_rows) == 12, f"RQ3 N/A count !=12: {len(na_rows)}")

    expected = {"PAIRED_SEED": 36, "PAIRED_MEAN_STD": 8, "AGGREGATED_COMPARISON": 16, "NOT_APPLICABLE": 12}
    require(Counter(row["record_type"] for row in output) == expected, f"Invalid RQ3 output structure: {Counter(row['record_type'] for row in output)}")
    return sorted(output, key=lambda row: (
        DATASET_ORDER[row["dataset"]], MODEL_ORDER[row["model"]],
        ADAPTATION_ORDER[row["adaptation"]], METHOD_ORDER[row["comparison_method"]],
        {"PAIRED_SEED": 0, "PAIRED_MEAN_STD": 1, "AGGREGATED_COMPARISON": 2, "NOT_APPLICABLE": 3}[row["record_type"]],
        -1 if row["seed"] is None else int(row["seed"]),
    ))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in FIELDS})


def fmt_effect(row: dict[str, Any], metric: str) -> str:
    value = number(row.get(f"delta_{metric}"))
    if value is None:
        return "N/A" if row.get("status") == "N/A" else "—"
    std = number(row.get(f"delta_{metric}_std"))
    return f"{value:+.4f} ± {std:.4f}" if std is not None else f"{value:+.4f}"


def write_summary(path: Path, rq2: list[dict[str, Any]], rq3: list[dict[str, Any]]) -> None:
    rq2_aggregate = [row for row in rq2 if row["record_type"] == "PAIRED_MEAN_STD"]
    rq3_numeric = [row for row in rq3 if row["status"] == "COMPLETE"]
    rq3_na = [row for row in rq3 if row["status"] == "N/A"]

    # Compact RQ3 view: show three-seed summaries where predeclared; otherwise
    # the primary seed42 MC result. TS always has a three-seed summary; ensembles
    # are already single aggregate results.
    rq3_display: list[dict[str, Any]] = []
    for method in ("temperature_scaling", "mc_dropout", "deep_ensemble"):
        candidates = [row for row in rq3_numeric if row["comparison_method"] == method]
        cells = sorted(
            {(row["task"], row["dataset"], row["model"], row["adaptation"]) for row in candidates},
            key=lambda key: (DATASET_ORDER[key[1]], MODEL_ORDER[key[2]], ADAPTATION_ORDER[key[3]]),
        )
        for cell in cells:
            cell_rows = [row for row in candidates if (row["task"], row["dataset"], row["model"], row["adaptation"]) == cell]
            aggregates = [row for row in cell_rows if row["record_type"] in {"PAIRED_MEAN_STD", "AGGREGATED_COMPARISON"}]
            if aggregates:
                require(len(aggregates) == 1, f"Ambiguous RQ3 aggregate display row: {method}/{cell}")
                rq3_display.append(aggregates[0])
            else:
                primary = [row for row in cell_rows if row["record_type"] == "PAIRED_SEED" and int(row["seed"]) == 42]
                require(len(primary) == 1, f"Missing primary seed42 RQ3 row: {method}/{cell}")
                rq3_display.append(primary[0])

    lines = [
        "# Paired RQ2 and RQ3 Effect Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Inputs are limited to `reports/thesis_master_results.csv` and the validated prediction/manifests referenced by that master table. Prediction artifacts were used to confirm pairing compatibility; the reported effects are arithmetic differences between already-validated metrics. No model, training loop, or inference pipeline was executed.",
        "",
        "All deltas are absolute differences on the recorded 0–1 metric scale: comparison minus baseline. Positive Accuracy, Macro-F1, or mIoU deltas favor the comparison; negative NLL, Brier, or ECE-15 deltas favor the comparison. These signs are descriptive and are not significance tests.",
        "",
        "## RQ2 — Full fine-tuning versus frozen adaptation",
        "",
        "RQ2 contains 24 independently trained matched-seed contrasts and eight paired three-seed summaries. Aggregate effect standard deviations are sample standard deviations of the three seedwise deltas—not differences or combinations of the two arm standard deviations.",
        "",
        "| Task | Dataset | Model | Seeds | Δ Accuracy | Δ Macro-F1 | Δ mIoU | Δ NLL | Δ Brier | Δ ECE-15 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rq2_aggregate:
        lines.append(
            f"| {row['task']} | {DISPLAY[row['dataset']]} | {row['model'].upper()} | {row['contributing_seeds']} | "
            f"{fmt_effect(row, 'accuracy')} | {fmt_effect(row, 'macro_f1')} | {fmt_effect(row, 'miou')} | "
            f"{fmt_effect(row, 'nll')} | {fmt_effect(row, 'brier')} | {fmt_effect(row, 'ece_15')} |"
        )

    lines.extend([
        "",
        "## RQ3 — UQ method versus corresponding deterministic baseline",
        "",
        "The CSV contains 36 seed-level comparisons, eight paired three-seed summaries, 16 explicitly aggregate Ensemble-versus-member-mean comparisons, and 12 N/A declarations. The compact table below uses a three-seed paired summary where one was predeclared; otherwise MC Dropout is the primary seed42 pair. Deep Ensembles remain single aggregate results and are never pooled with member seeds.",
        "",
        "| Task | Dataset | Model | Adaptation | UQ method | Comparison basis | Δ Accuracy | Δ Macro-F1 | Δ mIoU | Δ NLL | Δ Brier | Δ ECE-15 |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rq3_display:
        if row["record_type"] == "PAIRED_MEAN_STD":
            basis = "paired seeds 42/43/44, mean ± sample std"
        elif row["record_type"] == "AGGREGATED_COMPARISON":
            basis = "M=3 ensemble vs exact member mean"
        else:
            basis = f"paired seed {row['seed']}"
        lines.append(
            f"| {row['task']} | {DISPLAY[row['dataset']]} | {row['model'].upper()} | {row['adaptation']} | {row['comparison_method']} | {basis} | "
            f"{fmt_effect(row, 'accuracy')} | {fmt_effect(row, 'macro_f1')} | {fmt_effect(row, 'miou')} | "
            f"{fmt_effect(row, 'nll')} | {fmt_effect(row, 'brier')} | {fmt_effect(row, 'ece_15')} |"
        )

    lines.extend([
        "",
        "### Intentionally not applicable",
        "",
        "| Task | Dataset | Model | Adaptation | Method | Reason |",
        "|---|---|---|---|---|---|",
    ])
    for row in rq3_na:
        lines.append(
            f"| {row['task']} | {DISPLAY[row['dataset']]} | {row['model'].upper()} | {row['adaptation']} | "
            f"{row['comparison_method']} | {row['applicability_reason']} |"
        )

    lines.extend([
        "",
        "## Interpretation boundaries",
        "",
        "- RQ2 and MC Dropout seed matches are design-paired independently trained models; only Temperature Scaling is a same-checkpoint post-hoc comparison.",
        "- Deep Ensemble effects compare one probability-mean ensemble with the mean of its exact seeds 42/43/44 members. They have no seedwise effect standard deviation.",
        "- Three-seed summaries are descriptive (n=3); no confidence intervals, hypothesis tests, or causal claims are made.",
        "- TreeSatAI Accuracy is strict multilabel exact-match accuracy, and its calibration quantities follow binary-label semantics; they are not directly pooled with EuroSAT multiclass values.",
        "- ECE means ECE-15 throughout. No sample-level bootstrap, reliability-bin, error-overlap, or uncertainty-diversity analysis was inferred from aggregate metrics.",
        "- TreeSatAI and segmentation Temperature Scaling remain N/A under the frozen final protocol; historical diagnostic artifacts are not reintroduced.",
        "",
        "Machine-readable tables:",
        "",
        "- `reports/rq2_adaptation_effects.csv`",
        "- `reports/rq3_uq_effects.csv`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(rq2: list[dict[str, Any]], rq3: list[dict[str, Any]]) -> None:
    require(Counter(row["record_type"] for row in rq2) == {"PAIRED_SEED": 24, "PAIRED_MEAN_STD": 8}, "RQ2 validation failed")
    require(Counter(row["record_type"] for row in rq3) == {"PAIRED_SEED": 36, "PAIRED_MEAN_STD": 8, "AGGREGATED_COMPARISON": 16, "NOT_APPLICABLE": 12}, "RQ3 validation failed")
    for row in (*rq2, *rq3):
        if row["status"] == "N/A":
            require(all(number(row.get(f"delta_{metric}")) is None for metric in METRICS), "N/A row contains an effect")
            continue
        for metric in METRICS:
            before = number(row.get(f"baseline_{metric}"))
            after = number(row.get(f"comparison_{metric}"))
            delta = number(row.get(f"delta_{metric}"))
            require((before is None) == (after is None) == (delta is None), f"Partial effect metric: {metric}")
            if delta is not None:
                assert_close(delta, after - before, f"effect arithmetic {row['rq']}/{metric}")
        if row["record_type"] == "AGGREGATED_COMPARISON":
            require(row["comparison_method"] == "deep_ensemble" and row["n_members"] == 3, "Invalid aggregate comparison")
            require(all(number(row.get(f"delta_{metric}_std")) is None for metric in METRICS), "Ensemble effect has an invented std")
        if row["record_type"] == "PAIRED_SEED":
            require(row["seed"] is not None and row["contributing_seeds"] == str(row["seed"]), "Invalid seed pair identity")


def main() -> None:
    for path in (RQ2_OUT, RQ3_OUT, SUMMARY_OUT):
        require(not path.exists(), f"Refusing to overwrite existing output: {path}")
    master = read_master()
    rq2 = build_rq2(master)
    rq3 = build_rq3(master)
    validate(rq2, rq3)
    staging = Path(tempfile.mkdtemp(prefix=".rq_effects_", dir=ROOT / "reports"))
    try:
        temp_rq2 = staging / RQ2_OUT.name
        temp_rq3 = staging / RQ3_OUT.name
        temp_summary = staging / SUMMARY_OUT.name
        write_csv(temp_rq2, rq2)
        write_csv(temp_rq3, rq3)
        write_summary(temp_summary, rq2, rq3)
        os.replace(temp_rq2, RQ2_OUT)
        os.replace(temp_rq3, RQ3_OUT)
        os.replace(temp_summary, SUMMARY_OUT)
    finally:
        staging.rmdir()
    print({
        "rq2_rows": len(rq2),
        "rq2_types": dict(Counter(row["record_type"] for row in rq2)),
        "rq3_rows": len(rq3),
        "rq3_types": dict(Counter(row["record_type"] for row in rq3)),
    })


if __name__ == "__main__":
    main()
