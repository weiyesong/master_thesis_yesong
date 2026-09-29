"""Read-only RQ1/RQ2 source crosscheck; writes only the sibling JSON evidence.

Run from any directory: python /workspace/reports/core_rq_completion_20260921/provenance_numeric_crosscheck.py
Classification: full deterministic parquet probabilities/labels, independent NumPy formulas.
Segmentation: original run test_metrics/confusion, not a second full probability replay.
Direction diagnostics: all deterministic ECE-15 bins, weighted gap and mixed-bin signs.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "reports/core_rq_audit_20260918"
OUT = Path(__file__).resolve().parent
TOLERANCE = 1e-6


def read_csv(name):
    with (AUDIT / name).open() as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def key(row):
    return tuple(row[k] for k in ("dataset", "model", "adaptation", "seed"))


def identity(row):
    return {k: row[k] for k in ("dataset", "model", "adaptation", "seed", "artifact_path")}


def right_closed_ece(confidence, correctness):
    membership = np.clip(np.searchsorted(np.linspace(0, 1, 16), confidence, side="left") - 1, 0, 14)
    return sum(abs((confidence[membership == b] - correctness[membership == b]).sum()) for b in range(15)) / len(confidence)


def classification(row):
    path = ROOT / row["artifact_path"]
    table = pq.read_table(path)
    probabilities = np.asarray(table["probabilities"].to_pylist(), dtype=np.float64)
    if row["dataset"] == "eurosat":
        target = np.asarray(table["true_label"].to_pylist())
        prediction = probabilities.argmax(1)
        correctness = prediction == target
        confidence = probabilities.max(1)
        one_hot = np.eye(probabilities.shape[1])[target]
        nll = -np.log(np.maximum(probabilities[np.arange(len(target)), target], 1e-12)).mean()
        brier = ((probabilities - one_hot) ** 2).sum(1).mean()
        tp = np.array([(prediction[target == k] == k).sum() for k in range(probabilities.shape[1])])
        predicted = np.bincount(prediction, minlength=probabilities.shape[1])
        support = np.bincount(target, minlength=probabilities.shape[1])
        f1 = np.mean(2 * tp / (predicted + support))
        accuracy = correctness.mean()
    else:
        target = np.asarray(table["label"].to_pylist(), dtype=bool)
        prediction = probabilities >= 0.5
        correctness = (prediction == target).ravel()
        confidence = np.maximum(probabilities, 1 - probabilities).ravel()
        nll = -(target * np.log(np.clip(probabilities, 1e-7, 1 - 1e-7)) + (~target) * np.log(np.clip(1 - probabilities, 1e-7, 1 - 1e-7))).mean()
        brier = ((probabilities - target) ** 2).mean()
        accuracy = (prediction == target).all(1).mean()
        tp = (prediction & target).sum(0)
        denominator = prediction.sum(0) + target.sum(0)
        f1 = np.mean(np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0))
    calculated = dict(accuracy=accuracy, macro_f1=f1, nll=nll, brier=brier,
                      ece_15=right_closed_ece(confidence, correctness),
                      mean_confidence_minus_accuracy=(confidence - correctness).mean())
    calculated = {k: float(v) for k, v in calculated.items()}
    differences = {k: v - float(row[k]) for k, v in calculated.items()}
    return {**identity(row), "basis": "full saved parquet probabilities and true labels",
            "artifact_sha256": sha(path), "sample_count": len(table), "calculated": calculated,
            "reference": {k: float(row[k]) for k in calculated}, "difference": differences,
            "pass": all(abs(v) <= TOLERANCE for v in differences.values())}


def segmentation(row):
    source = Path(row["artifact_path"]).parent.parent.parent.parent / "test_metrics.json"
    metrics = json.loads((ROOT / source).read_text())
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    true_positive = confusion.diagonal()
    iou = true_positive / (confusion.sum(0) + confusion.sum(1) - true_positive)
    calculated = {k: float(metrics[k]) for k in ("nll", "brier", "ece_15")}
    calculated.update(miou=float(iou.mean()), pixel_accuracy=float(true_positive.sum() / confusion.sum()))
    differences = {k: v - float(row[k]) for k, v in calculated.items()}
    return {**identity(row), "basis": "original test_metrics.json and independent confusion arithmetic",
            "source_path": str(source), "source_sha256": sha(ROOT / source),
            "confusion_matrix": confusion.tolist(), "calculated": calculated,
            "reference": {k: float(row[k]) for k in calculated}, "difference": differences,
            "pass": all(abs(v) <= TOLERANCE for v in differences.values())}


def main():
    classification_rows = [r for r in read_csv("classification_recomputed_metrics.csv") if r["uq_method"] == "deterministic"]
    segmentation_rows = [r for r in read_csv("segmentation_recomputed_metrics.csv") if r["uq_method"] == "deterministic"]
    checked_classification = [classification(r) for r in classification_rows]
    checked_segmentation = [segmentation(r) for r in segmentation_rows]
    metric_lookup = {key(r): r for r in classification_rows + segmentation_rows}
    direction_bins = defaultdict(list)
    for source in ("classification_reliability_bins.csv", "segmentation_reliability_bins.csv"):
        for row in read_csv(source):
            if row["uq_method"] == "deterministic" and row["kind"] == "decision_15":
                direction_bins[key(row)].append(row)
    directions = []
    for identifier, bins in sorted(direction_bins.items()):
        nonempty = [r for r in bins if int(r["count"]) > 0]
        count = sum(int(r["count"]) for r in nonempty)
        weighted_gap = sum(int(r["count"]) * float(r["signed_confidence_minus_outcome"]) for r in nonempty) / count
        ece = sum(int(r["count"]) * abs(float(r["signed_confidence_minus_outcome"])) for r in nonempty) / count
        ref = metric_lookup[identifier]
        positive = sum(float(r["signed_confidence_minus_outcome"]) > 1e-12 for r in nonempty)
        negative = sum(float(r["signed_confidence_minus_outcome"]) < -1e-12 for r in nonempty)
        differences = {"gap": weighted_gap - float(ref["mean_confidence_minus_accuracy"]), "ece_15": ece - float(ref["ece_15"])}
        directions.append({**identity(ref), "kind": "decision_15", "count": count,
                           "weighted_gap": weighted_gap, "ece_15_from_bins": ece,
                           "nonempty_bins": len(nonempty), "overconfident_bins": positive,
                           "underconfident_bins": negative, "both_signs_present": bool(positive and negative),
                           "difference": differences, "pass": all(abs(v) <= TOLERANCE for v in differences.values())})
    grouped = defaultdict(list)
    for row in directions:
        grouped[(row["dataset"], row["model"], row["adaptation"])].append(row)
    direction_summaries = []
    for (dataset, model, adaptation), rows in sorted(grouped.items()):
        gaps = np.array([r["weighted_gap"] for r in rows])
        direction_summaries.append(dict(dataset=dataset, model=model, adaptation=adaptation,
            seeds=[r["seed"] for r in rows], gap_mean=float(gaps.mean()), gap_min=float(gaps.min()),
            gap_max=float(gaps.max()), overall_positive_seeds=int((gaps > 0).sum()),
            overall_negative_seeds=int((gaps < 0).sum()), seeds_with_mixed_bin_signs=sum(r["both_signs_present"] for r in rows)))
    source_files = ["classification_recomputed_metrics.csv", "segmentation_recomputed_metrics.csv",
                    "classification_reliability_bins.csv", "segmentation_reliability_bins.csv"]
    evidence = {"created_utc": datetime.now(timezone.utc).isoformat(), "script_path": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha(__file__), "command": "python reports/core_rq_completion_20260921/provenance_numeric_crosscheck.py",
        "scope": "24 full classification prediction replays; 24 original segmentation metric/confusion checks; 48 deterministic ECE-15 bin reconstructions",
        "not_claimed": "This is not a second full segmentation probability replay or recovery of training-time source.",
        "tolerance": TOLERANCE,
        "source_tables": [{"path": str((AUDIT / f).relative_to(ROOT)), "sha256": sha(AUDIT / f)} for f in source_files],
        "classification": checked_classification, "segmentation": checked_segmentation,
        "direction_per_run": directions, "direction_summaries": direction_summaries,
        "all_pass": all(r["pass"] for r in checked_classification + checked_segmentation + directions)}
    (OUT / "provenance_numeric_crosscheck.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"classification_count": len(checked_classification), "segmentation_count": len(checked_segmentation),
                      "direction_run_count": len(directions), "all_pass": evidence["all_pass"]}))
    if not evidence["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
