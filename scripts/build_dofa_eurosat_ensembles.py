from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from models.calibration import plot_reliability_diagram
from scripts.experiment_manager import PROJECT_ROOT, write_code_snapshot_manifest


EXPECTED_SAMPLE_COUNT = 2714
EXPECTED_CLASS_COUNT = 10
N_BINS = 15
OUTPUT_RELATIVE_PATH = Path("results/final_thesis/dofa_eurosat_ensembles")
ARTIFACT_REPORT_RELATIVE_PATH = Path("reports/dofa_eurosat_frozen_artifacts.md")
RESULTS_REPORT_RELATIVE_PATH = Path("reports/dofa_eurosat_ensemble_results.csv")


@dataclass(frozen=True)
class FinalRun:
    adaptation: str
    seed: int
    run_id: str
    run_relative_path: str
    checkpoint_sha256: str

    def run_dir(self, project_root: Path) -> Path:
        return project_root / self.run_relative_path / self.run_id


FINAL_RUNS = (
    FinalRun(
        adaptation="frozen",
        seed=42,
        run_id="20260807T103233022656Z_eurosat_dofa_frozen_bnlinear_seed42_338e2700",
        run_relative_path="results/baselines/dofa_eurosat_frozen_bnlinear/runs",
        checkpoint_sha256="7cb285a3e45dcf6acb67c58e474d0b2bfed1aa78ddb6139ecd1dc261c84b54d0",
    ),
    FinalRun(
        adaptation="frozen",
        seed=43,
        run_id="20260807T104927933179Z_eurosat_dofa_frozen_bnlinear_seed43_3567033c",
        run_relative_path="results/baselines/dofa_eurosat_frozen_bnlinear/runs",
        checkpoint_sha256="67b3eb2be1b5f7bd2fb280fd975ac0583fc83bc3d4b3bc0eed2fb9f64c0b7960",
    ),
    FinalRun(
        adaptation="frozen",
        seed=44,
        run_id="20260807T110954653187Z_eurosat_dofa_frozen_bnlinear_seed44_36c48b49",
        run_relative_path="results/baselines/dofa_eurosat_frozen_bnlinear/runs",
        checkpoint_sha256="22e7fa5dae5d6e50606fc2c0e0707db513962efc13d62d6d71b22e2fe506a0b7",
    ),
    FinalRun(
        adaptation="full_finetune",
        seed=42,
        run_id="20260808T211626550846Z_eurosat_dofa_full_finetune_seed42_466f9f00",
        run_relative_path="results/baselines/dofa_eurosat_full_finetune/runs",
        checkpoint_sha256="dc5cbeaaada0ae763de0b916fa9de099613c0ca495156b4984c09105e7010ae6",
    ),
    FinalRun(
        adaptation="full_finetune",
        seed=43,
        run_id="20260808T215707981044Z_eurosat_dofa_full_finetune_seed43_53c8b069",
        run_relative_path="results/baselines/dofa_eurosat_full_finetune/runs",
        checkpoint_sha256="87635a994bd56920a2f0eab0dd1cddde528b2e6a277d57476c7d1dd67eb3d719",
    ),
    FinalRun(
        adaptation="full_finetune",
        seed=44,
        run_id="20260808T225252226548Z_eurosat_dofa_full_finetune_seed44_a9291cb9",
        run_relative_path="results/baselines/dofa_eurosat_full_finetune/runs",
        checkpoint_sha256="db55348289cad41a30a36365f96585b9dd1d98d877919cacdbc4b65b253fa730",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classification_metrics(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> Dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    accuracy = float(np.mean(predictions == labels))
    selected = np.clip(probabilities[np.arange(len(labels)), labels], np.finfo(np.float64).tiny, 1.0)
    nll = float(-np.log(selected).mean())
    targets = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    brier = float(np.square(probabilities - targets).sum(axis=1).mean())

    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= lower if index == 0 else confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += abs(float(correct[mask].mean()) - float(confidence[mask].mean())) * float(mask.mean())

    class_f1 = []
    for class_index in range(probabilities.shape[1]):
        true_positive = int(np.sum((labels == class_index) & (predictions == class_index)))
        predicted_count = int(np.sum(predictions == class_index))
        support = int(np.sum(labels == class_index))
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        class_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(class_f1)),
        "nll": nll,
        "brier": brier,
        "ece_15": ece,
    }


def load_and_validate_member(project_root: Path, run: FinalRun) -> Dict[str, Any]:
    run_dir = run.run_dir(project_root)
    checkpoint_path = run_dir / "best.pt"
    prediction_dir = run_dir / "predictions" / "test" / "deterministic"
    prediction_path = prediction_dir / "predictions.parquet"
    manifest_path = prediction_dir / "manifest.json"
    validation_path = prediction_dir / "validation_report.json"
    metrics_path = prediction_dir / "metrics.json"
    required = (checkpoint_path, prediction_path, manifest_path, validation_path, metrics_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing audited artifact(s) for {run.run_id}: {missing}")

    actual_checkpoint_hash = sha256_file(checkpoint_path)
    if actual_checkpoint_hash != run.checkpoint_sha256:
        raise ValueError(
            f"Checkpoint hash mismatch for {run.run_id}: {actual_checkpoint_hash} != {run.checkpoint_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    source = manifest.get("source", {})
    expected_source = {
        "run_id": run.run_id,
        "seed": run.seed,
        "adaptation_mode": run.adaptation,
        "split": "test",
        "uq_method": "deterministic",
        "checkpoint_sha256": run.checkpoint_sha256,
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(f"Manifest {key} mismatch for {run.run_id}: {source.get(key)!r} != {expected!r}")
    if manifest.get("sample_count") != EXPECTED_SAMPLE_COUNT or manifest.get("partial_export") is not False:
        raise ValueError(f"Prediction export is incomplete for {run.run_id}")
    if not validation.get("valid", False):
        raise ValueError(f"Prediction validation failed for {run.run_id}: {validation.get('errors')}")

    table = pd.read_parquet(prediction_path, engine="pyarrow")
    required_columns = {"sample_id", "true_label", "probabilities"}
    if not required_columns.issubset(table.columns):
        raise ValueError(f"Prediction columns missing for {run.run_id}: {sorted(required_columns - set(table.columns))}")
    sample_ids = table["sample_id"].astype(str).to_numpy()
    labels = table["true_label"].to_numpy(dtype=np.int64)
    probabilities = np.asarray(table["probabilities"].tolist(), dtype=np.float32)
    if probabilities.shape != (EXPECTED_SAMPLE_COUNT, EXPECTED_CLASS_COUNT):
        raise ValueError(f"Unexpected probability shape for {run.run_id}: {probabilities.shape}")
    if len(set(sample_ids.tolist())) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"Duplicate sample IDs for {run.run_id}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite probabilities for {run.run_id}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError(f"Probability rows do not sum to one for {run.run_id}")

    recomputed = classification_metrics(probabilities, labels)
    saved_names = {"accuracy": "accuracy", "macro_f1": "macro_f1", "nll": "nll", "brier": "brier", "ece_15": "ece"}
    for current_name, saved_name in saved_names.items():
        if not np.isclose(recomputed[current_name], float(saved_metrics[saved_name]), atol=1e-10, rtol=0):
            raise ValueError(
                f"Metric mismatch for {run.run_id} {current_name}: "
                f"{recomputed[current_name]} != {saved_metrics[saved_name]}"
            )
    return {
        "run": run,
        "run_dir": run_dir,
        "checkpoint_path": checkpoint_path,
        "prediction_path": prediction_path,
        "prediction_manifest_path": manifest_path,
        "sample_ids": sample_ids,
        "labels": labels,
        "class_names": list(manifest["class_names"]),
        "probabilities": probabilities,
        "metrics": recomputed,
    }


def build_ensemble(members: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    reference = members[0]
    for member in members[1:]:
        if not np.array_equal(member["sample_ids"], reference["sample_ids"]):
            raise ValueError("Ensemble member sample ID order differs")
        if not np.array_equal(member["labels"], reference["labels"]):
            raise ValueError("Ensemble member labels differ")
        if member["class_names"] != reference["class_names"]:
            raise ValueError("Ensemble member class mapping differs")
    member_probabilities = np.stack([member["probabilities"] for member in members], axis=1)
    ensemble_probabilities = member_probabilities.astype(np.float64).mean(axis=1)
    probability_sum_max_abs_error = float(np.max(np.abs(ensemble_probabilities.sum(axis=1) - 1.0)))
    if not np.allclose(ensemble_probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Averaged ensemble probability rows do not sum to one")
    return {
        "sample_ids": reference["sample_ids"],
        "labels": reference["labels"],
        "class_names": reference["class_names"],
        "member_probabilities": member_probabilities,
        "ensemble_probabilities": ensemble_probabilities,
        "probability_sum_max_abs_error": probability_sum_max_abs_error,
        "metrics": classification_metrics(ensemble_probabilities, reference["labels"]),
    }


def write_probability_parquet(path: Path, ensemble: Dict[str, Any], adaptation: str) -> None:
    probabilities = ensemble["ensemble_probabilities"]
    labels = ensemble["labels"]
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    sorted_probabilities = np.sort(probabilities, axis=1)
    safe_probabilities = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)
    table = pd.DataFrame(
        {
            "sample_id": ensemble["sample_ids"],
            "true_label": labels,
            "predicted_label": predictions,
            "correct": predictions == labels,
            "probabilities": [row.tolist() for row in probabilities],
            "maximum_softmax_probability": probabilities.max(axis=1),
            "top1_top2_probability_margin": sorted_probabilities[:, -1] - sorted_probabilities[:, -2],
            "predictive_entropy": -(probabilities * np.log(safe_probabilities)).sum(axis=1),
            "model_name": "dofa-base",
            "dataset": "eurosat",
            "adaptation_mode": adaptation,
            "split": "test",
            "uq_method": "deep_ensemble_probability_mean",
            "member_count": 3,
        }
    )
    arrow_table = pa.Table.from_pandas(table, preserve_index=False)
    probability_index = arrow_table.schema.get_field_index("probabilities")
    arrow_table = arrow_table.set_column(
        probability_index,
        "probabilities",
        pa.array([row.tolist() for row in probabilities], type=pa.list_(pa.float64())),
    )
    pq.write_table(arrow_table, path, compression="zstd")


def write_ensemble_outputs(
    output_dir: Path,
    adaptation: str,
    members: Sequence[Dict[str, Any]],
    ensemble: Dict[str, Any],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    reliability_dir = output_dir / "reliability"
    reliability_dir.mkdir()
    seeds = np.asarray([member["run"].seed for member in members], dtype=np.int64)
    run_ids = np.asarray([member["run"].run_id for member in members])
    checkpoint_hashes = np.asarray([member["run"].checkpoint_sha256 for member in members])

    member_probability_path = output_dir / "member_probabilities.npz"
    np.savez_compressed(
        member_probability_path,
        sample_ids=ensemble["sample_ids"],
        true_labels=ensemble["labels"],
        class_names=np.asarray(ensemble["class_names"]),
        seeds=seeds,
        run_ids=run_ids,
        checkpoint_sha256=checkpoint_hashes,
        probabilities=ensemble["member_probabilities"],
    )
    ensemble_probability_path = output_dir / "ensemble_probabilities.npz"
    np.savez_compressed(
        ensemble_probability_path,
        sample_ids=ensemble["sample_ids"],
        true_labels=ensemble["labels"],
        class_names=np.asarray(ensemble["class_names"]),
        member_seeds=seeds,
        probabilities=ensemble["ensemble_probabilities"],
        aggregation=np.asarray("arithmetic_mean_of_member_probabilities"),
    )
    ensemble_prediction_path = output_dir / "ensemble_predictions.parquet"
    write_probability_parquet(ensemble_prediction_path, ensemble, adaptation)

    reliability_paths = []
    for member in members:
        reliability_path = reliability_dir / f"seed{member['run'].seed}_deterministic_test_raw.png"
        plot_reliability_diagram(
            torch.from_numpy(member["probabilities"]),
            torch.from_numpy(member["labels"]),
            reliability_path,
            n_bins=N_BINS,
        )
        reliability_paths.append(reliability_path)
    ensemble_reliability_path = reliability_dir / "deep_ensemble_test_raw.png"
    plot_reliability_diagram(
        torch.from_numpy(ensemble["ensemble_probabilities"]),
        torch.from_numpy(ensemble["labels"]),
        ensemble_reliability_path,
        n_bins=N_BINS,
    )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(ensemble["metrics"], indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "eurosat",
        "model": "dofa-base",
        "adaptation": adaptation,
        "split": "test",
        "sample_count": int(len(ensemble["labels"])),
        "class_names": ensemble["class_names"],
        "member_axis_order": [
            {
                "index": index,
                "seed": member["run"].seed,
                "run_id": member["run"].run_id,
                "checkpoint_path": str(member["checkpoint_path"].resolve()),
                "checkpoint_sha256": member["run"].checkpoint_sha256,
                "prediction_path": str(member["prediction_path"].resolve()),
            }
            for index, member in enumerate(members)
        ],
        "aggregation": "arithmetic mean of member probabilities; logits were not averaged",
        "probability_sum_max_abs_error": ensemble["probability_sum_max_abs_error"],
        "metrics": {"n_bins": N_BINS, "brier_convention": "sum over classes, then mean over samples"},
        "arrays": {
            "member_probabilities": {
                "path": member_probability_path.name,
                "shape": list(ensemble["member_probabilities"].shape),
                "dtype": str(ensemble["member_probabilities"].dtype),
                "axes": ["sample", "member", "class"],
            },
            "ensemble_probabilities": {
                "path": ensemble_probability_path.name,
                "shape": list(ensemble["ensemble_probabilities"].shape),
                "dtype": str(ensemble["ensemble_probabilities"].dtype),
                "axes": ["sample", "class"],
            },
        },
        "outputs": {
            "ensemble_predictions": ensemble_prediction_path.name,
            "metrics": metrics_path.name,
            "deterministic_reliability": [str(path.relative_to(output_dir)) for path in reliability_paths],
            "ensemble_reliability": str(ensemble_reliability_path.relative_to(output_dir)),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "member_probabilities": str(member_probability_path),
        "ensemble_probabilities": str(ensemble_probability_path),
        "ensemble_predictions": str(ensemble_prediction_path),
        "metrics": str(metrics_path),
        "manifest": str(manifest_path),
        "ensemble_reliability": str(ensemble_reliability_path),
    }


def make_results_rows(
    members_by_adaptation: Dict[str, Sequence[Dict[str, Any]]],
    ensembles: Dict[str, Dict[str, Any]],
    output_root: Path,
) -> list[Dict[str, Any]]:
    rows = []
    for adaptation in ("frozen", "full_finetune"):
        for member in members_by_adaptation[adaptation]:
            rows.append(
                {
                    "dataset": "EuroSAT",
                    "model": "DOFA",
                    "adaptation": adaptation,
                    "result_type": "deterministic_member",
                    "seed": member["run"].seed,
                    "member_seeds": str(member["run"].seed),
                    "run_id": member["run"].run_id,
                    **member["metrics"],
                    "prediction_path": str(member["prediction_path"].resolve()),
                    "reliability_diagram": str(
                        (output_root / adaptation / "reliability" / f"seed{member['run'].seed}_deterministic_test_raw.png").resolve()
                    ),
                }
            )
        rows.append(
            {
                "dataset": "EuroSAT",
                "model": "DOFA",
                "adaptation": adaptation,
                "result_type": "deep_ensemble_probability_mean",
                "seed": "",
                "member_seeds": "42;43;44",
                "run_id": "",
                **ensembles[adaptation]["metrics"],
                "prediction_path": str((output_root / adaptation / "ensemble_predictions.parquet").resolve()),
                "reliability_diagram": str(
                    (output_root / adaptation / "reliability" / "deep_ensemble_test_raw.png").resolve()
                ),
            }
        )
    return rows


def artifact_report(members: Sequence[Dict[str, Any]], code_snapshot: Dict[str, Any]) -> str:
    lines = [
        "# DOFA–EuroSAT final artifact freeze manifest",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        "",
        "The six validation-NLL-selected `best.pt` checkpoints and their complete deterministic test exports below are the frozen final thesis artifacts. Historical run directories are immutable: do not overwrite, resume, or replace them. `last.pt` is not a final artifact.",
        "",
        "| Run ID | Checkpoint path | Checkpoint SHA-256 | Prediction path | Adaptation | Seed | Accuracy | Macro-F1 | NLL | Brier | ECE-15 |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for member in members:
        metrics = member["metrics"]
        lines.append(
            "| `{run_id}` | `{checkpoint}` | `{digest}` | `{prediction}` | {adaptation} | {seed} | "
            "{accuracy:.12f} | {macro_f1:.12f} | {nll:.12f} | {brier:.12f} | {ece_15:.12f} |".format(
                run_id=member["run"].run_id,
                checkpoint=member["checkpoint_path"].resolve(),
                digest=member["run"].checkpoint_sha256,
                prediction=member["prediction_path"].resolve(),
                adaptation=member["run"].adaptation,
                seed=member["run"].seed,
                **metrics,
            )
        )
    lines.extend(
        [
            "",
            "## Verification",
            "",
            "- All six checkpoint byte hashes match both this allowlist and the corresponding prediction manifest.",
            "- Every export is a validated, non-partial test export with 2,714 unique samples and 10-class probabilities.",
            "- Sample IDs, labels, row order, and class mappings match within each three-member ensemble.",
            "- The listed metrics were recomputed from the saved probabilities and match the existing metric artifacts.",
            f"- Ensemble-analysis code snapshot: `sha256:{code_snapshot['code_sha256']}` ({code_snapshot['file_count']} files).",
            "- No model was loaded, trained, fine-tuned, or used for inference; no Temperature Scaling was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv_exclusive(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(project_root: Path, validate_only: bool = False) -> Dict[str, Any]:
    members = [load_and_validate_member(project_root, run) for run in FINAL_RUNS]
    members_by_adaptation = {
        adaptation: [member for member in members if member["run"].adaptation == adaptation]
        for adaptation in ("frozen", "full_finetune")
    }
    for adaptation, adaptation_members in members_by_adaptation.items():
        if [member["run"].seed for member in adaptation_members] != [42, 43, 44]:
            raise ValueError(f"Unexpected member seeds for {adaptation}")
    ensembles = {
        adaptation: build_ensemble(adaptation_members)
        for adaptation, adaptation_members in members_by_adaptation.items()
    }
    validation_summary = {
        adaptation: ensemble["metrics"] for adaptation, ensemble in ensembles.items()
    }
    if validate_only:
        return {"validated": True, "ensemble_metrics": validation_summary}

    output_root = project_root / OUTPUT_RELATIVE_PATH
    artifact_report_path = project_root / ARTIFACT_REPORT_RELATIVE_PATH
    results_report_path = project_root / RESULTS_REPORT_RELATIVE_PATH
    existing_targets = [path for path in (output_root, artifact_report_path, results_report_path) if path.exists()]
    if existing_targets:
        raise FileExistsError(f"Refusing to overwrite existing final output(s): {existing_targets}")
    output_root.mkdir(parents=True, exist_ok=False)
    code_snapshot = write_code_snapshot_manifest(output_root, project_root=project_root)
    output_paths = {}
    for adaptation in ("frozen", "full_finetune"):
        output_paths[adaptation] = write_ensemble_outputs(
            output_root / adaptation,
            adaptation,
            members_by_adaptation[adaptation],
            ensembles[adaptation],
        )

    output_hashes = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            output_hashes[path.relative_to(output_root).as_posix()] = sha256_file(path)
    output_manifest_path = output_root / "output_hashes.json"
    output_manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "hash_algorithm": "sha256",
                "files": output_hashes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    artifact_report_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_report_path.open("x", encoding="utf-8") as handle:
        handle.write(artifact_report(members, code_snapshot))
    result_rows = make_results_rows(members_by_adaptation, ensembles, output_root)
    write_csv_exclusive(results_report_path, result_rows)
    return {
        "validated": True,
        "ensemble_metrics": validation_summary,
        "output_root": str(output_root),
        "artifact_report": str(artifact_report_path),
        "results_report": str(results_report_path),
        "code_snapshot": code_snapshot,
        "output_paths": output_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DOFA-EuroSAT ensembles from frozen test probabilities only.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = run(args.project_root.resolve(), validate_only=args.validate_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
