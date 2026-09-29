#!/usr/bin/env python3
"""Build the C7 self-contained research-data archive from validated outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "research_data"
REPORTS = ROOT / "reports"
ANALYSIS_SCRIPT = ROOT / "scripts/recompute_research_uncertainty.py"
FIXED_SUBSET = REPORTS / "mc_dropout_segmentation_research_subset_ids.json"
C2_AUDIT = REPORTS / "c2_segmentation_run_audits/20260824T162250Z/audit.csv"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def npz_headers(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as handle:
                version = np.lib.format.read_magic(handle)
                shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
            output[member[:-4]] = {
                "shape": list(shape),
                "dtype": str(dtype),
                "fortran_order": bool(fortran),
            }
    return output


def describe(path: Path) -> tuple[str, int | None, str]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        arrays = npz_headers(path)
        sample_count = None
        for key in ("sample_id", "sample_ids", "label", "labels", "true_labels", "probabilities"):
            if key in arrays and arrays[key]["shape"]:
                sample_count = int(arrays[key]["shape"][0])
                break
        return "npz", sample_count, json.dumps({"arrays": arrays}, sort_keys=True)
    if suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        columns = [{"name": field.name, "type": str(field.type)} for field in parquet.schema_arrow]
        return "parquet", int(parquet.metadata.num_rows), json.dumps({"columns": columns}, sort_keys=True)
    if suffix == ".json":
        return "json", None, "{}"
    if suffix == ".py":
        return "python", None, "{}"
    return suffix.lstrip(".") or "binary", None, "{}"


class Archive:
    def __init__(self, staging: Path) -> None:
        self.staging = staging
        self.rows: list[dict[str, Any]] = []
        self.ids: set[str] = set()

    @staticmethod
    def _copy_with_sha256(source: Path, destination: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as reader, destination.open("xb") as writer:
            for block in iter(lambda: reader.read(16 * 1024 * 1024), b""):
                writer.write(block)
                digest.update(block)
                size += len(block)
        shutil.copystat(source, destination, follow_symlinks=True)
        require(size == source.stat().st_size == destination.stat().st_size, f"Copy size mismatch: {source}")
        return digest.hexdigest(), size

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _record(
        self,
        destination: Path,
        *,
        artifact_id: str,
        task: str,
        dataset: str,
        model: str,
        adaptation: str,
        uq_method: str,
        seed: int | None,
        member_seeds: Iterable[int],
        split: str,
        role: str,
        source_paths: Iterable[Path],
        preservation_mode: str,
        sha256: str,
        size_bytes: int,
        probability_semantics: str,
        draw_axis: int | None,
        class_axis: int | None,
        validated_by: Iterable[Path],
        notes: str = "",
    ) -> None:
        require(artifact_id not in self.ids, f"Duplicate artifact ID: {artifact_id}")
        self.ids.add(artifact_id)
        file_format, sample_count, schema = describe(destination)
        self.rows.append(
            {
                "artifact_id": artifact_id,
                "task": task,
                "dataset": dataset,
                "model": model,
                "adaptation": adaptation,
                "uq_method": uq_method,
                "seed": seed,
                "member_seeds": [int(value) for value in member_seeds],
                "split": split,
                "role": role,
                "archive_path": str(destination.relative_to(self.staging)),
                "source_paths": [relative(path) for path in source_paths],
                "preservation_mode": preservation_mode,
                "format": file_format,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "sample_count": sample_count,
                "array_schema_json": schema,
                "probability_semantics": probability_semantics,
                "draw_axis": draw_axis,
                "class_axis": class_axis,
                "validated_by": [relative(path) for path in validated_by],
                "notes": notes,
            }
        )

    def copy(self, source: Path, destination_relative: str, **metadata: Any) -> Path:
        require(source.is_file(), f"Missing source artifact: {source}")
        destination = self.staging / destination_relative
        digest, size = self._copy_with_sha256(source, destination)
        self._record(
            destination,
            source_paths=[source],
            preservation_mode="independent_copy",
            sha256=digest,
            size_bytes=size,
            **metadata,
        )
        return destination

    def register_derived(self, destination: Path, source_paths: Iterable[Path], **metadata: Any) -> None:
        require(destination.is_file(), f"Derived artifact missing: {destination}")
        self._record(
            destination,
            source_paths=source_paths,
            preservation_mode="array_subset_extracted_from_validated_outputs",
            sha256=self._sha256(destination),
            size_bytes=destination.stat().st_size,
            **metadata,
        )


def metadata(
    artifact_id: str,
    task: str,
    dataset: str,
    model: str,
    adaptation: str,
    uq_method: str,
    role: str,
    validated_by: Iterable[Path],
    *,
    seed: int | None = None,
    member_seeds: Iterable[int] = (),
    semantics: str = "not_applicable",
    draw_axis: int | None = None,
    class_axis: int | None = None,
    split: str = "test",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "task": task,
        "dataset": dataset,
        "model": model,
        "adaptation": adaptation,
        "uq_method": uq_method,
        "seed": seed,
        "member_seeds": member_seeds,
        "split": split,
        "role": role,
        "probability_semantics": semantics,
        "draw_axis": draw_axis,
        "class_axis": class_axis,
        "validated_by": validated_by,
        "notes": notes,
    }


def build_segmentation_member_subset(
    output: Path,
    member_rows: list[dict[str, str]],
    sample_ids: list[str],
) -> list[Path]:
    member_probabilities = []
    member_run_ids = []
    labels = None
    valid_mask = None
    source_paths = []
    for row in sorted(member_rows, key=lambda value: int(value["seed"])):
        source = Path(row["run_dir"]) / "predictions/test/deterministic/predictions.npz"
        source_paths.append(source)
        with np.load(source, allow_pickle=False) as archive:
            available_ids = [str(value) for value in archive["sample_id"].tolist()]
            require(len(available_ids) == len(set(available_ids)), f"Non-unique segmentation IDs: {source}")
            positions = {value: index for index, value in enumerate(available_ids)}
            require(all(value in positions for value in sample_ids), f"Fixed subset absent from {source}")
            indices = np.asarray([positions[value] for value in sample_ids], dtype=np.int64)
            current_labels = archive["label"][indices]
            current_valid = archive["valid_mask"][indices].astype(bool)
            if labels is None:
                labels = current_labels
                valid_mask = current_valid
            else:
                require(np.array_equal(labels, current_labels), f"Member label mismatch: {source}")
                require(np.array_equal(valid_mask, current_valid), f"Member valid-mask mismatch: {source}")
            member_probabilities.append(archive["probabilities"][indices].astype(np.float32))
        member_run_ids.append(row["run_id"])
    probabilities = np.stack(member_probabilities, axis=1)
    require(np.isfinite(probabilities).all(), "Non-finite segmentation member probabilities")
    require(float(np.max(np.abs(probabilities.sum(axis=2, dtype=np.float64) - 1.0))) < 1.0e-4, "Member probabilities do not sum to one")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_id=np.asarray(sample_ids),
        label=labels,
        valid_mask=valid_mask,
        member_seeds=np.asarray([int(row["seed"]) for row in sorted(member_rows, key=lambda value: int(value["seed"]))], dtype=np.int64),
        member_run_ids=np.asarray(member_run_ids),
        probabilities=probabilities,
        axes=np.asarray(["sample", "member", "class", "height", "width"]),
    )
    return source_paths


def manifest_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("artifact_id", pa.string(), nullable=False),
            pa.field("task", pa.string(), nullable=False),
            pa.field("dataset", pa.string(), nullable=False),
            pa.field("model", pa.string(), nullable=False),
            pa.field("adaptation", pa.string(), nullable=False),
            pa.field("uq_method", pa.string(), nullable=False),
            pa.field("seed", pa.int32()),
            pa.field("member_seeds", pa.list_(pa.int32()), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("role", pa.string(), nullable=False),
            pa.field("archive_path", pa.string(), nullable=False),
            pa.field("source_paths", pa.list_(pa.string()), nullable=False),
            pa.field("preservation_mode", pa.string(), nullable=False),
            pa.field("format", pa.string(), nullable=False),
            pa.field("sha256", pa.string(), nullable=False),
            pa.field("size_bytes", pa.int64(), nullable=False),
            pa.field("sample_count", pa.int64()),
            pa.field("array_schema_json", pa.string(), nullable=False),
            pa.field("probability_semantics", pa.string(), nullable=False),
            pa.field("draw_axis", pa.int32()),
            pa.field("class_axis", pa.int32()),
            pa.field("validated_by", pa.list_(pa.string()), nullable=False),
            pa.field("notes", pa.string(), nullable=False),
        ]
    )


def write_schema(path: Path) -> None:
    value = {
        "schema_version": 1,
        "manifest": {
            "path": "manifest.parquet",
            "path_resolution": "archive_path is relative to the research_data directory; source_paths and validated_by are relative to the project root",
            "columns": {field.name: str(field.type) + (" (nullable)" if field.nullable else "") for field in manifest_schema()},
        },
        "axis_conventions": {
            "classification_mc": "probabilities[N,T,C]",
            "classification_ensemble": "probabilities[N,M,C]",
            "segmentation_mc_subset": "probabilities[N,T,C,H,W]",
            "segmentation_ensemble_subset": "probabilities[N,M,C,H,W]",
            "segmentation_full_mean": "probabilities[N,C,H,W]",
        },
        "probability_semantics": {
            "eurosat": "categorical multiclass; class probabilities sum to one",
            "treesatai": "independent Bernoulli multilabel probabilities; do not normalize across labels",
            "cloudsen12": "categorical pixelwise multiclass; class probabilities sum to one",
            "spacenet7": "categorical pixelwise multiclass; class probabilities sum to one",
        },
        "offline_uncertainty_outputs": {
            "mean_probabilities": "mean over axis 1",
            "predictive_entropy": "entropy of mean probability",
            "expected_predictive_entropy": "mean entropy across stochastic passes or members",
            "mi_style_disagreement": "predictive_entropy minus expected_predictive_entropy",
            "predictive_variance": "population variance across stochastic passes or members",
        },
        "formats": {
            "npz": "array_schema_json lists keys, shapes, dtypes, and Fortran-order flags",
            "parquet": "array_schema_json lists column names and Arrow types",
        },
    }
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_readme(path: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["role"] for row in rows)
    total = sum(int(row["size_bytes"]) for row in rows)
    lines = [
        "# Research-ready thesis data archive",
        "",
        "This C7 archive contains independent copies of validated C1–C5 prediction artifacts plus segmentation ensemble-member tensors extracted for the prospectively fixed research subset. It was built without training, checkpoint loading, model construction, or neural-network inference.",
        "",
        "## Integrity and portability",
        "",
        "- `manifest.parquet` is the authoritative index. Every row includes a SHA256, byte size, source path, validation source, array/column schema, and probability semantics.",
        "- Data artifacts are independent byte-for-byte copies, not symlinks or hard links. The directory can be moved as one unit.",
        "- `archive_path` is relative to this directory. Source and validation paths are project-root relative provenance references.",
        f"- Indexed artifacts: **{len(rows)}**; apparent archived size: **{total / 2**30:.2f} GiB**.",
        "",
        "## Coverage",
        "",
        f"- Classification deterministic prediction exports: {counts['classification_deterministic_predictions']} (all 8 cells × 3 seeds).",
        f"- Classification deterministic embedding files: {counts['classification_deterministic_embeddings']}.",
        f"- Classification Temperature-Scaling raw/calibrated exports: {counts['classification_temperature_scaling_predictions']}.",
        f"- Classification MC raw `[N,T,C]` files: {counts['classification_mc_raw_passes']}; MC embedding files: {counts['classification_mc_embeddings']}.",
        f"- Classification ensemble member `[N,M,C]` files: {counts['classification_ensemble_members']}.",
        f"- Segmentation deterministic probability files: {counts['segmentation_deterministic_predictions']}; dense/global representation files: {counts['segmentation_representations']}.",
        f"- Segmentation MC full-test uncertainty maps: {counts['segmentation_mc_full_maps']}; fixed-subset raw `[N,T,C,H,W]` files: {counts['segmentation_mc_raw_subset']}.",
        f"- Segmentation ensemble mean-probability files: {counts['segmentation_ensemble_mean']}; fixed-subset member `[N,M,C,H,W]` files: {counts['segmentation_ensemble_member_subset']}.",
        "",
        "Historical limitation: the six immutable DOFA–EuroSAT deterministic exports predate backbone-representation export and therefore have no deterministic test embedding. They are retained as-is. No inference was run to manufacture the missing embeddings. Test embeddings are present for the other 18 deterministic classification runs, and MC-Dropout archives include their own embeddings where produced.",
        "",
        "## Fixed segmentation research subset",
        "",
        "`metadata/fixed_research_subset_ids.json` is the C4/C5 predeclared selection. MC raw passes and derived ensemble-member subsets use exactly those IDs and their stored order. Derived member tensors are pure selection/stacking of the three saved deterministic probability arrays; member probabilities were not recomputed.",
        "",
        "## Checkpoint-free uncertainty analysis",
        "",
        "The bundled `recompute_uncertainty.py` imports only NumPy. It expects axis 1 to be the stochastic-pass/member axis and axis 2 to be the class axis.",
        "",
        "```bash",
        "python research_data/recompute_uncertainty.py \\",
        "  --input research_data/artifacts/classification/mc_dropout/eurosat/dofa/frozen/seed42/stochastic_outputs.npz \\",
        "  --output /tmp/eurosat_uncertainty.npz \\",
        "  --semantics categorical",
        "```",
        "",
        "For TreeSatAI, use `--semantics independent_bernoulli`. This computes per-label Bernoulli entropy and stores both per-label arrays and their sample-level sum. For segmentation research-subset tensors, keep categorical semantics and use a small `--batch-size` if memory is limited.",
        "",
        "Some legacy classification NPZ files store sample IDs as NumPy object arrays. Probability tensors remain readable with the secure default; the script skips those object IDs. Add `--trust-input-pickle` only for these SHA256-pinned archive files when sample-ID passthrough is required.",
        "",
        "The script recomputes predictive entropy, expected predictive entropy, MI-style disagreement, and population predictive variance. These are descriptive uncertainty quantities; MI-style disagreement is an epistemic proxy, not a claim of true epistemic uncertainty.",
        "",
        "## Recommended loading pattern",
        "",
        "Filter `manifest.parquet` by task/dataset/model/adaptation/UQ/role, resolve `archive_path` relative to `research_data/`, and verify SHA256 before long-term reuse. Parse `array_schema_json` before loading large NPZ files. Do not treat TreeSatAI probabilities as a multiclass simplex.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    require(not FINAL.exists(), f"Refusing to overwrite {FINAL}")
    require(ANALYSIS_SCRIPT.is_file(), f"Missing analysis script: {ANALYSIS_SCRIPT}")

    dofa_manifest_path = REPORTS / "dofa_eurosat_final_manifest.json"
    dofa_manifest = read_json(dofa_manifest_path)
    require(dofa_manifest["status"] == "FINAL_IMMUTABLE" and dofa_manifest["immutability"]["verified"], "DOFA-EuroSAT final manifest is not immutable")
    c1_audit_path = sorted((REPORTS / "c1_classification_run_audits").glob("*/c1_run_audit.json"))[-1]
    c1_runs = read_json(c1_audit_path)["runs"]
    require(len(c1_runs) == 18 and all(row["status"] == "PROMOTABLE" for row in c1_runs), "C1 audit is not fully promotable")
    c2_rows = read_csv(C2_AUDIT)
    require(len(c2_rows) == 24 and all(row["status"] == "PROMOTABLE" for row in c2_rows), "C2 audit is not fully promotable")
    c3_class_path = REPORTS / "c3_classification_results.csv"
    c3_class = read_csv(c3_class_path)
    require(len(c3_class) == 32, "Expected 32 C3 classification rows")
    c3_seg_path = REPORTS / "c3_segmentation_ensemble_results.csv"
    c3_seg = read_csv(c3_seg_path)
    require(len(c3_seg) == 8, "Expected eight C3 segmentation ensemble rows")
    c5_path = REPORTS / "mc_dropout_results.csv"
    c5_rows = read_csv(c5_path)
    require(len(c5_rows) == 24, "Expected 24 validated C5 rows")
    fixed = read_json(FIXED_SUBSET)
    require(fixed["selected_before_any_c4_stochastic_inference"], "Segmentation subset was not prospectively fixed")

    staging = Path(tempfile.mkdtemp(prefix=".research_data_", dir=ROOT))
    archive = Archive(staging)
    try:
        archive.copy(
            ANALYSIS_SCRIPT,
            "recompute_uncertainty.py",
            **metadata("archive.analysis", "archive", "all", "all", "all", "offline_analysis", "checkpoint_free_analysis_script", []),
        )
        archive.copy(
            FIXED_SUBSET,
            "metadata/fixed_research_subset_ids.json",
            **metadata("segmentation.fixed_subset.definition", "segmentation", "all", "all", "all", "all", "fixed_subset_definition", [FIXED_SUBSET], split="test"),
        )

        deterministic_classification: list[dict[str, Any]] = []
        for row in dofa_manifest["runs"]:
            deterministic_classification.append(
                {
                    "dataset": "eurosat", "model": "dofa", "adaptation": row["adaptation"], "seed": int(row["seed"]),
                    "prediction": Path(row["test_prediction_path"]), "embedding": None, "validated_by": dofa_manifest_path,
                }
            )
        for row in c1_runs:
            target = row["target"]
            prediction_dir = Path(row["prediction_export"]["path"])
            deterministic_classification.append(
                {
                    "dataset": target["dataset"], "model": target["model"], "adaptation": target["adaptation"], "seed": int(target["seed"]),
                    "prediction": prediction_dir / "predictions.parquet", "embedding": prediction_dir / "embeddings.npz", "validated_by": c1_audit_path,
                }
            )
        require(len(deterministic_classification) == 24, "Classification deterministic registry must contain 24 runs")
        for row in deterministic_classification:
            dataset, model, adaptation, seed = row["dataset"], row["model"], row["adaptation"], row["seed"]
            base = f"artifacts/classification/deterministic/{dataset}/{model}/{adaptation}/seed{seed}"
            semantics = "independent_bernoulli" if dataset == "treesatai" else "categorical"
            archive.copy(
                row["prediction"], f"{base}/predictions.parquet",
                **metadata(f"classification.deterministic.{dataset}.{model}.{adaptation}.seed{seed}.predictions", "classification", dataset, model, adaptation, "deterministic", "classification_deterministic_predictions", [row["validated_by"]], seed=seed, semantics=semantics, class_axis=1, notes="Contains logits, probabilities, labels/correctness; C1 schema also embeds backbone representations."),
            )
            if row["embedding"] is not None:
                archive.copy(
                    row["embedding"], f"{base}/embeddings.npz",
                    **metadata(f"classification.deterministic.{dataset}.{model}.{adaptation}.seed{seed}.embeddings", "classification", dataset, model, adaptation, "deterministic", "classification_deterministic_embeddings", [row["validated_by"]], seed=seed, notes="Test-split backbone representation export."),
                )

        temperature_rows = [row for row in c3_class if row["method"].startswith("temperature_scaling")]
        require(len(temperature_rows) == 24, "Expected 24 Temperature-Scaling rows")
        for row in temperature_rows:
            dataset, model, adaptation, seed = row["dataset"], row["model"], row["adaptation"], int(row["seed"])
            filename = "test_predictions.parquet" if dataset == "eurosat" else "test_predictions_raw_and_calibrated.parquet"
            source = Path(row["output_path"]) / filename
            archive.copy(
                source, f"artifacts/classification/temperature_scaling/{dataset}/{model}/{adaptation}/seed{seed}/{filename}",
                **metadata(f"classification.temperature_scaling.{dataset}.{model}.{adaptation}.seed{seed}.predictions", "classification", dataset, model, adaptation, "temperature_scaling", "classification_temperature_scaling_predictions", [c3_class_path], seed=seed, semantics="independent_bernoulli" if dataset == "treesatai" else "categorical", class_axis=1, notes="Contains untouched raw test logits/probabilities and calibrated test probabilities."),
            )

        for row in [value for value in c5_rows if value["task"] == "classification"]:
            dataset, model, adaptation, seed = row["dataset"], row["model"], row["adaptation"], int(row["seed"])
            output = Path(row["output_path"])
            audit = output / "c5_audit.json"
            require(read_json(audit)["passed"], f"Unvalidated C5 classification output: {output}")
            base = f"artifacts/classification/mc_dropout/{dataset}/{model}/{adaptation}/seed{seed}"
            semantics = "independent_bernoulli" if dataset == "treesatai" else "categorical"
            for filename, role, axes in (
                ("stochastic_outputs.npz", "classification_mc_raw_passes", (1, 2)),
                ("embeddings.npz", "classification_mc_embeddings", (None, None)),
                ("uncertainty_summaries.npz", "classification_mc_saved_summaries", (None, 1)),
            ):
                archive.copy(
                    output / filename, f"{base}/{filename}",
                    **metadata(f"classification.mc_dropout.{dataset}.{model}.{adaptation}.seed{seed}.{Path(filename).stem}", "classification", dataset, model, adaptation, "mc_dropout", role, [c5_path, audit], seed=seed, semantics=semantics if "embeddings" not in filename else "not_applicable", draw_axis=axes[0], class_axis=axes[1], notes="Raw MC tensor uses T=30." if filename == "stochastic_outputs.npz" else ""),
                )

        ensemble_class = [row for row in c3_class if row["method"] == "deep_ensemble_probability_mean"]
        require(len(ensemble_class) == 8, "Expected eight classification ensembles")
        for row in ensemble_class:
            dataset, model, adaptation = row["dataset"], row["model"], row["adaptation"]
            output = Path(row["output_path"])
            member_file = output / ("member_probabilities.npz" if (output / "member_probabilities.npz").exists() else "member_predictions.npz")
            semantics = "independent_bernoulli" if dataset == "treesatai" else "categorical"
            base = f"artifacts/classification/deep_ensemble/{dataset}/{model}/{adaptation}"
            archive.copy(
                member_file, f"{base}/member_outputs.npz",
                **metadata(f"classification.deep_ensemble.{dataset}.{model}.{adaptation}.members", "classification", dataset, model, adaptation, "deep_ensemble", "classification_ensemble_members", [c3_class_path, output / "manifest.json"], member_seeds=(42, 43, 44), semantics=semantics, draw_axis=1, class_axis=2, notes="Member probability tensor is [N,M,C]; TreeSatAI member file also preserves logits."),
            )
            archive.copy(
                output / "ensemble_probabilities.npz", f"{base}/ensemble_probabilities.npz",
                **metadata(f"classification.deep_ensemble.{dataset}.{model}.{adaptation}.mean", "classification", dataset, model, adaptation, "deep_ensemble", "classification_ensemble_mean", [c3_class_path, output / "manifest.json"], member_seeds=(42, 43, 44), semantics=semantics, class_axis=1, notes="Arithmetic mean of member probabilities, never logits."),
            )

        segmentation_registry: dict[tuple[str, str, str, int], dict[str, str]] = {}
        for row in c2_rows:
            key = (row["dataset"], row["model"], row["adaptation"], int(row["seed"]))
            segmentation_registry[key] = row
            output = Path(row["run_dir"]) / "predictions/test/deterministic"
            require(read_json(output / "validation_report.json")["valid"], f"Invalid C2 prediction export: {output}")
            dataset, model, adaptation, seed = *key[:3], key[3]
            base = f"artifacts/segmentation/deterministic/{dataset}/{model}/{adaptation}/seed{seed}"
            archive.copy(
                output / "predictions.npz", f"{base}/predictions.npz",
                **metadata(f"segmentation.deterministic.{dataset}.{model}.{adaptation}.seed{seed}.predictions", "segmentation", dataset, model, adaptation, "deterministic", "segmentation_deterministic_predictions", [C2_AUDIT, output / "validation_report.json"], seed=seed, semantics="categorical", class_axis=1, notes="Full test logits/probabilities, labels, predictions, correctness, confidence, entropy, and valid mask."),
            )
            archive.copy(
                output / "representations.npz", f"{base}/representations.npz",
                **metadata(f"segmentation.deterministic.{dataset}.{model}.{adaptation}.seed{seed}.representations", "segmentation", dataset, model, adaptation, "deterministic", "segmentation_representations", [C2_AUDIT, output / "validation_report.json"], seed=seed, notes="Global mean of final dense backbone feature map; dense feature maps were not retained."),
            )

        for row in [value for value in c5_rows if value["task"] == "segmentation"]:
            dataset, model, adaptation, seed = row["dataset"], row["model"], row["adaptation"], int(row["seed"])
            output = Path(row["output_path"])
            audit = output / "manifest.json"
            require(read_json(audit)["passed"], f"Unvalidated C5 segmentation output: {output}")
            base = f"artifacts/segmentation/mc_dropout/{dataset}/{model}/{adaptation}/seed{seed}"
            archive.copy(
                output / "aggregate_uncertainty_maps.npz", f"{base}/aggregate_uncertainty_maps.npz",
                **metadata(f"segmentation.mc_dropout.{dataset}.{model}.{adaptation}.seed{seed}.maps", "segmentation", dataset, model, adaptation, "mc_dropout", "segmentation_mc_full_maps", [c5_path, audit, FIXED_SUBSET], seed=seed, semantics="categorical", class_axis=1, notes="Full-test mean probabilities, predictions, uncertainty maps, and valid mask."),
            )
            archive.copy(
                output / "research_subset_stochastic_probabilities.npz", f"{base}/research_subset_stochastic_probabilities.npz",
                **metadata(f"segmentation.mc_dropout.{dataset}.{model}.{adaptation}.seed{seed}.raw_subset", "segmentation", dataset, model, adaptation, "mc_dropout", "segmentation_mc_raw_subset", [c5_path, audit, FIXED_SUBSET], seed=seed, semantics="categorical", draw_axis=1, class_axis=2, notes="Prospectively fixed research subset; raw tensor is [N,T,C,H,W] with T=30."),
            )

        for row in c3_seg:
            dataset, model, adaptation = row["dataset"], row["model"], row["adaptation"]
            output = Path(row["output_path"])
            ensemble_manifest = output / "manifest.json"
            require(read_json(ensemble_manifest)["aggregation"].startswith("arithmetic mean"), f"Invalid ensemble aggregation: {output}")
            base = f"artifacts/segmentation/deep_ensemble/{dataset}/{model}/{adaptation}"
            archive.copy(
                output / "ensemble_predictions.npz", f"{base}/ensemble_predictions.npz",
                **metadata(f"segmentation.deep_ensemble.{dataset}.{model}.{adaptation}.mean", "segmentation", dataset, model, adaptation, "deep_ensemble", "segmentation_ensemble_mean", [c3_seg_path, ensemble_manifest], member_seeds=(42, 43, 44), semantics="categorical", class_axis=1, notes="Full-test arithmetic mean of member probabilities."),
            )
            member_rows = [segmentation_registry[(dataset, model, adaptation, seed)] for seed in (42, 43, 44)]
            subset_output = staging / base / "research_subset_member_probabilities.npz"
            subset_ids = fixed["datasets"][dataset]["final_test_research_subset_ids"]
            sources = build_segmentation_member_subset(subset_output, member_rows, subset_ids)
            archive.register_derived(
                subset_output,
                source_paths=sources,
                **metadata(f"segmentation.deep_ensemble.{dataset}.{model}.{adaptation}.member_subset", "segmentation", dataset, model, adaptation, "deep_ensemble", "segmentation_ensemble_member_subset", [C2_AUDIT, c3_seg_path, ensemble_manifest, FIXED_SUBSET], member_seeds=(42, 43, 44), semantics="categorical", draw_axis=1, class_axis=2, notes="Pure fixed-ID extraction and stacking of saved deterministic member probabilities; tensor is [N,M,C,H,W]."),
            )

        write_schema(staging / "schema.json")
        write_readme(staging / "README.md", archive.rows)
        require(len(archive.rows) > 0, "Archive manifest would be empty")
        table = pa.Table.from_pylist(sorted(archive.rows, key=lambda row: row["artifact_id"]), schema=manifest_schema())
        pq.write_table(table, staging / "manifest.parquet", compression="zstd")
        os.replace(staging, FINAL)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    print(
        json.dumps(
            {
                "archive": str(FINAL),
                "artifacts": len(archive.rows),
                "size_gib": sum(int(row["size_bytes"]) for row in archive.rows) / 2**30,
                "roles": dict(sorted(Counter(row["role"] for row in archive.rows).items())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
