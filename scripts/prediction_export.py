from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from scripts.mc_dropout import activate_downstream_mc_dropout


SCHEMA_VERSION = 2
MAIN_TABLE_NAME = "predictions.parquet"
STOCHASTIC_NAME = "stochastic_outputs.npz"
EMBEDDINGS_NAME = "embeddings.npz"
MANIFEST_NAME = "manifest.json"
VALIDATION_NAME = "validation_report.json"


@dataclass
class PredictionBundle:
    table: pd.DataFrame
    manifest: Dict[str, Any]
    stochastic: Optional[Dict[str, np.ndarray]] = None
    embeddings: Optional[Dict[str, np.ndarray]] = None


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_probabilities(logits: torch.Tensor, classification_type: str = "multiclass") -> torch.Tensor:
    logits = logits.detach().float().cpu()
    if not torch.isfinite(logits).all():
        raise ValueError("Logits contain NaN or Inf")
    if classification_type == "multiclass":
        return torch.log_softmax(logits, dim=-1).exp()
    if classification_type == "multilabel":
        return torch.sigmoid(logits)
    raise ValueError("classification_type must be multiclass or multilabel")


def _to_matrix(series: pd.Series, dtype: np.dtype = np.float32) -> np.ndarray:
    return np.asarray(series.tolist(), dtype=dtype)


def _prediction_summaries(
    probabilities: np.ndarray,
    classification_type: str = "multiclass",
    threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    if probabilities.ndim != 2:
        raise ValueError(f"Expected [samples, classes] probabilities, got {probabilities.shape}")
    safe = np.clip(probabilities.astype(np.float64), np.finfo(np.float64).tiny, 1.0)
    if classification_type == "multilabel":
        predicted = (probabilities >= threshold).astype(np.int8)
        confidence = np.maximum(probabilities, 1.0 - probabilities).mean(axis=1)
        entropy = -(
            probabilities.astype(np.float64) * np.log(safe)
            + (1.0 - probabilities.astype(np.float64))
            * np.log(np.clip(1.0 - probabilities.astype(np.float64), np.finfo(np.float64).tiny, 1.0))
        ).sum(axis=1)
        return {
            "predicted": predicted,
            "maximum": confidence.astype(np.float32),
            "margin": (2.0 * np.abs(probabilities - 0.5)).mean(axis=1).astype(np.float32),
            "entropy": entropy.astype(np.float32),
        }
    if classification_type != "multiclass":
        raise ValueError("classification_type must be multiclass or multilabel")
    order = np.argsort(probabilities, axis=1)
    predicted = order[:, -1].astype(np.int64)
    top1 = probabilities[np.arange(len(probabilities)), order[:, -1]]
    top2 = probabilities[np.arange(len(probabilities)), order[:, -2]] if probabilities.shape[1] > 1 else np.zeros_like(top1)
    entropy = -(probabilities.astype(np.float64) * np.log(safe)).sum(axis=1)
    return {
        "predicted": predicted,
        "maximum": top1.astype(np.float32),
        "margin": (top1 - top2).astype(np.float32),
        "entropy": entropy.astype(np.float32),
    }


def _validate_inputs(
    sample_ids: Sequence[str],
    labels: np.ndarray,
    logits: np.ndarray,
    class_names: Sequence[str],
    classification_type: str,
) -> None:
    if logits.ndim != 2:
        raise ValueError(f"Deterministic logits must have shape [samples, classes], got {logits.shape}")
    expected_label_shape = (logits.shape[0],) if classification_type == "multiclass" else logits.shape
    if len(sample_ids) != logits.shape[0] or labels.shape != expected_label_shape:
        raise ValueError("sample_ids, labels, and logits have inconsistent sample dimensions")
    if logits.shape[1] != len(class_names):
        raise ValueError("Logit class dimension does not match class_names")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values must be unique and order-preserving")
    if not np.isfinite(logits).all():
        raise ValueError("Logits contain NaN or Inf")
    if classification_type == "multilabel" and not np.isin(labels, (0, 1)).all():
        raise ValueError("Multilabel targets must be binary multi-hot values")


def export_predictions(
    output_dir: str | Path,
    *,
    sample_ids: Sequence[str],
    true_labels: np.ndarray | torch.Tensor | Sequence[int],
    logits: np.ndarray | torch.Tensor,
    class_names: Sequence[str],
    model_name: str,
    dataset: str,
    adaptation_mode: str,
    seed: int,
    checkpoint: str,
    split: str,
    uq_method: str = "deterministic",
    corruption_type: str = "none",
    corruption_severity: int = 0,
    expected_count: Optional[int] = None,
    stochastic_logits: Optional[np.ndarray | torch.Tensor] = None,
    embeddings: Optional[np.ndarray | torch.Tensor] = None,
    classification_type: str = "multiclass",
    multilabel_threshold: float = 0.5,
    source: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    ids = [str(value) for value in sample_ids]
    labels = np.asarray(true_labels, dtype=np.int64)
    deterministic_logits = np.asarray(logits.detach().cpu() if isinstance(logits, torch.Tensor) else logits, dtype=np.float32)
    if classification_type not in {"multiclass", "multilabel"}:
        raise ValueError("classification_type must be multiclass or multilabel")
    if classification_type == "multilabel" and not 0.0 < float(multilabel_threshold) < 1.0:
        raise ValueError("multilabel_threshold must be strictly between zero and one")
    _validate_inputs(ids, labels, deterministic_logits, class_names, classification_type)
    probabilities = stable_probabilities(
        torch.from_numpy(deterministic_logits), classification_type=classification_type
    ).numpy().astype(np.float32)
    summaries = _prediction_summaries(probabilities, classification_type, multilabel_threshold)

    if embeddings is None:
        raise ValueError("Every final prediction export requires backbone representations")
    embedding_values = np.asarray(
        embeddings.detach().cpu() if isinstance(embeddings, torch.Tensor) else embeddings, dtype=np.float32
    )
    if embedding_values.ndim != 2 or embedding_values.shape[0] != len(ids):
        raise ValueError("backbone representations must have shape [samples, representation_dim]")
    if not np.isfinite(embedding_values).all():
        raise ValueError("Backbone representations contain NaN or Inf")

    if classification_type == "multiclass":
        correctness = summaries["predicted"] == labels
        label_column = labels
        prediction_column = summaries["predicted"]
    else:
        correctness = np.all(summaries["predicted"] == labels, axis=1)
        label_column = [row.tolist() for row in labels]
        prediction_column = [row.tolist() for row in summaries["predicted"]]

    table = pd.DataFrame(
        {
            "sample_id": ids,
            "label": label_column,
            "logits": [row.tolist() for row in deterministic_logits],
            "probabilities": [row.tolist() for row in probabilities],
            "prediction": prediction_column,
            "correctness": correctness,
            "confidence": summaries["maximum"],
            "predictive_entropy": summaries["entropy"],
            "backbone_representation": [row.tolist() for row in embedding_values],
            # Backward-compatible aliases retained for existing EuroSAT analysis code.
            "true_label": label_column,
            "predicted_label": prediction_column,
            "correct": correctness,
            "maximum_softmax_probability": summaries["maximum"],
            "top1_top2_probability_margin": summaries["margin"],
            "model_name": model_name,
            "dataset": dataset,
            "adaptation_mode": adaptation_mode,
            "seed": int(seed),
            "checkpoint": str(checkpoint),
            "split": split,
            "uq_method": uq_method,
            "corruption_type": corruption_type,
            "corruption_severity": int(corruption_severity),
        }
    )
    table_path = output_dir / MAIN_TABLE_NAME
    arrow_table = pa.Table.from_pandas(table, preserve_index=False)
    for column_name, values in (
        ("logits", deterministic_logits),
        ("probabilities", probabilities),
        ("backbone_representation", embedding_values),
    ):
        column_index = arrow_table.schema.get_field_index(column_name)
        float_lists = pa.array([row.tolist() for row in values], type=pa.list_(pa.float32()))
        arrow_table = arrow_table.set_column(column_index, column_name, float_lists)
    pq.write_table(arrow_table, table_path, compression="zstd")

    arrays: Dict[str, Any] = {
        "main_logits": {"file": MAIN_TABLE_NAME, "shape": list(deterministic_logits.shape), "dtype": "float32"},
        "main_probabilities": {"file": MAIN_TABLE_NAME, "shape": list(probabilities.shape), "dtype": "float32"},
    }
    stochastic_path = None
    if stochastic_logits is not None:
        raw_logits = np.asarray(
            stochastic_logits.detach().cpu() if isinstance(stochastic_logits, torch.Tensor) else stochastic_logits,
            dtype=np.float32,
        )
        if raw_logits.ndim != 3 or raw_logits.shape[0] != len(ids) or raw_logits.shape[2] != len(class_names):
            raise ValueError("stochastic_logits must have shape [samples, passes_or_members, classes]")
        raw_probabilities = stable_probabilities(
            torch.from_numpy(raw_logits), classification_type=classification_type
        ).numpy().astype(np.float32)
        stochastic_path = output_dir / STOCHASTIC_NAME
        np.savez_compressed(
            stochastic_path,
            sample_ids=np.asarray(ids),
            true_labels=labels,
            logits=raw_logits,
            probabilities=raw_probabilities,
        )
        arrays["stochastic_logits"] = {"file": STOCHASTIC_NAME, "shape": list(raw_logits.shape), "dtype": "float32"}
        arrays["stochastic_probabilities"] = {
            "file": STOCHASTIC_NAME,
            "shape": list(raw_probabilities.shape),
            "dtype": "float32",
            "axes": ["sample", "pass_or_member", "class"],
        }
        arrays["stochastic_logits"]["axes"] = ["sample", "pass_or_member", "class"]

    embeddings_path = output_dir / EMBEDDINGS_NAME
    np.savez_compressed(
        embeddings_path,
        sample_ids=np.asarray(ids),
        embeddings=embedding_values,
        backbone_representation=embedding_values,
    )
    arrays["backbone_representation"] = {
        "file": MAIN_TABLE_NAME,
        "redundant_file": EMBEDDINGS_NAME,
        "shape": list(embedding_values.shape),
        "dtype": "float32",
        "axes": ["sample", "feature"],
    }

    class_mapping = {str(index): name for index, name in enumerate(class_names)}
    source_values = source or {}
    split_total_count = source_values.get("split_total_count")
    partial_export = (
        len(ids) != int(split_total_count)
        if split_total_count is not None
        else expected_count is not None and len(ids) != expected_count
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "format": {"main_table": "parquet", "stochastic": "compressed_npz", "embeddings": "compressed_npz"},
        "sample_count": len(ids),
        "expected_count": int(expected_count if expected_count is not None else len(ids)),
        "partial_export": partial_export,
        "classification_type": classification_type,
        "multilabel_threshold": float(multilabel_threshold) if classification_type == "multilabel" else None,
        "class_names": list(class_names),
        "class_index_mapping": class_mapping,
        "arrays": arrays,
        "row_order": "Exactly the inference/DataLoader order; sample_id and every array share axis 0.",
        "prediction_semantics": (
            {
                "label": "binary multi-hot vector",
                "probabilities": "independent sigmoid probabilities",
                "prediction": "probability >= multilabel_threshold for each class",
                "correctness": "exact match across every class",
                "confidence": "mean probability assigned to each thresholded binary decision",
                "predictive_entropy": "sum of per-class Bernoulli entropies in nats",
            }
            if classification_type == "multilabel"
            else {
                "label": "single class index",
                "probabilities": "softmax probabilities",
                "prediction": "probability argmax",
                "correctness": "predicted class equals label",
                "confidence": "maximum softmax probability",
                "predictive_entropy": "categorical entropy in nats",
            }
        ),
        "source": {
            "model_name": model_name,
            "dataset": dataset,
            "adaptation_mode": adaptation_mode,
            "seed": int(seed),
            "checkpoint": str(checkpoint),
            "split": split,
            "uq_method": uq_method,
            "corruption_type": corruption_type,
            "corruption_severity": int(corruption_severity),
            **source_values,
        },
        "columns": {column: str(dtype) for column, dtype in table.dtypes.items()},
    }
    manifest_path = output_dir / MANIFEST_NAME
    _write_json(manifest_path, manifest)
    validation = validate_prediction_export(output_dir, expected_count=len(ids))
    validation_path = output_dir / VALIDATION_NAME
    _write_json(validation_path, validation)
    return {
        "table": table_path,
        "manifest": manifest_path,
        "validation": validation_path,
        **({"stochastic": stochastic_path} if stochastic_path else {}),
        "embeddings": embeddings_path,
    }


def export_stochastic_predictions(
    output_dir: str | Path,
    *,
    stochastic_logits: np.ndarray | torch.Tensor,
    **metadata: Any,
) -> Dict[str, Path]:
    """Export MC Dropout/ensemble members without discarding any raw member output."""
    raw_logits = torch.as_tensor(stochastic_logits).detach().float().cpu()
    if raw_logits.ndim != 3:
        raise ValueError("stochastic_logits must have shape [samples, passes_or_members, classes]")
    classification_type = str(metadata.get("classification_type", "multiclass"))
    member_probabilities = stable_probabilities(raw_logits, classification_type=classification_type)
    mean_probabilities = member_probabilities.mean(dim=1)
    if classification_type == "multiclass":
        aggregate_logits = mean_probabilities.clamp_min(torch.finfo(mean_probabilities.dtype).tiny).log()
    elif classification_type == "multilabel":
        eps = torch.finfo(mean_probabilities.dtype).eps
        clipped = mean_probabilities.clamp(eps, 1.0 - eps)
        aggregate_logits = torch.logit(clipped)
    else:
        raise ValueError("classification_type must be multiclass or multilabel")
    return export_predictions(
        output_dir,
        logits=aggregate_logits,
        stochastic_logits=raw_logits,
        **metadata,
    )


def load_prediction_export(output_dir: str | Path) -> PredictionBundle:
    output_dir = Path(output_dir)
    table = pd.read_parquet(output_dir / MAIN_TABLE_NAME, engine="pyarrow")
    manifest = json.loads((output_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    stochastic = None
    if (output_dir / STOCHASTIC_NAME).is_file():
        with np.load(output_dir / STOCHASTIC_NAME, allow_pickle=False) as archive:
            stochastic = {name: archive[name] for name in archive.files}
    embeddings = None
    if (output_dir / EMBEDDINGS_NAME).is_file():
        with np.load(output_dir / EMBEDDINGS_NAME, allow_pickle=False) as archive:
            embeddings = {name: archive[name] for name in archive.files}
    return PredictionBundle(table=table, manifest=manifest, stochastic=stochastic, embeddings=embeddings)


def validate_prediction_export(
    output_dir: str | Path, expected_count: Optional[int] = None, probability_tolerance: float = 1e-5
) -> Dict[str, Any]:
    bundle = load_prediction_export(output_dir)
    table = bundle.table
    schema_version = int(bundle.manifest.get("schema_version", 1))
    required_columns = (
        {
            "sample_id",
            "label",
            "logits",
            "probabilities",
            "prediction",
            "correctness",
            "confidence",
            "predictive_entropy",
            "backbone_representation",
        }
        if schema_version >= 2
        else {
            "sample_id",
            "true_label",
            "logits",
            "probabilities",
            "predicted_label",
            "correct",
            "maximum_softmax_probability",
            "predictive_entropy",
        }
    )
    errors: List[str] = []
    missing_columns = sorted(required_columns - set(table.columns))
    if missing_columns:
        errors.append(f"required prediction columns are missing: {missing_columns}")
        return {
            "valid": False,
            "errors": errors,
            "sample_count": len(table),
            "sample_ids_unique": not table["sample_id"].duplicated().any() if "sample_id" in table else False,
            "probability_sum_max_abs_error": None,
            "argmax_matches": None,
            "finite": False,
            "stochastic_shape": None,
            "embedding_shape": None,
        }
    classification_type = str(bundle.manifest.get("classification_type", "multiclass"))
    threshold = float(bundle.manifest.get("multilabel_threshold") or 0.5)
    label_column_name = "label" if "label" in table else "true_label"
    prediction_column_name = "prediction" if "prediction" in table else "predicted_label"
    correctness_column_name = "correctness" if "correctness" in table else "correct"
    confidence_column_name = "confidence" if "confidence" in table else "maximum_softmax_probability"
    logits = _to_matrix(table["logits"])
    probabilities = _to_matrix(table["probabilities"])
    labels = (
        table[label_column_name].to_numpy(dtype=np.int64)
        if classification_type == "multiclass"
        else _to_matrix(table[label_column_name], dtype=np.int64)
    )
    predictions = (
        table[prediction_column_name].to_numpy(dtype=np.int64)
        if classification_type == "multiclass"
        else _to_matrix(table[prediction_column_name], dtype=np.int64)
    )
    representations = _to_matrix(table["backbone_representation"]) if schema_version >= 2 else None
    if table["sample_id"].duplicated().any():
        errors.append("sample_id is not unique")
    target_count = expected_count if expected_count is not None else bundle.manifest.get("expected_count")
    if target_count is not None and len(table) != int(target_count):
        errors.append(f"sample count {len(table)} != expected {target_count}")
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        errors.append("main arrays contain NaN or Inf")
    probabilities_from_logits = stable_probabilities(
        torch.from_numpy(logits), classification_type=classification_type
    ).numpy()
    if not np.allclose(probabilities, probabilities_from_logits, atol=probability_tolerance, rtol=0):
        errors.append("saved probabilities do not match the configured logits transform")
    probability_sums = probabilities.sum(axis=1)
    if classification_type == "multiclass":
        if not np.allclose(probability_sums, 1.0, atol=probability_tolerance, rtol=0):
            errors.append("probability rows do not sum to one")
    elif not np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all():
        errors.append("multilabel probabilities are outside [0, 1]")
    summaries = _prediction_summaries(probabilities, classification_type, threshold)
    if not np.array_equal(summaries["predicted"], predictions):
        errors.append("probabilities do not match prediction")
    if not np.allclose(summaries["maximum"], table[confidence_column_name], atol=probability_tolerance):
        errors.append("confidence is inconsistent")
    if "top1_top2_probability_margin" in table and not np.allclose(
        summaries["margin"], table["top1_top2_probability_margin"], atol=probability_tolerance
    ):
        errors.append("top1_top2_probability_margin is inconsistent")
    if not np.allclose(summaries["entropy"], table["predictive_entropy"], atol=probability_tolerance):
        errors.append("predictive_entropy is inconsistent")
    expected_correctness = predictions == labels
    if classification_type == "multilabel":
        expected_correctness = expected_correctness.all(axis=1)
    if not np.array_equal(table[correctness_column_name].to_numpy(dtype=bool), expected_correctness):
        errors.append("correctness is inconsistent")
    if representations is not None and not np.isfinite(representations).all():
        errors.append("backbone representations contain NaN or Inf")
    if bundle.stochastic is not None:
        raw_logits = bundle.stochastic["logits"]
        raw_probabilities = bundle.stochastic["probabilities"]
        if raw_logits.ndim != 3 or raw_probabilities.shape != raw_logits.shape:
            errors.append("stochastic arrays do not have matching [samples, passes_or_members, classes] shapes")
        if raw_logits.shape[0] != len(table):
            errors.append("stochastic sample dimension does not match main table")
        if not np.isfinite(raw_logits).all() or not np.isfinite(raw_probabilities).all():
            errors.append("stochastic arrays contain NaN or Inf")
        if classification_type == "multiclass" and not np.allclose(
            raw_probabilities.sum(axis=-1), 1.0, atol=probability_tolerance, rtol=0
        ):
            errors.append("stochastic probability rows do not sum to one")
        if not np.array_equal(bundle.stochastic["sample_ids"].astype(str), table["sample_id"].astype(str).to_numpy()):
            errors.append("stochastic sample_id order differs from main table")
        if not np.allclose(raw_probabilities.mean(axis=1), probabilities, atol=probability_tolerance, rtol=0):
            errors.append("main probabilities do not equal mean stochastic probabilities")
    if bundle.embeddings is not None:
        if bundle.embeddings["embeddings"].shape[0] != len(table):
            errors.append("embedding sample dimension does not match main table")
        if not np.array_equal(bundle.embeddings["sample_ids"].astype(str), table["sample_id"].astype(str).to_numpy()):
            errors.append("embedding sample_id order differs from main table")
        if not np.isfinite(bundle.embeddings["embeddings"]).all():
            errors.append("embeddings contain NaN or Inf")
        if representations is not None and not np.allclose(
            bundle.embeddings["embeddings"], representations, atol=0.0, rtol=0.0
        ):
            errors.append("backbone representation column differs from embeddings.npz")
    elif schema_version >= 2:
        errors.append("backbone representation sidecar is missing")
    return {
        "valid": not errors,
        "errors": errors,
        "sample_count": len(table),
        "sample_ids_unique": not table["sample_id"].duplicated().any(),
        "classification_type": classification_type,
        "schema_version": schema_version,
        "probability_sum_max_abs_error": (
            float(np.max(np.abs(probability_sums - 1.0)))
            if len(table) and classification_type == "multiclass"
            else None
        ),
        "argmax_matches": (
            "probabilities do not match prediction" not in errors if classification_type == "multiclass" else None
        ),
        "finite": not any("NaN or Inf" in error for error in errors),
        "stochastic_shape": list(bundle.stochastic["logits"].shape) if bundle.stochastic is not None else None,
        "embedding_shape": list(bundle.embeddings["embeddings"].shape) if bundle.embeddings is not None else None,
    }


def recompute_metrics(output_dir: str | Path, n_bins: int = 15) -> Dict[str, Any]:
    bundle = load_prediction_export(output_dir)
    table = bundle.table
    probabilities = _to_matrix(table["probabilities"], dtype=np.float64)
    classification_type = str(bundle.manifest.get("classification_type", "multiclass"))
    threshold = float(bundle.manifest.get("multilabel_threshold") or 0.5)
    label_column_name = "label" if "label" in table else "true_label"
    if classification_type == "multilabel":
        labels = _to_matrix(table[label_column_name], dtype=np.int64)
        predictions = (probabilities >= threshold).astype(np.int64)
        accuracy = float(np.mean(np.all(predictions == labels, axis=1)))
        clipped = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0 - np.finfo(np.float32).eps)
        nll = float(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean())
        brier = float(np.square(probabilities - labels).mean())
        decision_confidence = np.maximum(probabilities, 1.0 - probabilities).reshape(-1)
        decision_correct = (predictions == labels).reshape(-1)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (decision_confidence >= lower if index == 0 else decision_confidence > lower) & (
                decision_confidence <= upper
            )
            if mask.any():
                ece += abs(
                    float(decision_correct[mask].mean()) - float(decision_confidence[mask].mean())
                ) * float(mask.mean())
        per_class = []
        confusion = []
        for class_index, class_name in enumerate(bundle.manifest["class_names"]):
            target = labels[:, class_index]
            prediction = predictions[:, class_index]
            tp = int(np.sum((target == 1) & (prediction == 1)))
            fp = int(np.sum((target == 0) & (prediction == 1)))
            fn = int(np.sum((target == 1) & (prediction == 0)))
            tn = int(np.sum((target == 0) & (prediction == 0)))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            confusion.append([[tn, fp], [fn, tp]])
            per_class.append(
                {
                    "class_index": class_index,
                    "class_name": class_name,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "support": int(target.sum()),
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
            )
        entropy = -(
            probabilities * np.log(clipped) + (1.0 - probabilities) * np.log(1.0 - clipped)
        ).sum(axis=1)
        return {
            "classification_type": classification_type,
            "accuracy": accuracy,
            "labelwise_accuracy": float(np.mean(predictions == labels)),
            "macro_f1": float(np.mean([item["f1"] for item in per_class])),
            "nll": nll,
            "brier": brier,
            "ece": ece,
            "mean_confidence": float(decision_confidence.mean()),
            "mean_predictive_entropy": float(entropy.mean()),
            "per_class": per_class,
            "confusion_matrix": confusion,
        }

    labels = table[label_column_name].to_numpy(dtype=np.int64)
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
    confusion = np.zeros((probabilities.shape[1], probabilities.shape[1]), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    per_class = []
    for class_index in range(probabilities.shape[1]):
        true_positive = int(confusion[class_index, class_index])
        predicted_count = int(confusion[:, class_index].sum())
        support = int(confusion[class_index, :].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        denominator = precision + recall
        f1 = 2 * precision * recall / denominator if denominator else 0.0
        per_class.append(
            {
                "class_index": class_index,
                "class_name": bundle.manifest["class_names"][class_index],
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        )
    safe_probabilities = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)
    entropy = -(probabilities * np.log(safe_probabilities)).sum(axis=1)
    return {
        "classification_type": classification_type,
        "accuracy": accuracy,
        "macro_f1": float(np.mean([item["f1"] for item in per_class])),
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "mean_confidence": float(confidence.mean()),
        "mean_predictive_entropy": float(entropy.mean()),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def save_evaluation_artifacts(output_dir: str | Path, n_bins: int = 15) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    metrics = recompute_metrics(output_dir, n_bins=n_bins)
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    class_names = load_prediction_export(output_dir).manifest["class_names"]
    np.save(output_dir / "confusion_matrix.npy", confusion)
    if metrics["classification_type"] == "multilabel":
        pd.DataFrame(
            [
                {
                    "class_index": item["class_index"],
                    "class_name": item["class_name"],
                    "tn": item["tn"],
                    "fp": item["fp"],
                    "fn": item["fn"],
                    "tp": item["tp"],
                }
                for item in metrics["per_class"]
            ]
        ).to_csv(output_dir / "confusion_matrix.csv", index=False)
    else:
        pd.DataFrame(confusion, index=class_names, columns=class_names).to_csv(output_dir / "confusion_matrix.csv")
    _write_json(output_dir / "per_class_metrics.json", {"per_class": metrics["per_class"]})
    _write_json(output_dir / "metrics.json", metrics)
    return metrics


@torch.no_grad()
def collect_deterministic_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    limit_batches: Optional[int] = None,
    embedding_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, Any]:
    model.eval()
    sample_ids: List[str] = []
    labels: List[torch.Tensor] = []
    logits: List[torch.Tensor] = []
    embeddings: List[torch.Tensor] = []
    for batch_index, batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        if "sample_id" not in batch:
            raise ValueError("Prediction export requires sample_id in every DataLoader batch")
        images = batch["image"].to(device)
        sample_ids.extend(str(value) for value in batch["sample_id"])
        labels.append(batch["label"].cpu())
        temporal_mask = batch.get("temporal_mask")
        if temporal_mask is not None:
            temporal_mask = temporal_mask.to(device)
            logits.append(model(images, temporal_mask=temporal_mask).cpu())
        else:
            logits.append(model(images).cpu())
        if embedding_extractor is not None:
            if temporal_mask is not None:
                embeddings.append(embedding_extractor(images, temporal_mask=temporal_mask).detach().cpu())
            else:
                embeddings.append(embedding_extractor(images).detach().cpu())
    if not logits:
        raise ValueError("Prediction loader produced no batches")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Prediction DataLoader produced duplicate sample_id values")
    return {
        "sample_ids": sample_ids,
        "labels": torch.cat(labels),
        "logits": torch.cat(logits),
        "embeddings": torch.cat(embeddings) if embeddings else None,
    }


def classification_embedding_extractor(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the common feature-extraction hook exposed by classifier wrappers."""
    extractor = getattr(model, "extract_features", None)
    if extractor is None or not callable(extractor):
        raise ValueError("Configured embedding export requires a classifier with extract_features(images)")
    return extractor


def dofa_embedding_extractor(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Backward-compatible alias for historical callers."""
    return classification_embedding_extractor(model)


@torch.no_grad()
def _collect_prediction_pass(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, limit_batches: Optional[int]
) -> Dict[str, Any]:
    sample_ids: List[str] = []
    labels: List[torch.Tensor] = []
    logits: List[torch.Tensor] = []
    for batch_index, batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        if "sample_id" not in batch:
            raise ValueError("Stochastic export requires sample_id in every DataLoader batch")
        sample_ids.extend(str(value) for value in batch["sample_id"])
        labels.append(batch["label"].cpu())
        images = batch["image"].to(device)
        temporal_mask = batch.get("temporal_mask")
        if temporal_mask is not None:
            logits.append(model(images, temporal_mask=temporal_mask.to(device)).cpu())
        else:
            logits.append(model(images).cpu())
    return {"sample_ids": sample_ids, "labels": torch.cat(labels), "logits": torch.cat(logits)}


@torch.no_grad()
def collect_mc_dropout_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    passes: int,
    limit_batches: Optional[int] = None,
) -> Dict[str, Any]:
    activation_audit = activate_downstream_mc_dropout(model)
    outputs = [_collect_prediction_pass(model, loader, device, limit_batches) for _ in range(passes)]
    reference_ids = outputs[0]["sample_ids"]
    reference_labels = outputs[0]["labels"]
    if any(output["sample_ids"] != reference_ids for output in outputs[1:]):
        raise ValueError("MC Dropout DataLoader order changed between passes")
    if any(not torch.equal(output["labels"], reference_labels) for output in outputs[1:]):
        raise ValueError("MC Dropout labels changed between passes")
    return {
        "sample_ids": reference_ids,
        "labels": reference_labels,
        "stochastic_logits": torch.stack([output["logits"] for output in outputs], dim=1),
        "activation_audit": activation_audit,
    }


@torch.no_grad()
def collect_deep_ensemble_predictions(
    models: Iterable[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    limit_batches: Optional[int] = None,
) -> Dict[str, Any]:
    outputs = []
    for model in models:
        model.eval()
        outputs.append(_collect_prediction_pass(model, loader, device, limit_batches))
    if not outputs:
        raise ValueError("Deep ensemble requires at least one model")
    reference_ids = outputs[0]["sample_ids"]
    reference_labels = outputs[0]["labels"]
    if any(output["sample_ids"] != reference_ids for output in outputs[1:]):
        raise ValueError("Deep ensemble DataLoader order changed between members")
    if any(not torch.equal(output["labels"], reference_labels) for output in outputs[1:]):
        raise ValueError("Deep ensemble labels changed between members")
    return {
        "sample_ids": reference_ids,
        "labels": reference_labels,
        "stochastic_logits": torch.stack([output["logits"] for output in outputs], dim=1),
    }
