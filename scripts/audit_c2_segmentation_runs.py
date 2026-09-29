from __future__ import annotations

"""Conservative promotion audit for the 24 deterministic C2 segmentation runs."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml

from scripts.experiment_manager import collect_code_snapshot
from scripts.segmentation_pipeline import SegmentationMetricAccumulator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    ("cloudsen12", "dofa", "frozen", "cloudsen12_frozen"),
    ("cloudsen12", "panopticon", "frozen", "cloudsen12_frozen"),
    ("cloudsen12", "dofa", "full_finetune", "cloudsen12_full_finetune"),
    ("cloudsen12", "panopticon", "full_finetune", "cloudsen12_full_finetune"),
    ("spacenet7", "dofa", "frozen", "spacenet7_frozen"),
    ("spacenet7", "panopticon", "frozen", "spacenet7_frozen"),
    ("spacenet7", "dofa", "full_finetune", "spacenet7_full_finetune"),
    ("spacenet7", "panopticon", "full_finetune", "spacenet7_full_finetune"),
)
EXPECTED_COUNTS = {"cloudsen12": 975, "spacenet7": 1152}
EXPECTED_CLASSES = {
    "cloudsen12": ["clear", "thick cloud", "thin cloud", "cloud shadow"],
    "spacenet7": ["background", "building"],
}
EXPECTED_IGNORE = {"cloudsen12": None, "spacenet7": 255}
AUDITOR_CODE_PATH = "scripts/audit_c2_segmentation_runs.py"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


@lru_cache(maxsize=16)
def _sha256(path_text: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _finite_scalars(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_scalars(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_scalars(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _snapshot_file_map(snapshot: Dict[str, Any]) -> Dict[str, str]:
    """Return the path-to-hash mapping from a code snapshot.

    A snapshot comparison must use the recorded file entries, rather than only
    comparing aggregate hashes, so that a narrowly allowed auditor-only change
    cannot conceal any second code/config change.
    """

    entries = snapshot.get("files")
    if not isinstance(entries, list):
        raise ValueError("snapshot files is not a list")
    result: Dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(
            entry.get("sha256"), str
        ):
            raise ValueError("snapshot contains an invalid file entry")
        path = entry["path"]
        if path in result:
            raise ValueError(f"snapshot repeats path: {path}")
        result[path] = entry["sha256"]
    if not result:
        raise ValueError("snapshot has no file entries")
    return result


def _snapshot_changed_paths(left: Dict[str, Any], right: Dict[str, Any]) -> set[str]:
    left_files = _snapshot_file_map(left)
    right_files = _snapshot_file_map(right)
    return {path for path in set(left_files) | set(right_files) if left_files.get(path) != right_files.get(path)}


def _compare_live_code_snapshot(expected: Dict[str, Any], live: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Allow an identical live tree or one content change to this auditor only.

    The auditor must remain runnable after it is hardened, while every training
    path and config remains byte-for-byte bound to the latest run/resume
    snapshot. Additions/removals are not accepted as an auditor self-change.

    >>> base = {"files": [{"path": AUDITOR_CODE_PATH, "sha256": "a"}, {"path": "scripts/train.py", "sha256": "b"}]}
    >>> _compare_live_code_snapshot(base, base)
    (True, [])
    >>> auditor_edit = {"files": [{"path": AUDITOR_CODE_PATH, "sha256": "c"}, {"path": "scripts/train.py", "sha256": "b"}]}
    >>> _compare_live_code_snapshot(base, auditor_edit)
    (True, ['scripts/audit_c2_segmentation_runs.py'])
    >>> training_edit = {"files": [{"path": AUDITOR_CODE_PATH, "sha256": "c"}, {"path": "scripts/train.py", "sha256": "d"}]}
    >>> _compare_live_code_snapshot(base, training_edit)
    (False, ['scripts/audit_c2_segmentation_runs.py', 'scripts/train.py'])
    """

    expected_files = _snapshot_file_map(expected)
    live_files = _snapshot_file_map(live)
    changed = sorted(
        path for path in set(expected_files) | set(live_files) if expected_files.get(path) != live_files.get(path)
    )
    if not changed:
        return True, changed
    auditor_content_edit = (
        changed == [AUDITOR_CODE_PATH]
        and AUDITOR_CODE_PATH in expected_files
        and AUDITOR_CODE_PATH in live_files
    )
    return auditor_content_edit, changed


def _attempt_disposition(
    summary_status: Optional[str],
    completed_epochs: int,
    has_checkpoint: bool,
    has_gradient_audit: bool,
    has_initialization_audit: bool,
) -> str:
    """Classify discovery evidence without promoting abandoned directories.

    >>> _attempt_disposition("completed", 2, True, True, True)
    'COMPLETED_CANDIDATE'
    >>> _attempt_disposition(None, 0, False, False, True)
    'NONCANDIDATE_ZERO_EPOCH_INITIALIZATION'
    >>> _attempt_disposition(None, 0, False, False, False)
    'NONCANDIDATE_METADATA_ONLY'
    >>> _attempt_disposition(None, 1, True, True, True)
    'NONCANDIDATE_INCOMPLETE_TRAINING'
    """

    if summary_status == "completed":
        return "COMPLETED_CANDIDATE"
    if completed_epochs > 0 or has_checkpoint or has_gradient_audit:
        return "NONCANDIDATE_INCOMPLETE_TRAINING"
    if has_initialization_audit:
        return "NONCANDIDATE_ZERO_EPOCH_INITIALIZATION"
    return "NONCANDIDATE_METADATA_ONLY"


def inspect_candidate_attempt(run_dir: Path) -> Dict[str, Any]:
    """Record why a discovered directory is or is not a promotion candidate."""

    summary_status: Optional[str] = None
    summary_error: Optional[str] = None
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        try:
            summary = _load_json(summary_path)
            if isinstance(summary, dict) and isinstance(summary.get("status"), str):
                summary_status = summary["status"]
            else:
                summary_error = "run_summary.json has no string status"
        except Exception as exc:
            summary_error = f"run_summary.json is unreadable: {type(exc).__name__}: {exc}"

    completed_epochs = 0
    history_error: Optional[str] = None
    history_path = run_dir / "training_history.json"
    if history_path.is_file():
        try:
            history = _load_json(history_path)
            if isinstance(history, list):
                completed_epochs = len(history)
            else:
                history_error = "training_history.json is not a list"
        except Exception as exc:
            history_error = f"training_history.json is unreadable: {type(exc).__name__}: {exc}"

    checkpoint_names = [name for name in ("best.pt", "last.pt") if (run_dir / name).is_file()]
    has_gradient_audit = (run_dir / "gradient_audit.json").is_file()
    initialization_audits = [
        name for name in ("model_audit.json", "optimizer_group_audit.json") if (run_dir / name).is_file()
    ]
    disposition = _attempt_disposition(
        summary_status,
        completed_epochs,
        bool(checkpoint_names),
        has_gradient_audit,
        bool(initialization_audits),
    )
    if disposition == "COMPLETED_CANDIDATE":
        reason = "run_summary.json records status=completed"
    elif summary_status is not None:
        reason = f"run_summary.json records non-completed status={summary_status!r}"
    elif disposition == "NONCANDIDATE_INCOMPLETE_TRAINING":
        reason = "training evidence exists, but no completed run summary exists"
    elif disposition == "NONCANDIDATE_ZERO_EPOCH_INITIALIZATION":
        reason = "initialization audits exist, but no completed epoch, checkpoint, gradient audit, or run summary exists"
    else:
        reason = "only run metadata exists; no initialization or training-completion evidence exists"
    diagnostics = [value for value in (summary_error, history_error) if value is not None]
    if diagnostics:
        reason += "; " + "; ".join(diagnostics)
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "disposition": disposition,
        "reason": reason,
        "evidence": {
            "summary_present": summary_path.is_file(),
            "summary_status": summary_status,
            "completed_epochs": completed_epochs,
            "checkpoints": checkpoint_names,
            "gradient_audit_present": has_gradient_audit,
            "initialization_audits": initialization_audits,
        },
    }


def _official_test_ids(dataset: str) -> set[str]:
    path = PROJECT_ROOT / "reports" / "dataset_manifests" / f"{dataset}_actual_manifest.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["sample_id"] for row in csv.DictReader(handle) if row["split"] == "test"}


def _confusion_difference_is_one_pixel(
    expected: Any,
    actual: Any,
) -> bool:
    """Return whether two confusion matrices differ by one prediction move.

    The final online accumulator runs on CUDA while the lossless float32 export
    is validated on CPU. A one-ULP class-boundary tie can therefore move one
    pixel between prediction columns without changing any target-row total.

    >>> _confusion_difference_is_one_pixel([[4, 1], [2, 3]], [[3, 2], [2, 3]])
    True
    >>> _confusion_difference_is_one_pixel([[4, 1], [2, 3]], [[2, 3], [2, 3]])
    False
    >>> _confusion_difference_is_one_pixel([[4, 1], [2, 3]], [[4, 1], [2, 3]])
    False
    """

    try:
        expected_array = np.asarray(expected, dtype=np.int64)
        actual_array = np.asarray(actual, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        expected_array.ndim != 2
        or expected_array.shape != actual_array.shape
        or expected_array.shape[0] != expected_array.shape[1]
        or np.array_equal(expected_array, actual_array)
    ):
        return False
    if not np.array_equal(expected_array.sum(axis=1), actual_array.sum(axis=1)):
        return False
    return int(np.abs(expected_array - actual_array).sum()) == 2


def _compare_metrics(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    errors: List[str],
    prefix: str,
    warnings: Optional[List[str]] = None,
) -> None:
    scalar_names = ["miou", "pixel_accuracy", "nll", "brier", "ece_15"]
    if "foreground_nll" in expected:
        scalar_names += ["foreground_ece_15", "foreground_nll", "foreground_brier"]
    for name in scalar_names:
        if name not in expected or name not in actual or not math.isclose(
            float(expected[name]), float(actual[name]), rel_tol=2e-6, abs_tol=2e-7
        ):
            errors.append(f"{prefix} metric mismatch: {name}")
    expected_confusion = expected.get("confusion_matrix")
    actual_confusion = actual.get("confusion_matrix")
    if expected_confusion != actual_confusion:
        if _confusion_difference_is_one_pixel(expected_confusion, actual_confusion):
            if warnings is not None:
                warnings.append(
                    f"{prefix} confusion matrix has one tolerated cross-device boundary-pixel move"
                )
        else:
            errors.append(f"{prefix} confusion matrix mismatch")
    for name, value in expected.get("per_class_iou", {}).items():
        observed = actual.get("per_class_iou", {}).get(name)
        if value is None or observed is None:
            if value != observed:
                errors.append(f"{prefix} per-class IoU mismatch: {name}")
        elif not math.isclose(float(value), float(observed), rel_tol=2e-6, abs_tol=2e-7):
            errors.append(f"{prefix} per-class IoU mismatch: {name}")


def deep_recompute_export(dataset: str, export_dir: Path) -> Dict[str, Any]:
    class_names = EXPECTED_CLASSES[dataset]
    ignore_index = EXPECTED_IGNORE[dataset]
    foreground = 1 if dataset == "spacenet7" else None
    accumulator = SegmentationMetricAccumulator(
        len(class_names), class_names, ignore_index, 15, foreground, 1
    )
    with np.load(export_dir / "predictions.npz", allow_pickle=False) as archive:
        logits = archive["logits"]
        labels = archive["label"]
        for start in range(0, logits.shape[0], 8):
            accumulator.update(
                torch.from_numpy(logits[start : start + 8]),
                torch.from_numpy(labels[start : start + 8]),
            )
    return accumulator.compute()


def audit_run(run_dir: Path, dataset: str, model: str, adaptation: str, seed: int, deep: bool) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    required = [
        "resolved_config.yaml", "code_snapshot.json", "environment.json", "input_artifacts.json",
        "model_audit.json", "optimizer_group_audit.json", "gradient_audit.json", "training_history.json",
        "best.pt", "last.pt", "segmentation_metrics.json", "results.json", "run_summary.json",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        return {"status": "BLOCKED", "run_dir": str(run_dir), "errors": [f"missing artifacts: {missing}"]}
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    summary = _load_json(run_dir / "run_summary.json")
    history = _load_json(run_dir / "training_history.json")
    metrics = _load_json(run_dir / "segmentation_metrics.json")
    experiment = next((item for item in config.get("experiments", []) if item["name"] == config.get("active_experiment")), None)
    if experiment is None:
        errors.append("active experiment is absent from resolved config")
    else:
        if experiment["model"].get("name") != model or experiment["model"].get("adaptation_mode") != adaptation:
            errors.append("model/adaptation identity mismatch")
    if config.get("data", {}).get("name") != dataset or int(config.get("seed", -1)) != seed:
        errors.append("dataset/seed identity mismatch")
    training = config.get("training", {})
    expected_patience = 10 if adaptation == "frozen" else 15
    protocol_checks = {
        "epochs": 50, "batch_size": 8, "weight_decay": 0.01,
        "head_learning_rate": 1e-3, "backbone_learning_rate": 1e-4,
    }
    for name, expected in protocol_checks.items():
        if training.get(name) != expected:
            errors.append(f"protocol mismatch: training.{name}")
    if training.get("early_stopping") != {"enabled": True, "patience": expected_patience, "min_delta": 0.0}:
        errors.append("early-stopping protocol mismatch")
    if training.get("checkpoint") != {"split": "val", "metric": "miou", "mode": "max"}:
        errors.append("checkpoint-selection protocol mismatch")
    if bool(config.get("dry_run")) or summary.get("status") != "completed":
        errors.append("run is dry or incomplete")

    snapshot = _load_json(run_dir / "code_snapshot.json")
    environment = _load_json(run_dir / "environment.json")
    if environment.get("code_version") != f"sha256:{snapshot.get('code_sha256')}":
        errors.append("environment/code snapshot mismatch")
    resume_event_paths = sorted((run_dir / "resume_events").glob("*/event.json"))
    latest_provenance_snapshot = snapshot
    allowed_resume_changes = {
        ".devcontainer/devcontainer.json",
        "scripts/run_experiments.py",
        AUDITOR_CODE_PATH,
    }
    for event_path in resume_event_paths:
        event = _load_json(event_path)
        event_dir = event_path.parent
        resume_snapshot_path = event_dir / "code_snapshot.json"
        archived_checkpoint = event_dir / "source_last.pt"
        if event.get("run_id") != run_dir.name or event.get("test_access") is not False:
            errors.append(f"invalid resume event identity/test-access flag: {event_path}")
        if event.get("original_code_sha256") != snapshot.get("code_sha256"):
            errors.append(f"resume event does not bind original snapshot: {event_path}")
        changed = event.get("code_changes", {})
        changed_paths = set(changed.get("changed", [])) | set(changed.get("added", [])) | set(changed.get("removed", []))
        if changed_paths - allowed_resume_changes or event.get("scientific_training_path_changed") is not False:
            errors.append(f"resume event contains unapproved code changes: {event_path}")
        resume_snapshot: Dict[str, Any] = {}
        if resume_snapshot_path.is_file():
            resume_snapshot = _load_json(resume_snapshot_path)
        if not resume_snapshot or resume_snapshot.get("code_sha256") != event.get("resume_code_sha256"):
            errors.append(f"resume code snapshot is missing or inconsistent: {event_path}")
        else:
            try:
                actual_resume_changes = _snapshot_changed_paths(snapshot, resume_snapshot)
            except ValueError as exc:
                errors.append(f"resume code snapshot file entries are invalid: {event_path}: {exc}")
            else:
                if actual_resume_changes - allowed_resume_changes:
                    errors.append(f"resume snapshot contains unapproved code changes: {event_path}")
                if actual_resume_changes != changed_paths:
                    errors.append(f"resume event code_changes do not match its archived snapshots: {event_path}")
            latest_provenance_snapshot = resume_snapshot
        if not archived_checkpoint.is_file() or file_sha256(archived_checkpoint) != event.get("checkpoint_sha256"):
            errors.append(f"archived resume checkpoint is missing or inconsistent: {event_path}")
    live_snapshot = collect_code_snapshot()
    try:
        live_compatible, live_changed_paths = _compare_live_code_snapshot(latest_provenance_snapshot, live_snapshot)
    except ValueError as exc:
        errors.append(f"live/latest code snapshot file entries are invalid: {exc}")
    else:
        if not live_compatible:
            errors.append(
                "live code snapshot differs from the run's latest provenance state outside the auditor itself: "
                f"{live_changed_paths}"
            )
    artifacts = _load_json(run_dir / "input_artifacts.json")
    if not artifacts.get("required") or not artifacts.get("all_verified"):
        errors.append("input artifacts were not all verified")
    for record in artifacts.get("records", {}).values():
        path = Path(record["path"])
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            errors.append(f"live input artifact hash mismatch: {record.get('label')}")

    if not history or [int(item.get("epoch", -1)) for item in history] != list(range(1, len(history) + 1)):
        errors.append("training history is not a contiguous non-empty prefix")
    elif not all(_finite_scalars(item) for item in history):
        errors.append("training history contains non-finite values")
    else:
        values = [float(item["val"]["miou"]) for item in history]
        best_value = max(values)
        best_epoch = values.index(best_value) + 1
        if int(summary.get("best_epoch", -1)) != best_epoch or not math.isclose(
            float(summary.get("best_value", math.nan)), best_value, rel_tol=0, abs_tol=1e-12
        ):
            errors.append("recorded best checkpoint is not earliest strict validation-mIoU maximum")
        if len(history) < 50 and len(history) != best_epoch + expected_patience:
            errors.append("early stopping did not occur at exact patience boundary")
        if int(summary.get("last_epoch", -1)) != len(history):
            errors.append("summary last epoch disagrees with history")

    model_audit = _load_json(run_dir / "model_audit.json")
    gradient = _load_json(run_dir / "gradient_audit.json")
    if model_audit.get("adaptation_mode") != adaptation:
        errors.append("model audit adaptation mismatch")
    backbone_grad = gradient.get("groups", {}).get("backbone", {})
    head_grad = gradient.get("groups", {}).get("head", {})
    if not head_grad.get("all_gradients_finite") or not head_grad.get("any_nonzero_gradient"):
        errors.append("decoder gradient audit failed")
    if adaptation == "frozen":
        if model_audit.get("backbone_trainable_parameters") != 0 or backbone_grad.get("gradient_parameter_tensors") != 0:
            errors.append("frozen backbone trainability/gradient invariant failed")
    elif model_audit.get("backbone_trainable_parameters", 0) <= 0 or not backbone_grad.get("any_nonzero_gradient"):
        errors.append("full-finetune backbone gradient invariant failed")

    best = _load_checkpoint(run_dir / "best.pt")
    last = _load_checkpoint(run_dir / "last.pt")
    if best.get("run_id") != run_dir.name or last.get("run_id") != run_dir.name:
        errors.append("checkpoint run identity mismatch")
    if int(best.get("epoch", -1)) != int(summary.get("best_epoch", -2)):
        errors.append("best checkpoint epoch mismatch")
    if int(last.get("epoch", -1)) != len(history):
        errors.append("last checkpoint epoch mismatch")
    if file_sha256(run_dir / "best.pt") != summary.get("best_checkpoint_sha256"):
        errors.append("best checkpoint SHA mismatch")
    if file_sha256(run_dir / "last.pt") != summary.get("last_checkpoint_sha256"):
        errors.append("last checkpoint SHA mismatch")
    for checkpoint_name, checkpoint in (("best", best), ("last", last)):
        if not all(bool(torch.isfinite(tensor).all()) for tensor in checkpoint["model"].values() if tensor.is_floating_point()):
            errors.append(f"{checkpoint_name} checkpoint contains non-finite model tensors")

    required_metrics = {"miou", "per_class_iou", "pixel_accuracy", "nll", "brier", "ece_15"}
    if not required_metrics.issubset(metrics.get("test", {})) or not _finite_scalars(metrics):
        errors.append("full finite segmentation metrics are missing")
    if dataset == "spacenet7":
        extras = {"foreground_ece_15", "classwise_calibration", "foreground_nll", "foreground_brier", "boundary_calibration"}
        if not extras.issubset(metrics.get("test", {})):
            errors.append("SpaceNet7 foreground/classwise/boundary metrics are missing")

    export_dir = run_dir / "predictions" / "test" / "deterministic"
    export_required = ["predictions.npz", "representations.npz", "per_image_metrics.csv", "manifest.json", "validation_report.json"]
    absent = [name for name in export_required if not (export_dir / name).is_file()]
    if absent:
        errors.append(f"test prediction export is incomplete: {absent}")
    else:
        manifest = _load_json(export_dir / "manifest.json")
        validation = _load_json(export_dir / "validation_report.json")
        expected_count = EXPECTED_COUNTS[dataset]
        if manifest.get("sample_count") != expected_count or not validation.get("valid"):
            errors.append("prediction manifest/validation count or validity mismatch")
        if manifest.get("arrays", {}).get("logits") != [expected_count, len(EXPECTED_CLASSES[dataset]), 224, 224]:
            errors.append("prediction logits shape mismatch")
        if manifest.get("representations", {}).get("shape") != [expected_count, 768]:
            errors.append("representation shape mismatch")
        with np.load(export_dir / "predictions.npz", allow_pickle=False) as archive:
            if set(archive.files) != {
                "sample_id", "label", "logits", "probabilities", "prediction", "correctness",
                "confidence", "predictive_entropy", "valid_mask",
            }:
                errors.append("prediction bundle field set mismatch")
            ids = archive["sample_id"].astype(str).tolist()
        if len(ids) != expected_count or set(ids) != _official_test_ids(dataset):
            errors.append("prediction sample IDs do not exactly cover official test split")
        with np.load(export_dir / "representations.npz", allow_pickle=False) as archive:
            representation_ids = archive["sample_id"].astype(str).tolist()
            representations = archive["representation"]
        if representation_ids != ids or representations.shape != (expected_count, 768) or not np.isfinite(representations).all():
            errors.append("representation values or alignment are invalid")
        with (export_dir / "per_image_metrics.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if [row.get("sample_id") for row in rows] != ids:
            errors.append("per-image metric rows are not aligned with predictions")
        if deep:
            recomputed = deep_recompute_export(dataset, export_dir)
            _compare_metrics(metrics["test"], recomputed, errors, "deep export", warnings)

    return {
        "status": "PROMOTABLE" if not errors else "BLOCKED",
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "dataset": dataset,
        "model": model,
        "adaptation": adaptation,
        "seed": seed,
        "best_epoch": summary.get("best_epoch"),
        "test_metrics": metrics.get("test"),
        "errors": errors,
        "warnings": warnings,
    }


def discover(dataset: str, model: str, adaptation: str, root_name: str, seed: int) -> List[Path]:
    root = PROJECT_ROOT / "results" / "final_thesis" / "segmentation" / root_name / "runs"
    if not root.is_dir():
        return []
    name = f"{dataset}_{model}_{adaptation}"
    matches = []
    for path in sorted(root.iterdir()):
        resolved = path / "resolved_config.yaml"
        if not path.is_dir() or not resolved.is_file():
            continue
        try:
            config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        except Exception:
            continue
        if config.get("active_experiment") == name and int(config.get("seed", -1)) == seed and not config.get("dry_run"):
            matches.append(path)
    return matches


def write_outputs(records: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "audit.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    fields = [
        "dataset", "model", "adaptation", "seed", "status", "run_id", "run_dir", "best_epoch",
        "errors", "warnings", "noncandidate_attempts",
    ]
    with (output_dir / "audit.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                **{name: record.get(name) for name in fields},
                "errors": " | ".join(record.get("errors", [])),
                "warnings": " | ".join(record.get("warnings", [])),
                "noncandidate_attempts": json.dumps(record.get("noncandidate_attempts", []), sort_keys=True),
            })
    lines = ["# C2 deterministic segmentation promotion audit", "", "| Dataset | Model | Adaptation | Seed | Status | Best epoch |", "|---|---|---|---:|---|---:|"]
    for record in records:
        lines.append(
            f"| {record['dataset']} | {record['model']} | {record['adaptation']} | {record['seed']} | "
            f"{record['status']} | {record.get('best_epoch') or ''} |"
        )
    blocked = [record for record in records if record["status"] != "PROMOTABLE"]
    lines += ["", f"Promotable: **{len(records) - len(blocked)}/{len(records)}**.", ""]
    for record in blocked:
        lines.append(f"- `{record['dataset']}/{record['model']}/{record['adaptation']}/seed{record['seed']}`: " + "; ".join(record["errors"]))
    warned = [record for record in records if record.get("warnings")]
    if warned:
        lines += ["", "## Tolerated numerical boundary cases", ""]
        for record in warned:
            lines.append(
                f"- `{record['dataset']}/{record['model']}/{record['adaptation']}/seed{record['seed']}`: "
                + "; ".join(record["warnings"])
            )
    preserved_attempts = [
        (record, attempt)
        for record in records
        for attempt in record.get("noncandidate_attempts", [])
    ]
    if preserved_attempts:
        lines += ["", "## Preserved non-candidate attempts", ""]
        for record, attempt in preserved_attempts:
            lines.append(
                f"- `{record['dataset']}/{record['model']}/{record['adaptation']}/seed{record['seed']}` — "
                f"`{attempt['run_id']}`: **{attempt['disposition']}**; {attempt['reason']}"
            )
    (output_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--deep", action="store_true", help="Recompute aggregate metrics from exported logits.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fail-on-nonpromotable", action="store_true")
    args = parser.parse_args()
    records: List[Dict[str, Any]] = []
    for dataset, model, adaptation, root_name in TARGETS:
        for seed in args.seeds:
            matches = discover(dataset, model, adaptation, root_name, seed)
            attempts = [inspect_candidate_attempt(path) for path in matches]
            completed = [
                path for path, attempt in zip(matches, attempts)
                if attempt["disposition"] == "COMPLETED_CANDIDATE"
            ]
            noncandidates = [
                attempt for attempt in attempts if attempt["disposition"] != "COMPLETED_CANDIDATE"
            ]
            if not completed:
                records.append({
                    "status": "MISSING", "dataset": dataset, "model": model, "adaptation": adaptation,
                    "seed": seed, "run_id": None, "run_dir": None, "best_epoch": None,
                    "incomplete_attempts": [attempt["run_dir"] for attempt in noncandidates],
                    "noncandidate_attempts": noncandidates,
                    "errors": ["no completed candidate found"],
                })
            elif len(completed) > 1:
                records.append({
                    "status": "AMBIGUOUS", "dataset": dataset, "model": model, "adaptation": adaptation,
                    "seed": seed, "run_id": None, "run_dir": None, "best_epoch": None,
                    "incomplete_attempts": [attempt["run_dir"] for attempt in noncandidates],
                    "noncandidate_attempts": noncandidates,
                    "errors": [f"multiple completed candidates: {[str(path) for path in completed]}"],
                })
            else:
                record = audit_run(completed[0], dataset, model, adaptation, seed, args.deep)
                record["incomplete_attempts"] = [attempt["run_dir"] for attempt in noncandidates]
                record["noncandidate_attempts"] = noncandidates
                records.append(record)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or PROJECT_ROOT / "reports" / "c2_segmentation_run_audits" / timestamp
    write_outputs(records, output_dir)
    counts = {status: sum(record["status"] == status for record in records) for status in {record["status"] for record in records}}
    print(json.dumps({"output_dir": str(output_dir), "counts": counts}, indent=2))
    if args.fail_on_nonpromotable and any(record["status"] != "PROMOTABLE" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
