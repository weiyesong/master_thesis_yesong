from __future__ import annotations

"""Validation-only C4 pilot checks for the frozen downstream MC Dropout protocol."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from scripts.experiment_manager import PROJECT_ROOT
from scripts.mc_dropout import activate_downstream_mc_dropout, designated_mc_dropout_modules
from scripts.prediction_export import checkpoint_sha256, export_predictions, validate_prediction_export
from scripts.run_experiments import (
    build_model,
    compute_task_classification_metrics,
    forward_classification_batch,
    make_dataloaders,
    move_batch,
)
from scripts.segmentation_pipeline import (
    build_segmentation_model,
    make_segmentation_dataloaders,
    move_segmentation_batch,
    segmentation_metrics,
)


PASSES = (10, 20, 30, 50)
FROZEN_PASSES = 30
DROPOUT_PROBABILITY = 0.10
OUTPUT_ROOT = PROJECT_ROOT / "results/c4_mc_dropout_pilots"
CLASSIFICATION_ROOT = OUTPUT_ROOT / "classification/eurosat_dofa_frozen"
SEGMENTATION_ROOT = OUTPUT_ROOT / "segmentation/cloudsen12_panopticon_frozen"
CLASSIFICATION_PILOT_OUTPUT = OUTPUT_ROOT / "validation_inference/classification"
SEGMENTATION_PILOT_OUTPUT = OUTPUT_ROOT / "validation_inference/segmentation"
SUBSET_MANIFEST = PROJECT_ROOT / "reports/mc_dropout_segmentation_research_subset_ids.json"
PROTOCOL_REPORT = PROJECT_ROOT / "reports/mc_dropout_protocol.md"
PILOT_REPORT = PROJECT_ROOT / "reports/mc_dropout_pilot_validation.md"
PRE_UQ_PROTOCOL = PROJECT_ROOT / "reports/pre_uq_protocol_freeze.md"
PRE_UQ_SHA256 = "150caf72c6085e4082a9e8ad619b604cdcbeb93b795226b1b9ec80b1d640d956"
SUBSET_SALT = "c4-downstream-mcd-segmentation-research-subset-v1"
FINAL_RESEARCH_SUBSET_SIZE = 32
PILOT_RESEARCH_SUBSET_SIZE = 2

# Fixed before pilot inference. Failing a bound means READY_FOR_FULL_MCD=NO; thresholds are never changed from results.
STABILITY_THRESHOLDS = {
    "mean_probability_mae": 0.005,
    "mean_probability_max_abs": 0.05,
    "metric_absolute_difference": 0.01,
    "uncertainty_mean_absolute_difference": 0.01,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
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


def _latest_completed_run(root: Path) -> Path:
    runs = sorted(
        path.parent for path in (root / "runs").glob("*/run_summary.json")
        if json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    )
    if len(runs) != 1:
        raise ValueError(f"Expected exactly one completed C4 pilot under {root}, found {len(runs)}")
    return runs[0]


def _run_material(run_root: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    run_dir = _latest_completed_run(run_root)
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    active = str(config["active_experiment"])
    experiment = next(item for item in config["experiments"] if item["name"] == active)
    checkpoint_path = run_dir / "best.pt"
    checkpoint = _load_checkpoint(checkpoint_path, device)
    if checkpoint.get("run_id") != run_dir.name:
        raise ValueError("C4 pilot checkpoint/run identity mismatch")
    return config, experiment, checkpoint_path, checkpoint


def _bn_state(model: nn.Module) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            values[name] = {
                "running_mean": None if module.running_mean is None else module.running_mean.detach().cpu().clone(),
                "running_var": None if module.running_var is None else module.running_var.detach().cpu().clone(),
                "num_batches_tracked": (
                    None if module.num_batches_tracked is None else module.num_batches_tracked.detach().cpu().clone()
                ),
            }
    return values


def _bn_state_equal(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> bool:
    if before.keys() != after.keys():
        return False
    for name in before:
        for key in before[name]:
            left, right = before[name][key], after[name][key]
            if left is None or right is None:
                if left is not right:
                    return False
            elif not torch.equal(left, right):
                return False
    return True


def _probability_uncertainty(probabilities: torch.Tensor) -> dict[str, torch.Tensor]:
    if probabilities.ndim not in (3, 5):
        raise ValueError("Stochastic probabilities must be [N,T,C] or [N,T,C,H,W]")
    class_dimension = 2
    safe = probabilities.clamp_min(1.0e-12)
    mean = probabilities.mean(dim=1)
    predictive_entropy = -(mean * mean.clamp_min(1.0e-12).log()).sum(dim=1)
    expected_entropy = -(probabilities * safe.log()).sum(dim=class_dimension).mean(dim=1)
    mutual_information = predictive_entropy - expected_entropy
    variance = probabilities.var(dim=1, unbiased=False)
    return {
        "mean_probabilities": mean,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "predictive_variance": variance,
    }


def _classification_convergence(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    probability_passes = logits.softmax(dim=2)
    rows: dict[str, Any] = {}
    summaries: dict[int, dict[str, torch.Tensor]] = {}
    for passes in PASSES:
        summary = _probability_uncertainty(probability_passes[:, :passes])
        summaries[passes] = summary
        aggregate_logits = summary["mean_probabilities"].clamp_min(1.0e-12).log()
        metrics = compute_task_classification_metrics(aggregate_logits, labels, "multiclass", n_bins=15)
        rows[str(passes)] = {
            "metrics": metrics,
            "uncertainty": {
                "mean_predictive_entropy": float(summary["predictive_entropy"].mean()),
                "mean_expected_entropy": float(summary["expected_entropy"].mean()),
                "mean_mutual_information": float(summary["mutual_information"].mean()),
                "mean_predictive_variance": float(summary["predictive_variance"].mean()),
            },
        }
    return _with_stability(rows, summaries, metric_names=("nll", "brier", "ece"))


def _segmentation_convergence(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
    ignore_index: int | None,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    summaries: dict[int, dict[str, torch.Tensor]] = {}
    for passes in PASSES:
        summary = _probability_uncertainty(probabilities[:, :passes])
        summaries[passes] = summary
        aggregate_logits = summary["mean_probabilities"].clamp_min(1.0e-12).log()
        metrics = segmentation_metrics(
            aggregate_logits, labels, class_names, ignore_index=ignore_index, n_bins=15
        )
        rows[str(passes)] = {
            "metrics": metrics,
            "uncertainty": {
                "mean_predictive_entropy": float(summary["predictive_entropy"].mean()),
                "mean_expected_entropy": float(summary["expected_entropy"].mean()),
                "mean_mutual_information": float(summary["mutual_information"].mean()),
                "mean_predictive_variance": float(summary["predictive_variance"].mean()),
            },
        }
    return _with_stability(rows, summaries, metric_names=("miou", "nll", "brier", "ece_15"))


def _with_stability(
    rows: dict[str, Any],
    summaries: dict[int, dict[str, torch.Tensor]],
    metric_names: Sequence[str],
) -> dict[str, Any]:
    thirty, fifty = summaries[30], summaries[50]
    probability_delta = (thirty["mean_probabilities"] - fifty["mean_probabilities"]).abs()
    metric_differences = {
        name: abs(float(rows["30"]["metrics"][name]) - float(rows["50"]["metrics"][name]))
        for name in metric_names
    }
    uncertainty_differences = {
        name: abs(float(rows["30"]["uncertainty"][name]) - float(rows["50"]["uncertainty"][name]))
        for name in rows["30"]["uncertainty"]
    }
    checks = {
        "mean_probability_mae": float(probability_delta.mean())
        <= STABILITY_THRESHOLDS["mean_probability_mae"],
        "mean_probability_max_abs": float(probability_delta.max())
        <= STABILITY_THRESHOLDS["mean_probability_max_abs"],
        "metric_absolute_differences": all(
            value <= STABILITY_THRESHOLDS["metric_absolute_difference"]
            for value in metric_differences.values()
        ),
        "uncertainty_mean_absolute_differences": all(
            value <= STABILITY_THRESHOLDS["uncertainty_mean_absolute_difference"]
            for value in uncertainty_differences.values()
        ),
    }
    return {
        "passes": rows,
        "comparison": {
            "reference": "T=50",
            "candidate": "T=30",
            "mean_probability_mae": float(probability_delta.mean()),
            "mean_probability_max_abs": float(probability_delta.max()),
            "metric_absolute_differences": metric_differences,
            "uncertainty_mean_absolute_differences": uncertainty_differences,
            "thresholds": STABILITY_THRESHOLDS,
            "checks": checks,
            "stable": all(checks.values()),
        },
    }


def _hash_rank(dataset: str, sample_id: str) -> str:
    return hashlib.sha256(f"{SUBSET_SALT}\0{dataset}\0{sample_id}".encode("utf-8")).hexdigest()


def _select_ids(dataset: str, values: Sequence[str], count: int) -> list[str]:
    unique = sorted(set(str(value) for value in values))
    if len(unique) != len(values):
        raise ValueError(f"Duplicate sample IDs in {dataset} candidate subset")
    return sorted(unique, key=lambda value: (_hash_rank(dataset, value), value))[:count]


def prepare_research_subsets() -> dict[str, Any]:
    if hashlib.sha256(PRE_UQ_PROTOCOL.read_bytes()).hexdigest() != PRE_UQ_SHA256:
        raise ValueError("Frozen pre-UQ protocol hash changed")
    datasets: dict[str, Any] = {}
    for dataset in ("cloudsen12", "spacenet7"):
        path = PROJECT_ROOT / f"reports/dataset_manifests/{dataset}_actual_manifest.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
        split_name = "test"
        test_ids = [record["sample_id"] for record in records if record["split"] == split_name]
        validation_ids = [record["sample_id"] for record in records if record["split"] == "validation"]
        datasets[dataset] = {
            "source_manifest": str(path.resolve()),
            "source_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "official_test_count": len(test_ids),
            "final_test_research_subset_size": FINAL_RESEARCH_SUBSET_SIZE,
            "final_test_research_subset_ids": _select_ids(dataset, test_ids, FINAL_RESEARCH_SUBSET_SIZE),
        }
        if dataset == "cloudsen12":
            datasets[dataset]["pilot_validation_research_subset_size"] = PILOT_RESEARCH_SUBSET_SIZE
            datasets[dataset]["pilot_validation_research_subset_ids"] = _select_ids(
                f"{dataset}:validation", validation_ids, PILOT_RESEARCH_SUBSET_SIZE
            )
    payload = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "selected_before_any_c4_stochastic_inference": True,
        "selection_method": (
            "Take the lowest SHA256 ranks of salt\\0dataset\\0sample_id within the official split; "
            "selection is independent of images, labels, predictions, and uncertainty."
        ),
        "selection_salt": SUBSET_SALT,
        "datasets": datasets,
    }
    _write_json_exclusive(SUBSET_MANIFEST, payload)
    return payload


def run_classification_pilot(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    config, experiment, checkpoint_path, checkpoint = _run_material(CLASSIFICATION_ROOT, device)
    model = build_model(config, experiment["model"], int(config["data"]["num_classes"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    loader = make_dataloaders(config)["val"]
    raw_batch = next(iter(loader))
    batch = move_batch(raw_batch, device)
    sample_ids = [str(value) for value in raw_batch["sample_id"]]
    labels = batch["label"].detach().cpu()
    checkpoint_before = checkpoint_sha256(checkpoint_path)
    batchnorm_before = _bn_state(model)

    model.eval()
    with torch.no_grad():
        deterministic_first = forward_classification_batch(model, batch)
        deterministic_second = forward_classification_batch(model, batch)
        embeddings = model.extract_features(batch["image"]).detach().cpu()
    deterministic_repeated_equal = bool(torch.equal(deterministic_first, deterministic_second))
    activation_audit = activate_downstream_mc_dropout(model)
    logits_parts = []
    with torch.no_grad():
        for _ in range(max(PASSES)):
            logits_parts.append(forward_classification_batch(model, batch).detach().float().cpu())
    stochastic_logits = torch.stack(logits_parts, dim=1)

    activate_downstream_mc_dropout(model)
    with torch.no_grad():
        torch.manual_seed(1729)
        controlled_first = forward_classification_batch(model, batch).detach().cpu()
        torch.manual_seed(1729)
        controlled_second = forward_classification_batch(model, batch).detach().cpu()
    same_seed_repeat_equal = bool(torch.equal(controlled_first, controlled_second))
    stochastic_nonidentical = bool(torch.any(stochastic_logits[:, 1:] != stochastic_logits[:, :-1]))
    batchnorm_unchanged = _bn_state_equal(batchnorm_before, _bn_state(model))
    checkpoint_unchanged = checkpoint_before == checkpoint_sha256(checkpoint_path)
    no_gradients = all(parameter.grad is None for parameter in model.parameters())
    convergence = _classification_convergence(stochastic_logits, labels)

    if CLASSIFICATION_PILOT_OUTPUT.exists():
        raise FileExistsError(CLASSIFICATION_PILOT_OUTPUT)
    probabilities = stochastic_logits.softmax(dim=2)
    uncertainty = _probability_uncertainty(probabilities)
    aggregate_logits = uncertainty["mean_probabilities"].clamp_min(1.0e-12).log()
    export_predictions(
        CLASSIFICATION_PILOT_OUTPUT,
        sample_ids=sample_ids,
        true_labels=labels,
        logits=aggregate_logits,
        class_names=config["data"]["class_names"],
        model_name="DOFA",
        dataset="eurosat",
        adaptation_mode="frozen",
        seed=42,
        checkpoint=str(checkpoint_path),
        split="validation_pilot_subset",
        uq_method="downstream_head_mc_dropout",
        expected_count=len(sample_ids),
        stochastic_logits=stochastic_logits,
        embeddings=embeddings,
        source={
            "run_id": checkpoint["run_id"],
            "checkpoint_sha256": checkpoint_before,
            "split_total_count": len(sample_ids),
            "official_split": "validation",
            "pilot_partial_split": True,
            "passes_collected": max(PASSES),
        },
    )
    np.savez_compressed(
        CLASSIFICATION_PILOT_OUTPUT / "uncertainty_summaries.npz",
        sample_id=np.asarray(sample_ids),
        label=labels.numpy(),
        probabilities=uncertainty["mean_probabilities"].numpy().astype(np.float32),
        prediction=uncertainty["mean_probabilities"].argmax(dim=1).numpy(),
        predictive_entropy=uncertainty["predictive_entropy"].numpy().astype(np.float32),
        expected_entropy=uncertainty["expected_entropy"].numpy().astype(np.float32),
        mutual_information=uncertainty["mutual_information"].numpy().astype(np.float32),
        predictive_variance=uncertainty["predictive_variance"].numpy().astype(np.float32),
    )
    _write_json_exclusive(CLASSIFICATION_PILOT_OUTPUT / "convergence.json", convergence)
    classification_run = _latest_completed_run(CLASSIFICATION_ROOT)
    model_audit = json.loads((classification_run / "model_audit.json").read_text(encoding="utf-8"))
    gradient_audit = json.loads((classification_run / "gradient_audit.json").read_text(encoding="utf-8"))
    training_history = json.loads((classification_run / "training_history.json").read_text(encoding="utf-8"))
    checks = {
        "checkpoint_selection_is_validation_nll_min": (
            checkpoint.get("selection_metric") == "nll"
            and config["training"]["checkpoint"] == {"split": "val", "metric": "nll", "mode": "min"}
        ),
        "one_designated_dropout": len(designated_mc_dropout_modules(model)) == 1,
        "dropout_probability_is_0_10": math.isclose(
            float(designated_mc_dropout_modules(model)[0][1].p), DROPOUT_PROBABILITY
        ),
        "dropout_active_during_training": model_audit["training_mode_policy"]["head_training"]
        and model_audit["training_mode_policy"]["head_dropout"] == "configured",
        "training_gradients_valid": gradient_audit["performed"]
        and gradient_audit["all_expected_parameters_received_gradient"]
        and not gradient_audit["zero_gradient_parameters"]
        and not gradient_audit["nonfinite_gradient_parameters"]
        and float(training_history[0]["train"]["head_gradient_norm"]) > 0.0,
        "dropout_active_during_mc_inference": activation_audit["designated_modules"][0]["training"],
        "batchnorm_eval_and_buffers_unchanged": batchnorm_unchanged
        and not activation_audit["batchnorm_training_modules"],
        "backbone_stochasticity_disabled": not activation_audit["backbone_stochastic_training_modules"],
        "deterministic_repeated_outputs_equal": deterministic_repeated_equal,
        "same_dropout_seed_reproduces_output": same_seed_repeat_equal,
        "stochastic_outputs_nonidentical": stochastic_nonidentical,
        "no_gradients_during_mc_inference": no_gradients,
        "checkpoint_file_unchanged": checkpoint_unchanged,
        "stochastic_shape_n_t_c": list(stochastic_logits.shape)
        == [len(sample_ids), max(PASSES), int(config["data"]["num_classes"])],
        "export_validation_passed": validate_prediction_export(CLASSIFICATION_PILOT_OUTPUT)["valid"],
        "t30_stable_relative_to_t50": convergence["comparison"]["stable"],
    }
    payload = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "task": "classification",
        "pilot_cell": "EuroSAT / DOFA / frozen / seed 42",
        "split_used_for_mc_convergence": "official validation; one fixed pilot batch",
        "test_access": False,
        "run_dir": str(classification_run),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_before,
        "sample_ids": sample_ids,
        "activation_audit": activation_audit,
        "checks": checks,
        "passed": all(checks.values()),
    }
    _write_json_exclusive(CLASSIFICATION_PILOT_OUTPUT / "pilot_audit.json", payload)
    return payload


def run_segmentation_pilot(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    subset_manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    fixed_pilot_ids = subset_manifest["datasets"]["cloudsen12"]["pilot_validation_research_subset_ids"]
    config, experiment, checkpoint_path, checkpoint = _run_material(SEGMENTATION_ROOT, device)
    model = build_segmentation_model(config, experiment["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    loader = make_segmentation_dataloaders(config)["val"]
    raw_batch = next(iter(loader))
    batch = move_segmentation_batch(raw_batch, device)
    sample_ids = [str(value) for value in raw_batch["sample_id"]]
    labels = batch["mask"].detach().cpu()
    dataset_ids = [str(value) for value in loader.dataset.sample_ids]
    id_to_index = {sample_id: index for index, sample_id in enumerate(dataset_ids)}
    if any(sample_id not in id_to_index for sample_id in fixed_pilot_ids):
        raise ValueError("Predeclared pilot research-subset ID is absent from official validation")
    fixed_items = [loader.dataset[id_to_index[sample_id]] for sample_id in fixed_pilot_ids]
    fixed_images = torch.stack([item["image"] for item in fixed_items]).to(device)
    fixed_labels = torch.stack([item["mask"] for item in fixed_items]).cpu()
    checkpoint_before = checkpoint_sha256(checkpoint_path)
    batchnorm_before = _bn_state(model)

    model.eval()
    with torch.no_grad():
        deterministic_first = model(batch["image"])
        deterministic_second = model(batch["image"])
    deterministic_repeated_equal = bool(torch.equal(deterministic_first, deterministic_second))
    activation_audit = activate_downstream_mc_dropout(model)
    probability_parts = []
    with torch.no_grad():
        for _ in range(max(PASSES)):
            probability_parts.append(model(batch["image"]).softmax(dim=1).detach().float().cpu())
    stochastic_probabilities = torch.stack(probability_parts, dim=1)
    fixed_probability_parts = []
    activate_downstream_mc_dropout(model)
    with torch.no_grad():
        for _ in range(max(PASSES)):
            fixed_probability_parts.append(model(fixed_images).softmax(dim=1).detach().float().cpu())
    fixed_stochastic_probabilities = torch.stack(fixed_probability_parts, dim=1)

    activate_downstream_mc_dropout(model)
    with torch.no_grad():
        torch.manual_seed(2718)
        controlled_first = model(batch["image"]).detach().cpu()
        torch.manual_seed(2718)
        controlled_second = model(batch["image"]).detach().cpu()
    same_seed_repeat_equal = bool(torch.equal(controlled_first, controlled_second))
    stochastic_nonidentical = bool(
        torch.any(stochastic_probabilities[:, 1:] != stochastic_probabilities[:, :-1])
    )
    batchnorm_unchanged = _bn_state_equal(batchnorm_before, _bn_state(model))
    checkpoint_unchanged = checkpoint_before == checkpoint_sha256(checkpoint_path)
    no_gradients = all(parameter.grad is None for parameter in model.parameters())
    convergence = _segmentation_convergence(
        stochastic_probabilities,
        labels,
        config["data"]["class_names"],
        config["data"].get("ignore_index"),
    )

    if SEGMENTATION_PILOT_OUTPUT.exists():
        raise FileExistsError(SEGMENTATION_PILOT_OUTPUT)
    SEGMENTATION_PILOT_OUTPUT.mkdir(parents=True, exist_ok=False)
    uncertainty = _probability_uncertainty(stochastic_probabilities)
    mean_probabilities = uncertainty["mean_probabilities"]
    prediction = mean_probabilities.argmax(dim=1)
    confidence = mean_probabilities.max(dim=1).values
    np.savez_compressed(
        SEGMENTATION_PILOT_OUTPUT / "aggregate_uncertainty_maps.npz",
        sample_id=np.asarray(sample_ids),
        label=labels.numpy(),
        probabilities=mean_probabilities.numpy().astype(np.float32),
        prediction=prediction.numpy(),
        confidence=confidence.numpy().astype(np.float32),
        predictive_entropy=uncertainty["predictive_entropy"].numpy().astype(np.float32),
        expected_entropy=uncertainty["expected_entropy"].numpy().astype(np.float32),
        mutual_information=uncertainty["mutual_information"].numpy().astype(np.float32),
        predictive_variance=uncertainty["predictive_variance"].numpy().astype(np.float32),
    )
    np.savez_compressed(
        SEGMENTATION_PILOT_OUTPUT / "research_subset_stochastic_probabilities.npz",
        sample_id=np.asarray(fixed_pilot_ids),
        label=fixed_labels.numpy(),
        probabilities=fixed_stochastic_probabilities.numpy().astype(np.float32),
        axes=np.asarray(["sample", "pass", "class", "height", "width"]),
    )
    _write_json_exclusive(SEGMENTATION_PILOT_OUTPUT / "convergence.json", convergence)
    model_audit = json.loads(
        (_latest_completed_run(SEGMENTATION_ROOT) / "model_audit.json").read_text(encoding="utf-8")
    )
    gradient_audit = json.loads(
        (_latest_completed_run(SEGMENTATION_ROOT) / "gradient_audit.json").read_text(encoding="utf-8")
    )
    checks = {
        "checkpoint_selection_is_validation_miou_max": (
            checkpoint.get("selection_metric") == "miou"
            and config["training"]["checkpoint"] == {"split": "val", "metric": "miou", "mode": "max"}
        ),
        "one_designated_dropout2d": len(designated_mc_dropout_modules(model)) == 1
        and isinstance(designated_mc_dropout_modules(model)[0][1], nn.Dropout2d),
        "dropout_probability_is_0_10": math.isclose(
            float(designated_mc_dropout_modules(model)[0][1].p), DROPOUT_PROBABILITY
        ),
        "dropout_active_during_training": len(model_audit["mc_dropout"]) == 1,
        "training_gradients_valid": gradient_audit["groups"]["head"]["all_gradients_finite"]
        and gradient_audit["groups"]["head"]["all_trainable_parameters_have_gradients"]
        and gradient_audit["groups"]["head"]["any_nonzero_gradient"]
        and gradient_audit["groups"]["backbone"]["gradient_parameter_tensors"] == 0,
        "dropout_active_during_mc_inference": activation_audit["designated_modules"][0]["training"],
        "batchnorm_eval_and_buffers_unchanged": batchnorm_unchanged
        and not activation_audit["batchnorm_training_modules"],
        "backbone_stochasticity_disabled": not activation_audit["backbone_stochastic_training_modules"],
        "deterministic_repeated_outputs_equal": deterministic_repeated_equal,
        "same_dropout_seed_reproduces_output": same_seed_repeat_equal,
        "stochastic_outputs_nonidentical": stochastic_nonidentical,
        "no_gradients_during_mc_inference": no_gradients,
        "checkpoint_file_unchanged": checkpoint_unchanged,
        "aggregate_map_schema_valid": list(mean_probabilities.shape)
        == [len(sample_ids), int(config["data"]["num_classes"]), *labels.shape[-2:]],
        "research_subset_stochastic_schema_valid": list(fixed_stochastic_probabilities.shape)
        == [PILOT_RESEARCH_SUBSET_SIZE, max(PASSES), int(config["data"]["num_classes"]), *labels.shape[-2:]],
        "t30_stable_relative_to_t50": convergence["comparison"]["stable"],
        "test_not_evaluated_by_pilot_training": not json.loads(
            (_latest_completed_run(SEGMENTATION_ROOT) / "run_summary.json").read_text(encoding="utf-8")
        )["test_evaluation_performed"],
    }
    payload = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "task": "segmentation",
        "pilot_cell": "CloudSEN12 / Panopticon / frozen / seed 42",
        "split_used_for_mc_convergence": "official validation; one fixed pilot batch",
        "test_access": False,
        "run_dir": str(_latest_completed_run(SEGMENTATION_ROOT)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_before,
        "sample_ids": sample_ids,
        "research_subset_ids": fixed_pilot_ids,
        "research_subset_selection": "frozen SHA256 ranking within official validation before inference",
        "activation_audit": activation_audit,
        "checks": checks,
        "passed": all(checks.values()),
    }
    _write_json_exclusive(SEGMENTATION_PILOT_OUTPUT / "pilot_audit.json", payload)
    return payload


def _format_classification_rows(convergence: dict[str, Any]) -> list[str]:
    lines = [
        "| T | Accuracy | NLL | Brier | ECE-15 | Pred. entropy | Expected entropy | MI | Variance |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for passes in PASSES:
        row = convergence["passes"][str(passes)]
        metric, uq = row["metrics"], row["uncertainty"]
        lines.append(
            f"| {passes} | {metric['accuracy']:.6f} | {metric['nll']:.6f} | "
            f"{metric['brier']:.6f} | {metric['ece']:.6f} | {uq['mean_predictive_entropy']:.6f} | "
            f"{uq['mean_expected_entropy']:.6f} | {uq['mean_mutual_information']:.6f} | "
            f"{uq['mean_predictive_variance']:.6f} |"
        )
    return lines


def _format_segmentation_rows(convergence: dict[str, Any]) -> list[str]:
    lines = [
        "| T | mIoU | NLL | Brier | ECE-15 | Pred. entropy | Expected entropy | Disagreement/MI | Variance |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for passes in PASSES:
        row = convergence["passes"][str(passes)]
        metric, uq = row["metrics"], row["uncertainty"]
        lines.append(
            f"| {passes} | {metric['miou']:.6f} | {metric['nll']:.6f} | {metric['brier']:.6f} | "
            f"{metric['ece_15']:.6f} | {uq['mean_predictive_entropy']:.6f} | "
            f"{uq['mean_expected_entropy']:.6f} | {uq['mean_mutual_information']:.6f} | "
            f"{uq['mean_predictive_variance']:.6f} |"
        )
    return lines


def finalize_reports() -> dict[str, Any]:
    classification_audit = json.loads(
        (CLASSIFICATION_PILOT_OUTPUT / "pilot_audit.json").read_text(encoding="utf-8")
    )
    segmentation_audit = json.loads(
        (SEGMENTATION_PILOT_OUTPUT / "pilot_audit.json").read_text(encoding="utf-8")
    )
    classification_convergence = json.loads(
        (CLASSIFICATION_PILOT_OUTPUT / "convergence.json").read_text(encoding="utf-8")
    )
    segmentation_convergence = json.loads(
        (SEGMENTATION_PILOT_OUTPUT / "convergence.json").read_text(encoding="utf-8")
    )
    ready = bool(classification_audit["passed"] and segmentation_audit["passed"])
    code_snapshot = json.loads((OUTPUT_ROOT / "code_snapshot.json").read_text(encoding="utf-8"))
    protocol_lines = [
        "# MC Dropout Protocol",
        "",
        f"Created: {utc_now()}",
        "",
        "Status: frozen prospectively from `reports/pre_uq_protocol_freeze.md`.",
        "",
        "## Method scope",
        "",
        "The thesis method is **downstream-head/decoder MC Dropout**. DOFA and Panopticon backbone architectures, native dropout, DropPath, and stochastic depth are not altered or enabled. Classification uses one `Dropout(p=0.10)` immediately before the final linear classifier (after EuroSAT's fixed BatchNorm). Segmentation uses one `Dropout2d(p=0.10)` on the common decoder's final feature map immediately before its 1×1 classifier. Placement is identical across backbone families within each task.",
        "",
        "The deterministic comparator retains its original zero-dropout head/decoder. The MC model is newly trained with the one designated dropout layer; a deterministic checkpoint is never converted by inference-only dropout insertion.",
        "",
        "## Training and checkpoint selection",
        "",
        "- Classification: minimum validation NLL; earliest strict tie.",
        "- Segmentation: maximum validation mIoU; earliest strict tie.",
        "- Validation used for checkpoint selection is deterministic with dropout disabled.",
        "- Test metrics, calibration, and uncertainty never participate in selection.",
        "",
        "## Stochastic inference",
        "",
        "- Freeze model weights; call full-model evaluation mode, then activate exactly the one designated downstream dropout module.",
        "- BatchNorm remains in evaluation mode and its running buffers must be unchanged.",
        "- Use `torch.no_grad()` and identical preprocessed input tensors for all passes.",
        "- Run exactly `T=30` passes for final experiments. Arithmetic-mean probabilities—not logits or decisions—are predictive probabilities.",
        "- Preserve predictive entropy, expected entropy, their difference as MI-style disagreement, and per-class/per-pixel probability variance.",
        "",
        "## Pass-count validation rule",
        "",
        "Pilots evaluate nested prefixes T=10/20/30/50 on validation only. T=30 is accepted relative to T=50 only when probability MAE ≤0.005, maximum absolute probability difference ≤0.05, each requested metric changes by ≤0.01, and each mean uncertainty summary changes by ≤0.01. Failure blocks the full launch; thresholds are not adjusted after results.",
        "",
        "## Output schema",
        "",
        "Classification saves `[N,T,C]` raw logits/probabilities plus sample ID, label, probability-mean prediction, predictive entropy, expected entropy, MI-style disagreement, probability variance `[N,C]`, and backbone representation.",
        "",
        "Segmentation saves full-test probability-mean maps, prediction/confidence maps, predictive entropy, expected entropy, disagreement, and variance. Full `[N_subset,T,C,H,W]` stochastic probabilities are retained only for the predeclared research subset when full storage is excessive. The exact final CloudSEN12 and SpaceNet7 test IDs were fixed by label/prediction-independent SHA256 ranking before pilot stochastic inference in `reports/mc_dropout_segmentation_research_subset_ids.json`.",
        "",
        "## Replication",
        "",
        "The frozen core is 24 new runs: seed 42 for all 16 cells, plus seeds 43/44 only for EuroSAT/DOFA/frozen, TreeSatAI/Panopticon/full-finetune, CloudSEN12/Panopticon/frozen, and SpaceNet7/DOFA/full-finetune. MC passes are repeated posterior-style draws, not independent training replicates.",
        "",
        "## Frozen values",
        "",
        "| Quantity | Value |",
        "|---|---|",
        "| Dropout probability | 0.10 for both tasks |",
        "| Classification placement | after optional fixed BatchNorm, before final Linear |",
        "| Segmentation placement | final decoder feature map, before 1×1 classifier |",
        "| Backbone stochasticity | disabled during MC inference |",
        "| Final stochastic passes | 30 |",
        "| Calibration bins | 15 |",
        "| Aggregation | arithmetic mean of probabilities |",
        "",
        f"Pilot inference code snapshot: `sha256:{code_snapshot['code_sha256']}` ({code_snapshot['file_count']} files).",
        "",
    ]
    pilot_lines = [
        "# MC Dropout Pilot Validation",
        "",
        f"Created: {utc_now()}",
        "",
        "No deterministic experiment was modified. Two new one-epoch, one-training-batch pilots were run, and all MC convergence analysis used one fixed official-validation batch. Test data were not evaluated or inspected.",
        "",
        "The classification batch is a deterministic loader-prefix technical subset and contains one EuroSAT class; its absolute task metrics are not thesis results. It is used only to test stochastic-mode behavior and nested-pass convergence. The segmentation batch is likewise a technical subset. Final MC results must use the complete official evaluation split.",
        "",
        "## Classification pilot",
        "",
        f"Cell: {classification_audit['pilot_cell']}. Checkpoint: `{classification_audit['checkpoint']}`.",
        "",
        *_format_classification_rows(classification_convergence),
        "",
        "T=30 vs T=50: probability MAE "
        f"{classification_convergence['comparison']['mean_probability_mae']:.6f}, maximum difference "
        f"{classification_convergence['comparison']['mean_probability_max_abs']:.6f}; stable = "
        f"**{str(classification_convergence['comparison']['stable']).upper()}**.",
        "",
        "## Segmentation pilot",
        "",
        f"Cell: {segmentation_audit['pilot_cell']}. Checkpoint: `{segmentation_audit['checkpoint']}`.",
        "",
        *_format_segmentation_rows(segmentation_convergence),
        "",
        "T=30 vs T=50: probability MAE "
        f"{segmentation_convergence['comparison']['mean_probability_mae']:.6f}, maximum difference "
        f"{segmentation_convergence['comparison']['mean_probability_max_abs']:.6f}; stable = "
        f"**{str(segmentation_convergence['comparison']['stable']).upper()}**.",
        "",
        "## Validation checks",
        "",
        "| Check | Classification | Segmentation |",
        "|---|---|---|",
    ]
    check_names = sorted(set(classification_audit["checks"]) | set(segmentation_audit["checks"]))
    for name in check_names:
        left = classification_audit["checks"].get(name)
        right = segmentation_audit["checks"].get(name)
        pilot_lines.append(
            f"| `{name}` | {'PASS' if left else ('—' if left is None else 'FAIL')} | "
            f"{'PASS' if right else ('—' if right is None else 'FAIL')} |"
        )
    pilot_lines.extend(
        [
            "",
            "The checks cover training-time activation, inference-time activation, finite/nonzero gradients from the training run audits, deterministic repeated outputs, same-seed reproducibility, non-identical stochastic outputs, BatchNorm evaluation/buffer preservation, disabled backbone stochasticity, checkpoint immutability, pass-count convergence, and storage shapes.",
            "",
            f"## READY_FOR_FULL_MCD = {'YES' if ready else 'NO'}",
            "",
            "This readiness decision authorizes the frozen full MC Dropout matrix only if it is YES; it does not launch that matrix.",
            "",
        ]
    )
    _write_text_exclusive(PROTOCOL_REPORT, "\n".join(protocol_lines))
    _write_text_exclusive(PILOT_REPORT, "\n".join(pilot_lines))
    return {
        "protocol_report": str(PROTOCOL_REPORT),
        "pilot_report": str(PILOT_REPORT),
        "ready_for_full_mcd": ready,
        "code_snapshot_sha256": code_snapshot["code_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-research-subsets")
    classification = subparsers.add_parser("classification-pilot")
    classification.add_argument("--device", required=True)
    segmentation = subparsers.add_parser("segmentation-pilot")
    segmentation.add_argument("--device", required=True)
    subparsers.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "prepare-research-subsets":
        result = prepare_research_subsets()
    elif args.command == "classification-pilot":
        result = run_classification_pilot(args.device)
    elif args.command == "segmentation-pilot":
        result = run_segmentation_pilot(args.device)
    else:
        result = finalize_reports()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
