from __future__ import annotations

"""Prepare and audit the frozen C5 MC-Dropout training registry.

This module creates only new C5 configuration/registry artifacts.  It never
opens a test dataloader and never edits a deterministic run artifact.
"""

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.c3_calibration_ensembles import classification_registry, segmentation_registry
from scripts.experiment_manager import PROJECT_ROOT


CONFIG_ROOT = PROJECT_ROOT / "configs/c5_mc_dropout"
OUTPUT_ROOT = PROJECT_ROOT / "results/final_thesis/mc_dropout"
REGISTRY_PATH = OUTPUT_ROOT / "run_registry.json"
PROTOCOL_PATH = PROJECT_ROOT / "reports/mc_dropout_protocol.md"
PILOT_PATH = PROJECT_ROOT / "reports/mc_dropout_pilot_validation.md"
PRE_UQ_PATH = PROJECT_ROOT / "reports/pre_uq_protocol_freeze.md"

EXPECTED_PROTOCOL_SHA256 = "5d528c67546bee3faffc48cfd268a9aacce332575af8c23df6beb1b4966d4e17"
EXPECTED_PILOT_SHA256 = "7a4c7c9390460bd20a8a20f9fc5ff56c40ab65de2151921a11c963835c7e4aae"
EXPECTED_PRE_UQ_SHA256 = "150caf72c6085e4082a9e8ad619b604cdcbeb93b795226b1b9ec80b1d640d956"

SOURCE_CONFIGS = {
    ("eurosat", "dofa", "frozen"): "configs/eurosat_dofa_frozen_baseline.yaml",
    ("eurosat", "dofa", "full_finetune"): "configs/eurosat_dofa_full_finetune.yaml",
    ("eurosat", "panopticon", "frozen"): "configs/eurosat_panopticon_frozen_baseline.yaml",
    ("eurosat", "panopticon", "full_finetune"): "configs/eurosat_panopticon_full_finetune.yaml",
    ("treesatai", "dofa", "frozen"): "configs/treesatai_frozen_final.yaml",
    ("treesatai", "panopticon", "frozen"): "configs/treesatai_frozen_final.yaml",
    ("treesatai", "dofa", "full_finetune"): "configs/treesatai_full_finetune_final.yaml",
    ("treesatai", "panopticon", "full_finetune"): "configs/treesatai_full_finetune_final.yaml",
    ("cloudsen12", "dofa", "frozen"): "configs/cloudsen12_frozen_final.yaml",
    ("cloudsen12", "panopticon", "frozen"): "configs/cloudsen12_frozen_final.yaml",
    ("cloudsen12", "dofa", "full_finetune"): "configs/cloudsen12_full_finetune_final.yaml",
    ("cloudsen12", "panopticon", "full_finetune"): "configs/cloudsen12_full_finetune_final.yaml",
    ("spacenet7", "dofa", "frozen"): "configs/spacenet7_frozen_final.yaml",
    ("spacenet7", "panopticon", "frozen"): "configs/spacenet7_frozen_final.yaml",
    ("spacenet7", "dofa", "full_finetune"): "configs/spacenet7_full_finetune_final.yaml",
    ("spacenet7", "panopticon", "full_finetune"): "configs/spacenet7_full_finetune_final.yaml",
}

ROBUSTNESS_CELLS = {
    ("eurosat", "dofa", "frozen"),
    ("treesatai", "panopticon", "full_finetune"),
    ("cloudsen12", "panopticon", "frozen"),
    ("spacenet7", "dofa", "full_finetune"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _check_frozen_gate() -> None:
    expected = {
        PRE_UQ_PATH: EXPECTED_PRE_UQ_SHA256,
        PROTOCOL_PATH: EXPECTED_PROTOCOL_SHA256,
        PILOT_PATH: EXPECTED_PILOT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"Frozen C4 artifact hash changed: {path}")
    if "READY_FOR_FULL_MCD = YES" not in PILOT_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("C4 did not authorize the full MC-Dropout matrix")


def _deterministic_comparators() -> dict[tuple[str, str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for member in [*classification_registry(), *segmentation_registry()]:
        key = (member.dataset, member.model, member.adaptation, member.seed)
        if key in result:
            raise ValueError(f"Duplicate deterministic comparator: {key}")
        result[key] = {
            "run_id": member.run_id,
            "run_dir": str(member.run_dir.resolve()),
            "checkpoint_path": str(member.checkpoint_path.resolve()),
            "checkpoint_sha256": member.checkpoint_sha256,
            "prediction_path": str(member.prediction_path.resolve()),
            "prediction_sha256": sha256_file(member.prediction_path),
        }
    return result


def _derive_config(cell: tuple[str, str, str]) -> tuple[dict[str, Any], Path, str]:
    dataset, model, adaptation = cell
    source_path = PROJECT_ROOT / SOURCE_CONFIGS[cell]
    source_hash = sha256_file(source_path)
    config = copy.deepcopy(_read_yaml(source_path))
    matches = [
        item for item in config["experiments"]
        if item["model"]["name"] == model and item["model"]["adaptation_mode"] == adaptation
    ]
    if len(matches) != 1:
        raise ValueError(f"Source config does not resolve one experiment for {cell}")
    experiment = copy.deepcopy(matches[0])
    task = "classification" if dataset in {"eurosat", "treesatai"} else "segmentation"
    experiment_name = f"c5_mcd_{dataset}_{model}_{adaptation}"
    experiment["name"] = experiment_name
    if task == "classification":
        head = experiment["model"].setdefault("head", {})
        head["dropout"] = 0.10
    else:
        decoder = experiment["model"].setdefault("decoder", {})
        decoder["dropout"] = 0.10
    seeds = [42, 43, 44] if cell in ROBUSTNESS_CELLS else [42]
    output_dir = OUTPUT_ROOT / task / dataset / model / adaptation
    config.update(
        {
            "seed": 42,
            "seeds": seeds,
            "device": "auto",
            "output_dir": str(output_dir.resolve()),
            "results_csv": str((output_dir / "results.csv").resolve()),
            "experiments": [experiment],
            "prediction_export": {"enabled": False},
            "evaluation": {"test_enabled": False},
            "visualization": {"enabled": False, "track_gradient_norm": True},
            "uq_methods": [
                {
                    "name": "downstream_mc_dropout",
                    "dropout_probability": 0.10,
                    "stochastic_passes": 30,
                    "aggregation": "arithmetic_mean_probabilities",
                }
            ],
            "provenance": {
                "require_input_hashes": True,
                "protocol_id": "C5-FINAL-MCD-2026-08-26-v1",
                "protocol_path": str(PROTOCOL_PATH.resolve()),
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            },
            "c5_frozen_protocol": {
                "pre_uq_protocol_path": str(PRE_UQ_PATH.resolve()),
                "pre_uq_protocol_sha256": EXPECTED_PRE_UQ_SHA256,
                "pilot_validation_path": str(PILOT_PATH.resolve()),
                "pilot_validation_sha256": EXPECTED_PILOT_SHA256,
                "source_deterministic_config": str(source_path.resolve()),
                "source_deterministic_config_sha256": source_hash,
                "test_embargo_during_training": True,
            },
        }
    )
    # Old immutable DOFA-EuroSAT configs predate mandatory artifact provenance.
    # These metadata fields do not alter their historical preprocessing: the
    # absence of data.normalization intentionally retains the old RGB constants.
    if dataset == "eurosat" and model == "dofa":
        experiment["model"]["weights_sha256"] = (
            "4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1"
        )
        config["experiments"] = [experiment]
        config["data"].update(
            {
                "artifact_path": "./data/EuroSATallBands.zip",
                "artifact_sha256": "751f070f9bffa2eed48b24ca2dd0b02959280c08837e8c9a5532a67ba611df59",
                "split_manifest_sha256": "c5cadc7936394f0678307f7db25c55a2dc19f73ddb93891b772c16221e8abf22",
                "wavelength_source": "immutable historical nominal Sentinel-2 RGB centers",
            }
        )
        if "normalization" in config["data"]:
            raise ValueError("DOFA-EuroSAT C5 must retain the immutable comparator's historical normalization")
    return config, source_path, source_hash


def prepare() -> dict[str, Any]:
    _check_frozen_gate()
    if CONFIG_ROOT.exists() or REGISTRY_PATH.exists():
        raise FileExistsError("C5 preparation artifacts already exist; refusing to overwrite")
    CONFIG_ROOT.mkdir(parents=True, exist_ok=False)
    comparators = _deterministic_comparators()
    rows: list[dict[str, Any]] = []
    for cell in sorted(SOURCE_CONFIGS):
        dataset, model, adaptation = cell
        config, source_path, source_hash = _derive_config(cell)
        config_path = CONFIG_ROOT / f"{dataset}_{model}_{adaptation}.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        task = "classification" if dataset in {"eurosat", "treesatai"} else "segmentation"
        seeds = [42, 43, 44] if cell in ROBUSTNESS_CELLS else [42]
        for seed in seeds:
            comparator_key = (*cell, seed)
            if comparator_key not in comparators:
                raise ValueError(f"Missing seed-matched deterministic comparator: {comparator_key}")
            rows.append(
                {
                    "task": task,
                    "dataset": dataset,
                    "model": model,
                    "adaptation": adaptation,
                    "seed": seed,
                    "robustness_subset": cell in ROBUSTNESS_CELLS,
                    "experiment": config["experiments"][0]["name"],
                    "config_path": str(config_path.resolve()),
                    "config_sha256": sha256_file(config_path),
                    "source_deterministic_config": str(source_path.resolve()),
                    "source_deterministic_config_sha256": source_hash,
                    "output_root": config["output_dir"],
                    "deterministic_comparator": comparators[comparator_key],
                }
            )
    if len(rows) != 24 or sum(row["seed"] == 42 for row in rows) != 16:
        raise RuntimeError("C5 registry is not the frozen 24-run/16-primary matrix")
    payload = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "ready_gate": True,
        "training_run_count": len(rows),
        "primary_seed42_run_count": sum(row["seed"] == 42 for row in rows),
        "robustness_extra_run_count": sum(row["seed"] in {43, 44} for row in rows),
        "dropout_probability": 0.10,
        "stochastic_passes": 30,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "pilot_sha256": EXPECTED_PILOT_SHA256,
        "pre_uq_sha256": EXPECTED_PRE_UQ_SHA256,
        "training_test_embargo": True,
        "runs": rows,
    }
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return payload


def audit() -> dict[str, Any]:
    _check_frozen_gate()
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    checks = {
        "run_count_24": len(payload["runs"]) == 24,
        "primary_seed42_count_16": sum(row["seed"] == 42 for row in payload["runs"]) == 16,
        "extra_seed_count_8": sum(row["seed"] in {43, 44} for row in payload["runs"]) == 8,
        "config_hashes_valid": all(
            sha256_file(Path(row["config_path"])) == row["config_sha256"] for row in payload["runs"]
        ),
        "deterministic_checkpoints_unchanged": all(
            sha256_file(Path(row["deterministic_comparator"]["checkpoint_path"]))
            == row["deterministic_comparator"]["checkpoint_sha256"]
            for row in payload["runs"]
        ),
        "deterministic_predictions_unchanged": all(
            sha256_file(Path(row["deterministic_comparator"]["prediction_path"]))
            == row["deterministic_comparator"]["prediction_sha256"]
            for row in payload["runs"]
        ),
    }
    checks["passed"] = all(checks.values())
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "audit"))
    args = parser.parse_args()
    value = prepare() if args.command == "prepare" else audit()
    print(json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
