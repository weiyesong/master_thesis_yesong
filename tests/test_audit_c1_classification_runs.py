from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from scripts.audit_c1_classification_runs import (
    EXPECTED_HASHES,
    LEGACY_RESUME_SOURCE_EPOCH,
    RESUME_CODE_CHANGE_ALLOWLIST,
    RESUME_PROTECTED_PROVENANCE,
    TREESATAI_CHANNELS,
    TREESATAI_CLASSES,
    TREESATAI_MEAN,
    TREESATAI_STD,
    TREESATAI_WAVELENGTHS_NM,
    Target,
    audit_targets,
    csv_rows,
    expected_targets,
    legacy_exact_train_generator_state_sha256,
    markdown_report,
    protocol_errors,
    validate_code_snapshot,
    validate_history,
    validate_resume_provenance,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_snapshot_bundle(directory: Path, files: dict[str, str]) -> dict:
    entries = [{"path": path, "sha256": digest, "size_bytes": 1} for path, digest in sorted(files.items())]
    aggregate = hashlib.sha256()
    for item in entries:
        aggregate.update(item["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(item["sha256"].encode("ascii"))
        aggregate.update(b"\n")
    digest = aggregate.hexdigest()
    snapshot = {
        "schema_version": 1,
        "algorithm": "sha256",
        "code_sha256": digest,
        "file_count": len(entries),
        "files": entries,
    }
    (directory / "code_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (directory / "environment.json").write_text(
        json.dumps(
            {
                "code_version": f"sha256:{digest}",
                "code_snapshot": {"code_sha256": digest, "file_count": len(entries)},
            }
        ),
        encoding="utf-8",
    )
    return snapshot


def make_resume_fixture(run_dir: Path) -> tuple[Target, str, list[dict], dict, dict, dict]:
    target = Target("treesatai", "dofa", "full_finetune", "treesatai_dofa_full_finetune", 42)
    run_id = "synthetic-resumed-run"
    event_dir = run_dir / "resume_events" / "event-1"
    event_dir.mkdir(parents=True)

    unchanged = "1" * 64
    original_files = {
        "scripts/run_experiments.py": "2" * 64,
        "tests/test_experiment_configuration.py": "3" * 64,
        "scripts/unchanged.py": unchanged,
    }
    resume_files = {
        "scripts/run_experiments.py": "4" * 64,
        "tests/test_experiment_configuration.py": "5" * 64,
        "scripts/unchanged.py": unchanged,
    }
    original_snapshot = write_snapshot_bundle(run_dir, original_files)
    resume_snapshot = write_snapshot_bundle(event_dir, resume_files)

    gradient = {
        "performed": True,
        "expected_parameter_tensors": 2,
        "all_expected_parameters_received_gradient": True,
        "missing_gradient_parameters": [],
        "zero_gradient_parameters": [],
        "nonfinite_gradient_parameters": [],
        "epoch": 1,
    }
    for name in RESUME_PROTECTED_PROVENANCE:
        path = run_dir / name
        if name in {"code_snapshot.json", "environment.json"}:
            continue
        if name == "gradient_audit.json":
            path.write_text(json.dumps(gradient), encoding="utf-8")
        else:
            path.write_text(f"protected {name}\n", encoding="utf-8")

    history = []
    for epoch in range(1, 7):
        epoch_gradient = {key: value for key, value in gradient.items() if key != "epoch"}
        if epoch != 1:
            epoch_gradient = {
                "performed": False,
                "all_expected_parameters_received_gradient": None,
                "missing_gradient_parameters": [],
                "zero_gradient_parameters": [],
                "nonfinite_gradient_parameters": [],
            }
        history.append(
            {
                "epoch": epoch,
                "train": {"gradient_audit": epoch_gradient},
                "val": {"nll": 1.0 / epoch},
                "backbone_learning_rate": 1.0e-4,
            }
        )
    batches_per_epoch = 63
    batches = []
    for epoch in range(1, 7):
        for batch in range(1, batches_per_epoch + 1):
            batches.append(
                {
                    "epoch": epoch,
                    "batch": batch,
                    "global_step": len(batches) + 1,
                    "loss": 1.0 / epoch,
                }
            )

    source_path = event_dir / f"source_last_epoch{LEGACY_RESUME_SOURCE_EPOCH}.pt"
    torch.save(
        {
            "run_id": run_id,
            "epoch": 5,
            "best_epoch": 5,
            "best_metric": 0.2,
            "epochs_without_improvement": 0,
            "selection_metric": "nll",
            "config": treesatai_full_config(),
            "history": history[:5],
            "batch_history": batches[: 5 * batches_per_epoch],
            "model": {},
        },
        source_path,
    )
    final_last_path = run_dir / "last.pt"
    torch.save({"history": history, "batch_history": batches}, final_last_path)
    pd.DataFrame(batches).to_csv(run_dir / "batch_metrics.csv", index=False)

    protected = {name: file_sha256(run_dir / name) for name in RESUME_PROTECTED_PROVENANCE}
    request = {
        "status": "started",
        "run_id": run_id,
        "seed": target.seed,
        "experiment": target.experiment,
        "dataset": target.dataset,
        "model": target.model,
        "adaptation": target.adaptation,
        "source_checkpoint": {
            "path": str(final_last_path),
            "archived_path": str(source_path),
            "sha256": file_sha256(source_path),
            "size_bytes": source_path.stat().st_size,
            "epoch": 5,
            "best_epoch": 5,
            "best_metric": 0.2,
            "epochs_without_improvement": 0,
        },
        "next_epoch": 6,
        "original_code_sha256": original_snapshot["code_sha256"],
        "resume_code_sha256": resume_snapshot["code_sha256"],
        "code_changes": [
            {
                "path": path,
                "before_sha256": original_files[path],
                "after_sha256": resume_files[path],
            }
            for path in sorted(RESUME_CODE_CHANGE_ALLOWLIST)
        ],
        "protected_provenance_sha256": protected,
        "config_verified_exact": True,
        "input_artifacts_verified_exact": True,
        "model_audit_verified_exact": True,
        "optimizer_groups_verified_exact": True,
        "gradient_audit_preserved": True,
        "generator_restore_method": "legacy_exact_sampler_fast_forward",
        "train_generator_state_sha256": legacy_exact_train_generator_state_sha256(42, 4000, 5),
        "train_dataset_length": 4000,
        "partial_epoch_policy": "discard_uncheckpointed_work_and_restart_next_epoch",
        "test_access": False,
    }
    (event_dir / "resume_request.json").write_text(json.dumps(request), encoding="utf-8")
    checkpoint = {
        "best": {"sha256": "6" * 64},
        "last": {"sha256": file_sha256(final_last_path)},
    }
    curve = {"best_epoch": 6, "best_validation_nll": 1.0 / 6}
    completion = {
        "status": "completed",
        "run_id": run_id,
        "resumed_from_epoch": 6,
        "last_epoch": 6,
        "best_epoch": 6,
        "best_metric": 1.0 / 6,
        "best_checkpoint_sha256": checkpoint["best"]["sha256"],
        "last_checkpoint_sha256": checkpoint["last"]["sha256"],
        "final_test_export": str(run_dir / "predictions" / "test" / "deterministic"),
        "test_access_during_promotion": False,
    }
    (event_dir / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
    summary = {
        "training_segments": {"resumed_from_epoch": 6},
        "resume_event": str(event_dir),
    }
    return target, run_id, history, curve, checkpoint, summary


def treesatai_full_config() -> dict:
    wavelengths = list(TREESATAI_WAVELENGTHS_NM)
    return {
        "seed": 42,
        "dry_run": False,
        "task": "classification_multilabel",
        "provenance": {
            "require_input_hashes": True,
            "protocol_id": "B5-2026-08-15-v1",
            "protocol_path": "./reports/final_training_protocol.md",
            "protocol_sha256": EXPECTED_HASHES["training_protocol"],
        },
        "active_experiment": "treesatai_dofa_full_finetune",
        "data": {
            "name": "treesatai",
            "protocol": "geobench2_treesatai",
            "source": "GEO-Bench-2@fd9d0b664e6fb0faba54636bdff4906634debd4b",
            "artifact_sha256": EXPECTED_HASHES["treesatai_artifact"],
            "extracted_manifest_sha256": EXPECTED_HASHES["treesatai_manifest"],
            "input_bands": "s2",
            "channels": list(TREESATAI_CHANNELS),
            "wavelengths_nm": wavelengths,
            "image_size": 224,
            "temporal_protocol": "shared_encoder_per_timestamp_then_mean_features",
            "multilabel": True,
            "num_classes": 15,
            "class_names": list(TREESATAI_CLASSES),
            "normalize": True,
            "normalization": {
                "method": "channelwise_standardization",
                "statistics_split": "train",
                "mean": list(TREESATAI_MEAN),
                "std": list(TREESATAI_STD),
            },
            "augmentations": {"train": "none", "val": "none", "calibration": "none", "test": "none"},
        },
        "training": {
            "epochs": 100,
            "batch_size": 64,
            "optimizer": {"name": "adamw"},
            "learning_rate": 1.0e-4,
            "backbone_learning_rate": 1.0e-4,
            "head_learning_rate": 1.0e-3,
            "weight_decay": 0.01,
            "warmup": {"enabled": True, "epochs": 5, "type": "linear"},
            "gradient_clipping": {"enabled": False},
            "layerwise_lr_decay": {"enabled": False},
            "early_stopping": {"enabled": True, "patience": 15, "min_delta": 0.0},
            "checkpoint": {"split": "val", "metric": "nll", "mode": "min"},
            "limit_train_batches": None,
            "limit_val_batches": None,
        },
        "metrics": {"n_bins": 15, "multilabel_threshold": 0.5},
        "uq_methods": [{"name": "none"}],
        "reproducibility": {"deterministic": True, "warn_only": False},
        "prediction_export": {
            "enabled": True,
            "splits": ["test"],
            "save_backbone_representation": True,
            "save_embeddings": True,
        },
        "experiments": [
            {
                "name": "treesatai_dofa_full_finetune",
                "model": {
                    "name": "dofa",
                    "size": "base",
                    "pretrained": True,
                    "adaptation_mode": "full_finetune",
                    "actual_wavelengths": [value / 1000.0 for value in wavelengths],
                    "actual_wavelength_units": "micrometers",
                    "weights_sha256": EXPECTED_HASHES["dofa_weights"],
                    "head": {"architecture": "linear", "dropout": 0.0},
                },
            }
        ],
    }


class C1AuditUtilityTests(unittest.TestCase):
    def test_target_matrix_supports_seed42_then_all_final_seeds(self):
        seed42 = expected_targets([42])
        self.assertEqual(len(seed42), 6)
        self.assertEqual({target.seed for target in seed42}, {42})
        all_seeds = expected_targets([42, 43, 44])
        self.assertEqual(len(all_seeds), 18)
        with self.assertRaises(ValueError):
            expected_targets([41])

    def test_frozen_b5_protocol_is_checked_exactly(self):
        config = treesatai_full_config()
        target = Target("treesatai", "dofa", "full_finetune", "treesatai_dofa_full_finetune", 42)
        self.assertEqual(protocol_errors(config, target), [])
        config["training"]["batch_size"] = 32
        errors = protocol_errors(config, target)
        self.assertTrue(any("batch_size" in error for error in errors))

    def test_history_uses_earliest_minimum_and_exact_early_stop(self):
        config = treesatai_full_config()
        config["training"]["early_stopping"]["patience"] = 2
        target = Target("treesatai", "dofa", "full_finetune", "treesatai_dofa_full_finetune", 42)
        nll_values = [1.0, 0.8, 0.9, 1.1]
        history = []
        for epoch, nll in enumerate(nll_values, start=1):
            factor = min(1.0, epoch / 5.0)
            history.append(
                {
                    "epoch": epoch,
                    "train": {
                        "loss": 1.0,
                        "accuracy": 0.1,
                        "gradient_norm": 1.0,
                        "backbone_gradient_norm": 0.5,
                        "head_gradient_norm": 0.5,
                    },
                    "val": {
                        "accuracy": 0.1,
                        "labelwise_accuracy": 0.5,
                        "macro_f1": 0.1,
                        "nll": nll,
                        "ece": 0.1,
                        "brier": 0.2,
                        "mean_confidence": 0.6,
                        "predictive_entropy": 2.0,
                    },
                    "warmup_factor": factor,
                    "backbone_learning_rate": 1.0e-4 * factor,
                    "head_learning_rate": 1.0e-3 * factor,
                }
            )
        errors: list[str] = []
        curve = validate_history(history, {"best_epoch": 2}, config, target, errors)
        self.assertEqual(errors, [])
        self.assertEqual(curve["best_epoch"], 2)
        self.assertEqual(curve["termination"], "early_stopping")

    def test_code_snapshot_aggregate_and_environment_must_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            file_hash = "a" * 64
            aggregate = hashlib.sha256()
            aggregate.update(b"scripts/example.py\0")
            aggregate.update(file_hash.encode("ascii"))
            aggregate.update(b"\n")
            digest = aggregate.hexdigest()
            (run_dir / "code_snapshot.json").write_text(
                json.dumps(
                    {
                        "code_sha256": digest,
                        "files": [{"path": "scripts/example.py", "sha256": file_hash, "size_bytes": 1}],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "environment.json").write_text(
                json.dumps({"code_version": f"sha256:{digest}"}), encoding="utf-8"
            )
            errors: list[str] = []
            evidence = validate_code_snapshot(run_dir, errors)
            self.assertEqual(errors, [])
            self.assertEqual(evidence["code_sha256"], digest)

    def test_non_resumed_run_keeps_ordinary_audit_path(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Target("eurosat", "panopticon", "frozen", "eurosat_panopticon_frozen_bnlinear", 42)
            errors: list[str] = []
            evidence = validate_resume_provenance(
                Path(directory),
                "ordinary-run",
                target,
                {},
                [],
                {},
                {},
                errors,
            )
            self.assertEqual(errors, [])
            self.assertEqual(evidence, {"resumed": False, "event_count": 0})

    def test_legacy_sampler_fast_forward_matches_recorded_real_run_state(self):
        self.assertEqual(
            legacy_exact_train_generator_state_sha256(43, 18866, 5),
            "3a1d82cd9cdcd35b58f07f114e70d5d404ebcd98be512a823c75961df7331904",
        )

    def test_complete_resume_chain_is_accepted_and_unallowlisted_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            target, run_id, history, curve, checkpoint, summary = make_resume_fixture(run_dir)
            errors: list[str] = []
            evidence = validate_resume_provenance(
                run_dir,
                run_id,
                target,
                summary,
                history,
                curve,
                checkpoint,
                errors,
            )
            self.assertEqual(errors, [])
            self.assertTrue(evidence["resumed"])
            self.assertTrue(evidence["combined_history_valid"])
            self.assertTrue(evidence["combined_batches_valid"])
            self.assertEqual(evidence["completion_status"], "completed")

            event_dir = Path(summary["resume_event"])
            resume_snapshot = json.loads((event_dir / "code_snapshot.json").read_text(encoding="utf-8"))
            files = {item["path"]: item["sha256"] for item in resume_snapshot["files"]}
            files["scripts/unchanged.py"] = "7" * 64
            updated = write_snapshot_bundle(event_dir, files)
            request_path = event_dir / "resume_request.json"
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["resume_code_sha256"] = updated["code_sha256"]
            request["code_changes"].append(
                {
                    "path": "scripts/unchanged.py",
                    "before_sha256": "1" * 64,
                    "after_sha256": "7" * 64,
                }
            )
            request_path.write_text(json.dumps(request), encoding="utf-8")
            errors = []
            validate_resume_provenance(
                run_dir,
                run_id,
                target,
                summary,
                history,
                curve,
                checkpoint,
                errors,
            )
            self.assertTrue(any("allowlist" in error for error in errors))

    def test_missing_runs_are_not_promoted(self):
        targets = expected_targets([42])
        results = audit_targets(Path("/workspace"), targets, {target.key: [] for target in targets})
        self.assertEqual({item["status"] for item in results}, {"MISSING"})
        self.assertFalse(any(item["promotion_eligible"] for item in results))

    def test_test_metric_values_are_report_only(self):
        target = Target("eurosat", "panopticon", "frozen", "eurosat_panopticon_frozen_bnlinear", 42)
        run = {
            "target": target.__dict__,
            "target_key": target.key,
            "run_id": "valid-run",
            "run_dir": "/tmp/valid-run",
            "status": "PROMOTABLE",
            "promotion_eligible": True,
            "errors": [],
            "validation_curve": {"best_validation_nll": 1.0},
            "test_metrics_report_only": {"accuracy": 0.0, "macro_f1": 0.0, "nll": 100.0, "brier": 1.0, "ece": 1.0},
        }
        row = csv_rows([run])[0]
        self.assertTrue(row["promotion_eligible"])
        self.assertEqual(row["reported_test_accuracy_excluded"], 0.0)
        payload = {
            "created_at_utc": "now",
            "summary": {"promotable": 1, "blocked": 0, "missing": 0, "ambiguous": 0},
            "runs": [run],
        }
        report = markdown_report(payload)
        self.assertIn("excluded", report.lower())
        self.assertIn("PROMOTABLE", report)


if __name__ == "__main__":
    unittest.main()
