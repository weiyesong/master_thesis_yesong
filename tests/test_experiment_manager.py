import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.experiment_manager import (
    collect_code_snapshot,
    collect_environment_metadata,
    make_invocation_summary_path,
    make_run_id,
    resolve_config,
    set_global_seed,
    write_code_snapshot_manifest,
    write_resolved_config,
)


class ExperimentManagerTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "seed": 42,
            "data": {
                "split_files": {"train": "train.txt", "val": "val.txt"},
                "channels": ["B04", "B03", "B02"],
                "wavelengths_nm": [665, 560, 490],
            },
            "training": {"epochs": 20, "learning_rate": 1e-3},
            "metrics": {"n_bins": 15},
            "uq_methods": [{"name": "none"}],
            "experiments": [{"name": "unit", "model": {"name": "dofa", "adaptation_mode": "frozen"}}],
            "dry_run_settings": {"epochs": 1, "limit_train_batches": 2, "limit_val_batches": 1},
        }

    def test_seed_repeats_python_numpy_and_torch(self):
        set_global_seed(123)
        first = (random.random(), np.random.rand(), torch.rand(1).item())
        set_global_seed(123)
        second = (random.random(), np.random.rand(), torch.rand(1).item())
        self.assertEqual(first, second)

    def test_arbitrary_integer_seed_is_supported(self):
        seed = 2**80 + 123
        set_global_seed(seed)
        first = (random.random(), np.random.rand(), torch.rand(1).item())
        set_global_seed(seed)
        second = (random.random(), np.random.rand(), torch.rand(1).item())
        self.assertEqual(first, second)

    def test_non_integer_seed_is_rejected(self):
        self.config["seed"] = 42.5
        with self.assertRaisesRegex(ValueError, "seed must be an integer"):
            resolve_config(self.config, Path("config.yaml"))

    def test_panopticon_model_and_full_finetune_are_configurable(self):
        self.config["seed"] = -17
        self.config["data"].update(
            {
                "channels": ["B04", "B03", "B02"],
                "wavelengths_nm": [665, 560, 490],
            }
        )
        self.config["experiments"][0]["model"] = {
            "name": "panopticon",
            "size": "base",
            "adaptation_mode": "full_finetune",
        }
        resolved = resolve_config(self.config, Path("config.yaml"))
        model = resolved["experiments"][0]["model"]
        self.assertEqual(resolved["seed"], -17)
        self.assertEqual(model["name"], "panopticon")
        self.assertFalse(model["freeze_backbone"])
        self.assertEqual(resolved["data"]["wavelengths_nm"], [665.0, 560.0, 490.0])
        self.assertEqual(model["actual_wavelengths"], [665.0, 560.0, 490.0])
        self.assertEqual(model["actual_wavelength_units"], "nanometers")

    def test_so2sat_accepts_only_official_embedded_partition(self):
        self.config["data"] = {
            "name": "so2sat",
            "protocol": "geobench2_m-so2sat",
            "channels": ["B02", "B03", "B04"],
            "wavelengths_nm": [493, 560, 665],
        }
        resolved = resolve_config(self.config, Path("so2sat.yaml"))
        self.assertEqual(resolved["data"]["protocol"], "geobench2_m-so2sat")

    def test_so2sat_rejects_random_or_local_split_definition(self):
        self.config["data"] = {
            "name": "so2sat",
            "protocol": "geobench2_m-so2sat",
            "val_fraction": 0.1,
        }
        with self.assertRaisesRegex(ValueError, "random split"):
            resolve_config(self.config, Path("so2sat.yaml"))

        del self.config["data"]["val_fraction"]
        self.config["data"]["split_files"] = {"train": "train.txt", "val": "val.txt"}
        with self.assertRaisesRegex(ValueError, "embedded"):
            resolve_config(self.config, Path("so2sat.yaml"))

    def test_hardcoded_so2sat_split_counts_are_rejected(self):
        self.config["data"] = {
            "name": "so2sat",
            "protocol": "geobench2_m-so2sat",
            "expected_split_counts": {"train": 1, "val": 1, "test": 1},
        }
        with self.assertRaisesRegex(ValueError, "hard-code"):
            resolve_config(self.config, Path("so2sat.yaml"))

    def test_dofa_resolution_records_converted_model_wavelengths(self):
        self.config["data"].update(
            {"channels": ["B04", "B03", "B02"], "wavelengths_nm": [665, 560, 490]}
        )
        resolved = resolve_config(self.config, Path("config.yaml"))
        model = resolved["experiments"][0]["model"]
        self.assertEqual(model["actual_wavelengths"], [0.665, 0.56, 0.49])
        self.assertEqual(model["actual_wavelength_units"], "micrometers")

    def test_micrometer_values_in_canonical_nm_field_are_rejected(self):
        self.config["data"].update(
            {"channels": ["B04", "B03", "B02"], "wavelengths_nm": [0.665, 0.560, 0.490]}
        )
        with self.assertRaisesRegex(ValueError, "canonical optical nanometers"):
            resolve_config(self.config, Path("config.yaml"))

    def test_explicit_normalization_requires_train_statistics(self):
        self.config["data"].update(
            {
                "channels": ["B04", "B03", "B02"],
                "normalization": {
                    "method": "channelwise_standardization",
                    "statistics_split": "test",
                    "mean": [1.0, 2.0, 3.0],
                    "std": [1.0, 1.0, 1.0],
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "train split only"):
            resolve_config(self.config, Path("config.yaml"))

    def test_run_ids_are_unique_and_include_seed(self):
        first = make_run_id("experiment", 42)
        second = make_run_id("experiment", 42)
        self.assertNotEqual(first, second)
        self.assertIn("seed42", first)

    def test_invocation_summary_paths_are_unique(self):
        first = make_invocation_summary_path(Path("results"), [42, 43, 44])
        second = make_invocation_summary_path(Path("results"), [42, 43, 44])
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, Path("results/summaries"))

    def test_code_snapshot_is_content_addressed_and_excludes_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            (project_root / "scripts").mkdir()
            (project_root / "scripts" / "experiment.py").write_text("value = 1\n", encoding="utf-8")
            (project_root / "configs").mkdir()
            (project_root / "configs" / "run.yaml").write_text("seed: 42\n", encoding="utf-8")
            (project_root / "results").mkdir()
            (project_root / "results" / "ignored.json").write_text('{"volatile": true}\n', encoding="utf-8")

            first = collect_code_snapshot(project_root)
            second = collect_code_snapshot(project_root)
            self.assertEqual(first["code_sha256"], second["code_sha256"])
            self.assertEqual(first["file_count"], 2)
            self.assertNotIn("results/ignored.json", {item["path"] for item in first["files"]})

            (project_root / "scripts" / "experiment.py").write_text("value = 2\n", encoding="utf-8")
            changed = collect_code_snapshot(project_root)
            self.assertNotEqual(first["code_sha256"], changed["code_sha256"])

    def test_new_run_environment_has_valid_code_version(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "project"
            run_dir = Path(directory) / "run"
            project_root.mkdir()
            run_dir.mkdir()
            (project_root / "entrypoint.py").write_text("print('research')\n", encoding="utf-8")
            snapshot = write_code_snapshot_manifest(run_dir, project_root=project_root)
            environment = collect_environment_metadata(code_snapshot=snapshot)
            self.assertRegex(environment["code_version"], r"^sha256:[0-9a-f]{64}$")
            self.assertEqual(environment["code_snapshot"]["file_count"], 1)
            self.assertTrue((run_dir / "code_snapshot.json").is_file())
            self.assertEqual(
                environment["geobench2"]["lock"]["git_commit"],
                "fd9d0b664e6fb0faba54636bdff4906634debd4b",
            )

    def test_dry_run_overrides_and_resolved_config_write(self):
        resolved = resolve_config(self.config, Path("config.yaml"), dry_run=True)
        self.assertEqual(resolved["training"]["epochs"], 1)
        self.assertEqual(resolved["training"]["limit_train_batches"], 2)
        self.assertTrue(resolved["experiments"][0]["model"]["freeze_backbone"])
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            path = write_resolved_config(resolved, run_dir)
            saved = yaml.safe_load(path.read_text())
            self.assertTrue(saved["dry_run"])

    def test_checkpoint_selection_must_use_validation(self):
        self.config["training"]["checkpoint"] = {"split": "test", "metric": "nll", "mode": "min"}
        with self.assertRaisesRegex(ValueError, "validation"):
            resolve_config(self.config, Path("config.yaml"))

    def test_random_augmentation_forbidden_on_calibration_and_test(self):
        self.config["data"]["augmentations"] = {"calibration": "random_crop", "test": "none"}
        with self.assertRaisesRegex(ValueError, "calibration"):
            resolve_config(self.config, Path("config.yaml"))



if __name__ == "__main__":
    unittest.main()
