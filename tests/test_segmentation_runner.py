import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scripts.run_experiments import run_segmentation_experiment
from scripts.segmentation_pipeline import CommonUNetDecoder, FoundationSegmentationModel


class _ToyDenseAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, image):
        return torch.nn.functional.adaptive_avg_pool2d(self.backbone(image), (2, 2))


class _ToyDataset(Dataset):
    def __init__(self, split):
        self.split = split

    def __len__(self):
        return 2

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index + 11)
        return {
            "image": torch.randn(3, 8, 8, generator=generator),
            "mask": torch.randint(0, 2, (8, 8), generator=generator),
            "sample_id": f"{self.split}_{index}",
        }


class SegmentationRunnerTests(unittest.TestCase):
    def test_runner_writes_early_stopping_checkpoints_metrics_and_full_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "seed": 42,
                "seeds": [42],
                "device": "cpu",
                "output_dir": str(root / "results"),
                "results_csv": str(root / "results.csv"),
                "task": "segmentation",
                "data": {
                    "name": "spacenet7",
                    "class_names": ["background", "building"],
                    "num_classes": 2,
                    "ignore_index": 255,
                    "foreground_class_index": 1,
                },
                "training": {
                    "epochs": 2,
                    "batch_size": 2,
                    "optimizer": {"name": "adamw"},
                    "backbone_learning_rate": 1e-4,
                    "head_learning_rate": 1e-3,
                    "weight_decay": 0.01,
                    "warmup": {"enabled": False, "epochs": 0},
                    "early_stopping": {"enabled": True, "patience": 1, "min_delta": 0.0},
                    "checkpoint": {"split": "val", "metric": "miou", "mode": "max"},
                    "limit_train_batches": None,
                    "limit_val_batches": None,
                },
                "metrics": {"n_bins": 15, "boundary_radius": 1},
                "reproducibility": {"deterministic": True, "warn_only": False},
                "prediction_export": {"enabled": True, "splits": ["test"]},
                "dry_run": False,
                "experiments": [],
            }
            experiment = {
                "name": "spacenet7_toy_frozen",
                "task": "segmentation",
                "model": {
                    "name": "dofa",
                    "adaptation_mode": "frozen",
                    "freeze_backbone": True,
                    "expected_frozen_backbone_parameters": [],
                },
            }
            config["experiments"] = [experiment]

            def loaders(_config, generator=None):
                return {
                    split: DataLoader(
                        _ToyDataset(split), batch_size=2, shuffle=split == "train", generator=generator
                    )
                    for split in ("train", "val", "test")
                }

            def model_factory(_config, _model_config):
                return FoundationSegmentationModel(
                    _ToyDenseAdapter(), CommonUNetDecoder(4, 2, decoder_channels=(8, 4)), True
                )

            with patch("scripts.run_experiments.make_segmentation_dataloaders", side_effect=loaders), patch(
                "scripts.run_experiments.build_segmentation_model", side_effect=model_factory
            ), patch(
                "scripts.run_experiments.collect_input_artifact_provenance",
                return_value={"required": False, "all_verified": False, "records": {}},
            ):
                result = run_segmentation_experiment(config, experiment)

            run_dir = Path(result.checkpoint_path).parent
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            self.assertFalse(summary["test_access_during_training_or_selection"])
            self.assertTrue((run_dir / "best.pt").is_file())
            self.assertTrue((run_dir / "last.pt").is_file())
            export = run_dir / "predictions" / "test" / "deterministic"
            for name in ("predictions.npz", "representations.npz", "per_image_metrics.csv", "manifest.json"):
                self.assertTrue((export / name).is_file(), name)
            manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["representations"]["shape"], [2, 4])


if __name__ == "__main__":
    unittest.main()
