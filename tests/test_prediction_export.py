import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.calibration import compute_classification_metrics
from scripts.prediction_export import (
    export_predictions,
    export_stochastic_predictions,
    load_prediction_export,
    recompute_metrics,
    save_evaluation_artifacts,
    validate_prediction_export,
)


class PredictionExportTests(unittest.TestCase):
    def metadata(self):
        return {
            "sample_ids": ["sample_3", "sample_1", "sample_4", "sample_0", "sample_2"],
            "true_labels": np.array([0, 1, 2, 1, 0]),
            "class_names": ["zero", "one", "two"],
            "model_name": "toy-model",
            "dataset": "toy-data",
            "adaptation_mode": "frozen",
            "seed": 42,
            "checkpoint": "toy.pt",
            "split": "val",
            "uq_method": "deterministic",
        }

    def test_deterministic_roundtrip_and_metric_recomputation(self):
        logits = torch.tensor(
            [[3.0, 1.0, -1.0], [0.1, 2.0, 0.2], [-2.0, 0.0, 2.0], [0.0, 0.5, 0.4], [1.0, 0.8, -0.2]]
        )
        embeddings = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prediction_export"
            export_predictions(output, logits=logits, embeddings=embeddings, expected_count=5, **self.metadata())
            bundle = load_prediction_export(output)
            self.assertEqual(bundle.table["sample_id"].tolist(), self.metadata()["sample_ids"])
            self.assertEqual(bundle.embeddings["embeddings"].shape, (5, 4))
            required = {
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
            self.assertTrue(required <= set(bundle.table.columns))
            np.testing.assert_allclose(
                np.asarray(bundle.table["backbone_representation"].tolist()), embeddings.numpy()
            )
            report = validate_prediction_export(output, expected_count=5)
            self.assertTrue(report["valid"], report)
            expected = compute_classification_metrics(logits, torch.tensor(self.metadata()["true_labels"]), n_bins=15).to_dict()
            actual = recompute_metrics(output, n_bins=15)
            for name in ("accuracy", "nll", "brier", "ece"):
                self.assertAlmostEqual(actual[name], expected[name], places=6)
            saved = save_evaluation_artifacts(output, n_bins=15)
            self.assertEqual(len(saved["per_class"]), 3)
            self.assertEqual(np.asarray(saved["confusion_matrix"]).shape, (3, 3))
            for filename in ("metrics.json", "per_class_metrics.json", "confusion_matrix.npy", "confusion_matrix.csv"):
                self.assertTrue((output / filename).is_file())

    def test_stochastic_roundtrip_preserves_every_member(self):
        raw_logits = torch.arange(60, dtype=torch.float32).reshape(5, 4, 3) / 10
        metadata = self.metadata()
        metadata["uq_method"] = "mc_dropout"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stochastic_export"
            export_stochastic_predictions(
                output,
                stochastic_logits=raw_logits,
                embeddings=torch.ones(5, 4),
                expected_count=5,
                **metadata,
            )
            bundle = load_prediction_export(output)
            self.assertEqual(bundle.stochastic["logits"].shape, (5, 4, 3))
            np.testing.assert_allclose(bundle.stochastic["logits"], raw_logits.numpy())
            self.assertTrue(validate_prediction_export(output, expected_count=5)["valid"])

    def test_multilabel_export_uses_sigmoid_and_exact_match_semantics(self):
        metadata = self.metadata()
        metadata["true_labels"] = np.array(
            [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int64
        )
        logits = torch.tensor(
            [[2.0, -1.0, 0.1], [-2.0, 3.0, -1.0], [1.0, 1.0, -2.0], [-1.0, -1.0, 2.0], [0.0, -2.0, -3.0]]
        )
        embeddings = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "multilabel_export"
            export_predictions(
                output,
                logits=logits,
                embeddings=embeddings,
                expected_count=5,
                classification_type="multilabel",
                multilabel_threshold=0.5,
                **metadata,
            )
            bundle = load_prediction_export(output)
            probabilities = np.asarray(bundle.table["probabilities"].tolist())
            np.testing.assert_allclose(probabilities, torch.sigmoid(logits).numpy(), atol=1e-6)
            np.testing.assert_array_equal(bundle.table["prediction"].iloc[0], np.array([1, 0, 1]))
            self.assertEqual(bundle.manifest["classification_type"], "multilabel")
            self.assertTrue(validate_prediction_export(output, expected_count=5)["valid"])
            metrics = save_evaluation_artifacts(output, n_bins=15)
            self.assertEqual(metrics["classification_type"], "multilabel")
            self.assertEqual(np.asarray(metrics["confusion_matrix"]).shape, (3, 2, 2))

    def test_duplicate_sample_id_is_rejected(self):
        metadata = self.metadata()
        metadata["sample_ids"][1] = metadata["sample_ids"][0]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unique"):
                export_predictions(Path(directory) / "invalid", logits=torch.zeros(5, 3), **metadata)


if __name__ == "__main__":
    unittest.main()
