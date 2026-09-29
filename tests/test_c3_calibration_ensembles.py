import unittest

import numpy as np

from scripts.c3_calibration_ensembles import (
    classification_metrics,
    fit_positive_multilabel_temperature,
    fit_positive_temperature,
    multilabel_reliability_bins,
    stable_sigmoid,
    stable_softmax,
)


class C3CalibrationEnsembleTests(unittest.TestCase):
    def test_positive_temperature_improves_overconfident_logits(self):
        logits = np.asarray(
            [[8.0, -8.0], [8.0, -8.0], [-8.0, 8.0], [-8.0, 8.0]], dtype=np.float64
        )
        labels = np.asarray([0, 1, 1, 1], dtype=np.int64)
        fit = fit_positive_temperature(logits, labels)
        self.assertGreater(fit["temperature"], 0.0)
        self.assertLessEqual(fit["calibration_nll_after"], fit["calibration_nll_before"])

    def test_multiclass_probability_mean_is_not_logit_mean(self):
        first = np.asarray([[9.0, 0.0, 0.0]])
        second = np.asarray([[0.0, 2.0, 0.0]])
        probability_mean = (stable_softmax(first) + stable_softmax(second)) / 2.0
        logit_mean_probability = stable_softmax((first + second) / 2.0)
        self.assertFalse(np.allclose(probability_mean, logit_mean_probability))
        self.assertAlmostEqual(float(probability_mean.sum()), 1.0)

    def test_multilabel_metrics_use_binary_decision_conventions(self):
        probabilities = np.asarray([[0.9, 0.2], [0.4, 0.8]], dtype=np.float64)
        labels = np.asarray([[1, 0], [0, 1]], dtype=np.int64)
        metrics = classification_metrics(probabilities, labels, "multilabel")
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertGreater(metrics["nll"], 0.0)
        self.assertGreaterEqual(metrics["ece_15"], 0.0)

    def test_multilabel_positive_temperature_preserves_predictions_and_argmax(self):
        logits = np.asarray(
            [[4.0, -2.0, 1.0], [-3.0, 2.0, -0.5], [1.5, -1.0, 3.0]], dtype=np.float64
        )
        labels = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=np.int64)
        fit = fit_positive_multilabel_temperature(logits, labels)
        self.assertGreater(fit["temperature"], 0.0)
        self.assertLessEqual(fit["fit_nll_after"], fit["fit_nll_before"])
        raw = stable_sigmoid(logits)
        calibrated = stable_sigmoid(logits / fit["temperature"])
        np.testing.assert_array_equal(raw >= 0.5, calibrated >= 0.5)
        np.testing.assert_array_equal(raw.argmax(axis=1), calibrated.argmax(axis=1))

    def test_multilabel_reliability_ece_matches_metric_convention(self):
        probabilities = np.asarray([[0.9, 0.2], [0.4, 0.8]], dtype=np.float64)
        labels = np.asarray([[1, 0], [0, 1]], dtype=np.int64)
        bins = multilabel_reliability_bins(probabilities, labels)
        metrics = classification_metrics(probabilities, labels, "multilabel")
        self.assertAlmostEqual(bins["ece_15"], metrics["ece_15"], places=14)


if __name__ == "__main__":
    unittest.main()
