import unittest

import numpy as np

from scripts.c3_calibration_ensembles import classification_metrics
from scripts.c6_build_final_results import (
    ADAPTATIONS, DATASETS_CLASS, DATASETS_SEG, MODELS,
    bin_ece, calibration_bins, classification_bins, na_temperature_row, temperature_scope,
)


class FinalReportingTests(unittest.TestCase):
    def test_exact_edges_are_right_closed_and_empty_bins_preserved(self):
        data = calibration_bins(np.array([0., .25, .5, .75, 1.]), np.array([0, 1, 0, 1, 1]), bins=4)
        np.testing.assert_array_equal(data['count'], [2, 1, 1, 1])
        self.assertAlmostEqual(data['mean_confidence'][0], .125)
        empty = calibration_bins(np.array([1.]), np.array([1.]), bins=4)
        np.testing.assert_array_equal(empty['count'], [0, 0, 0, 1])
        self.assertTrue(np.isnan(empty['mean_confidence'][:3]).all())

    def test_multilabel_main_plot_matches_decision_ece_and_not_positive_diagnostic(self):
        probabilities = np.array([[.1, .55], [.2, .45], [.9, .25]], dtype=np.float64)
        labels = np.array([[0, 1], [1, 0], [0, 1]])
        data = classification_bins(labels, probabilities, multilabel=True)
        expected = classification_metrics(probabilities, labels, 'multilabel')['ece_15']
        self.assertAlmostEqual(bin_ece(data), expected, places=14)
        self.assertTrue((data['mean_confidence'][data['count'] > 0] >= .5).all())
        positive = classification_bins(labels, probabilities, multilabel=True, positive_label=True)
        self.assertNotAlmostEqual(bin_ece(data), bin_ece(positive), places=7)
        self.assertEqual(int(data['count'].sum()), labels.size)

    def test_segmentation_top_confidence_is_not_binary_transformed(self):
        data = calibration_bins(np.array([.3, .4]), np.array([0, 1]))
        self.assertAlmostEqual(np.nansum(data['mean_confidence'] * data['count']) / 2, .35)
        self.assertAlmostEqual(bin_ece(data), .45)

    def test_scope_respects_master_na_even_when_historical_results_exist(self):
        rows = []
        for dataset in (*DATASETS_CLASS, *DATASETS_SEG):
            for model in MODELS:
                for adaptation in ADAPTATIONS:
                    applicable = dataset == 'eurosat'
                    rows.append(dict(dataset=dataset, model=model, adaptation=adaptation,
                                     task='classification' if dataset in DATASETS_CLASS else 'segmentation',
                                     uq_method='temperature_scaling',
                                     record_type='MEAN_STD' if applicable else 'NOT_APPLICABLE',
                                     status='COMPLETE' if applicable else 'N/A', applicability_reason='dated policy'))
        scope = temperature_scope(rows)
        self.assertEqual(sum(row['status'] == 'N/A' for row in scope.values()), 12)
        excluded = na_temperature_row(('treesatai', 'dofa', 'frozen'), scope)
        self.assertEqual(excluded['status'], 'N/A')
        self.assertEqual(excluded['n_runs'], 0)
        self.assertNotIn('ece_15', excluded)
        with self.assertRaises(RuntimeError):
            temperature_scope(rows + [rows[0]])
        with self.assertRaises(RuntimeError):
            temperature_scope(rows[:-1])


if __name__ == '__main__':
    unittest.main()
