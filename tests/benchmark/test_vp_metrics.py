import unittest

import numpy as np

from netllm_litevlm.evaluation.vp_metrics import (
    corrected_rotation_aware_rmse,
    evaluate_vp_metrics,
    mae,
    mean_angular_error,
    rmse,
    upstream_rmse,
)


class VPMetricsTest(unittest.TestCase):
    def test_linear_metrics(self):
        prediction = np.array([0.0, 2.0, 4.0])
        target = np.array([0.0, 1.0, 2.0])
        self.assertAlmostEqual(mae(prediction, target), 1.0)
        self.assertAlmostEqual(rmse(prediction, target), np.sqrt(5.0 / 3.0))

    def test_rotation_aware_metrics_and_upstream_distinction(self):
        prediction = np.array([359.0, 1.0])
        target = np.array([1.0, 359.0])
        self.assertEqual(mean_angular_error(prediction, target), 2.0)
        self.assertEqual(corrected_rotation_aware_rmse(prediction, target), 2.0)
        self.assertEqual(upstream_rmse(prediction, target, rotation=True), 358.0)

    def test_checkpoint_controls_metric_validity(self):
        invalid = evaluate_vp_metrics(None, None, checkpoint_available=False)
        self.assertFalse(invalid.metric_valid)
        self.assertIsNone(invalid.mae)
        self.assertEqual(
            invalid.metric_invalid_reason, "trained checkpoint unavailable"
        )

        valid = evaluate_vp_metrics(
            np.array([359.0]),
            np.array([1.0]),
            test_loss=0.25,
            checkpoint_available=True,
        )
        self.assertTrue(valid.metric_valid)
        self.assertEqual(valid.mae, 2.0)
        self.assertEqual(valid.rmse, 2.0)
        self.assertEqual(valid.upstream_rmse, 358.0)
        self.assertEqual(valid.test_loss, 0.25)

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            mae(np.zeros((1, 2)), np.zeros((2, 1)))


if __name__ == "__main__":
    unittest.main()
