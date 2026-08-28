#!/usr/bin/env python3

import unittest

import numpy as np

from metalrobo.mlx_policy_worker import (
    _difficulty_balanced_retention_weights,
)


class DifficultyBalancedRetentionWeightsTest(unittest.TestCase):
    def test_priority_band_receives_requested_total_authority(self) -> None:
        weights = _difficulty_balanced_retention_weights(
            np.ones(5, dtype=np.float32),
            np.asarray([2, 2, 3, 3, 3], dtype=np.int32),
            priority_band=2,
            priority_factor=1.25,
        )

        priority_total = float(np.sum(weights[:2]))
        other_total = float(np.sum(weights[2:]))
        self.assertAlmostEqual(priority_total / other_total, 1.25, places=6)
        self.assertLessEqual(float(np.max(weights)), 1.0)

    def test_absent_priority_band_balances_represented_bands(self) -> None:
        weights = _difficulty_balanced_retention_weights(
            np.ones(6, dtype=np.float32),
            np.asarray([3, 3, 4, 4, 4, 4], dtype=np.int32),
            priority_band=2,
            priority_factor=1.25,
        )

        self.assertAlmostEqual(float(np.sum(weights[:2])), 2.0, places=6)
        self.assertAlmostEqual(float(np.sum(weights[2:])), 2.0, places=6)
        np.testing.assert_allclose(
            weights,
            np.asarray([1.0, 1.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
