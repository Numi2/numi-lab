#!/usr/bin/env python3

import unittest

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from metalrobo.mlx_policy_learning import (
    _actor_observation_extension_only_gradients,
)

from metalrobo.mlx_policy_worker import (
    _blend_retention_and_rollout_teacher_targets,
    _difficulty_balanced_retention_weights,
)


class DifficultyBalancedRetentionWeightsTest(unittest.TestCase):
    def test_extension_only_gradients_freeze_inherited_actor(self) -> None:
        gradients = {
            "actor": {
                "layers": [
                    {
                        "weight": mx.ones((2, 5)),
                        "bias": mx.ones((2,)),
                    },
                    {"weight": mx.ones((1, 2))},
                ]
            },
            "critic": {"weight": mx.ones((1, 3))},
            "log_standard_deviation": mx.ones((1,)),
        }

        masked = dict(tree_flatten(
            _actor_observation_extension_only_gradients(gradients, 3, 1)
        ))

        np.testing.assert_array_equal(
            np.asarray(masked["actor.layers.0.weight"]),
            [[0, 0, 0, 1, 0], [0, 0, 0, 1, 0]],
        )
        np.testing.assert_array_equal(
            np.asarray(masked["actor.layers.0.bias"]),
            [0, 0],
        )
        np.testing.assert_array_equal(
            np.asarray(masked["actor.layers.1.weight"]),
            [[0, 0]],
        )
        np.testing.assert_array_equal(
            np.asarray(masked["log_standard_deviation"]),
            [0],
        )
        np.testing.assert_array_equal(
            np.asarray(masked["critic.weight"]),
            [[1, 1, 1]],
        )

    def test_route_teacher_blends_without_erasing_reference_targets(self) -> None:
        reference = np.asarray([[0.0, 0.5], [0.2, -0.2]], dtype=np.float32)
        teacher = np.asarray([[1.0, -0.5], [1.0, 1.0]], dtype=np.float32)
        blended = _blend_retention_and_rollout_teacher_targets(
            reference,
            teacher,
            np.asarray([1.0, 0.0], dtype=np.float32),
            0.1,
        )
        np.testing.assert_allclose(blended[0], [0.1, 0.4])
        np.testing.assert_allclose(blended[1], reference[1])

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
