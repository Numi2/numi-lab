"""Regression tests for physics-error isolation in HyperPolicy training."""

import unittest
import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None

if mx is not None:
    from metalrobo.hyperpolicy.training_pipeline import (
        _pad_motion_batch,
        _teacher_weights,
        specialist_batch_from_rollout,
    )
    from metalrobo.hyperpolicy.mlx_specialist import (
        _weighted_mean,
        MLXSpecialistAdapterLearner,
    )
    from metalrobo.hyperpolicy.base import HyperBaseLayer, HyperBasePolicy
    from metalrobo.mlx_policy_learning import NativePolicyRollout, _TRANSITION_DTYPE
    from metalrobo.hyperpolicy.common import _ACTIVATION_IDENTITY
    from metalrobo.hyperpolicy.mlx_model import (
        ARDYHyperPolicyConfiguration,
        PhaseVaryingLowRankActor,
    )


@unittest.skipIf(mx is None, "MLX is not installed")
class HyperPolicyPhysicsErrorTest(unittest.TestCase):
    def setUp(self):
        self.previous_device = mx.default_device()
        mx.set_default_device(mx.gpu if os.environ.get("NUMI_TEST_MLX_DEVICE") == "gpu" else mx.cpu)

    def tearDown(self):
        mx.set_default_device(self.previous_device)

    @staticmethod
    def rollout(rewards, tracking=None, errors=None, done=None, timeout=None):
        count = len(rewards)
        transitions = np.zeros(count, dtype=[
            ("reward", "f4"), ("physics_error", "u4"),
            ("done", "u4"), ("timeout", "u4"),
        ])
        transitions["reward"] = rewards
        if errors is not None:
            transitions["physics_error"] = errors
        if done is not None:
            transitions["done"] = done
        if timeout is not None:
            transitions["timeout"] = timeout
        outcomes = {} if tracking is None else {
            "tracking_score": np.asarray(tracking, dtype=np.float32)
        }
        return SimpleNamespace(transitions=transitions, outcomes=outcomes)

    def test_failed_reward_and_tracking_do_not_change_accepted_weights(self):
        first = self.rollout([1, 2, 3, 4], [0.2, 0.4, 0.6, 0.8],
                             errors=[0, 1, 0, 0])
        second = self.rollout([1, 2000, 3, 4], [0.2, -500, 0.6, 0.8],
                              errors=[0, 1, 0, 0])
        weights = _teacher_weights(first)
        changed = _teacher_weights(second)
        np.testing.assert_allclose(weights[[0, 2, 3]], changed[[0, 2, 3]])
        self.assertEqual(float(weights[1]), 0.0)
        self.assertEqual(float(changed[1]), 0.0)

    def test_failed_reward_does_not_change_reward_only_ranking(self):
        first = self.rollout([1, 2, 3, 4], errors=[0, 1, 0, 0])
        second = self.rollout([1, 2000, 3, 4], errors=[0, 1, 0, 0])
        np.testing.assert_allclose(_teacher_weights(first)[[0, 2, 3]],
                                   _teacher_weights(second)[[0, 2, 3]])

    def test_termination_is_excluded_and_timeout_is_included(self):
        baseline = self.rollout([1, 2, 3, 4], [0.2, 0.4, 0.6, 0.8],
                                done=[0, 1, 1, 0], timeout=[0, 0, 1, 0])
        extreme = self.rollout([1, -2000, 3, 4], [0.2, 500, 0.6, 0.8],
                               done=[0, 1, 1, 0], timeout=[0, 0, 1, 0])
        a, b = _teacher_weights(baseline), _teacher_weights(extreme)
        self.assertEqual(float(a[1]), 0)
        self.assertGreater(float(a[2]), 0)
        np.testing.assert_allclose(a[[0, 2, 3]], b[[0, 2, 3]])

    def test_empty_valid_and_single_valid(self):
        invalid = self.rollout([1, 2], errors=[1, 1])
        np.testing.assert_array_equal(_teacher_weights(invalid), [0, 0])
        one = self.rollout([1, 2], [0.2, 100], errors=[0, 1])
        self.assertAlmostEqual(float(_teacher_weights(one)[0]), 0.45, places=6)

    def test_clean_rewards_and_tracking_retain_original_semantics(self):
        clean = self.rollout([1, 2, 3], [0.2, 0.4, 0.6])
        np.testing.assert_allclose(_teacher_weights(clean), [0.1, 0.55, 1.0],
                                   atol=1e-6)

    @staticmethod
    def training_fixture(error):
        rollout = HyperPolicyPhysicsErrorTest.rollout(
            [1, 2, 3, 4], [0.2, 0.4, 0.6, 0.8], errors=[0, error, 0, 0]
        )
        rollout.sample_count = 4
        rollout.action_count = 1
        rollout.actor_observation_count = 1
        rollout.actor_observations = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
        rollout.teacher_actions = np.array([[1], [2], [3], [4]], dtype=np.float32)
        rollout.outcomes["hyper_policy_phase"] = np.array([0, 0.25, 0.5, 1], dtype=np.float32)
        motion = SimpleNamespace(
            frame_count=2, features=np.zeros((2, 1), np.float32),
            knot_phases=np.array([0, 1], np.float32),
            knot_event_features=np.zeros((2, 1), np.float32),
            phases=np.array([0, 1], np.float32), validate=lambda: None,
        )
        policy = SimpleNamespace(
            action_count=1, knot_phases=np.array([0, 1], np.float32),
            reference_phases=np.array([0, 1], np.float32),
            reference_actions=np.array([[0], [1]], np.float32),
            validate=lambda: None,
        )
        specialist = SimpleNamespace(
            coefficients=np.zeros((1, 2, 1), np.float32),
            authority=np.zeros((1, 2, 1), np.float32),
            phase_rate_multiplier=np.ones((1, 2), np.float32),
        )
        config = SimpleNamespace(maximum_frames=2, feature_count=1,
                                 event_feature_count=1, coefficient_count=1,
                                 action_count=1)
        item = SimpleNamespace(motion=motion, generated_policy=policy,
                               specialist=specialist, rollout=rollout, qualified=True)
        return rollout, policy, config, item

    def test_specialist_batch_and_mlx_loss_ignore_failed_row(self):
        rollout, policy, config, _ = self.training_fixture(1)
        baseline = specialist_batch_from_rollout(
            rollout=rollout, motion_policy=policy, configuration=config
        )
        rollout.transitions["reward"][1] = -1000
        rollout.outcomes["tracking_score"][1] = 1000
        rollout.teacher_actions[1] = -1e30
        rollout.actor_observations[1] = 1e30
        changed = specialist_batch_from_rollout(
            rollout=rollout, motion_policy=policy, configuration=config
        )
        def loss(batch):
            squared = mx.square(mx.array(batch.teacher_actions[:, 0]))
            return float(_weighted_mean(squared, mx.array(batch.teacher_weights)).item())
        self.assertEqual(loss(baseline), loss(changed))
        np.testing.assert_array_equal(baseline.teacher_weights, changed.teacher_weights)
        self.assertEqual(changed.teacher_weights.shape, (3,))
        np.testing.assert_array_equal(baseline.actor_observations, changed.actor_observations)
        np.testing.assert_array_equal(baseline.teacher_actions, changed.teacher_actions)

    def test_specialist_refuses_when_no_sample_is_physically_valid(self):
        rollout, policy, config, _ = self.training_fixture(1)
        rollout.transitions["physics_error"][:] = 1
        with self.assertRaisesRegex(ValueError, "no physically valid"):
            specialist_batch_from_rollout(
                rollout=rollout, motion_policy=policy, configuration=config
            )

    def test_real_specialist_update_ignores_failed_row(self):
        rollout, policy, _, _ = self.training_fixture(1)
        config = ARDYHyperPolicyConfiguration(
            feature_count=1, action_count=1, coefficient_count=1,
            event_feature_count=1, model_width=2, transformer_blocks=1,
            attention_heads=1, feed_forward_width=2, maximum_frames=2,
            decoder_hidden_width=2, decoder_layers=1,
        )
        layer = HyperBaseLayer(
            input_count=1, output_count=1, rank=1, activation=_ACTIVATION_IDENTITY,
            weight=np.array([[0.2]], dtype=np.float32), bias=np.zeros(1, np.float32),
            adapter_down=np.array([[1]], dtype=np.float32),
            adapter_up=np.array([[1]], dtype=np.float32),
            adapter_bias_basis=np.array([[1]], dtype=np.float32),
        )
        base = HyperBasePolicy(
            id="test", revision=1, world_fingerprint=1, task_fingerprint=2,
            observation_fingerprint=3, action_fingerprint=4,
            observation_mean=np.zeros(1, np.float32),
            observation_inverse_standard_deviation=np.ones(1, np.float32),
            layers=(layer,), coefficient_limits=np.ones(1, np.float32),
            action_bias=np.zeros(1, np.float32), action_scale=np.ones(1, np.float32),
            observation_clip=10, action_clip=10,
        )
        base = replace(base, fingerprint=base.computed_fingerprint())
        def update():
            batch = specialist_batch_from_rollout(
                rollout=rollout, motion_policy=policy, configuration=config
            )
            learner = MLXSpecialistAdapterLearner(
                actor=PhaseVaryingLowRankActor(base), hyper_base=base,
                configuration=config, motion_count=1, maximum_knot_count=2,
            )
            metrics = learner.update(batch)
            return metrics, learner.export()
        first, a = update()
        rollout.transitions["reward"][1] = 1e20
        rollout.outcomes["tracking_score"][1] = -1e20
        rollout.teacher_actions[1] = 1e30
        rollout.actor_observations[1] = -1e30
        second, b = update()
        self.assertEqual(first, second)
        self.assertTrue(np.isfinite(list(first.values())).all())
        np.testing.assert_array_equal(a.coefficients, b.coefficients)
        np.testing.assert_array_equal(a.authority, b.authority)
        np.testing.assert_array_equal(a.phase_rate_multiplier, b.phase_rate_multiplier)

    def test_clean_meta_training_preserves_outcome_label(self):
        _, _, config, item = self.training_fixture(0)
        good = _pad_motion_batch([item], config)
        np.testing.assert_array_equal(good.failure_targets, [0])
        item.qualified = False
        bad_outcome = _pad_motion_batch([item], config)
        np.testing.assert_array_equal(bad_outcome.failure_targets, [1])

    def test_public_policy_batch_rejects_failed_physics(self):
        transitions = np.zeros(4, dtype=_TRANSITION_DTYPE)
        r = NativePolicyRollout(
            id="test", task_fingerprint=1, policy_fingerprint=2, policy_revision=1,
            environment_count=2, control_step_count=2,
            actor_observation_count=1, critic_observation_count=1, action_count=1,
            motion_feature_count=0,
            actor_observations=np.zeros(4, np.float32),
            critic_observations=np.zeros(4, np.float32),
            motion_features=np.empty(0, np.float32),
            latents=np.zeros(4, np.float32),
            old_log_probabilities=np.zeros(4, np.float32),
            old_values=np.zeros(4, np.float32),
            bootstrap_values=np.zeros(2, np.float32),
            transitions=transitions,
        )
        self.assertIsNotNone(r.policy_batch())
        transitions["physics_error"][1] = 1
        with self.assertRaisesRegex(ValueError, "physics failure"):
            r.policy_batch()

    def test_meta_training_rejects_numerical_failure_as_outcome(self):
        _, _, config, item = self.training_fixture(1)
        with self.assertRaisesRegex(ValueError, "physics failure"):
            _pad_motion_batch([item], config)


if __name__ == "__main__":
    unittest.main()
