"""Whole-update rate scheduling, independently measured KL and restart contracts."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import tempfile
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from metalrobo.mlx_policy_learning import NativePolicyRollout, _TRANSITION_DTYPE
from metalrobo.mlx_policy_worker import (
    _configuration_record, _restore_learner_state, _write_learner_state,
    _PPO_RESUMABLE_SCHEDULE_FIELDS, _PPO_EXPLICIT_LEARNING_RATE_OVERRIDE_FIELDS,
)
from metalrobo.mlx_recurrent_policy_learning import (
    MLXRecurrentPolicyLearner, MLXRecurrentPolicyBatch, MLXRecurrentPPOConfiguration,
)


def configuration(**changes):
    return replace(MLXRecurrentPPOConfiguration(
        hidden_sizes=(4,), critic_hidden_sizes=(4,), sequence_length=4,
        minibatch_size=24, update_epochs=2, learning_rate=.01,
        adaptive_kl_mode="post-step-decision-groups", seed=1814), **changes)


def rollout(learner, *, selected=True, empty=False):
    rng = np.random.default_rng(1815)
    T, N, A = 7, 3, 2  # Tail windows are padded; a rare head precedes four beak choices.
    actor = rng.normal(size=(T, N, 2)).astype('f4')
    critic = rng.normal(size=(T, N, 1)).astype('f4')
    resets = np.zeros((T, N), bool); resets[0] = True
    means, before = learner.model.actor_mean(mx.array(actor), mx.zeros((N, 3)), mx.array(resets))
    old = np.array(means)
    latent = old + rng.normal(size=old.shape).astype('f4') * .7
    logp = np.array(learner.model.log_probability(means, learner.model.log_standard_deviation, mx.array(latent)))
    values = np.array(learner.model.value(mx.array(critic)))
    transitions = np.zeros(T*N, dtype=_TRANSITION_DTYPE)
    transitions['policy_revision'] = learner.revision
    transitions['reward'].reshape(T, N)[-1] = [0, 20, 50]
    mask = np.zeros((T, N), 'f4')
    if not empty: mask[1] = 3; mask[3:] = 2
    key = 'policy_action_selection_mask'
    return NativePolicyRollout(
        id='final-group-KL-padding', task_fingerprint=1, policy_fingerprint=2,
        policy_revision=learner.revision, environment_count=N, control_step_count=T,
        actor_observation_count=2, critic_observation_count=1, action_count=A,
        motion_feature_count=0, format_version=9 if selected else 8,
        actor_observations=actor.reshape(-1), critic_observations=critic.reshape(-1),
        motion_features=np.empty(0, 'f4'), latents=latent.reshape(-1),
        old_log_probabilities=logp.reshape(-1), old_values=values.reshape(-1),
        bootstrap_values=np.zeros(N, 'f4'), transitions=transitions, memory_hidden_count=3,
        memory_before=np.array(before).reshape(-1), memory_resets=resets.astype('u4').reshape(-1),
        outcomes={key:mask.reshape(-1)} if selected else {},
        outcome_units={key:'action-mask'} if selected else {},
        outcome_directions={key:0} if selected else {}), old


def update(learner, r):
    return learner.update(MLXRecurrentPolicyBatch.from_rollout(
        r, discount=learner.configuration.discount, gae_lambda=learner.configuration.gae_lambda))


def state(learner):
    return {prefix+k:np.array(v) for prefix, tree in (
        ('model.', learner.model.trainable_parameters()), ('optimizer.', learner.optimizer.state))
        for k,v in tree_flatten(tree)}


class RecurrentAdaptiveKLTest(unittest.TestCase):
    def test_final_conditional_gaussian_kl_independent_of_frequency_and_padding(self):
        for selected in (True, False):
            learner = MLXRecurrentPolicyLearner.initialize(2, 1, 2, 3, configuration())
            r, old = rollout(learner, selected=selected)
            oldstd = np.array(learner.model.log_standard_deviation, dtype='f8')
            metrics = update(learner, r)
            # Reconstruct each truncated recurrent context without batch.gather or its masks.
            final = np.empty_like(old, dtype='f8')
            for start in (0, 4):
                end = min(start+4, 7)
                means,_ = learner.model.actor_mean(
                    mx.array(r.actor_observations.reshape(7,3,2)[start:end]),
                    mx.array(r.memory_before.reshape(7,3,3)[start]),
                    mx.array(r.memory_resets.reshape(7,3)[start:end].astype(bool)))
                final[start:end] = np.array(means, dtype='f8')
            newstd = np.array(learner.model.log_standard_deviation, dtype='f8')
            kl = newstd-oldstd + .5*(np.exp(2*(oldstd-newstd)) +
                (old.astype('f8')-final)**2*np.exp(-2*newstd)-1)
            if selected:
                groups = {3:(float(kl[1].sum(axis=-1).mean()), 3),
                          2:(float(kl[3:,:,1].mean()), 12)}
            else:
                groups = {3:(float(kl.sum(axis=-1).mean()), 21)}
            for mask,(expected,count) in groups.items():
                self.assertEqual(metrics[f'post_update_decision_group_{mask}_count'], count)
                self.assertAlmostEqual(metrics[f'post_update_decision_group_{mask}_kl'], expected, places=6)
            self.assertAlmostEqual(metrics['post_update_max_decision_group_kl'],
                max(x[0] for x in groups.values()), places=6)
            self.assertAlmostEqual(metrics['post_update_kl_divergence'],
                sum(x[0]*x[1] for x in groups.values())/sum(x[1] for x in groups.values()), places=6)

    def test_rate_only_changes_after_all_optimizer_steps_and_respects_bounds(self):
        for target,changes,expected in ((100., {'maximum_learning_rate':.013}, .013),
                                        (1e-9, {'minimum_learning_rate':.009}, .009)):
            c = configuration(target_kl=target, **changes)
            adaptive = MLXRecurrentPolicyLearner.initialize(2,1,2,3,c)
            fixed = MLXRecurrentPolicyLearner.initialize(2,1,2,3,replace(c,adaptive_learning_rate=False))
            r,_ = rollout(adaptive)
            metrics = update(adaptive,r); update(fixed,r)
            # A per-minibatch controller would fail this exact model/Adam comparison.
            for k,v in state(fixed).items():
                if k != 'optimizer.learning_rate': np.testing.assert_array_equal(state(adaptive)[k],v,err_msg=k)
            self.assertAlmostEqual(metrics['learning_rate_used'],.01,places=8)
            self.assertAlmostEqual(metrics['learning_rate_next_update'],expected,places=8)
            self.assertAlmostEqual(float(adaptive.optimizer.learning_rate.item()),expected,places=8)
            self.assertEqual(metrics['learning_rate_adaptation_kl'],metrics['post_update_max_decision_group_kl'])

    def test_disabled_target_and_absent_decisions_do_not_change_rate(self):
        for changes,empty in (({'adaptive_learning_rate':False},False), ({'target_kl':None},False), ({},True)):
            learner = MLXRecurrentPolicyLearner.initialize(2,1,2,3,configuration(**changes))
            r,_ = rollout(learner,empty=empty); metrics = update(learner,r)
            self.assertAlmostEqual(metrics['learning_rate_next_update'],.01,places=8)
            if empty:
                self.assertEqual(metrics['post_update_max_decision_group_kl'],0)
                self.assertFalse(any(k.startswith('post_update_decision_group_') for k in metrics))

    def test_serialized_mode_is_strict_and_legacy_record_is_unchanged(self):
        c = configuration()
        legacy = MLXRecurrentPolicyLearner.initialize(2,1,2,3,replace(c,adaptive_kl_mode='legacy-pre-step'))
        historical = asdict(legacy.configuration); historical.pop('adaptive_kl_mode')
        self.assertEqual(_configuration_record(legacy),json.dumps(historical,sort_keys=True,separators=(',',':'),allow_nan=False))
        with self.assertRaisesRegex(ValueError,'KL mode'): configuration(adaptive_kl_mode='bad').validate()
        learner = MLXRecurrentPolicyLearner.initialize(2,1,2,3,c)
        for field in ('contract_version','world_fingerprint','task_fingerprint',
                      'observation_fingerprint','action_fingerprint'):
            setattr(learner,field,1)
        r,_ = rollout(learner); update(learner,r)
        with tempfile.TemporaryDirectory() as d:
            pack = learner.write_policy_pack(Path(d)/'policy.policypack',policy_id='adaptive-kl-test')
            saved = _write_learner_state(learner,Path(d)/'learner.safetensors')
            _, metadata = mx.load(saved,return_metadata=True)
            self.assertEqual(json.loads(metadata['configuration'])['adaptive_kl_mode'],c.adaptive_kl_mode)
            for fields in (_PPO_RESUMABLE_SCHEDULE_FIELDS,
                           _PPO_RESUMABLE_SCHEDULE_FIELDS|_PPO_EXPLICIT_LEARNING_RATE_OVERRIDE_FIELDS):
                wrong = MLXRecurrentPolicyLearner.from_policy_pack(pack,legacy.configuration)
                with self.assertRaisesRegex(ValueError,'configuration'):
                    _restore_learner_state(wrong,saved,ppo_resumable_schedule_fields=fields)
            restored = MLXRecurrentPolicyLearner.from_policy_pack(pack,c)
            self.assertTrue(_restore_learner_state(restored,saved))
            nextroll,_ = rollout(learner)
            a = update(learner,nextroll); b = update(restored,nextroll)
            for k,v in state(learner).items(): np.testing.assert_array_equal(state(restored)[k],v,err_msg=k)
            for k in ('learning_rate_used','learning_rate_next_update','post_update_max_decision_group_kl'):
                self.assertEqual(a[k],b[k])


if __name__=='__main__': unittest.main()
