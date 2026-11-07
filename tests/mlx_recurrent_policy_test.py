"""Return boundaries and sequence coverage independent of simulator fixtures."""
from dataclasses import replace
import unittest
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from metalrobo.mlx_policy_learning import NativePolicyRollout, _TRANSITION_DTYPE
from metalrobo.mlx_recurrent_policy_learning import MLXRecurrentPolicyBatch, MLXRecurrentPPOConfiguration, MLXRecurrentPolicyLearner


def example():
    transitions=np.zeros(3,dtype=_TRANSITION_DTYPE)
    transitions["reward"]=[.5,.6,.7]
    transitions["done"]=[0,1,0]
    transitions["timeout"]=[0,1,0]
    transitions["timeout_bootstrap_value"]=[0,5,0]
    transitions["policy_revision"]=1
    return NativePolicyRollout(id="timeout-reset-example",task_fingerprint=1,policy_fingerprint=2,
        policy_revision=1,environment_count=1,control_step_count=3,actor_observation_count=2,
        critic_observation_count=1,action_count=1,motion_feature_count=0,
        actor_observations=np.arange(6,dtype=np.float32),critic_observations=np.arange(3,dtype=np.float32),
        motion_features=np.empty(0,np.float32),latents=np.zeros(3,np.float32),old_log_probabilities=np.zeros(3,np.float32),
        old_values=np.array([1,2,3],np.float32),bootstrap_values=np.array([4],np.float32),transitions=transitions,
        memory_hidden_count=2,memory_before=np.zeros(6,np.float32),memory_resets=np.array([1,0,1],np.uint32))


class RecurrentReturnBoundaryTest(unittest.TestCase):
    def test_timeout_bootstrap_and_reset_cut(self):
        r=example();batch=MLXRecurrentPolicyBatch.from_rollout(r,discount=.9,gae_lambda=.8)
        np.testing.assert_allclose(batch.advantages[:,0],[3.532,3.1,1.3],atol=1e-6,rtol=0)
        np.testing.assert_allclose(batch.returns[:,0],[4.532,5.1,4.3],atol=1e-6,rtol=0)
        changed=r.transitions.copy();changed["reward"][2]=700
        other=MLXRecurrentPolicyBatch.from_rollout(replace(r,transitions=changed),discount=.9,gae_lambda=.8)
        np.testing.assert_array_equal(batch.advantages[:2],other.advantages[:2])

    def test_reset_requires_a_terminal_edge(self):
        r=example();changed=r.transitions.copy();changed["done"][1]=0;changed["timeout"][1]=0
        with self.assertRaisesRegex(ValueError,"reset/terminal"):
            MLXRecurrentPolicyBatch.from_rollout(replace(r,transitions=changed))

    def test_terminal_edge_requires_a_reset(self):
        with self.assertRaisesRegex(ValueError,"reset/terminal"):
            MLXRecurrentPolicyBatch.from_rollout(replace(example(),memory_resets=np.array([1,0,0],np.uint32)))

    def test_timeout_cannot_be_nonterminal(self):
        r=example();changed=r.transitions.copy();changed["done"][1]=0
        with self.assertRaisesRegex(ValueError,"nonterminal timeout"):
            MLXRecurrentPolicyBatch.from_rollout(replace(r,transitions=changed))

    def test_window_tail_is_lossless_and_zero_padded(self):
        batch=MLXRecurrentPolicyBatch.from_rollout(example())
        windows=batch.windows(2)
        np.testing.assert_array_equal(windows,[[0,0,2],[2,0,1]])
        tables,indices,valid=batch.gather(windows,2)
        np.testing.assert_array_equal(np.sort(indices[valid]),[0,1,2])
        np.testing.assert_array_equal(tables["valid"],[[1,1],[1,0]])
        np.testing.assert_array_equal(tables["actor_observations"][:,0],[[0,1],[2,3]])
        np.testing.assert_array_equal(tables["actor_observations"][1,1],[0,0])
        self.assertEqual(tables["advantages"][1,1],0)
        self.assertTrue(tables["resets"][1,1])

    def test_dense_and_flat_paths_are_not_interchangeable(self):
        r=example()
        with self.assertRaisesRegex(ValueError,"contiguous sequence"):
            r.policy_batch()
        with self.assertRaisesRegex(ValueError,"recurrent rollout"):
            MLXRecurrentPolicyBatch.from_rollout(replace(r,memory_hidden_count=0))

    def test_minibatch_cannot_split_a_sequence(self):
        with self.assertRaisesRegex(ValueError,"sequence_length"):
            MLXRecurrentPPOConfiguration(sequence_length=8,minibatch_size=7).validate()


class RecurrentInitializationTest(unittest.TestCase):
    def config(self, seed=19):
        return MLXRecurrentPPOConfiguration(hidden_sizes=(11, 5), critic_hidden_sizes=(9,),
            sequence_length=4, minibatch_size=8, seed=seed)

    def test_seeded_initialization_and_separate_architectures(self):
        a=MLXRecurrentPolicyLearner.initialize(13,7,3,8,self.config())
        b=MLXRecurrentPolicyLearner.initialize(13,7,3,8,self.config())
        c=MLXRecurrentPolicyLearner.initialize(13,7,3,8,self.config(20))
        arrays=lambda x: {k:np.array(v) for k,v in tree_flatten(x.model.parameters())}
        first,second,other=arrays(a),arrays(b),arrays(c)
        self.assertEqual(first.keys(),second.keys())
        for key in first: np.testing.assert_array_equal(first[key],second[key])
        self.assertFalse(np.array_equal(first["actor.memory.recurrent_weights"],other["actor.memory.recurrent_weights"]))
        self.assertEqual(a.model.actor.memory["input_weights"].shape,(24,5))
        self.assertEqual(a.model.actor.memory["recurrent_weights"].shape,(24,8))
        self.assertEqual([(x.weight.shape,x.activation) for x in a.model.actor.encoder],
            [((11,13),2),((5,11),2)])
        self.assertEqual([x.weight.shape for x in a.model.critic],[(9,7),(1,9)])

    def test_finite_bounded_encoder_and_reset_cut_at_extreme_inputs(self):
        a=MLXRecurrentPolicyLearner.initialize(13,7,3,8,self.config())
        x=mx.full((3,2,13),1e20);initial=mx.full((2,8),.75)
        resets=mx.array([[True,False],[False,False],[True,True]])
        means,before=a.model.actor_mean(x,initial,resets)
        self.assertTrue(np.isfinite(np.array(means)).all())
        self.assertTrue(np.all(np.abs(np.array(before))<=1))
        np.testing.assert_array_equal(np.array(before)[0,0],np.zeros(8))
        np.testing.assert_array_equal(np.array(before)[2],np.zeros((2,8)))
        np.testing.assert_array_equal(np.array(before)[0,1],np.full(8,.75))

    def test_zero_output_preserves_encoder_and_memory(self):
        a=MLXRecurrentPolicyLearner.initialize(13,7,3,8,self.config())
        memory=np.array(a.model.actor.memory["input_weights"])
        encoder=np.array(a.model.actor.encoder[0].weight)
        a.zero_actor_output()
        means,_=a.model.actor_mean(mx.ones((3,2,13)),mx.zeros((2,8)),mx.zeros((3,2),mx.bool_))
        np.testing.assert_array_equal(np.array(means),np.zeros((3,2,3)))
        np.testing.assert_array_equal(np.array(a.model.actor.memory["input_weights"]),memory)
        np.testing.assert_array_equal(np.array(a.model.actor.encoder[0].weight),encoder)

    def test_invalid_capacity_rejects_before_construction(self):
        for dimensions in ((0,7,3,8),(13,0,3,8),(13,7,0,8),(13,7,3,0)):
            with self.assertRaisesRegex(ValueError,"dimensions"):
                MLXRecurrentPolicyLearner.initialize(*dimensions,self.config())
        with self.assertRaisesRegex(ValueError,"nonempty tanh encoder"):
            MLXRecurrentPolicyLearner.initialize(13,7,3,8,replace(self.config(),hidden_sizes=()))


if __name__=="__main__":unittest.main()
