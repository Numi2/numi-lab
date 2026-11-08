"""Independent final-policy Gaussian KL and actual Adam displacement checks."""
import unittest
import math
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from metalrobo.mlx_policy_learning import NativePolicyRollout, _TRANSITION_DTYPE
from metalrobo.mlx_recurrent_policy_learning import MLXRecurrentPolicyLearner, MLXRecurrentPPOConfiguration, MLXRecurrentPolicyBatch

class RecurrentOptimizerMetricsTest(unittest.TestCase):
 def test_final_factor_kl_and_applied_update_against_numpy(self):
  c=MLXRecurrentPPOConfiguration(hidden_sizes=(4,),critic_hidden_sizes=(4,),sequence_length=5,
    minibatch_size=15,update_epochs=2,learning_rate=.01,maximum_gradient_norm=.1,
    adaptive_learning_rate=False,seed=814)
  l=MLXRecurrentPolicyLearner.initialize(2,1,2,3,c)
  rng=np.random.default_rng(815);T,N,A=5,3,2
  actor=rng.normal(size=(T,N,2)).astype('f4');critic=rng.normal(size=(T,N,1)).astype('f4')
  resets=np.zeros((T,N),bool);resets[0]=True;initial=mx.zeros((N,3))
  means,before=l.model.actor_mean(mx.array(actor),initial,mx.array(resets));old=np.array(means)
  oldstd=np.array(l.model.log_standard_deviation);latent=old+rng.normal(size=old.shape).astype('f4')*.7
  logp=np.array(l.model.log_probability(means,mx.array(oldstd),mx.array(latent)))
  values=np.array(l.model.value(mx.array(critic)));transitions=np.zeros(T*N,dtype=_TRANSITION_DTYPE)
  transitions['policy_revision']=l.revision;transitions['reward'].reshape(T,N)[-1]=[0,20,50]
  mask=np.full((T,N),2,'f4');mask[1]=3
  r=NativePolicyRollout(id='independent-final-KL',task_fingerprint=1,policy_fingerprint=2,
    policy_revision=l.revision,environment_count=N,control_step_count=T,actor_observation_count=2,
    critic_observation_count=1,action_count=A,motion_feature_count=0,format_version=9,
    actor_observations=actor.reshape(-1),critic_observations=critic.reshape(-1),motion_features=np.empty(0,'f4'),
    latents=latent.reshape(-1),old_log_probabilities=logp.reshape(-1),old_values=values.reshape(-1),
    bootstrap_values=np.zeros(N,'f4'),transitions=transitions,memory_hidden_count=3,
    memory_before=np.array(before).reshape(-1),memory_resets=resets.astype('u4').reshape(-1),
    outcomes={'policy_action_selection_mask':mask.reshape(-1)},outcome_units={'policy_action_selection_mask':'action-mask'},
    outcome_directions={'policy_action_selection_mask':0})
  params=lambda:{k:np.array(v,dtype='f8') for k,v in tree_flatten(l.model.trainable_parameters()) if k.startswith('actor.') or k=='log_standard_deviation'}
  oldparams=params();m=l.update(MLXRecurrentPolicyBatch.from_rollout(r,discount=c.discount,gae_lambda=c.gae_lambda))
  final,_=l.model.actor_mean(mx.array(actor),initial,mx.array(resets));new=np.array(final,dtype='f8');std=np.array(l.model.log_standard_deviation,dtype='f8')
  ref=std-oldstd+.5*(np.exp(2*(oldstd-std))+(old.astype('f8')-new)**2*np.exp(-2*std)-1)
  selections=r.action_selections().reshape(T,N,A).astype(bool)
  for action,count in ((0,3),(1,15)):
   self.assertEqual(m[f'post_update_action_{action}_decision_count'],count)
   self.assertAlmostEqual(m[f'post_update_action_{action}_kl'],float(ref[:,:,action][selections[:,:,action]].mean()),places=6)
  self.assertAlmostEqual(m['post_update_kl_divergence'],float(np.where(selections,ref,0).sum()/15),places=6)
  self.assertGreater(m['post_update_kl_divergence'],m['kl_divergence'])
  after=params();delta=math.sqrt(sum(np.square(after[k]-v).sum() for k,v in oldparams.items()))
  self.assertAlmostEqual(m['actor_total_parameter_delta_norm'],delta,places=6)
  self.assertLess(m['minimum_gradient_clip_scale'],1)
  self.assertLess(m['clipped_actor_gradient_norm'],m['actor_gradient_norm'])
  self.assertGreater(m['actor_parameter_delta_norm'],0)

if __name__=='__main__':unittest.main()
