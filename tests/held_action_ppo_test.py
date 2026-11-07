"""Selected-factor PPO: independent FP64 differentiation and absent-decision controls."""
from dataclasses import replace
import unittest
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_recurrent_policy_test import example
from metalrobo.mlx_recurrent_policy_learning import MLXRecurrentPolicyLearner, MLXRecurrentPPOConfiguration
from metalrobo.recurrent_ppo_admission import fp64_loss

class HeldActionPPOTest(unittest.TestCase):
    def fixture(self):
        c=MLXRecurrentPPOConfiguration(hidden_sizes=(5,),critic_hidden_sizes=(4,),sequence_length=4,
             minibatch_size=8,entropy_coefficient=.03,seed=813)
        l=MLXRecurrentPolicyLearner.initialize(2,1,3,4,c)
        rng=np.random.default_rng(814);actor=rng.normal(size=(4,2,2)).astype('f4')
        critic=rng.normal(size=(4,2,1)).astype('f4');initial=np.full((2,4),.1,'f4')
        resets=np.array([[1,1],[0,0],[0,1],[0,0]],bool);valid=np.array([[1,1],[1,1],[1,1],[1,0]],'f4')
        means,_=l.model.actor_mean(mx.array(actor),mx.array(initial),mx.array(resets))
        old=np.array(means)+.02;std=np.array(l.model.log_standard_deviation)-.01
        latents=old+rng.normal(size=old.shape).astype('f4')*.3
        values=np.array(l.model.value(mx.array(critic)))
        mask=np.array([[[1,0,1],[0,1,0]],[[0,0,0],[0,0,0]],[[0,0,1],[1,0,0]],[[0,1,0],[0,0,0]]],'f4')
        args=[actor,critic,latents,np.full((4,2),-99,'f4'),old,std,values,
              np.array([[.7,-.4],[.2,.3],[-.5,.6],[.4,0]],'f4'),values+.2,initial,resets,valid,mask]
        return l,args
    def evaluate(self,l,args):
        (loss,metrics),g=l._loss_and_gradient(*[mx.array(x) for x in args]);mx.eval(loss,metrics,g)
        return float(loss),{k:np.array(v) for k,v in tree_flatten(g)}
    def test_selected_factor_loss_and_all_parameter_family_gradients_fp64(self):
        l,args=self.fixture();loss,grads=self.evaluate(l,args)
        p={k:np.array(v,dtype='f8') for k,v in tree_flatten(l.model.trainable_parameters())}
        args64=[np.asarray(x,dtype='f8') if np.asarray(x).dtype!=bool else x for x in args]
        ref=fp64_loss(p,l._source_pack,args64,l.configuration)
        self.assertLess(abs(loss-ref),3e-6)
        for key,g in grads.items():
            ix=np.unravel_index(np.argmax(np.abs(g)),g.shape);v=p[key][ix];eps=1e-5
            p[key][ix]=v+eps;a=fp64_loss(p,l._source_pack,args64,l.configuration)
            p[key][ix]=v-eps;b=fp64_loss(p,l._source_pack,args64,l.configuration);p[key][ix]=v
            self.assertLess(abs((a-b)/(2*eps)-float(g[ix])),3e-5,key)
    def test_ignored_proposals_and_full_logp_have_no_loss_or_gradient_authority(self):
        l,args=self.fixture();a,ga=self.evaluate(l,args);changed=list(args)
        changed[2]=np.where(args[-1],args[2],args[2]+50);changed[3]=args[3]+100
        b,gb=self.evaluate(l,changed);self.assertEqual(a,b)
        for k in ga:np.testing.assert_array_equal(ga[k],gb[k],err_msg=k)
    def test_no_decisions_has_zero_actor_gradient_and_trains_critic(self):
        l,args=self.fixture();args[-1]=np.zeros_like(args[-1]);_,g=self.evaluate(l,args)
        for k in g:
            if k.startswith('actor.') or k=='log_standard_deviation':np.testing.assert_array_equal(g[k],np.zeros_like(g[k]))
        self.assertTrue(any(np.any(x) for k,x in g.items() if k.startswith('critic.')))
    def test_all_selected_factors_equal_legacy_loss_and_gradients(self):
        l,args=self.fixture();args[-1]=np.ones_like(args[-1]);args[3]=np.array(l.model.log_probability(mx.array(args[4]),mx.array(args[5]),mx.array(args[2])))
        a,ga=self.evaluate(l,args);b,gb=self.evaluate(l,args[:-1]);self.assertAlmostEqual(a,b,places=6)
        for k in ga:np.testing.assert_allclose(ga[k],gb[k],atol=3e-6,rtol=0)
    def test_typed_v9_selection_contract_rejects_malformed_bits(self):
        r=example();key='policy_action_selection_mask'
        r=replace(r,format_version=9,outcomes={key:np.array([1,0,1],'f4')},outcome_units={key:'action-mask'},outcome_directions={key:0})
        np.testing.assert_array_equal(r.action_selections(),[[1],[0],[1]])
        bad=[replace(r,format_version=8),replace(r,outcomes={}),replace(r,memory_hidden_count=0),
             replace(r,outcome_units={key:'rate'}),replace(r,outcome_directions={key:1}),
             replace(r,action_count=25),replace(r,teacher_actions=np.ones(3))]
        for x in (-1,.5,2,np.nan,np.inf):bad.append(replace(r,outcomes={key:np.array([1,x,1],'f4')}))
        for item in bad:
            with self.assertRaises(ValueError):item.action_selections()

if __name__=='__main__':unittest.main()
