"""Initialization-only GRU retention: exact legacy path and analytic unforced decay."""
import unittest
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from metalrobo.gru_memory import gru_sequence
from metalrobo.mlx_recurrent_policy_learning import MLXRecurrentPolicyLearner,MLXRecurrentPPOConfiguration

class InitialMemoryTimescaleTest(unittest.TestCase):
    def initialize(self,**kw):
        c=MLXRecurrentPPOConfiguration(hidden_sizes=(7,5),critic_hidden_sizes=(6,),seed=571)
        return MLXRecurrentPolicyLearner.initialize(3,2,2,8,c,**kw)
    def arrays(self,l):return {k:np.array(v) for k,v in tree_flatten(l.model.parameters())}
    def test_explicit_legacy_is_byte_exact_and_all_four_biases_zero(self):
        a,b=self.arrays(self.initialize()),self.arrays(self.initialize(maximum_memory_timescale_ticks=0))
        self.assertEqual(a.keys(),b.keys())
        for k in a:np.testing.assert_array_equal(a[k],b[k],err_msg=k)
        np.testing.assert_array_equal(a['actor.memory.input_bias'],np.zeros(24,'f4'))
        np.testing.assert_array_equal(a['actor.memory.recurrent_bias'],np.zeros(24,'f4'))
    def test_only_update_gate_input_bias_changes(self):
        a,b=self.arrays(self.initialize()),self.arrays(self.initialize(maximum_memory_timescale_ticks=256))
        for k in a:
            if k!='actor.memory.input_bias':np.testing.assert_array_equal(a[k],b[k],err_msg=k)
        tau=np.geomspace(2.,256.,8);expected=np.zeros(24,'f4');expected[8:16]=-np.log(np.expm1(1/tau))
        np.testing.assert_array_equal(b['actor.memory.input_bias'],expected)
    def test_isolated_cell_has_analytic_retention_and_trainable_gate_gradient(self):
        l=self.initialize(maximum_memory_timescale_ticks=256)
        p={k:mx.zeros_like(v) for k,v in l.model.actor.memory.items()};p['input_bias']=l.model.actor.memory['input_bias']
        x=mx.zeros((32,1,5));h=mx.full((1,8),.5);reset=mx.zeros((32,1),mx.bool_)
        y,_=gru_sequence(x,h,reset,p);actual=np.array(y)[-1,0];expected=.5*np.exp(-32/np.geomspace(2.,256.,8))
        np.testing.assert_allclose(actual,expected,rtol=0,atol=3e-6)
        g=mx.grad(lambda b:mx.sum(gru_sequence(x,h,reset,{**p,'input_bias':b})[0][-1]))(p['input_bias'])
        self.assertTrue(np.isfinite(np.array(g)).all());self.assertTrue(np.all(np.array(g)[8:16]>0))
    def test_invalid_initializer_rejects_before_parameter_construction(self):
        for value in (-1,1,4097,2.5,True,None):
            with self.assertRaisesRegex(ValueError,'timescale'):self.initialize(maximum_memory_timescale_ticks=value)
        self.initialize(maximum_memory_timescale_ticks=2);self.initialize(maximum_memory_timescale_ticks=4096)

if __name__=='__main__':unittest.main()
