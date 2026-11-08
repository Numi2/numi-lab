"""Actual update32 replay: archived generating data, model/Adam, no physics."""
from pathlib import Path
import json,sys
import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
from metalrobo.mlx_policy_worker import _restore_learner_state,_configuration_record
from metalrobo.mlx_policy_learning import read_policy_rollout_pack
from metalrobo.mlx_recurrent_policy_learning import MLXRecurrentPolicyLearner,MLXRecurrentPolicyBatch,MLXRecurrentPPOConfiguration

root=Path(sys.argv[1]);results={}
for arm in ('fixed','adaptive'):
    before=root/arm/'checkpoints/revision-00000096.trainingcheckpoint'
    after=root/arm
    _,metadata=mx.load(before/'learner.safetensors',return_metadata=True)
    configuration=json.loads(metadata['configuration'])
    for k in ('hidden_sizes','critic_hidden_sizes'):configuration[k]=tuple(configuration[k])
    c=MLXRecurrentPPOConfiguration(**configuration)
    learner=MLXRecurrentPolicyLearner.from_policy_pack(before/'policy.policypack',c)
    assert _restore_learner_state(learner,before/'learner.safetensors')
    r=read_policy_rollout_pack(after/'training.rolloutpack')
    steps=[];original=learner._compiled_training_step
    def traced(*args):
        loss,metrics=original(*args);mx.eval(loss,metrics,learner.model.state,learner.optimizer.state)
        steps.append({k:float(v.item())for k,v in metrics.items()});return loss,metrics
    learner._compiled_training_step=traced
    metrics=learner.update(MLXRecurrentPolicyBatch.from_rollout(r,discount=c.discount,gae_lambda=c.gae_lambda))
    expected,meta=mx.load(after/'learner.safetensors',return_metadata=True)
    actual={**{'model.'+k:v for k,v in tree_flatten(learner.model.trainable_parameters())},
            **{'optimizer.'+k:v for k,v in tree_flatten(learner.optimizer.state)}}
    assert actual.keys()==expected.keys()
    for key in actual:assert np.array(actual[key]).tobytes()==np.array(expected[key]).tobytes(),(arm,key)
    assert _configuration_record(learner)==meta['configuration']
    assert str(learner.revision)==meta['policy_revision']=='97'
    metrics.pop('update_seconds')
    results[arm]={'status':'all_model_and_Adam_arrays_byte_exact','arrays':len(actual),
        'configuration_bytes_exact':True,'steps':steps,'final_metrics':metrics}
with Path(sys.argv[2]).open('x')as f:json.dump(results,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({k:{'arrays':v['arrays'],'steps':[{m:s[m]for m in (
    'actor_gradient_norm','critic_gradient_norm','gradient_clip_scale','clipped_actor_gradient_norm','actor_parameter_delta_norm')}
    for s in v['steps']]}for k,v in results.items()}))
