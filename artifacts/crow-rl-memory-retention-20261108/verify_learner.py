from pathlib import Path
import hashlib,json,sys
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
root=Path(sys.argv[1]);m=json.loads(Path(sys.argv[2]).read_text());sys.path.insert(0,str(root/'python'))
from metalrobo.mlx_recurrent_policy_learning import MLXRecurrentPolicyLearner,MLXRecurrentPPOConfiguration
from metalrobo.mlx_policy_worker import _restore_learner_state
config=MLXRecurrentPPOConfiguration(**json.loads(m['learner_reference']['metadata']['configuration']))
l=MLXRecurrentPolicyLearner.from_policy_pack(root/'input/checkpoint/policy.policypack',config,library_path=root/'lib/libmetalrobo.dylib')
if not _restore_learner_state(l,root/'input/checkpoint/learner.safetensors'):raise RuntimeError('Learner restore rejected')
actual={prefix+k:np.array(v) for prefix,tree in [('model.',l.model.state),('optimizer.',l.optimizer.state)] for k,v in tree_flatten(tree)}
expected=m['learner_reference']['tensors']
if actual.keys()!=expected.keys():raise RuntimeError('Restored learner topology differs')
for k,v in actual.items():
 e=expected[k]
 if str(v.dtype)!=e['dtype'] or list(v.shape)!=e['shape'] or hashlib.sha256(v.tobytes()).hexdigest()!=e['sha256']:raise RuntimeError('Restored learner differs: '+k)
print(json.dumps({'status':'passed','actual_model_and_Adam_tensors_exact':len(actual),'policy_revision':l.revision,'physical_steps_advanced':0}))
