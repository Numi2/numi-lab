from pathlib import Path
import gzip,hashlib,json,shutil
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-stage-selected-policy-20261109';B=R/'build/crow-stage-selected-policy-20261109';lib=R/'build/crow-hierarchical-goal-memory-20261109/libmetalrobo.dylib'
old=R/'build/crow-goal-memory-dagger-training-20261109/recovery-imitation.policypack';new=B/'frozen-route-actor.policypack'
p=read_policy_pack(old,library_path=str(lib))
layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in p.layers]
write_policy_pack(new,policy_id='crow_v15_goal_memory_frozen_route_actor',revision=p.revision,contract_version=p.contract_version,world_fingerprint=p.world_fingerprint,task_fingerprint=p.task_fingerprint,observation_fingerprint=p.observation_fingerprint,action_fingerprint=p.action_fingerprint,layers=layers,observation_mean=p.observation_mean,observation_inverse_standard_deviation=p.observation_inverse_standard_deviation,action_bias=p.action_bias,action_scale=p.action_scale,observation_clip=p.observation_clip,action_clip=p.action_clip,library_path=str(lib))
n=read_policy_pack(new,library_path=str(lib))
assert not n.critic_layers and len(n.action_log_standard_deviation)==0
assert all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) for x,y in zip(p.layers,n.layers))
for name in ['observation_mean','observation_inverse_standard_deviation','action_bias','action_scale']:
 assert np.array_equal(getattr(p,name),getattr(n,name)),name
with new.open('rb') as f,(A/'frozen-route-actor.policypack.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:shutil.copyfileobj(f,z)
(A/'base-policy.json').write_text(json.dumps({'source_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'base_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'actor_weights_identical':True,'observation_and_action_transforms_identical':True,'critic_and_stochastic_head_removed':True},indent=2)+'\n')
