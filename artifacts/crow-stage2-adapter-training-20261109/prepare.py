from pathlib import Path
import hashlib,json,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-stage2-adapter-training-20261109';B=R/'build/crow-stage2-adapter-training-20261109';lib=R/'build/crow-hierarchical-goal-memory-20261109/libmetalrobo.dylib'
actor=read_policy_pack(R/'build/crow-route-interpolation-20261109/route-75.policypack',library_path=str(lib))
old=read_policy_pack(R/'build/crow-gate-recovery-training-20261109/recovery-imitation.policypack',library_path=str(lib))
for name in ['contract_version','world_fingerprint','task_fingerprint','observation_fingerprint','action_fingerprint']:
 assert getattr(actor,name)==getattr(old,name),name
for name in ['observation_mean','observation_inverse_standard_deviation','action_bias','action_scale']:
 assert np.array_equal(getattr(actor,name),getattr(old,name)),name
assert np.all(actor.observation_mean[141:163]==0) and np.all(actor.observation_inverse_standard_deviation[141:163]==1)
def layers(p):return [PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in p]
out=B/'trainable-initial.policypack'
write_policy_pack(out,policy_id='crow_stage2_initial',revision=max(actor.revision,old.revision)+1,contract_version=actor.contract_version,world_fingerprint=actor.world_fingerprint,task_fingerprint=actor.task_fingerprint,observation_fingerprint=actor.observation_fingerprint,action_fingerprint=actor.action_fingerprint,layers=layers(actor.layers),critic_layers=layers(old.critic_layers),observation_mean=actor.observation_mean,observation_inverse_standard_deviation=actor.observation_inverse_standard_deviation,critic_observation_mean=old.critic_observation_mean,critic_observation_inverse_standard_deviation=old.critic_observation_inverse_standard_deviation,action_log_standard_deviation=old.action_log_standard_deviation,action_bias=actor.action_bias,action_scale=actor.action_scale,observation_clip=actor.observation_clip,action_clip=actor.action_clip,library_path=str(lib))
p=read_policy_pack(out,library_path=str(lib));assert all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) for x,y in zip(p.layers,actor.layers))
(A/'initial.json').write_text(json.dumps({'trainable_initial_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'actor_source_sha256':hashlib.sha256((R/'build/crow-route-interpolation-20261109/route-75.policypack').read_bytes()).hexdigest(),'actor_layers_identical':True,'critic_used_only_to_resume_supervised_training':True},indent=2)+'\n')
