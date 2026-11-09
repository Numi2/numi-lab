from pathlib import Path
import hashlib,json,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-stage2-head-experiment-20261109';B=R/'build/crow-stage2-head-experiment-20261109';lib=R/'build/crow-hierarchical-goal-memory-20261109/libmetalrobo.dylib'
source=R/'build/crow-stage2-adapter-training-20261109/recovery-imitation.policypack';p=read_policy_pack(source,library_path=str(lib))
def cp(ls):return [PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in ls]
layers=cp(p.layers)
assert np.array_equal(p.action_scale[:4],p.action_scale[4:8]) and np.array_equal(p.action_bias[:4],p.action_bias[4:8])
layers[-1].weights[4:8,:]=layers[-1].weights[:4,:];layers[-1].bias[4:8]=layers[-1].bias[:4]
common=dict(revision=p.revision+1,contract_version=p.contract_version,world_fingerprint=p.world_fingerprint,task_fingerprint=p.task_fingerprint,observation_fingerprint=p.observation_fingerprint,action_fingerprint=p.action_fingerprint,layers=layers,observation_mean=p.observation_mean,observation_inverse_standard_deviation=p.observation_inverse_standard_deviation,action_bias=p.action_bias,action_scale=p.action_scale,observation_clip=p.observation_clip,action_clip=p.action_clip,library_path=str(lib))
full=B/'trainable-initial.policypack';actor=B/'initial-actor.policypack'
write_policy_pack(full,policy_id='crow_stage2_head_trainable',critic_layers=cp(p.critic_layers),critic_observation_mean=p.critic_observation_mean,critic_observation_inverse_standard_deviation=p.critic_observation_inverse_standard_deviation,action_log_standard_deviation=p.action_log_standard_deviation,**common)
write_policy_pack(actor,policy_id='crow_stage2_head_initial',**common)
a=read_policy_pack(actor,library_path=str(lib))
assert np.array_equal(a.layers[-1].weights[4:8],a.layers[-1].weights[:4]) and np.array_equal(a.layers[-1].bias[4:8],a.layers[-1].bias[:4])
(A/'initial.json').write_text(json.dumps({'full_sha256':hashlib.sha256(full.read_bytes()).hexdigest(),'actor_sha256':hashlib.sha256(actor.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'stage2_head_identical_to_original_commands':True},indent=2)+'\n')
