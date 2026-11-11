from pathlib import Path
import hashlib,json,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/"artifacts/crow-v15-localized-route-student-imitation-20261111";B=R/"build/crow-v15-localized-route-student-imitation-20261111"
lib=R/"build/crow-v15-hip-roll-journey-integration-20261111/libmetalrobo.dylib"
actor=read_policy_pack(R/"build/crow-v15-near-slalom-student-imitation-20261111/route-imitation.policypack",library_path=str(lib))
critic=read_policy_pack(R/"build/crow-deterministic-route-training-20261109/recovery-imitation.policypack",library_path=str(lib))
assert not actor.critic_layers and len(critic.critic_layers)>0 and len(critic.critic_observation_mean)==106 and len(critic.action_log_standard_deviation)==18
def layers(p):return [PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in p]
out=B/"trainable-initial.policypack"
write_policy_pack(out,policy_id="crow_v15_prefix_imitation_training_wrapper",revision=actor.revision+1,contract_version=actor.contract_version,world_fingerprint=actor.world_fingerprint,task_fingerprint=actor.task_fingerprint,observation_fingerprint=actor.observation_fingerprint,action_fingerprint=actor.action_fingerprint,layers=layers(actor.layers),critic_layers=layers(critic.critic_layers),observation_mean=actor.observation_mean,observation_inverse_standard_deviation=actor.observation_inverse_standard_deviation,critic_observation_mean=critic.critic_observation_mean,critic_observation_inverse_standard_deviation=critic.critic_observation_inverse_standard_deviation,action_log_standard_deviation=critic.action_log_standard_deviation,action_bias=actor.action_bias,action_scale=actor.action_scale,observation_clip=actor.observation_clip,action_clip=actor.action_clip,library_path=str(lib))
p=read_policy_pack(out,library_path=str(lib))
assert len(p.layers)==len(actor.layers) and all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) and x.activation==y.activation for x,y in zip(p.layers,actor.layers))
assert all(np.array_equal(getattr(p,k),getattr(actor,k)) for k in ("observation_mean","observation_inverse_standard_deviation","action_bias","action_scale"))
assert p.observation_clip==actor.observation_clip and p.action_clip==actor.action_clip
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(A/"initial.json").write_text(json.dumps({"actor_source_sha256":h(R/"build/crow-v15-near-slalom-student-imitation-20261111/route-imitation.policypack"),"critic_source_sha256":h(R/"build/crow-deterministic-route-training-20261109/recovery-imitation.policypack"),"training_wrapper_sha256":h(out),"actor_layers_and_transforms_identical":True,"scope":"The borrowed critic has a different historical task contract and is only an unused wrapper requirement for supervised actor imitation. No critic value or gradient is used; it will be stripped from the output policy."},indent=2)+"\n")
