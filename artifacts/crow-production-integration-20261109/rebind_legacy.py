from pathlib import Path
import hashlib,json
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();B=R/'build/crow-production-integration-20261109';A=R/'artifacts/crow-production-integration-20261109';lib=R/'build/lib/libmetalrobo.dylib'
source=R/'build/crow-extended-flight-20261109/collision-on/extended.policypack';dest=B/'legacy-regression.policypack'
p=read_policy_pack(source,library_path=str(lib))
def layers(xs):return [PolicyDenseLayerArtifact(weights=x.weights,bias=x.bias,activation=x.activation) for x in xs]
write_policy_pack(dest,policy_id='v15_legacy_regression_rebound',revision=p.revision,contract_version=p.contract_version,world_fingerprint=p.world_fingerprint,task_fingerprint=309395803524675639,observation_fingerprint=p.observation_fingerprint,action_fingerprint=p.action_fingerprint,layers=layers(p.layers),critic_layers=layers(p.critic_layers),observation_mean=p.observation_mean,observation_inverse_standard_deviation=p.observation_inverse_standard_deviation,critic_observation_mean=p.critic_observation_mean,critic_observation_inverse_standard_deviation=p.critic_observation_inverse_standard_deviation,action_log_standard_deviation=p.action_log_standard_deviation,action_bias=p.action_bias,action_scale=p.action_scale,observation_clip=p.observation_clip,action_clip=p.action_clip,library_path=str(lib))
(A/'legacy-rebind.json').write_text(json.dumps({'source_task_fingerprint':p.task_fingerprint,'rebound_task_fingerprint':309395803524675639,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'rebound_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'purpose':'same actor and critic weights for a controlled regression, not a trained or qualified policy'},indent=2)+'\n')
