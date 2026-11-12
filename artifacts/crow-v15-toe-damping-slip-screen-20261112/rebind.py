from pathlib import Path
import hashlib,json,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-v15-toe-damping-slip-screen-20261112';B=R/'build/crow-v15-toe-damping-slip-screen-20261112';cmd=json.loads((A/'preflight-command.json').read_text());report=json.loads((A/'preflight-result.json').read_text());world,task,obs,act=report['new_fingerprints']
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
reports=[]
for option,label in [('--policy-pack','landing'),('--base-policy-pack','route')]:
 source=Path(cmd[cmd.index(option)+1]);src=read_policy_pack(source,library_path=str(B/'libmetalrobo.dylib'));assert not src.critic_layers and len(src.action_log_standard_deviation)==0
 layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers];target=B/(label+'-candidate.policypack')
 write_policy_pack(target,policy_id='crow_toe_damping_diagnostic_'+label,revision=src.revision+1,contract_version=src.contract_version,world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,layers=layers,observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(B/'libmetalrobo.dylib'))
 dst=read_policy_pack(target,library_path=str(B/'libmetalrobo.dylib'));assert len(src.layers)==len(dst.layers) and all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) and x.activation==y.activation for x,y in zip(src.layers,dst.layers));assert all(np.array_equal(getattr(src,k),getattr(dst,k)) for k in ('observation_mean','observation_inverse_standard_deviation','action_bias','action_scale'));assert src.observation_clip==dst.observation_clip and src.action_clip==dst.action_clip
 reports.append({'actor':label,'source_sha256':h(source),'target_sha256':h(target),'identical_layers_and_transforms':True})
(A/'policy-binding.json').write_text(json.dumps({'old_fingerprints':report['old_fingerprints'],'new_fingerprints':report['new_fingerprints'],'actors':reports},indent=2)+'\n')
print('rebound',len(reports),'actor packs with identical layers and transforms')
