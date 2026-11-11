from pathlib import Path
import hashlib,json,re,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-v15-knee-compliance-screen-20261111';B=R/'build/crow-v15-knee-compliance-screen-20261111';cmd=json.loads((A/'preflight-command.json').read_text());err=(A/'preflight.stderr').read_text();m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',err)
if not m:raise RuntimeError('Expected changed contract fingerprint absent: '+err[-1200:])
world,task,obs,act=map(int,m.groups());reports=[];h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for option,label in [('--policy-pack','landing'),('--base-policy-pack','route')]:
 source=Path(cmd[cmd.index(option)+1]);src=read_policy_pack(source,library_path=str(B/'libmetalrobo.dylib'));assert not src.critic_layers and len(src.action_log_standard_deviation)==0
 layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers];target=B/(label+'.policypack')
 write_policy_pack(target,policy_id='crow_knee_compliance_diagnostic_'+label,revision=src.revision+1,contract_version=src.contract_version,world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,layers=layers,observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(B/'libmetalrobo.dylib'))
 dst=read_policy_pack(target,library_path=str(B/'libmetalrobo.dylib'));assert len(src.layers)==len(dst.layers) and all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) and x.activation==y.activation for x,y in zip(src.layers,dst.layers));assert all(np.array_equal(getattr(src,k),getattr(dst,k)) for k in ('observation_mean','observation_inverse_standard_deviation','action_bias','action_scale'));assert src.observation_clip==dst.observation_clip and src.action_clip==dst.action_clip
 cmd[cmd.index(option)+1]=str(target);reports.append({'actor':label,'source_sha256':h(source),'target_sha256':h(target),'identical_layers_and_transforms':True})
(A/'policy-binding.json').write_text(json.dumps({'fingerprints':[world,task,obs,act],'reports':reports,'scope':'Diagnostic rebinding only; matching fingerprints or weights do not prove observation semantic equivalence.'},indent=2)+'\n');(A/'rebound-command.json').write_text(json.dumps(cmd,indent=2)+'\n');print('bound diagnostic policies')
