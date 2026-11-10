from pathlib import Path
import hashlib,json,re,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd(); A=R/'artifacts/crow-v15-toe-damping-probe-20261110'; B=R/'build/crow-v15-toe-damping-probe-20261110'; L=B/'libmetalrobo.dylib'
base=(R/'build/crow-v15-long-perch-host-20261110/c_api.cpp').read_text(); altered=(B/'c_api.cpp').read_text(); old='d.drive={.1f,.003f,.00005f,0}'; new='d.drive={.1f,.0045f,.00005f,0}'; assert base.count(old)==1 and altered==base.replace(old,new)
m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',(A/'compatibility.stderr').read_text()); assert m
world,task,obs,act=map(int,m.groups()); h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); results={}
for name,srcpath in [('landing',R/'build/crow-v15-long-perch-host-20261110/landing.policypack'),('route',R/'build/crow-v15-stage-aware-learning-20261110/corrective-actor.policypack')]:
 src=read_policy_pack(srcpath,library_path=str(L)); assert not src.critic_layers and len(src.action_log_standard_deviation)==0
 layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers]; out=B/(name+'.policypack')
 write_policy_pack(out,policy_id='crow_toe_damping_probe_'+name,revision=src.revision+1,contract_version=src.contract_version,world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,layers=layers,observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(L))
 dst=read_policy_pack(out,library_path=str(L)); assert len(dst.layers)==len(src.layers)
 assert all(x.activation==y.activation and np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) for x,y in zip(src.layers,dst.layers))
 for k in ('observation_mean','observation_inverse_standard_deviation','action_bias','action_scale'):assert np.array_equal(getattr(src,k),getattr(dst,k)),k
 for k in ('contract_version','observation_clip','action_clip'):assert getattr(src,k)==getattr(dst,k),k
 assert (dst.world_fingerprint,dst.task_fingerprint,dst.observation_fingerprint,dst.action_fingerprint)==(world,task,obs,act)
 results[name]={'source_sha256':h(srcpath),'candidate_sha256':h(out),'layer_shapes':[[*x.weights.shape] for x in src.layers],'byte_identical_arrays':True}
(A/'rebind-results.json').write_text(json.dumps({'candidate_fingerprints':{'world':world,'task':task,'observation':obs,'action':act},'source_host_sha256':h(R/'build/crow-v15-long-perch-host-20261110/libmetalrobo.dylib'),'candidate_host_sha256':h(L),'source_single_change_verified':True,'policies':results,'limitations':'Diagnostic rebinding of identical tensors to changed body contract; no training, physical qualification, or biological validation. Task, observation and action source definitions are unchanged; native contract checks remain active.'},indent=2)+'\n')
print(json.dumps(results,indent=2))
