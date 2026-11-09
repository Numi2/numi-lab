from pathlib import Path
import gzip,hashlib,json,shutil
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-route-interpolation-20261109';B=R/'build/crow-route-interpolation-20261109';lib=R/'build/crow-hierarchical-goal-memory-20261109/libmetalrobo.dylib'
old=R/'build/crow-gate-recovery-training-20261109/frozen-route-actor.policypack';new=R/'build/crow-gate-second-route-training-20261109/frozen-route-actor.policypack'
a=read_policy_pack(old,library_path=str(lib));b=read_policy_pack(new,library_path=str(lib))
for name in ['contract_version','world_fingerprint','task_fingerprint','observation_fingerprint','action_fingerprint','observation_clip','action_clip']:
 assert getattr(a,name)==getattr(b,name),name
for name in ['observation_mean','observation_inverse_standard_deviation','action_bias','action_scale']:
 assert np.array_equal(getattr(a,name),getattr(b,name)),name
assert len(a.layers)==len(b.layers)
results=[]
for x in [25,50,75]:
 alpha=x/100
 layers=[]
 for u,v in zip(a.layers,b.layers):
  assert u.activation==v.activation and u.weights.shape==v.weights.shape and u.bias.shape==v.bias.shape
  layers.append(PolicyDenseLayerArtifact(weights=((1-alpha)*u.weights+alpha*v.weights).astype(np.float32),bias=((1-alpha)*u.bias+alpha*v.bias).astype(np.float32),activation=u.activation))
 path=B/f'route-{x}.policypack'
 write_policy_pack(path,policy_id=f'crow_v15_route_interpolation_{x}',revision=max(a.revision,b.revision)+1,contract_version=a.contract_version,world_fingerprint=a.world_fingerprint,task_fingerprint=a.task_fingerprint,observation_fingerprint=a.observation_fingerprint,action_fingerprint=a.action_fingerprint,layers=layers,observation_mean=a.observation_mean,observation_inverse_standard_deviation=a.observation_inverse_standard_deviation,action_bias=a.action_bias,action_scale=a.action_scale,observation_clip=a.observation_clip,action_clip=a.action_clip,library_path=str(lib))
 p=read_policy_pack(path,library_path=str(lib));assert not p.critic_layers and len(p.action_log_standard_deviation)==0
 with path.open('rb') as f,(A/(path.name+'.gz')).open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:shutil.copyfileobj(f,z)
 h=hashlib.sha256(path.read_bytes()).hexdigest()
 assert hashlib.sha256(gzip.decompress((A/(path.name+'.gz')).read_bytes())).hexdigest()==h
 results.append({'alpha':alpha,'policy_sha256':h})
(A/'policies.json').write_text(json.dumps({'old_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'new_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'results':results},indent=2)+'\n')
