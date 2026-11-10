from pathlib import Path
import gzip,hashlib,json,re,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-v15-substep-contract-20261110';D=R/'artifacts/crow-v15-full-substep-sensitivity-20261110';B=R/'build/crow-v15-full-substep-sensitivity-20261110';lib=R/'build/crow-v15-long-perch-host-20261110/libmetalrobo.dylib'; lines=(A/'audit.json').read_text();r=json.loads(lines)
assert r['returncode']==0 and len(r['records'])==3
for line in r['records']:
 assert all(k+'=1' in line for k in ('compile','world_except_dt','model','layout','task','header','constraints')),line
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();report={}
for sub in (8,16):
 with gzip.open(D/('j-original-'+str(sub)+'.stderr.gz'),'rt') as f:err=f.read()
 m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',err);assert m
 fp=tuple(map(int,m.groups()));report[str(sub)]={'fingerprints':list(fp),'policies':{}}
 for name,srcpath in [('landing',R/'build/crow-v15-long-perch-host-20261110/landing.policypack'),('route',R/'build/crow-v15-second-clearance-learning-20261110/corrective-actor.policypack')]:
  src=read_policy_pack(srcpath,library_path=str(lib));assert not src.critic_layers and len(src.action_log_standard_deviation)==0
  out=B/(name+'-'+str(sub)+'.policypack');layers=[PolicyDenseLayerArtifact(weights=z.weights.copy(),bias=z.bias.copy(),activation=z.activation) for z in src.layers]
  write_policy_pack(out,policy_id='crow_v15_substep_diagnostic_'+name,revision=src.revision+1,contract_version=src.contract_version,world_fingerprint=fp[0],task_fingerprint=fp[1],observation_fingerprint=fp[2],action_fingerprint=fp[3],layers=layers,observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(lib))
  dst=read_policy_pack(out,library_path=str(lib));assert all(x.activation==y.activation and np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) for x,y in zip(src.layers,dst.layers)) and len(src.layers)==len(dst.layers)
  for k in ('observation_mean','observation_inverse_standard_deviation','action_bias','action_scale'):assert np.array_equal(getattr(src,k),getattr(dst,k))
  for k in ('contract_version','observation_clip','action_clip'):assert getattr(src,k)==getattr(dst,k)
  assert (dst.world_fingerprint,dst.task_fingerprint,dst.observation_fingerprint,dst.action_fingerprint)==fp
  x=np.random.default_rng(72).normal(size=(16,src.actor_observation_count)).astype(np.float32);err=float(np.max(np.abs(src.actor_mean(x)-dst.actor_mean(x))));assert err<2e-5
  report[str(sub)]['policies'][name]={'source_sha256':sha(srcpath),'rebound_sha256':sha(out),'identical_arrays':True,'max_checked_output_error':err}
(A/'rebind-results.json').write_text(json.dumps({'settings':report,'limits':'Diagnostic copies bound to observed native contracts after exact compiled-model/operator comparison. The changed numerical physics is not qualified.'},indent=2)+'\n');print(json.dumps(report,indent=2))
