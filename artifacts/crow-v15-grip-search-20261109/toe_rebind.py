from pathlib import Path
import gzip,hashlib,json,os,re,subprocess
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-v15-grip-search-20261109';B=R/'build/crow-v15-grip-search-20261109';C=B/'toe-envelope';assert json.loads((A/'toe-envelope-contract-comparison.json').read_text())['all_diagnostic_lines_equal']
cmd=json.loads((A/'toe-integration-neutral-bindings.json').read_text())['argv'];cmd[cmd.index('--steps')+1]='1';env={**os.environ,'DYLD_LIBRARY_PATH':str(C),'PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=120);(A/'toe-envelope-original-policy-rejection.stderr').write_text(p.stderr)
m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',p.stderr)
if not p.returncode or not m:raise RuntimeError('Expected changed contract rejection: '+p.stderr[-1000:])
world,task,obs,act=map(int,m.groups());reports=[]
for label,flag in [('route','--base-policy-pack'),('landing','--policy-pack')]:
 source=Path(cmd[cmd.index(flag)+1]);src=read_policy_pack(source,library_path=str(C/'libmetalrobo.dylib'));assert not src.critic_layers and len(src.action_log_standard_deviation)==0
 layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers];out=C/(label+'.policypack')
 write_policy_pack(out,policy_id='crow_toe_envelope_diagnostic_'+label,revision=src.revision+1,contract_version=src.contract_version,world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,layers=layers,observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(C/'libmetalrobo.dylib'))
 dst=read_policy_pack(out,library_path=str(C/'libmetalrobo.dylib'));assert len(src.layers)==len(dst.layers) and all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) and x.activation==y.activation for x,y in zip(src.layers,dst.layers))
 for n in ['observation_mean','observation_inverse_standard_deviation','action_bias','action_scale']:assert np.array_equal(getattr(src,n),getattr(dst,n))
 with out.open('rb') as f,(A/('toe-envelope-'+label+'.policypack.gz')).open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:z.write(f.read())
 reports.append({'label':label,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'rebound_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'layers_and_transforms_equal':True})
report={'old_pack_rejected':True,'world':world,'task':task,'observation':obs,'action':act,'actors':reports,'scope':'diagnostic rebind only; altered physical body'};(A/'toe-rebind-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
