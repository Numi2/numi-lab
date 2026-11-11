from pathlib import Path
import hashlib,json,os,re,subprocess
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-v15-landing-approach-dev-20261111';B=R/'build/crow-v15-landing-approach-dev-20261111';L=B/'trace';D=R/'build/crow-v15-course-collisions-enabled-20261111/enabled';O=A/'binding';O.mkdir(exist_ok=True);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cmd=json.loads((R/'artifacts/crow-v15-teacher-clearance-route-dev-20261111/height/command.json').read_text());cmd[cmd.index('--metallib')+1]=str(B/'tracking.metallib');cmd[cmd.index('--steps')+1]='2';cmd[cmd.index('--state-trace')+1]=str(B/'bind-trace.tsv');cmd[cmd.index('--crow-replay-pack')+1]=str(B/'bind-replay.json');env={**os.environ,'DYLD_LIBRARY_PATH':str(L),'PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,capture_output=True,text=True,timeout=180);(O/'initial.stdout').write_text(p.stdout);(O/'initial.stderr').write_text(p.stderr)
m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',p.stderr)
if not m:raise RuntimeError('Expected changed task fingerprint was absent: '+p.stderr[-1200:])
world,task,obs,act=map(int,m.groups());reports=[]
for option,label in [('--policy-pack','landing'),('--base-policy-pack','route')]:
 source=D/(label+'.policypack');src=read_policy_pack(source,library_path=str(D/'libmetalrobo.dylib'));assert not src.critic_layers and len(src.action_log_standard_deviation)==0
 layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers];target=B/(label+'.policypack')
 write_policy_pack(target,policy_id='crow_landing_horizon_'+label,revision=src.revision+1,contract_version=src.contract_version,world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,layers=layers,observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(L/'libmetalrobo.dylib'))
 dst=read_policy_pack(target,library_path=str(L/'libmetalrobo.dylib'));assert len(src.layers)==len(dst.layers) and all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) and x.activation==y.activation for x,y in zip(src.layers,dst.layers));assert all(np.array_equal(getattr(src,k),getattr(dst,k)) for k in ('observation_mean','observation_inverse_standard_deviation','action_bias','action_scale'));assert src.observation_clip==dst.observation_clip and src.action_clip==dst.action_clip
 cmd[cmd.index(option)+1]=str(target);reports.append({'actor':label,'source_sha256':h(source),'target_sha256':h(target),'identical_layers_and_transforms':True})
(O/'policy-binding.json').write_text(json.dumps({'fingerprints':[world,task,obs,act],'reports':reports},indent=2)+'\n');p=subprocess.run(cmd,cwd=R,env=env,capture_output=True,text=True,timeout=180);(O/'rebound.stdout').write_text(p.stdout);(O/'rebound.stderr').write_text(p.stderr)
if p.returncode:raise RuntimeError('rebound preflight failed: '+p.stderr[-1500:])
(O/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');print('rebound preflight passed')
