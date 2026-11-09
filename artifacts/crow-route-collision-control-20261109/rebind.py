from pathlib import Path
import hashlib,json,os,re,subprocess
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-route-collision-control-20261109';B=R/'build/crow-route-collision-control-20261109'
lib=B/'libmetalrobo.dylib';old=R/'build/crow-phase-policy-20261109/build/crow-envelope-restore-policy/build/crow-flight-source-diagnostics-01/versioned-flight-bench.policypack';new=B/'collision-enabled.policypack'
cmd=json.loads((R/'artifacts/crow-route-teacher-v1-20261109/route-results.json').read_text())['argv']
for k,v in [('--steps','1'),('--state-trace',str(B/'contract.tsv'))]:cmd[cmd.index(k)+1]=v
env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=120)
(A/'old-contract.stdout').write_text(p.stdout);(A/'old-contract.stderr').write_text(p.stderr)
m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',p.stderr)
if not p.returncode or not m:raise RuntimeError('Expected collision world mismatch was not reported: '+p.stderr[-4000:])
world,task,obs,act=map(int,m.groups());src=read_policy_pack(old,library_path=str(lib))
def layers(xs):return [PolicyDenseLayerArtifact(weights=x.weights,bias=x.bias,activation=x.activation) for x in xs]
write_policy_pack(new,policy_id='crow_v15_collision_route_bench',revision=1,contract_version=src.contract_version,world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,layers=layers(src.layers),critic_layers=layers(src.critic_layers),observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,critic_observation_mean=src.critic_observation_mean,critic_observation_inverse_standard_deviation=src.critic_observation_inverse_standard_deviation,action_log_standard_deviation=src.action_log_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(lib))
dst=read_policy_pack(new,library_path=str(lib))
def same(x,y):return np.array_equal(x,y)
actor=all(same(x.weights,y.weights) and same(x.bias,y.bias) and x.activation==y.activation for x,y in zip(src.layers,dst.layers)) and len(src.layers)==len(dst.layers)
critic=all(same(x.weights,y.weights) and same(x.bias,y.bias) and x.activation==y.activation for x,y in zip(src.critic_layers,dst.critic_layers)) and len(src.critic_layers)==len(dst.critic_layers)
fields=['observation_mean','observation_inverse_standard_deviation','critic_observation_mean','critic_observation_inverse_standard_deviation','action_log_standard_deviation','action_bias','action_scale']
arrays={f:same(getattr(src,f),getattr(dst,f)) for f in fields}
if not actor or not critic or not all(arrays.values()):raise RuntimeError('Policy tensor mismatch')
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'old_policy_rejected':True,'actor_layers_equal':actor,'critic_layers_equal':critic,'other_arrays_equal':arrays,'old_policy_sha256':h(old),'new_policy_sha256':h(new),'world':world,'task':task,'observation':obs,'action':act,'observer_library_sha256':h(lib)}
(A/'contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
