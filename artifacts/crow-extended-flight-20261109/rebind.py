from pathlib import Path
import hashlib,json,os,re,subprocess
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-extended-flight-20261109';B=R/'build/crow-extended-flight-20261109'
srcpacks={'collision-on':R/'build/crow-route-collision-control-20261109/collision-enabled.policypack','collision-off':R/'build/crow-phase-policy-20261109/build/crow-envelope-restore-policy/build/crow-flight-source-diagnostics-01/versioned-flight-bench.policypack'}
out={}
for mode,old in srcpacks.items():
 lib=B/mode/'libmetalrobo.dylib';new=B/mode/'extended.policypack'
 cmd=json.loads((R/'artifacts/crow-frozen-flight-law-20261109/20261120/tracking-results.json').read_text())['argv']
 for k,v in [('--steps','1'),('--policy-pack',str(old)),('--state-trace',str(B/mode/'contract.tsv'))]:cmd[cmd.index(k)+1]=v
 env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent),'PYTHONDONTWRITEBYTECODE':'1'};env.pop('CROW_CONTACT_TRACE',None)
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=120);(A/(mode+'-old.stderr')).write_text(p.stderr)
 m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',p.stderr)
 if not p.returncode or not m:raise RuntimeError('Expected horizon contract mismatch: '+p.stderr[-4000:])
 world,task,obs,act=map(int,m.groups());src=read_policy_pack(old,library_path=str(lib))
 def layers(xs):return [PolicyDenseLayerArtifact(weights=x.weights,bias=x.bias,activation=x.activation) for x in xs]
 write_policy_pack(new,policy_id='crow_v15_extended_flight_'+mode,revision=1,contract_version=src.contract_version,world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,layers=layers(src.layers),critic_layers=layers(src.critic_layers),observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,critic_observation_mean=src.critic_observation_mean,critic_observation_inverse_standard_deviation=src.critic_observation_inverse_standard_deviation,action_log_standard_deviation=src.action_log_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(lib))
 dst=read_policy_pack(new,library_path=str(lib));fields=['observation_mean','observation_inverse_standard_deviation','critic_observation_mean','critic_observation_inverse_standard_deviation','action_log_standard_deviation','action_bias','action_scale'];eq=lambda x,y:np.array_equal(x,y)
 same=all(eq(x.weights,y.weights) and eq(x.bias,y.bias) and x.activation==y.activation for l,r in [(src.layers,dst.layers),(src.critic_layers,dst.critic_layers)] for x,y in zip(l,r)) and all(eq(getattr(src,f),getattr(dst,f)) for f in fields)
 if not same:raise RuntimeError('Policy payload differs')
 h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 out[mode]={'old_rejected':True,'payload_equal':same,'world':world,'task':task,'observation':obs,'action':act,'old_policy_sha256':h(old),'new_policy_sha256':h(new),'library_sha256':h(lib)}
(A/'contracts.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
