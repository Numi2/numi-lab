from pathlib import Path
import hashlib,json,os,re,subprocess
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-hierarchical-contract-20261109';B=R/'build/crow-hierarchical-contract-20261109'
lib=B/'libmetalrobo.dylib';old=R/'build/crow-extended-flight-20261109/collision-on/extended.policypack';new=B/'hierarchical-initial.policypack'
cmd=json.loads((R/'artifacts/crow-hierarchical-oracle-20261109/route-results.json').read_text())['argv']
for k,v in [('--steps','1'),('--policy-pack',str(old)),('--state-trace',str(B/'old-contract.tsv'))]:cmd[cmd.index(k)+1]=v
env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'PYTHONDONTWRITEBYTECODE':'1'};env.pop('CROW_CONTACT_TRACE',None);env.pop('CROW_SUPPORT_TRACE',None)
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=120);(A/'old-contract.stderr').write_text(p.stderr)
m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',p.stderr)
if not p.returncode or not m:raise RuntimeError('Expected hierarchy contract rejection: '+p.stderr[-4000:])
world,task,obs,act=map(int,m.groups());src=read_policy_pack(old,library_path=str(lib))
layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers]
assert layers[-1].weights.shape[0]==18 and layers[-1].bias.shape==(18,)
layers[-1].weights[:4,:]=0
layers[-1].bias[:4]=np.array([0.5,0,0,(1.2-.975)/.325],dtype=np.float32)
critic=[PolicyDenseLayerArtifact(weights=x.weights,bias=x.bias,activation=x.activation) for x in src.critic_layers]
write_policy_pack(new,policy_id='crow_v15_hierarchical_initial_development',revision=1,contract_version=src.contract_version,world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,layers=layers,critic_layers=critic,observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,critic_observation_mean=src.critic_observation_mean,critic_observation_inverse_standard_deviation=src.critic_observation_inverse_standard_deviation,action_log_standard_deviation=src.action_log_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(lib))
dst=read_policy_pack(new,library_path=str(lib))
assert np.array_equal(dst.layers[-1].weights[:4],np.zeros_like(dst.layers[-1].weights[:4]))
assert np.array_equal(dst.layers[-1].weights[4:],src.layers[-1].weights[4:])
assert np.array_equal(dst.layers[-1].bias[4:],src.layers[-1].bias[4:])
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'old_policy_rejected':True,'task_id':'american_crow_v15_hierarchical_navigation_development_v1','world':world,'task':task,'observation':obs,'action':act,'old_policy_sha256':h(old),'new_policy_sha256':h(new),'library_sha256':h(lib),'first_four_action_heads':'zero weights, constant initial speed 0.25 m/s, lateral 0, heading error 0, height 1.20 m','other_action_heads':'unchanged','note':'new policy has new task semantics; explicit warm start, not a qualified or trained navigator'}
(A/'contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
