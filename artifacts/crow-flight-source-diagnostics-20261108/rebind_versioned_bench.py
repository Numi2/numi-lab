from pathlib import Path
import json,os,re,subprocess,hashlib
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-flight-source-diagnostics-20261108';B=R/'build/crow-flight-source-diagnostics-01'
lib=B/'versioned-flight-bench-lib/libmetalrobo.dylib';shader=B/'versioned-flight-bench.metallib'
old=B/'obstacle-free-policy.policypack';new=B/'versioned-flight-bench.policypack'
base=json.load(open(A/'flight-bench-results.json'))['argv']
base[base.index('--metallib')+1]=str(shader);base[base.index('--policy-pack')+1]=str(old)
base[base.index('--steps')+1]='1'
env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent),'PYTHONDONTWRITEBYTECODE':'1'}
r=subprocess.run(base,cwd=R,env=env,text=True,capture_output=True,timeout=120)
(A/'versioned-bench-old-policy.stderr').write_text(r.stderr)
m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',r.stderr)
if not r.returncode or not m:raise RuntimeError('Old contract was not rejected: '+r.stderr)
world,task,obs,act=map(int,m.groups())
src=read_policy_pack(old,library_path=str(lib))
def layers(xs):return [PolicyDenseLayerArtifact(weights=x.weights,bias=x.bias,activation=x.activation) for x in xs]
write_policy_pack(new,policy_id='crow_v15_versioned_flight_phase_bench',revision=1,contract_version=src.contract_version,
 world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,
 layers=layers(src.layers),critic_layers=layers(src.critic_layers),observation_mean=src.observation_mean,
 observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,critic_observation_mean=src.critic_observation_mean,
 critic_observation_inverse_standard_deviation=src.critic_observation_inverse_standard_deviation,
 action_log_standard_deviation=src.action_log_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,
 observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(lib))
dest=read_policy_pack(new,library_path=str(lib))
equal=all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) for x,y in zip(src.layers,dest.layers))
if not equal:raise RuntimeError('Actor tensors differ')
(A/'versioned-bench-contract.json').write_text(json.dumps({'old_policy_rejected':True,'actor_weights_identical':equal,
 'old_policy_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'new_policy_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),
 'world':world,'task':task,'observation':obs,'action':act},indent=2)+'\n')
print('old policy rejected; actor tensors identical')
