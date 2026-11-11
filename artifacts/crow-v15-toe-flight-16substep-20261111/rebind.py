from pathlib import Path
import gzip,hashlib,json,re
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-v15-toe-flight-16substep-20261111';D=R/'build/crow-v15-toe-damping-isolation-20261110/zero';B=R/'build/crow-v15-toe-flight-16substep-20261111';B.mkdir(exist_ok=True)
with gzip.open(A/'initial-contract-failure.stderr.gz','rt') as f:msg=f.read()
m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',msg);assert m
world,task,obs,act=map(int,m.groups());source=R/'build/crow-v15-toe-damping-isolation-20261110/zero-4/zero.policypack';target=B/'zero.policypack';src=read_policy_pack(source,library_path=str(D/'libmetalrobo.dylib'));layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers];write_policy_pack(target,policy_id='crow_substep_zero_16',revision=1,contract_version=src.contract_version,world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,layers=layers,observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(D/'libmetalrobo.dylib'));dst=read_policy_pack(target,library_path=str(D/'libmetalrobo.dylib'));assert len(src.layers)==len(dst.layers) and all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) and np.count_nonzero(y.weights)==0 and np.count_nonzero(y.bias)==0 for x,y in zip(src.layers,dst.layers)) and np.count_nonzero(dst.action_bias)==0
for field in ('observation_mean','observation_inverse_standard_deviation','action_bias','action_scale'):assert np.array_equal(getattr(src,field),getattr(dst,field))
assert src.observation_clip==dst.observation_clip and src.action_clip==dst.action_clip
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();r={'source_sha256':h(source),'target_sha256':h(target),'zero_layers_verified':True,'normalization_and_clipping_preserved':True,'contract':{'world':world,'task':task,'observation':obs,'action':act}};(A/'rebind.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
