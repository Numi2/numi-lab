from pathlib import Path
import hashlib,json,os,re,subprocess
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-v15-toe-ground-32-rebound-20261111';B=R/'build/crow-v15-toe-ground-32-rebound-20261111';lib=R/'build/crow-v15-two-segment-frozen-flight-dense-contact-20261111/libmetalrobo.dylib';err=(R/'artifacts/crow-v15-toe-ground-substeps-screen-20261111/stderr.gz')
import gzip
err=gzip.open(err,'rt').read();m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',err);assert m,err
world,task,obs,act=map(int,m.groups());source=R/'build/crow-v15-two-segment-frozen-flight-contact-map-20261111/two-segment-zero.policypack';src=read_policy_pack(source,library_path=str(lib));assert not src.critic_layers and len(src.action_log_standard_deviation)==0;assert all(not np.any(x.weights) and not np.any(x.bias) for x in src.layers) and not np.any(src.action_bias)
layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers];target=B/'zero-32.policypack';write_policy_pack(target,policy_id='crow_two_segment_frozen_flight_zero_diagnostic',revision=src.revision+1,contract_version=src.contract_version,world_fingerprint=world,task_fingerprint=task,observation_fingerprint=obs,action_fingerprint=act,layers=layers,observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=str(lib));dst=read_policy_pack(target,library_path=str(lib));assert len(src.layers)==len(dst.layers) and all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) and x.activation==y.activation for x,y in zip(src.layers,dst.layers))
for field in ('observation_mean','observation_inverse_standard_deviation','action_bias','action_scale'):assert np.array_equal(getattr(src,field),getattr(dst,field))
assert src.observation_clip==dst.observation_clip and src.action_clip==dst.action_clip
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();report={'source_policy_sha256':h(source),'rebound_policy_sha256':h(target),'identical_layers_and_transforms':True,'zero_output_verified':True,'world':world,'task':task,'observation':obs,'action':act};(A/'policy-binding.json').write_text(json.dumps(report,indent=2)+'\n')
