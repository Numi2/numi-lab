"""Rebind existing unchanged tensors to the collision-on manifest for a smoke test.

This is a compatibility experiment, not training or policy qualification.
"""
from pathlib import Path
import hashlib,json,re
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
r=Path('/Users/n/Developer/numi-crow-rl-20261107');a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114'
err=(r/'artifacts/crow-v15-course-collision-restored-20261114/smoke.stderr').read_text()
m=re.search(r'policy=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\) task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',err);assert m
fingerprints=list(map(int,m.groups()[4:]));reports=[]
for label in ('landing','route'):
    source=r/'build/crow-v15-mirrored-pronation-20261113'/(label+'-candidate.policypack');target=b/(label+'-rebound.policypack');library=str(b/'libmetalrobo.dylib')
    src=read_policy_pack(source,library_path=library);assert not src.critic_layers and len(src.action_log_standard_deviation)==0
    layers=[PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in src.layers]
    write_policy_pack(target,policy_id='crow_recorded_restart_diagnostic_'+label,revision=src.revision+1,contract_version=src.contract_version,world_fingerprint=fingerprints[0],task_fingerprint=fingerprints[1],observation_fingerprint=fingerprints[2],action_fingerprint=fingerprints[3],layers=layers,observation_mean=src.observation_mean,observation_inverse_standard_deviation=src.observation_inverse_standard_deviation,action_bias=src.action_bias,action_scale=src.action_scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=library)
    dst=read_policy_pack(target,library_path=library)
    assert len(src.layers)==len(dst.layers) and all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) and x.activation==y.activation for x,y in zip(src.layers,dst.layers))
    assert all(np.array_equal(getattr(src,k),getattr(dst,k)) for k in ('observation_mean','observation_inverse_standard_deviation','action_bias','action_scale'))
    reports.append({'policy':label,'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'rebound_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'identical_tensors':True})
(a/'policy-rebind.json').write_text(json.dumps({'fingerprints':fingerprints,'reports':reports,'scope':'Compatibility-only; no new training, no performance claim.'},indent=2)+'\n')
print('identical tensors rebound for diagnostic use')
