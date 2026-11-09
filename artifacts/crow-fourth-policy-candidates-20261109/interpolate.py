from pathlib import Path
import gzip,hashlib,json,os,shutil,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-fourth-policy-candidates-20261109';B=R/'build/crow-fourth-stage3-training-20261109';lib=R/'build/lib/libmetalrobo.dylib'
alpha=float(os.environ['CROW_ALPHA']);name='mix'+str(round(alpha*100))
p0=R/'build/crow-expanded-stage3-specialist-20261109/checkpoint-160.policypack';p1=R/'build/crow-fourth-stage3-training-20261109/checkpoint-160.policypack'
a=read_policy_pack(p0,library_path=str(lib));b=read_policy_pack(p1,library_path=str(lib))
for k in ['contract_version','world_fingerprint','task_fingerprint','observation_fingerprint','action_fingerprint','observation_clip','action_clip']:assert getattr(a,k)==getattr(b,k),k
for k in ['observation_mean','observation_inverse_standard_deviation','action_bias','action_scale']:assert np.array_equal(getattr(a,k),getattr(b,k)),k
layers=[]
for x,y in zip(a.layers,b.layers):
 assert x.activation==y.activation and x.weights.shape==y.weights.shape
 w=((1-alpha)*x.weights+alpha*y.weights).astype(np.float32);bias=((1-alpha)*x.bias+alpha*y.bias).astype(np.float32)
 layers.append(PolicyDenseLayerArtifact(weights=w,bias=bias,activation=x.activation))
out=B/('checkpoint-'+name+'.policypack')
write_policy_pack(out,policy_id='crow_waypoint_interpolation_'+name,revision=max(a.revision,b.revision)+1,contract_version=a.contract_version,world_fingerprint=a.world_fingerprint,task_fingerprint=a.task_fingerprint,observation_fingerprint=a.observation_fingerprint,action_fingerprint=a.action_fingerprint,layers=layers,observation_mean=a.observation_mean,observation_inverse_standard_deviation=a.observation_inverse_standard_deviation,action_bias=a.action_bias,action_scale=a.action_scale,observation_clip=a.observation_clip,action_clip=a.action_clip,library_path=str(lib))
with out.open('rb') as f,(A/(name+'.policypack.gz')).open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:shutil.copyfileobj(f,z)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert hashlib.sha256(gzip.decompress((A/(name+'.policypack.gz')).read_bytes())).hexdigest()==sha(out)
(A/(name+'.json')).write_text(json.dumps({'alpha':alpha,'base_sha256':sha(p0),'updated_sha256':sha(p1),'output_sha256':sha(out),'scope':'development candidate, linear parameter interpolation, requires full flight validation'},indent=2)+'\n')
print(name,flush=True)
