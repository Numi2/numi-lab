from pathlib import Path
import gzip,hashlib,json,os,shutil,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-stage3-specialist-experiment-20261109';B=R/'build/crow-stage3-specialist-experiment-20261109';lib=R/'build/lib/libmetalrobo.dylib'
epoch=os.environ['CROW_EPOCH'];stage2=os.environ.get('CROW_STAGE2','baseline')
paths=[R/'build/crow-stage2-adapter-training-20261109/frozen-route-actor.policypack',
       R/('build/crow-route-no-vision-diagnostic-20261109/frozen-route-actor.policypack' if stage2=='baseline' else f'build/crow-expanded-stage2-training-20261109/checkpoint-{stage2}.policypack'),
       R/f'build/crow-expanded-stage3-specialist-20261109/checkpoint-{epoch}.policypack']
pol=[read_policy_pack(p,library_path=str(lib)) for p in paths]
native=read_policy_pack(R/'build/crow-mixed-native-integration-20261109/route.policypack',library_path=str(lib))
for p in pol:
 for k in ['contract_version','world_fingerprint','observation_fingerprint','action_fingerprint','observation_clip','action_clip']:assert getattr(p,k)==getattr(native,k),k
 for k in ['observation_mean','observation_inverse_standard_deviation','action_bias','action_scale']:assert np.array_equal(getattr(p,k),getattr(native,k)),k
layers=[];a=[p.layers[0] for p in pol]
layers.append(PolicyDenseLayerArtifact(weights=np.concatenate([p.weights for p in a]),bias=np.concatenate([p.bias for p in a]),activation=a[0].activation))
for i in [1,2]:
 a=[p.layers[i] for p in pol];rows,cols=a[0].weights.shape;w=np.zeros((3*rows,3*cols),dtype=np.float32)
 for n,p in enumerate(a):w[n*rows:(n+1)*rows,n*cols:(n+1)*cols]=p.weights
 layers.append(PolicyDenseLayerArtifact(weights=w,bias=np.concatenate([p.bias for p in a]),activation=a[0].activation))
a=[p.layers[-1] for p in pol];out,cols=a[0].weights.shape;w=np.zeros((out,3*cols),dtype=np.float32);w[:,:cols]=a[0].weights;bias=a[0].bias.copy()
for n,slot in [(1,4),(2,8)]:
 w[slot:slot+4,:cols]=0;w[slot:slot+4,n*cols:(n+1)*cols]=a[n].weights[:4];bias[slot:slot+4]=a[n].bias[:4]
layers.append(PolicyDenseLayerArtifact(weights=w,bias=bias,activation=a[0].activation))
name=f'stage3-{epoch}-stage2-{stage2}';path=B/(name+'.policypack')
write_policy_pack(path,policy_id='crow_diagnostic_'+name,revision=max(x.revision for x in pol)+1,contract_version=native.contract_version,world_fingerprint=native.world_fingerprint,task_fingerprint=native.task_fingerprint,observation_fingerprint=native.observation_fingerprint,action_fingerprint=native.action_fingerprint,layers=layers,observation_mean=native.observation_mean,observation_inverse_standard_deviation=native.observation_inverse_standard_deviation,action_bias=native.action_bias,action_scale=native.action_scale,observation_clip=native.observation_clip,action_clip=native.action_clip,library_path=str(lib))
p=read_policy_pack(path,library_path=str(lib));x=np.random.default_rng(71).normal(size=(64,816)).astype(np.float32);c=p.actor_mean(x)
errors=[float(np.max(np.abs(z.actor_mean(x)[:,:4]-c[:,slot:slot+4]))) for z,slot in zip(pol,[0,4,8])];assert max(errors)<2e-5,errors
with path.open('rb') as f,(A/(name+'.policypack.gz')).open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:shutil.copyfileobj(f,z)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert hashlib.sha256(gzip.decompress((A/(name+'.policypack.gz')).read_bytes())).hexdigest()==sha(path)
(A/(name+'.json')).write_text(json.dumps({'sources':[{'path':str(p),'sha256':sha(p)} for p in paths],'candidate_sha256':sha(path),'max_output_error':errors,'scope':'diagnostic shader only; output slots 8-11 are stage3 expert'},indent=2)+'\n')
print(name,errors,flush=True)
