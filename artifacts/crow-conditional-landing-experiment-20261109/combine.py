from pathlib import Path
import gzip,hashlib,json,os,shutil,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-conditional-landing-experiment-20261109';B=R/'build/crow-conditional-landing-experiment-20261109';lib=R/'build/lib/libmetalrobo.dylib';epoch=os.environ['CROW_EPOCH']
paths=[R/'build/crow-four-expert-native-integration-20261109/landing.policypack',R/f'build/crow-late-landing-training-20261109/landing-{epoch}.policypack']
pol=[read_policy_pack(p,library_path=str(lib)) for p in paths];ref=pol[0]
for p in pol:
 for k in ['contract_version','world_fingerprint','task_fingerprint','observation_fingerprint','action_fingerprint','observation_clip','action_clip']:assert getattr(p,k)==getattr(ref,k),k
 for k in ['observation_mean','observation_inverse_standard_deviation','action_bias','action_scale']:assert np.array_equal(getattr(p,k),getattr(ref,k)),k
layers=[];a=[p.layers[0] for p in pol];layers.append(PolicyDenseLayerArtifact(weights=np.concatenate([p.weights for p in a]),bias=np.concatenate([p.bias for p in a]),activation=a[0].activation))
for i in [1,2]:
 a=[p.layers[i] for p in pol];rows,cols=a[0].weights.shape;w=np.zeros((2*rows,2*cols),dtype=np.float32)
 for n,p in enumerate(a):w[n*rows:(n+1)*rows,n*cols:(n+1)*cols]=p.weights
 layers.append(PolicyDenseLayerArtifact(weights=w,bias=np.concatenate([p.bias for p in a]),activation=a[0].activation))
a=[p.layers[-1] for p in pol];out,cols=a[0].weights.shape;w=np.zeros((out,2*cols),dtype=np.float32);w[:,:cols]=a[0].weights;bias=a[0].bias.copy()
w[4:8,:cols]=0;w[4:8,cols:2*cols]=a[1].weights[:4];bias[4:8]=a[1].bias[:4]
layers.append(PolicyDenseLayerArtifact(weights=w,bias=bias,activation=a[0].activation))
name='conditional-'+epoch;path=B/(name+'.policypack')
write_policy_pack(path,policy_id='crow_conditional_landing_'+epoch,revision=max(x.revision for x in pol)+1,contract_version=ref.contract_version,world_fingerprint=ref.world_fingerprint,task_fingerprint=ref.task_fingerprint,observation_fingerprint=ref.observation_fingerprint,action_fingerprint=ref.action_fingerprint,layers=layers,observation_mean=ref.observation_mean,observation_inverse_standard_deviation=ref.observation_inverse_standard_deviation,action_bias=ref.action_bias,action_scale=ref.action_scale,observation_clip=ref.observation_clip,action_clip=ref.action_clip,library_path=str(lib))
p=read_policy_pack(path,library_path=str(lib));x=np.random.default_rng(117).normal(size=(64,816)).astype(np.float32);c=p.actor_mean(x)
errors=[float(np.max(np.abs(z.actor_mean(x)[:,:4]-c[:,slot:slot+4]))) for z,slot in zip(pol,[0,4])];assert max(errors)<2e-5,errors
with path.open('rb') as f,(A/(name+'.policypack.gz')).open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:shutil.copyfileobj(f,z)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert hashlib.sha256(gzip.decompress((A/(name+'.policypack.gz')).read_bytes())).hexdigest()==sha(path)
(A/(name+'.json')).write_text(json.dumps({'sources':[{'path':str(p),'sha256':sha(p)} for p in paths],'candidate_sha256':sha(path),'max_output_error':errors,'scope':'diagnostic only, requires new fingerprinted task'},indent=2)+'\n')
print(name,errors,flush=True)
