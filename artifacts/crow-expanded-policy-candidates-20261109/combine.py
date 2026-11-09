from pathlib import Path
import gzip,hashlib,json,os,shutil,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-expanded-policy-candidates-20261109';B=R/'build/crow-expanded-policy-candidates-20261109';lib=R/'build/lib/libmetalrobo.dylib'
v2=os.environ['CROW_STAGE2'];v3=os.environ['CROW_STAGE3'];name=os.environ['CROW_CANDIDATE']
routepath=R/(f'build/crow-expanded-stage3-training-20261109/checkpoint-{v3}.policypack' if v3!='baseline' else 'build/crow-stage2-adapter-training-20261109/frozen-route-actor.policypack')
expertpath=R/(f'build/crow-expanded-stage2-training-20261109/checkpoint-{v2}.policypack' if v2!='baseline' else 'build/crow-route-no-vision-diagnostic-20261109/frozen-route-actor.policypack')
route=read_policy_pack(routepath,library_path=str(lib));expert=read_policy_pack(expertpath,library_path=str(lib));native=read_policy_pack(R/'build/crow-mixed-native-integration-20261109/route.policypack',library_path=str(lib))
for k in ['contract_version','world_fingerprint','observation_fingerprint','action_fingerprint','observation_clip','action_clip']:
 assert getattr(route,k)==getattr(expert,k)==getattr(native,k),k
for k in ['observation_mean','observation_inverse_standard_deviation','action_bias','action_scale']:
 assert np.array_equal(getattr(route,k),getattr(expert,k)) and np.array_equal(getattr(route,k),getattr(native,k)),k
layers=[]
u,v=route.layers[0],expert.layers[0]
layers.append(PolicyDenseLayerArtifact(weights=np.concatenate([u.weights,v.weights]),bias=np.concatenate([u.bias,v.bias]),activation=u.activation))
for u,v in zip(route.layers[1:-1],expert.layers[1:-1]):
 rows,cols=u.weights.shape;w=np.zeros((2*rows,2*cols),dtype=np.float32);w[:rows,:cols]=u.weights;w[rows:,cols:]=v.weights
 layers.append(PolicyDenseLayerArtifact(weights=w,bias=np.concatenate([u.bias,v.bias]),activation=u.activation))
u,v=route.layers[-1],expert.layers[-1];out,cols=u.weights.shape;w=np.zeros((out,2*cols),dtype=np.float32);w[:,:cols]=u.weights;bias=u.bias.copy();w[4:8,:cols]=0;w[4:8,cols:]=v.weights[:4];bias[4:8]=v.bias[:4]
layers.append(PolicyDenseLayerArtifact(weights=w,bias=bias,activation=u.activation))
path=B/(name+'.policypack')
write_policy_pack(path,policy_id='crow_expanded_'+name,revision=max(route.revision,expert.revision)+1,contract_version=native.contract_version,world_fingerprint=native.world_fingerprint,task_fingerprint=native.task_fingerprint,observation_fingerprint=native.observation_fingerprint,action_fingerprint=native.action_fingerprint,layers=layers,observation_mean=native.observation_mean,observation_inverse_standard_deviation=native.observation_inverse_standard_deviation,action_bias=native.action_bias,action_scale=native.action_scale,observation_clip=native.observation_clip,action_clip=native.action_clip,library_path=str(lib))
p=read_policy_pack(path,library_path=str(lib));assert not p.critic_layers and not len(p.action_log_standard_deviation)
x=np.random.default_rng(71).normal(size=(64,816)).astype(np.float32);a=route.actor_mean(x);b=expert.actor_mean(x);c=p.actor_mean(x)
e1=float(np.max(np.abs(a[:,:4]-c[:,:4])));e2=float(np.max(np.abs(b[:,:4]-c[:,4:8])));assert e1<2e-5 and e2<2e-5,(e1,e2)
with path.open('rb') as f,(A/(name+'.policypack.gz')).open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:shutil.copyfileobj(f,z)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert hashlib.sha256(gzip.decompress((A/(name+'.policypack.gz')).read_bytes())).hexdigest()==sha(path)
(A/(name+'.json')).write_text(json.dumps({'route_input':str(routepath),'expert_input':str(expertpath),'route_sha256':sha(routepath),'expert_sha256':sha(expertpath),'candidate_sha256':sha(path),'route_max_error':e1,'expert_max_error':e2,'native_contract_from':sha(R/'build/crow-mixed-native-integration-20261109/route.policypack'),'critic_and_stochastic_head_removed':True},indent=2)+'\n')
print(name,e1,e2,flush=True)
