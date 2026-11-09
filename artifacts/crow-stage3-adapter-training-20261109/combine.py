from pathlib import Path
import gzip,hashlib,json,os,shutil,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();A=R/'artifacts/crow-stage3-adapter-training-20261109';B=R/'build/crow-stage3-adapter-training-20261109';lib=R/'build/crow-hierarchical-goal-memory-20261109/libmetalrobo.dylib'
route=read_policy_pack(R/'build/crow-stage3-adapter-training-20261109/frozen-route-actor.policypack',library_path=str(lib))
expert_path=R/('build/crow-stage3-adapter-training-20261109/frozen-route-actor.policypack' if os.environ.get('CROW_EXPERT_IDENTITY')=='1' else 'build/crow-route-no-vision-diagnostic-20261109/frozen-route-actor.policypack')
expert=read_policy_pack(expert_path,library_path=str(lib))
for k in ['contract_version','world_fingerprint','task_fingerprint','observation_fingerprint','action_fingerprint','observation_clip','action_clip']:
 assert getattr(route,k)==getattr(expert,k),k
for k in ['observation_mean','observation_inverse_standard_deviation','action_bias','action_scale']:
 assert np.array_equal(getattr(route,k),getattr(expert,k)),k
assert len(route.layers)==len(expert.layers)==4
layers=[]
u,v=route.layers[0],expert.layers[0]
layers.append(PolicyDenseLayerArtifact(weights=np.concatenate([u.weights,v.weights],axis=0),bias=np.concatenate([u.bias,v.bias]),activation=u.activation))
for u,v in zip(route.layers[1:-1],expert.layers[1:-1]):
 assert u.activation==v.activation and u.weights.shape==v.weights.shape
 rows,cols=u.weights.shape;w=np.zeros((2*rows,2*cols),dtype=np.float32);w[:rows,:cols]=u.weights;w[rows:,cols:]=v.weights
 layers.append(PolicyDenseLayerArtifact(weights=w,bias=np.concatenate([u.bias,v.bias]),activation=u.activation))
u,v=route.layers[-1],expert.layers[-1];assert u.weights.shape==v.weights.shape and u.weights.shape[0]==18
out,cols=u.weights.shape;w=np.zeros((out,2*cols),dtype=np.float32);w[:,:cols]=u.weights;bias=u.bias.copy()
w[4:8,:cols]=0;w[4:8,cols:]=v.weights[:4];bias[4:8]=v.bias[:4]
layers.append(PolicyDenseLayerArtifact(weights=w,bias=bias,activation=u.activation))
identity=os.environ.get('CROW_EXPERT_IDENTITY')=='1';path=B/('combined-identity.policypack' if identity else 'combined-expert.policypack')
write_policy_pack(path,policy_id='crow_stage2_combined',revision=max(route.revision,expert.revision)+1,contract_version=route.contract_version,world_fingerprint=route.world_fingerprint,task_fingerprint=route.task_fingerprint,observation_fingerprint=route.observation_fingerprint,action_fingerprint=route.action_fingerprint,layers=layers,observation_mean=route.observation_mean,observation_inverse_standard_deviation=route.observation_inverse_standard_deviation,action_bias=route.action_bias,action_scale=route.action_scale,observation_clip=route.observation_clip,action_clip=route.action_clip,library_path=str(lib))
p=read_policy_pack(path,library_path=str(lib));assert not p.critic_layers and not len(p.action_log_standard_deviation)
rng=np.random.default_rng(20270222);x=rng.normal(size=(64,816)).astype(np.float32)
a=route.actor_mean(x);b=expert.actor_mean(x);c=p.actor_mean(x)
error_route=float(np.max(np.abs(a[:,:4]-c[:,:4])));error_expert=float(np.max(np.abs(b[:,:4]-c[:,4:8])))
assert error_route<2e-5 and error_expert<2e-5,(error_route,error_expert)
if not identity:
 with path.open('rb') as f,(A/'combined-expert.policypack.gz').open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:shutil.copyfileobj(f,z)
 assert hashlib.sha256(gzip.decompress((A/'combined-expert.policypack.gz').read_bytes())).hexdigest()==hashlib.sha256(path.read_bytes()).hexdigest()
(A/('combine-identity.json' if identity else 'combine-expert.json')).write_text(json.dumps({'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'route_max_abs_error':error_route,'expert_max_abs_error':error_expert,'layer_shapes':[list(x.weights.shape) for x in p.layers],'expert_path':str(expert_path),'semantics':'experimental specialized waypoint-two output head'},indent=2)+'\n')
print('combined errors',error_route,error_expert,flush=True)
