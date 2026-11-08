from pathlib import Path
import hashlib,json,sys
import numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack
R=Path.cwd();B=R/'build/crow-active-navigation-transfer-01';A=R/'artifacts/crow-active-navigation-transfer-20261108'
lib=R/'build/lib/libmetalrobo.dylib'
src_path=B/'source.policypack';new_contract=R/'build/crow-active-navigation-train-smoke-01/initial.policypack'
assert hashlib.sha256(src_path.read_bytes()).hexdigest()=='cc79a6d0a9c10f4964944323ea9b0f8311340d334eb297032c8a9529f68fc570'
src=read_policy_pack(src_path,library_path=lib); target=read_policy_pack(new_contract,library_path=lib)
assert (src.actor_observation_count,src.action_count)==(807,15)
assert (target.actor_observation_count,target.action_count)==(816,18)
offset=207
layers=[]
for i,l in enumerate(src.layers):
 w=np.array(l.weights,copy=True);b=np.array(l.bias,copy=True)
 if i==0:w=np.concatenate([w[:,:offset],np.zeros((w.shape[0],9),np.float32),w[:,offset:]],axis=1)
 if i==len(src.layers)-1:
  w=np.concatenate([w,np.zeros((3,w.shape[1]),np.float32)],axis=0);b=np.concatenate([b,np.zeros(3,np.float32)])
 layers.append(PolicyDenseLayerArtifact(weights=w,bias=b,activation=l.activation))
mean=np.concatenate([src.effective_observation_mean[:offset],np.zeros(9,np.float32),src.effective_observation_mean[offset:]])
inv=np.concatenate([src.effective_observation_inverse_standard_deviation[:offset],np.ones(9,np.float32),src.effective_observation_inverse_standard_deviation[offset:]])
bias=np.concatenate([src.effective_action_bias,np.zeros(3,np.float32)])
scale=np.concatenate([src.effective_action_scale,np.ones(3,np.float32)])
out=B/'transferred.policypack'
write_policy_pack(out,policy_id='crow_v15_navigation_transfer_diagnostic',revision=1,contract_version=target.contract_version,
 world_fingerprint=target.world_fingerprint,task_fingerprint=target.task_fingerprint,
 observation_fingerprint=target.observation_fingerprint,action_fingerprint=target.action_fingerprint,
 layers=layers,observation_mean=mean,observation_inverse_standard_deviation=inv,action_bias=bias,action_scale=scale,
 observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=lib)
# A separate stochastic pack carries a fresh v15 critic for an exploratory
# continuation; the deterministic screen artifact above is never overwritten.
assert len(target.action_log_standard_deviation)==18 and target.critic_observation_count==106
critic=[PolicyDenseLayerArtifact(weights=l.weights,bias=l.bias,activation=l.activation) for l in target.critic_layers]
trainable=B/'trainable.policypack'
write_policy_pack(trainable,policy_id='crow_v15_navigation_transfer_trainable',revision=1,contract_version=target.contract_version,
 world_fingerprint=target.world_fingerprint,task_fingerprint=target.task_fingerprint,
 observation_fingerprint=target.observation_fingerprint,action_fingerprint=target.action_fingerprint,
 layers=layers,critic_layers=critic,observation_mean=mean,observation_inverse_standard_deviation=inv,
 critic_observation_mean=target.effective_critic_observation_mean,
 critic_observation_inverse_standard_deviation=target.effective_critic_observation_inverse_standard_deviation,
 action_log_standard_deviation=target.action_log_standard_deviation,
 action_bias=bias,action_scale=scale,observation_clip=src.observation_clip,action_clip=src.action_clip,library_path=lib)
dst=read_policy_pack(out,library_path=lib)
def actor(p,x):
 x=np.clip((x-p.effective_observation_mean)*p.effective_observation_inverse_standard_deviation,-p.observation_clip,p.observation_clip)
 for l in p.layers:
  x=x@l.weights.T+l.bias
  if l.activation==3:x=np.where(x>0,x,np.expm1(np.minimum(x,0)))
 return x
rng=np.random.default_rng(20261108)
old=rng.normal(size=(16,807)).astype(np.float32)
new=np.concatenate([old[:,:offset],rng.normal(size=(16,9)).astype(np.float32),old[:,offset:]],axis=1)
a=actor(src,old);b=actor(dst,new)
delta=float(np.max(np.abs(a-b[:,:15])));head=float(np.max(np.abs(b[:,15:])))
assert delta<1e-4 and head==0,(delta,head)
result={'source_sha256':hashlib.sha256(src_path.read_bytes()).hexdigest(),
        'target_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source_actions':15,'target_actions':18,
        'source_observations':807,'target_observations':816,'insert_offset':offset,'new_inputs':9,'new_outputs':3,
        'cpu_max_original_logit_difference':delta,'cpu_max_new_action_logit':head,
        'scope':'synthetic numeric projection only; different embodiment, camera, physics and course behavior require independent measurement'}
(A/'transfer-result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result))
