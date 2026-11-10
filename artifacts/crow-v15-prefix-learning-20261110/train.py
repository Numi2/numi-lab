from pathlib import Path
import gzip,hashlib,json,time
from functools import partial
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from metalrobo.mlx_policy_learning import MLXPolicyLearner,MLXPPOConfiguration,read_policy_rollout_pack
R=Path.cwd();A=R/'artifacts/crow-v15-prefix-learning-20261110';B=R/'build/crow-v15-prefix-learning-20261110';H=R/'build/crow-v15-long-perch-host-20261110';S=R/'artifacts/crow-v15-student-recovery-20261110'
qual=json.loads((S/'qualification.json').read_text());assert qual['eligible_for_development_routing_prefix_training']
roll=H/'development-j-student-recovery.rollout';lib=H/'libmetalrobo.dylib';initial=B/'trainable-initial.policypack';r=read_policy_rollout_pack(roll,library_path=str(lib));assert r.control_step_count==7000 and r.environment_count==1 and r.action_count==18
obs=r.actor_observations.reshape(7000,1,r.actor_observation_count)[:,0,:];labels=r.teacher_actions.reshape(7000,1,18)[:,0,:4];stage=r.outcome('navigation_waypoints_reached').reshape(7000,1)[:,0].astype(np.int32)
end=json.loads((S/'analysis.json').read_text())['nav5_step'];steps=np.arange(1,end-1,5);assert np.all((stage[steps]>=0)&(stage[steps]<=4));x=obs[steps].copy();y=labels[steps].copy();s=stage[steps];factor=np.where(steps<1100,4.,1.).astype(np.float32);assert np.isfinite(x).all() and np.isfinite(y).all() and np.max(np.abs(y))<=1.00001
val=np.arange(len(steps))%5==0;tr=~val;tot=np.bincount(s[tr],weights=factor[tr],minlength=5);assert np.all(tot>0);weights=(factor*sum(tr)/(5*tot[s])).astype(np.float32)
learner=MLXPolicyLearner.from_policy_pack(initial,MLXPPOConfiguration(learning_rate=1e-4,seed=20261110),library_path=str(lib));model=learner.model
def infer(a):
 out=[]
 for i in range(0,len(a),1024):
  z=model.actor_mean(mx.array(a[i:i+1024],dtype=mx.float32));mx.eval(z);out.append(np.asarray(z))
 return np.concatenate(out)
reference=infer(x);opt=optim.Adam(learning_rate=1e-4,bias_correction=True);opt.init(model.trainable_parameters())
def loss_fn(model,x,y,w,head):
 pred=model.actor_mean(x);err=mx.square(pred[:,:4]-y)*mx.array([1.,2.,.5,1.5],dtype=mx.float32)
 return mx.mean(mx.sum(err,axis=1)*w)+2*mx.mean(mx.square(pred[:,15:]-head))
lossgrad=nn.value_and_grad(model,loss_fn);state=[model.state,opt.state]
@partial(mx.compile,inputs=state,outputs=state)
def step(x,y,w,h):
 loss,grad=lossgrad(model,x,y,w,h);opt.update(model,grad);return loss
rng=np.random.default_rng(20261110);train_ids=np.flatnonzero(tr);history=[];start=time.monotonic()
for epoch in range(1,121):
 order=rng.permutation(train_ids);losses=[]
 for j in range(0,len(order),256):
  idx=order[j:j+256];loss=step(mx.array(x[idx]),mx.array(y[idx]),mx.array(weights[idx]),mx.array(reference[idx,15:]));mx.eval(loss,model.state,opt.state);losses.append(float(loss))
 if epoch in (1,20,60,120):
  pred=infer(x[val]);history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'in_trajectory_validation_mae':np.mean(np.abs(pred[:,:4]-y[val]),axis=0).tolist(),'head_mse':float(np.mean((pred[:,15:]-reference[val,15:])**2)),'seconds':round(time.monotonic()-start,2)});print(json.dumps(history[-1]),flush=True)
out=B/'route-prefix.policypack';learner.revision+=1;learner.write_policy_pack(out,policy_id='crow_v15_development_prefix_actor_v1',library_path=str(lib));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with out.open('rb') as src,(A/'route-prefix.policypack.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz:gz.write(src.read())
report={'rollout_sha256':sha(roll),'initial_policy_sha256':sha(initial),'output_policy_sha256':sha(out),'seed':20261110,'end_before_waypoint_5_step':end,'train_samples':int(tr.sum()),'in_trajectory_validation_samples':int(val.sum()),'train_stage_counts':np.bincount(s[tr],minlength=5).tolist(),'initial_validation_mae':np.mean(np.abs(reference[val,:4]-y[val]),axis=0).tolist(),'history':history,'limits':'Single used layout and student-prefix recovery; diagnostic split shares trajectory and is not held out. No landing samples or independent validation.'}
(A/'training-results.json').write_text(json.dumps(report,indent=2)+'\n')
