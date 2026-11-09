from pathlib import Path
import hashlib,json,os,time
from functools import partial
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from metalrobo.mlx_policy_learning import MLXPolicyLearner,MLXPPOConfiguration,read_policy_rollout_pack
R=Path.cwd();A=R/'artifacts/crow-goal-memory-imitation-20261109';B=R/'build/crow-goal-memory-imitation-20261109';B.mkdir(parents=True,exist_ok=True)
LIB=R/'build/crow-hierarchical-goal-memory-20261109/libmetalrobo.dylib'; initial=R/'build/crow-hierarchical-goal-memory-20261109/hierarchical-initial.policypack'
courses=['training','held-out-a','held-out-b']; data={'train':[],'validation':[]};input_hashes={}
for course in courses:
 manifest=json.loads((R/'artifacts/crow-goal-memory-demonstrations-20261109'/course/'manifest.json').read_text())
 pack=R/'build/crow-goal-memory-demonstrations-20261109'/(course+'.rollout')
 digest=hashlib.sha256(pack.read_bytes()).hexdigest()
 assert digest==manifest['raw_pack_sha256'] and manifest['round_trip_verified']
 r=read_policy_rollout_pack(pack,library_path=str(LIB))
 assert r.control_step_count==3000 and r.environment_count==8 and r.action_count==18
 assert not np.any(r.transitions['physics_error'])
 obs=r.actor_observations.reshape(3000,8,r.actor_observation_count)
 labels=r.teacher_actions.reshape(3000,8,18)
 stage=r.outcome('navigation_waypoints_reached').reshape(3000,8)
 valid={x['lane'] for x in manifest['lanes'] if x['eligible']}
 for part, lanes in [('train',list(range(6))),('validation',[6,7])]:
  for lane in lanes:
   if lane not in valid:continue
   xs=obs[1::5,lane,:].copy();ys=labels[1::5,lane,:4].copy();ss=stage[1::5,lane].astype(np.int32).copy()
   assert np.isfinite(xs).all() and np.isfinite(ys).all() and np.max(np.abs(ys))<=1.00001
   data[part].append((xs,ys,ss,course,lane))
 input_hashes[course]=digest
def join(part):
 chunks=data[part]
 assert chunks
 x=np.concatenate([a[0] for a in chunks]); y=np.concatenate([a[1] for a in chunks]); stage=np.concatenate([a[2] for a in chunks])
 counts=np.bincount(stage,minlength=6);assert all(counts>0)
 weights=(len(stage)/(6*counts[stage])).astype(np.float32)
 return x,y,stage,weights,counts
train_x,train_y,train_stage,train_w,train_counts=join('train')
val_x,val_y,val_stage,val_w,val_counts=join('validation')
config=MLXPPOConfiguration(learning_rate=1e-4,seed=20261220)
learner=MLXPolicyLearner.from_policy_pack(initial,config,library_path=str(LIB))
model=learner.model
def baseline_heads(arr):
 out=[]
 for i in range(0,len(arr),2048):
  y=model.actor_mean(mx.array(arr[i:i+2048],dtype=mx.float32))[:,15:]
  mx.eval(y);out.append(np.asarray(y))
 return np.concatenate(out)
train_head=baseline_heads(train_x);val_head=baseline_heads(val_x)
optimizer=optim.Adam(learning_rate=1e-4,bias_correction=True);optimizer.init(model.trainable_parameters())
def loss_fn(model,x,y,w,reference):
 pred=model.actor_mean(x)
 err=mx.square(pred[:,:4]-y)*mx.array([1.0,2.0,0.5,1.5],dtype=mx.float32)
 command=mx.mean(mx.sum(err,axis=1)*w)
 retention=mx.mean(mx.square(pred[:,15:]-reference))
 return command+2.0*retention
lossgrad=nn.value_and_grad(model,loss_fn)
state=[model.state,optimizer.state]
@partial(mx.compile,inputs=state,outputs=state)
def step(x,y,w,h):
 loss,grad=lossgrad(model,x,y,w,h)
 optimizer.update(model,grad)
 return loss
rng=np.random.default_rng(20261220);history=[];start=time.monotonic()
for epoch in range(1,121):
 order=rng.permutation(len(train_x));losses=[]
 for offset in range(0,len(order),1024):
  idx=order[offset:offset+1024]
  loss=step(mx.array(train_x[idx]),mx.array(train_y[idx]),mx.array(train_w[idx]),mx.array(train_head[idx]))
  mx.eval(loss,model.state,optimizer.state);losses.append(float(loss))
 if epoch in (1,10,20,40,60,80,100,120):
  pred=[]
  for i in range(0,len(val_x),2048):
   z=model.actor_mean(mx.array(val_x[i:i+2048]));mx.eval(z);pred.append(np.asarray(z))
  pred=np.concatenate(pred)
  row={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_command_mae':np.mean(np.abs(pred[:,:4]-val_y),axis=0).tolist(),'validation_head_mse':float(np.mean((pred[:,15:]-val_head)**2)),'seconds':round(time.monotonic()-start,2)}
  history.append(row);print(json.dumps(row),flush=True)
output=B/'goal-memory-imitation.policypack'
learner.revision+=1
learner.write_policy_pack(output,policy_id='crow_v15_goal_memory_imitation_v1',library_path=str(LIB))
report={'input_rollout_sha256':input_hashes,'initial_policy_sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),'output_policy_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'seed':20261220,'stride':5,'training_samples':len(train_x),'validation_samples':len(val_x),'training_stage_counts':train_counts.tolist(),'validation_stage_counts':val_counts.tolist(),'teacher_action_indices':[0,1,2,3],'epochs':120,'learning_rate':1e-4,'history':history,'training_lanes':[[x[3],x[4]] for x in data['train']],'validation_lanes':[[x[3],x[4]] for x in data['validation']],'note':'supervised teacher imitation only; no autonomous success claim'}
(A/'training-results.json').write_text(json.dumps(report,indent=2)+'\n')

import gzip,shutil
with output.open('rb') as f, (A/'goal-memory-imitation.policypack.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:shutil.copyfileobj(f,z)
assert hashlib.sha256(gzip.decompress((A/'goal-memory-imitation.policypack.gz').read_bytes())).hexdigest()==hashlib.sha256(output.read_bytes()).hexdigest()
