from pathlib import Path
import gzip,hashlib,json,shutil,time
from functools import partial
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from metalrobo.mlx_policy_learning import MLXPolicyLearner,MLXPPOConfiguration,read_policy_rollout_pack

R=Path.cwd()
A=R/'artifacts/crow-terminal-specialist-training-20261109'
B=R/'build/crow-terminal-specialist-training-20261109'
B.mkdir(parents=True,exist_ok=True)
LIB=R/'build/crow-hierarchical-goal-memory-20261109/libmetalrobo.dylib'
initial=R/'build/crow-deterministic-terminal-training-20261109/recovery-imitation.policypack'
courses=['training','held-out-a','held-out-b']
data={'train':[],'validation':[]}
hashes={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
for kind in ('demonstrations','terminal-dagger','deterministic-terminal-dagger','full-student-dagger'):
 for course in courses:
  root=R/('artifacts/crow-'+(kind if kind in ('deterministic-terminal-dagger','full-student-dagger') else 'goal-memory-'+kind)+'-20261109')
  manifest=json.loads((root/course/'manifest.json').read_text())
  pack=R/('build/crow-'+(kind if kind in ('deterministic-terminal-dagger','full-student-dagger') else 'goal-memory-'+kind)+'-20261109')/(course+'.rollout')
  digest=sha(pack)
  assert digest==manifest['raw_pack_sha256'] and manifest['round_trip_verified']
  r=read_policy_rollout_pack(pack,library_path=str(LIB))
  n=8 if kind in ('demonstrations','deterministic-terminal-dagger','full-student-dagger') else 4
  assert r.control_step_count==3000 and r.environment_count==n and r.action_count==18
  obs=r.actor_observations.reshape(3000,n,r.actor_observation_count)
  labels=r.teacher_actions.reshape(3000,n,18)
  stage=r.outcome('navigation_waypoints_reached').reshape(3000,n)
  valid=sorted(x['lane'] for x in manifest['lanes'] if x['eligible'])
  if kind=='demonstrations':
   groups={'train':[x for x in valid if x<6],'validation':[x for x in valid if x>=6]}
  else:
   groups={'train':valid[:-1],'validation':valid[-1:]}
  steps=np.arange(1,3000,5)
  for part,lanes in groups.items():
   for lane in lanes:
    xs=obs[steps,lane,:].copy()
    ys=labels[steps,lane,:4].copy()
    ss=stage[steps,lane].astype(np.int32).copy()
    chosen=ss>=5
    xs=xs[chosen];ys=ys[chosen];ss=ss[chosen];selected_steps=steps[chosen]
    if not len(ss):continue
    factor=np.ones(len(selected_steps),dtype=np.float32)
    if kind=='terminal-dagger':factor[selected_steps<=manifest['student_takeover_steps'][lane]]=8
    if kind=='deterministic-terminal-dagger':factor[selected_steps<=manifest['student_takeover_steps'][lane]]=12
    if kind=='full-student-dagger':factor[selected_steps<=manifest['student_takeover_steps'][lane]]=16
    assert np.isfinite(xs).all() and np.isfinite(ys).all() and np.max(np.abs(ys))<=1.00001
    assert np.all((ss>=0)&(ss<=5))
    data[part].append((xs,ys,ss,factor,kind,course,lane))
  hashes[kind+'/'+course]=digest
def join(part):
 chunks=data[part]
 x=np.concatenate([a[0] for a in chunks])
 y=np.concatenate([a[1] for a in chunks])
 stage=np.concatenate([a[2] for a in chunks])
 factor=np.concatenate([a[3] for a in chunks])
 totals=np.bincount(stage,weights=factor,minlength=6)
 assert totals[5]>0 and np.all(stage==5)
 weights=(factor*len(stage)/totals[5]).astype(np.float32)
 return x,y,stage,weights,np.bincount(stage,minlength=6)
train_x,train_y,train_stage,train_w,train_counts=join('train')
val_x,val_y,val_stage,val_w,val_counts=join('validation')
config=MLXPPOConfiguration(learning_rate=1e-4,seed=20270124)
learner=MLXPolicyLearner.from_policy_pack(initial,config,library_path=str(LIB))
model=learner.model
def baseline_heads(arr):
 out=[]
 for i in range(0,len(arr),2048):
  y=model.actor_mean(mx.array(arr[i:i+2048],dtype=mx.float32))[:,15:]
  mx.eval(y);out.append(np.asarray(y))
 return np.concatenate(out)
train_head=baseline_heads(train_x);val_head=baseline_heads(val_x)
optimizer=optim.Adam(learning_rate=1e-4,bias_correction=True)
optimizer.init(model.trainable_parameters())
def loss_fn(model,x,y,w,reference):
 pred=model.actor_mean(x)
 err=mx.square(pred[:,:4]-y)*mx.array([1.,2.,.5,1.5],dtype=mx.float32)
 return mx.mean(mx.sum(err,axis=1)*w)+2*mx.mean(mx.square(pred[:,15:]-reference))
lossgrad=nn.value_and_grad(model,loss_fn)
state=[model.state,optimizer.state]
@partial(mx.compile,inputs=state,outputs=state)
def step(x,y,w,h):
 loss,grad=lossgrad(model,x,y,w,h)
 optimizer.update(model,grad)
 return loss
rng=np.random.default_rng(20270124);history=[];start=time.monotonic()
for epoch in range(1,301):
 order=rng.permutation(len(train_x));losses=[]
 for offset in range(0,len(order),1024):
  idx=order[offset:offset+1024]
  loss=step(mx.array(train_x[idx]),mx.array(train_y[idx]),mx.array(train_w[idx]),mx.array(train_head[idx]))
  mx.eval(loss,model.state,optimizer.state);losses.append(float(loss))
 if epoch in (1,20,40,80,120,160,200,250,300):
  pred=[]
  for i in range(0,len(val_x),2048):
   z=model.actor_mean(mx.array(val_x[i:i+2048]));mx.eval(z);pred.append(np.asarray(z))
  pred=np.concatenate(pred)
  row={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_command_mae':np.mean(np.abs(pred[:,:4]-val_y),axis=0).tolist(),'validation_head_mse':float(np.mean((pred[:,15:]-val_head)**2)),'seconds':round(time.monotonic()-start,2)}
  history.append(row);print(json.dumps(row),flush=True)
output=B/'recovery-imitation.policypack'
learner.revision+=1
learner.write_policy_pack(output,policy_id='crow_v15_terminal_specialist_v1',library_path=str(LIB))
compressed=A/'recovery-imitation.policypack.gz'
with output.open('rb') as f,compressed.open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:shutil.copyfileobj(f,z)
assert hashlib.sha256(gzip.decompress(compressed.read_bytes())).hexdigest()==sha(output)
report={'input_rollout_sha256':hashes,'initial_policy_sha256':sha(initial),'output_policy_sha256':sha(output),'seed':20270124,'stride':5,'training_samples':len(train_x),'validation_samples':len(val_x),'training_stage_counts':train_counts.tolist(),'validation_stage_counts':val_counts.tolist(),'epochs':300,'learning_rate':1e-4,'history':history,'training_lanes':[[x[4],x[5],x[6]] for x in data['train']],'validation_lanes':[[x[4],x[5],x[6]] for x in data['validation']],'note':'supervised teacher and successful recovery imitation only; no autonomous success claim'}
(A/'training-results.json').write_text(json.dumps(report,indent=2)+'\n')
