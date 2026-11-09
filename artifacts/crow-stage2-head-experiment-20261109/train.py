from pathlib import Path
import gzip,hashlib,json,shutil,time
from functools import partial
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from metalrobo.mlx_policy_learning import MLXPolicyLearner,MLXPPOConfiguration,read_policy_rollout_pack,_actor_observation_extension_only_gradients
R=Path.cwd();A=R/'artifacts/crow-stage2-head-experiment-20261109';B=R/'build/crow-stage2-head-experiment-20261109'
LIB=R/'build/crow-hierarchical-goal-memory-20261109/libmetalrobo.dylib'
initial=B/'trainable-initial.policypack'
data={'train':[],'validation':[]};hashes={};zero_checks=0
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def consume(kind,name,root,pack,n):
 global zero_checks
 manifest=json.loads((root/'manifest.json').read_text())
 digest=sha(pack);assert digest==manifest['raw_pack_sha256'] and manifest['round_trip_verified']
 r=read_policy_rollout_pack(pack,library_path=str(LIB))
 assert r.control_step_count==3000 and r.environment_count==n and r.action_count==18 and r.actor_observation_count==816
 obs=r.actor_observations.reshape(3000,n,816);labels=r.teacher_actions.reshape(3000,n,18);stage=r.outcome('navigation_waypoints_reached').reshape(3000,n)
 valid=sorted(x['lane'] for x in manifest['lanes'] if x['eligible'])
 if kind=='demonstrations':groups={'train':[x for x in valid if x<6],'validation':[x for x in valid if x>=6]}
 else:groups={'train':valid[:-1],'validation':valid[-1:]}
 steps=np.arange(1,3000,5)
 for part,lanes in groups.items():
  for lane in lanes:
   ss=stage[steps,lane].astype(np.int32)
   # Transition outcome is published after the actor observation: use the
   # observation's gated phase, not the post-step outcome, at boundaries.
   active=np.isclose(obs[steps,lane,147],0.4,atol=1e-6)
   outside=~active
   assert np.all(obs[steps[outside],lane,141:163]==0),('gated indices differ',kind,name,lane)
   zero_checks+=int(np.sum(outside))
   chosen=active; selected=steps[chosen]
   if not len(selected):continue
   xs=obs[selected,lane,:].copy();ys=labels[selected,lane,:4].copy()
   factor=np.ones(len(selected),dtype=np.float32)
   if kind=='deterministic-dagger':factor[selected<=manifest['student_takeover_steps'][lane]]=8
   if kind=='gate-recovery-dagger':factor[selected<=manifest['student_takeover_steps'][lane]]=12
   if kind=='production-gate':factor[selected<=manifest['student_takeover_steps'][lane]]=24
   assert np.isfinite(xs).all() and np.isfinite(ys).all() and np.max(np.abs(ys))<=1.00001
   data[part].append((xs,ys,factor,kind,name,lane))
 hashes[kind+'/'+name]=digest
for kind in ['demonstrations','deterministic-dagger','gate-recovery-dagger']:
 for course in ['training','held-out-a','held-out-b']:
  prefix=kind if kind!='demonstrations' else 'goal-memory-demonstrations'
  consume(kind,course,R/f'artifacts/crow-{prefix}-20261109'/course,R/f'build/crow-{prefix}-20261109'/(course+'.rollout'),8)
for batch in ['a1','a2','a3']:
 consume('production-gate',batch,R/'artifacts/crow-production-gate-dagger-20261109'/batch,R/'build/crow-production-gate-dagger-20261109'/(batch+'.rollout'),8)
def join(part):
 c=data[part];assert c
 x=np.concatenate([z[0] for z in c]);y=np.concatenate([z[1] for z in c]);f=np.concatenate([z[2] for z in c])
 return x,y,(f*len(f)/sum(f)).astype(np.float32)
train_x,train_y,train_w=join('train');val_x,val_y,val_w=join('validation')
config=MLXPPOConfiguration(learning_rate=1e-4,seed=20270218)
learner=MLXPolicyLearner.from_policy_pack(initial,config,library_path=str(LIB));model=learner.model
before=[(np.asarray(l.weight).copy(),np.asarray(l.bias).copy()) for l in model.actor.layers if hasattr(l,'weight')]
optimizer=optim.Adam(learning_rate=1e-4,bias_correction=True);optimizer.init(model.trainable_parameters())
def loss_fn(model,x,y,w):
 pred=model.actor_mean(x)
 return mx.mean(mx.sum(mx.square(pred[:,4:8]-y)*mx.array([1.,2.,.5,1.5],dtype=mx.float32),axis=1)*w)
lossgrad=nn.value_and_grad(model,loss_fn);state=[model.state,optimizer.state]
@partial(mx.compile,inputs=state,outputs=state)
def step(x,y,w):
 loss,grad=lossgrad(model,x,y,w);grad=_actor_observation_extension_only_gradients(grad,141,22,output_action_indices=(4,5,6,7))
 optimizer.update(model,grad)
 return loss
rng=np.random.default_rng(20270218);history=[];start=time.monotonic()
for epoch in range(1,301):
 order=rng.permutation(len(train_x));losses=[]
 for off in range(0,len(order),1024):
  ix=order[off:off+1024];loss=step(mx.array(train_x[ix]),mx.array(train_y[ix]),mx.array(train_w[ix]))
  mx.eval(loss,model.state,optimizer.state);losses.append(float(loss))
 if epoch in (1,20,40,80,120,160,200,250,300):
  pred=[]
  for i in range(0,len(val_x),2048):
   z=model.actor_mean(mx.array(val_x[i:i+2048]));mx.eval(z);pred.append(np.asarray(z))
  row={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_command_mae':np.mean(np.abs(np.concatenate(pred)[:,4:8]-val_y),axis=0).tolist(),'seconds':round(time.monotonic()-start,2)}
  history.append(row);print(json.dumps(row),flush=True)
after=[(np.asarray(l.weight).copy(),np.asarray(l.bias).copy()) for l in model.actor.layers if hasattr(l,'weight')]
assert len(before)==len(after)
for i,((w0,b0),(w1,b1)) in enumerate(zip(before,after)):
 if i==len(before)-1:
  assert np.array_equal(b0[:4],b1[:4]) and np.array_equal(b0[8:],b1[8:])
  assert np.array_equal(w0[:4],w1[:4]) and np.array_equal(w0[8:],w1[8:])
  assert np.any(w0[4:8]!=w1[4:8])
  continue
 assert np.array_equal(b0,b1)
 if i==0:
  assert np.array_equal(w0[:,:141],w1[:,:141]) and np.array_equal(w0[:,163:],w1[:,163:])
  assert np.any(w0[:,141:163]!=w1[:,141:163])
 else:assert np.array_equal(w0,w1)
output=B/'recovery-imitation.policypack';learner.revision+=1
learner.write_policy_pack(output,policy_id='crow_v15_stage2_head',library_path=str(LIB))
with output.open('rb') as f,(A/'recovery-imitation.policypack.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:shutil.copyfileobj(f,z)
assert hashlib.sha256(gzip.decompress((A/'recovery-imitation.policypack.gz').read_bytes())).hexdigest()==sha(output)
report={'input_rollout_sha256':hashes,'initial_policy_sha256':sha(initial),'output_policy_sha256':sha(output),'seed':20270218,'training_samples':len(train_x),'validation_samples':len(val_x),'protected_offstage_zero_samples':zero_checks,'only_first_layer_columns_141_to_162_and_output_rows_4_to_7_changed':True,'history':history,'training_lanes':[[x[3],x[4],x[5]] for x in data['train']],'validation_lanes':[[x[3],x[4],x[5]] for x in data['validation']]}
(A/'training-results.json').write_text(json.dumps(report,indent=2)+'\n')
