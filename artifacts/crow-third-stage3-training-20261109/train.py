from pathlib import Path
import gzip,hashlib,json,shutil,time
from functools import partial
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from metalrobo.mlx_policy_learning import MLXPolicyLearner,MLXPPOConfiguration,read_policy_rollout_pack,_actor_observation_extension_only_gradients
R=Path.cwd();A=R/'artifacts/crow-third-stage3-training-20261109';B=R/'build/crow-third-stage3-training-20261109'
LIB=R/'build/crow-hierarchical-goal-memory-20261109/libmetalrobo.dylib'
initial=R/'build/crow-expanded-stage3-specialist-20261109/checkpoint-160.policypack'
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
   active=np.isclose(obs[steps,lane,169],0.6,atol=1e-6)
   outside=~active
   assert np.all(obs[steps[outside],lane,163:185]==0),('gated indices differ',kind,name,lane)
   zero_checks+=int(np.sum(outside))
   chosen=active; selected=steps[chosen]
   if not len(selected):continue
   xs=obs[selected,lane,:].copy();ys=labels[selected,lane,:4].copy()
   factor=np.ones(len(selected),dtype=np.float32)
   if kind=='deterministic-dagger':factor[selected<=manifest['student_takeover_steps'][lane]]=8
   if kind=='gate-recovery-dagger':factor[selected<=manifest['student_takeover_steps'][lane]]=12
   if kind in ('mixed-stage3','expanded-stage3','expanded-gate'):factor[selected<=manifest['student_takeover_steps'][lane]]=5
   if kind.startswith('third-'):factor[selected<=manifest['student_takeover_steps'][lane]]=8
   assert np.isfinite(xs).all() and np.isfinite(ys).all() and np.max(np.abs(ys))<=1.00001
   data[part].append((xs,ys,factor,kind,name,lane))
 hashes[kind+'/'+name]=digest
for kind in ['demonstrations','deterministic-dagger','gate-recovery-dagger']:
 for course in ['training','held-out-a','held-out-b']:
  prefix=kind if kind!='demonstrations' else 'goal-memory-demonstrations'
  consume(kind,course,R/f'artifacts/crow-{prefix}-20261109'/course,R/f'build/crow-{prefix}-20261109'/(course+'.rollout'),8)
for batch in ['training','held-out-a','held-out-b']:
 consume('mixed-stage3',batch,R/'artifacts/crow-mixed-stage3-dagger-20261109'/batch,R/'build/crow-mixed-stage3-dagger-20261109'/(batch+'.rollout'),8)
for kind,directory in [('expanded-stage3','crow-expanded-dagger-20261109'),('expanded-gate','crow-expanded-gate-dagger-20261109')]:
 for batch in ['layout-c','layout-d']:
  consume(kind,batch,R/'artifacts'/directory/batch,R/'build'/directory/(batch+'.rollout'),16)
for schedule in ['early','late','terminal']:
 for batch in ['layout-e','layout-f']:
  consume('third-'+schedule,batch,R/'artifacts/crow-third-dagger-20261109'/schedule/batch,R/'build/crow-third-dagger-20261109'/schedule/(batch+'.rollout'),16)
def join(part):
 c=data[part];assert c
 x=np.concatenate([z[0] for z in c]);y=np.concatenate([z[1] for z in c]);f=np.concatenate([z[2] for z in c])
 return x,y,(f*len(f)/sum(f)).astype(np.float32)
train_x,train_y,train_w=join('train');val_x,val_y,val_w=join('validation')
config=MLXPPOConfiguration(learning_rate=1e-4,seed=20270304)
learner=MLXPolicyLearner.from_policy_pack(initial,config,library_path=str(LIB));model=learner.model
before=[(np.asarray(l.weight).copy(),np.asarray(l.bias).copy()) for l in model.actor.layers if hasattr(l,'weight')]
optimizer=optim.Adam(learning_rate=1e-4,bias_correction=True);optimizer.init(model.trainable_parameters())
def loss_fn(model,x,y,w):
 pred=model.actor_mean(x)
 return mx.mean(mx.sum(mx.square(pred[:,:4]-y)*mx.array([1.,2.,.5,1.5],dtype=mx.float32),axis=1)*w)
lossgrad=nn.value_and_grad(model,loss_fn);state=[model.state,optimizer.state]
@partial(mx.compile,inputs=state,outputs=state)
def step(x,y,w):
 loss,grad=lossgrad(model,x,y,w);
 optimizer.update(model,grad)
 return loss
rng=np.random.default_rng(20270304);history=[];start=time.monotonic()
for epoch in range(1,161):
 order=rng.permutation(len(train_x));losses=[]
 for off in range(0,len(order),1024):
  ix=order[off:off+1024];loss=step(mx.array(train_x[ix]),mx.array(train_y[ix]),mx.array(train_w[ix]))
  mx.eval(loss,model.state,optimizer.state);losses.append(float(loss))
 if epoch in (1,20,40,80,120,160):
  pred=[]
  for i in range(0,len(val_x),2048):
   z=model.actor_mean(mx.array(val_x[i:i+2048]));mx.eval(z);pred.append(np.asarray(z))
  row={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_command_mae':np.mean(np.abs(np.concatenate(pred)[:,:4]-val_y),axis=0).tolist(),'seconds':round(time.monotonic()-start,2)}
  history.append(row);print(json.dumps(row),flush=True)
  if epoch in (20,40,80,160):
   learner.write_policy_pack(B/f'checkpoint-{epoch}.policypack',policy_id=f'crow_v15_expanded_stage3_{epoch}',library_path=str(LIB))
output=B/'recovery-imitation.policypack';learner.revision+=1
learner.write_policy_pack(output,policy_id='crow_v15_stage3_adapter',library_path=str(LIB))
with output.open('rb') as f,(A/'recovery-imitation.policypack.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:shutil.copyfileobj(f,z)
assert hashlib.sha256(gzip.decompress((A/'recovery-imitation.policypack.gz').read_bytes())).hexdigest()==sha(output)
report={'input_rollout_sha256':hashes,'initial_policy_sha256':sha(initial),'output_policy_sha256':sha(output),'seed':20270304,'training_samples':len(train_x),'validation_samples':len(val_x),'protected_offstage_zero_samples':zero_checks,'full_actor_specialist':True,'history':history,'training_lanes':[[x[3],x[4],x[5]] for x in data['train']],'validation_lanes':[[x[3],x[4],x[5]] for x in data['validation']]}
(A/'training-results.json').write_text(json.dumps(report,indent=2)+'\n')
