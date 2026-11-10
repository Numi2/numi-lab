from pathlib import Path
import gzip,hashlib,json,time,shutil
from functools import partial
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from metalrobo.mlx_policy_learning import MLXPolicyLearner,MLXPPOConfiguration,read_policy_rollout_pack
R=Path.cwd();A=R/'artifacts/crow-v15-stage-aware-learning-20261110';B=R/'build/crow-v15-stage-aware-learning-20261110';H=R/'build/crow-v15-long-perch-host-20261110';lib=H/'libmetalrobo.dylib';initial=B/'trainable-initial.policypack'
sources=[('initial_j','crow-v15-student-recovery-20261110','development-j-student-recovery',1100),('late_j','crow-v15-late-student-recovery-20261110','development-j-late-student-recovery',2050),('i','crow-v15-i-student-recovery-20261110','development-i-student-recovery',1800),('corrective_i','crow-v15-corrective-i-recovery-20261110','development-i-corrective-recovery',1700),('anchored_j','crow-v15-anchored-j-recovery-20261110','development-j-anchored-recovery',2000),('anchored_i','crow-v15-anchored-i-recovery-20261110','development-i-anchored-recovery',2100),('approach_early_j','crow-v15-approach-j-early-recovery-20261110','development-j-approach-early-recovery',2100),('approach_early_i','crow-v15-approach-i-early-recovery-20261110','development-i-approach-early-recovery',2200),('foot_i','crow-v15-foot-placement-probe-20261110','development-i-foot-placement',2200),('placement_j','crow-v15-placement-j-recovery-20261110','development-j-placement-recovery',2100),('newactor_j','crow-v15-newactor-j-recovery-20261110','development-j-newactor-recovery',2200),('newactor_i','crow-v15-newactor-i-recovery-20261110','development-i-newactor-recovery',2100)]
chunks=[];source_info=[];temps=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for label,folder,name,takeover in sources:
 D=R/'artifacts'/folder;qual=json.loads((D/'qualification.json').read_text())
 if not qual['eligible_for_development_routing_prefix_training']:
  source_info.append({'source':label,'included':False,'reason':'failed qualification'});continue
 arc=D/(name+'.rollout.gz');raw=B/(name+'.training-tmp.rollout')
 with gzip.open(arc,'rb') as inp,raw.open('wb') as out:shutil.copyfileobj(inp,out)
 temps.append(raw);r=read_policy_rollout_pack(raw,library_path=str(lib));assert r.control_step_count==7000 and r.environment_count==1 and r.action_count==18
 obs=r.actor_observations.reshape(7000,1,r.actor_observation_count)[:,0,:];labels=r.teacher_actions.reshape(7000,1,18)[:,0,:];stage=r.outcome('navigation_waypoints_reached').reshape(7000,1)[:,0].astype(np.int32);end=json.loads((D/'analysis.json').read_text())['nav5_step'];steps=np.arange(1,end-1,5)
 assert np.all((stage[steps]>=0)&(stage[steps]<=4));x=obs[steps].copy();raw_labels=labels[steps].copy();y=raw_labels[:,:4].copy();ss=stage[steps];
 # Teacher writes the first four labels and duplicates them into the exact
 # active expert block. Decode that block from the rollout itself, avoiding
 # ambiguity at accepted waypoint-transition boundaries.
 matches=np.stack([np.all(np.isclose(raw_labels[:,4*k:4*k+4],y,rtol=0,atol=1e-7),axis=1) for k in (1,2,3)],axis=1)
 assert np.all(np.max(np.abs(y),axis=1)>1e-6) and np.all(matches.sum(axis=1)<=1), label
 groups=np.where(matches.any(axis=1),matches.argmax(axis=1)+1,0).astype(np.int32)
 expected=np.clip(ss-1,0,3); mismatch=int(np.sum(groups!=expected));factor=np.where((steps>=takeover-400)&(steps<takeover),8.,np.where(steps<takeover,4.,1.)).astype(np.float32);val=np.arange(len(steps))%5==0;assert np.isfinite(x).all() and np.isfinite(y).all() and np.max(np.abs(y))<=1.00001
 chunks.append((x,y,ss,factor,val,groups));source_info.append({'source':label,'included':True,'rollout_sha256':sha(raw),'samples':len(steps),'end_before_waypoint_5_step':end,'student_takeover':takeover,'label_stage_mismatches':mismatch,'expert_group_counts':np.bincount(groups,minlength=4).tolist()})
assert len(chunks)>=2
x=np.concatenate([z[0] for z in chunks]);y=np.concatenate([z[1] for z in chunks]);s=np.concatenate([z[2] for z in chunks]);factor=np.concatenate([z[3] for z in chunks]);val=np.concatenate([z[4] for z in chunks]);groups=np.concatenate([z[5] for z in chunks]);groupmask=np.eye(4,dtype=np.float32)[groups];tr=~val;tot=np.bincount(s[tr],weights=factor[tr],minlength=5);assert np.all(tot>0);weights=(factor*sum(tr)/(5*tot[s])).astype(np.float32)
learner=MLXPolicyLearner.from_policy_pack(initial,MLXPPOConfiguration(learning_rate=1e-4,seed=20261111),library_path=str(lib));model=learner.model
def infer(a):
 out=[]
 for i in range(0,len(a),1024):
  z=model.actor_mean(mx.array(a[i:i+1024],dtype=mx.float32));mx.eval(z);out.append(np.asarray(z))
 return np.concatenate(out)
reference=infer(x);opt=optim.Adam(learning_rate=1e-4,bias_correction=True);opt.init(model.trainable_parameters())
def loss_fn(model,x,y,w,head,anchor,mask,group):
 pred=model.actor_mean(x);selected=mx.sum(mx.reshape(pred[:,:16],(-1,4,4))*mx.expand_dims(group,2),axis=1);err=mx.square(selected-y)*mx.array([1.,2.,.5,1.5],dtype=mx.float32)
 return mx.mean(mx.sum(err,axis=1)*w)+2*mx.mean(mx.square(pred[:,16:]-head))+2*mx.mean(mx.sum(mx.square(pred[:,:4]-anchor),axis=1)*mask)
lossgrad=nn.value_and_grad(model,loss_fn);state=[model.state,opt.state]
@partial(mx.compile,inputs=state,outputs=state)
def step(x,y,w,h,anchor,mask,group):
 loss,grad=lossgrad(model,x,y,w,h,anchor,mask,group);opt.update(model,grad);return loss
rng=np.random.default_rng(20261111);train_ids=np.flatnonzero(tr);history=[];start=time.monotonic()
for epoch in range(1,121):
 order=rng.permutation(train_ids);losses=[]
 for j in range(0,len(order),256):
  idx=order[j:j+256];loss=step(mx.array(x[idx]),mx.array(y[idx]),mx.array(weights[idx]),mx.array(reference[idx,16:]),mx.array(reference[idx,:4]),mx.array((s[idx]==0).astype(np.float32)),mx.array(groupmask[idx]));mx.eval(loss,model.state,opt.state);losses.append(float(loss))
 if epoch in (1,20,60,120):
  pred=infer(x[val]);history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'in_trajectory_validation_mae':np.mean(np.abs(pred[:,:16].reshape(-1,4,4)[np.arange(sum(val)),groups[val]]-y[val]),axis=0).tolist(),'head_mse':float(np.mean((pred[:,16:]-reference[val,16:])**2)),'seconds':round(time.monotonic()-start,2)});print(json.dumps(history[-1]),flush=True)
out=B/'corrective.policypack';learner.revision+=1;learner.write_policy_pack(out,policy_id='crow_v15_corrective_routing_development',library_path=str(lib))
with out.open('rb') as src,(A/'corrective.policypack.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz:shutil.copyfileobj(src,gz)
report={'sources':source_info,'initial_policy_sha256':sha(initial),'output_policy_sha256':sha(out),'seed':20261111,'train_samples':int(tr.sum()),'in_trajectory_validation_samples':int(val.sum()),'train_stage_counts':np.bincount(s[tr],minlength=5).tolist(),'initial_validation_mae':np.mean(np.abs(reference[val,:16].reshape(-1,4,4)[np.arange(sum(val)),groups[val]]-y[val]),axis=0).tolist(),'history':history,'limits':'Development trajectories and in-trajectory diagnostic only; no terminal landing samples or independent holdout.'}
(A/'training-results.json').write_text(json.dumps(report,indent=2)+'\n')
for p in temps:p.unlink()
