from pathlib import Path
import hashlib,json,math,time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from metalrobo.mlx_policy_learning import MLXPolicyLearner,MLXPPOConfiguration,read_policy_rollout_pack,read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact,write_policy_pack

R=Path.cwd();A=R/"artifacts/crow-v15-localized-route-student-imitation-20261111";B=R/"build/crow-v15-localized-route-student-imitation-20261111"
P=R/"artifacts/crow-v15-route-prefix-demonstrations-20261111";qualification=json.loads((P/"qualification.json").read_text())
pack=R/"build/crow-v15-route-prefix-demonstrations-20261111/teacher-20261126-rollout.bin"
lib=R/"build/crow-v15-hip-roll-journey-integration-20261111/libmetalrobo.dylib"
initial=B/"trainable-initial.policypack"
r=read_policy_rollout_pack(pack,library_path=str(lib));n=r.environment_count;t=r.control_step_count
assert (n,t,r.action_count)==(8,3800,18)
obs=r.actor_observations.reshape(t,n,r.actor_observation_count)
labels=r.teacher_actions.reshape(t,n,18)
stages=r.outcome("navigation_waypoints_reached").reshape(t,n)
validation=set(qualification["validation_lanes"]);valid=set(qualification["eligible_lanes"])
assert validation and validation<valid
data={"train":[],"validation":[]}
for lane in sorted(valid):
 row=qualification["lanes"][lane]
 assert row["eligible_route_prefix"] and not row["physical_failure"]
 end=int(row["first_fifth_waypoint_step"])
 times=np.arange(400,end,5)
 stage=stages[times,lane].astype(np.int32)
 use=(stage>=0)&(stage<=4);times=times[use];stage=stage[use]
 offsets=np.where(stage>=4,12,np.where(stage==3,8,np.where(stage==2,4,0))).astype(np.int32)
 indices=offsets[:,None]+np.arange(4,dtype=np.int32)[None,:]
 x=obs[times,lane].copy();y=labels[times,lane,:4].copy()
 assert np.isfinite(x).all() and np.isfinite(y).all() and np.max(np.abs(y))<=1.00001
 data["validation" if lane in validation else "train"].append((x,y,indices,stage,lane))
# Add only approaches that later had supported opposing grip, ending before the first foot impulse.
perch=json.loads((R/'artifacts/crow-v15-teacher-perch-balanced-approach-control-20261111/teacher-20261126/perch-analysis.json').read_text())
selected={row['lane']:row for row in perch['rows'] if row['lane'] in (0,4,6)}
assert set(selected)=={0,4,6} and all(x['last_25_half_weight_bilateral_opposed'] and not x['physical_failure'] for x in selected.values())
for lane,row in sorted(selected.items()):
 start=int(qualification['lanes'][lane]['first_fifth_waypoint_step']);end=int(row['first_perch_foot_impulse_step'])
 times=np.arange(start,end,5);times=times[stages[times,lane]>=5]
 x=obs[times,lane].copy();y=labels[times,lane,:4].copy();offset=np.where(times>=2600,4,0).astype(np.int32);idx=offset[:,None]+np.arange(4,dtype=np.int32)[None,:];stage=np.full(len(times),5,dtype=np.int32)
 assert len(times)>0 and np.isfinite(x).all() and np.isfinite(y).all()
 data['validation' if lane==6 else 'train'].append((x,y,idx,stage,lane))
# Only student-visited pre-takeover states whose subsequent teacher first episode reached all five waypoints are used.
recovery=R/'artifacts/crow-v15-student-state-teacher-recovery-20261111/recovery-20261126'
recovery_audit=json.loads((recovery/'analysis.json').read_text());recover_rollout=R/'build/crow-v15-student-state-teacher-recovery-20261111/recovery-20261126-rollout.bin'
rr=read_policy_rollout_pack(recover_rollout,library_path=str(lib));assert (rr.environment_count,rr.control_step_count,rr.action_count)==(8,3000,18)
ro=rr.actor_observations.reshape(3000,8,rr.actor_observation_count);rl=rr.teacher_actions.reshape(3000,8,18);rs=rr.outcome('navigation_waypoints_reached').reshape(3000,8)
for lane in range(7):
 row=recovery_audit['rows'][lane];audit=recovery_audit['pre_takeover_parity'][str(lane)]
 assert not row['physical_failure'] and row['max_waypoints']==5 and audit['first_difference'] is None and audit['first_missing'] is None
 times=np.arange(400,900+100*lane,5);stage=rs[times,lane].astype(np.int32);use=(stage>=0)&(stage<5);times=times[use];stage=stage[use]
 offset=np.where(stage>=4,12,np.where(stage==3,8,np.where(stage==2,4,0))).astype(np.int32);idx=offset[:,None]+np.arange(4,dtype=np.int32)[None,:]
 x=ro[times,lane].copy();y=rl[times,lane,:4].copy();assert np.isfinite(x).all() and np.isfinite(y).all()
 data['validation' if lane in (0,6) else 'train'].append((x,y,idx,stage,lane))
# Corrective labels on later student-visited states with verified safe teacher continuation.
near=R/'artifacts/crow-v15-near-slalom-teacher-recovery-20261111/recovery-20261126'
na=json.loads((near/'analysis.json').read_text());npk=R/'build/crow-v15-near-slalom-teacher-recovery-20261111/recovery-20261126-rollout.bin'
nr=read_policy_rollout_pack(npk,library_path=str(lib));assert (nr.environment_count,nr.control_step_count,nr.action_count)==(8,3000,18)
no=nr.actor_observations.reshape(3000,8,nr.actor_observation_count);nl=nr.teacher_actions.reshape(3000,8,18);ns=nr.outcome('navigation_waypoints_reached').reshape(3000,8)
for lane in range(8):
 row=na['rows'][lane];audit=na['pre_takeover_parity'][str(lane)];cutoff=1500+100*(lane%3)
 assert not row['physical_failure'] and row['max_waypoints']==5 and audit['first_difference'] is None and audit['first_missing'] is None and audit['compared']==cutoff-1
 times=np.arange(400,cutoff,5);stage=ns[times,lane].astype(np.int32);use=(stage>=0)&(stage<5);times=times[use];stage=stage[use]
 offset=np.where(stage>=4,12,np.where(stage==3,8,np.where(stage==2,4,0))).astype(np.int32);idx=offset[:,None]+np.arange(4,dtype=np.int32)[None,:]
 x=no[times,lane].copy();y=nl[times,lane,:4].copy();assert len(times)>0 and np.isfinite(x).all() and np.isfinite(y).all()
 data['validation' if lane in (0,6) else 'train'].append((x,y,idx,stage,lane))

# Only student states whose matching teacher continuation reached all waypoints without a physical failure.
late=R/"artifacts/crow-v15-late-route-teacher-recovery-20261111/recovery-20261126"
la=json.loads((late/"analysis.json").read_text());lp=R/"build/crow-v15-late-route-teacher-recovery-20261111/recovery-20261126-rollout.bin"
lr=read_policy_rollout_pack(lp,library_path=str(lib));assert (lr.environment_count,lr.control_step_count,lr.action_count)==(8,3800,18)
lo=lr.actor_observations.reshape(3800,8,lr.actor_observation_count);ll=lr.teacher_actions.reshape(3800,8,18);ls=lr.outcome("navigation_waypoints_reached").reshape(3800,8)
for lane in (6,):
 row=la["rows"][lane];audit=la["pre_takeover_parity"][str(lane)];cutoff=2100+100*(lane%4)
 assert not row["physical_failure"] and row["max_waypoints"]==5 and audit["first_difference"] is None and audit["first_missing"] is None and audit["compared"]==cutoff-1
 times=np.arange(1400,cutoff,5);stage=ls[times,lane].astype(np.int32);use=(stage>=0)&(stage<5);times=times[use];stage=stage[use]
 offset=np.where(stage>=4,12,np.where(stage==3,8,np.where(stage==2,4,0))).astype(np.int32);idx=offset[:,None]+np.arange(4,dtype=np.int32)[None,:]
 x=lo[times,lane].copy();y=ll[times,lane,:4].copy();assert len(times)>0 and np.isfinite(x).all() and np.isfinite(y).all()
 data["validation" if lane==0 else "train"].append((x,y,idx,stage,lane))

def joined(part):
 z=data[part];assert z
 x=np.concatenate([v[0] for v in z]);y=np.concatenate([v[1] for v in z]);idx=np.concatenate([v[2] for v in z]);stage=np.concatenate([v[3] for v in z])
 counts=np.bincount(stage,minlength=6);assert (counts>0).all()
 weights=(len(stage)/(6.0*counts[stage])).astype(np.float32)
 mask=np.ones((len(stage),18),np.float32);mask[np.arange(len(stage))[:,None],idx]=0
 return x,y,idx,stage,weights,mask,counts
train=joined("train");val=joined("validation")
learner=MLXPolicyLearner.from_policy_pack(initial,MLXPPOConfiguration(learning_rate=2e-5,seed=20261227),library_path=str(lib))
model=learner.model
def baseline(x):
 outputs=[]
 for i in range(0,len(x),1024):
  z=model.actor_mean(mx.array(x[i:i+1024]));mx.eval(z);outputs.append(np.asarray(z))
 return np.concatenate(outputs)
anchor=baseline(train[0])
opt=optim.Adam(learning_rate=2e-5,bias_correction=True);opt.init(model.trainable_parameters())
def loss(model,x,y,idx,weights,mask,base):
 p=model.actor_mean(x);chosen=mx.take_along_axis(p,idx,axis=1)
 imitation=mx.sum(mx.square(chosen-y)*mx.array([1.,2.,.5,1.5]),axis=1)
 return mx.mean(imitation*weights)+0.5*mx.mean(mx.sum(mx.square(p-base)*mask,axis=1))+2.0*mx.mean(mx.sum(mx.square(p-base),axis=1))
valuegrad=nn.value_and_grad(model,loss)
def metrics(d):
 pred=baseline(d[0]);chosen=np.take_along_axis(pred,d[2],axis=1);err=np.abs(chosen-d[1])
 return {"mae_by_command":np.mean(err,axis=0).tolist(),"mae_by_stage":[float(np.mean(err[d[3]==k])) for k in range(6)]}
before=metrics(val);rng=np.random.default_rng(20261227);history=[];start=time.monotonic()
for epoch in range(1,31):
 order=rng.permutation(len(train[0]));batchloss=[]
 for j in range(0,len(order),1024):
  ix=order[j:j+1024]
  l,g=valuegrad(model,mx.array(train[0][ix]),mx.array(train[1][ix]),mx.array(train[2][ix]),mx.array(train[4][ix]),mx.array(train[5][ix]),mx.array(anchor[ix]))
  opt.update(model,g);mx.eval(l,model.state,opt.state);batchloss.append(float(l))
 if epoch in (1,10,20,30):
  item={"epoch":epoch,"train_loss":float(np.mean(batchloss)),"validation":metrics(val),"elapsed_s":round(time.monotonic()-start,2)}
  history.append(item);print(json.dumps(item),flush=True)
trained=B/"trained-with-unused-critic.policypack";learner.revision+=1
learner.write_policy_pack(trained,policy_id="crow_v15_qualified_route_prefix_training_wrapper_v1",library_path=str(lib))
p=read_policy_pack(trained,library_path=str(lib))
def copy_layers(items):return [PolicyDenseLayerArtifact(weights=x.weights.copy(),bias=x.bias.copy(),activation=x.activation) for x in items]
output=B/"route-imitation.policypack"
write_policy_pack(output,policy_id="crow_v15_qualified_route_prefix_imitation_v1",revision=p.revision+1,contract_version=p.contract_version,world_fingerprint=p.world_fingerprint,task_fingerprint=p.task_fingerprint,observation_fingerprint=p.observation_fingerprint,action_fingerprint=p.action_fingerprint,layers=copy_layers(p.layers),observation_mean=p.observation_mean,observation_inverse_standard_deviation=p.observation_inverse_standard_deviation,action_bias=p.action_bias,action_scale=p.action_scale,observation_clip=p.observation_clip,action_clip=p.action_clip,library_path=str(lib))
q=read_policy_pack(output,library_path=str(lib))
assert not q.critic_layers and all(np.array_equal(x.weights,y.weights) and np.array_equal(x.bias,y.bias) and x.activation==y.activation for x,y in zip(p.layers,q.layers))
assert all(np.array_equal(getattr(p,k),getattr(q,k)) for k in ("observation_mean","observation_inverse_standard_deviation","action_bias","action_scale"))
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
result={"data_scope":"Teacher route prefixes, pre-contact approaches from three lanes, and student-visited states with matched teacher recovery on one used layout and seed. Failed first episodes excluded. Correlated validation is not unused-layout evidence; teacher contact balancing is not reproduced by the actor.","train_lanes":sorted(valid-validation),"validation_lanes":sorted(validation),"train_samples":len(train[0]),"validation_samples":len(val[0]),"train_stage_counts":train[6].tolist(),"validation_stage_counts":val[6].tolist(),"validation_before":before,"history":history,"input_rollout_sha256":sha(pack),"initial_training_wrapper_sha256":sha(initial),"original_actor_sha256":sha(R/"build/crow-v15-near-slalom-student-imitation-20261111/route-imitation.policypack"),"unused_critic_stripped":True,"trained_policy_sha256":sha(output),"host_sha256":sha(lib),"seed":20261227}
result['train_lanes']={'teacher_route':[1,3,4,5,7],'student_recovery':[1,2,3,4,5],'pre_contact_approach':[0,4]}
result['validation_lanes']={'teacher_route':[0,6],'student_recovery':[0,6],'pre_contact_approach':[6]}
result['validation_caveat']='Approach lane 0 is used in training while its route-prefix samples are in validation; all lanes share a course and seed. This is correlated development validation.'
result['approach_command_offset']='For pre-contact approach, use output slots 4–7 at step >=2600, slots 0–3 earlier; physical landing balance is not learned.'
result['student_recovery_rollout_sha256']=sha(recover_rollout)
result['near_slalom_rollout_sha256']=sha(npk)
result['near_slalom_training_lanes']=[1,2,3,4,5,7]
result['near_slalom_validation_lanes']=[0,6]
result['data_scope'] += ' Includes eight new matching student trajectories with safe teacher recovery; all share the development seed and layout.'
result["late_recovery_rollout_sha256"]=sha(lp);result["late_recovery_train_lanes"]=[6];result["late_recovery_validation_lanes"]=[];result["anchor_penalty_all_outputs"]=2.0;result["learning_rate"]=2e-5;result["epochs"]=30;result["validation_caveat"]="Correlated development split; late recovery lane 6 is used for training and was previously validation. All share one seed and layout."
(A/"training.json").write_text(json.dumps(result,indent=2)+"\n")
