from pathlib import Path
import gzip,shutil,hashlib,json,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack,read_policy_rollout_pack
R=Path.cwd();A=R/'artifacts/crow-v15-policy-error-audit-20261110';H=R/'build/crow-v15-long-perch-host-20261110';lib=H/'libmetalrobo.dylib';arc=R/'artifacts/crow-v15-approach-j-early-recovery-20261110/development-j-approach-early-recovery.rollout.gz';temp=H/'audit-approach-j.tmp.rollout'
with gzip.open(arc,'rb') as f,temp.open('wb') as g:shutil.copyfileobj(f,g)
try:
 r=read_policy_rollout_pack(temp,library_path=str(lib));obs=r.actor_observations.reshape(7000,1,r.actor_observation_count)[:,0,:];labels=r.teacher_actions.reshape(7000,1,18)[:,0,:4];stage=r.outcome('navigation_waypoints_reached').reshape(7000,1)[:,0].astype(int)
 models=[('original',H/'route.policypack'),('prefix',R/'build/crow-v15-prefix-learning-20261110/route-prefix-actor.policypack'),('anchored',R/'build/crow-v15-anchored-learning-20261110/corrective-actor.policypack'),('approach',R/'build/crow-v15-approach-learning-20261110/corrective-actor.policypack'),('recovery',R/'build/crow-v15-recovery-learning-20261110/corrective-actor.policypack')];out={}
 for name,path in models:
  model=read_policy_pack(path,library_path=str(lib));pred=model.actor_mean(obs)[:,:4];res={}
  for label,sel in [(f'stage_{s}',(stage==s)&(np.arange(7000)<2100)) for s in range(3)]+[('last_200_before_takeover',(np.arange(7000)>=1900)&(np.arange(7000)<2100))]:
   if not sel.any():continue
   err=pred[sel]-labels[sel];res[label]={'count':int(sel.sum()),'mae':np.mean(abs(err),axis=0).tolist(),'bias':np.mean(err,axis=0).tolist(),'lateral_saturation_fraction':float(np.mean(abs(pred[sel,1])>.98)),'teacher_lateral_mean':float(labels[sel,1].mean()),'actor_lateral_mean':float(pred[sel,1].mean())}
  out[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'groups':res}
 out['limits']='The sample is used development student states and part of recovery training; output is not an independent holdout or a substitute for safe autonomous rollout.';(A/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
finally:temp.unlink(missing_ok=True)
