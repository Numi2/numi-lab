from pathlib import Path
import gzip,shutil,json,numpy as np
from metalrobo.mlx_policy_learning import read_policy_pack,read_policy_rollout_pack
R=Path.cwd();A=R/'artifacts/crow-v15-paired-label-diagnostic-20261110';H=R/'build/crow-v15-long-perch-host-20261110';lib=str(H/'libmetalrobo.dylib');arc=R/'artifacts/crow-v15-teacher-terminal-handoff-20261110/j-handoff.rollout.gz';tmp=H/'j-handoff-diagnostic.tmp.rollout'
with gzip.open(arc,'rb') as inp,tmp.open('wb') as out:shutil.copyfileobj(inp,out)
r=read_policy_rollout_pack(tmp,library_path=lib);tmp.unlink();obs=r.actor_observations.reshape(7000,r.actor_observation_count);labels=r.teacher_actions.reshape(7000,18);src=R/'build/crow-v15-stage-aware-learning-20261110/corrective-actor.policypack';dst=R/'build/crow-v15-paired-corrective-learning-20261110/corrective-actor.policypack';models={'before':read_policy_pack(src,library_path=lib),'paired':read_policy_pack(dst,library_path=lib)}
y=labels[:,:4];matches=np.stack([np.all(np.isclose(labels[:,4*k:4*k+4],y,rtol=0,atol=1e-7),axis=1) for k in (1,2,3)],axis=1);groups=np.where(matches.any(axis=1),matches.argmax(axis=1)+1,0)
results={}
for name,model in models.items():
 pred=model.actor_mean(obs).reshape(-1,18);sel=pred[:,:16].reshape(-1,4,4)[np.arange(7000),groups];report=[]
 for lo,hi in [(1500,1800),(1800,2050),(2050,2250),(2250,2450),(2450,2600),(2600,2699)]:
  p=sel[lo:hi];t=y[lo:hi];report.append({'steps':[lo,hi],'mean_absolute_error':np.mean(np.abs(p-t),axis=0).tolist(),'mean_signed_error':np.mean(p-t,axis=0).tolist(),'max_abs_error':np.max(np.abs(p-t),axis=0).tolist()})
 results[name]=report
(A/'results.json').write_text(json.dumps({'comparison':results,'limits':'Correlated teacher-recovery observations; not the new actor trajectory or an independent success estimate.'},indent=2)+'\n');print(json.dumps(results,indent=2))
