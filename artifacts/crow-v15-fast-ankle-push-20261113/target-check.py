from pathlib import Path
import collections,gzip,json,math
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-fast-ankle-push-20261113';run=A/'push-nowings-20261127-rebound-fast80/run.stderr.gz';target={};events=[]
with gzip.open(run,'rt') as f:
 for l in f:
  if l.startswith('CROW_DRIVE_TARGET '):
   z=l.split();t,e=map(int,z[1:3]);v=list(map(float,z[3:]));assert len(v)==40 and all(map(math.isfinite,v));assert (e,t) not in target;target[e,t]=v
  elif l.startswith('CROW_TOE_V_SUBSTEP '):
   z=l.split();t,k,e=map(int,z[1:4]);v=list(map(float,z[4:]));assert len(v)==16
   if e>=2 and 348<=t<=380:
    for j,x in enumerate(v):
     if abs(x)>=11.99:events.append({'lane':e,'step':t,'substep':k,'toe_dof_index':j,'velocity_rad_s':x})
assert len(target)==8*46,len(target)
rows=[{'lane':e,'step':t,'left_rear_target':target[e,t][28],'right_rear_target':target[e,t][36]} for e in range(8) for t in range(335,381)]
out={'target_samples':len(target),'target_rows':rows,'toe_speed_threshold_events':events,'toe_dof_counts':dict(collections.Counter(x['toe_dof_index'] for x in events)),'scope':'Copied target after task binding, before solver clamp; velocity events are >=11.99 rad/s and do not alone prove clipping. Observational used-seed data.'};(A/'target-check.json').write_text(json.dumps(out,indent=2)+'\n');print('target rows',len(target),'threshold events',len(events),'by toe',out['toe_dof_counts']);print('launch rear target ranges',[(e,min(target[e,t][28] for t in range(348,373)),max(target[e,t][28] for t in range(348,373)),min(target[e,t][36] for t in range(348,373)),max(target[e,t][36] for t in range(348,373))) for e in range(2,8)])
