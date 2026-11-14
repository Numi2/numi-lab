from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-perch-toe-substep-velocity-20261114');B=Path('artifacts/crow-v15-roll32-return-placement-ablation-20261114')
def state(p):
 d={};v={}
 with gzip.open(p/'trace.log.gz','rt') as f:
  for line in f:
   x=line.split()
   if x and x[0] in ('step','control','event','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','placement'):
    e,t=map(int,x[1:3]);d.setdefault((e,t),[]).append((x[0],x[3:]))
    if x[0]=='step':i=x.index('v');v[e,t]=list(map(float,x[i+1:]))
 return d,v
def subs(p):
 c={};v={};errs=[]
 with gzip.open(p/'substeps.log.gz','rt') as f:
  for line in f:
   x=line.split()
   if not x:continue
   if 'ERROR' in x[0]:errs.append(line.strip())
   if x[0] in ('CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP'):
    t,k,e=map(int,x[1:4]);d=c if x[0]=='CROW_TWO_TOE_SUBSTEP' else v;key=e,t,k
    if key in d:errs.append('duplicate '+str(key))
    d[key]=x[4:] if d is c else list(map(float,x[4:]))
 return c,v,errs
a,av=state(A);b,bv=state(B);c,v,err=subs(A);bc,_,berr=subs(B);rows=[]
for e in (0,1):
 statebad=[t for t in range(450) if (e,t) not in a or (e,t) not in b or a[e,t]!=b[e,t]];cbad=[(t,k) for t in range(1,451) for k in range(32) if c.get((e,t,k))!=bc.get((e,t,k)) or (e,t,k) not in c];vmissing=[(t,k) for t in range(1,451) for k in range(32) if (e,t,k) not in v or len(v[e,t,k])!=16 or not all(map(math.isfinite,v[e,t,k]))];epoch=[]
 for t in range(1,451):
  if (e,t,31) in v and (e,t-1) in av:epoch.extend(abs(x-y) for x,y in zip(v[e,t,31],av[e,t-1][22:38]))
 # Parse root z directly from the first three values after the q marker.
 fall=None
 with gzip.open(A/'trace.log.gz','rt') as f:
  for line in f:
   x=line.split()
   if x and x[0]=='step' and int(x[1])==e and float(x[x.index('q')+3])<.1:fall=int(x[2]);break
 peak=max((max(map(abs,vals)) for (lane,t,k),vals in v.items() if lane==e and (fall is None or t-1<fall)),default=None);hits=sum(any(abs(z)>=11.99 for z in vals) for (lane,t,k),vals in v.items() if lane==e and (fall is None or t-1<fall))
 rows.append({'lane':e,'state_command_event_mismatch_count':len(statebad),'contact_substep_mismatch_count':len(cbad),'missing_invalid_velocity_substeps':len(vmissing),'final_substep_accepted_velocity_max_abs_difference_rad_s':max(epoch,default=None),'first_root_fall_zero_based':fall,'pre_fall_substep_toe_peak_rad_s':peak,'pre_fall_substeps_with_toe_speed_threshold':hits})
out={'rows':rows,'observer_errors':err+berr,'scope':'passive same-start comparison; final-substep velocity check does not independently synchronize contact epoch or establish biological realism'};(A/'comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
