from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-airborne-preclose-regrip-20261114');states={};fall={};samples={};dups=[];errors=[]
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if x and x[0]=='step':
   e,t=map(int,x[1:3]);i=x.index('v');states[e,t]=list(map(float,x[i+1:]));
   if float(x[x.index('q')+3])<.1:fall.setdefault(e,t)
with gzip.open(A/'substeps.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if not x:continue
  if 'ERROR' in x[0]:errors.append(line.strip())
  if x[0]=='CROW_TOE_V_SUBSTEP':
   t,k,e=map(int,x[1:4]);key=e,t,k
   if key in samples:dups.append(key)
   samples[key]=list(map(float,x[4:]))
rows=[]
for e in (0,1):
 bad=[(t,k) for t in range(1,451) for k in range(32) if (e,t,k) not in samples or len(samples[e,t,k])!=16 or not all(map(math.isfinite,samples[e,t,k]))];dif=[]
 for t in range(1,451):
  if (e,t,31) in samples and (e,t-1) in states:dif.extend(abs(a-b) for a,b in zip(samples[e,t,31],states[e,t-1][22:38]))
 selected=[(t,k,a) for (lane,t,k),a in samples.items() if lane==e and (e not in fall or t-1<fall[e])]; hits=[(t,k) for t,k,a in selected if any(abs(z)>=11.99 for z in a)];rows.append({'lane':e,'missing_invalid_substep_count':len(bad),'first_root_fall_zero_based':fall.get(e),'pre_fall_max_toe_speed_rad_s':max((max(map(abs,a)) for t,k,a in selected),default=None),'pre_fall_threshold_substep_count':len(hits),'first_threshold_substeps':hits[:5],'final_substep_accepted_max_abs_difference_rad_s':max(dif,default=None)})
out={'rows':rows,'duplicate_sample_count':len(dups),'observer_errors':errors,'scope':'destination toe velocities at physics substeps; contact/state epoch alignment not established'};(A/'velocity-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
