from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-rear-toe-effort-counterfactual-20261114'); B=Path('artifacts/crow-v15-early-stance-coupling-20261114')
def readstates(path):
 out={}
 with gzip.open(path,'rt') as f:
  for line in f:
   z=line.split()
   if z and z[0]=='step':
    e,t=map(int,z[1:3]);i=z.index('q');j=z.index('v');out[e,t]={'q':list(map(float,z[i+1:j])),'v':list(map(float,z[j+1:]))}
 return out
new=readstates(A/'trace.log.gz');old=readstates(B/'trace.log.gz');samples={};errors=[]
with gzip.open(A/'substeps.log.gz','rt') as f:
 for line in f:
  a=line.split()
  if not a:continue
  if 'ERROR' in a[0]:errors.append(line.strip())
  if a[0]=='CROW_TOE_DRIVE_EFFORT':
   t,k,e=map(int,a[1:4]);key=(e,t,k)
   if key in samples:errors.append('duplicate effort '+str(key))
   samples[key]=list(map(float,a[4:]))
rows=[]
for e in (0,1):
 missing=[(t,k) for t in range(1,451) for k in range(32) if (e,t,k) not in samples or len(samples[e,t,k])!=16 or not all(map(math.isfinite,samples[e,t,k]))]; effort=[samples[e,t,k][14] for t in range(1,451) for k in range(32) if (e,t,k) in samples];mid=[samples[e,t,k][14] for t in range(280,361) for k in range(32) if (e,t,k) in samples];first=next((t for t in range(450) if new.get((e,t))!=old.get((0,t))),None)
 rows.append({'lane':e,'states':sum((e,t) in new for t in range(450)),'first_accepted_state_different_from_archived_control_zero_based':first,'missing_or_invalid_effort_substeps':len(missing),'max_abs_rear_proximal_effort_nm':max(map(abs,effort),default=None),'saturated_fraction_intervals_280_360':sum(abs(v)>=.03996 for v in mid)/len(mid) if mid else None,'rear_proximal_angles':{str(t):new[e,t]['q'][37] for t in (280,300,310,320,340,360) if (e,t) in new},'observed_effort_within_new_limit':bool(effort) and max(map(abs,effort))<=.040001})
out={'lanes':rows,'observer_errors':errors,'scope':'Constructed-state parameter counterfactual; applied effort is modeled implicit-drive output and does not establish biological torque; contact epochs not independently synchronized.'};(A/'counterfactual-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
