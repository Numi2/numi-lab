from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-roll32-rear-knee-cap-20261114');rows={e:{'positions':[],'speeds':[],'targets':[],'fall':None} for e in (0,1)}
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if not x or x[0] not in ('step','balance'):continue
  e,t=int(x[1]),int(x[2]);r=rows[e]
  if x[0]=='balance':r['targets'].append(float(x[-1]))
  else:
   qi=x.index('q');vi=x.index('v');q=list(map(float,x[qi+1:vi]));v=list(map(float,x[vi+1:]));r['positions']+=q[39:41];r['speeds'] += [abs(x) for x in v[38:40]]
   if q[2]<.1 and r['fall'] is None:r['fall']=t
out=[]
for e,r in rows.items():
 cap=.32;out.append({'lane':e,'target_range_rad':[min(r['targets']),max(r['targets'])],'target_near_cap_intervals':sum(abs(x)>=cap-1e-6 for x in r['targets']),'joint_position_range_rad':[min(r['positions']),max(r['positions'])],'position_within_physical_035_limit':all(-.350001<=x<=.350001 for x in r['positions']),'max_accepted_joint_speed_rad_s':max(r['speeds']),'first_fall_zero_based':r['fall']})
o={'rows':out,'scope':'accepted control states only; not substep speed, actuator effort or biological limits'};(A/'roll-limits.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
