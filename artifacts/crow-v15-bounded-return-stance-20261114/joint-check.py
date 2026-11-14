from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-bounded-return-stance-20261114');rows={0:[],1:[]}
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if x and x[0]=='step':
   e=int(x[1]);q=x.index('q');v=x.index('v');qs=list(map(float,x[q+1:v]));vs=list(map(float,x[v+1:]));rows[e].append((qs[15],qs[16],vs[14],vs[15]))
out={}
for e,r in rows.items():
 out[str(e)]={'samples':len(r),'knee_position_range_rad':[min(x[0] for x in r),max(x[0] for x in r)],'ankle_position_range_rad':[min(x[1] for x in r),max(x[1] for x in r)],'peak_abs_knee_speed_rad_s':max(abs(x[2]) for x in r),'peak_abs_ankle_speed_rad_s':max(abs(x[3]) for x in r),'within_position_limits':all(-.100001<=x[0]<=1.550001 and -.900001<=x[1]<=.700001 for x in r),'scope':'accepted control states, not substep actuator effort measurements'}
(A/'joint-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
