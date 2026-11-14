from pathlib import Path
import gzip,json
p=Path('artifacts/crow-v15-bilateral-stance-recenter-20261114');events={}
with gzip.open(p/'trace.log.gz','rt') as f:
 for line in f:
  z=line.split()
  if z and z[0]=='event':
   e,t=map(int,z[1:3]);events[e,t]=list(map(float,z[3:]))
out=[]
for e in (0,1):
 def right(v):return v[3]>.001 and v[6]>.000001 and v[7]<-.000001
 def left(v):return v[2]>.000006 and v[4]>.000001 and v[5]<-.000001
 ret=next((t for t in range(450) if events[e,t][8]>0),None)
 out.append({'lane':e,'return_zero_based':ret,'first_postreturn_right_gate_loss':next((t for t in range(ret,450) if not right(events[e,t])),None) if ret is not None else None,'first_postreturn_left_opposing_loaded':next((t for t in range(ret,450) if left(events[e,t])),None) if ret is not None else None,'scope':'accepted control-interval event only; epochs not independently aligned'})
(p/'support-timing.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
