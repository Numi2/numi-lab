from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-return-hip-rear-feedback-20261114');x={};control={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  a=l.split()
  if not a:continue
  if a[0]=='returnHip':x[int(a[1]),int(a[2])]=float(a[3])
  if a[0]=='control':control[int(a[1]),int(a[2])]=list(map(float,a[3:]))
rows=[]
for e in (0,1):
 ts=sorted(t for k,t in x if k==e);vals=[x[e,t] for t in ts];steps=[abs(b-a) for a,b in zip(vals,vals[1:])];rows.append({'lane':e,'samples':len(vals),'maximum_correction_rad':max(vals,default=0),'minimum_correction_rad':min(vals,default=0),'max_slew_rad':max(steps,default=0),'bound_pass':max(vals,default=0)<=.010001 and min(vals,default=0)>=-1e-7 and max(steps,default=0)<=.001001})
o={'rows':rows,'note':'Target correction bounds only; gating is implemented from prior contact and must be audited against raw event timing.'};(A/'return-hip-check.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
