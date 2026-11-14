from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-stance-pitch-damping-20261114');u={0:[],1:[]}
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if x and x[0]=='pitchcontrol':u[int(x[1])].append(float(x[-1]))
rows=[]
for e in (0,1):
 v=u[e];steps=[abs(b-a) for a,b in zip(v,v[1:])];lim,rate=((.06 if e==0 else .10),.002);rows.append({'lane':e,'samples':len(v),'max_abs_correction_rad':max(map(abs,v)),'max_slew_rad_per_interval':max(steps),'within_bounds':len(v)==450 and max(map(abs,v))<=lim+1e-6 and max(steps)<=rate+1e-6})
o={'rows':rows,'pass':all(x['within_bounds'] for x in rows),'scope':'control targets, not measured physical effort'};(A/'correction-check.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
