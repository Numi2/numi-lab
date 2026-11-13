from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-supported-regrip-placement-20261113');p={0:[],1:[]}
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if x and x[0]=='placement':p[int(x[1])].append((int(x[2]),float(x[3])))
r={}
for e in (0,1):
 z=p[e];cap=(.045,.075)[e];r[str(e)]={'samples':len(z),'first_nonzero_zero_based':next((t for t,x in z if x>0),None),'max_offset_rad':max((x for _,x in z),default=None),'max_slew_rad':max((abs(a[1]-b[1]) for a,b in zip(z,z[1:])),default=None),'within_cap':len(z)==450 and max(x for _,x in z)<=cap+1e-6,'within_slew':len(z)==450 and max(abs(a[1]-b[1]) for a,b in zip(z,z[1:]))<=.003001}
(A/'placement-check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
