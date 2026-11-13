from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-supported-foot-placement-return-20261113');r={0:[],1:[]};targets={0:{},1:{}}
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if not x:continue
  if x[0]=='placement':r[int(x[1])].append((int(x[2]),float(x[3]),int(x[4]),int(x[5])))
  elif x[0]=='jointtarget':targets[int(x[1])].setdefault(int(x[4-1]),[]).append(float(x[4]))
out={}
for e,v in r.items():
 increments=[(a,b) for a,b in zip(v,v[1:]) if b[1]>a[1]+1e-7]
 out[str(e)]={'samples':len(v),'max_fraction':max(z[1] for z in v),'max_slew':max((b[1]-a[1] for a,b in zip(v,v[1:])),default=0.),'increments_without_support':sum(not b[2] for a,b in increments),'increments_without_posture':sum(not b[3] for a,b in increments),'target_ranges_rad':{str(j):[min(z),max(z)] for j,z in targets[e].items()},'scope':'logged targets and accepted support gates, not substep torque'}
(A/'placement-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
