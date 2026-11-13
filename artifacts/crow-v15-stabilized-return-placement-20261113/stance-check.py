from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-stabilized-return-placement-20261113');prep={0:[],1:[]};target={0:{},1:{}};state={0:[],1:[]}
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if not x:continue
  e=int(x[1]) if len(x)>1 and x[1] in ('0','1') else None
  if e is None:continue
  if x[0]=='stanceprep':prep[e].append((int(x[2]),float(x[3]),int(x[4])))
  elif x[0]=='stancetarget':target[e].setdefault(int(x[3]),[]).append(float(x[4]))
  elif x[0]=='step':
   q=x.index('q');v=x.index('v');qs=list(map(float,x[q+1:v]));vs=list(map(float,x[v+1:]));state[e].append((qs[18],qs[19],vs[17],vs[18]))
out={}
for e in (0,1):
 p=prep[e];r=state[e];inc=[(a,b) for a,b in zip(p,p[1:]) if b[1]>a[1]+1e-7]
 out[str(e)]={'prep_samples':len(p),'max_fraction':max((x[1] for x in p),default=0),'max_slew':max((b[1]-a[1] for a,b in zip(p,p[1:])),default=0),'increments_without_50_supported':sum(b[2]<50 for a,b in inc),'target_ranges_rad':{str(j):[min(v),max(v)] for j,v in target[e].items()},'accepted_knee_range_rad':[min(x[0] for x in r),max(x[0] for x in r)],'accepted_ankle_range_rad':[min(x[1] for x in r),max(x[1] for x in r)],'accepted_peak_speeds_rad_s':[max(abs(x[2]) for x in r),max(abs(x[3]) for x in r)],'within_model_position_limits':all(-.100001<=x[0]<=1.550001 and -.900001<=x[1]<=.700001 for x in r),'scope':'accepted states and logged target gates, not substep actuator effort'}
(A/'stance-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
