from pathlib import Path
import gzip,json
A=Path(__file__).resolve().parent; data={0:[],1:[]}; targets={0:[],1:[]}
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if not x: continue
  if x[0]=='step':
   e=int(x[1]);q=x.index('q');v=x.index('v');qs=list(map(float,x[q+1:v]));vs=list(map(float,x[v+1:]));data[e].append((qs[17],vs[16]))
  elif x[0]=='rightHipCommand':targets[int(x[1])].append(float(x[3]))
rows=[]
for e in (0,1):
 d=data[e];u=targets[e];rows.append({'lane':e,'state_samples':len(d),'target_samples':len(u),'position_range_rad':[min(x[0] for x in d),max(x[0] for x in d)],'peak_accepted_speed_rad_s':max(abs(x[1]) for x in d),'target_range_rad':[min(u),max(u)],'position_in_bounds':len(d)==450 and all(-.850001<=x[0]<=.850001 for x in d),'target_in_bounds':len(u)==450 and all(-.850001<=x<=.850001 for x in u)})
out={'rows':rows,'scope':'accepted states and commanded targets; not substep speed or drive effort'};(A/'stance-hip-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
