from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-pitch-clearance-combined-20261114');states={};ev={};imp=collections.defaultdict(lambda:[0.]*37)
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if not x:continue
  if x[0]=='step':
   q=x.index('q');v=x.index('v');states[int(x[1]),int(x[2])+1]=(list(map(float,x[q+1:v])),list(map(float,x[v+1:])))
  elif x[0]=='event':ev[int(x[1]),int(x[2])+1]=(bool(int(x[3])),float(x[13]))
with gzip.open(A/'substeps.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if x and x[0]=='CROW_TWO_TOE_SUBSTEP':
   t,e=int(x[1]),int(x[3]);a=imp[e,t]
   for i,v in enumerate(map(float,x[6:])):a[i]+=v
out={}
for e in (0,1):
 counts=collections.Counter();first={};sample=[]
 for t in range(230,451):
  q,v=states[e,t]
  if q[2]<.1:break
  rel,gap=ev[e,t];a=imp[e,t];tilt=math.acos(max(-1,min(1,1-2*(q[3]**2+q[4]**2))));speed=math.sqrt(sum(x*x for x in v[:3]));front=sum(a[29+i] for i in range(6));rear=sum(a[29+i] for i in (6,7));g={'released':rel,'clearance':gap>=.001,'released_foot_unloaded':abs(a[2])<=1e-7,'stance_upward':a[1]>=.5*.45004*9.81*.02,'stance_front':front>=.0002,'stance_rear':rear<=-.0002,'nonfoot':abs(a[4])<=1e-7,'tilt':tilt<=.35,'speed':speed<=.12}
  for k,ok in g.items():
   if not ok:counts[k]+=1;first.setdefault(k,t)
  if t in (230,235,240,250,260,270,280,290,300,310,320,330):sample.append({'interval':t,'failed':[k for k,v in g.items() if not v],'tilt_rad':tilt,'speed_m_s':speed,'stance_front_Ns':front,'stance_rear_Ns':rear,'clearance_m':gap})
 out[str(e)]={'failure_counts_before_height_gate':counts,'first_failure_intervals':first,'samples':sample}
(A/'gate-failures.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
