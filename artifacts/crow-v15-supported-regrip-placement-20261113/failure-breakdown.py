from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-supported-regrip-placement-20261113');state={};events={};imp=collections.defaultdict(lambda:[0.]*37)
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if not x:continue
  if x[0]=='step':
   q=x.index('q');v=x.index('v');state[int(x[1]),int(x[2])+1]=(list(map(float,x[q+1:v])),list(map(float,x[v+1:])))
  elif x[0]=='event':events[int(x[1]),int(x[2])+1]=(bool(int(x[3])),float(x[13]))
with gzip.open(A/'substeps.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if x and x[0]=='CROW_TWO_TOE_SUBSTEP':
   t,e=int(x[1]),int(x[3]);a=imp[e,t]
   for i,v in enumerate(map(float,x[6:])):a[i]+=v
out={}
for e in (0,1):
 c=collections.Counter();first={};fall=None
 for t in range(1,451):
  if (e,t) not in state:continue
  q,v=state[e,t]
  if q[2]<.1:fall=t;break
  release,clear=events[e,t]
  if not release:continue
  a=imp[e,t];tilt=math.acos(max(-1.,min(1.,1.-2*(q[3]**2+q[4]**2))));speed=math.sqrt(sum(x*x for x in v[:3]));checks={'clearance':clear>=.001,'left_no_contact':abs(a[2])<=1e-7,'right_support':a[1]>=.5*.45004*9.81*.02,'right_opposing_toes':sum(a[29+i] for i in range(6))>=.0002 and sum(a[29+i] for i in (6,7))<=-.0002,'no_nonfoot':abs(a[4])<=1e-7,'tilt':tilt<=.35,'speed':speed<=.12}
  c['release_intervals']+=1
  for k,ok in checks.items():
   if not ok:c[k+'_failed']+=1;first.setdefault(k,t)
  if all(checks.values()):c['all_passed']+=1
 out[str(e)]={'first_fall':fall,'counts':dict(c),'first_failed_interval_by_check':first}
(A/'failure-breakdown.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
