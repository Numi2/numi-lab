from pathlib import Path
import gzip,json,math,collections
A=Path('artifacts/crow-v15-clearance-posture-gate-20261114');state={};events={};imp=collections.defaultdict(lambda:[0.]*37)
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if not x:continue
  if x[0]=='step':
   q=x.index('q');v=x.index('v');state[int(x[1]),int(x[2])+1]=(list(map(float,x[q+1:v])),list(map(float,x[v+1:])))
  elif x[0]=='event':events[int(x[1]),int(x[2])+1]=bool(int(x[11]))
with gzip.open(A/'substeps.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if x and x[0]=='CROW_TWO_TOE_SUBSTEP':
   t,e=int(x[1]),int(x[3]);a=imp[e,t]
   for i,v in enumerate(map(float,x[6:])):a[i]+=v
out={}
for e in (0,1):
 passed=[];first_finish=next((t for t in range(1,451) if events.get((e,t))),None);fail=collections.Counter()
 for t in range(first_finish or 451,451):
  if (e,t) not in state:continue
  q,v=state[e,t]
  if q[2]<.1:break
  a=imp[e,t];tilt=math.acos(max(-1,min(1,1-2*(q[3]**2+q[4]**2))));speed=math.sqrt(sum(x*x for x in v[:3]));left=sum(a[21+i] for i in range(6))>=.0002 and sum(a[21+i] for i in (6,7))<=-.0002;right=sum(a[29+i] for i in range(6))>=.0002 and sum(a[29+i] for i in (6,7))<=-.0002;checks={'both_up':min(a[0],a[1])>.0002 and a[0]+a[1]>=.5*.45004*9.81*.02,'left_opp':left,'right_opp':right,'no_nonfoot':abs(a[4])<=1e-7,'tilt':tilt<=.35,'speed':speed<=.12}
  for k,vv in checks.items():
   if not vv:fail[k]+=1
  if all(checks.values()):passed.append(t)
 streaks=[]
 for t in passed:
  if not streaks or t!=streaks[-1][-1]+1:streaks.append([t])
  else:streaks[-1].append(t)
 out[str(e)]={'return_step':first_finish,'qualifying_steps':passed,'streaks':[[s[0],s[-1],len(s)] for s in streaks],'failed_gate_counts':dict(fail)}
(A/'regrip-intervals.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
