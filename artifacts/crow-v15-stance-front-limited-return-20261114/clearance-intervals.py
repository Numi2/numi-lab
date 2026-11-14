from pathlib import Path
import gzip,json,math,collections
A=Path('artifacts/crow-v15-stance-front-limited-return-20261114');state={};events={};imp=collections.defaultdict(lambda:[0.]*37)
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if not x:continue
  if x[0]=='step':
   q=x.index('q');v=x.index('v');state[int(x[1]),int(x[2])+1]=(list(map(float,x[q+1:v])),list(map(float,x[v+1:])) )
  elif x[0]=='event':events[int(x[1]),int(x[2])+1]=(bool(int(x[3])),float(x[13]))
with gzip.open(A/'substeps.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if x and x[0]=='CROW_TWO_TOE_SUBSTEP':
   t,e=int(x[1]),int(x[3]);a=imp[e,t]
   for i,v in enumerate(map(float,x[6:])):a[i]+=v
out={}
for e in (0,1):
 passed=[];seq=[]
 for t in range(1,451):
  if (e,t) not in state:continue
  q,v=state[e,t]
  if q[2]<.1:break
  release,gap=events[e,t]
  if not release:continue
  a=imp[e,t];tilt=math.acos(max(-1,min(1,1-2*(q[3]**2+q[4]**2))));speed=math.sqrt(sum(x*x for x in v[:3]));rightopp=sum(a[29+i] for i in range(6))>=.0002 and sum(a[29+i] for i in (6,7))<=-.0002;ok=gap>=.001 and abs(a[2])<=1e-7 and a[1]>=.5*.45004*9.81*.02 and rightopp and abs(a[4])<=1e-7 and tilt<=.35 and speed<=.12
  if ok:
   x,y,z,w=q[3:7];pitch=math.asin(max(-1.,min(1.,2*(w*y-z*x))));passed.append({'step':t,'gap_m':gap,'right_up_impulse_Ns':a[1],'root_pitch_rad':pitch,'pitch_rate_rad_s':v[4],'root_speed_m_s':speed,'root_x_m':q[0]})
 for p in passed:
  if not seq or p['step']!=seq[-1][-1]+1:seq.append([p['step']])
  else:seq[-1].append(p['step'])
 out[str(e)]={'passed_intervals':passed,'streaks':[[v[0],v[-1],len(v)] for v in seq]}
(A/'clearance-intervals.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
