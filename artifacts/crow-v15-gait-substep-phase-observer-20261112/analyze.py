from pathlib import Path
import gzip,json,math,collections,os,subprocess
A=Path('artifacts/crow-v15-gait-substep-phase-observer-20261112');P=Path('artifacts/crow-v15-magpie-phase-fit-screen-20261112/candidate-20261127-rebound-phase');q={};sub={}
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0]=='CROW_Q_SUBSTEP':
   t,k,e=map(int,z[1:4]);v=list(map(float,z[4:]));assert len(v)==41 and (e,t,k) not in q;q[e,t,k]=v
  elif z[0]=='CROW_TWO_TOE_SUBSTEP':
   t,k,e=map(int,z[1:4])
   if 201<=t<=300:
    v=list(map(float,z[6:]));assert len(v)==37 and (e,t,k) not in sub;sub[e,t,k]=v
assert len(q)==len(sub)==12800 and set(q)==set(sub)
inp=''.join(f'{e} {(t-201)*16+k} '+' '.join(map(str,v))+' '+' '.join(['0']*40)+'\n' for (e,t,k),v in sorted(q.items()))
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())}
p=subprocess.run(['build/crow-v15-gait-substep-phase-observer-20261112/kinematics'],input=inp,text=True,capture_output=True,check=True,env=env);b={}
for line in p.stdout.splitlines():
 z=line.split();assert z[0]=='body';key=int(z[1]),int(z[2]),z[3];assert key not in b;b[key]=list(map(float,z[4:]))
assert len(b)==12800*8

def rot(q,v):
 x,y,z,w=q;u=(x,y,z);d=sum(u[i]*v[i] for i in range(3));c=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);s=w*w-sum(x*x for x in u);return [s*v[i]+2*d*u[i]+2*w*c[i] for i in range(3)]
def point(b,v):
 q=rot(b[3:7],v);return [q[i]+b[i] for i in range(3)]
def angle(h,k,v):
 x=[h[i]-k[i] for i in range(3)];y=[v[i]-k[i] for i in range(3)];d=sum(x[i]*y[i] for i in range(3))/(math.sqrt(sum(i*i for i in x))*math.sqrt(sum(i*i for i in y)));return math.degrees(math.acos(max(-1,min(1,d))))
records=[];rejects=collections.Counter();maxerr=0.
for e in range(8):
 for side,sd in enumerate(('left','right')):
  samples=[]
  for t in range(201,301):
   for k in range(16):
    z=sub[e,t,k];v=z[5+side*8:13+side*8];samples.append((z[2+side]>1e-7,sum(v[:6])>1e-7,sum(v[6:])>1e-7))
  active=[i for i,v in enumerate(samples) if v[0]];groups=[]
  for i in active:
   if not groups or i-groups[-1][-1]>3:groups.append([i])
   else:groups[-1].append(i)
  events=[]
  for g in groups:
   start,end=g[0],g[-1]
   if start==0 or end==len(samples)-1:rejects['partial_contact']+=1;events.append(None);continue
   front=[i for i in range(start,end+1) if samples[i][1]];rear=[i for i in range(start,end+1) if samples[i][2]]
   if not front or not rear:rejects['missing_toe_group']+=1;events.append(None);continue
   if (end-start+1)*.00125<.04:rejects['contact_under_40ms']+=1;events.append(None);continue
   events.append({'rear_first':rear[0],'front_last':front[-1]})
  for prev,nxt in zip(events,events[1:]):
   if not prev or not nxt:rejects['incomplete_pair']+=1;continue
   start,end=prev['front_last'],nxt['rear_first'];indices=list(range(start+1,end))
   # Match the older criterion of at least three 20 ms endpoint samples so
   # only resolution, not recovery selection, differs in the comparison.
   endpoints=[i for i in indices if i%16==15]
   if len(endpoints)<3:rejects['recovery_under_three_endpoints']+=1;continue
   knee=[];ankle=[]
   for i in indices:
    thigh=b[e,i,'crow_'+sd+'_thigh'];sh=b[e,i,'crow_'+sd+'_shank'];foot=b[e,i,'crow_'+sd+'_foot'];toe=b[e,i,'crow_'+sd+'_front_toe_2_proximal'];hip=point(thigh,[0,0,.031625]);k1=point(thigh,[0,0,-.031625]);k2=point(sh,[0,0,.031625]);an=point(sh,[0,0,-.031625]);an2=point(foot,[-.02875,0,.0092]);toeprox=point(toe,[-.009,0,0]);maxerr=max(maxerr,math.dist(k1,k2),math.dist(an,an2));knee.append(angle(hip,k1,an));ankle.append(angle(k1,an,toeprox))
   ki=min(range(len(indices)),key=lambda i:knee[i]);ai=min(range(len(indices)),key=lambda i:ankle[i]);delta=(ai-ki)*.00125
   records.append({'lane':e,'side':sd,'lag_rad':[0.,.3,.6,.9][e%4],'recovery_seconds':(end-start)*.00125,'knee_min_deg':knee[ki],'ankle_min_deg':ankle[ai],'ankle_minus_knee_minimum_seconds':delta,'order':'knee_first' if delta>0 else 'ankle_first' if delta<0 else 'same_substep','start':start,'end':end,'knee_min_index':indices[ki],'ankle_min_index':indices[ai]})
out={'records':records,'counts_by_lag':{str(lag):dict(collections.Counter(x['order'] for x in records if x['lag_rad']==lag)) for lag in (0.,.3,.6,.9)},'rejections':dict(rejects),'max_joint_anchor_error_m':maxerr,'scope':'1.25 ms synchronized model q and contact substeps, same event selection as endpoint analysis. Anatomical mapping, biological sample, surface and speed remain unvalidated. Angle minima are not a robust estimate of extension onset; discrete impulse/contact timing is not a measured EMG or video timing. No causal or biological qualification.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
