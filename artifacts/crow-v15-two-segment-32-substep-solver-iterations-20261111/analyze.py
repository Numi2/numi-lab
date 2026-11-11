from pathlib import Path
import collections,gzip,json,math,numpy as np
A=Path('artifacts/crow-v15-two-segment-32-substep-solver-iterations-20261111');R=Path('artifacts/crow-v15-two-segment-grip-screen-20261111');state={};pose={};controls={};ref={};refc={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  a=l.split()
  if not a:continue
  if a[0]=='step':
   e=int(a[1]);t=int(a[2])+1;qi=a.index('q');vi=a.index('v');state[e,t]={'q':list(map(float,a[qi+1:vi])),'v':list(map(float,a[vi+1:]))}
  elif a[0]=='pose':pose.setdefault((int(a[1]),int(a[2])+1),{})[a[3]]=list(map(float,a[4:]))
  elif a[0]=='control':controls[int(a[1]),int(a[2])+1]=list(map(float,a[3:]))
for l in (R/'trace.log').read_text().splitlines():
 a=l.split()
 if not a:continue
 if a[0]=='step':
  t=int(a[1])+1;qi=a.index('q');vi=a.index('v');ref[t]={'q':list(map(float,a[qi+1:vi])),'v':list(map(float,a[vi+1:]))}
 elif a[0]=='control':refc[int(a[1])+1]=list(map(float,a[2:]))
by=collections.defaultdict(lambda:[0.]*37);coverage=collections.Counter();bad=[];errors=[]
with gzip.open(A/'substeps.log.gz','rt') as f:
 for l in f:
  a=l.split()
  if a and (a[0]=='CROW_SUBSTEP_ERROR' or a[0]=='failure'):errors.append(l.strip())
  if a and a[0]=='CROW_TWO_TOE_SUBSTEP':
   t,sub,e=map(int,a[1:4]);key=(e,t,sub);coverage[key]+=1
   if int(a[4])>int(a[5]) or len(a[6:])!=37:bad.append(key)
   for i,v in enumerate(map(float,a[6:])):by[e,t][i]+=v
missing=[(e,t,i) for e in range(8) for t in range(1,351) for i in range(32) if coverage[e,t,i]!=1];diff=[t for t in range(1,351) if state.get((0,t))!=ref.get(t) or controls.get((0,t))!=refc.get(t)]
def rot(q):
 x,y,z,w=q;return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def gap(toe):
 P=rot(toe[3:]);Q=np.eye(3);d=np.array([0.,0.,-.055])-np.array(toe[:3]);axes=[*P.T,*Q.T]+[np.cross(P[:,i],Q[:,j]) for i in range(3) for j in range(3)];v=[]
 for u in axes:
  n=np.linalg.norm(u)
  if n>1e-8:
   u=u/n;v.append(abs(d@u)-sum(abs(P.T@u)*[.009,.0025,.0092])-sum(abs(Q.T@u)*[.025,.72,.055]))
 return float(max(v))
def longest(a):
 best=cur=0
 for v in a:cur=cur+1 if v else 0;best=max(best,cur)
 return best
rows=[]
for e in range(8):
 support=[];opp=[];qual=[];gaps=[];poseMissing=[];fall=None;nonfoot=[]
 for t in range(1,351):
  x=state.get((e,t))
  if x is None:support.append(False);opp.append(False);qual.append(False);continue
  v=by[e,t];q=x['q'];u=x['v'];tilt=math.acos(max(-1.,min(1.,1.-2*(q[3]**2+q[4]**2))));speed=math.sqrt(sum(z*z for z in u[:3]));s=min(v[0],v[1])>.0002 and sum(v[:2])>=.5*.45004*9.81*.02 and v[4]<=1e-7 and tilt<=.35 and speed<=.12;o=all(sum(v[21+base+i] for i in range(6))>=.0002 and sum(v[21+base+i] for i in (6,7))<=-.0002 for base in (0,8));support.append(s);opp.append(o);qual.append(s and o)
  if v[4]>1e-7:nonfoot.append(t)
  if fall is None and q[2]<.1:fall=t
  if o:
   p=pose.get((e,t),{})
   if len(p)!=16:poseMissing.append(t)
   gaps.extend(gap(z) for z in p.values())
 g=min(gaps) if gaps else None
 rows.append({'lane':e,'initial_x_shift_m':0.,'toe_flexion_cap_rad':.55 if e==0 else 1.05,'hip_balance_gain':0. if e==0 else 1.,'velocity_damping_seconds':.2,'rear_toe_cap_rad':[.55,1.05,.65,.75,.85,.95,1.15,1.2][e],'states':sum((e,t) in state for t in range(1,351)),'support_intervals':sum(support),'longest_supported_intervals':longest(support),'opposed_intervals':sum(opp),'longest_supported_opposed_intervals':longest(qual),'terminal_100_supported_opposed':all(qual[-100:]),'minimum_opposed_endpoint_gap_m':g,'opposed_pose_missing_steps':poseMissing,'first_root_below_0_1m':fall,'nonfoot_impulse_intervals':len(nonfoot),'first_front_activation':[next((t for t in range(1,351) if controls.get((e,t),[0]*6)[i]>0),None) for i in (1,4)],'pass':longest(qual)>=50 and all(qual[-100:]) and fall is None and g is not None and g>=-.0005 and not poseMissing and not nonfoot})
out={'observed_records':len(coverage),'missing_count':len(missing),'bad_count':len(bad),'errors':errors,'lane_zero_difference_from_old_solver_control':len(diff),'first_difference_from_old_solver_control':diff[:1],'lanes':rows,'scope':'constructed development stationary starts; no flight arrival, release, continuous geometry or biological qualification'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
