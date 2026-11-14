from pathlib import Path
import collections,gzip,json,math,numpy as np
A=Path('artifacts/crow-v15-supported-return-gate-20261114');state={};pose={};events={};by=collections.defaultdict(lambda:[0.]*37);coverage=collections.Counter();bad=[];errors=[]
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  a=l.split()
  if not a:continue
  if a[0]=='step':
   e,t=int(a[1]),int(a[2])+1;qi=a.index('q');vi=a.index('v');state[e,t]={'q':list(map(float,a[qi+1:vi])),'v':list(map(float,a[vi+1:]))}
  elif a[0]=='pose':pose.setdefault((int(a[1]),int(a[2])+1),{})[a[3]]=list(map(float,a[4:]))
  elif a[0]=='event':events[int(a[1]),int(a[2])+1]={'release':bool(int(a[3])),'support_streak':int(a[4]),'finished':bool(int(a[11])),'clear_count':int(a[12]),'sensor_clearance':float(a[13])}
with gzip.open(A/'substeps.log.gz','rt') as f:
 for l in f:
  a=l.split()
  if a and a[0] in ('CROW_SUBSTEP_ERROR','failure'):errors.append(l.strip())
  if a and a[0]=='CROW_TWO_TOE_SUBSTEP':
   t,sub,e=map(int,a[1:4]);key=e,t,sub;coverage[key]+=1
   if int(a[4])>int(a[5]) or len(a[6:])!=37:bad.append(key)
   for i,v in enumerate(map(float,a[6:])):by[e,t][i]+=v
missing=[(e,t,i) for e in range(2) for t in range(1,451) for i in range(32) if coverage[e,t,i]!=1]
def rot(q):
 x,y,z,w=q;return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def gap(obj,half):
 P=rot(obj[3:]);Q=np.eye(3);d=np.array([0.,0.,-.055])-np.array(obj[:3]);axes=[*P.T,*Q.T]+[np.cross(P[:,i],Q[:,j]) for i in range(3) for j in range(3)];v=[]
 for u in axes:
  n=np.linalg.norm(u)
  if n>1e-8:
   u=u/n;v.append(abs(d@u)-sum(abs(P.T@u)*half)-sum(abs(Q.T@u)*[.025,.72,.055]))
 return float(max(v))
def longest(a):
 best=cur=0
 for v in a:cur=cur+1 if v else 0;best=max(best,cur)
 return best
rows=[]
for e in range(2):
 sensor_errors=[];trigger=next((t for t in range(1,451) if events.get((e,t),{}).get('release')),None);finish=next((t for t in range(1,451) if events.get((e,t),{}).get('finished')),None);qual=[];grip=[];maxclear=-float('inf');fall=None;peak_right_up=0.;nonfoot=0;missingposes=[]
 for t in range(1,451):
  x=state.get((e,t));p=pose.get((e,t),{});v=by[e,t];release=trigger is not None and t>=trigger
  if not x:qual.append(False);grip.append(False);continue
  q=x['q'];u=x['v'];
  if fall is None and q[2]<.1:fall=t
  if fall is not None:qual.append(False);grip.append(False);continue
  tilt=math.acos(max(-1.,min(1.,1.-2*(q[3]**2+q[4]**2))));speed=math.sqrt(sum(z*z for z in u[:3]));names=[n for n in p if n.startswith('crow_left_') and 'toe' in n];has=len(names)==8 and 'crow_left_foot' in p
  if release and not has:missingposes.append(t)
  clear=min([gap(p[n],[.009,.0025,.0092]) for n in names]+([gap(p['crow_left_foot'],[.01,.01725,.0092])] if 'crow_left_foot' in p else [])) if has else -float('inf')
  if has and (e,t) in events:sensor_errors.append(abs(clear-events[e,t]['sensor_clearance']))
  if release:maxclear=max(maxclear,clear)
  rightopp=sum(v[21+8+i] for i in range(6))>=.0002 and sum(v[21+8+i] for i in (6,7))<=-.0002
  ok=release and has and clear>=.001 and abs(v[2])<=1e-7 and v[1]>=.5*.45004*9.81*.02 and rightopp and abs(v[4])<=1e-7 and tilt<=.35 and speed<=.12
  qual.append(ok);leftopp=sum(v[21+i] for i in range(6))>=.0002 and sum(v[21+i] for i in (6,7))<=-.0002;grip.append(finish is not None and t>=finish and min(v[0],v[1])>.0002 and v[0]+v[1]>=.5*.45004*9.81*.02 and leftopp and rightopp and abs(v[4])<=1e-7 and tilt<=.35 and speed<=.12);peak_right_up=max(peak_right_up,v[1]);nonfoot+=abs(v[4])>1e-7
 rows.append({'states_observed':sum((e,t) in state for t in range(1,451)),'lane':e,'stance_knee_target_offset_rad':.075,'initial_x_shift_m':.002,'sensor_clearance_max_abs_error_m':max(sensor_errors) if sensor_errors else None,'release_trigger_step':trigger,'return_step':finish,'terminal_100_bilateral_supported_grip':all(grip[-100:]),'longest_qualified_release_intervals':longest(qual),'peak_minimum_left_clearance_m':maxclear if maxclear!=-float('inf') else None,'maximum_right_upward_impulse_per_interval':peak_right_up,'nonfoot_impulse_intervals':nonfoot,'first_root_below_0_1m':fall,'missing_pose_steps':missingposes,'pass':len(sensor_errors)==450 and max(sensor_errors)<=3e-6 and trigger is not None and finish is not None and longest(qual)>=100 and all(grip[-100:]) and fall is None and not nonfoot and not missingposes and all((e,t) in state for t in range(1,451)) and not errors})
out={'lanes':rows,'missing_substeps':len(missing),'bad_substeps':len(bad),'errors':errors,'scope':'constructed stationary grip-to-release development test'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
