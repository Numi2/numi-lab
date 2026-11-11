from pathlib import Path
import collections,json,math,numpy as np
A=Path('artifacts/crow-v15-two-segment-recentered-grip-20261111');R=Path('artifacts/crow-v15-two-segment-contact-admission-20261111');state={};pose={};control={};ref={}
for l in (A/'trace.log').read_text().splitlines():
 a=l.split()
 if not a:continue
 if a[0]=='step':
  qi=a.index('q');vi=a.index('v');state[int(a[1])+1]={'q':list(map(float,a[qi+1:vi])),'v':list(map(float,a[vi+1:]))}
 elif a[0]=='pose':pose.setdefault(int(a[1])+1,{})[a[2]]=list(map(float,a[3:]))
 elif a[0]=='control':control[int(a[1])+1]=list(map(float,a[2:]))
for l in (R/'trace.log').read_text().splitlines():
 a=l.split()
 if a and a[0]=='step':
  qi=a.index('q');vi=a.index('v');ref[int(a[1])+1]={'q':list(map(float,a[qi+1:vi])),'v':list(map(float,a[vi+1:]))}
records={};errors=[]
for l in (A/'substeps.log').read_text().splitlines():
 a=l.split()
 if a and (a[0]=='CROW_SUBSTEP_ERROR' or a[0]=='failure'):errors.append(l)
 if a and a[0]=='CROW_TWO_TOE_SUBSTEP':records[tuple(map(int,a[1:4]))]={'required':int(a[4]),'capacity':int(a[5]),'v':list(map(float,a[6:]))}
end=len(state);missing=[(t,i) for t in range(1,end+1) for i in range(16) if (t,i,0) not in records];bad=[k for k,v in records.items() if v['required']>v['capacity'] or len(v['v'])!=37];diff=[t for t in range(1,11) if state.get(t)!=ref.get(t)];by=collections.defaultdict(lambda:[0.]*37)
for (t,sub,lane),r in records.items():
 if lane==0:
  for i,v in enumerate(r['v']):by[t][i]+=v
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
support=[];opp=[];qual=[];gaps=[];rows=[]
for t in range(1,end+1):
 v=by[t];q=state[t]['q'];u=state[t]['v'];tilt=math.acos(max(-1.,min(1.,1.-2*(q[3]**2+q[4]**2))));speed=math.sqrt(sum(x*x for x in u[:3]));s=min(v[0],v[1])>.0002 and sum(v[:2])>=.5*.45004*9.81*.02 and v[4]<=1e-7 and tilt<=.35 and speed<=.12;o=all(sum(v[21+base+i] for i in range(6))>=.0002 and sum(v[21+base+i] for i in (6,7))<=-.0002 for base in (0,8));support.append(s);opp.append(o);qual.append(s and o)
 if o:
  pp=pose.get(t,{});gaps.extend(gap(x) for x in pp.values());rows.append({'step':t,'toe_count':len(pp),'minimum_gap_m':min((gap(x) for x in pp.values()),default=None)})
out={'steps':end,'different_initial_x_shift_m':-0.010,'observed_records':len(records),'missing':missing,'bad':bad,'errors':errors,'support_intervals':sum(support),'longest_supported_intervals':longest(support),'opposed_intervals':sum(opp),'supported_opposed_intervals':sum(qual),'longest_supported_opposed_intervals':longest(qual),'minimum_opposed_endpoint_gap_m':min(gaps) if gaps else None,'opposed_steps':rows,'first_front_activation':[next((t for t,x in sorted(control.items()) if x[i]>0),None) for i in (1,4)],'last_root_q':state[end]['q'][:7] if end else None,'last_root_v':state[end]['v'][:6] if end else None,'scope':'single initialized stationary perch screen; endpoint OBB overlaps and substep impulses, no continuous geometry, release, flight, route or biological qualification'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('missing','bad','opposed_steps')},indent=2));print('missing',len(missing),'bad',len(bad))
