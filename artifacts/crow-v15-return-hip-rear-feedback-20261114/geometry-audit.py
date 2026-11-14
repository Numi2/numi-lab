from pathlib import Path
import gzip,json,math,numpy as np
A=Path('artifacts/crow-v15-return-hip-rear-feedback-20261114');R=A.parent
cases={'control':('crow-v15-return-hip-rear-feedback-20261114',0),'placement':('crow-v15-return-hip-rear-feedback-20261114',1)}
def gap(obj,half):
 x,y,z,w=obj[3:];P=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]);Q=np.eye(3);d=np.array([0.,0.,-.055])-np.array(obj[:3]);axes=[*P.T,*Q.T]+[np.cross(P[:,i],Q[:,j]) for i in range(3) for j in range(3)];return max(abs(d@(u/np.linalg.norm(u)))-sum(abs(P.T@(u/np.linalg.norm(u)))*half)-sum(abs(Q.T@(u/np.linalg.norm(u)))*[.025,.72,.055]) for u in axes if np.linalg.norm(u)>1e-8)
out={}
for name,(folder,e) in cases.items():
 poses={};events={};contacts={};states={}
 with gzip.open(R/folder/'trace.log.gz','rt') as f:
  for line in f:
   x=line.split()
   if not x or x[0] not in ('pose','event','contact','step') or int(x[1])!=e:continue
   t=int(x[2])
   if x[0]=='pose' and x[3].startswith('crow_left_') and ('toe' in x[3] or 'foot' in x[3]):poses.setdefault(t,{})[x[3]]=list(map(float,x[4:]))
   elif x[0]=='event':events[t]=list(map(float,x[3:]))
   elif x[0]=='contact' and ('crow_left_' in x[3] or 'crow_left_' in x[4]):
    n=x[3] if x[3].startswith('crow_left_') else x[4];contacts.setdefault(t,{}).setdefault(n,0.);contacts[t][n]+=float(x[5])
   elif x[0]=='step':q=x.index('q');states[t]=float(x[q+3])
 fall=next((t for t in sorted(states) if states[t]<.1),None);table=[]
 for t in (157,198,208,209,212,220,230,240,250,270,290,310):
  if t not in poses or (fall is not None and t>=fall):continue
  p=poses[t];groups={}
  for group,prefix in (('foot','crow_left_foot'),('front','crow_left_front_toe'),('rear','crow_left_rear_toe')):
   names=[n for n in p if n==prefix or n.startswith(prefix)];half=[.01,.01725,.0092] if group=='foot' else [.009,.0025,.0092];g=[gap(p[n],half) for n in names]
   groups[group]={'min_box_gap_m':min(g),'max_box_gap_m':max(g),'centers_x_m':[p[n][0] for n in sorted(names)],'last_substep_impulse_Ns':sum(contacts.get(t,{}).get(n,0.) for n in names)}
  table.append({'zero_based_tick':t,'left_front_x_impulse_Ns':events[t][4],'left_rear_x_impulse_Ns':events[t][5],'groups':groups})
 post=[t for t in sorted(events) if t>=209 and (fall is None or t<fall)];positive=[t for t in post if events[t][4]>.000001];out[name]={'source':folder,'lane':e,'first_root_below_0_1m_zero_based':fall,'postreturn_ticks':len(post),'postreturn_ticks_with_positive_front_normal':len(positive),'first_positive_front_normal_ticks':positive[:10],'samples':table}
(A/'geometry-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
