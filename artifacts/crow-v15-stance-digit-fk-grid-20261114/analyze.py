from pathlib import Path
import collections,gzip,json,math,numpy as np
A=Path('artifacts/crow-v15-stance-digit-fk-grid-20261114');T=Path('artifacts/crow-v15-contact-transfer-no-placement-20261114/trace.log.gz');poses={};rows=collections.defaultdict(list)
with gzip.open(T,'rt') as f:
 for l in f:
  x=l.split()
  if x and x[0]=='pose' and int(x[1])==0 and int(x[2]) in (290,295,297,300,305,310) and x[3].startswith('crow_right_') and ('foot' in x[3] or 'toe' in x[3]):poses.setdefault(int(x[2]),{})[x[3]]=list(map(float,x[4:]))
def gap(obj,half):
 x,y,z,w=obj[3:];P=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]);d=np.array([0.,0.,-.055])-np.array(obj[:3]);axes=[*P.T,*np.eye(3).T]+[np.cross(P[:,i],np.eye(3)[:,j]) for i in range(3) for j in range(3)];return max(abs(d@(u/np.linalg.norm(u)))-sum(abs(P.T@(u/np.linalg.norm(u)))*half)-sum(abs(np.eye(3).T@(u/np.linalg.norm(u)))*[.025,.72,.055]) for u in axes if np.linalg.norm(u)>1e-8)
with (A/'output.tsv').open() as f:
 for line in f:
  z=line.split();lane,t=int(z[0]),int(z[1]);dh,dd,_,foot,front,rear=map(float,z[2:]);rows[t].append({'proximal_delta_rad':dh,'distal_delta_rad':dd,'foot_gap_m':foot,'front_gap_m':front,'rear_gap_m':rear})
out=[]
for t in (290,295,297,300,305,310):
 p=poses[t];groups={'foot':['crow_right_foot'],'front':[n for n in p if n.startswith('crow_right_front_toe')],'rear':[n for n in p if n.startswith('crow_right_rear_toe')]};assert len(groups['front'])==6 and len(groups['rear'])==2
 observed={k:min(gap(p[n],[.01,.01725,.0092] if k=='foot' else [.009,.0025,.0092]) for n in names) for k,names in groups.items()};zero=next(x for x in rows[t] if x['proximal_delta_rad']==0 and x['distal_delta_rad']==0);err=max(abs(observed[k]-zero[k+'_gap_m']) for k in groups);candidates=[x for x in rows[t] if x['foot_gap_m']>=-.0002 and -.0006<=x['front_gap_m']<=0 and -.0006<=x['rear_gap_m']<=0];best=min(candidates,key=lambda x:math.hypot(x['proximal_delta_rad'],x['distal_delta_rad'])) if candidates else None
 out.append({'tick':t,'samples':len(rows[t]),'zero_pose_max_gap_error_m':err,'recorded_gaps_m':observed,'zero_offset_gaps_m':{k:zero[k+'_gap_m'] for k in groups},'eligible_count':len(candidates),'minimum_joint_change_candidate':best})
result={'rows':out,'mapping_check_pass':all(x['zero_pose_max_gap_error_m']<=1e-7 for x in out),'scope':'fixed-state oriented-box gap calculation only; no dynamics, forces, contact normal or biological qualification'};(A/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
