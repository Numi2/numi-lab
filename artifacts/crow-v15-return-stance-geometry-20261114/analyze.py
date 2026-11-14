from pathlib import Path
import gzip,json,math,numpy as np
A=Path('artifacts/crow-v15-return-stance-geometry-20261114');P=Path('artifacts/crow-v15-pitch-normal-support-20261114/trace.log.gz');poses={};steps={295,300,302,304,306}
with gzip.open(P,'rt') as f:
 for line in f:
  z=line.split()
  if z and z[0]=='pose' and z[1]=='0' and int(z[2]) in steps and (z[3]=='crow_right_foot' or z[3].startswith('crow_right_front_toe') or z[3].startswith('crow_right_rear_toe')):poses.setdefault(int(z[2]),{})[z[3]]=list(map(float,z[4:]))
def gap(o,half):
 x,y,z,w=o[3:];p=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]);d=np.array([0.,0.,-.055])-np.array(o[:3]);q=np.eye(3);axes=[*p.T,*q.T]+[np.cross(p[:,i],q[:,j]) for i in range(3) for j in range(3)];return float(max(abs(d@(u/np.linalg.norm(u)))-sum(abs(p.T@(u/np.linalg.norm(u)))*half)-sum(abs(q.T@(u/np.linalg.norm(u)))*[.025,.72,.055]) for u in axes if np.linalg.norm(u)>1e-8))
ref={}
for t,p in poses.items():
 front=[v for k,v in p.items() if 'front_toe' in k];rear=[v for k,v in p.items() if 'rear_toe' in k];assert len(front)==6 and len(rear)==2
 ref[t]={'foot':gap(p['crow_right_foot'],[.01,.01725,.0092]),'front':min(gap(v,[.009,.0025,.0092]) for v in front),'rear':min(gap(v,[.009,.0025,.0092]) for v in rear)}
assert set(ref)==steps
out={}
with gzip.open(A/'output.tsv.gz','rt') as f:
 for line in f:
  lane,t,dh,dk,da,foot,front,rear=map(float,line.split());t=int(t);o=out.setdefault(t,{'count':0,'zero':None,'smallest_contact_pose':None});o['count']+=1;d={'hip_rad':dh,'knee_rad':dk,'ankle_rad':da,'foot_m':foot,'front_m':front,'rear_m':rear,'norm_rad':math.sqrt(dh*dh+dk*dk+da*da)}
  if dh==dk==da==0:o['zero']=d
  if -.0006<=front<=0 and -.0006<=rear<=0 and foot>=-.0002 and (o['smallest_contact_pose'] is None or d['norm_rad']<o['smallest_contact_pose']['norm_rad']):o['smallest_contact_pose']=d
for t,o in out.items():
 assert o['zero'] is not None
 o['zero_gap_error_m']={g:abs(o['zero'][g+'_m']-ref[t][g]) for g in ('foot','front','rear')};assert max(o['zero_gap_error_m'].values())<1e-6
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
