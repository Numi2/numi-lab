from pathlib import Path
from collections import defaultdict
import gzip,json,numpy as np
A=Path('artifacts/crow-v15-stance-kinematic-sensitivity-20261114');d=defaultdict(dict);lines=(A/'output.tsv').read_text().splitlines()
for line in lines[2:]:
 e,t,j,dx,*g=line.split();d[int(e),int(t),int(j)][float(dx)]=list(map(float,g))
poses=defaultdict(dict)
with gzip.open('artifacts/crow-v15-com-stance-balance-20261114/trace.log.gz','rt') as f:
 for line in f:
  z=line.split()
  if z and z[0]=='pose' and z[3].startswith('crow_right_') and ('foot' in z[3] or 'toe' in z[3]):poses[int(z[1]),int(z[2])][z[3]]=list(map(float,z[4:]))
def gap(obj,half):
 x,y,z,w=obj[3:];P=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]);Q=np.eye(3);delta=np.array([0.,0.,-.055])-np.array(obj[:3]);axes=[*P.T,*Q.T]+[np.cross(P[:,i],Q[:,j]) for i in range(3) for j in range(3)];return max(abs(delta@(u/np.linalg.norm(u)))-sum(abs(P.T@(u/np.linalg.norm(u)))*half)-sum(abs(Q.T@(u/np.linalg.norm(u)))*[.025,.72,.055]) for u in axes if np.linalg.norm(u)>1e-8)
rows=[];errors=[]
for e in (0,1):
 for t in (200,210,220,225,230,235,240):
  p=poses[e,t];other=[]
  for prefix,half in [('crow_right_foot',[.01,.01725,.0092]),('crow_right_front_toe',[.009,.0025,.0092]),('crow_right_rear_toe',[.009,.0025,.0092])]:
   objs=[v for k,v in p.items() if k==prefix or k.startswith(prefix)];assert objs;other.append(min(gap(v,half) for v in objs))
  base=d[e,t,17][0.];err=max(abs(a-b) for a,b in zip(base,other));errors.append(err)
  deriv={str(j):[(a-b)/.04 for a,b in zip(d[e,t,j][.02],d[e,t,j][-.02])] for j in (17,18,19)}
  rows.append({'lane':e,'tick':t,'base_gaps_m':base,'derivatives_m_per_rad_hip_knee_ankle':deriv,'independent_gap_error_m':err})
out={'hashes':lines[:2],'geometry_crosscheck_max_error_m':max(errors),'geometry_crosscheck_pass':bool(max(errors)<1e-6),'rows':rows,'scope':'fixed-state finite differences; no dynamics or contacts'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
