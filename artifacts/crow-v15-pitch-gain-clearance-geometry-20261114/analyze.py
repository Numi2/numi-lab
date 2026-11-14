from pathlib import Path
from collections import defaultdict
import gzip,json,numpy as np
A=Path('artifacts/crow-v15-pitch-gain-clearance-geometry-20261114');data=defaultdict(dict);poses=defaultdict(dict)
for line in (A/'output.tsv').read_text().splitlines():
 e,t,j,dx,*v=line.split();data[int(e),int(t),int(j)][float(dx)]=list(map(float,v))
with gzip.open('artifacts/crow-v15-pitch-gain-clearance-20261114/trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if x and x[0]=='pose' and int(x[2]) in (270,275,279,280) and x[3].startswith('crow_left_') and ('toe' in x[3] or 'foot' in x[3]):poses[int(x[1]),int(x[2])][x[3]]=list(map(float,x[4:]))
def gap(obj,half):
 x,y,z,w=obj[3:];P=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]);Q=np.eye(3);d=np.array([0.,0.,-.055])-np.array(obj[:3]);axes=[*P.T,*Q.T]+[np.cross(P[:,i],Q[:,j]) for i in range(3) for j in range(3)];return max(abs(d@(u/np.linalg.norm(u)))-sum(abs(P.T@(u/np.linalg.norm(u)))*half)-sum(abs(Q.T@(u/np.linalg.norm(u)))*[.025,.72,.055]) for u in axes if np.linalg.norm(u)>1e-8)
rows=[];errors=[]
for e in (0,1):
 for t in (270,275,279,280):
  base=data[e,t,14][0.];deriv={str(j):[(a-b)/.04 for a,b in zip(data[e,t,j][.02],data[e,t,j][-.02])] for j in (14,15,16)};p=poses[e,t];g=[]
  for prefix,half in (('crow_left_foot',[.01,.01725,.0092]),('crow_left_front_toe',[.009,.0025,.0092]),('crow_left_rear_toe',[.009,.0025,.0092])):
   names=[n for n in p if n==prefix or n.startswith(prefix)];assert names;g.append(min(gap(p[n],half) for n in names))
  err=max(abs(a-b) for a,b in zip(base,g));errors.append(err);rows.append({'lane':e,'tick':t,'base_gaps_m':base,'gap_derivative_m_per_rad_hip_knee_ankle':deriv,'geometry_crosscheck_max_error_m':err})
out={'binary_hashes':(A/'hashes.txt').read_text().splitlines(),'max_crosscheck_error_m':max(errors),'crosscheck_pass':bool(max(errors)<1e-6),'rows':rows,'scope':'Fixed-state hypothetical geometry, not dynamic contact, reachable motion or stability.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
