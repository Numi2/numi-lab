from pathlib import Path
import gzip,json,numpy as np,hashlib
A=Path('artifacts/crow-v15-rear-toe-geometry-audit-20261111')
folders=['crow-v15-matched-balance-grip-screen-20261111','crow-v15-rear-grip-reacquisition-20261111']
def rot(q):
 x,y,z,w=q;return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def gap(toe,ground):
 a=np.array(toe[:3]);Q=rot(toe[3:]);P=rot(ground[3:]);b=np.array(ground[:3])+P@np.array([0,0,-.055]);d=b-a;axes=[*Q.T,*P.T]+[np.cross(Q[:,i],P[:,j]) for i in range(3) for j in range(3)];g=[]
 for u in axes:
  n=np.linalg.norm(u)
  if n>1e-8:
   u=u/n;g.append(abs(d@u)-sum(abs(Q.T@u)*[.018,.0025,.0092])-sum(abs(P.T@u)*[.025,.72,.055]))
 return float(max(g))
out={}
for folder in folders:
 f=Path('artifacts')/folder/'results.txt.gz';poses={};ground=None
 with gzip.open(f,'rt') as stream:
  for l in stream:
   v=l.split()
   if not v:continue
   if v[0]=='groundpose':ground=list(map(float,v[3:]))
   if v[0]=='pose' and 'rear_toe' in v[4] and int(v[3]) in (80,100,120,140,160,180,200,220):poses[int(v[3]),v[4]]=list(map(float,v[5:]))
 assert ground is not None and len(poses)==16
 out[folder]={'input_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'samples':[{'step':t,'toe':name,'x_m':p[0],'z_m':p[2],'sat_gap_m':gap(p,ground)} for (t,name),p in sorted(poses.items())]}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
