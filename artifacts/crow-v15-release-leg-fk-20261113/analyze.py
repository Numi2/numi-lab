from pathlib import Path
import gzip,json,math
A=Path(__file__).resolve().parent;rows={};poses={};states={}
for line in (A/'results.tsv').read_text().splitlines():
 x=line.split();k=tuple(map(int,x[:3]));assert k not in rows;rows[k]=list(map(float,x[3:]));assert len(rows[k])==9 and all(map(math.isfinite,rows[k]))
with gzip.open(A.parent/'crow-v15-perch-weight-transfer-strong-20261113/trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if x and x[0]=='pose' and x[3] in ('crow_left_foot','crow_right_foot'):poses[int(x[1]),int(x[2]),x[3]]=list(map(float,x[4:7]))
  elif x and x[0]=='step':
   q=x.index('q');v=x.index('v');states[int(x[1]),int(x[2])]=list(map(float,x[q+1:v]))
keys={(a,b) for a,b,c in rows if c==0};assert len(keys)==643 and len(rows)==7*len(keys)
errs=[]
for e,t in keys:
 for name,ofs in [('crow_left_foot',3),('crow_right_foot',6)]:errs.extend(abs(a-b) for a,b in zip(rows[e,t,0][ofs:ofs+3],poses[e,t,name]))
assert max(errs)<1e-6
out={'sample_count':len(keys),'foot_coordinate_checks':len(errs),'foot_max_abs_error_m':max(errs),'lanes':{}}
for e,steps in [(0,[120,130,140,150,160,180]),(1,[145,155,165,175,185,205])]:
 d={}
 for t in steps:
  x=rows[e,t,0];q=states[e,t];p={'q_hip_knee_ankle_rad':q[14:17],'left_foot_minus_com_m':[x[i+3]-x[i] for i in range(3)]}
  for name,a,b in [('hip',1,2),('knee',3,4),('ankle',5,6)]:
   plus,minus=rows[e,t,a],rows[e,t,b];p[name+'_d_foot_minus_com_m_per_rad']=[((plus[i+3]-plus[i])-(minus[i+3]-minus[i]))/.02 for i in range(3)]
  d[str(t)]=p
 out['lanes'][str(e)]=d
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
