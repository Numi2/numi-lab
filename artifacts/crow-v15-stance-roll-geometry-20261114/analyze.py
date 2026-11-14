from pathlib import Path
from collections import defaultdict
import gzip,json
A=Path(__file__).resolve().parent;data=defaultdict(dict);bal={}
for l in (A/'output.tsv').read_text().splitlines():
 e,t,j,dx,*v=l.split();data[int(e),int(t),int(j)][float(dx)]=list(map(float,v))
with gzip.open(A.parent/'crow-v15-combined-hip28-pitch14-20261114/trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if x and x[0]=='balance':bal[int(x[1]),int(x[2])]=list(map(float,x[3:]))
rows=[];errors=[]
for e in (0,1):
 for t in (250,260,270,280,290,300,315):
  base=data[e,t,40][0.]; logged=bal[e,t+1];err=max(abs(base[0]-logged[2]),abs(base[1]-logged[3]));errors.append(err)
  rows.append({'lane':e,'tick':t,'com_relative_stance_foot_m':base,'derivative_m_per_rad':{str(j):[(a-b)/.04 for a,b in zip(data[e,t,j][.02],data[e,t,j][-.02])] for j in (39,40)},'crosscheck_xy_error_m':err})
out={'binary_hashes':(A/'hashes.txt').read_text().splitlines(),'max_xy_crosscheck_error_m':max(errors),'crosscheck_pass':max(errors)<1e-6,'rows':rows,'scope':'fixed-root hypothetical kinematics only; does not predict contact-constrained dynamics'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
