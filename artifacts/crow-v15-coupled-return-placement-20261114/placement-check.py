from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-coupled-return-placement-20261114');d={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if x and x[0]=='placement':d[int(x[1]),int(x[2])]=float(x[3])
rows=[]
for e in (0,1):
 v=[d[e,t] for t in range(450) if (e,t) in d];j=[b-a for a,b in zip(v,v[1:])]
 rows.append({'lane':e,'samples':len(v),'min':min(v,default=None),'max':max(v,default=None),'max_increment':max(j,default=None),'nondecreasing':all(x>=-1e-7 for x in j),'within_registered_cap':len(v)==450 and max(v)<=([1.,.2][e]+1e-6) and max(j,default=1)<=.050001})
o={'rows':rows,'pass':all(x['within_registered_cap'] and x['nondecreasing'] for x in rows)};(A/'placement-check.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
