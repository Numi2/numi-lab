from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-return-hip-rear-feedback-20261114');B=Path('artifacts/crow-v15-stance-front-com-balance-20261114')
def read(p):
 s={};u={}
 with gzip.open(p/'trace.log.gz','rt') as f:
  for l in f:
   x=l.split()
   if not x:continue
   if x[0]=='step':
    q=x.index('q');v=x.index('v');s[int(x[1]),int(x[2])]=list(map(float,x[q+1:v]+x[v+1:]))
   elif x[0]=='control':u[int(x[1]),int(x[2])]=list(map(float,x[3:]))
 return s,u
s,u=read(A);ref,_=read(B);on=next((t for t in range(450) if u[0,t]!=u[1,t]),None);pre=[max(abs(a-b) for a,b in zip(s[0,t],s[1,t])) for t in range(on or 0)];baseline=[max(abs(a-b) for a,b in zip(s[0,t],ref[1,t])) for t in range(450)]
o={'first_control_difference_zero_based':on,'prefix_pairs':len(pre),'prefix_exact':on is not None and max(pre,default=1)==0,'archived_baseline_pairs':len(baseline),'archived_baseline_exact':len(baseline)==450 and max(baseline)==0};(A/'comparison.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
