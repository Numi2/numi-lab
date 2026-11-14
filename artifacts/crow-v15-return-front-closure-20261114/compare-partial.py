from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-return-front-closure-20261114');B=Path('artifacts/crow-v15-stance-front-com-balance-20261114')
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
s,u=read(A);ref,_=read(B);common=sorted(set(t for e,t in s if e==0 and (1,t) in s));on=next((t for t in common if u.get((0,t))!=u.get((1,t))),None);pre=[max(abs(a-b) for a,b in zip(s[0,t],s[1,t])) for t in common if on is not None and t<on];base=[max(abs(a-b) for a,b in zip(s[0,t],ref[1,t])) for t in common if (1,t) in ref]
o={'common_states':len(common),'last_zero_based':max(common),'first_control_difference_zero_based':on,'prefix_count':len(pre),'prefix_exact':on is not None and len(pre)==on and max(pre,default=1)==0,'control_reference_count':len(base),'control_reference_exact':len(base)==len(common) and max(base,default=1)==0,'note':'Partial run only; no complete 450 state parity.'};(A/'comparison-partial.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
