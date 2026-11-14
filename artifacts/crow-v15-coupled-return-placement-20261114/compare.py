from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-coupled-return-placement-20261114');B=Path('artifacts/crow-v15-return-toe-closure-rate-20261113')
def read(p):
 s={};targets={}
 with gzip.open(p/'trace.log.gz','rt') as f:
  for line in f:
   x=line.split()
   if not x:continue
   if x[0]=='step':
    q=x.index('q');v=x.index('v');s[int(x[1]),int(x[2])]=list(map(float,x[q+1:v]+x[v+1:]))
   elif x[0]=='hipcommand':targets[int(x[1]),int(x[2]),14]=float(x[3])
   elif x[0]=='jointtarget' and x[3]=='16':targets[int(x[1]),int(x[2]),16]=float(x[4])
 return s,targets
s,u=read(A);ref,v=read(B);on=next((t for t in range(450) if any(abs(u[0,t,j]-u[1,t,j])>1e-8 for j in (14,16))),None)
pre=[max(abs(a-b) for a,b in zip(s[0,t],s[1,t])) for t in range(on or 0) if (0,t) in s and (1,t) in s];base=[max(abs(a-b) for a,b in zip(s[0,t],ref[0,t])) for t in range(450) if (0,t) in s and (0,t) in ref]
o={'first_joint_target_difference_zero_based':on,'prefix_pairs':len(pre),'prefix_exact':on is not None and len(pre)==on and max(pre,default=1)==0,'archived_baseline_pairs':len(base),'archived_baseline_exact':len(base)==450 and max(base,default=1)==0,'target_ranges':{str(j):[min(u[e,t,j] for e in (0,1) for t in range(450)),max(u[e,t,j] for e in (0,1) for t in range(450))] for j in (14,16)},'note':'Common COM-balance, contact and placement terms are state-dependent; the intervention increments are bounded in source.'}
(A/'comparison.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
