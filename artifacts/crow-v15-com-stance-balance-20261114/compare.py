from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-com-stance-balance-20261114')
B=Path('artifacts/crow-v15-return-toe-closure-rate-20261113')
def read(p):
 states={}; targets={}
 with gzip.open(p/'trace.log.gz','rt') as f:
  for line in f:
   x=line.split()
   if not x:continue
   if x[0]=='step':
    q=x.index('q');v=x.index('v');states[int(x[1]),int(x[2])]=list(map(float,x[q+1:v]+x[v+1:]))
   elif x[0]=='rightHipCommand':targets[int(x[1]),int(x[2])]=float(x[3])
 return states,targets
s,u=read(A);ref,v=read(B)
on=next((t for t in range(450) if (0,t) in u and (1,t) in u and abs(u[0,t]-u[1,t])>1e-8),None)
pre=[max(abs(a-b) for a,b in zip(s[0,t],s[1,t])) for t in range(on or 0) if (0,t) in s and (1,t) in s]
baseline=[max(abs(a-b) for a,b in zip(s[0,t],ref[0,t])) for t in range(450) if (0,t) in s and (0,t) in ref]
steps=[abs(u[e,t]-u[e,t-1]) for e in (0,1) for t in range(1,450) if (e,t) in u and (e,t-1) in u]
o={'first_hip_target_difference_zero_based':on,'prefix_pairs':len(pre),'prefix_exact':on is not None and len(pre)==on and max(pre,default=1)==0,'archived_baseline_pairs':len(baseline),'archived_baseline_exact':len(baseline)==450 and max(baseline,default=1)==0,'max_hip_target_step_rad':max(steps,default=None),'note':'Hip command contains both pitch correction and common COM balance; correction rate is bounded in source.'}
(A/'comparison.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
