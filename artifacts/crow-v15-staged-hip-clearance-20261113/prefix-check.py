from pathlib import Path
import gzip,json
R=Path('artifacts');A=R/'crow-v15-staged-hip-clearance-20261113';B=R/'crow-v15-staged-perch-release-20261113'
def read(p):
 state={};lift={}
 with gzip.open(p/'trace.log.gz','rt') as f:
  for line in f:
   x=line.split()
   if x and x[0]=='step':
    q=x.index('q');v=x.index('v');state[int(x[1]),int(x[2])]=list(map(float,x[q+1:v]+x[v+1:]))
   elif x and x[0]=='stage':lift[int(x[1]),int(x[2])]=float(x[8])
 return state,lift
ref,rl=read(B);cur,cl=read(A);out={}
for e in (0,1):
 ts=sorted(t for lane,t in cur if lane==e);on=next((t for t in ts if cl[e,t]>0),None);refon=next((t for lane,t in sorted(rl) if lane==e and rl[e,t]>0),None);check=[t for t in ts if on is None or t<on];missing=[t for t in check if (e,t) not in ref];errs=[abs(a-b) for t in check if (e,t) in ref for a,b in zip(cur[e,t],ref[e,t])];out[str(e)]={'first_lift_zero_based':on,'reference_first_lift_zero_based':refon,'matched_pre_lift_states':len(check)-len(missing),'missing':missing,'max_abs_qv_error':max(errs,default=None),'prefix_match':on==refon and bool(errs) and not missing and max(errs)<=1e-6}
(A/'prefix-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
