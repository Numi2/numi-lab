from pathlib import Path
import gzip,json
R=Path('artifacts');A=R/'crow-v15-active-clearance-regrip-20261113';B=R/'crow-v15-supported-pitch-feedback-20261113'
def read(p):
 state={};r={}
 with gzip.open(p/'trace.log.gz','rt') as f:
  for line in f:
   x=line.split()
   if x and x[0]=='step':
    q=x.index('q');v=x.index('v');state[int(x[1]),int(x[2])]=list(map(float,x[q+1:v]+x[v+1:]))
   elif x and x[0]=='reposition':r[int(x[1]),int(x[2])]=float(x[4])
 return state,r
ref,_=read(B);cur,rr=read(A);out={}
for e in (0,1):
 ts=sorted(t for lane,t in cur if lane==e);on=next((t for t in ts if rr.get((e,t),0)>0),None);check=[t for t in ts if on is None or t<on];missing=[t for t in check if (1,t) not in ref];errs=[abs(a-b) for t in check if (1,t) in ref for a,b in zip(cur[e,t],ref[1,t])];out[str(e)]={'reference_lane':1,'first_reposition_command_zero_based':on,'matched_preintervention_states':len(check)-len(missing),'missing':missing,'max_abs_qv_error':max(errs,default=None),'prefix_match':bool(errs) and not missing and max(errs)<=1e-6}
(A/'prefix-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
