from pathlib import Path
import gzip,json
R=Path('artifacts');A=R/'crow-v15-perch-weight-transfer-strong-20261113';B=R/'crow-v15-hiproll-grip-release-20261113'
def read(p):
 s={};b={}
 with gzip.open(p/'trace.log.gz','rt') as f:
  for line in f:
   x=line.split()
   if x and x[0]=='step':
    qi=x.index('q');vi=x.index('v');s[int(x[1]),int(x[2])]=list(map(float,x[qi+1:vi]+x[vi+1:]))
   elif x and x[0]=='balance':b[int(x[1]),int(x[2])]=list(map(float,x[3:]))
 return s,b
ref,_=read(B);cur,bal=read(A);out={}
for e in (0,1):
 ts=sorted(t for lane,t in cur if lane==e);on=next((t for t in ts if bal[e,t][0]),None);check=[t for t in ts if on is None or t<on];missing=[t for t in check if (e,t) not in ref];errs=[abs(a-b) for t in check if (e,t) in ref for a,b in zip(cur[e,t],ref[e,t])];out[str(e)]={'weight_transfer_start_zero_based':on,'matched_preintervention_states':len(check)-len(missing),'missing':missing,'max_abs_qv_error':max(errs,default=None),'prefix_match':bool(errs) and not missing and max(errs)<=1e-6,'max_abs_roll_target_rad':max((abs(bal[e,t][-1]) for t in ts),default=0),'max_readiness_streak':max((int(bal[e,t][1]) for t in ts),default=0)}
(A/'prefix-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
