from pathlib import Path
import gzip,json
R=Path('artifacts');A=R/'crow-v15-perch-release-ankle-20261113';B=R/'crow-v15-perch-weight-transfer-strong-20261113'
def read(p):
 s={};ev={}
 with gzip.open(p/'trace.log.gz','rt') as f:
  for line in f:
   x=line.split()
   if x and x[0]=='step':
    qi=x.index('q');vi=x.index('v');s[int(x[1]),int(x[2])]=list(map(float,x[qi+1:vi]+x[vi+1:]))
   elif x and x[0]=='event':ev[int(x[1]),int(x[2])]=bool(int(x[3]))
 return s,ev
ref,rev=read(B);cur,ev=read(A);out={}
for e in (0,1):
 ts=sorted(t for lane,t in cur if lane==e);on=next((t for t in ts if ev[e,t]),None);check=[t for t in ts if on is None or t<on];missing=[t for t in check if (e,t) not in ref];errs=[abs(a-b) for t in check if (e,t) in ref for a,b in zip(cur[e,t],ref[e,t])];out[str(e)]={'release_start_zero_based':on,'baseline_release_start_zero_based':next((t for t in ts if rev.get((e,t))),None),'matched_prerelease_states':len(check)-len(missing),'missing':missing,'max_abs_qv_error':max(errs,default=None),'prefix_match':bool(errs) and not missing and max(errs)<=1e-6}
(A/'prefix-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
