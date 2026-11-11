from pathlib import Path
import gzip,json
D=Path('artifacts/crow-v15-hip-roll-negative-shift-20261111');R=Path('artifacts/crow-v15-hip-roll-filtered-release-20261111')
def read(d):
 data={}
 with gzip.open(d/'trace.log.gz','rt') as f:
  for l in f:
   a=l.split()
   if a and a[0]=='step' and int(a[1]) in (6,7):
    e,t=map(int,a[1:3]);i=a.index('q');j=a.index('v');data[e,t]=(a[i+1:j],a[j+1:])
 return data
a,b=read(R),read(D);common=sorted(a.keys()&b.keys());r={'reference':str(R),'compared_endpoint_states':len(common),'first_difference':next((list(k) for k in common if a[k]!=b[k]),None),'reference_only':len(a.keys()-b.keys()),'repeat_only':len(b.keys()-a.keys()),'scope':'Zero-drive lanes 6 and 7, q/v endpoint samples; batch rejection censors the negative experiment.'};(D/'zero-comparison.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
