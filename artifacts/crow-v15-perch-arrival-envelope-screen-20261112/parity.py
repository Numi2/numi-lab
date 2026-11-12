from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-perch-arrival-envelope-screen-20261112');R=Path('artifacts/crow-v15-perch-smaller-negative-hip-screen-20261112')
def read(root,mapping):
 out={}
 for name in ('trace.log.gz','substeps.log.gz'):
  with gzip.open(root/'mixed'/name,'rt') as f:
   for l in f:
    z=l.split()
    if not z:continue
    if z[0]=='step':
     e,t=map(int,z[1:3])
     if e in mapping:
      i=z.index('q');out['step',mapping[e],t]=z[i:]
    elif z[0] in ('CROW_TWO_TOE_SUBSTEP','CROW_TOE_VELOCITY'):
     t,k,e=map(int,z[1:4])
     if e in mapping:out[z[0],mapping[e],t,k]=z[4:]
 return out
a=read(A,{4:0,5:1});b=read(R,{0:0,1:1});bad=[k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)]
out={'candidate_samples':len(a),'reference_samples':len(b),'mismatch_count':len(bad),'mismatch_examples':bad[:8],'pass':len(a)==len(b)==71500 and not bad}
(A/'baseline-parity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if not out['pass']:raise SystemExit('Baseline parity failed; do not interpret arrival arms')
