from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-release-com-velocity-grid-20261111');B=Path('artifacts/crow-v15-release-balance-continuity-20261111')
def read(p):
 r={}
 with gzip.open(p/'results.txt.gz','rt') as f:
  for l in f:
   v=l.split()
   if v and v[0]=='trace':r[(int(v[1]),int(v[3]))]=v[4:]
 return r
a,b=read(A),read(B);d=[(1,t) for t in range(1100) if a.get((1,t))!=b.get((1,t))];d += [(i,t) for i in range(2,17) for t in range(777) if a.get((i,t))!=b.get((1,t))];r={'reference_samples':1100,'early_samples_per_candidate':777,'differences':d};(A/'parity.json').write_text(json.dumps(r,indent=2)+'\n');print({'differences':len(d),'first':d[:10]});assert not d
