from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-regrip-rear-force-feedback-20261111');B=Path('artifacts/crow-v15-single-foot-release-20261111')
def read(p):
 d={};release={}
 with gzip.open(p/'results.txt.gz','rt') as f:
  for l in f:
   v=l.split()
   if v and v[0]=='trace':d[(int(v[1]),int(v[3]))]=v[4:]
   elif v and v[0]=='foot_release_requested':
    if p==B:release[1]=int(v[1])
    else:release[int(v[1])]=int(v[2])
 return d,release
a,ar=read(A);b,br=read(B);d=[(1,t) for t in range(1500) if a.get((1,t))!=b.get((1,t))];d += [(i,t) for i in range(2,17) for t in range(877) if a.get((i,t))!=b.get((1,t))];d += [('release',i) for i in range(1,17) if ar.get(i)!=br.get(1)];r={'reference_samples':1500,'early_samples_per_candidate':877,'releases':ar,'differences':d};(A/'parity.json').write_text(json.dumps(r,indent=2)+'\n');print({'differences':len(d),'first':d[:10],'releases':ar});assert not d
