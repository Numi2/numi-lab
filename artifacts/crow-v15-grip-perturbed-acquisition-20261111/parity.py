from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-grip-perturbed-acquisition-20261111');B=Path('artifacts/crow-v15-regrip-low-load-rate-screen-20261111')
def read(p):
 d={};release={}
 with gzip.open(p/'results.txt.gz','rt') as f:
  for line in f:
   v=line.split()
   if v and v[0]=='trace':d[int(v[1]),int(v[3])]=v[4:]
   elif v and v[0]=='foot_release_requested':release[int(v[1])]=int(v[2])
 return d,release
a,ar=read(A);b,br=read(B);diff=[(1,t) for t in range(1500) if a.get((1,t))!=b.get((15,t))]
if ar.get(1)!=br.get(15):diff.append(('release',1))
(A/'parity.json').write_text(json.dumps({'reference_samples':1500,'releases':ar,'differences':diff},indent=2)+'\n');print('reference differences',len(diff),'releases',ar);assert not diff
