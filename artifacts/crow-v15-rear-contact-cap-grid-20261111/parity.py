from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-rear-contact-cap-grid-20261111');B=Path('artifacts/crow-v15-front-contact-regulation-20261111')
def read(p,lane):
 out={}
 with gzip.open(p/'results.txt.gz','rt') as f:
  for line in f:
   v=line.split()
   if v and v[0]=='trace' and int(v[1])==lane and int(v[3])<750:out[int(v[3])]=v[4:]
 return out
a,b=read(A,1),read(B,14);diff=[i for i in range(750) if a.get(i)!=b.get(i)];r={'lane':1,'reference_lane':14,'samples':len(a),'differences':len(diff),'first_differences':diff[:20]};(A/'parity.json').write_text(json.dumps(r,indent=2)+'\n');print(r);assert len(a)==len(b)==750 and not diff
