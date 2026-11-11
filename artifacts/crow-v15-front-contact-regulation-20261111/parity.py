from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-front-contact-regulation-20261111')
def read(p,lane,n):
 out={}
 with gzip.open(p,'rt') as f:
  for line in f:
   v=line.split()
   if v and v[0]=='trace' and int(v[1])==lane and int(v[3])<n:out[int(v[3])]=v[4:]
 return out
reports=[]
for lane,path,ref,n in [(1,'artifacts/crow-v15-ankle-range-grid-20261111/results.txt.gz',5,500),(2,'artifacts/crow-v15-com-damping-grid-20261111/results.txt.gz',5,750)]:
 a=read(A/'results.txt.gz',lane,n);b=read(path,ref,n);diff=[i for i in range(n) if a.get(i)!=b.get(i)];reports.append({'lane':lane,'reference_lane':ref,'samples':len(a),'differences':len(diff),'first_differences':diff[:20]});assert len(a)==len(b)==n and not diff,reports
(A/'parity.json').write_text(json.dumps(reports,indent=2)+'\n');print(reports)
