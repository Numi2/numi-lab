from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-gait-onset-ramp-20261111');B=Path('artifacts/crow-v15-sustained-walk-20261111')
def read(path):
 sup={};sub={}
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(line.split(' ',1)[1]);
    if d['step']<=500:sup[d['lane'],d['step']]=d
   elif line.startswith('CROW_WALK_SUBSTEP '):
    v=line.split();l,t,k=int(v[3]),int(v[1]),int(v[2]);
    if t<=500:sub[l,t,k]=list(map(float,v[6:]))
 return sup,sub
a,aa=read(A/'baseline.stderr.gz');b,bb=read(B/'walk.stderr.gz');diff=[k for k in set(a)|set(b) if k not in a or k not in b or any(a[k][f]!=b[k][f] for f in ('q','v'))];cdiff=[k for k in set(aa)|set(bb) if aa.get(k)!=bb.get(k)];out={'qv_differences':diff[:20],'contact_differences':cdiff[:20],'qv_difference_count':len(diff),'contact_difference_count':len(cdiff)};(A/'parity.json').write_text(json.dumps(out,indent=2)+'\n');print(out);assert not diff and not cdiff
