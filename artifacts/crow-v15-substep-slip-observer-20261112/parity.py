from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-substep-slip-observer-20261112')
paths=[Path('artifacts/crow-v15-toe-preload-matched-trial-20261112/candidate-20261127-rebound-low'),A/'candidate-20261127-rebound-low']
def read(p):
 s={};c={}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);
    if 1<=x['step']<=300:s[x['lane'],x['step']]=x
   elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=l.split();t,k,e=map(int,z[1:4]);
    if 1<=t<=300:c[e,t,k]=z[4:]
 return s,c
(a,b),(c,d)=map(read,paths)
assert len(a)==len(c)==2400 and len(b)==len(d)==38400
assert a.keys()==c.keys() and b.keys()==d.keys()
sd=[k for k in a if a[k]!=c[k]]
cd=[k for k in b if b[k]!=d[k]]
out={'states':len(a),'substeps':len(b),'state_differences':len(sd),'substep_differences':len(cd),'examples':(sd+cd)[:10]}
(A/'parity.json').write_text(json.dumps(out,indent=2)+'\n');print(out);assert not sd and not cd
