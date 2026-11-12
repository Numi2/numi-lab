from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-corvid-ankle-support-gate-20261112')
paths=[Path('artifacts/crow-v15-toe-preload-matched-trial-20261112/candidate-20261127-rebound-low'),A/'candidate-20261127-rebound-fine']
def read(path):
 d={}
 with gzip.open(path/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]); key=('state',x['lane'],x['step'],0)
    if x['step']<=200:d[key]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);key=('contact',e,t,k)
    if t<=200:d[key]=z[4:]
 return d
a,b=map(read,paths)
assert len(a)==27200 and set(a)==set(b),(len(a),len(b))
differences=[k for k in a if a[k]!=b[k]]
result={'state_samples':1600,'contact_substeps':25600,'differences':differences}
(A/'prefix-parity-fine.json').write_text(json.dumps(result,indent=2)+'\n')
print('prefix differences',len(differences));assert not differences
