from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-sensor-threshold-audit-20261112'); ds=[Path('artifacts/crow-v15-materialized-contact-feedback-20261112/candidate-20261127-rebound-fine'),A/'candidate-20261127-rebound-fine']
def read(d):
 out={}
 with gzip.open(d/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);
    if 1<=x['step']<=300:out['state',x['lane'],x['step'],0]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 1<=t<=300:out['contact',e,t,k]=z[4:]
 return out
x,y=map(read,ds);assert len(x)==40800 and x.keys()==y.keys(),(len(x),len(y));diff=[k for k in x if x[k]!=y[k]];out={'states':2400,'contact_substeps':38400,'difference_count':len(diff),'first_differences':diff[:20]};(A/'observer-repeat-parity.json').write_text(json.dumps(out,indent=2)+'\n');print(out);assert not diff
