from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-aerodynamic-wrench-audit-20261112');ds=[Path('artifacts/crow-v15-common-command-pronation-screen-20261112/candidate-20261126-rebound-flexed'),A/'candidate-20261126-rebound-aero']
def read(d):
 out={}
 with gzip.open(d/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);
    if 1<=x['step']<=1200:out['state',x['lane'],x['step'],0]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 1<=t<=1200:out['contact',e,t,k]=z[4:]
 return out
x,y=map(read,ds);assert len(x)==8*1200*17 and x.keys()==y.keys(),(len(x),len(y));diff=[k for k in x if x[k]!=y[k]];out={'states':9600,'contact_substeps':153600,'difference_count':len(diff),'first_differences':diff[:20]};(A/'parity.json').write_text(json.dumps(out,indent=2)+'\n');print(out);assert not diff
