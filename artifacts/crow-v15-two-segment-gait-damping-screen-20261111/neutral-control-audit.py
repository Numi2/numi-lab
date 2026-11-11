from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-two-segment-gait-damping-screen-20261111');B=Path('artifacts/crow-v15-two-segment-gait-stance-screen-20261111')
def read(path):
 d={}
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1])
    if x['lane']==5:d[x['step']]=x
 return d
a=read(A/'stderr.gz');b=read(B/'stderr.gz');missing=[t for t in range(1,302) if t not in a or t not in b];different=[t for t in range(1,302) if t in a and t in b and a[t]!=b[t]];out={'lane':5,'samples_expected':301,'samples_current':len(a),'samples_reference':len(b),'missing_steps':missing,'different_steps':different,'pass':not missing and not different};(A/'neutral-control-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
