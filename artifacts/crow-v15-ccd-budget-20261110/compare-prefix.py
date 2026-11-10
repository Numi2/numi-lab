import gzip,json
from pathlib import Path
A=Path('artifacts/crow-v15-ccd-budget-20261110')
def load(path):
 d={}
 with gzip.open(path,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);t=x['step']
    if t<=3839:d[t]=x
 return d
x=load('artifacts/crow-v15-guided-descent-20261110/development-j-descent.stderr.gz');y=load(A/'development-j-budget.stderr.gz');common=sorted(set(x)&set(y));differences=[];worst=0
for t in common:
 diff=max(abs(a-b) for k in ('q','v') for a,b in zip(x[t][k],y[t][k]));worst=max(worst,diff)
 if diff:differences.append({'step':t,'max_abs_qv':diff})
r={'common_samples':len(common),'first_step':common[0] if common else None,'last_step':common[-1] if common else None,'max_abs_qv_difference':worst,'nonidentical_samples':len(differences),'first_difference':differences[0] if differences else None};(A/'prefix-parity.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
