from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-walk-step-strength-screen-20261111/trial');B=Path('artifacts/crow-v15-perch-support-gate-screen-20261111/trial')
def read(p):
 d={}
 for z in sorted(p.glob('stderr-*.gz')):
  with gzip.open(z,'rt') as f:
   for l in f:
    if l.startswith('CROW_SUPPORT_SAMPLE '):
     x=json.loads(l.split(' ',1)[1]);e=x['lane'];t=x['step']
     if t<=355:d[e,t]=x
 return d
x=read(A);y=read(B);rows=[]
for e in range(8):
 first=next((t for t in range(1,356) if any(x[e,t][k]!=y[e,t][k] for k in ('q','v','policy_action'))),None);rows.append({'lane':e,'first_state_or_action_difference_step':first})
Path('artifacts/crow-v15-walk-step-strength-screen-20261111/paired-prefix.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
