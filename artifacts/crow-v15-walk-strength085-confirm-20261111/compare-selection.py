from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-walk-strength085-confirm-20261111/trial-20261126');B=Path('artifacts/crow-v15-walk-step-strength-screen-20261111/trial')
def read(p):
 d={}
 for z in sorted(p.glob('stderr-*.gz')):
  with gzip.open(z,'rt') as f:
   for l in f:
    if l.startswith('CROW_SUPPORT_SAMPLE '):
     x=json.loads(l.split(' ',1)[1]);e=x['lane'];t=x['step']
     if e in (4,5) and t<=355:d[e,t]=x
 return d
x=read(A);y=read(B);assert len(x)==len(y)==2*355
r=[{'lane':e,'first_state_or_action_difference_step':next((t for t in range(1,356) if any(x[e,t][k]!=y[e,t][k] for k in ('q','v','policy_action'))),None)} for e in (4,5)]
assert all(z['first_state_or_action_difference_step'] is None for z in r),r
Path('artifacts/crow-v15-walk-strength085-confirm-20261111/selection-parity.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
