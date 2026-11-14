from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-stance-roll-authority-20261114'); s={};b={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  a=l.split()
  if not a or a[0] not in ('step','balance'):continue
  e,t=int(a[1]),int(a[2])
  if a[0]=='step':
   qi=a.index('q');vi=a.index('v');s[e,t]=list(map(float,a[qi+1:vi]+a[vi+1:]))
  elif a[0]=='balance':b[e,t]=list(map(float,a[3:]))
first_roll=next(t for t in range(450) if b[0,t][-1]!=b[1,t][-1]);first_state=next((t for t in range(450) if s[0,t]!=s[1,t]),None)
out={'first_roll_target_difference_zero_based':first_roll,'first_state_difference_zero_based':first_state,'state_exact_before_first_roll_target':all(s[0,t]==s[1,t] for t in range(first_roll)),'caveat':'target and state logging epochs not independently verified'}
(A/'roll-prefix.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
