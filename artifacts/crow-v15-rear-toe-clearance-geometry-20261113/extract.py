from pathlib import Path
import gzip,json,hashlib
A=Path('artifacts/crow-v15-rear-toe-clearance-geometry-20261113');S=A.parent/'crow-v15-active-clearance-regrip-20261113';selected={208,210,212,215,220,230};states={};poses={}
with gzip.open(S/'trace.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if not x or x[0] not in ('step','pose') or x[1]!='0' or int(x[2]) not in selected:continue
  t=int(x[2])
  if x[0]=='step':
   q=x.index('q');v=x.index('v');states[t]=x[q+1:v]+x[v+1:]
  elif x[3]=='crow_left_foot' or (x[3].startswith('crow_left_') and 'toe' in x[3]):poses[t,x[3]]=list(map(float,x[4:]))
assert set(states)==selected and len(poses)==54
(A/'states.tsv').write_text(''.join('0 '+str(t)+' '+' '.join(states[t])+'\n' for t in sorted(selected)))
(A/'expected-poses.json').write_text(json.dumps({'trace_sha256':hashlib.sha256((S/'trace.log.gz').read_bytes()).hexdigest(),'poses':{str(t):{n:p for (u,n),p in poses.items() if u==t} for t in selected}},indent=2)+'\n')
