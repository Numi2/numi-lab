from pathlib import Path
import gzip,json,hashlib
A=Path(__file__).resolve().parent;S=A.parent/'crow-v15-supported-pitch-feedback-20261113';select={157,208,212,220,240};states={};poses={}
with gzip.open(S/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if not x or x[0] not in ('step','pose') or int(x[1])!=1 or int(x[2]) not in select:continue
  t=int(x[2])
  if x[0]=='step':
   q=x.index('q');v=x.index('v');states[t]=x[q+1:v]+x[v+1:]
  elif x[3]=='crow_left_foot' or (x[3].startswith('crow_left_') and 'toe' in x[3]):poses[t,x[3]]=list(map(float,x[4:]))
assert set(states)==select and len(poses)==45
(A/'states.tsv').write_text(''.join('1 '+str(t)+' '+' '.join(states[t])+'\n' for t in sorted(select)))
(A/'expected-poses.json').write_text(json.dumps({'trace_sha256':hashlib.sha256((S/'trace.log.gz').read_bytes()).hexdigest(),'poses':{str(t):{n:p for (u,n),p in poses.items() if u==t} for t in select}},indent=2)+'\n')
