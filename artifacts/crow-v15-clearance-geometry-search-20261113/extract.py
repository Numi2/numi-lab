from pathlib import Path
import gzip,json,hashlib
A=Path(__file__).resolve().parent;S=A.parent/'crow-v15-staged-perch-release-20261113';stage=json.loads((S/'stage-check.json').read_text());choose={int(e):v['first_nonzero_lift_zero_based']-1 for e,v in stage.items()};assert all(t>=0 for t in choose.values());states={};poses={}
with gzip.open(S/'trace.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if not x or x[0] not in ('step','pose'):continue
  e,t=map(int,x[1:3])
  if t!=choose[e]:continue
  if x[0]=='step':
   q=x.index('q');v=x.index('v');states[e]=x[q+1:v]+x[v+1:]
  elif x[3]=='crow_left_foot' or (x[3].startswith('crow_left_') and 'toe' in x[3]):poses[e,x[3]]=list(map(float,x[4:]))
assert len(states)==2 and len(poses)==18
(A/'states.tsv').write_text(''.join(str(e)+' '+str(choose[e])+' '+' '.join(states[e])+'\n' for e in sorted(states)))
(A/'expected-poses.json').write_text(json.dumps({'selected_steps_zero_based':choose,'trace_sha256':hashlib.sha256((S/'trace.log.gz').read_bytes()).hexdigest(),'poses':{str(e):{name:pose for (lane,name),pose in poses.items() if lane==e} for e in choose}},indent=2)+'\n')
print('selected',choose)
