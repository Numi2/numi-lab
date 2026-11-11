from pathlib import Path
import gzip,json
ref=Path('artifacts/crow-v15-student-state-landing-eval-20261111/autonomous-20261126');cur=Path('artifacts/crow-v15-explicit-goal-final-approach-control-20261111/autonomous-20261126')
d=json.loads((ref/'run.stdout').read_text());first=d['first_episode_navigation'];ends=first['steps_by_environment'];complete=first['first_completion_step_by_environment']
def read(p):
 out={}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e=x['lane'];t=x['step']
    if t<=ends[e]:out[e,t]=x
 return out
a=read(ref);b=read(cur);rows=[]
for e in range(8):
 end=complete[e] if complete[e] else ends[e];dif=[];missing=[]
 for t in range(1,end):
  x=a.get((e,t));y=b.get((e,t))
  if x is None or y is None:missing.append(t);continue
  if any(x[k]!=y[k] for k in ('q','v','policy_action','contacts')):dif.append(t)
 rows.append({'lane':e,'comparison_end_exclusive':end,'first_difference':dif[:1],'different_samples':len(dif),'first_missing':missing[:1]})
out={'rows':rows,'scope':'Endpoint accepted q/v, recorded actor outputs and contacts through the interval before the original fifth-waypoint threshold; teacher off in both. Matching endpoints do not prove all substeps identical.'};(cur/'comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
