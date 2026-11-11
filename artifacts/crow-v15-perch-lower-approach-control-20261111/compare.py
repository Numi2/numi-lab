from pathlib import Path
import gzip,json
ref=Path('artifacts/crow-v15-perch-alignment-gate-control-20261111/autonomous-20261126');cur=Path('artifacts/crow-v15-perch-lower-approach-control-20261111/autonomous-20261126')
contact=json.loads((ref/'perch-analysis.json').read_text())['rows'];ends=[r['first_episode_steps'] for r in contact]
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
 first_contact=contact[e]['first_perch_foot_impulse_step'] or ends[e]+1;dif=[];missing=[]
 for t in range(1,ends[e]+1):
  x=a.get((e,t));y=b.get((e,t))
  if x is None or y is None:missing.append(t);continue
  if any(x[k]!=y[k] for k in ('q','v','policy_action','contacts')):dif.append(t)
 rows.append({'lane':e,'first_reference_perch_foot_impulse':first_contact if first_contact<=ends[e] else None,'first_difference':dif[0] if dif else None,'differences_before_first_perch_contact':sum(t<first_contact for t in dif),'missing_before_first_perch_contact':sum(t<first_contact for t in missing)})
out={'rows':rows,'scope':'Endpoint accepted q/v, actor output and contact comparison only. Both trials run without a teacher. Substep support is analyzed separately.'};(cur/'comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
