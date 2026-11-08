import json,math
from pathlib import Path
cases=json.loads(Path('candidate-trace.json').read_text())['cases'];out=[]
for c in cases:
 for i,r in enumerate(c['rows']):
  n,u,v=r['impulse'];a,b=r['post_relative'][1:];limit=c['mu']*n
  p=u-a;q=v-b;norm=math.hypot(p,q)
  if norm>limit and norm>0:p*=limit/norm;q*=limit/norm
  residual=math.hypot(p-u,q-v)
  assert math.isfinite(residual)
  out.append({'requested':c['iterations'],'mu':c['mu'],'vn':c['initial_vn'],'dir':c['dir'],'row':i,'residual_m_s':residual})
summary={'rows':len(out),'max_requested_2_m_s':max(x['residual_m_s'] for x in out if x['requested']==2),'max_requested_64_m_s':max(x['residual_m_s'] for x in out if x['requested']==64),'violations':sum(x['residual_m_s']>.002 for x in out),'per_row':out}
Path('independent-natural.json').write_text(json.dumps(summary,indent=2)+'\n');print({k:v for k,v in summary.items() if k!='per_row'})
