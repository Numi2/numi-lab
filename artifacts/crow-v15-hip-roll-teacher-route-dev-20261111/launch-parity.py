from pathlib import Path
import gzip,json
def read(p):
 d={}
 with gzip.open(p,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1])
    if x['step']<=400:d[x['lane'],x['step']]=x
 return d
a=read(Path('artifacts/crow-v15-hip-roll-autonomous-route-dev-20261111/autonomous-20261126/run.stderr.gz'))
b=read(Path('artifacts/crow-v15-hip-roll-teacher-route-dev-20261111/teacher-20261126/run.stderr.gz'))
out=[]
for e in range(8):
 missing=[t for t in range(1,401) if (e,t) not in a or (e,t) not in b]
 delta=[t for t in range(1,401) if (e,t) in a and (e,t) in b and (a[e,t]['q']!=b[e,t]['q'] or a[e,t]['v']!=b[e,t]['v'])]
 out.append({'lane':e,'missing_steps':missing,'first_difference_step':delta[0] if delta else None,'different_steps':len(delta)})
Path('artifacts/crow-v15-hip-roll-teacher-route-dev-20261111/launch-parity.json').write_text(json.dumps(out,indent=2)+'\n')
print(out)
