import gzip,json
from pathlib import Path
A=Path('artifacts/crow-v15-launch-transition-20261110')
paths={'baseline':Path('artifacts/crow-v15-integrated-push-causality-20261110/diagnostic-reserved-final-j-push.stderr.gz')}
paths.update({a:A/('diagnostic-reserved-final-j-'+a+'.stderr.gz') for a in ('extension','forward')})
rows={}
for name,path in paths.items():
 d={}
 for line in gzip.open(path,'rt'):
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   z=json.loads(line.split(' ',1)[1]);t=z['step']
   if t in (650,652,655,660,665,670,680,700,730,750):d[t]={'x':z['q'][0],'z':z['q'][2],'vx':z['v'][0],'vz':z['v'][2],'leg_q':z['q'][14:17]}
 rows[name]=d
print(json.dumps(rows,indent=2));(A/'analysis.json').write_text(json.dumps(rows,indent=2)+'\n')
