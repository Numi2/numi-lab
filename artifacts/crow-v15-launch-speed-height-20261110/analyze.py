import gzip,json
from pathlib import Path
A=Path('artifacts/crow-v15-launch-speed-height-20261110');rows={}
for arm in ('low','mid','high'):
 d={}
 for line in gzip.open(A/('diagnostic-reserved-final-j-'+arm+'.stderr.gz'),'rt'):
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   z=json.loads(line.split(' ',1)[1]);t=z['step']
   if t in (650,652,655,660,665,670,680,700,730,750,800,900,1000):d[t]={'x':z['q'][0],'z':z['q'][2],'vx':z['v'][0],'vz':z['v'][2]}
 rows[arm]=d
(A/'analysis.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
