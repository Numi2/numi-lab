import gzip,json
from pathlib import Path
A=Path('artifacts/crow-v15-leg-push-sweep-20261110');rows={}
for arm in ('hip','fore','deep'):
 d={}
 for line in gzip.open(A/('diagnostic-reserved-final-j-'+arm+'.stderr.gz'),'rt'):
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   z=json.loads(line.split(' ',1)[1]);t=z['step']
   if 650<=t<=670:
    active=[c for c in z['contacts'] if c['impulses'][0]>1e-7]
    d[t]={'x':z['q'][0],'z':z['q'][2],'vx':z['v'][0],'vz':z['v'][2],'contacts':len(z['contacts']),'active_contact_pairs':sorted(set((c['a_name'],c['b_name']) for c in active))}
 rows[arm]=d
(A/'analysis.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
