from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-intermediate-seed27-20261114');entries=[];rows=[]
for arm in ('push','hold'):
 with gzip.open(A/(arm+'-window.json.gz'),'rt') as f:j=json.load(f)
 for x in j['state']:
  if x['lane'] in (6,7) and 350<=x['step']<=500:
   idx=len(entries);entries.append([arm,x['lane'],x['step']]);rows.append(str(idx)+' 0 '+' '.join(map(str,x['q']+x['v'])))
assert len(entries)==604
q=next(x['q'] for x in j['state'] if x['lane']==6 and x['step']==350);zero=[0.]*40;vel=zero.copy();vel[:3]=[.3,-.2,.1]
for v in (zero,vel):rows.append(str(len(rows))+' 0 '+' '.join(map(str,q+v)))
(A/'fk-input.tsv').write_text('\n'.join(rows)+'\n');(A/'fk-keys.json').write_text(json.dumps(entries)+'\n')
