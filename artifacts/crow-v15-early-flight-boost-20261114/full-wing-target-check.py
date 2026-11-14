from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-early-flight-boost-20261114');d={}
for arm in ('push','hold'):
 with gzip.open(A/(arm+'-window.json.gz'),'rt') as f:x=json.load(f)
 targets={(v[0],v[1]):v[2:] for v in x['target']}
 if len(targets)!=len(x['target']):raise ValueError('duplicate target records')
 d[arm]=targets
rows=[]
for e in (6,7):
 missing=[t for t in range(335,501) if (e,t) not in d['push'] or (e,t) not in d['hold']]
 ds=[abs(d['push'][e,t][j]-d['hold'][e,t][j]) for t in range(335,501) if (e,t) in d['push'] and (e,t) in d['hold'] for j in (8,9,10,11)]
 rows.append({'lane':e,'missing':missing,'count':len(ds),'max_difference_rad':max(ds,default=None),'match':not missing and len(ds)==664 and max(ds,default=1)<1e-5})
o={'rows':rows,'scope':'Supplemental check added after run; full wing target comparison through interval 500.'}
(A/'full-wing-target-check.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
