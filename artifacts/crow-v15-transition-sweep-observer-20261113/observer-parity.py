from pathlib import Path
import gzip,json,hashlib
A=Path('artifacts/crow-v15-transition-sweep-observer-20261113'); O=Path('artifacts/crow-v15-hop-wing-transition-20261113'); rows=[]
for arm in ('push','hold'):
 ds=[]
 for base,tag in ((O,'rebound-transition'),(A,'rebound-sweep')):
  p=base/(arm+'-wing-20261126-'+tag)/'run.stderr.gz';d={}
  with gzip.open(p,'rt') as f:
   for line in f:
    if line.startswith('CROW_SUPPORT_SAMPLE '):
     x=json.loads(line.split(' ',1)[1]);
     if x['lane'] in (2,3):
      key=x['lane'],x['step'];assert key not in d;d[key]=x
  ds.append(d)
 for e in (2,3):
  missing=[t for t in range(1,501) if any((e,t) not in d for d in ds)]
  mismatch={k:[t for t in range(1,501) if all((e,t) in d for d in ds) and ds[0][e,t][k]!=ds[1][e,t][k]] for k in ('q','v','policy_action','contacts')}
  rows.append({'arm':arm,'lane':e,'missing':missing,'mismatches':mismatch})
valid=all(not r['missing'] and not any(r['mismatches'].values()) for r in rows);out={'valid':valid,'rows':rows,'scope':'Exact accepted-state and action equality on same candidate lanes versus prior shorter-window observer. Other lanes intentionally have altered target commands.'};(A/'observer-parity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if not valid:raise SystemExit('observer parity failed')
