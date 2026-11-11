from pathlib import Path
import gzip,json,collections
A=Path('artifacts/crow-v15-toe-speed-limit-all-lanes-20261111');p=Path('artifacts/crow-v15-toe-speed-limit-observer-20261111/stderr.gz');hits=collections.Counter();steps=collections.Counter();toes=collections.Counter();seen=set();minsep=None
with gzip.open(p,'rt') as f:
 for l in f:
  if not l.startswith('CROW_SUPPORT_SAMPLE '):continue
  x=json.loads(l.split(' ',1)[1]);seen.add((x['step'],x['lane']))
  for c in x['contacts']:
   z=c['separation'];minsep=z if minsep is None else min(minsep,z)
  inds=[i for i,z in enumerate(x['v'][22:38]) if abs(z)>=11.99]
  if inds:
   hits[x['lane']]+=1;steps[x['step']]+=1
   for i in inds:toes[i]+=1
out={'observed_lane_steps':len(seen),'lanes_with_hits':len(hits),'lane_interval_hits':sum(hits.values()),'hits_by_lane':dict(hits),'hits_by_step':dict(steps),'hits_by_toe':dict(toes),'minimum_final_substep_contact_separation_m':minsep,'scope':'retrospective accepted endpoint and last-substep contact analysis; earlier sparse flight sampling missed these limits'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('lanes',len(hits),'hits',sum(hits.values()),'observed',len(seen),'minsep',minsep)
