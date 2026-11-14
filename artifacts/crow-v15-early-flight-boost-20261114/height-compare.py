from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-early-flight-boost-20261114')
B=Path('artifacts/crow-v15-intermediate-wing-amplitude-20261114')
def states(p):
 with gzip.open(p,'rt') as f: a=json.load(f)['state']
 d={(x['lane'],x['step']):x for x in a}
 if len(a)!=len(d): raise ValueError('duplicate state records')
 return d
old=states(B/'push-window.json.gz')
new={arm:states(A/(arm+'-window.json.gz')) for arm in ('push','hold')}
missing=[];diff=[]
for t in range(1,396):
 a=old.get((6,t));b=new['push'].get((6,t))
 if a is None or b is None:missing.append(t);continue
 for key in ('q','v','policy_action','contacts'):
  if a[key]!=b[key]:diff.append([t,key])
rows=[]
for arm,d in new.items():
 for e in (6,7):
  samples=[(t,d[e,t]['q'][2]) for t in range(450,501) if (e,t) in d]
  rows.append({'arm':arm,'lane':e,'count':len(samples),'min_height_m':min((x[1] for x in samples),default=None),'below_floor':[[t,h] for t,h in samples if h<.25],'passes_height':len(samples)==51 and all(h>=.25 for _,h in samples)})
out={'old_push_lane6_preboost_steps':395,'missing':missing,'differences':diff,'exact':not missing and not diff,'height_450_to_500':rows,'scope':'Used-seed development screen. State and contact timing in these candidate lanes remains inferred.'}
(A/'height-compare.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
