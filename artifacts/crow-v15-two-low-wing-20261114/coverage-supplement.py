from pathlib import Path
from collections import Counter
import gzip,json
A=Path('artifacts/crow-v15-two-low-wing-20261114')
base=json.loads((A/'contact-onset-audit.json').read_text());rows=[]
for arm in ('push','hold'):
 with gzip.open(A/(arm+'-window.json.gz'),'rt') as f:d=json.load(f)
 for lane in (6,7):
  old=next(x for x in base['rows'] if x['arm']==arm and x['lane']==lane);first=old['first_unloaded'];out={'arm':arm,'lane':lane,'first_unloaded':first,'streams':{}}
  for typ in ('contact','toe','aero'):
   c=Counter((x[1],x[2]) for x in d[typ] if x[0]==lane)
   full={(t,k) for t in range(335,501) for k in range(16)}
   pre={(t,k) for t in range(371,first) for k in range(16)} if first else set()
   out['streams'][typ]={'missing_335_500':len(full-c.keys()),'duplicate_335_500':sum(c[x]-1 for x in full if c[x]>1),'missing_371_before_unload':len(pre-c.keys()),'last_recorded':max((t for t,k in c),default=None)}
  rows.append(out)
result={'rows':rows,'note':'The original audit required aerodynamic samples through step 500, but aero tracing stops after step 400. It also summed an incomplete held-leg pre-unload aerodynamic window; interpret that number only as a partial sum. Retained original audit is unchanged.'}
(A/'coverage-supplement.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
