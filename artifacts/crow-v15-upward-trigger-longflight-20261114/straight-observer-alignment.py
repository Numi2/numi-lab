from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-upward-trigger-longflight-20261114');P=A/'straight-20261126-rebound-longflight'
x=json.load(gzip.open(P/'replay.json.gz','rt'))['payload'];lane=x['environment'];frames=x['frames'];states={(s['lane'],s['step']):s for s in json.load(gzip.open(A/'straight-reference.json.gz','rt'))}
rows=[]
for offset in (0,1):
 for key in ('q','v'):
  comparisons=[];missing=[]
  for frame in frames:
   s=states.get((lane,frame['step']+offset))
   if s is None:missing.append(frame['step']);continue
   comparisons.append(max(abs(a-b) for a,b in zip(frame[key],s[key])))
  rows.append({'offset':offset,'field':key,'compared':len(comparisons),'max_abs_difference':max(comparisons,default=None),'count_exceeding_1e-6':sum(y>1e-6 for y in comparisons),'missing':missing[:10]})
out={'scope':'same-run native replay versus support observer on replay lane only; contact epoch not tested','lane':lane,'rows':rows}
(A/'straight-observer-alignment.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
