from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-flight-release-latch-20261114');P=A/'turn-20261127-rebound-latch27';x=json.load(gzip.open(P/'replay.json.gz','rt'))['payload'];lane=x['environment'];assert lane==7
states={(s['lane'],s['step']):s for s in json.load(gzip.open(A/'seed27-turn-reference.json.gz','rt'))};rows=[]
for offset in (0,1):
 for key in ('q','v'):
  ds=[];missing=[]
  for frame in x['frames']:
   s=states.get((lane,frame['step']+offset))
   if s is None:missing.append(frame['step']);continue
   ds.append(max(abs(a-b) for a,b in zip(frame[key],s[key])))
  rows.append({'offset':offset,'field':key,'compared':len(ds),'max_abs_difference':max(ds,default=None),'count_exceeding_1e-6':sum(y>1e-6 for y in ds),'missing':missing[:10]})
o={'scope':'same-run right-turn candidate replay versus support observer; contact timing not established','lane':lane,'rows':rows};(A/'seed27-observer-alignment.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
