from pathlib import Path
import collections,gzip,json
A=Path('artifacts/crow-v15-toe-damping-slip-screen-20261112')
paths={'baseline':Path('artifacts/crow-v15-substep-slip-observer-20261112/candidate-20261127-rebound-low/run.stderr.gz'),'candidate':A/'candidate-20261127-rebound-low/run.stderr.gz'}
out={}
for name,p in paths.items():
 counts=collections.Counter();seen=set()
 with gzip.open(p,'rt') as f:
  for l in f:
   z=l.split()
   if z and z[0]=='CROW_TOE_V_SUBSTEP':
    t,k,e=map(int,z[1:4])
    if 1<=t<=300:
     seen.add((e,t,k))
     if max(abs(float(v)) for v in z[4:])>=11.99:counts[str(e)+('_early' if t<=200 else '_walk')]+=1
 assert len(seen)==38400
 out[name]={'seen_substeps':len(seen),'toe_cap_substeps_by_lane_and_phase':dict(counts)}
(A/'toe-cap-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
