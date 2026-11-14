from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-observer-timing-audit-20261114');p=Path('artifacts/crow-v15-launch-candidate-replay-20261114/push-20261126-rebound-lane7/run.stderr.gz');states={};subs={}
with gzip.open(p,'rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);states[x['lane'],x['step']]=len(x['contacts'])
  elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
   z=line.split();t,k,e,count=map(int,z[1:5])
   if k==15:subs[e,t]=count
common=states.keys() & subs.keys();bad=[(k,states[k],subs[k]) for k in sorted(common) if states[k]!=subs[k]]
out={'comparisons':len(common),'mismatched_counts':len(bad),'examples':bad[:10],'scope':'same-label contact count matches last logged substep; count alone does not prove full contact identity'}
(A/'contact-timing.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
