from pathlib import Path
import gzip,json
R=Path.cwd();report={}
def samples(path):
 with gzip.open(path,'rt') as f:
  return {(x['lane'],x['step']):x for line in f if line.startswith('CROW_FLIGHT_SAMPLE ') for x in [json.loads(line.split(' ',1)[1])]}
for course in ['training','held-out-a','held-out-b']:
 old=samples(R/'artifacts/crow-hierarchical-demonstrations-20261109'/course/'rollout.stderr.gz')
 new=samples(R/'artifacts/crow-goal-memory-demonstrations-20261109'/course/'rollout.stderr.gz')
 keys=set(old)|set(new)
 bad=[k for k in keys if k not in old or k not in new or old[k]['q']!=new[k]['q'] or old[k]['v']!=new[k]['v']]
 report[course]={'samples':len(keys),'mismatches':len(bad),'first_mismatches':[list(x) for x in bad[:5]]}
assert all(v['samples']>0 and v['mismatches']==0 for v in report.values()),report
(R/'artifacts/crow-goal-memory-demonstrations-20261109/physics-parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
