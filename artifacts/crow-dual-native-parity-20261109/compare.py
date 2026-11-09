from pathlib import Path
import gzip,json
R=Path.cwd();A=R/'artifacts/crow-dual-native-parity-20261109'
def samples(path):
 with gzip.open(path,'rt') as f:
  return {(x['lane'],x['step']):x for line in f if line.startswith('CROW_FLIGHT_SAMPLE ') for x in [json.loads(line.split(' ',1)[1])]}
rows=[]
for course,diagnostic in [
 ('reserved-final-c',R/'artifacts/crow-joint-specialist-eval-20261109/stage3-160-stage2-80/reserved-final-c'),
 ('reserved-final-d',R/'artifacts/crow-joint-specialist-eval-20261109/stage3-160-stage2-80/reserved-final-d'),
 ('held-out-a',R/'artifacts/crow-joint-fresh-dev-20261109/stage3-160-stage2-80/held-out-a')]:
 native=A/'integrated'/course
 da=json.loads((diagnostic/'results.json').read_text());db=json.loads((native/'results.json').read_text())
 a=samples(diagnostic/'rollout.stderr.gz');b=samples(native/'rollout.stderr.gz')
 qa={k:(x['q'],x['v']) for k,x in a.items()};qb={k:(x['q'],x['v']) for k,x in b.items()}
 outcomes=['steps_by_environment','maximum_waypoints_by_environment','physical_failure_by_environment','first_completion_step_by_environment']
 eq={k:da['first_episode'][k]==db['first_episode'][k] for k in outcomes}
 row={'course':course,'samples':len(qa),'qv_exact':qa==qb,'outcomes_exact':eq,'supported_exact':[x['supported_finish'] for x in da['lanes']]==[x['supported_finish'] for x in db['lanes']]}
 rows.append(row)
 print(row,flush=True)
assert all(x['qv_exact'] and all(x['outcomes_exact'].values()) and x['supported_exact'] for x in rows)
(A/'comparison.json').write_text(json.dumps({'rows':rows,'scope':'diagnostic versus integrated teacher-off first episodes, sampled q/v and complete first-outcome records'},indent=2)+'\n')
