from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-stronger-disturbance-20261114');D=A/'control-20261127-rebound-observer-control';S=Path('artifacts/crow-v15-equal-launch-onset-20261113/straight-20261127-rebound-equallong');ref={(x['lane'],x['step']):x for x in json.load(gzip.open(A/'straight-reference-states.json.gz','rt'))};new={};forces={};errors=[]
with gzip.open(D/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);k=x['lane'],x['step'];assert k not in new;new[k]=x;continue
  x=line.split()
  if not x:continue
  if 'OBSERVER_ERROR' in x[0] or x[0]=='CROW_SUBSTEP_ERROR':errors.append(line.strip())
  if x[0]=='CROW_PRE_AERO_WRENCH':
   t,k,e=map(int,x[1:4]);key=e,t,k;assert key not in forces;forces[key]=list(map(float,x[4:]))
stdout=json.loads((D/'run.stdout').read_text());orig=json.loads((A/'straight-reference-stdout.json').read_text());meta=stdout['first_episode_navigation'];base=orig['first_episode_navigation'];inputs=json.loads((D/'inputs.json').read_text());rinputs=json.loads((S/'inputs.json').read_text());parity=[]
for e in range(8):
 stop=min(base['steps_by_environment'][e],meta['steps_by_environment'][e]);bad=[t for t in range(1,stop+1) if (e,t) not in ref or (e,t) not in new or any(new[e,t][k]!=ref[e,t][k] for k in ('q','v','policy_action','contacts'))];parity.append({'lane':e,'through':stop,'mismatch_count':len(bad),'first_mismatches':bad[:5]})
missing=sum((e,t,k) not in forces for e in range(8) for t in range(895,946) for k in range(16));finite=all(len(v)==6 and all(map(math.isfinite,v)) for v in forces.values());maxforce=max((max(map(abs,v)) for v in forces.values()),default=None)
checks={'expected_shader':inputs['metallib_sha256']==rinputs['metallib_sha256'],'same_policy_inputs':all(inputs[k]==rinputs[k] for k in ('landing_policy_sha256','route_policy_sha256','frozen_flight_sha256')),'teacher_off':stdout['birdflow_journey_teacher'] is False,'same_episode_lengths':meta['steps_by_environment']==base['steps_by_environment'],'same_failure_outcomes':meta['physical_failure_by_environment']==base['physical_failure_by_environment'],'all_first_episode_states_match':all(not x['mismatch_count'] for x in parity),'wrench_samples_complete':missing==0,'wrenches_finite_and_zero':finite and maxforce is not None and maxforce<=1e-9,'no_observer_errors':not errors}
out={'checks':checks,'pass':all(checks.values()),'parity':parity,'wrench_missing':missing,'wrench_max_abs':maxforce,'errors':errors,'scope':'observational host parity control on used seed'};(A/'control-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
