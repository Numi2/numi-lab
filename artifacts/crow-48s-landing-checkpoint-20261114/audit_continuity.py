import gzip, json, math
from pathlib import Path

root=Path(__file__).parent
runs=['near-perch-250','near-perch-landingmode-250','near-perch-landingmode-500',
      'pitch-balance-negative-600','pitch-balance-positive-600','pitch-balance-negative-1000']
results={}
for run in runs:
    path=root/run
    if not (path/'replay.json').exists(): continue
    payload=json.loads((path/'replay.json').read_text())['payload']
    ends=[{'step':f['step'],'reason':f['termination_reason'],'timeout':f['timeout']}
          for f in payload['frames'] if f['done']]
    jumps=[];previous=None
    with gzip.open(path/'stderr.gz','rt') as stream:
        for line in stream:
            if not line.startswith('CROW_SUPPORT_SAMPLE '): continue
            sample=json.loads(line.split(' ',1)[1])
            if previous is not None:
                displacement=math.sqrt(sum((sample['q'][i]-previous['q'][i])**2 for i in range(3)))
                if displacement>.03:
                    jumps.append({'before_observer_step':previous['step'],'after_observer_step':sample['step'],
                                  'displacement_m':displacement})
            previous=sample
    results[run]={'replay_done_events':ends,'root_jumps_over_3cm':jumps,
                  'continuous_from_checkpoint':not ends and not jumps,
                  'note':'Observer and replay indices differ; termination reason 2 is MR_TASK_TERMINATION_TILT.'}
(root/'continuity-audit.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
