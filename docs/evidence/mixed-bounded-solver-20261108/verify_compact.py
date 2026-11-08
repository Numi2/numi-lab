import json
from pathlib import Path
before=json.loads(Path('compact.json').read_text())['runs'];after=json.loads(Path('compact-v2.json').read_text())['runs']
assert len(before)==len(after)==3
for old,run in zip(before,after):
 assert run['succeeded'] and run['successful_steps']==4 and not run['failed_steps'] and run['rigid_matches_baseline']
 assert all(run[f'repeated_env{k}'] for k in range(4))
 assert all(e['status']==0 and e['contacts']==1 and e['sweeps']==2 and e['queue_flags']&(1<<28) for e in run['env'])
 assert {k:v for k,v in run.items() if k not in ('repeated_env2','repeated_env3')}==old
print('8-lane compact packet: four environments and all exposed repeats match')
