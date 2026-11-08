import json
from pathlib import Path
a=json.loads(Path('isolation.json').read_text())['runs'];b=json.loads(Path('isolation-count-fixed.json').read_text())['runs']
assert a[:2]==b[:2]
old=a[2];new=b[2]
assert old['env'][0]['status']==new['env'][0]['status']==11
assert old['env'][0]['sweeps']==0 and new['env'][0]['sweeps']==2
assert old['env'][0]['normal_residual']==new['env'][0]['normal_residual']
assert old['env'][0]['restored'] and new['env'][0]['restored']
assert old['env'][1]==new['env'][1] and old['rigid_matches_baseline'] and new['rigid_matches_baseline']
assert all(x['repeated_env0'] and x['repeated_env1'] for x in b)
assert b[1]['env'][0]['status']==0 and b[1]['env'][0]['normal_residual']<=.002
assert b[1]['env'][1]['status']==0 and b[1]['rigid_matches_baseline']
print('two-environment isolation, status count, rollback and repeat checks passed')
