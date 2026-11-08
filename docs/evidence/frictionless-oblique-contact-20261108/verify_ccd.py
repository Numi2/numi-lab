import json
from pathlib import Path
p=Path(__file__).resolve().parent
a=json.loads((p/'ccd-baseline.json').read_text())['runs'];b=json.loads((p/'ccd-candidate.json').read_text())['runs']
assert len(a)==len(b)==12
for x,y in zip(a,b):
 x.pop('elapsed_with_initialization');y.pop('elapsed_with_initialization')
 assert x==y,(x,y)
 assert x['repeat_identical'] and x['rejected_state_restored'] and x['rejected_contact_cache_empty']
assert sum(not x['accepted'] for x in a if x['fixture'] in (12,14))==4
assert all(x['accepted'] and x['normal_impulse_Ns']==0 for x in a if x['fixture'] in (13,15))
print('all 12 match baseline exactly excluding timing; four thin impacts reject; four near misses accepted with zero impulse; repeats and exposed rollback pass')
