import json
from pathlib import Path
p=Path(__file__).parent
base=json.loads((p/'baseline.json').read_text())['cases']
fixed=json.loads((p/'corrected.json').read_text())['cases']
assert len(base)==len(fixed)==36
different=[]
for i,(a,b) in enumerate(zip(base,fixed)):
 a=dict(a);b=dict(b);a.pop('solver_residual');b.pop('solver_residual')
 if a!=b:different.append(i)
assert not different,different
print('36 cases: all exposed records except the normal residual match exactly')
