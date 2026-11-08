"""Exact output repeat, omitting diagnostic elapsed time."""
import json
from pathlib import Path
assert Path('symmetry-final.json').read_bytes()==Path('symmetry-final-repeat.json').read_bytes()
a=json.loads(Path('dynamics-n4.json').read_text());b=json.loads(Path('dynamics-n4-repeat.json').read_text())
for case in ('full','quarter'):
 del a[case]['elapsed_s'];del b[case]['elapsed_s']
assert a==b
print('static and n4 dynamic non-timing outputs reproduced exactly')
