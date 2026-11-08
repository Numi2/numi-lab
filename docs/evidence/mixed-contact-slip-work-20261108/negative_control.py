import json,shutil,subprocess,sys
from pathlib import Path
p=Path('negative-control');p.mkdir(exist_ok=True)
shutil.copy('control.json',p/'control.json');shutil.copy('analyze.py',p/'analyze.py')
d=json.loads(Path('trace.json').read_text())
case=next(x for x in d['cases'] if x['iterations']==64 and x['mu']==.5 and x['initial_vn']==-1 and x['dir']==0)
r=next(x for x in case['rows'] if not (x['flags']&8) and x['body_a']==0 and x['body_b']==1)
r['post_relative'][1:]=r['impulse'][1:]
(p/'trace.json').write_text(json.dumps(d,indent=2)+'\n')
r=subprocess.run([sys.executable,'analyze.py'],cwd=p,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(p/'execution.log').write_text(r.stdout);assert r.returncode==0,r.stdout
result=json.loads((p/'analysis.json').read_text())['counts']
assert result['positive_tangent_work']>=1 and result['sphere_independent_velocity_mismatch']>=1,result
print('corrupted slip detected by dissipation and independent velocity checks')
