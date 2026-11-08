import json,subprocess,sys,tempfile
from pathlib import Path
here=Path.cwd();results={}
for name in ('corrupt-target','corrupt-geometry'):
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp)
  for f in ('baseline.json','witness.json','velocity.json','analyze.py'):(root/f).write_bytes((here/f).read_bytes())
  p=root/'witness.json';d=json.loads(p.read_text())
  if name=='corrupt-target':d['steps'][7]['rows'][1]['scratch'][11]+=0.001
  else:d['steps'][7]['rows'][1]['point'][0]+=0.001
  p.write_text(json.dumps(d))
  r=subprocess.run([sys.executable,'analyze.py'],cwd=root,capture_output=True,text=True)
  results[name]={'exit_code':r.returncode,'detected':r.returncode!=0,'stderr':r.stderr}
assert all(x['detected'] for x in results.values())
Path('negative-control.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
