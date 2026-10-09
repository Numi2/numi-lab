from pathlib import Path
import json,subprocess,time,hashlib
r=Path(__file__).resolve().parent
d=json.loads((r/'run-declaration.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for p,h in d['immutable_assets'].items():assert sha(p)==h,p
start=time.monotonic()
with (r/'owner-stdout.log').open('w') as log:
 result=subprocess.run(d['argv'],cwd=r,stdout=log,stderr=subprocess.STDOUT)
changed={p:{'before':h,'after':sha(p)} for p,h in d['immutable_assets'].items() if sha(p)!=h}
(r/'execution.json').write_text(json.dumps({'returncode':result.returncode,'wall_seconds':time.monotonic()-start,'changed_inputs':changed,'declaration_sha256':sha(r/'run-declaration.json')},indent=2)+'\n')
print((r/'execution.json').read_text(),flush=True)
