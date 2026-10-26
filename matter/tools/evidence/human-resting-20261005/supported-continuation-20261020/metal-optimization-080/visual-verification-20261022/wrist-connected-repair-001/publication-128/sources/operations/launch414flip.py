from pathlib import Path
import json,hashlib,time,subprocess
p=Path('/Users/numi/Work/numi-human-connected-surface-069/D/local-two-parent-retriangulation-414/run-001-pass002');j=json.load(open(p/'launch.json'))
for f,h in j['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
a=j['argv'];start=time.monotonic()
with(p/'execution.log').open('x')as log:r=subprocess.run([a[0],'/tmp/numi-human-geometry-slot.py',*a],stdout=log,stderr=subprocess.STDOUT)
o=dict(returncode=r.returncode,wall_seconds=time.monotonic()-start,log_sha256=hashlib.sha256((p/'execution.log').read_bytes()).hexdigest(),native_admission=False,accepted_physical_steps=0)
(p/'execution.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o))

