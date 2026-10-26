from pathlib import Path
import subprocess,json,time,hashlib
p=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-415/run-001-launch');a=json.load(open(p/'argv.json'));j=json.load(open(p/'preflight.json'))
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
assert sha(p/'argv.json')==j['argv_sha256']
for s,v in j['input_manifest'].items():assert sha(s)==v['sha256'],s
assert not Path(a[a.index('--output')+1]).exists()
cmd=[a[0],'/tmp/numi-human-geometry-slot.py',*a];start=time.monotonic()
with (p/'fit.log').open('x')as log:cp=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
r=dict(argv=cmd,exit_code=cp.returncode,wall_seconds=time.monotonic()-start,log_sha256=sha(p/'fit.log'),preflight_sha256=sha(p/'preflight.json'),native_admission=False,accepted_physical_steps=0)
(p/'execution.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));assert cp.returncode==0

