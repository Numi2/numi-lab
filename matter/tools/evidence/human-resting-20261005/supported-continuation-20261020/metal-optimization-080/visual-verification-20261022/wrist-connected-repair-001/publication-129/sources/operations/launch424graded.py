from pathlib import Path
import json,hashlib,subprocess,time
D=Path('/Users/numi/Work/numi-human-connected-surface-069');P=D/'D/postfit-backtrack-424/run-001';py='/Users/n/numi-human-prep-venv-20261005/bin/python';w=D/'D/local-backtracking-config-419/run_validated_local_backtracking_419.py';cfg=P/'backtrack-config.json'
a=[py,str(w),str(cfg),'--timeout-seconds','900'];start=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(f):sha(f)for f in (list(w.parent.glob('*.py'))+list((D/'D/graded-local-step-423').glob('*.py'))+[cfg])}
(P/'launch.json').write_text(json.dumps(dict(argv=a,pins=pins,native_admission=False,accepted_physical_steps=0),indent=2)+'\n')
with(P/'execution.log').open('x')as f:r=subprocess.run(a,stdout=f,stderr=subprocess.STDOUT)
assert all(sha(k)==v for k,v in pins.items())
o=dict(returncode=r.returncode,wall_seconds=time.monotonic()-start,log_sha256=sha(P/'execution.log'),native_admission=False,accepted_physical_steps=0)
(P/'execution.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o))

