import datetime,subprocess
from pathlib import Path
d=Path(__file__).resolve().parent;root=d.parents[2];b=Path('/Users/numi/numi-lab-main-upload-limited-build-20261108')
if (d/'preflight.sha256-check').exists():raise SystemExit('refusing overwrite')
with (d/'preflight.sha256-check').open('w') as f:
 if subprocess.run(['shasum','-a','256','-c','declared-identities.sha256'],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode:raise SystemExit('preflight failed')
cases=[('acceptance',['python3',str(root/'tests/ccd/world/regression.py'),str(b/'ccd-world'),str(b/'MetalRobo.metallib'),str(d/'acceptance')]),('kinematic',['python3',str(root/'tests/ccd/world/kinematic_rollback_regression.py'),str(b/'ccd-kinematic'),str(b/'MetalRobo.metallib')])]
for name,command in cases:
 start=datetime.datetime.now(datetime.timezone.utc)
 try:
  with (d/(name+'.stdout')).open('w') as out,(d/(name+'.stderr')).open('w') as err:rc=str(subprocess.run(command,cwd=root,stdout=out,stderr=err,timeout=180).returncode)
 except subprocess.TimeoutExpired:rc='timeout'
 end=datetime.datetime.now(datetime.timezone.utc);(d/(name+'.timing')).write_text(f'start_utc={start.isoformat()}\nend_utc={end.isoformat()}\nexit={rc}\n');print(f'{name} exit={rc} start={start.isoformat()} end={end.isoformat()}',flush=True)
 if rc!='0':break
with (d/'postflight.sha256-check').open('w') as f:post=subprocess.run(['shasum','-a','256','-c','declared-identities.sha256'],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode
print(f'postflight={post}',flush=True)
