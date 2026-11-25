import datetime,subprocess
from pathlib import Path
d=Path(__file__).resolve().parent;root=d.parents[2]
if (d/'preflight.sha256-check').exists():raise SystemExit('refusing overwrite')
with (d/'preflight.sha256-check').open('w') as f:
 if subprocess.run(['shasum','-a','256','-c','declared-identities.sha256'],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode:raise SystemExit('preflight failed')
shader='/Users/numi/numi-lab-main-upload-limited-build-20261108/MetalRobo.metallib'
for name,binary in [('initial','/Users/numi/numi-lab-main-upload-limited-build-20261108/probe-candidate'),('nonempty','/Users/numi/numi-lab-nonempty-upload-build-20261108/probe-candidate')]:
 start=datetime.datetime.now(datetime.timezone.utc)
 try:
  with (d/(name+'.stdout')).open('w') as o,(d/(name+'.stderr')).open('w') as e:
   rc=str(subprocess.run(['python3',str(root/'tests/ccd/world/upload_abort_regression.py'),binary,shader],cwd=root,stdout=o,stderr=e,timeout=300).returncode)
 except subprocess.TimeoutExpired:rc='timeout'
 end=datetime.datetime.now(datetime.timezone.utc);(d/(name+'.timing')).write_text(f'start_utc={start.isoformat()}\nend_utc={end.isoformat()}\nexit={rc}\n');print(f'{name} exit={rc} start={start.isoformat()} end={end.isoformat()}',flush=True)
 if rc!='0':break
with (d/'postflight.sha256-check').open('w') as f:post=subprocess.run(['shasum','-a','256','-c','declared-identities.sha256'],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode
print(f'postflight={post}',flush=True)
