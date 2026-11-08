import datetime,subprocess
from pathlib import Path
d=Path(__file__).resolve().parent;b=Path('/Users/numi/numi-lab-main-upload-shared-baseline-build-20261108')
if (d/'preflight.sha256-check').exists():raise SystemExit('refusing overwrite')
with (d/'preflight.sha256-check').open('w') as f:
 if subprocess.run(['shasum','-a','256','-c','declared-identities.sha256'],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode:raise SystemExit('preflight failed')
start=datetime.datetime.now(datetime.timezone.utc)
try:
 with (d/'control.stdout').open('w') as o,(d/'control.stderr').open('w') as e:
  rc=str(subprocess.run([str(b/'probe-baseline'),str(b/'MetalRobo.metallib'),str(d/'control.bin'),'control','shared'],cwd=d,stdout=o,stderr=e,timeout=120).returncode)
except subprocess.TimeoutExpired:rc='timeout'
end=datetime.datetime.now(datetime.timezone.utc)
(d/'control.timing').write_text(f'start_utc={start.isoformat()}\nend_utc={end.isoformat()}\nexit={rc}\n')
with (d/'postflight.sha256-check').open('w') as f:post=subprocess.run(['shasum','-a','256','-c','declared-identities.sha256'],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode
print(f'exit={rc} start={start.isoformat()} end={end.isoformat()} postflight={post}',flush=True)
