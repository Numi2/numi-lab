import datetime,subprocess
from pathlib import Path
d=Path(__file__).resolve().parent;b=Path('/Users/numi/numi-lab-upload-cmake-config-20261108')
if (d/'preflight.sha256-check').exists():raise SystemExit('refusing overwrite')
with (d/'preflight.sha256-check').open('w') as f:
 if subprocess.run(['shasum','-a','256','-c','declared-identities.sha256'],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode:raise SystemExit('preflight failed')
start=datetime.datetime.now(datetime.timezone.utc)
try:
 with (d/'ctest.stdout').open('w') as o,(d/'ctest.stderr').open('w') as e:
  rc=str(subprocess.run(['ctest','--test-dir',str(b),'-R','^metalrobo\\.upload_abort\\.','--output-on-failure'],stdout=o,stderr=e,timeout=650).returncode)
except subprocess.TimeoutExpired:rc='timeout'
end=datetime.datetime.now(datetime.timezone.utc);(d/'ctest.timing').write_text(f'start_utc={start.isoformat()}\nend_utc={end.isoformat()}\nexit={rc}\n')
with (d/'postflight.sha256-check').open('w') as f:post=subprocess.run(['shasum','-a','256','-c','declared-identities.sha256'],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode
print(f'ctest exit={rc} start={start.isoformat()} end={end.isoformat()} postflight={post}',flush=True)
