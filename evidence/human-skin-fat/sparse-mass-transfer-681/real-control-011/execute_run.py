import datetime, hashlib, json, subprocess, sys, time
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'prelaunch.json').read_text())
run = manifest['run']
start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
start = time.monotonic()
with open(run['stdout_path'], 'wb') as out, open(run['stderr_path'], 'wb') as err:
    process = subprocess.run(run['argv'], cwd=run['cwd'], stdout=out, stderr=err, check=False)
elapsed = time.monotonic() - start
end_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
Path(run['exit_code_path']).write_text(str(process.returncode) + '\n')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
(root / 'timing.json').write_text(json.dumps({
    'start_utc': start_utc, 'end_utc': end_utc, 'elapsed_seconds': elapsed,
    'exit_code': process.returncode,
    'stdout_sha256': sha(run['stdout_path']), 'stderr_sha256': sha(run['stderr_path'])
}, indent=2) + '\n')
print(json.dumps({'exit_code': process.returncode, 'elapsed_seconds': elapsed,
                  'stdout_path': run['stdout_path'], 'stderr_path': run['stderr_path']}))
