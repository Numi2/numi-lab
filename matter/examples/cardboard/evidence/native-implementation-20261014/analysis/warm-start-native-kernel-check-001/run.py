from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time

B = Path('/Users/home/cardboard-implementation-20261014')
P = B / 'analysis/base-cgls-warm-start-refinement-overlay-159-002'
O = B / 'analysis/warm-start-native-kernel-check-001'
O.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = P / 'source-overlay.json'
assert sha(manifest) == '29043b640b6fde5fc66c864ff84d80400cb09771fb5e455368afdd7ae76c7f8d'
record = json.loads(manifest.read_text())
executable = P / 'native/build/warm_start_check'
library = P / 'native/build/WarmStartCheck.metallib'
assert sha(executable) == record['validation']['checker_executable_linked_not_run']['sha256']
assert sha(library) == record['validation']['metallib']['sha256']
source = B / 'snapshots/source-right-grip-telemetry-157'
frozen = json.loads((source / 'frozen-source.json').read_text())['source_sha256']
for name, digest in frozen.items():
    assert sha(source / name) == digest, name
processes = subprocess.check_output(['ps', '-axo', 'pid=,stat=,comm='], text=True)
active = [line for line in processes.splitlines()
          if (line.split(None, 2)[-1].endswith('/numi-matter-cardboard-probe')
              or line.split(None, 2)[-1].endswith('/warm_start_check'))]
assert not active, active
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
begin = time.monotonic()
run = subprocess.run([str(executable)], capture_output=True, timeout=60)
(O / 'stdout.log').write_bytes(run.stdout)
(O / 'stderr.log').write_bytes(run.stderr)
receipt = {'schema': 'numi.cardboard.native-warm-start-kernel-check.v1',
           'package_manifest_sha256': sha(manifest), 'argv': [str(executable)],
           'executable_sha256': sha(executable), 'metallib_sha256': sha(library),
           'baseline_sources_verified': len(frozen), 'started_at_utc': started,
           'wall_seconds': time.monotonic() - begin, 'returncode': run.returncode,
           'gpu_execution': True, 'physical_transactions': 0,
           'full_solver_or_physical_qualification': False}
(O / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
shutil.copyfile(__file__, O / 'run.py')
(O / 'manifest.json').write_text(json.dumps({p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(O.iterdir()) if p.is_file()}, indent=2) + '\n')
print(json.dumps(receipt))
print(run.stdout.decode())
print(run.stderr.decode())
