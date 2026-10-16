from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
S = B / 'snapshots/source-warm-cgls-refinement-169'
O = B / 'analysis/warm-start-native-kernel-check-002'
A = B / 'analysis/source169-build-cpu-validation-001'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(A / 'manifest.json') == '75a84e24dc81219ab8f8f275a329d14cece6d4c826f7cde3790c25d2cf0bc27c'
assert sha(S / 'frozen-source.json') == 'f5054091c27a7e1add41630aae1b45ee6ecc3e6d4112ee85c931cc4c70f44ea4'
sources = json.loads((S / 'frozen-source.json').read_text())['source_sha256']
assert len(sources) == 804
for name, expected in sources.items():
    assert sha(S / name) == expected, name
audit = json.loads((A / 'verification.json').read_text())
package = json.loads((A / 'manifest.json').read_text())
for row in package['files']:
    assert sha(A / row['path']) == row['sha256'], row['path']
for row in audit['build_artifacts_verified'] + audit['build_metadata_verified']:
    assert sha(Path(row['path'])) == row['sha256'], row['path']
build = R / 'build-cardboard-provenance-169'
exe = build / 'numi-matter-cardboard-warm-start-check'
library = build / 'shaders/CardboardWarmStartCheck.metallib'
processes = subprocess.check_output(['ps', '-axo', 'pid=,stat=,comm='], text=True)
active = []
preserved_stopped_debugger_targets = []
for line in processes.splitlines():
    fields = line.split(None, 2)
    if len(fields) == 3 and not fields[1].startswith('Z') and (
            'numi-matter-' in Path(fields[2]).name or
            Path(fields[2]).name in {'warm_start_check', 'sparse_scan_check'}):
        if fields[0] == '77441' and fields[1] == 'T' and fields[2] == str(
                R / 'build-cardboard-world/matter/numi-matter-cardboard-paper-material-check'):
            parent = subprocess.check_output(['ps', '-p', fields[0], '-o', 'ppid='], text=True).strip()
            debugger = subprocess.check_output(['ps', '-p', parent, '-o', 'comm='], text=True).strip()
            assert parent == '76106' and Path(debugger).name == 'lldb'
            preserved_stopped_debugger_targets.append({'pid': int(fields[0]), 'state': fields[1],
                'command': fields[2], 'parent_pid': int(parent), 'parent_command': debugger})
        else:
            active.append(line)
assert not active, active
O.mkdir(exist_ok=False)
env = {k: v for k, v in os.environ.items() if not k.startswith('NM_FEM_')}
removed = sorted(set(os.environ) - set(env))
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
begin = time.monotonic()
timed_out = False
try:
    run = subprocess.run([str(exe)], capture_output=True, timeout=60, env=env)
    stdout, stderr, code = run.stdout, run.stderr, run.returncode
except subprocess.TimeoutExpired as error:
    stdout, stderr, code = error.stdout or b'', error.stderr or b'', -9
    timed_out = True
(O / 'stdout.log').write_bytes(stdout)
(O / 'stderr.log').write_bytes(stderr)
receipt = {'schema': 'numi.cardboard.native-warm-start-kernel-check.v2',
           'source_snapshot': str(S), 'frozen_source_sha256': sha(S / 'frozen-source.json'),
           'cpu_validation_manifest_sha256': sha(A / 'manifest.json'),
           'argv': [str(exe)], 'executable_sha256': sha(exe),
           'metallib_sha256': sha(library), 'source_files_verified': len(sources),
           'started_at_utc': started, 'wall_seconds': time.monotonic() - begin,
           'returncode': code, 'timed_out': timed_out, 'gpu_execution': True,
           'physical_transactions': 0, 'removed_parent_environment_keys': removed,
           'preexisting_stopped_debugger_targets_preserved': preserved_stopped_debugger_targets,
           'full_solver_or_physical_qualification': False,
           'prior_failed_checker': str(B / 'analysis/warm-start-native-kernel-check-001')}
(O / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(O / 'run.py').write_bytes(Path(__file__).read_bytes())
(O / 'manifest.json').write_text(json.dumps({p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)}
     for p in sorted(O.iterdir()) if p.is_file()}, indent=2) + '\n')
print(json.dumps(receipt))
print(stdout.decode())
print(stderr.decode())
