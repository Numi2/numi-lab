from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time

B = Path('/Users/n/cardboard-implementation-20261014-host-macmini')
S = B / 'snapshots/source-checkpoint-warm-breakdown-173'
D = B / 'build-source173-001'
O = B / 'analysis/source173-host-validation-002'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not O.exists()
assert sha(S / 'frozen-source.json') == '6d7e76fdb07702b0b0727fc6610ae979c4e0e161ad92f6888036b1dfef3b53b3'
assert sha(D / 'source-build.json') == '96396de7fa14197a6759583d8b7bb053c412e08f1051b947502c1cc38b185229'
frozen = json.loads((S / 'frozen-source.json').read_text())
build = json.loads((D / 'source-build.json').read_text())
assert build['success'] and build['source_sha256'] == frozen['source_sha256']
for name, digest in frozen['source_sha256'].items():
    assert sha(S / name) == digest, name
for name, digest in build['artifacts_sha256'].items():
    assert sha(name) == digest, name
O.mkdir(parents=True)
environment = {k: v for k, v in os.environ.items() if not k.startswith('NM_FEM_') and k not in {
    'NUMI_CARDBOARD_STEP_TIMING', 'NUMI_CARDBOARD_NATIVE_TIMING', 'NUMI_MATTER_GPU_TIMING_DENSE45'}}
environment['PATH'] = '/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
commands = [
    ['/opt/homebrew/bin/ctest', '--test-dir', str(D), '-R',
     r'^matter\.compiler\.cardboard_(shipping_snapshot|coupon_resume)$', '--output-on-failure', '-j', '1'],
    [sys.executable, '-B', str(S / 'matter/tools/test_cardboard_solver_environment.py')],
    [str(D / 'numi-matter-cardboard-warm-start-check')]]
receipts = []
for index, command in enumerate(commands):
    if index == 2:
        workers = [line for line in subprocess.check_output(['ps', '-axo', 'pid=,stat=,comm='], text=True).splitlines()
                   if any(key in line for key in ('numi-matter-', 'sparse-scan-checker', 'source469-material-replay-check'))
                   and not line.split()[1].startswith('Z')]
        assert not workers, workers
        free = os.statvfs(B)
        assert free.f_bavail * free.f_frsize > 2 * 1024 ** 3
    start = time.monotonic()
    result = subprocess.run(command, env=environment, capture_output=True, timeout=90)
    (O / ('command-%02d.stdout.log' % index)).write_bytes(result.stdout)
    (O / ('command-%02d.stderr.log' % index)).write_bytes(result.stderr)
    row = {'argv': command, 'returncode': result.returncode, 'wall_seconds': time.monotonic() - start,
           'gpu_kernel_execution': index == 2, 'physical_transactions': 0,
           'stdout_sha256': hashlib.sha256(result.stdout).hexdigest(),
           'stderr_sha256': hashlib.sha256(result.stderr).hexdigest()}
    receipts.append(row)
    (O / ('command-%02d.receipt.json' % index)).write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row), flush=True)
    assert result.returncode == 0, result.stderr.decode()
    if index == 2:
        assert re.search(rb'checks=264; metal_kernel_checks_executed=1; physical_transactions=0', result.stdout)
report = {'schema': 'numi.cardboard.macmini-source173-validation.v1', 'host_tag': 'macmini',
    'source_frozen_sha256': sha(S / 'frozen-source.json'), 'source_build_sha256': sha(D / 'source-build.json'),
    'source_files_verified': len(frozen['source_sha256']), 'artifact_sha256': build['artifacts_sha256'],
    'cpu_test_cases': 13, 'native_helper_checks': 264, 'physical_transactions': 0,
    'receipts': receipts, 'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'full_solver_qualification': False, 'original_m4_host_memory_qualification': False}
(O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
(O / 'manifest.json').write_text(json.dumps({'files_sha256': {
    p.name: sha(p) for p in sorted(O.iterdir()) if p.is_file()}}, indent=2) + '\n')
print(json.dumps(report), flush=True)
