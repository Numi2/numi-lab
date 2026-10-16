from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
S = B / 'snapshots/source-checkpoint-warm-breakdown-173'
D = R / 'build-cardboard-provenance-173'
A = B / 'analysis/source173-build-cpu-validation-001'
O = B / 'analysis/source173-warm-kernel-check-001'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not O.exists()
    assert sha(S / 'frozen-source.json') == '6d7e76fdb07702b0b0727fc6610ae979c4e0e161ad92f6888036b1dfef3b53b3'
    assert sha(D / 'source-build.json') == '0306631506b565f12a77cd44fb3acaabb012cb0dbeefe1e0eb82a9befd43b9a2'
    assert sha(A / 'manifest.json') == '50d69d976294b19b3dd21d25adc7627ef607361846b9486792828c2b2ac72cf9'
    frozen = json.loads((S / 'frozen-source.json').read_text())
    built = json.loads((D / 'source-build.json').read_text())
    assert len(frozen['source_sha256']) == 804 and built['source_sha256'] == frozen['source_sha256']
    for name, digest in frozen['source_sha256'].items():
        assert sha(S / name) == digest, name
    for name, digest in built['artifacts_sha256'].items():
        assert sha(name) == digest, name
    for name, digest in json.loads((A / 'manifest.json').read_text())['files_sha256'].items():
        assert sha(A / name) == digest, name
    cpu = json.loads((A / 'report.json').read_text())
    assert cpu['cpu_test_cases_reported'] == 29 and cpu['gpu_executions'] == 0
    previous = B / 'solver/source169-default-off-parity-472/receipt.json'
    assert previous.is_file() and json.loads(previous.read_text())['exit_code'] is not None
    active = []
    preserved = []
    for line in subprocess.check_output(['ps', '-axo', 'pid=,ppid=,stat=,comm='], text=True).splitlines():
        fields = line.split(None, 3)
        if len(fields) != 4 or fields[2].startswith('Z'):
            continue
        if 'numi-matter-' not in Path(fields[3]).name and Path(fields[3]).name not in {'warm_start_check', 'sparse_scan_check'}:
            continue
        if fields[:3] == ['77441', '76106', 'T'] and fields[3] == str(R / 'build-cardboard-world/matter/numi-matter-cardboard-paper-material-check'):
            debugger = subprocess.check_output(['ps', '-p', '76106', '-o', 'comm='], text=True).strip()
            assert Path(debugger).name == 'lldb'
            preserved.append(line)
        else:
            active.append(line)
    assert not active, active
    O.mkdir()
    command = [str(D / 'numi-matter-cardboard-warm-start-check')]
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith('NM_FEM_') and key not in {
                       'NUMI_CARDBOARD_STEP_TIMING', 'NUMI_CARDBOARD_NATIVE_TIMING',
                       'NUMI_MATTER_GPU_TIMING_DENSE45'}}
    start = time.monotonic()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    timed_out = False
    try:
        result = subprocess.run(command, capture_output=True, timeout=60, env=environment)
        stdout, stderr, code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr, code = error.stdout or b'', error.stderr or b'', -9
        timed_out = True
    (O / 'stdout.log').write_bytes(stdout)
    (O / 'stderr.log').write_bytes(stderr)
    match = re.search(rb'checks=(\d+); metal_kernel_checks_executed=1; physical_transactions=0', stdout)
    receipt = {'schema': 'numi.cardboard.source173.warm-kernel-check.v1',
        'argv': command, 'frozen_source_sha256': sha(S / 'frozen-source.json'),
        'source_build_sha256': sha(D / 'source-build.json'),
        'cpu_manifest_sha256': sha(A / 'manifest.json'), 'previous472_receipt_sha256': sha(previous),
        'started_at_utc': started, 'wall_seconds': time.monotonic() - start,
        'returncode': code, 'timed_out': timed_out, 'checks_reported': int(match[1]) if match else None,
        'gpu_execution': True, 'physical_transactions': 0,
        'stopped_debugger_targets_preserved': preserved,
        'removed_parent_environment_keys': sorted(set(os.environ) - set(environment)),
        'production_material_operator_or_full_cgls_recurrence_exercised': False,
        'full_solver_or_physical_qualification': False}
    (O / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    shutil.copy2(__file__, O / 'run.py')
    (O / 'manifest.json').write_text(json.dumps({'files_sha256': {
        p.name: sha(p) for p in sorted(O.iterdir()) if p.is_file()}}, indent=2) + '\n')
    print(json.dumps(receipt))
    print(stdout.decode())
    print(stderr.decode())
    assert code == 0 and not timed_out and match is not None

if __name__ == '__main__':
    main()
