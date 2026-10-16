from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

B = Path('/Users/home/cardboard-implementation-20261014')
S = B / 'snapshots/source-checkpoint-warm-breakdown-173'
D = Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-173')
O = B / 'analysis/source173-build-cpu-validation-001'
PY = '/Users/home/.pyenv/versions/3.10.12/bin/python3'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not O.exists()
    frozen = json.loads((S / 'frozen-source.json').read_text())
    built = json.loads((D / 'source-build.json').read_text())
    assert sha(S / 'frozen-source.json') == '6d7e76fdb07702b0b0727fc6610ae979c4e0e161ad92f6888036b1dfef3b53b3'
    assert built['success'] is True and not built['changed_sources']
    assert frozen['source_sha256'] == built['source_sha256'] and len(frozen['source_sha256']) == 804
    assert built['targets'] == ['numi-matter-cardboard-probe', 'numi-matter-cardboard-warm-start-check',
                               'numi-matter-cardboard-shipping-snapshot-check', 'numi-matter-cardboard-coupon-resume-check']
    for name, digest in frozen['source_sha256'].items():
        assert sha(S / name) == digest, name
    for name, digest in built['artifacts_sha256'].items():
        assert sha(name) == digest, name
    for name, digest in built['build_metadata_sha256'].items():
        assert sha(D / name) == digest, name
    O.mkdir()
    tests = [
        ('checkpoint-ctest', ['ctest', '--test-dir', str(D), '-R',
            '^matter\\.compiler\\.cardboard_(shipping_snapshot|coupon_resume)$', '--output-on-failure', '-j', '1'],
            '100% tests passed, 0 tests failed out of 2'),
        ('analysis-ctest', ['ctest', '--test-dir', str(D), '-L', 'cpu', '-R', '^matter\\.analysis\\.',
            '--output-on-failure', '-j', '1'], '100% tests passed, 0 tests failed out of 14'),
        ('resolution-ctest', ['ctest', '--test-dir', str(D), '-R',
            '^matter\\.analysis\\.cardboard_(resolution_study|convergence)$', '--output-on-failure', '-j', '1'],
            '100% tests passed, 0 tests failed out of 2'),
        ('environment-unittest', [PY, '-B', str(S / 'matter/tools/test_cardboard_solver_environment.py')], 'Ran 11 tests'),
    ]
    receipts = []
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for name, command, expected in tests:
        start = time.monotonic()
        result = subprocess.run(command, cwd=S, env=environment, capture_output=True)
        (O / (name + '.stdout.log')).write_bytes(result.stdout)
        (O / (name + '.stderr.log')).write_bytes(result.stderr)
        receipt = {'argv': command, 'cwd': str(S), 'exit_code': result.returncode,
            'elapsed_seconds': time.monotonic() - start,
            'stdout_sha256': sha(O / (name + '.stdout.log')),
            'stderr_sha256': sha(O / (name + '.stderr.log'))}
        (O / (name + '.receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
        receipts.append(receipt)
        text = result.stdout.decode() + result.stderr.decode()
        assert result.returncode == 0 and expected in text, name
        assert not re.search(r'^\s*\d+/\d+ Test.*matter\.metal\.', text, re.M), name
    report = {'schema': 'numi.cardboard.source173.cpu-validation.v1',
        'frozen_source_sha256': sha(S / 'frozen-source.json'), 'source_files_verified': 804,
        'build_record_sha256': sha(D / 'source-build.json'), 'build_targets': built['targets'],
        'artifact_sha256': built['artifacts_sha256'], 'build_metadata_sha256': built['build_metadata_sha256'],
        'cpu_test_cases_reported': 29, 'receipts': receipts,
        'gpu_executions': 0, 'physical_transactions': 0,
        'qualification': 'Successful source-bound native build and focused CPU contract validation; no native trajectory, checkpoint restore, convergence or physical qualification.'}
    (O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    shutil.copy2(__file__, O / 'validate.py')
    (O / 'manifest.json').write_text(json.dumps({'files_sha256': {
        str(p.relative_to(O)): sha(p) for p in sorted(O.rglob('*')) if p.is_file()}}, indent=2) + '\n')
    print(json.dumps({'report_sha256': sha(O / 'report.json'), 'manifest_sha256': sha(O / 'manifest.json'),
        'cpu_test_cases_reported': 29, 'gpu_executions': 0}))

if __name__ == '__main__':
    main()
