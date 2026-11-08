from pathlib import Path
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True
R = Path.cwd()
A = Path('artifacts/crow-rl-implicit-drive-audit-20261108')
G = Path('artifacts/crow-rl-git-recovery-implicit-drive-20261108')
M = A / 'payload/manifest.json'
W = Path('build/crow-implicit-drive-git-recovery-01')
B = Path('build/crow-implicit-drive-audit-build-01')
T = Path('.numi/retained-runtime/implicit-drive-audit-20261108')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def df():
    return subprocess.check_output(['df', '-k', '/System/Volumes/Data'], text=True)


assert not W.exists(), 'Recovery must use a fresh destination'
assert subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == ''
files = json.loads(M.read_text())['files']
by_path = {f['path']: f for f in files}
commands = [
    {'id': 'calibration', 'binary': str(B / 'probe'), 'args': ['--calibrate'],
     'exit': 0, 'stdout_reference': str(B / 'calibration.stdout')},
    {'id': 'production', 'binary': str(B / 'probe'), 'args': ['SHADER', 'NEW_TSV'],
     'exit': 0, 'tsv_reference': str(B / 'native.tsv'), 'rows': 96},
    {'id': 'invalid-metadata', 'binary': str(B / 'bounded-probe'),
     'args': ['SHADER', 'NEW_TSV'], 'exit': 1,
     'tsv_reference': str(B / 'bounded.tsv'), 'rows': 0,
     'stderr_tail': 'Bounded motor IR invalid: authored constraint payload count is invalid'},
    {'id': 'bounded-v2', 'binary': str(B / 'bounded-v2-probe'),
     'args': ['SHADER', 'NEW_TSV'], 'exit': 0,
     'tsv_reference': str(B / 'bounded-v2.tsv'), 'rows': 144},
]
plan = {
    'purpose': 'Execute exact committed native evidence from Git-restored inputs before raw cleanup.',
    'manifest': str(M), 'manifest_sha256': sha(M),
    'commands': commands,
    'inputs': {str(p): by_path[str(p)]['sha256'] for p in [
        B / 'probe', B / 'bounded-probe', B / 'bounded-v2-probe',
        T / 'libmetalrobo.dylib', T / 'metalrobo/MetalRobo.metallib']},
    'criteria': 'Exact calibration stdout and all three native TSV byte sequences, including original pre-physics rejection; expected exit codes and restored dylib load identity. No scientific outcome promotion.',
    'budget': {'invocations': 4, 'native_microstep_conditions': 240,
               'timeout_per_invocation_seconds': 90, 'recovery_scratch_max_bytes': 16000000},
    'scope': 'One 96-condition production diagnosis, one 144-condition bounded alternative and invalid-metadata fixture. Does not replay whole crow release, learning, biological or throughput qualification.',
    'cleanup': 'Record exact path/hash/mode/mtime inventory, persist lsof return/stdout/stderr, validate unchanged files and closed handles, then remove only enumerated recovery files. Keep all replay logs compressed in Git.',
    'script_sha256': sha(Path(__file__)),
}
write(G / 'plan.json', plan)
subprocess.run(['git', 'add', str(G / 'plan.json'), str(G / 'recover.py')], check=True)
subprocess.run(['git', 'commit', '--quiet', '-m', 'Seal native implicit motor Git recovery'], check=True)
subprocess.run(['git', 'push', 'origin', 'HEAD'], check=True)
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert sha(M) == plan['manifest_sha256'] and sha(Path(__file__)) == plan['script_sha256']

spec = importlib.util.spec_from_file_location('crow_evidence_archive', R / 'tools/crow_evidence_archive.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)
W.mkdir()
archive.verify(M, restore=W)
results = []
try:
    for c in commands:
        binary = R / W / c['binary']
        assert sha(binary) == by_path[c['binary']]['sha256']
        shader = R / W / T / 'metalrobo/MetalRobo.metallib'
        output = R / W / (c['id'] + '.tsv')
        args = [str(shader) if a == 'SHADER' else str(output) if a == 'NEW_TSV' else a for a in c['args']]
        env = os.environ.copy()
        env.update(DYLD_LIBRARY_PATH=str(R / W / T), DYLD_PRINT_LIBRARIES='1')
        stdout = W / (c['id'] + '.stdout')
        stderr = W / (c['id'] + '.stderr')
        start = time.monotonic()
        with stdout.open('xb') as out, stderr.open('xb') as err:
            run = subprocess.run([str(binary), *args], env=env, stdout=out, stderr=err, timeout=90)
        result = {'id': c['id'], 'command': [str(binary), *args], 'exit': run.returncode,
                  'seconds': time.monotonic() - start}
        for stream in (stdout, stderr):
            dest = G / (stream.name + '.gz')
            dest.write_bytes(gzip.compress(stream.read_bytes(), mtime=0))
        results.append(result)
        write(G / 'partial-results.json', {'sealed_head': head, 'commands': results})
        assert run.returncode == c['exit'], result
        loads = [s for s in stderr.read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
        assert loads and all(s.endswith(str(R / W / T / 'libmetalrobo.dylib')) for s in loads), loads
        result['loader_lines'] = loads
        if c.get('stderr_tail'):
            assert stderr.read_text().splitlines()[-1] == c['stderr_tail']
        if c.get('stdout_reference'):
            assert stdout.read_bytes() == (W / c['stdout_reference']).read_bytes()
            result['stdout_byte_exact'] = True
        if c.get('tsv_reference'):
            reference = W / c['tsv_reference']
            assert output.read_bytes() == reference.read_bytes(), c['id'] + ' TSV differs'
            with output.open() as stream:
                rows = list(csv.DictReader(stream, delimiter='\t'))
            assert len(rows) == c['rows']
            result.update(tsv_byte_exact=True, rows=len(rows), tsv_sha256=sha(output))
        print(json.dumps({'completed': c['id'], 'exit': result['exit'], 'exact': True}), flush=True)

    inventory = []
    for p in sorted(W.rglob('*')):
        assert not p.is_symlink()
        if not p.is_file():
            continue
        s = p.stat()
        assert stat.S_ISREG(s.st_mode)
        inventory.append({'path': str(p), 'bytes': s.st_size, 'sha256': sha(p),
                          'mode': stat.S_IMODE(s.st_mode), 'mtime_ns': s.st_mtime_ns})
    total = sum(f['bytes'] for f in inventory)
    assert total <= plan['budget']['recovery_scratch_max_bytes']
    before = df()
    handles = subprocess.run(['/usr/sbin/lsof', '-Fpn', '+D', str(W)], capture_output=True, text=True, timeout=60)
    guard = {'exit': handles.returncode, 'stdout': handles.stdout, 'stderr': handles.stderr,
             'files': inventory, 'bytes': total, 'df_before': before}
    write(G / 'cleanup-guard.json', guard)
    assert handles.returncode in (0, 1) and not handles.stdout and not handles.stderr, 'Recovery handles remain; guard retained'
    for f in inventory:
        p = Path(f['path'])
        s = p.stat()
        assert s.st_size == f['bytes'] and stat.S_IMODE(s.st_mode) == f['mode'] and s.st_mtime_ns == f['mtime_ns'] and sha(p) == f['sha256']
    for f in inventory:
        Path(f['path']).unlink()
    for root, dirs, names in os.walk(W, topdown=False):
        assert not names
        Path(root).rmdir()
    report = {'verdict': 'executed_git_restored_native_recovery_passed', 'sealed_head': head,
              'manifest_sha256': sha(M), 'commands': results, 'native_microstep_conditions': 240,
              'invalid_metadata_native_microsteps': 0, 'removed_temporary_files': len(inventory),
              'removed_temporary_bytes': total, 'open_handles': 0, 'df_before': before, 'df_after': df(),
              'scope': plan['scope']}
    write(G / 'verification.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'commands'}), flush=True)
except Exception as error:
    write(G / 'failure.json', {'error': repr(error), 'completed': results, 'temporary_tree_retained': str(W)})
    raise
