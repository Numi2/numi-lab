from pathlib import Path
import gzip, hashlib, json, os, stat, subprocess, sys, time

R = Path.cwd()
G = R / 'artifacts/crow-rl-git-recovery-motor-contact-20261108'
X = R / 'build/crow-motor-contact-git-recovery-01'
sys.path.insert(0, str(R / 'tools'))
from crow_evidence_archive import tree, Objects, content


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


plan = json.loads((G / 'plan.json').read_text())
entries = tree()
with Objects() as objects:
    raw = objects.read(entries[plan['manifest']][1])
    assert hashlib.sha256(raw).hexdigest() == plan['manifest_sha256']
    m = json.loads(raw)
    assert m['summary']['source_bytes'] <= plan['restore_max_bytes']
    X.mkdir(exist_ok=False)
    for f in m['files']:
        p = X / f['path']
        p.parent.mkdir(parents=True, exist_ok=True)
        data = b''.join(content(f, entries, objects))
        assert len(data) == f['bytes'] and hashlib.sha256(data).hexdigest() == f['sha256']
        p.write_bytes(data)
        p.chmod(f['mode'])
        os.utime(p, ns=(f['mtime_ns'], f['mtime_ns']))
print(json.dumps({'restored_files': len(m['files']), 'restored_bytes': m['summary']['source_bytes']}), flush=True)
B = X / 'build/crow-motor-contact-build-01'
T1 = X / '.numi/retained-runtime/motor-contact-20261108'
T2 = X / '.numi/retained-runtime/motor-contact-factor-fix-20261108'
commands, checks = [], []


def execute(name, runtime, args, expected):
    env = os.environ.copy()
    env.update(DYLD_LIBRARY_PATH=str(runtime), DYLD_PRINT_LIBRARIES='1',
               CROW_PROBE_METALLIB=str(runtime / 'metalrobo/MetalRobo.metallib'))
    for key in ('METALROBO_METALLIB', 'METALROBO_CCD_FAILURE_CAPTURE', 'METALROBO_BOUNDED_FAILURE_CAPTURE'):
        env.pop(key, None)
    start = time.monotonic()
    with (G / (name + '.stdout')).open('xb') as out, (G / (name + '.stderr')).open('xb') as err:
        try:
            code = subprocess.run([str(a) for a in args], env=env, stdout=out, stderr=err,
                                  timeout=plan['timeout_seconds_each']).returncode
        except subprocess.TimeoutExpired:
            code = None
    loads = [s for s in (G / (name + '.stderr')).read_text().splitlines()
             if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
    rec = dict(name=name, command=[str(a) for a in args], expected_exit_code=expected,
               exit_code=code, seconds=time.monotonic() - start, loader_lines=loads)
    commands.append(rec)
    write(G / 'execution.json', commands)
    print(json.dumps(rec), flush=True)
    assert code == expected and loads and all(s.endswith(str(runtime / 'libmetalrobo.dylib')) for s in loads), rec


def exact(actual, reference):
    rec = dict(actual=str(actual.relative_to(X)), reference=str(reference.relative_to(X)),
               byte_exact=actual.read_bytes() == reference.read_bytes(), sha256=sha(actual))
    checks.append(rec)
    write(G / 'comparisons.json', checks)
    assert rec['byte_exact'], rec


execute('original-split', T1, [T1 / 'probe', T1 / 'metalrobo/MetalRobo.metallib', X / 'original.tsv'], 0)
exact(X / 'original.tsv', B / 'native.tsv')
execute('original-empty-factor-failure', T1,
        [T1 / 'coupled-probe', T1 / 'metalrobo/MetalRobo.metallib', X / 'empty-factor.tsv', '--coupled'], 1)
assert 'contact_quality_diagnostics=10.000000' in (G / 'original-empty-factor-failure.stderr').read_text()
exact(X / 'empty-factor.tsv', B / 'coupled.tsv')
execute('fixed-factor-momentum-failure', T2,
        [T2 / 'probe', T2 / 'metalrobo/MetalRobo.metallib', X / 'factor-fixed.tsv', '--coupled-unused'], 1)
assert 'Coupled force/momentum validity gate failed' in (G / 'fixed-factor-momentum-failure.stderr').read_text()
exact(X / 'factor-fixed.tsv', B / 'coupled-unused.tsv')
execute('tight-momentum-replay', T2,
        [T2 / 'probe-precision', T2 / 'metalrobo/MetalRobo.metallib', X / 'momentum.tsv', '--momentum-replay'], 0)
exact(X / 'momentum.tsv', B / 'momentum-replay.tsv')
execute('tight-joint-comparison', T2,
        [T2 / 'probe-precision', T2 / 'metalrobo/MetalRobo.metallib', X / 'joint.tsv', '--coupled-precision'], 0)
exact(X / 'joint.tsv', B / 'coupled-precision.tsv')
execute('contact-owner', T2, [T2 / 'contact-probe', T2 / 'metalrobo/MetalRobo.metallib'], 0)
execute('default-crow', T2, [B / 'fixture-default', X / 'default-crow', '0', '1', '32'], 0)
physical_names = []
for reference in sorted((B / 'default-crow').iterdir()):
    if reference.is_file() and reference.name != 'runtime.json':
        exact(X / 'default-crow' / reference.name, reference)
        physical_names.append(reference.name)
assert len(physical_names) == 10

known = {(f['sha256'], f['bytes']): f['path'] for f in m['files']}
inventory = []
(G / 'recovery-blobs').mkdir(exist_ok=False)
for p in sorted(X.rglob('*')):
    if not p.is_file():
        continue
    s = p.lstat()
    assert stat.S_ISREG(s.st_mode) and not p.is_symlink()
    h = sha(p)
    item = dict(path=str(p.relative_to(R)), bytes=s.st_size, sha256=h,
                mode=stat.S_IMODE(s.st_mode), mtime_ns=s.st_mtime_ns)
    if (h, s.st_size) in known:
        item['archive_source'] = known[(h, s.st_size)]
    else:
        dest = G / 'recovery-blobs' / (h + '.gz')
        if not dest.exists():
            dest.write_bytes(gzip.compress(p.read_bytes(), compresslevel=6, mtime=0))
        item['recovery_blob'] = str(dest.relative_to(R))
    inventory.append(item)
assert sum(f['bytes'] for f in inventory) <= plan['temporary_budget_bytes']
write(G / 'temporary-inventory.json', inventory)
handles = subprocess.run(['/usr/sbin/lsof', '-Fpn', '+D', str(X)], capture_output=True, text=True, timeout=60)
write(G / 'temporary-open-handles.json', dict(exit_code=handles.returncode, stdout=handles.stdout, stderr=handles.stderr))
assert handles.returncode in (0, 1) and not handles.stdout and not handles.stderr
report = dict(manifest=plan['manifest'], manifest_sha256=plan['manifest_sha256'],
              restored_files=len(m['files']), restored_bytes=m['summary']['source_bytes'],
              native_commands=len(commands), byte_exact_comparisons=len(checks),
              all_compared_bytes_exact=all(c['byte_exact'] for c in checks),
              temporary_files=len(inventory), temporary_bytes=sum(f['bytes'] for f in inventory),
              new_recovery_blobs=len(list((G / 'recovery-blobs').glob('*.gz'))),
              open_handles=0, policy_or_profile_promoted=False,
              limits='Scoped native recovery of existing observations; no new scientific qualification. Old uninitialized-factor failure is checked by its typed diagnostic, with its defined one-row prefix compared exactly.')
write(G / 'verification.json', report)
print(json.dumps(report), flush=True)
