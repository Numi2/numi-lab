from pathlib import Path
import csv, gzip, hashlib, json, os, stat, subprocess, sys, time

R = Path.cwd()
G = R / 'artifacts/crow-rl-git-recovery-native-drive-ir-20261108'
X = R / 'build/crow-native-drive-ir-git-recovery-01'
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
B = X / 'build/crow-native-drive-ir-build-01'
T1 = X / '.numi/retained-runtime/native-drive-ir-20261108'
T2 = X / '.numi/retained-runtime/native-drive-ir-capacity-20261108'
T3 = X / '.numi/retained-runtime/native-drive-ir-zero-gain-20261108'
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


execute('old-capacity-admission', T1, [B/'crow-fixture', X/'old-admission', '0', '1', '32', '2'], 1)
assert 'constraint program exceeds the world block capacity' in (G/'old-capacity-admission.stderr').read_text()

def directory_exact(actual, reference):
    for p in sorted(reference.iterdir()):
        if p.is_file() and p.name != 'runtime.json':
            exact(actual/p.name, p)

execute('capacity-crow-failure', T2, [B/'crow-fixture', X/'capacity-crow', '0', '1', '32', '2'], 1)
directory_exact(X/'capacity-crow', B/'crow-mode2-capacity')
execute('old-zero-gain-rejection', T3, [T3/'probe', T2/'metalrobo/MetalRobo.metallib', X/'zero-old.tsv', '--zero-gain'], 1)
assert 'Native zero-gain physical gate failed' in (G/'old-zero-gain-rejection.stderr').read_text()
exact(X/'zero-old.tsv', B/'zero-old.tsv')
execute('current-zero-gain', T3, [T3/'probe', T3/'metalrobo/MetalRobo.metallib', X/'zero-new.tsv', '--zero-gain'], 0)
exact(X/'zero-new.tsv', B/'zero-new.tsv')
execute('current-native-rows', T3, [T3/'probe', T3/'metalrobo/MetalRobo.metallib', X/'native.tsv', '--native-drive'], 0)
old=list(csv.DictReader((B/'native-zero.tsv').open(),delimiter='\t'))
new=list(csv.DictReader((X/'native.tsv').open(),delimiter='\t'))
assert len(old)==len(new)==576
for a,b in zip(old,new):
    assert {k:v for k,v in a.items() if k!='gpu_ms'}=={k:v for k,v in b.items() if k!='gpu_ms'},(a,b)
write(G/'native-physical-comparison.json',dict(physical_records_byte_exact=576,excluded_columns=['gpu_ms'],all_original_and_recovered_timings_retained=True))
execute('constraint-ir-owner', T3, [T3/'ir-probe'], 0)
execute('contact-owner', T3, [T3/'contact-probe', T3/'metalrobo/MetalRobo.metallib'], 0)
execute('default-crow', T3, [B/'fixture-default', X/'default-crow', '0', '1', '32'], 0)
directory_exact(X/'default-crow', B/'default-crow-zero')
execute('current-crow-failure', T3, [B/'crow-fixture', X/'current-crow', '0', '1', '32', '2'], 1)
directory_exact(X/'current-crow', B/'crow-mode2-zero')

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
              native_physical_rows_compared_without_gpu_timing=576,
              limits='Scoped native recovery of existing observations. Original capacity admission, zero-gain rejection and current crow solver failure are executed. All source files and old hybrid controls are Git-byte verified; the old hybrid/split-quality trials are not re-executed. No scientific qualification or promotion.')
write(G / 'verification.json', report)
print(json.dumps(report), flush=True)
