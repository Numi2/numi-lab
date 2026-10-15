from pathlib import Path
import hashlib, json, os, shutil, subprocess

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
S = B / 'snapshots/source-bounded-native-cgls-141'
N = B / 'snapshots/source-bounded-native-cgls-mask-repair-142'
K = B / 'analysis/native-base-cgls-mask-repair-142-001/core-v1/source/matter/src/metal/base_cgls.metalinc'
O = B / 'analysis/source142-native-cgls-mask-integration-001'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(K) == '439f702b6b92b913cff204ba3fd2a73757280cf68f9b7808c5cefcfa3de9b04e'
receipt = B / 'native/base-cgls-mask-repair-check-447/receipt.json'
proof = json.loads(receipt.read_text())
assert proof['exit_code'] == 0 and proof['source_inputs_unchanged'] and not proof['timed_out']
frozen = json.loads((S / 'frozen-source.json').read_text())
support = json.loads((S / 'configure-support-source.json').read_text())
for path, digest in {**frozen['source_sha256'], **support['files']}.items():
    assert sha(S / path) == digest, path
assert not N.exists() and not O.exists()
O.mkdir()
space = os.statvfs(B)
available = space.f_bavail * space.f_frsize
assert available > 700_000_000, available
(O / 'disk-preflight.json').write_text(json.dumps(dict(available_bytes=available, minimum_bytes=700_000_000), indent=2)+'\n')
clone = subprocess.run(['/bin/cp', '-cR', str(S), str(N)], capture_output=True, text=True)
(O / 'clone.log').write_text(clone.stdout + clone.stderr)
assert clone.returncode == 0
relative = 'matter/src/metal/base_cgls.metalinc'
before = (S / relative).read_text()
after = K.read_text()
old = 'if (mask.w != 0.0f || any((mask.xyz != float3(0.0f)) &'
new = 'if (mask.w != 0.0f || any((mask.xyz == float3(0.0f)) &'
assert before.count(old) == 1 and before.replace(old, new) == after
shutil.copy2(K, N / relative)
frozen['source_sha256'][relative] = sha(K)
frozen['base_snapshot'] = str(S)
frozen['scope'] = ('Source141 plus the measured first-layer CGLS initial-mask predicate repair. '
    'Only measure_initial mask!=0 becomes mask==0 in the constrained-input test. '
    'Native446 confirms the original invalid-input transition; native447 passes the unchanged synthetic checker. '
    'No recurrence, tolerance, material, physical operator, accepted history, final certificate or rollback change. '
    'All enabled production FEM CGLS qualification remains pending.')
(N / 'frozen-source.json').write_text(json.dumps(frozen, indent=2)+'\n')
for path, digest in {**frozen['source_sha256'], **support['files']}.items():
    assert sha(N / path) == digest, path
(O / 'integration.json').write_text(json.dumps(dict(source=str(N), base=str(S),
    changed_files={relative:sha(K)}, native447_receipt_sha256=sha(receipt),
    selected_sources=len(frozen['source_sha256']), physical_steps=0,
    status='AWAITING_BUILD_AND_PRODUCTION_NATIVE_CHECKS'), indent=2)+'\n')
(O / 'assemble.py').write_bytes(Path(__file__).read_bytes())
command = ['/Users/home/.pyenv/versions/3.10.12/bin/python3',
    str(N / 'matter/tools/cardboard_source_bound_build.py'), '--source-snapshot', str(N),
    '--build-dir', str(R / 'build-cardboard-provenance-142'), '--cmake-source-relative', '.',
    '--jobs', '4', '--target', 'numi-matter-cardboard-probe']
(B / 'build-command-142.json').write_text(json.dumps(command, indent=2)+'\n')
with (B / 'build142-runner.log').open('x') as output:
    code = subprocess.call(command, stdout=output, stderr=subprocess.STDOUT)
(O / 'post-build.json').write_text(json.dumps(dict(build_exit_code=code,
    source_build_sha256=sha(R / 'build-cardboard-provenance-142/source-build.json')), indent=2)+'\n')
print('build142', code, flush=True)
raise SystemExit(code)
