from pathlib import Path
import hashlib
import json
import shutil
import subprocess

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
S = B / 'snapshots/source-coupon-cache-admission-139'
N = B / 'snapshots/source-bounded-native-cgls-141'
K = B / 'analysis/native-base-cgls-candidate-139-overlay-001/core-v1/source/matter/src/metal/base_cgls.metalinc'
A = B / 'analysis/native-cgls-runtime-overlay-140-001'
O = B / 'analysis/source141-native-cgls-integration-001'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(K) == '24563b7d4d54c894c1f6ace75dca060429159752500c394e78a69a4fc0ae3b2c'
binding = json.loads((A / 'final-runtime-binding.json').read_text())
assert sha(A / 'candidate/matter/src/runtime.mm') == binding['candidate_runtime_sha256']
assert sha(A / 'runtime-adapters.metalinc') == binding['runtime_adapters_sha256']
frozen = json.loads((S / 'frozen-source.json').read_text())
support = json.loads((S / 'configure-support-source.json').read_text())
for path, digest in {**frozen['source_sha256'], **support['files']}.items():
    assert sha(S / path) == digest, path
assert not N.exists() and not O.exists()
O.mkdir()
clone = subprocess.run(['/bin/cp', '-cR', str(S), str(N)], capture_output=True, text=True)
(O / 'clone.log').write_text(clone.stdout + clone.stderr)
assert clone.returncode == 0
shutil.copy2(A / 'candidate/matter/src/runtime.mm', N / 'matter/src/runtime.mm')
shutil.copy2(K, N / 'matter/src/metal/base_cgls.metalinc')
(N / 'matter/src/metal/base_cgls_runtime.metalinc').write_text(
    'namespace numi_matter_metal {\n\n' + (A / 'runtime-adapters.metalinc').read_text() +
    '\n} // namespace numi_matter_metal\n')
p = N / 'matter/src/metal/fgmres.metalinc'
p.write_text(p.read_text() + '\n#include "base_cgls.metalinc"\n#include "base_cgls_runtime.metalinc"\n')
p = N / 'matter/CMakeLists.txt'
text = p.read_text()
old = '            src/metal/fgmres.metalinc\n'
assert text.count(old) == 1
p.write_text(text.replace(old, old + '            src/metal/base_cgls.metalinc\n            src/metal/base_cgls_runtime.metalinc\n'))
changed_paths = ['matter/src/runtime.mm', 'matter/src/metal/fgmres.metalinc',
                 'matter/src/metal/base_cgls.metalinc', 'matter/src/metal/base_cgls_runtime.metalinc',
                 'matter/CMakeLists.txt']
changed = {path: sha(N / path) for path in changed_paths}
frozen['source_sha256'].update(changed)
frozen['base_snapshot'] = str(S)
frozen['scope'] = ('Source139 plus default-off bounded native CGLS32/64 direction generation. '
    'Actual cached unsymmetric A and transpose A^T, separate selected-only B/x/r/s/p/q and mask/map scratch, '
    'strict finite/denominator/input checks, no physical operator/material/history/acceptance tolerance change. '
    'Only after primary Armijo exhaustion on selected Newton states; installer requires original Newton certificate '
    'but clears its latch before testing the new direction through existing nonlinear Armijo/final certificate/rollback. '
    'Bounded optional read-only scratch capture at one canonical Newton index. Awaiting all native qualification.')
(N / 'frozen-source.json').write_text(json.dumps(frozen, indent=2) + '\n')
for path, digest in {**frozen['source_sha256'], **support['files']}.items():
    assert sha(N / path) == digest, path
(O / 'integration.json').write_text(json.dumps(dict(
    source=str(N), base=str(S), core_kernel_sha256=sha(K),
    runtime_binding_sha256=sha(A / 'final-runtime-binding.json'), changed_files=changed,
    selected_sources=len(frozen['source_sha256']), physical_steps=0,
    status='AWAITING_BUILD_AND_NATIVE_CHECKS'
), indent=2) + '\n')
(O / 'assemble.py').write_bytes(Path(__file__).read_bytes())
command = ['/Users/home/.pyenv/versions/3.10.12/bin/python3',
    str(N / 'matter/tools/cardboard_source_bound_build.py'), '--source-snapshot', str(N),
    '--build-dir', str(R / 'build-cardboard-provenance-141'), '--cmake-source-relative', '.',
    '--jobs', '4', '--target', 'numi-matter-cardboard-probe']
(B / 'build-command-141.json').write_text(json.dumps(command, indent=2) + '\n')
with (B / 'build141-runner.log').open('x') as output:
    code = subprocess.call(command, stdout=output, stderr=subprocess.STDOUT)
(O / 'post-build.json').write_text(json.dumps(dict(
    build_exit_code=code,
    source_build_sha256=sha(R / 'build-cardboard-provenance-141/source-build.json')
), indent=2) + '\n')
print('build141', code, flush=True)
raise SystemExit(code)
