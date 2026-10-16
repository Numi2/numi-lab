from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys

sys.dont_write_bytecode = True
B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
S = B / 'snapshots/source-diagnostic-and-resume-167-final'
T = B / 'snapshots/source-reference-jacobian-168'
O = B / 'analysis/source168-reference-integration-001'
assert not T.exists() and not O.exists()
O.mkdir()
T.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
frozen = json.loads((S / 'frozen-source.json').read_text())
support = json.loads((S / 'configure-support-source.json').read_text())
files = dict(frozen['source_sha256'])
files.update(support['files'])
assert len(files) == 1834
for name, digest in files.items():
    assert sha(S / name) == digest, name
    destination = T / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(S / name, destination)
changes = ['matter/CMakeLists.txt', 'matter/tools/cardboard_bilinear_fp64_reference.py',
           'matter/tools/test_cardboard_reference_fb_jacobian.py',
           'matter/tools/cardboard_reference_fb_tet846_fixture.json']
for name in changes:
    shutil.copy2(R / name, T / name)
for name, digest in frozen['source_sha256'].items():
    if name not in changes:
        assert sha(R / name) == digest, name
spec = importlib.util.spec_from_file_location('builder168', T / 'matter/tools/cardboard_source_bound_build.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
selected = module.source_bindings(T)
assert len(selected) == 802
support_files = {name: sha(T / name) for name in files if name not in selected}
new_frozen = dict(frozen, base_snapshot=str(S), source_sha256=selected,
                  scope='Source167 native code unchanged; independent FP64 FB Jacobian repair and portable numerical witness. CPU test registration only.')
(T / 'frozen-source.json').write_text(json.dumps(new_frozen, indent=2) + '\n')
(T / 'configure-support-source.json').write_text(json.dumps(dict(support, files=support_files), indent=2) + '\n')
report = {'schema': 'numi.cardboard.source168-reference-integration.v1',
          'base_source_count': len(frozen['source_sha256']), 'source_count': len(selected),
          'base_frozen_sha256': sha(S / 'frozen-source.json'),
          'frozen_source_sha256': sha(T / 'frozen-source.json'),
          'source': str(T), 'changed_or_added': {name: sha(T / name) for name in changes},
          'native_runtime_shader_probe_unchanged': True,
          'source166003_manifest_sha256': sha(B / 'analysis/bilinear-capture-layer-audit-source166-003/manifest.json'),
          'fixture_precision_digits': 90, 'fp64_derivative_relative_l2_gate': 1e-6,
          'initial_arbitrary_1e12_test_failure_retained': 'analysis/reference003-portable-test-attempt-001',
          'projection_gates_unchanged': True, 'gpu_execution': False}
(O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
shutil.copyfile(__file__, O / 'prepare.py')
print(json.dumps(report))
