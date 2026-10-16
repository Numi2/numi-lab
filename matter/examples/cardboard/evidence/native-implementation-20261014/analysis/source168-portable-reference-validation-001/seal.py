from pathlib import Path
import hashlib
import json
import shutil

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
S = B / 'snapshots/source-reference-jacobian-168'
BUILD = R / 'build-cardboard-provenance-168'
O = B / 'analysis/source168-portable-reference-validation-001'
O.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
frozen = json.loads((S / 'frozen-source.json').read_text())
build = json.loads((BUILD / 'source-build.json').read_text())
assert build['success'] and build['source_sha256'] == frozen['source_sha256']
for name, expected in frozen['source_sha256'].items():
    assert sha(S / name) == expected == sha(R / name), name
for path, expected in build['artifacts_sha256'].items():
    assert sha(Path(path)) == expected, path
text = (BUILD / 'Testing/Temporary/LastTest.log').read_text()
assert text.count('Test Passed.') == 6 and 'Test Failed.' not in text
for name in ['coupon_resume', 'inside_clearance', 'shipping_clearance_memory',
             'regional_consumer_capture', 'bilinear_capture_layers', 'reference_fb_jacobian']:
    assert 'cardboard_' + name in text, name
for name in ['frozen-source.json', 'configure-support-source.json']:
    shutil.copyfile(S / name, O / name)
for name in ['source-build.json', 'provenance-command-0.log', 'provenance-command-1.log', 'CTestTestfile.cmake']:
    shutil.copyfile(BUILD / name, O / name)
shutil.copyfile(BUILD / 'Testing/Temporary/LastTest.log', O / 'LastTest.log')
report = {
    'schema': 'numi.cardboard.portable-reference-validation.v1',
    'source_snapshot': str(S), 'selected_sources_verified': len(frozen['source_sha256']),
    'source_build_sha256': sha(BUILD / 'source-build.json'),
    'cpu_artifacts_verified': len(build['artifacts_sha256']), 'ctest_passed': 6,
    'fp64_derivative_relative_l2_gate': 1e-6,
    'projection_acceptance_limits_unchanged': [2e-12, 2e-11],
    'native_runtime_shader_probe_unchanged_from_source167': True,
    'source167_native_build_receipt_sha256': sha(R / 'build-cardboard-provenance-167-final/source-build.json'),
    'scope': '802 source hashes equal current working tree. CPU reference/registration build only. Source167 supplies the separately bound unchanged native probe; its source snapshot remains immutable.',
    'initial_test_failure': 'analysis/reference003-portable-test-attempt-001',
    'native_gpu_execution': False,
    'qualification': 'Portable numerical regression only. Full nonlinear solve, convergence, calibration, assembly and crush/impact remain unqualified.'
}
(O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
shutil.copyfile(__file__, O / 'seal.py')
(O / 'manifest.json').write_text(json.dumps({p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(O.iterdir()) if p.is_file()}, indent=2) + '\n')
print(json.dumps({'report_sha256': sha(O / 'report.json'), 'manifest_sha256': sha(O / 'manifest.json')}))
