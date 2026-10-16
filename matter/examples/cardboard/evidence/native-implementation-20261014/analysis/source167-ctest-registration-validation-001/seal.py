from pathlib import Path
import hashlib
import json
import shutil

B = Path('/Users/home/cardboard-implementation-20261014')
BUILD = Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-167-final')
O = B / 'analysis/source167-ctest-registration-validation-001'
O.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
log = BUILD / 'Testing/Temporary/LastTest.log'
text = log.read_text()
names = ['coupon_resume', 'inside_clearance', 'shipping_clearance_memory', 'regional_consumer_capture']
for name in names:
    assert f'matter.compiler.cardboard_{name}' in text
assert text.count('Test Passed.') == 4
assert 'Test Failed.' not in text
assert str(B / 'snapshots/source-diagnostic-and-resume-167-final/matter/src/runtime.mm') in text
shutil.copyfile(log, O / 'LastTest.log')
shutil.copyfile(BUILD / 'CTestTestfile.cmake', O / 'CTestTestfile.cmake')
report = {
    'schema': 'numi.cardboard.ctest-registration-validation.v1',
    'source_build_sha256': sha(BUILD / 'source-build.json'),
    'source167_validation_report_sha256': sha(B / 'analysis/source167-native-build-and-cpu-validation-002/report.json'),
    'registration_overlay_sha256': sha(B / 'analysis/source167-test-registration-002/source-overlay.json'),
    'tests': names,
    'passed': 4,
    'scope': 'CPU helper registration and explicit runtime source argument; no GPU execution or physical qualification.'
}
(O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
shutil.copyfile(__file__, O / 'seal.py')
manifest = {str(p.relative_to(O)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted(O.iterdir()) if p.is_file()}
(O / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'report_sha256': sha(O / 'report.json'), 'manifest_sha256': sha(O / 'manifest.json')}))
