from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
P = B / 'analysis/bilinear-capture-layer-audit-source166-003'
O = B / 'analysis/reference003-portable-integration-001'
O.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P / 'manifest.json') == '23d77dc36a39e15c4275ba4784816d33e56d1929714c9eabc28d3029ab32c611'
source = P / 'candidate/matter/tools/cardboard_bilinear_fp64_reference.py'
assert sha(source) == 'abb2f33b00fd2fbac0097c35814833dbe957313a8a0cc65da3a70c030e8f1a0e'
destination = R / 'matter/tools/cardboard_bilinear_fp64_reference.py'
assert sha(destination) == 'ea5cba69f600f446cc1c129b03bf1fa578d4f26d2d0d62fa1e63b4fa4c8248d5'
mp_script = B / 'analysis/native469-independent-projection-002/solve_mp_projection.py'
spec = importlib.util.spec_from_file_location('portable_mp_witness', mp_script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.mp.mp.dps = 90
_, _, inputs, _, _, _ = module.load_capture_bindings()
inp = inputs[846]
p, F, old, E, C, fun = module.model(inp['p'], inp['F'], inp['old'])
report = json.loads((B / 'analysis/native469-independent-projection-002/run-sealed002/report.json').read_text())
root = next(row for row in report['resolved_rows'] if row['tet'] == 846)['precision_results']['110']['root_state_mp_decimal']
states = {
    'elastic_trial': [float(v) for v in module.elastic_trial_seed(p, F, old, E, C)],
    'resolved_root': [float(module.mp.mpf(v)) for v in root]
}
rows = []
for name, values in states.items():
    exact = [module.exact_mpf(v) for v in values]
    derivative = []
    for column in range(15):
        def one(value):
            state = list(exact)
            state[column] = value
            return fun(state)[12]
        derivative.append(module.mp.nstr(module.mp.diff(one, exact[column]), 85))
    rows.append({'name': name, 'state': values, 'fb_jacobian_90_digit': derivative})
fixture = {
    'schema': 'numi.cardboard.independent-fb-jacobian-fixture.v1',
    'native_capture_sha256': module.EXPECTED['capture'],
    'high_precision_witness_manifest_sha256': sha(B / 'analysis/native469-independent-projection-002/manifest.json'),
    'independent_equations_script_sha256': sha(mp_script),
    'parameters': inp['p'].tolist(), 'F': inp['F'].tolist(), 'accepted_history': inp['old'].tolist(),
    'precision_digits': 90, 'tet_id': 846, 'rows': rows,
    'scope': 'Numerical reference regression only. Expected derivatives use independently authored unsimplified MP equations at exact promoted binary64 fixture inputs.'
}
fixture_path = R / 'matter/tools/test-data/cardboard/reference-fb-tet846.json'
assert not fixture_path.exists()
fixture_path.parent.mkdir(parents=True, exist_ok=True)
fixture_path.write_text(json.dumps(fixture, indent=2, allow_nan=False) + '\n')
new_source = source.read_text().replace(
    '# residuals. Complex-step derivatives use the same analytic expression.',
    '# residuals. jacobian() differentiates the authored FB row directly.')
destination.write_text(new_source)
result = {'schema': 'numi.cardboard.reference003-portable-integration.v1',
          'reference_package_manifest_sha256': sha(P / 'manifest.json'),
          'reference_sha256': sha(destination), 'fixture_sha256': sha(fixture_path),
          'comment_only_delta_from_sealed_reference': True,
          'external_absolute_path_test_not_installed': True,
          'native_source_unchanged': True, 'gpu_execution': False}
(O / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
shutil.copyfile(__file__, O / 'integrate.py')
shutil.copyfile(fixture_path, O / 'reference-fb-tet846.json')
print(json.dumps(result))
