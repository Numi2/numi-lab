from pathlib import Path
import hashlib
import importlib.util
import json

B = Path('/Users/home/cardboard-implementation-20261014')
A = B / 'solver/cache-reuse-full-history-parity-441'
C = B / 'solver/cgls-disabled-history-parity-445'
O = B / 'analysis/native445-cgls-disabled-history-parity-001'
assert not O.exists()
spec = importlib.util.spec_from_file_location('history441', B / 'analysis/native419-history-parity-001/audit_419.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
receipts = {str(p): json.loads((p / 'receipt.json').read_text()) for p in (A, C)}
assert all(r['execution_started'] and r['source_inputs_unchanged'] and r['exit_code'] == 2
           and not r['timed_out'] for r in receipts.values())
rows = h.compare_tree(A / 'output', C / 'output')
differences = [r for r in rows if not r['equal']]
assert len(rows) == 35
assert [r['path'] for r in differences] == ['manifest.json', 'rejected-conditioning-000002/manifest.json'], differences
assert {r['field'] for r in differences[0]['json_differences']} == {
    '$/materials/glue_path', '$/materials/liner_path', '$/materials/medium_path'}
assert [r['field'] for r in differences[1]['json_differences']] == ['$/device_program_fingerprint']
assert (A / 'stdout.log').read_bytes() == (C / 'stdout.log').read_bytes()
assert (A / 'first-status5.bin').read_bytes() == (C / 'first-status5.bin').read_bytes()
def events(root, prefix):
    return [line for line in (root / 'stderr.log').read_text().splitlines() if line.startswith(prefix)]
arm = [events(p, 'fem_residual_armijo_iteration=') for p in (A, C)]
true = [events(p, 'fem_true_residual_cycle=') for p in (A, C)]
assert len(arm[0]) == len(arm[1]) == 64 and arm[0] == arm[1]
assert len(true[0]) == len(true[1]) == 4096 and true[0] == true[1]
excluded = ('cardboard_step_timing=', 'fem_coupled_capture_', 'fem_status5_capture_',
            'fem_residual_armijo_iteration=', 'fem_residual_armijo_trace_enabled=')
clean = lambda p: [line for line in (p / 'stderr.log').read_text().splitlines() if not line.startswith(excluded)]
assert clean(A) == clean(C)
notice_a = json.loads(events(A, 'fem_residual_armijo_trace_enabled=')[0].split('=', 1)[1])
notice_c = json.loads(events(C, 'fem_residual_armijo_trace_enabled=')[0].split('=', 1)[1])
assert notice_a == notice_c
_, pa, ea = h.command_parts(B / 'native-441-command.json')
_, pc, ec = h.command_parts(B / 'native-445-command.json')
assert h.normalized_probe(pa) == h.normalized_probe(pc)
for key in ['NM_FEM_STATUS5_CAPTURE_PATH', 'NM_FEM_COUPLED_CAPTURE_PATH']:
    ea.pop(key)
    ec.pop(key)
assert ec['NM_FEM_CACHE_PROJECTION_REUSE'] == '1'
assert ea == ec
assert 'NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR' not in ec
build = Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-141/source-build.json')
assert json.loads(build.read_text())['success']
O.mkdir()
report = dict(
    schema='numi.cardboard.cgls_disabled_native_attempted_history_parity.v1',
    verdict='PASS_for_exact_attempted_history_only',
    output_files=35, byte_exact_output_files=33, metadata_differences=differences,
    armijo_records_byte_exact=64, true_residual_records_byte_exact=4096,
    stdout_byte_exact=True, first_status5_payload_byte_exact=True,
    all_other_numerical_status_fields_byte_exact=True,
    only_notice_difference='none',
    physical_probe_argv_equal_after_path_normalization=True,
    environment_difference='none; all CGLS selectors and capture paths absent',
    source_build141_sha256=sha(build), receipts=receipts,
    run_bindings={str(p): {name: sha(p / name) for name in ['receipt.json', 'stderr.log', 'output/result.json']} for p in (A, C)},
    qualification_limit='Both complete attempted histories reject loading control1 at Newton48 after one accepted control. No complete coupon, convergence or performance qualification. Source141 CGLS helpers remain default-off; enabled native qualification is separate. Timing is one observation per source; no speedup claim.')
(O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
(O / 'audit.py').write_bytes(Path(__file__).read_bytes())
(O / 'manifest.json').write_text(json.dumps(dict(files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in O.iterdir()}), indent=2) + '\n')
print('native445parity 33/35 physical files;64Armijo/4096trueR exact')
