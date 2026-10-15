#!/usr/bin/env python3
"""Strict read-only comparison of source-125 status5 capture run 415 to run 409."""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

BASE = Path('/Users/home/cardboard-implementation-20261014')
A = BASE / 'solver/two-stage-certified-coupon-409'
B = BASE / 'solver/two-stage-coupon-first-descent-capture-415'
S123 = BASE / 'snapshots/source-two-stage-material-merit-123'
S125 = BASE / 'snapshots/source-status5-vtccd-diagnostics-125'
PKG = Path(__file__).resolve().parent
DECODER = PKG / 'decode_status5'

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_json(path: Path):
    return json.loads(path.read_text())

def compare_tree(left: Path, right: Path):
    l = {p.relative_to(left).as_posix(): p for p in left.rglob('*') if p.is_file()}
    r = {p.relative_to(right).as_posix(): p for p in right.rglob('*') if p.is_file()}
    rows = []
    for name in sorted(l.keys() | r.keys()):
        if name not in l or name not in r:
            rows.append({'path': name, 'equal': False, 'reason': 'path_missing'})
            continue
        x, y = l[name].read_bytes(), r[name].read_bytes()
        row = {'path': name, 'left_bytes': len(x), 'right_bytes': len(y),
               'left_sha256': hashlib.sha256(x).hexdigest(),
               'right_sha256': hashlib.sha256(y).hexdigest(), 'equal': x == y}
        if x != y and name.endswith('.json'):
            jx, jy = json.loads(x), json.loads(y)
            differences = []
            def walk(a, b, path='$'):
                if type(a) is not type(b):
                    differences.append({'field': path, 'left': a, 'right': b}); return
                if isinstance(a, dict):
                    for key in sorted(a.keys() | b.keys()):
                        if key not in a or key not in b:
                            differences.append({'field': path + '/' + key, 'missing_side': 'left' if key not in a else 'right'})
                        elif a[key] != b[key]: walk(a[key], b[key], path + '/' + key)
                elif isinstance(a, list):
                    if len(a) != len(b): differences.append({'field': path, 'left_count': len(a), 'right_count': len(b)})
                    for i, (u, v) in enumerate(zip(a, b)):
                        if u != v: walk(u, v, path + f'/{i}')
                else:
                    differences.append({'field': path, 'left': a, 'right': b})
            walk(jx, jy)
            row['json_differences'] = differences
        rows.append(row)
    return rows

def log_events(path: Path, prefix: str):
    return [line for line in path.read_text().splitlines() if line.startswith(prefix)]

def main():
    ra, rb = read_json(A/'receipt.json'), read_json(B/'receipt.json')
    assert ra['execution_started'] and rb['execution_started']
    assert ra['exit_code'] == rb['exit_code'] == 2
    assert ra['timed_out'] is rb['timed_out'] is False
    assert ra['source_inputs_unchanged'] and rb['source_inputs_unchanged']
    files = compare_tree(A/'output', B/'output')
    assert len(files) == 35 and all(x['path'] for x in files)
    diffs = [x for x in files if not x['equal']]
    assert [x['path'] for x in diffs] == ['manifest.json', 'rejected-conditioning-000002/manifest.json']
    manifest_diffs = diffs[0]['json_differences']
    assert {x['field'] for x in manifest_diffs} == {
        '$/materials/glue_path', '$/materials/liner_path', '$/materials/medium_path'}
    reject_diffs = diffs[1]['json_differences']
    assert len(reject_diffs) == 1 and reject_diffs[0]['field'] == '$/device_program_fingerprint'
    # All material content and compiled-world/output files are checked by the tree hashes above.
    arm_a = log_events(A/'stderr.log', 'fem_residual_armijo_iteration=')
    arm_b = log_events(B/'stderr.log', 'fem_residual_armijo_iteration=')
    true_a = log_events(A/'stderr.log', 'fem_true_residual_cycle=')
    true_b = log_events(B/'stderr.log', 'fem_true_residual_cycle=')
    assert len(arm_a) == len(arm_b) == 64 and arm_a == arm_b
    assert len(true_a) == len(true_b) == 4096 and true_a == true_b
    ignorable = ('cardboard_step_timing=', 'fem_coupled_capture_enabled=',
                 'fem_coupled_capture_skipped=', 'fem_status5_capture_complete=')
    clean_a = [x for x in (A/'stderr.log').read_text().splitlines() if not x.startswith(ignorable)]
    clean_b = [x for x in (B/'stderr.log').read_text().splitlines() if not x.startswith(ignorable)]
    assert len(clean_a) == len(clean_b) == 4164 and clean_a == clean_b
    assert (A/'stdout.log').read_bytes() == (B/'stdout.log').read_bytes()
    result_a, result_b = read_json(A/'output/result.json'), read_json(B/'output/result.json')
    assert result_a == result_b and result_a['status'] == 'failed'
    assert result_a['accepted_steps'] == 1 and result_a['physical_validation'] is False
    assert sha(A/'output/compiled.nmatterpack') == sha(B/'output/compiled.nmatterpack')

    status_manifest = read_json(B/'first-status5.json')
    status_payload = B/'first-status5.bin'
    assert status_manifest['records_present'] == 1
    assert status_manifest['payload']['sha256'] == sha(status_payload)
    decoded = json.loads(subprocess.check_output([str(DECODER), str(status_payload)], text=True))
    rows = [x for x in decoded['records'] if x['capture'][0] == 1]
    assert len(rows) == 1
    record = rows[0]
    assert record['capture'] == [1, 31, 1, 2]
    assert record['context'][0] == 1 and record['context'][3] == 26
    assert record['identity'] == [0, 1464, 1464, 0]
    assert record['status'][0] == 5 and record['status'][3] == 1464
    stress_max = record['status'][9]
    limit = record['material_validity'][2]
    assert stress_max > limit > 0

    accepted26 = None
    for line in arm_b:
        value = json.loads(line.split('=', 1)[1])
        if value.get('newton_iteration') == 26:
            accepted26 = value['environments'][0]
            break
    assert accepted26 is not None and accepted26['accepted'] is True
    assert accepted26['selected_alpha'] == 0.0078125

    source_hashes = {}
    for snap in (S123, S125):
        manifest = snap/'frozen-source.json'
        source_hashes[snap.name] = {
            'frozen_source_manifest_sha256': sha(manifest),
            'file_count': len(read_json(manifest)['source_sha256'])}
    out = {
      'schema': 'numi.cardboard.native415_status5_parity.v1',
      'verdict': 'physical_trace_and_outputs_match_with_expected_metadata_differences; captured_private_trial_is_inadmissible',
      'claim_limit': 'Read-only native evidence; both runs fail and neither qualifies a complete physical cycle.',
      'runs': {
        '409': {'path': str(A), 'receipt_sha256': sha(A/'receipt.json'), 'stderr_sha256': sha(A/'stderr.log'), 'exit_code': ra['exit_code'], 'timed_out': ra['timed_out']},
        '415': {'path': str(B), 'receipt_sha256': sha(B/'receipt.json'), 'stderr_sha256': sha(B/'stderr.log'), 'exit_code': rb['exit_code'], 'timed_out': rb['timed_out']}
      },
      'source_bindings': source_hashes,
      'shared_output': {'path_count': len(files), 'byte_identical_count': sum(x['equal'] for x in files), 'differences': diffs,
        'compiled_nmatterpack_sha256': sha(A/'output/compiled.nmatterpack'), 'result_json_byte_identical': True},
      'log_comparison': {'armijo_iteration_records': len(arm_a), 'armijo_iteration_records_byte_identical': True,
        'fresh_true_residual_records': len(true_a), 'fresh_true_residual_records_byte_identical': True,
        'stderr_lines_after_removing_timing_and_declared_capture_metadata': len(clean_a), 'remaining_stderr_lines_byte_identical': True,
        'stdout_byte_identical': True},
      'status5': {'manifest_sha256': sha(B/'first-status5.json'), 'payload_sha256': sha(status_payload),
        'payload_bytes': len(status_payload.read_bytes()), 'records_present': decoded['records_present'],
        'capture': record['capture'], 'context': record['context'], 'identity': record['identity'],
        'status': record['status'], 'detF': record['status'][8], 'reported_max_abs_transformed_stress': stress_max,
        'material_validity': record['material_validity'], 'stress_limit': limit,
        'stress_limit_ratio': stress_max/limit, 'trial_index': record['identity'][3],
        'alpha_lane_value_in_capture': record['scalars'][1], 'alpha_lane_flag': bool(record['payload'][0] & 64),
        'alpha_value_is_known_incorrect': True, 'actual_selected_newton26_alpha_from_armijo_trace': accepted26['selected_alpha'],
        'local_status_semantics': 'Private trial index 0, stage 31; status 5 at tetrahedron 1464. Newton 26 later accepts alpha 1/128. The capture does not store the per-trial alpha correctly.',
        'qualification': 'Stress exceeds the material maximum by 17.918528%. This is a rejected private candidate, not accepted state or the terminal Newton-48 failure. Because the capture does not include the projection-valid bit, it does not prove projection succeeded; the stress threshold condition is independently violated.'},
      'limitations': ['Coupled candidate capture was skipped because environment_count=2 is outside its scope; no complete coupled capture was produced.',
        'status5 capture records only the first private status-5 witness for this root and does not show every trial.',
        'Both coupon runs exit 2 after one accepted step and remain failed/incomplete; parity does not imply convergence or physical qualification.']
    }
    (PKG/'audit.json').write_text(json.dumps(out, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'verdict':out['verdict'],'output_byte_identical_count':out['shared_output']['byte_identical_count'],
      'output_count':len(files),'armijo_exact':len(arm_a),'true_residual_exact':len(true_a),
      'stress':stress_max,'limit':limit,'accepted_alpha_newton26':accepted26['selected_alpha']},indent=2))

if __name__ == '__main__': main()
