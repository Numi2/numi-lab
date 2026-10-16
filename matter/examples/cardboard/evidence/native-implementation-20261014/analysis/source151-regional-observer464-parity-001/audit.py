#!/usr/bin/env python3
"""Strict CPU-only full-attempt parity audit for native runs 462 and 464.

This compares retained artifacts only. It does not launch a solver or qualify
the failed coupon. The only ignored differences are explicitly enumerated
observer configuration/markers, run-specific paths, and timing records.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import pathlib
import re
import sys
from typing import Any

B = pathlib.Path('/Users/home/cardboard-implementation-20261014')
O = pathlib.Path(__file__).resolve().parent
L = B / 'solver/cgls-block-right-disabled-parity-462'
R = B / 'solver/cgls-regional-state-capture-464'
CMD_L = B / 'native-462-command.json'
CMD_R = B / 'analysis/native464-regional-capture-command-001/native-464-command.json'
READER = B / 'analysis/native464-regional-capture-command-001/validator/read_regional_capture_v2.py'
READER_DIR = READER.parent

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def normalize_text(value: str) -> str:
    # Explicit run/source/build substitutions only. No numeric or scientific
    # values are normalized.
    for old, new in (
        (str(L), '<case>'), (str(R), '<case>'),
        ('/Users/home/cardboard-implementation-20261014/snapshots/source-cgls-block-right-preconditioned-149', '<snapshot>'),
        ('/Users/home/cardboard-implementation-20261014/snapshots/source-cgls-regional-state-capture-151', '<snapshot>'),
        ('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-149', '<build>'),
        ('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-151', '<build>'),
    ):
        value = value.replace(old, new)
    return value

def parse_stderr(path: pathlib.Path) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if '=' not in line:
            raise AssertionError(f'unparsed stderr line {path}:{index}')
        key, raw = line.split('=', 1)
        rows.append((key, json.loads(raw)))
    return rows

def normalized_event(row: tuple[str, Any]) -> tuple[str, Any]:
    key, value = row
    # Normalize only run-local path values in observer completion markers.
    if isinstance(value, dict):
        value = json.loads(json.dumps(value))
        for field in ('manifest', 'path'):
            if isinstance(value.get(field), str):
                value[field] = normalize_text(value[field])
        if isinstance(value.get('payload'), dict) and isinstance(value['payload'].get('path'), str):
            value['payload']['path'] = normalize_text(value['payload']['path'])
    return key, value

def flatten_diff(a: Any, b: Any, path: str = '') -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            p = f'{path}/{key}'
            if key not in a or key not in b:
                out.append({'path': p, 'left': 'missing' if key not in a else a[key], 'right': 'missing' if key not in b else b[key]})
            else:
                out.extend(flatten_diff(a[key], b[key], p))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append({'path': path, 'left_length': len(a), 'right_length': len(b)})
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(flatten_diff(x, y, f'{path}/{i}'))
    elif a != b:
        out.append({'path': path, 'left': a, 'right': b})
    return out

def command_env(argv: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for i, item in enumerate(argv[:-1]):
        if item == '--env':
            key, value = argv[i + 1].split('=', 1)
            if key in env:
                raise AssertionError(f'duplicate env key {key}')
            env[key] = value
    return env

def command_tail(argv: list[str]) -> list[str]:
    if '--' not in argv:
        raise AssertionError('native command is missing executable separator')
    return argv[argv.index('--') + 1:]

def main() -> None:
    # Verify the immutable native run receipts and failed-result classification.
    receipt_l, receipt_r = read_json(L / 'receipt.json'), read_json(R / 'receipt.json')
    assert receipt_l['execution_started'] and receipt_r['execution_started']
    assert receipt_l['exit_code'] == receipt_r['exit_code'] == 2
    assert receipt_l['timed_out'] is False and receipt_r['timed_out'] is False
    assert receipt_l['termination_errors'] == receipt_r['termination_errors'] == []
    assert receipt_l['source_inputs_unchanged'] is True and receipt_r['source_inputs_unchanged'] is True
    result_l, result_r = read_json(L / 'output/result.json'), read_json(R / 'output/result.json')
    assert result_l == result_r
    assert result_l['status'] == 'failed' and result_l['accepted_steps'] == 1
    assert result_l['physical_validation'] is False

    # The actual probe argv after '--' must match exactly after only source,
    # build, case and output path normalization.
    cmd_l, cmd_r = read_json(CMD_L), read_json(CMD_R)
    tail_l = [normalize_text(x) for x in command_tail(cmd_l)]
    tail_r = [normalize_text(x) for x in command_tail(cmd_r)]
    assert tail_l == tail_r, 'physical/solver/native argv differs beyond explicit path mapping'
    env_l, env_r = command_env(cmd_l), command_env(cmd_r)
    for mapping in (env_l, env_r):
        for key in ('NM_FEM_COUPLED_CAPTURE_PATH', 'NM_FEM_STATUS5_CAPTURE_PATH', 'NM_FEM_BASE_CGLS_CAPTURE_PATH'):
            if key in mapping:
                mapping[key] = normalize_text(mapping[key])
    removed = {'NM_FEM_COUPLED_CAPTURE_SELECTOR', 'NM_FEM_COUPLED_CAPTURE_PATH'}
    added = {'NM_FEM_BASE_CGLS_STATE_CAPTURE_REGIONAL': '1'}
    env_l_prod = {k: v for k, v in env_l.items() if k not in removed}
    env_r_prod = dict(env_r)
    assert all(env_r_prod.pop(k, None) == v for k, v in added.items())
    assert env_l_prod == env_r_prod, 'unexpected non-observer environment delta'
    assert env_l['NM_FEM_COUPLED_CAPTURE_SELECTOR'] == '1:48'
    assert 'NM_FEM_COUPLED_CAPTURE_SELECTOR' not in env_r

    # Complete output-tree inventory: exact relative names, exact bytes except
    # the three absolute material source paths in output/manifest.json.
    def tree(root: pathlib.Path) -> dict[str, pathlib.Path]:
        return {p.relative_to(root).as_posix(): p for p in root.rglob('*') if p.is_file()}
    out_l, out_r = tree(L / 'output'), tree(R / 'output')
    assert set(out_l) == set(out_r)
    output_diffs = []
    for name in sorted(out_l):
        left, right = out_l[name].read_bytes(), out_r[name].read_bytes()
        if left != right:
            output_diffs.append(name)
    assert len(out_l) == 35 and output_diffs == ['manifest.json']
    manifest_l, manifest_r = read_json(out_l['manifest.json']), read_json(out_r['manifest.json'])
    manifest_diffs = flatten_diff(manifest_l, manifest_r)
    allowed_manifest_diffs = {
        '/materials/glue_path', '/materials/liner_path', '/materials/medium_path'
    }
    assert {x['path'] for x in manifest_diffs} == allowed_manifest_diffs
    assert all(normalize_text(x['left']) == normalize_text(x['right']) for x in manifest_diffs)
    assert (L / 'stdout.log').read_bytes() == (R / 'stdout.log').read_bytes()

    # Shared legacy snapshots must match byte-for-byte, including payloads.
    binary_names = ('base-cgls.bin', 'base-cgls.bin.state.bin', 'first-status5.bin')
    binary_checks = {}
    for name in binary_names:
        lb, rb = (L / name).read_bytes(), (R / name).read_bytes()
        assert lb == rb, f'{name} differs'
        binary_checks[name] = {'bytes': len(lb), 'sha256': sha(L / name), 'byte_identical': True}
    assert binary_checks['base-cgls.bin']['bytes'] == 409368
    assert binary_checks['base-cgls.bin.state.bin']['bytes'] == 4327464
    assert binary_checks['first-status5.bin']['bytes'] == 13568

    state_l, state_r = read_json(L / 'base-cgls.bin.state.json'), read_json(R / 'base-cgls.bin.state.json')
    state_diffs = flatten_diff(state_l, state_r)
    assert [x['path'] for x in state_diffs] == ['/micro_generation']
    assert state_diffs[0]['right'] == 2
    status_l, status_r = read_json(L / 'first-status5.json'), read_json(R / 'first-status5.json')
    status_diffs = flatten_diff(status_l, status_r)
    assert [x['path'] for x in status_diffs] == ['/payload/path']
    assert normalize_text(status_l['payload']['path']) == normalize_text(status_r['payload']['path'])
    assert status_l['payload']['sha256'] == status_r['payload']['sha256'] == sha(L / 'first-status5.bin')

    # Numeric/production trace rows must match with every non-timing row
    # compared. Only the explicitly conflicting legacy observer and its
    # expected skip are removed from 462; 464 adds its regional capture marker.
    trace_l, trace_r = parse_stderr(L / 'stderr.log'), parse_stderr(R / 'stderr.log')
    timing_l = [x for x in trace_l if x[0] == 'cardboard_step_timing']
    timing_r = [x for x in trace_r if x[0] == 'cardboard_step_timing']
    production_l = [x for x in trace_l if x[0] != 'cardboard_step_timing']
    production_r = [x for x in trace_r if x[0] != 'cardboard_step_timing']
    observer_only_l = [x for x in production_l if x[0] in {'fem_coupled_capture_enabled', 'fem_coupled_capture_skipped'}]
    observer_only_r = [x for x in production_r if x[0] == 'fem_base_cgls_regional_capture']
    assert [x[0] for x in observer_only_l] == ['fem_coupled_capture_enabled', 'fem_coupled_capture_skipped']
    assert len(observer_only_r) == 1
    common_l = [normalized_event(x) for x in production_l if x[0] not in {'fem_coupled_capture_enabled', 'fem_coupled_capture_skipped'}]
    common_r = [normalized_event(x) for x in production_r if x[0] != 'fem_base_cgls_regional_capture']
    assert common_l == common_r, 'a non-timing/common production event differs'
    counts_l = collections.Counter(k for k, _ in trace_l)
    counts_r = collections.Counter(k for k, _ in trace_r)
    for key, expected in {
        'fem_true_residual_cycle': 4096,
        'fem_residual_armijo_iteration': 112,
        'fem_base_cgls_direction': 64,
    }.items():
        assert counts_l[key] == counts_r[key] == expected

    # Strict reader validates observer payload inventory, joins and geometry.
    sys.path.insert(0, str(READER_DIR))
    spec = importlib.util.spec_from_file_location('regional_capture_reader_v2', READER)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    regional_manifest = R / 'base-cgls.bin.regional.json'
    regional_read = module.read_capture(regional_manifest)
    assert regional_read['identity_joined'] is True
    assert regional_read['regional_payload_bytes'] == 77619992
    assert regional_read['regional_array_count'] == 30
    assert regional_read['graph_layout_checksum_recomputed'] is True
    assert regional_read['graph_source_topology_matches_joined_state'] is True
    assert regional_read['graph_source_free_mask_matches_joined_state'] is True
    assert regional_read['all_operator_arrays_finite'] is True
    assert regional_read['augmented_regional_arrays_admission'].startswith('UNADMITTED:')
    regional_meta = read_json(regional_manifest)
    array_meta = {x['name']: x for x in regional_meta['arrays']}
    selected_entry = array_meta['primary_policy_selected_environment']
    regional_payload = (R / 'base-cgls.bin.regional.bin').read_bytes()
    selected_offset = selected_entry['offset']
    selected_words = [int.from_bytes(regional_payload[selected_offset+i*4:selected_offset+(i+1)*4], 'little')
                      for i in range(regional_meta['environment_count'])]
    assert selected_words == [0, 0]
    assert regional_meta['capture_point'] == 'selected_newton_after_fgmres_regional_rap_before_cgls_direction_construction_or_installation'
    assert regional_meta['newton_iteration'] == 48
    assert regional_meta['source_physics_fingerprint_u64_hex'] == regional_meta['world_fingerprint_u64_hex'] == '75c1264bc96ac2d9'
    assert regional_meta['device_program_fingerprint_u64_hex'] == '9ce5de35a9e1be1c'

    # Source/build binding overview: every frozen source file hash is verified
    # against its snapshot; expected source delta is exactly the regional observer.
    snap_l = B / 'snapshots/source-cgls-block-right-preconditioned-149'
    snap_r = B / 'snapshots/source-cgls-regional-state-capture-151'
    frozen_l, frozen_r = read_json(snap_l / 'frozen-source.json'), read_json(snap_r / 'frozen-source.json')
    hashes_l, hashes_r = frozen_l['source_sha256'], frozen_r['source_sha256']
    assert all(sha(snap_l / rel) == digest for rel, digest in hashes_l.items())
    assert all(sha(snap_r / rel) == digest for rel, digest in hashes_r.items())
    source_delta = {
        rel: {'source149': hashes_l.get(rel), 'source151': hashes_r.get(rel)}
        for rel in sorted(set(hashes_l) | set(hashes_r))
        if hashes_l.get(rel) != hashes_r.get(rel)
    }
    assert set(source_delta) == {
        'matter/src/runtime.mm',
        'matter/tools/read_regional_capture.py',
        'matter/tools/test_read_regional_capture.py',
    }

    # Record exact source/build/run inputs and computed verdict.
    run_bindings = {}
    build_bindings = {}
    for label, case, prediction in (('462', L, read_json(L / 'prediction.json')),
                                    ('464', R, read_json(R / 'prediction.json'))):
        run_bindings[label] = {
            'case': str(case), 'receipt_sha256': sha(case / 'receipt.json'),
            'prediction_sha256': sha(case / 'prediction.json'),
            'source_snapshot': prediction['source_snapshot'],
            'runtime_dir': prediction['runtime_dir'],
            'build_provenance': prediction['build_provenance'],
            'executed_build_verified': prediction['executed_build_verified'],
            'receipt': read_json(case / 'receipt.json'),
        }
        assert prediction['executed_build_verified'] is True
        build = read_json(pathlib.Path(prediction['build_provenance']))
        snapshot = pathlib.Path(prediction['source_snapshot'])
        frozen = read_json(snapshot / 'frozen-source.json')
        assert build['success'] is True
        assert build['source_snapshot'] == str(snapshot)
        assert build['source_sha256'] == frozen['source_sha256']
        artifact_results = {}
        for artifact_path, expected_sha in build['artifacts_sha256'].items():
            actual_path = pathlib.Path(artifact_path)
            assert sha(actual_path) == expected_sha, f'build artifact hash mismatch: {actual_path}'
            artifact_results[str(actual_path)] = expected_sha
        build_bindings[label] = {
            'source_build_json': prediction['build_provenance'],
            'source_build_json_sha256': sha(pathlib.Path(prediction['build_provenance'])),
            'success': build['success'],
            'source_file_count': len(build['source_sha256']),
            'all_source_hashes_match_frozen_snapshot': True,
            'artifact_hashes_verified': artifact_results,
        }
    regional_sha = sha(R / 'base-cgls.bin.regional.bin')

    report = {
        'schema': 'numi.cardboard.source151_regional_observer_full_attempt_parity.v1',
        'verdict': 'PASS_FULL_ATTEMPT_PARITY_WITH_OBSERVER_METADATA_DEFECT',
        'claim': 'The source151 regional state observer did not change the attempted physical/solver history relative to source149; both runs retain the same failed control-step outcome.',
        'runs': run_bindings,
        'builds': build_bindings,
        'native_argv_after_path_mapping_equal': True,
        'environment_difference': {
            '462_only_mutually_exclusive_legacy_coupled_observer': {
                'NM_FEM_COUPLED_CAPTURE_SELECTOR': env_l['NM_FEM_COUPLED_CAPTURE_SELECTOR'],
                'NM_FEM_COUPLED_CAPTURE_PATH': 'run-local path; coupled observer skipped because environment_count=2',
            },
            '464_only_regional_observer': added,
            'all_other_environment_values_equal_after_path_mapping': True,
        },
        'source_changes': source_delta,
        'source_snapshot_inventory': {
            '149_count': len(hashes_l), '151_count': len(hashes_r),
            'all_frozen_hashes_reverified': True,
            'interpretation': 'source151 adds the regional observer runtime change and two reader/test tools; no other frozen source file hashes differ.',
        },
        'attempt_outcome': {
            'same_result_json': result_l == result_r,
            'status': result_l['status'], 'accepted_steps': result_l['accepted_steps'],
            'physical_validation': result_l['physical_validation'],
            'stdout_byte_identical': True,
            'receipts': 'both exit 2; neither timed out; no termination error; source inputs unchanged',
        },
        'output_tree': {
            'file_count_each': len(out_l), 'relative_names_equal': True,
            'byte_identical_files': len(out_l) - len(output_diffs),
            'raw_difference': output_diffs,
            'normalized_manifest_differences': [x['path'] for x in manifest_diffs],
        },
        'production_trace': {
            'true_residual_cycle_rows': counts_l['fem_true_residual_cycle'],
            'armijo_rows': counts_l['fem_residual_armijo_iteration'],
            'cgls_direction_rows': counts_l['fem_base_cgls_direction'],
            'all_common_non_timing_event_rows_equal': True,
            'common_non_timing_row_count': len(common_l),
            'timing_rows_excluded': {'462': len(timing_l), '464': len(timing_r)},
            'explicit_observer_rows': {
                '462': [k for k, _ in observer_only_l],
                '464': [k for k, _ in observer_only_r],
            },
        },
        'legacy_capture_payloads': binary_checks,
        'legacy_sidecar_metadata_differences': {
            'state_json': state_diffs,
            'status5_json': status_diffs,
            'payloads_remain_byte_identical': True,
        },
        'regional_observer': {
            'manifest': str(regional_manifest), 'manifest_sha256': sha(regional_manifest),
            'payload': str(R / 'base-cgls.bin.regional.bin'),
            'payload_sha256': regional_sha,
            'reader_result': regional_read,
            'selection_field_caveat': {
                'array': 'primary_policy_selected_environment',
                'raw_words': selected_words,
                'meaning': 'Source runtime capture code appends this buffer before CGLS direction construction/installation; the selection buffer is only written by the later CGLS layout/selection path. The serialized words are therefore pre-layout scratch, not current selection state. They are zero for both environments in this capture.',
                'admission': 'do not use this field as authoritative current selection; retain as a metadata defect pending a separately versioned observer repair.',
                'source_line_bindings': {
                    'capture_array_append': {'file': str(snap_r/'matter/src/runtime.mm'), 'line': 23173},
                    'capture_invocation_before_direction_construction': {'file': str(snap_r/'matter/src/runtime.mm'), 'line': 24204},
                    'later_cgls_selection_buffer_use': {'file': str(snap_r/'matter/src/runtime.mm'), 'line': 23948},
                },
            },
            'interpretation_limit': 'The reader confirms exact joined state, topology, graph layout, finite arrays, and valid base data. Augmented RAP is unadmitted because the consuming restart cycle is not captured; projected material-history currentness is unavailable. This sidecar is diagnostic only.',
        },
        'limitations': [
            'The experiment is full attempted-history parity for an observer-only change, not a successful solver or physical qualification.',
            'Both coupon attempts fail the same step after one accepted step.',
            'The legacy coupled observer was omitted from 464 because the two observers are mutually exclusive; its 462 marker records an environment-count-out-of-scope skip with zero payload.',
            'The regional observer does not establish that the augmented RAP belongs to the consumed restart cycle.',
        ],
    }
    (O / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    inputs = [CMD_L, CMD_R, L/'prediction.json', R/'prediction.json', L/'receipt.json', R/'receipt.json',
              L/'stdout.log', R/'stdout.log', L/'stderr.log', R/'stderr.log',
              L/'base-cgls.bin', R/'base-cgls.bin', L/'base-cgls.bin.state.bin', R/'base-cgls.bin.state.bin',
              L/'base-cgls.bin.state.json', R/'base-cgls.bin.state.json',
              L/'first-status5.bin', R/'first-status5.bin', L/'first-status5.json', R/'first-status5.json',
              L/'output/result.json', R/'output/result.json', L/'output/manifest.json', R/'output/manifest.json',
              READER, READER_DIR/'read_regional_capture_v1.py', regional_manifest, R/'base-cgls.bin.regional.bin',
              snap_l/'frozen-source.json', snap_r/'frozen-source.json',
              B/'analysis/native462-launch-guard-001/receipt.json',
              B/'analysis/native464-regional-capture-command-001/launch-guard/receipt.json',
              B/'analysis/native464-regional-capture-command-001/preflight/command-validation.json',
              B/'analysis/native464-regional-capture-command-001/source151-regional-capture-bindings.json',
              O/'README.md', O/'failed-attempt001.json', O/'failed-attempt002.json',
              pathlib.Path(run_bindings['462']['build_provenance']),
              pathlib.Path(run_bindings['464']['build_provenance'])]
    for build_record in build_bindings.values():
        inputs.extend(pathlib.Path(p) for p in build_record['artifact_hashes_verified'])
    for rel in sorted(out_l):
        inputs.extend([out_l[rel], out_r[rel]])
    manifest = {
        'schema': 'numi.cardboard.source151_regional_observer_full_attempt_parity_manifest.v1',
        'created_by': 'audit.py', 'audit_sha256': sha(pathlib.Path(__file__)),
        'report_sha256': sha(O/'report.json'),
        'package_artifacts': {
            'README.md': sha(O/'README.md'),
            'failed-attempt001.json': sha(O/'failed-attempt001.json'),
            'failed-attempt002.json': sha(O/'failed-attempt002.json'),
        },
        'inputs': {str(p): sha(p) for p in sorted(set(inputs), key=str)},
        'source_snapshot_file_counts': {'149': len(hashes_l), '151': len(hashes_r)},
        'source_snapshot_sha256': {'149': sha(snap_l/'frozen-source.json'), '151': sha(snap_r/'frozen-source.json')},
        'regional_reader_sha256': sha(READER),
    }
    (O / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'verdict': report['verdict'], 'output_files': len(out_l),
                      'trace_counts': {k: counts_l[k] for k in ('fem_true_residual_cycle','fem_residual_armijo_iteration','fem_base_cgls_direction')},
                      'common_non_timing_rows': len(common_l), 'regional_payload_sha256': regional_sha,
                      'report_sha256': sha(O/'report.json'), 'manifest_sha256': sha(O/'manifest.json')}, indent=2))

if __name__ == '__main__':
    main()
