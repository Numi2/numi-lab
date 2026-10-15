#!/usr/bin/env python3
"""Strict adjudication of native Metal CCD attempt 357 and related traces."""
from __future__ import annotations
import hashlib
import json
import math
import pathlib
import shutil
import struct
import sys
from typing import Any

BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014')
HERE = pathlib.Path(__file__).resolve().parent
CCD_PACKAGE = BASE / 'analysis/private-ccd-metal-compare-007'
CCD_RUN = BASE / 'analysis/private-ccd-metal-native-357'
TRACE_TOOL = BASE / 'analysis/private-trace-health-001/trace_health.py'
TRACE_CASES = [
    ('355', 'whole-mortar-free-body-baseline-355', None, None),
    ('358', 'first-fatal-private-trial-358', 'private-trial.coupled.json',
     'after_deformable_contact_before_free_body_terms'),
    ('359', 'first-fatal-center-359', 'center-capture.coupled.json', None),
]
sys.path.insert(0, str(TRACE_TOOL.parent))
from trace_health import (  # type: ignore  # noqa: E402
    adjudicate_capture, adjudicate_sequence, parse_armijo_log, read_capture_status,
    sha256_bytes,
)


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: pathlib.Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f'{path}: nonstandard JSON constant {value}')
    return json.loads(path.read_text(), parse_constant=reject)


def strict_json_bytes(data: bytes, where: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f'{where}: nonstandard JSON constant {value}')
    return json.loads(data, parse_constant=reject)


def copy_raw(src: pathlib.Path, dst: pathlib.Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return {'source_path': str(src), 'package_path': str(dst.relative_to(HERE)),
            'bytes': dst.stat().st_size, 'sha256': sha(dst)}


def adjudicate_ccd() -> dict[str, Any]:
    manifest_path = CCD_PACKAGE / 'manifest.json'
    manifest = strict_json(manifest_path)
    result_path = CCD_RUN / 'native-result.json'
    prediction_path = CCD_RUN / 'prediction.json'
    receipt_path = CCD_RUN / 'receipt.json'
    result = strict_json(result_path)
    prediction = strict_json(prediction_path)
    receipt = strict_json(receipt_path)
    input_manifest_path = CCD_PACKAGE / manifest['cases']['manifest']
    case_manifest = strict_json(input_manifest_path)
    payload_path = CCD_PACKAGE / manifest['cases']['payload']
    payload = payload_path.read_bytes()

    # Check every sealed 007 package file, not just the native runner inputs.
    bad_files = []
    for entry in manifest['files']:
        path = CCD_PACKAGE / entry['path']
        if (not path.is_file() or path.stat().st_size != entry['bytes']
                or sha(path) != entry['sha256']):
            bad_files.append(entry['path'])
    if bad_files:
        raise ValueError(f'sealed 007 files differ: {bad_files}')
    if sha(manifest_path) != prediction.get('package_manifest_sha256'):
        raise ValueError('357 prediction does not bind the exact 007 package manifest')

    build = strict_json(CCD_PACKAGE / 'build/build-manifest.json')
    file_sha = {entry['path']: entry['sha256'] for entry in manifest['files']}
    expected_bindings = {
        'contact_source_sha256': manifest['source']['contact_source_sha256'],
        'frozen_source_manifest_sha256': manifest['source']['frozen_manifest_sha256'],
        'generated_metal_sha256': file_sha['build/private_ee_ccd_diagnostic.metal'],
        'build_manifest_sha256': manifest['build_manifest_sha256'],
        'case_manifest_sha256': manifest['cases']['manifest_sha256'],
        'case_payload_sha256': manifest['cases']['payload_sha256'],
        'metallib_sha256': file_sha['build/private_ee_ccd_diagnostic.metallib'],
    }
    binding_checks = {key: result.get(key) == value for key, value in expected_bindings.items()}
    if not all(binding_checks.values()):
        raise ValueError(f'native 357 hash binding mismatch: {binding_checks}')
    if (result.get('schema') != 'numi.matter.private_ee_ccd_metal_diagnostic_result.v2'
            or result.get('gpu_result') is not True
            or result.get('solver_acceptance_evaluated') is not False
            or result.get('physics_acceptance_evaluated') is not False):
        raise ValueError('native result schema or diagnostic-only flags mismatch')
    if (receipt.get('exit_code') != 0 or receipt.get('inputs_unchanged') is not True
            or receipt.get('physical_transactions') != 0):
        raise ValueError('357 receipt does not describe a clean diagnostic execution')
    argv = prediction.get('argv')
    sealed_argv = manifest.get('native_run_argv')
    if not isinstance(argv, list) or not isinstance(sealed_argv, list):
        raise ValueError('missing native argv binding')
    if argv[:-1] != sealed_argv[:-1] or argv[-1] != str(result_path):
        raise ValueError('357 argv differs from sealed package except for its output path')

    if len(payload) != case_manifest['payload_bytes'] or sha(payload_path) != case_manifest['payload_sha256']:
        raise ValueError('case payload byte count/hash mismatch')
    if len(case_manifest['cases']) != 4 or len(result.get('cases', [])) != 4:
        raise ValueError('expected exactly four packed native cases')
    expected_order = [(333, 16, 20), (333, 17, 20), (334, 16, 21), (334, 17, 21)]
    if case_manifest['case_stride_bytes'] != 144 or case_manifest['case_stride_float4'] != 9:
        raise ValueError('unexpected input record stride')
    capture_bindings = []
    for capture in case_manifest['captures']:
        cap_path = pathlib.Path(capture['manifest_path'])
        cap = strict_json(cap_path)
        cap_sha = sha(cap_path)
        pay_meta = cap.get('payload')
        cap_payload = pathlib.Path(pay_meta['file'])
        capture_ok = (
            cap_sha == capture['manifest_sha256']
            and sha(cap_payload) == capture['payload_sha256'] == pay_meta['sha256']
            and cap.get('root') == 1201 and cap.get('iteration') == 12
            and cap.get('trial') in (20, 21)
        )
        if not capture_ok:
            raise ValueError(f'input capture join failed for {cap_path}')
        capture_bindings.append({
            'capture_case': capture['case'], 'root': cap['root'],
            'iteration': cap['iteration'], 'trial': cap['trial'],
            'capture_manifest_path': str(cap_path), 'capture_manifest_sha256': cap_sha,
            'capture_payload_path': str(cap_payload), 'capture_payload_sha256': sha(cap_payload),
            'primitive_nodes': capture['primitive_nodes'],
            'candidate_identity': capture['candidate_identity'],
            'object_dt_f32_hex': capture['object_dt_f32_hex'],
            'thickness_f32_hex': capture['thickness_f32_hex'],
        })

    rows = []
    for idx, (case_meta, native) in enumerate(zip(case_manifest['cases'], result['cases'])):
        expected = expected_order[idx]
        key = (case_meta['capture_case'], case_meta['budget'], case_meta['capture_trial'])
        native_key = (native.get('capture_case'), native.get('budget'), case_meta['capture_trial'])
        if key != expected or native_key != expected or native.get('index') != idx:
            raise ValueError(f'case order/capture join mismatch at index {idx}: {key}, {native_key}, native_index={native.get("index")}')
        raw = b''.join(bytes.fromhex(word) for word in case_meta['records_f32_hex'])
        if len(raw) != 144 or payload[idx * 144:(idx + 1) * 144] != raw:
            raise ValueError(f'packed little-endian input words differ at index {idx}')
        if case_meta['edge_node_ids'] != [[13, 14], [28, 32]] or case_meta['kind'] != 'edge_edge':
            raise ValueError(f'unexpected input feature mapping at index {idx}')
        inst = native['instrumented']
        iterations = inst['iterations']
        if [it.get('iteration') for it in iterations] != list(range(len(iterations))):
            raise ValueError(f'incomplete or reordered native iteration trace at index {idx}')
        if len(iterations) != native['budget']:
            raise ValueError(f'iteration loop count does not equal authored budget at index {idx}')
        prod = native['production_budget16']
        parity = inst.get('budget16_bitwise_matches_production')
        if native['budget'] == 16:
            if parity is not True:
                raise ValueError(f'16-step instrumented scalar differs from production at index {idx}')
            if any(inst[k] != prod[k] for k in ('hit', 'certified', 'classification', 'toi', 'separation_m')):
                raise ValueError(f'16-step displayed scalars differ at index {idx}')
        elif native['budget'] == 17 and parity is not None:
            raise ValueError(f'17-step case incorrectly claims production bitwise parity at index {idx}')
        last = iterations[-1]['time_distance_effective_thickness_speed']
        t, distance, effective_thickness, speed = map(float, last)
        gap = distance - effective_thickness
        next_step = max(gap / speed, 1.0e-6) if speed > 0 else None
        next_t = t + next_step if next_step is not None else None
        rows.append({
            'index': idx, 'capture_case': native['capture_case'],
            'capture_trial': case_meta['capture_trial'], 'budget': native['budget'],
            'input_record_sha256': hashlib.sha256(raw).hexdigest(),
            'edge_node_ids': case_meta['edge_node_ids'],
            'production16': prod,
            'instrumented_result': {k: inst.get(k) for k in (
                'hit', 'certified', 'classification', 'toi', 'separation_m',
                'budget16_bitwise_matches_production')},
            'iterations_executed': len(iterations),
            'last_sample': {
                'iteration': iterations[-1]['iteration'], 'time': t,
                'distance_m': distance, 'effective_thickness_m': effective_thickness,
                'clearance_m': gap, 'speed_bound_m_per_normalized_time': speed,
                'conservative_step_from_last_sample': next_step,
                'predicted_next_normalized_time_from_reported_floats': next_t,
            },
        })

    raw_files = []
    for src, rel in [
        (result_path, 'raw/ccd357/native-result.json'),
        (prediction_path, 'raw/ccd357/prediction.json'),
        (receipt_path, 'raw/ccd357/receipt.json'),
        (CCD_RUN / 'stderr.log', 'raw/ccd357/stderr.log'),
        (CCD_RUN / 'stdout.log', 'raw/ccd357/stdout.log'),
        (manifest_path, 'raw/ccd357/package-manifest.json'),
        (input_manifest_path, 'raw/ccd357/captured-ee-cases.json'),
        (payload_path, 'raw/ccd357/captured-ee-cases.float4.bin'),
        (CCD_PACKAGE / 'build/build-manifest.json', 'raw/ccd357/build-manifest.json'),
    ]:
        raw_files.append(copy_raw(src, HERE / rel))

    return {
        'package_manifest_sha256': sha(manifest_path),
        'sealed_package_file_count': len(manifest['files']),
        'sealed_package_file_mismatches': [],
        'native_result_sha256': sha(result_path),
        'prediction_sha256': sha(prediction_path),
        'receipt_sha256': sha(receipt_path),
        'native_receipt': receipt,
        'native_result_flags': {k: result.get(k) for k in (
            'gpu_result', 'solver_acceptance_evaluated', 'physics_acceptance_evaluated', 'qualification')},
        'hash_binding_checks': binding_checks,
        'native_argv_matches_sealed_package_except_output_path': True,
        'input_payload': {
            'path': str(payload_path), 'bytes': len(payload), 'sha256': sha(payload_path),
            'case_manifest_path': str(input_manifest_path),
            'case_manifest_sha256': sha(input_manifest_path),
            'stride_bytes': case_manifest['case_stride_bytes'],
            'record_count': case_manifest['case_count'],
        },
        'source_capture_bindings': capture_bindings,
        'cases': rows,
        'interpretation': {
            'budget16': 'both captured cases match the unchanged production CCD scalar output bit-for-bit and remain uncertified after 16 iterations',
            'budget17': 'both source-extracted 17-step clones classify certified_miss with no hit; this does not change or qualify the production 16-step kernel',
            'last_step_analysis': 'the recorded 16th sample advances conservatively to normalized time about 0.999548, still inside [0,1]; at the next sample the safe advancement exceeds the endpoint (about 1.00488), consistent with the exact source branch that certifies a miss when time exceeds 1 + 8*kEpsilon. The result schema does not report an explicit branch code, so branch attribution is based on source order plus the serialized sample values.',
            'scope_limit': 'CCD classification only. It does not evaluate residual Armijo, solver acceptance, transaction acceptance, or physical validity of the enclosing candidate.',
        },
        'raw_files': raw_files,
    }


def adjudicate_trace_case(case_id: str, dirname: str, capture_name: str | None,
                          phase: str | None) -> dict[str, Any]:
    source = BASE / 'native' / dirname
    log_path = source / 'stderr.log'
    prediction_path = source / 'prediction.json'
    receipt_path = source / 'receipt.json'
    rows = parse_armijo_log(log_path.read_bytes())
    seq = adjudicate_sequence(rows, expected_root=1201, environment_index=0)
    pred = strict_json(prediction_path)
    rec = strict_json(receipt_path)
    if rec.get('source_inputs_unchanged') is not True or rec.get('timed_out') is not False:
        raise ValueError(f'{case_id} receipt has wrong source/timeout status')
    capture = None
    capture_status = None
    decision = None
    raw_dir = HERE / f'raw/trace{case_id}'
    raw_files = [
        copy_raw(log_path, raw_dir / 'stderr.log'),
        copy_raw(source / 'stdout.log', raw_dir / 'stdout.log'),
        copy_raw(prediction_path, raw_dir / 'prediction.json'),
        copy_raw(receipt_path, raw_dir / 'receipt.json'),
    ]
    if capture_name:
        capture_path = source / capture_name
        capture = strict_json(capture_path)
        raw_files.append(copy_raw(capture_path, raw_dir / capture_name))
        if phase:
            capture_status = read_capture_status(capture_path, phase)
        decision = adjudicate_capture(seq, capture, capture_trial_status=capture_status)
    selected = []
    picks = sorted(set([0, max(0, (seq['first_fatal_iteration'] or 0) - 1),
                        seq['first_fatal_iteration'] or 0,
                        (seq['first_fatal_iteration'] or 0) + 1,
                        seq['record_count'] - 1]))
    by_iter = {r['newton_iteration']:r for r in rows}
    for iteration in picks:
        if iteration in by_iter:
            selected.append(by_iter[iteration])
    return {
        'case_id': case_id,
        'native_directory': str(source),
        'prediction_sha256': sha(prediction_path), 'receipt_sha256': sha(receipt_path),
        'stderr_sha256': sha(log_path),
        'source_snapshot': pred.get('source_snapshot'),
        'source_build_manifest': pred.get('build_provenance'),
        'receipt': rec,
        'source_input_hash_map_sha256': hashlib.sha256(
            json.dumps(pred.get('inputs_sha256', {}), sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'raw_armijo_records': rows,
        'trace_adjudication': {
            'root': seq['root'], 'environment_index': seq['environment_index'],
            'record_count': seq['record_count'],
            'iteration_sequence': seq['iteration_sequence'],
            'first_fatal_iteration': seq['first_fatal_iteration'],
            'rows': seq['rows'], 'selected_summary_rows': selected,
        },
        'capture_manifest': None if capture is None else {
            'path': str(source / capture_name), 'sha256': sha(source / capture_name),
            'schema': capture.get('schema'), 'root': capture.get('root'),
            'iteration': capture.get('iteration'), 'trial': capture.get('trial'),
            'selected_phase': capture.get('selected_phase'),
            'capture_point': capture.get('capture_point'),
            'payload_bytes': capture.get('payload', {}).get('byte_count'),
            'payload_sha256': capture.get('payload', {}).get('sha256'),
            'selected_private_status': capture_status,
        },
        'capture_comparison_decision': decision,
        'raw_files': raw_files,
        'physical_verdict': 'failed native transaction; trace/capture does not qualify a converged physical step',
    }


def main() -> None:
    ccd = adjudicate_ccd()
    traces = [adjudicate_trace_case(*spec) for spec in TRACE_CASES]
    # Verify the independent on/off runs preserve the same source inputs,
    # all 32 trace rows, and the nine non-log transaction state files.
    def normalized_records(case: dict[str, Any]) -> list[dict[str, Any]]:
        return [{k: v for k, v in row.items() if k != '_source_line'}
                for row in case['raw_armijo_records']]
    trace_rows_equal = all(
        normalized_records(traces[0]) == normalized_records(item)
        for item in traces[1:])
    input_maps_equal = len({item['source_input_hash_map_sha256'] for item in traces}) == 1
    dirs = [BASE / 'native' / spec[1] for spec in TRACE_CASES]
    def state_map(directory: pathlib.Path) -> dict[str, tuple[int, str]]:
        output = {}
        for path in (directory / 'diagnostic-state').rglob('*'):
            if path.is_file() and path.suffix != '.log':
                output[str(path.relative_to(directory / 'diagnostic-state'))] = (path.stat().st_size, sha(path))
        return output
    states = [state_map(d) for d in dirs]
    states_equal = all(x == states[0] for x in states[1:])
    if not (trace_rows_equal and input_maps_equal and states_equal):
        raise ValueError('355/358/359 source, trace, or production-state parity check failed')
    report = {
        'schema': 'numi.cardboard.private_ccd_metal_adjudication_357.v1',
        'adjudicator': str(HERE / 'adjudicate.py'),
        'strict_json': True,
        'ccd_native357': ccd,
        'related_native_trace_health': {
            'adjudicator_path': str(TRACE_TOOL), 'adjudicator_sha256': sha(TRACE_TOOL),
            'cases': traces,
            'capture_off_on_parity': {
                'run_ids': ['355', '358', '359'],
                'all_32_armijo_records_equal': trace_rows_equal,
                'source_build_input_hash_maps_equal': input_maps_equal,
                'diagnostic_state_nonlog_file_counts': [len(x) for x in states],
                'all_nonlog_transaction_state_files_equal': states_equal,
                'comparison_scope': 'nine diagnostic-state transaction .bin/.json files per run; per-run diagnostic .log files excluded',
            },
            'interpretation': 'All three logs have first solver-fatal status at Newton 7. Newton 7 has a finite predecision center base, but its line search is rejected; the selected private trial at 358 separately fails contact rebuild with status 6/index 283. The Newton 7 center capture at 359 is not an accepted trial pair. Later residual rows are failure-latched and invalid as physical center residuals.',
        },
        'qualification': 'No physical or solver acceptance. CCD checker is a bounded native Metal diagnostic; 355/358/359 are failed transaction traces.',
    }
    encoded = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    (HERE / 'report.json').write_bytes(encoded)
    (HERE / 'report.sha256').write_text(sha256_bytes(encoded) + '  report.json\n')
    entries = []
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and path.name not in {'package-manifest.json', 'package-manifest.sha256'}:
            entries.append({'path':str(path.relative_to(HERE)), 'bytes':path.stat().st_size, 'sha256':sha(path)})
    package = {
        'schema':'numi.cardboard.private_ccd_metal_adjudication_package.v1',
        'report_sha256':sha(HERE/'report.json'),
        'adjudicator_sha256':sha(HERE/'adjudicate.py'),
        'trace_health_source_sha256':sha(TRACE_TOOL),
        'files':entries,
    }
    pbytes=(json.dumps(package,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    (HERE/'package-manifest.json').write_bytes(pbytes)
    (HERE/'package-manifest.sha256').write_text(sha256_bytes(pbytes)+'  package-manifest.json\n')
    print(json.dumps({
        'report':str(HERE/'report.json'),'report_sha256':sha(HERE/'report.json'),
        'package_manifest_sha256':sha(HERE/'package-manifest.json'),
        'ccd_cases':[{k:c[k] for k in ('capture_case','budget','instrumented_result','iterations_executed','last_sample')}
                     for c in ccd['cases']],
        'trace_first_fatal':{c['case_id']:c['trace_adjudication']['first_fatal_iteration'] for c in traces},
        'trace_capture_decisions':{c['case_id']:c['capture_comparison_decision'] for c in traces if c['capture_comparison_decision'] is not None},
        'trace_parity':report['related_native_trace_health']['capture_off_on_parity'],
    },indent=2))

if __name__ == '__main__':
    main()
