#!/usr/bin/env python3
"""Verify the source-bound MP90 sensitivity package without recomputing endpoints."""
from decimal import Decimal
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise SystemExit(message)

def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, expected in manifest['artifacts'].items():
        path = ROOT / name
        require(path.is_file(), f'missing artifact: {name}')
        require(path.stat().st_size == expected['bytes'], f'byte-size mismatch: {name}')
        require(sha(path) == expected['sha256'], f'hash mismatch: {name}')
    for item in manifest['source_bindings']:
        path = Path(item['path'])
        require(path.is_file(), f'missing source input: {item["role"]}')
        require(sha(path) == item['sha256'], f'source input hash mismatch: {item["role"]}')

    report = json.loads((ROOT / 'report.json').read_text())
    common = json.loads((ROOT / 'common-epsilon-1e-7.json').read_text())
    summary = json.loads((ROOT / 'summary.json').read_text())
    require(report['bindings']['capture_sha256'] == manifest['source_index']['capture_sha256'], 'capture binding mismatch')
    require(report['bindings']['native_jsonl_sha256'] == manifest['source_index']['native300_accurate_sha256'], 'native300 binding mismatch')
    require(common['bindings']['capture_sha256'] == manifest['source_index']['capture_sha256'], 'common-epsilon capture mismatch')
    require(common['bindings']['native_jsonl_sha256'] == manifest['source_index']['native300_accurate_sha256'], 'common-epsilon native mismatch')
    require(summary['qualification'] is False, 'package must remain diagnostic only')
    require(all(summary['classification_checks'].values()), 'classification check false')
    require(len(summary['points']) == 2, 'expected exactly two selected points')
    tol = Decimal('1e-70')
    for point in report['results']:
        require(point['source_inputs']['Fdot_relationship_preserved'] is True, 'Fdot relationship not preserved')
        require(point['source_inputs']['accepted_history_policy'].startswith('same captured accepted_state'), 'accepted history changed')
        require(point['center_projection']['converged'], 'center projection failed')
        require(Decimal(point['center_projection']['terminal_scaled_residual']) <= tol, 'center residual gate failed')
        require(Decimal(point['center_projection']['terminal_newton_correction_inf']) <= tol, 'center correction gate failed')
        for sweep in point['nine_direction_endpoint_sweep']:
            require(sweep['all_endpoint_projections_converged'], 'matrix endpoint failed')
            require(not sweep['branch_transition_columns'], 'matrix branch transition present')
            for endpoint in sweep['endpoint_records']:
                require(endpoint['same_yield_and_FB_branch'], 'matrix endpoint branch mismatch')
                for side in ('plus', 'minus'):
                    projection = endpoint[side]['projection']
                    require(projection['converged'], f'{side} endpoint failed')
                    require(Decimal(projection['terminal_scaled_residual']) <= tol, f'{side} residual gate failed')
                    require(Decimal(projection['terminal_newton_correction_inf']) <= tol, f'{side} correction gate failed')
        for sweep in point['captured_Fdot_endpoint_sweep']:
            for side in ('plus_projection', 'minus_projection'):
                projection = sweep[side]
                require(projection['converged'], f'Fdot {side} failed')
                require(Decimal(projection['terminal_scaled_residual']) <= tol, f'Fdot {side} residual gate failed')
                require(Decimal(projection['terminal_newton_correction_inf']) <= tol, f'Fdot {side} correction gate failed')

    native = summary['native306_context']
    require(native['receipt']['execution_started'] and native['receipt']['exit_code'] == 0 and not native['receipt']['timed_out'], 'native306 receipt invalid')
    for policy, points in native['policies'].items():
        for point in points.values():
            require(point['physical_steps'] == 0 and point['physical_validation'] is False, f'native306 {policy} must remain diagnostic-only')
            require(point['accepted_history_unchanged'] and point['center_repeats_bit_identical'], f'native306 {policy} diagnostic integrity failed')
    print(json.dumps({'verified': True, 'artifacts': len(manifest['artifacts']), 'source_inputs': len(manifest['source_bindings']), 'tets': [x['point']['tet_id'] for x in report['results']], 'classification_only': True}, indent=2))

if __name__ == '__main__':
    main()
