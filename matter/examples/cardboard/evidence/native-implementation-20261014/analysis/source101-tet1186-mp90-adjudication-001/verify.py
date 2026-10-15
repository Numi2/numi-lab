#!/usr/bin/env python3
"""Fail-closed integrity and metric checks for the tet1186 MP90 package."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'manifest.json'

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def need(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit('FAIL: ' + message)

def close(actual: float, expected: float, rel: float = 2e-12) -> bool:
    return math.isfinite(actual) and abs(actual-expected) <= rel * max(abs(expected), 1e-300)

def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    need(manifest['schema'] == 'numi.cardboard.source101-tet1186-mp90-adjudication.v1', 'manifest schema')
    for name, item in manifest['artifacts'].items():
        p = HERE / item['path']
        need(p.is_file(), f'missing artifact {name}: {p}')
        need(sha(p) == item['sha256'], f'artifact hash mismatch: {name}')
    for name, item in manifest['inputs'].items():
        p = Path(item['path'])
        need(p.exists(), f'missing bound input {name}: {p}')
        need(sha(p) == item['sha256'], f'input hash mismatch: {name}')

    result = json.loads((HERE / 'output-002/result.json').read_text())
    detail = json.loads((HERE / 'output-003/direction-roundoff.json').read_text())
    need(result['status'] == 'completed', 'MP90 root replay did not complete')
    need(result['capture_identity'] == {'environment': 0, 'material_index': 0, 'newton_iteration': 23,
        'root': 2, 'tet_id': 1186, 'world_fingerprint_hex': '6de3475316b7e8b9'}, 'capture identity')
    need(result['precision']['mpmath_dps'] == 90 and result['precision']['max_newton_iterations'] == 48, 'precision/cap')
    need(len(result['results']) == 2, 'expected two perturbation cases')
    need(detail['accepted_history_fixed'] is True and detail['mpmath_dps'] == 90, 'history/precision contract')
    need(len(detail['results']) == 2, 'direction decomposition cases')

    expected = {
        1e-7: {'nominal': 9.60242126761476e-7, 'realized': 5.888653981152615e-12,
               'actual_vs_nominal': 9.602436994924189e-7, 'v3_fd_vs_mp90': 2.5607747993148527e-7},
        3e-8: {'nominal': 1.5857681873279995e-6, 'realized': 5.299790306224242e-13,
               'actual_vs_nominal': 1.585768323540667e-6, 'v3_fd_vs_mp90': 5.245402001801941e-7},
    }
    by_target = {float(x['target_deltaF_frobenius']): x for x in detail['results']}
    need(set(by_target) == set(expected), 'target set')
    for target, exp in expected.items():
        row = by_target[target]
        need(row['projection_converged_center_plus_minus'] is True, f'projection status {target}')
        need(close(float(row['mp90_fd_vs_mp90_tangent_nominal']), exp['nominal']), f'nominal metric {target}')
        need(close(float(row['mp90_fd_vs_mp90_tangent_realized']), exp['realized']), f'realized metric {target}')
        need(close(float(row['tangent_action_actual_realized_vs_nominal']['relative_l2']), exp['actual_vs_nominal']), f'direction action delta {target}')
        need(close(float(row['fp64_central_fd_vs_mp90_fd']['relative_l2']), exp['v3_fd_vs_mp90']), f'endpoint FD delta {target}')
    need(manifest['verdict']['uniform_v3_gate_changed'] is False, 'must not change uniform gate')
    need(manifest['verdict']['universal_noise_floor_claimed'] is False, 'must not claim universal noise floor')
    print('PASS: MP90 package bindings and point-specific assertions verified; uniform v3 gate remains unchanged.')

if __name__ == '__main__':
    main()
