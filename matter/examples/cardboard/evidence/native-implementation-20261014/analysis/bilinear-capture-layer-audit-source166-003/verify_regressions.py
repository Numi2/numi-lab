#!/usr/bin/env python3
"""Hash and schema-check the source166-003 419/469 full-reader regressions."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

BASE = Path('/Users/home/cardboard-implementation-20261014')
ROOT = BASE / 'analysis/bilinear-capture-layer-audit-source166-003'
READER = ROOT / 'candidate/matter/tools/cardboard_bilinear_capture_layers.py'
REFERENCE = ROOT / 'candidate/matter/tools/cardboard_bilinear_fp64_reference.py'
GEOMETRY = BASE / 'snapshots/source-right-grip-telemetry-157/matter/tools/cardboard_cached_operator_audit.py'
MP_REPORT = BASE / 'analysis/native469-independent-projection-002/run-sealed002/report.json'
MP_STATES = BASE / 'analysis/native469-independent-projection-002/run-sealed002/mp_states.npz'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_one(label: str, case_rel: str, output_rel: str) -> dict:
    case = BASE / case_rel
    output = ROOT / output_rel
    report_path = output / 'report.json'
    arrays_path = output / 'arrays.npz'
    manifest_path = output / 'manifest.json'
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if report['tools_sha256']['cardboard_bilinear_capture_layers.py'] != sha(READER):
        raise SystemExit(f'{label}: reader source hash mismatch')
    if report['tools_sha256']['cardboard_bilinear_fp64_reference.py'] != sha(REFERENCE):
        raise SystemExit(f'{label}: reference source hash mismatch')
    if report['tools_sha256']['cardboard_cached_operator_audit.py'] != sha(GEOMETRY):
        raise SystemExit(f'{label}: geometry source hash mismatch')
    for rel, expected in manifest['files_sha256'].items():
        path = output / rel
        if not path.is_file() or sha(path) != expected:
            raise SystemExit(f'{label}: output file hash mismatch: {path}')
    for raw_path, expected in manifest['inputs_sha256'].items():
        path = Path(raw_path)
        if not path.is_file() or sha(path) != expected:
            raise SystemExit(f'{label}: input file hash mismatch: {path}')
    arrays = np.load(arrays_path)
    if arrays['reference_C'].shape != (1920, 9, 9) or arrays['native_C'].shape != (1920, 9, 9):
        raise SystemExit(f'{label}: 9x9 tangent matrix shape mismatch')
    center = report['states'][0]
    matrix = center['cached_material_matrix_comparison_Pa']
    if matrix['rows'] != 1920 * 9:
        raise SystemExit(f'{label}: reader did not compare all nine tangent columns')
    if len(report['states']) != 5:
        raise SystemExit(f'{label}: full five-state reader output missing')
    if any(report['six_qualification_axes'].values()):
        raise SystemExit(f'{label}: this diagnostic reader must not promote qualification axes')
    return {
        'case': str(case), 'output': str(output),
        'scope': report['scope'],
        'capture_sha256': report['bindings']['capture_sha256'],
        'prediction_sha256': report['bindings']['prediction_sha256'],
        'receipt_sha256': report['bindings']['receipt_sha256'],
        'output_manifest_sha256': sha(manifest_path),
        'report_sha256': sha(report_path), 'arrays_sha256': sha(arrays_path),
        'five_state_projection': [
            {'state': s['state_index'], 'valid': s['projection']['valid_rows'],
             'rows': s['projection']['rows'], 'failure_count': len(s['projection']['failures']),
             'failures': s['projection']['failures'],
             'max_scaled_residual': s['projection']['max_scaled_residual'],
             'max_dynamic_correction': s['projection']['max_dynamic_correction']}
            for s in report['states']
        ],
        'center_tangent_columns': {
            'matrix_shape': list(arrays['reference_C'].shape),
            'compared_rows': matrix['rows'], 'finite_rows': matrix['finite_rows'],
            'max_valid_row_relative_l2': matrix['max_relative_l2'],
            'relative_gate': matrix['relative_gate'], 'all_rows_at_gate': matrix['all_rows_at_relative_gate'],
        },
        'fp64_sweep': report['fp64_directional_sweep'],
        'qualification_axes': report['six_qualification_axes'],
    }


results = {
    'schema': 'numi.cardboard.source166003-full-reader-regression-audit.v1',
    'classification': 'CPU full-reader regressions of an independent reference Jacobian fix; no native execution or physical qualification',
    'source166_003': {
        'reader_sha256': sha(READER), 'reference_sha256': sha(REFERENCE),
        'base_source166_002_reference_sha256': sha(BASE / 'analysis/bilinear-capture-layer-audit-source166-002/candidate/matter/tools/cardboard_bilinear_fp64_reference.py'),
        'patch_scope': 'only the independent reference Jacobian FB output row and a new MP regression test; rationalized residual value and strict gates unchanged',
    },
    'mp_witness': {'report': str(MP_REPORT), 'report_sha256': sha(MP_REPORT),
                   'states': str(MP_STATES), 'states_sha256': sha(MP_STATES)},
    'runs': [
        verify_one('native419', 'solver/two-stage-first-failure-rich-fd-419', 'validation/regression419-005'),
        verify_one('native469', 'solver/final-newton-fd-capture-469', 'validation/regression469-001'),
    ],
    'tet846_mp_state_comparison': None,
    'global_interpretation': 'The analytic row resolves the independent-reference tet846 FB derivative artifact. Remaining 469 strict residual misses stay inconclusive; 419 side-state misses remain visible. All qualification axes remain false.',
}

with np.load(ROOT / 'validation/regression469-001/arrays.npz') as arrays:
    paper_ids = np.flatnonzero(arrays['material'] != 2)
    row846 = int(np.flatnonzero(paper_ids == 846)[0])
    with np.load(MP_STATES) as mp_states:
        d = arrays['s0_reference_state'][row846] - mp_states['tet_846']
    results['tet846_mp_state_comparison'] = {
        'global_tet_id': 846, 'paper_row': row846,
        'l2_absolute': float(np.linalg.norm(d)), 'max_abs_component': float(np.max(np.abs(d))),
        'interpretation': 'The FP64 reader passes this point after the analytic FB-row fix; it is compared to the independent 110-digit local MP root.'
    }

(ROOT / 'validation/summary.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
print(json.dumps({'summary': str(ROOT / 'validation/summary.json'),
                  '419_center_valid': results['runs'][0]['five_state_projection'][0]['valid'],
                  '469_center_valid': results['runs'][1]['five_state_projection'][0]['valid'],
                  'tet846_mp_state_l2': results['tet846_mp_state_comparison']['l2_absolute']}, sort_keys=True))
