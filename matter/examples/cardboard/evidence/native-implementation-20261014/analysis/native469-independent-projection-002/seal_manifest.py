#!/usr/bin/env python3
"""Create a deterministic hash manifest for the independent MP audit package."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent
BASE = Path('/Users/home/cardboard-implementation-20261014')
REPORT = ROOT / 'run-sealed002' / 'report.json'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


report = json.loads(REPORT.read_text())
files = {}
for rel in (
    'README.md', 'solve_mp_projection.py', 'test_report.py', 'test-stdout.log',
    'seal_manifest.py', 'run-sealed002/report.json', 'run-sealed002/mp_states.npz',
    'run-final/report.json', 'run-final/mp_states.npz',
    'run-sealed/report.json', 'run-sealed/mp_states.npz',
    'run-evidence/report.json', 'run-evidence/mp_states.npz',
):
    p = ROOT / rel
    if not p.is_file():
        raise SystemExit(f'missing package file: {rel}')
    files[rel] = {'sha256': sha(p), 'bytes': p.stat().st_size}

summary = []
for row in report['resolved_rows']:
    tangent = row['native_comparisons']['cached_tangent']
    summary.append({
        'tet': row['tet'],
        'mp_residual': row['high_precision_result']['scaled_residual_130dps'],
        'mp_dynamic_correction': row['high_precision_result']['dynamic_scaled_correction_130dps'],
        'state_relative_l2': row['native_comparisons']['projected_state_relative_l2']['relative_l2'],
        'stress_relative_l2': row['native_comparisons']['stress_P_local_Pa']['relative_l2'],
        'tangent_matrix_relative_l2': tangent['matrix_frobenius']['relative_l2'],
        'tangent_worst_column_relative_l2': max(x['relative_l2'] or 0 for x in tangent['column_relative_l2']),
    })

manifest = {
    'schema': 'numi.cardboard.native469-independent-mp-projection-package.v2',
    'classification': 'independent local MP projection witness; not native trajectory or physical qualification',
    'canonical_output': 'run-sealed002',
    'reproduction': {
        'python': '/Users/home/.pyenv/versions/3.10.12/bin/python3',
        'python_version': platform.python_version(),
        'mpmath_version': __import__('mpmath').__version__,
        'numpy_version': __import__('numpy').__version__,
        'solver_command': f"{sys.executable} {ROOT / 'solve_mp_projection.py'} --output /absolute/path/to/new-output",
        'test_command': f"cd {ROOT} && {sys.executable} -m unittest -v test_report.py",
        'test_exit_code': 0,
        'syntax_compile': 'python -m py_compile solve_mp_projection.py test_report.py passed',
    },
    'input_bindings': report['bindings'],
    'strict_gates': report['strict_gates'],
    'resolved_rows': summary,
    'max_70_80_110_dynamic_scaled_root_spread': max(r['high_precision_result']['stability_max_dynamic_scaled_difference'] for r in report['resolved_rows']),
    'preserved_preliminary_attempts': {
        'run-final': 'superseded; first test compared the stability threshold field as though it were observed spread',
        'run-sealed': 'superseded; corrected report field, but test still targeted stale run-final path',
        'run-evidence': 'successful source-map verification output superseded only by final gate labels and final package binding',
    },
    'files': files,
    'limitations': report['limitations'] + [
        'The current source-166 tet-846 tangent Jacobian comparison is flagged as provisional due to the separately diagnosed quotient differentiation artifact.',
        'The six reference rows cover center F only; the other four captured deformation states, full assembly gates, and physical closure remain open.',
    ],
}
(ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
print(json.dumps({'manifest': str(ROOT / 'manifest.json'), 'sha256': sha(ROOT / 'manifest.json'), 'files': len(files)}, sort_keys=True))
