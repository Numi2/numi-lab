"""Tighten candidate-seeded FP64 projections for the captured local tangents.

This is offline analysis only. It imports the independent handwritten equations
and the earlier candidate-seeded replay routine; it never calls Metal/compiler
code. Inputs and prior tol=2e-14 outputs are hash-bound in report.json.
"""
from pathlib import Path
import sys, json, hashlib, time
import numpy as np

BASE = Path('/Users/home/cardboard-implementation-20261014')
ROOT = BASE / 'analysis/bilinear15-fp64-replay-267'
DIAG = ROOT / 'diagnosis-001'
sys.path.insert(0, str(ROOT))
import bilinear15_reference as ref
from importlib.machinery import SourceFileLoader
reproj = SourceFileLoader('captured_seed_reprojection', str(DIAG / 'captured_seed_reprojection.py')).load_module()

CAPTURES = {
    'N0': BASE / 'solver/sourcefit15-fg32-control1-capture-cycle-258/operator-capture',
    'N63': BASE / 'solver/sourcefit15-fg32-control1-newton63-capture-266/operator-capture',
}
PRIOR = {
    'N0': ROOT / 'early-output-002',
    'N63': ROOT / 'late-output-002',
}
TARGET_TETS = {'N0': [750, 751, 758, 763, 857], 'N63': [857, 1698]}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def as_words(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)

def capture_pair(record, key):
    high = as_words(record[key])
    low = as_words(record.get(key + '_low', np.zeros_like(record[key])))
    return high + low

def yield_gap(p, z):
    q, _ = ref.hill(p, z[6:12])
    return float(1 + p[17]*p[19]*p[21]*z[13]/p[10]
                   + p[18]*p[20]*p[22]*z[14]/p[10] - q)

def run_one(label, capture_path, prior_dir):
    records = [json.loads(s) for s in capture_path.read_text().splitlines() if s.strip()]
    header = records[0]
    assert header['schema'] == 'numi.cardboard.fem-newton-capture.v1'
    assert records[-1]['record'] == 'integrity' and records[-1]['production_inputs_bitwise_unchanged']
    params = as_words(header['parameters_by_environment'])
    tets = [r for r in records if r.get('record') == 'tetrahedron' and r.get('material_index') in (0, 1)]
    prior = json.loads((prior_dir / 'report.json').read_text())
    with np.load(prior_dir / 'rows.npz') as z:
        old14 = z['fp64_refined'].astype(np.float64)
        old_native = z['native'].astype(np.float64)
    assert len(tets) == len(old14) == len(old_native) == 3456
    root15, native, valid_indices = [], [], []
    selected = []
    failed = []
    start = time.monotonic()
    selected_ids = set(TARGET_TETS[label])
    for i, t in enumerate(tets):
        c = t['center']; env = int(t['environment']); mi = int(t['material_index'])
        p = params[env, 28*mi:28*(mi+1)]
        old = as_words(c['accepted_state'])
        f = capture_pair(c, 'F_local')
        candidate = capture_pair(c, 'projected_state')
        direction = as_words(c['F_direction_local_per_s'])
        z, solve = reproj.project_from_candidate(p, f, old, candidate, 2e-15, maxiter=80)
        if solve['status'] != 'converged':
            failed.append({'array_index': i, 'tet_id': int(t['tet_id']), 'status': solve['status'],
                           'iterations': solve.get('iterations'), 'residual': solve.get('residual')})
            continue
        tangent = ref.tangent(p, f, old, z, direction)
        root15.append(tangent); native.append(as_words(c['tangent_local_Pa_per_s'])); valid_indices.append(i)
        if int(t['tet_id']) not in selected_ids:
            continue
        diag = c.get('projection', c.get('projection_diagnostics'))
        cap_norm, _ = reproj.norm(p, f, old, candidate)
        selected.append({
            'array_index': i, 'tet_id': int(t['tet_id']), 'environment': env, 'material_index': mi,
            'native_projection_diagnostics': diag,
            'independent_candidate_scaled_residual': cap_norm,
            'candidate_residual_to_shader_report_ratio': cap_norm / diag[0] if diag and diag[0] else None,
            'candidate_yield_gap': yield_gap(p, candidate),
            'strict_root_yield_gap': yield_gap(p, z),
            'fb_epsilon': float(p[26]),
            'candidate_delta_multiplier': float(candidate[12] - old[12]),
            'strict_root_delta_multiplier': float(z[12] - old[12]),
            'candidate_fb_residual_unscaled': float(ref.residual(p, f, old, candidate)[12]),
            'candidate_to_strict_root_state_linf': float(np.max(np.abs(candidate-z))),
            'candidate_to_strict_root_tangent_rel_l2': reproj.rel_l2(tangent, ref.tangent(p, f, old, candidate, direction)),
            'tol14_to_tol15_tangent_rel_l2': reproj.rel_l2(tangent, old14[i]),
            'strict_root_residual': solve['residual'],
            'strict_root_iterations': solve['iterations'],
            'strict_root_state': z.tolist(),
        })
    roots = np.asarray(root15); natives = np.asarray(native)
    diffs = roots - natives
    old14 = old14[np.asarray(valid_indices, dtype=np.int64)]
    return {
        'capture': str(capture_path), 'capture_sha256': sha(capture_path),
        'prior_tol14_report_sha256': sha(prior_dir / 'report.json'),
        'prior_tol14_rows_sha256': sha(prior_dir / 'rows.npz'),
        'source_root': header['root'], 'newton_iteration': header['newton_iteration'],
        'records': len(tets), 'tol15_converged': len(roots), 'tol15_failed': failed,
        'strict_tol15_runtime_seconds': time.monotonic()-start,
        'native_vs_tol15_global_l2_relative': float(np.linalg.norm(diffs)/np.linalg.norm(roots)),
        'tol14_vs_tol15_global_l2_relative': float(np.linalg.norm(roots-old14)/np.linalg.norm(roots)),
        'max_tet_native_vs_tol15_relative_l2': float(max(reproj.rel_l2(roots[i], natives[i]) for i in range(len(roots)))),
        'max_tet_tol14_vs_tol15_relative_l2': float(max(reproj.rel_l2(roots[i], old14[i]) for i in range(len(roots)))),
        'selected_tets': selected,
    }

if __name__ == '__main__':
    started = time.monotonic()
    results = {label: run_one(label, path, PRIOR[label]) for label, path in CAPTURES.items()}
    source = BASE / 'snapshots/source-stateless-compensated-and-validation-admission-077'
    report = {
        'schema': 'numi.cardboard.bilinear15-root-stability.v1',
        'method': 'Independent handwritten FP64 equations; captured candidate seed; same unweighted authored-row L-infinity Armijo merit as production; strict stopping tolerance 2e-15*(1+state infinity norm). Native tangent is read from exact captured operator records.',
        'bindings': {
            'source_snapshot': str(source),
            'liner_asset_sha256': sha(source/'matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter'),
            'medium_asset_sha256': sha(source/'matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter'),
            'common_metalinc_sha256': sha(source/'matter/src/metal/common.metalinc'),
            'reference_equations_sha256': sha(ROOT/'bilinear15_reference.py'),
            'candidate_seed_solver_sha256': sha(DIAG/'captured_seed_reprojection.py'),
            'analysis_script_sha256': sha(__file__),
        },
        'captures': results,
        'limits': [
            'Offline local return-map replay only; no production source edits or GPU rerun.',
            'The strict-root tangent comparison diagnoses local projection error at the captured Newton linearization, not the full global solve.',
            'No constitutive calibration, complete coupon, corrugated box, or physical qualification claim.',
            'A candidate-seeded solve is used because the older elastic-seed reference merit path stalls for selected late-yield rows; this is a reference-path issue, not evidence that the native local solve failed.',
        ],
        'wall_seconds': time.monotonic()-started,
    }
    out = Path(__file__).parent / 'report.json'
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
