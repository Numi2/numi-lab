#!/usr/bin/env python3
"""Optional high-precision fallback for source166003's failed 469 rows.

This is a separate reference result. It reads exact stored FP32-derived inputs,
starts from accepted history plus the elastic trial stress, and never seeds
from a native projected state. It does not rewrite the source166003 FP64 result
or convert its failures into FP64 passes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np

BASE = Path('/Users/home/cardboard-implementation-20261014')
INPUT = BASE / 'analysis/bilinear-capture-layer-audit-source166-003/validation/regression469-001'
CAPTURE = BASE / 'solver/final-newton-fd-capture-469/diagnostics/final-newton'
PREDICTION = BASE / 'solver/final-newton-fd-capture-469/prediction.json'
RECEIPT = BASE / 'solver/final-newton-fd-capture-469/receipt.json'
STDERR = BASE / 'solver/final-newton-fd-capture-469/stderr.log'
SOURCE_DIR = BASE / 'analysis/bilinear-capture-layer-audit-source166-003/candidate/matter/tools'
REFERENCE = SOURCE_DIR / 'cardboard_bilinear_fp64_reference.py'
GEOMETRY = SOURCE_DIR / 'cardboard_bilinear_capture_layers.py'
AUDIT = BASE / 'snapshots/source-right-grip-telemetry-157/matter/tools/cardboard_cached_operator_audit.py'
SOURCE_REPORT = json.loads((INPUT / 'report.json').read_text())
RESIDUAL_SCALE = np.array([1e4] * 6 + [1.0] * 6 + [1e4] * 3, dtype=np.float64)
RESIDUAL_GATE = mp.mpf('2e-12')
CORRECTION_GATE = mp.mpf('2e-11')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def exact_mpf(value):
    if isinstance(value, (float, np.floating)):
        n, d = float(value).as_integer_ratio()
    elif isinstance(value, (int, np.integer)):
        n, d = int(value), 1
    else:
        raise TypeError(type(value).__name__)
    return mp.mpf(n) / d


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def verify_bindings():
    manifest = json.loads((INPUT / 'manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        actual = sha(INPUT / name)
        if actual != expected:
            raise ValueError(f'469 regression {name} hash mismatch: {actual}')
    for raw_path, expected in manifest['inputs_sha256'].items():
        path = Path(raw_path)
        actual = sha(path)
        if actual != expected:
            raise ValueError(f'469 input hash mismatch {path}: {actual}')
    bindings = SOURCE_REPORT['bindings']
    source_root = Path(bindings['source_snapshot'])
    source_hashes = bindings['source_hashes_verified_against_prediction']
    if not isinstance(source_hashes, dict) or len(source_hashes) != 787:
        raise ValueError('source report does not contain the expected 787-file source binding')
    mismatches = []
    for rel, expected in source_hashes.items():
        path = source_root / rel.removeprefix('source/')
        if not path.is_file() or sha(path) != expected:
            mismatches.append(rel)
    if mismatches:
        raise ValueError(f'{len(mismatches)} source snapshot hash mismatches: {mismatches[:4]}')
    return {
        'input_files': {name: sha(INPUT / name) for name in manifest['files_sha256']},
        'native_files': {p.name: sha(p) for p in (CAPTURE, PREDICTION, RECEIPT, STDERR)},
        'source_snapshot': str(source_root),
        'source_file_count': len(source_hashes),
        'source_hash_map_sha256': hashlib.sha256(json.dumps(source_hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'source_metadata_sha256': sha(source_root / 'frozen-source.json'),
        'reference_sha256': sha(REFERENCE),
        'geometry_reader_sha256': sha(GEOMETRY),
    }


def model(p0, f0, old0):
    p = [exact_mpf(x) for x in p0]
    F = mp.matrix([[exact_mpf(f0[i, j]) for j in range(3)] for i in range(3)])
    old = [exact_mpf(x) for x in old0]
    I = mp.eye(3)
    C = mp.matrix([
        [p[1], p[4], p[5], 0, 0, 0],
        [p[4], p[2], p[6], 0, 0, 0],
        [p[5], p[6], p[3], 0, 0, 0],
        [0, 0, 0, 2 * p[9], 0, 0],
        [0, 0, 0, 0, 2 * p[8], 0],
        [0, 0, 0, 0, 0, 2 * p[7]],
    ]) / p[10]
    Eten = (F.T * F - I) / 2
    E = mp.matrix([Eten[0, 0], Eten[1, 1], Eten[2, 2], Eten[1, 2], Eten[0, 2], Eten[0, 1]])

    def residual(z):
        ss = z[6:12]
        dl = z[12] - old[12]
        a, b, c, d, e, v = ss
        q = mp.sqrt(p[11] * (b - c) ** 2 + p[12] * (c - a) ** 2 + p[13] * (a - b) ** 2
                    + 2 * p[14] * d * d + 2 * p[15] * e * e + 2 * p[16] * v * v + p[25] ** 2)
        flow = [p[12] * (a - c) + p[13] * (a - b), p[11] * (b - c) + p[13] * (b - a),
                p[11] * (c - b) + p[12] * (c - a), p[14] * d, p[15] * e, p[16] * v]
        flow = [x / q for x in flow]
        r = [mp.mpf(0)] * 15
        for j in range(6):
            r[j] = z[j] - old[j] - dl * flow[j]
        elastic = C * (E - mp.matrix(z[:6]))
        for j in range(6):
            r[6 + j] = ss[j] - elastic[j]
        pos = [(ss[j] + mp.sqrt(ss[j] ** 2 + p[23] ** 2)) / 2 for j in range(3)]
        wd = sum(x * x for x in pos) + p[25] ** 2
        r[13] = z[13] - old[13] - dl * pos[0] ** 2 / wd
        r[14] = z[14] - old[14] - dl * pos[1] ** 2 / wd
        y = 1 + p[17] * p[19] * p[21] * z[13] / p[10] + p[18] * p[20] * p[22] * z[14] / p[10] - q
        root = mp.sqrt(dl * dl + y * y + p[26] ** 2)
        r[12] = root - dl - y
        return r

    return p, F, old, E, C, residual


def scaled_residual(r):
    weights = [mp.mpf('1e4')] * 6 + [mp.mpf(1)] * 6 + [mp.mpf('1e4')] * 3
    return max(abs(r[i]) * weights[i] for i in range(15))


def jacobian(fun, z):
    J = mp.matrix(15, 15)
    for j in range(15):
        for i in range(15):
            def one(x):
                zz = list(z)
                zz[j] = x
                return fun(zz)[i]
            J[i, j] = mp.diff(one, z[j])
    return J


def elastic_seed(p, F, old, E, C):
    stress = C * (E - mp.matrix(old[:6]))
    return list(old[:6]) + list(stress) + list(old[12:])


def solve_one(p0, F0, old0, precision, max_iterations=48):
    mp.mp.dps = precision
    p, F, old, E, C, fun = model(p0, F0, old0)
    z = elastic_seed(p, F, old, E, C)
    history = []
    correction = mp.inf
    for iteration in range(max_iterations + 1):
        r = fun(z)
        rn = scaled_residual(r)
        J = jacobian(fun, z)
        d = list(mp.lu_solve(J, -mp.matrix(r)))
        scales = [max(abs(old[k]), abs(z[k]), p[26]) for k in range(15)]
        correction = max(abs(d[k]) / scales[k] for k in range(15))
        history.append({'iteration': iteration, 'scaled_residual': mp.nstr(rn, 28),
                        'dynamic_scaled_correction': mp.nstr(correction, 28)})
        if rn <= RESIDUAL_GATE and correction <= CORRECTION_GATE:
            return {'ok': True, 'z': z, 'residual': rn, 'correction': correction,
                    'iterations': iteration, 'history': history, 'reason': 'both unchanged strict gates met'}
        alpha = mp.mpf(1)
        accepted = False
        for backtrack in range(112):
            trial = [z[k] + alpha * d[k] for k in range(15)]
            trial_norm = scaled_residual(fun(trial))
            if mp.isfinite(trial_norm) and trial_norm < rn:
                history[-1]['accepted_alpha'] = mp.nstr(alpha, 16)
                history[-1]['backtracks'] = backtrack
                z, accepted = trial, True
                break
            alpha /= 2
        if not accepted:
            history[-1].update({'accepted_alpha': None, 'backtracks': 112})
            return {'ok': False, 'z': z, 'residual': rn, 'correction': correction,
                    'iterations': iteration, 'history': history, 'reason': 'strict residual descent stalled'}
    return {'ok': False, 'z': z, 'residual': scaled_residual(fun(z)), 'correction': correction,
            'iterations': max_iterations, 'history': history, 'reason': 'iteration ceiling'}


def mp_stress(p0, F0, z):
    p = [exact_mpf(x) for x in p0]
    F = mp.matrix([[exact_mpf(F0[i, j]) for j in range(3)] for i in range(3)])
    E = (F.T * F - mp.eye(3)) / 2
    ev = mp.matrix([E[0, 0], E[1, 1], E[2, 2], E[1, 2], E[0, 2], E[0, 1]])
    C = mp.matrix([
        [p[1], p[4], p[5], 0, 0, 0], [p[4], p[2], p[6], 0, 0, 0],
        [p[5], p[6], p[3], 0, 0, 0], [0, 0, 0, 2 * p[9], 0, 0],
        [0, 0, 0, 0, 2 * p[8], 0], [0, 0, 0, 0, 0, 2 * p[7]],
    ]) / p[10]
    sb = p[10] * (C * (ev - mp.matrix(z[:6])))
    S = mp.matrix([[sb[0], sb[5], sb[4]], [sb[5], sb[1], sb[3]], [sb[4], sb[3], sb[2]]])
    P = F * S
    return P


def mp_branch_values(p0, old0, z):
    p = [exact_mpf(x) for x in p0]
    old = [exact_mpf(x) for x in old0]
    a, b, c, d, e, v = z[6:12]
    q = mp.sqrt(p[11] * (b-c)**2 + p[12] * (c-a)**2 + p[13] * (a-b)**2
                + 2*p[14]*d*d + 2*p[15]*e*e + 2*p[16]*v*v + p[25]**2)
    dl = z[12] - old[12]
    y = 1 + p[17]*p[19]*p[21]*z[13]/p[10] + p[18]*p[20]*p[22]*z[14]/p[10] - q
    den = mp.sqrt(dl*dl + y*y + p[26]*p[26]) + dl + y
    return {'lambda_increment': dl, 'yield_residual': y, 'fb_denominator': den}


def mp_tangent(p0, F0, old0, z):
    """Differentiate the authored residual and stress by mpmath implicit function theorem."""
    p, F, old, _, _, residual = model(p0, F0, old0)
    J = jacobian(residual, z)
    result = mp.matrix(9, 9)
    for col in range(9):
        fi, fj = divmod(col, 3)

        def perturbed_F(t):
            f = mp.matrix(F)
            f[fi, fj] += t
            return f

        rhs = mp.matrix(15, 1)
        for i in range(15):
            def r_component(t):
                fp = perturbed_F(t)
                # Rebuild exactly from mp values rather than rounding the
                # perturbed deformation gradient through binary64.
                pmp = [exact_mpf(x) for x in p0]
                oldmp = [exact_mpf(x) for x in old0]
                Fp = fp
                I = mp.eye(3)
                Cm = mp.matrix([[pmp[1], pmp[4], pmp[5], 0, 0, 0], [pmp[4], pmp[2], pmp[6], 0, 0, 0],
                                [pmp[5], pmp[6], pmp[3], 0, 0, 0], [0, 0, 0, 2*pmp[9], 0, 0],
                                [0, 0, 0, 0, 2*pmp[8], 0], [0, 0, 0, 0, 0, 2*pmp[7]]]) / pmp[10]
                et = (Fp.T * Fp - I) / 2
                ev = mp.matrix([et[0,0], et[1,1], et[2,2], et[1,2], et[0,2], et[0,1]])
                def fun(zv):
                    ss = zv[6:12]; dl = zv[12] - oldmp[12]
                    a,b,c,d,e,v = ss
                    q = mp.sqrt(pmp[11]*(b-c)**2+pmp[12]*(c-a)**2+pmp[13]*(a-b)**2+2*pmp[14]*d*d+2*pmp[15]*e*e+2*pmp[16]*v*v+pmp[25]**2)
                    flow = [pmp[12]*(a-c)+pmp[13]*(a-b),pmp[11]*(b-c)+pmp[13]*(b-a),pmp[11]*(c-b)+pmp[12]*(c-a),pmp[14]*d,pmp[15]*e,pmp[16]*v]
                    flow = [x/q for x in flow]
                    rr=[mp.mpf(0)]*15
                    for k in range(6): rr[k]=zv[k]-oldmp[k]-dl*flow[k]
                    elastic=Cm*(ev-mp.matrix(zv[:6]))
                    for k in range(6): rr[6+k]=ss[k]-elastic[k]
                    pos=[(ss[k]+mp.sqrt(ss[k]**2+pmp[23]**2))/2 for k in range(3)]
                    wd=sum(x*x for x in pos)+pmp[25]**2
                    rr[13]=zv[13]-oldmp[13]-dl*pos[0]**2/wd
                    rr[14]=zv[14]-oldmp[14]-dl*pos[1]**2/wd
                    y=1+pmp[17]*pmp[19]*pmp[21]*zv[13]/pmp[10]+pmp[18]*pmp[20]*pmp[22]*zv[14]/pmp[10]-q
                    rr[12]=mp.sqrt(dl*dl+y*y+pmp[26]**2)-dl-y
                    return rr
                return fun(z)[i]
            rhs[i] = mp.diff(r_component, mp.mpf(0))
        dz = mp.lu_solve(J, -rhs)

        for out in range(9):
            oi, oj = divmod(out, 3)

            def stress_component(t):
                fp = perturbed_F(t)
                zp = [z[k] + t * dz[k] for k in range(15)]
                pp = [exact_mpf(x) for x in p0]
                et = (fp.T * fp - mp.eye(3)) / 2
                ev = mp.matrix([et[0,0], et[1,1], et[2,2], et[1,2], et[0,2], et[0,1]])
                Cm = mp.matrix([[pp[1], pp[4], pp[5], 0, 0, 0], [pp[4], pp[2], pp[6], 0, 0, 0],
                                [pp[5], pp[6], pp[3], 0, 0, 0], [0, 0, 0, 2*pp[9], 0, 0],
                                [0, 0, 0, 0, 2*pp[8], 0], [0, 0, 0, 0, 0, 2*pp[7]]]) / pp[10]
                sb = pp[10] * (Cm * (ev - mp.matrix(zp[:6])))
                Sm = mp.matrix([[sb[0], sb[5], sb[4]], [sb[5], sb[1], sb[3]], [sb[4], sb[3], sb[2]]])
                return (fp * Sm)[oi, oj]
            result[out, col] = mp.diff(stress_component, mp.mpf(0))
    return result


def metric(actual, reference):
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    b = np.asarray(reference, dtype=np.float64).reshape(-1)
    d = a - b
    denom = float(np.linalg.norm(b))
    return {'relative_l2': float(np.linalg.norm(d) / denom) if denom else None,
            'absolute_l2': float(np.linalg.norm(d)), 'reference_l2': denom,
            'max_abs_component': float(np.max(np.abs(d)))}


def mp_matrix_metric(actual, reference):
    diffs = [actual[i, j] - reference[i, j] for i in range(actual.rows) for j in range(actual.cols)]
    ref_norm = mp.sqrt(sum(reference[i, j] ** 2 for i in range(reference.rows) for j in range(reference.cols)))
    diff_norm = mp.sqrt(sum(x * x for x in diffs))
    return {'relative_l2': mp.nstr(diff_norm / ref_norm if ref_norm else mp.nan, 40),
            'absolute_l2': mp.nstr(diff_norm, 40),
            'reference_l2': mp.nstr(ref_norm, 40),
            'max_abs_component': mp.nstr(max(abs(x) for x in diffs), 40)}


def paper_matrix_fp64(ref, p, f, old, z):
    n = len(p)
    out = np.empty((n, 9, 9), dtype=np.float64)
    for col in range(9):
        h = np.zeros((n, 3, 3), dtype=np.float64)
        h.reshape(n, 9)[:, col] = 1.0
        out[:, :, col] = ref.tangent(p, f, old, z, h).reshape(n, 9)
    return out


def run(output: Path):
    if output.exists():
        raise FileExistsError(f'output already exists; preserving prior evidence: {output}')
    t0 = time.time()
    bindings = verify_bindings()
    archive = np.load(INPUT / 'arrays.npz')
    source_report = json.loads((INPUT / 'report.json').read_text())
    ref = import_module('bilinear_ref166003', REFERENCE)
    sys.modules['cardboard_bilinear_fp64_reference'] = ref
    sys.path.insert(0, str(AUDIT.parent))
    import_module('cardboard_cached_operator_audit', AUDIT)
    paper_tets = np.flatnonzero(archive['material'] != 2)
    paper_index = {int(tet): i for i, tet in enumerate(paper_tets)}
    failed_by_state = []
    for state in source_report['states']:
        failed_by_state.append({int(x['tet']): x['reason'] for x in state['projection']['failures']})
    header = None
    with CAPTURE.open('r', encoding='utf8') as f:
        header = json.loads(next(f))
    p32 = np.asarray(header['parameters_by_environment'][0][28:56], dtype=np.float32)
    p = p32.astype(np.float64)
    glue = np.asarray(header['parameters_by_environment'][0][56:59], dtype=np.float64)
    if p.shape != (28,) or len(glue) != 3:
        raise ValueError(f'expected 28 paper and 3 starch parameters, got {p.shape}/{glue.shape}')
    failed_ids = sorted({tet for state in failed_by_state for tet in state})
    if not failed_ids:
        raise ValueError('no FP64 failures in the bound report')
    # Bind all native center/side stresses directly from exact capture records.
    native_stress = [dict() for _ in range(5)]
    native_projection_metadata = [dict() for _ in range(5)]
    with CAPTURE.open('r', encoding='utf8') as f:
        next(f)
        for line in f:
            row = json.loads(line)
            tet = row.get('tet_id')
            if row.get('record') == 'tetrahedron' and row.get('environment') == 0 and tet in failed_ids:
                center = row['center']
                native_stress[0][int(tet)] = np.asarray(center['P_local_Pa'], dtype=np.float32).astype(np.float64)
                native_projection_metadata[0][int(tet)] = {
                    'projection_diagnostics_raw': center.get('projection'),
                    'flags_status_determinant_raw': center.get('flags_terminal_status_parameter_count'),
                    'hill_q_lambda_a_b_raw': center.get('hill_q_lambda_a_b'),
                }
                for si in range(1, 5):
                    side = row['sides'][si - 1]
                    native_stress[si][int(tet)] = np.asarray(side['P_local_Pa'], dtype=np.float32).astype(np.float64)
                    native_projection_metadata[si][int(tet)] = {
                        'projection_diagnostics_raw': side.get('projection_diagnostics'),
                        'flags_status_determinant_raw': side.get('flags_status_determinant'),
                        'constitutive_kind_state_count_flags_valid_raw': side.get('constitutive_kind_state_count_flags_valid'),
                        'hill_q_lambda_a_b_raw': side.get('hill_q_lambda_a_b'),
                    }
    if any(set(native_stress[si]) != set(failed_ids) for si in range(5)):
        raise ValueError('missing native captured stress for at least one failed point/state')

    report_rows = []
    mp_states = {}
    mp_tangents = {}
    mp_stresses = {}
    # Build mixed source-reference fields for all five states. Existing FP64-success
    # rows keep their original results; only exact failed rows use MP data.
    state_matrices = []
    state_stresses = []
    state_finite_counts = []
    state_names = ['center', 's1', 's2', 's3', 's4']
    F_keys = ['F_center', 's1_F', 's2_F', 's3_F', 's4_F']
    for si, (name, Fkey) in enumerate(zip(state_names, F_keys)):
        Fpaper = archive[Fkey][paper_tets]
        oldpaper = archive['accepted_paper_history']
        valid = archive[f's{si}_valid']
        z_fp64 = archive[f's{si}_reference_state']
        good_indices = np.flatnonzero(valid)
        # Keep the source166 reference matrix for center, and construct other states
        # only from source166003's own valid FP64 projected rows.
        if si == 0:
            Cfull = archive['reference_C'].copy()
        else:
            Cfull = np.full((len(archive['material']), 9, 9), np.nan, dtype=np.float64)
        if good_indices.size:
            Cfull[paper_tets[good_indices]] = paper_matrix_fp64(
                ref, np.repeat(p[None, :], len(good_indices), axis=0), Fpaper[good_indices],
                oldpaper[good_indices], z_fp64[good_indices])
        # The starch law is elastic and independent of paper projection.
        glue_ids = np.flatnonzero(archive['material'] == 2)
        Fglue = archive[Fkey][glue_ids]
        for col in range(9):
            h = np.zeros((len(glue_ids), 3, 3), dtype=np.float64)
            h.reshape(len(glue_ids), 9)[:, col] = 1.0
            Cfull[glue_ids, :, col] = ref.glue_tangent_action(
                Fglue, h, np.full(len(glue_ids), glue[1]), np.full(len(glue_ids), glue[2])).reshape(len(glue_ids), 9)
        Sfull = archive[f's{si}_reference_stress'].copy()
        # Existing stress arrays already contain the independently computed glue rows.
        for tet, reason in failed_by_state[si].items():
            row = paper_index[tet]
            p_row = p.copy()
            F_row = archive[Fkey][tet].copy()
            old_row = oldpaper[row].copy()
            precision_results = {}
            solves = {}
            for dps in (70, 90, 110):
                result = solve_one(p_row, F_row, old_row, dps)
                solves[dps] = result
                precision_results[str(dps)] = {
                    'strict_gates_pass': bool(result['ok']), 'iterations': int(result['iterations']),
                    'scaled_residual': mp.nstr(result['residual'], 36),
                    'dynamic_scaled_correction': mp.nstr(result['correction'], 36),
                    'reason': result['reason'], 'iteration_history': result['history'],
                }
            if not all(solves[d]['ok'] for d in (70, 90, 110)):
                report_rows.append({'state': name, 'tet': tet, 'original_fp64_failure': reason,
                                    'precision_results': precision_results, 'fallback_status': 'unresolved_mp'})
                continue
            mp.mp.dps = 140
            z_mp = solves[110]['z']
            zspread = {}
            for a, b in ((70, 90), (90, 110), (70, 110)):
                za, zb = solves[a]['z'], solves[b]['z']
                zspread[f'{a}_vs_{b}'] = max(
                    abs(za[k] - zb[k]) / max(abs(solves[110]['z'][k]), abs(old_row[k]), exact_mpf(p[26]))
                    for k in range(15))
            stable = max(zspread.values()) < mp.mpf('1e-12')
            zfloat = np.asarray([float(x) for x in z_mp], dtype=np.float64)
            zold = old_row
            rounded_state_error = max(
                abs(exact_mpf(zfloat[k]) - z_mp[k]) /
                max(abs(old_row[k]), abs(z_mp[k]), exact_mpf(p[26])) for k in range(15))
            # Final strict gates recomputed at 140 digits, independent of line-search status.
            pmp, Fmp, oldmp, _, _, fun = model(p_row, F_row, old_row)
            rr = fun(z_mp)
            JJ = jacobian(fun, z_mp)
            correction = list(mp.lu_solve(JJ, -mp.matrix(rr)))
            scales = [max(abs(oldmp[k]), abs(z_mp[k]), pmp[26]) for k in range(15)]
            final_r = scaled_residual(rr)
            final_c = max(abs(correction[k]) / scales[k] for k in range(15))
            strict_ok = final_r <= RESIDUAL_GATE and final_c <= CORRECTION_GATE and stable
            mp.mp.dps = 105
            Cmp90 = mp_tangent(p_row, F_row, old_row, solves[90]['z'])
            Cmp90_mp = Cmp90.copy()
            Cmp9064 = np.asarray([[float(Cmp90[i, j]) for j in range(9)] for i in range(9)], dtype=np.float64)
            mp.mp.dps = 140
            Cmp = mp_tangent(p_row, F_row, old_row, z_mp)
            Cmp64 = np.asarray([[float(Cmp[i, j]) for j in range(9)] for i in range(9)], dtype=np.float64)
            z64batch = zfloat[None, :]
            Cfp64 = paper_matrix_fp64(ref, p_row[None, :], F_row[None, :, :], old_row[None, :], z64batch)[0]
            stress_mp_mat = mp_stress(p_row, F_row, z_mp)
            Smp = np.asarray([[float(stress_mp_mat[i, j]) for j in range(3)] for i in range(3)], dtype=np.float64)
            Sfp64 = ref.stress(p_row[None, :], F_row[None, :, :], z64batch)[0]
            Cfull[tet] = Cmp64
            Sfull[tet] = Smp
            key = f'{si}:{tet}'
            mp_states[key] = zfloat
            mp_tangents[key] = Cmp64
            mp_stresses[key] = Smp
            branch = mp_branch_values(p_row, old_row, z_mp)
            native_state = archive[f's{si}_native_state'][row]
            native_stress_at_native_state = ref.stress(
                p_row[None, :], F_row[None, :, :], native_state[None, :])[0]
            native_branch = archive[f's{si}_native_branch_values'][row]
            mp_class = [branch['lambda_increment'] > 0, branch['yield_residual'] > 0,
                        branch['fb_denominator'] > mp.mpf('1e-14')]
            native_class = [bool(native_branch[0] > 0), bool(native_branch[1] > 0),
                            bool(native_branch[2] > 1e-14)]
            fp64_mp_tangent = metric(Cmp64, Cfp64)
            native_mp_tangent = metric(archive['native_C'][tet], Cmp64) if si == 0 else None
            report_rows.append({
                'state': name, 'state_index': si, 'tet': int(tet), 'paper_row': int(row),
                'original_fp64_failure': reason,
                'fallback_status': 'mp_gates_pass' if strict_ok else 'mp_root_or_stability_gate_failed',
                'precision_results': precision_results,
                'mp110_final_scaled_residual': mp.nstr(final_r, 36),
                'mp110_final_dynamic_correction': mp.nstr(final_c, 36),
                'mp_root_relative_spread_70_90_110': {k: mp.nstr(v, 32) for k, v in zspread.items()},
                'mp_root_stability_gate_1e-12': bool(stable),
                'mp_root_float64_rounding_max_dynamic_scaled_component': mp.nstr(rounded_state_error, 32),
                'mp_ift_90_vs_110_same_equations': mp_matrix_metric(Cmp90_mp, Cmp),
                'mp_ift_vs_fp64_ift_at_same_rounded_mp_root': {**fp64_mp_tangent,
                    'relative_l2_le_1e-6': bool(fp64_mp_tangent['relative_l2'] is not None and fp64_mp_tangent['relative_l2'] <= 1e-6)},
                'native_captured_state_vs_mp_root': metric(native_state, zfloat),
                'native_state_source166_scaled_residual': float(archive[f's{si}_native_residual'][row]),
                'native_projection_metadata_raw': native_projection_metadata[si][tet],
                'native_center_tangent_vs_mp_ift': ({**native_mp_tangent,
                    'relative_l2_le_1e-3': bool(native_mp_tangent['relative_l2'] is not None and native_mp_tangent['relative_l2'] <= 1e-3)} if si == 0 else None),
                'native_captured_stress_vs_mp_stress': metric(native_stress[si][tet], Smp),
                'native_captured_stress_vs_fp64_stress_at_native_state': metric(native_stress[si][tet], native_stress_at_native_state),
                'fp64_stress_at_rounded_mp_root_vs_mp_stress': metric(Sfp64, Smp),
                'mp_branch_values': {k: mp.nstr(v, 36) for k, v in branch.items()},
                'native_branch_values': [float(x) for x in native_branch],
                'independent_branch_predicates': {'lambda_positive': bool(mp_class[0]), 'yield_residual_positive': bool(mp_class[1]),
                    'fb_denominator_above_1e-14': bool(mp_class[2]),
                    'native_capture_predicates_match': bool(native_class == mp_class)},
                'interpretation': 'MP-only fallback reference for a row that failed the original source166003 FP64 projection. This does not turn the FP64 row into a pass or qualify native assembly.'
            })
        state_matrices.append(Cfull)
        state_stresses.append(Sfull)
        state_finite_counts.append({
            'state': name, 'material_tangent_all_1920_finite': bool(np.isfinite(Cfull).all()),
            'finite_tangent_rows': int(np.isfinite(Cfull).all(axis=(1, 2)).sum()),
            'material_stress_all_1920_finite': bool(np.isfinite(Sfull).all()),
            'finite_stress_rows': int(np.isfinite(Sfull).all(axis=(1, 2)).sum()),
            'used_mp_rows': int(len(failed_by_state[si])),
            'remaining_nonfinite_tangent_rows': np.flatnonzero(~np.isfinite(Cfull).all(axis=(1,2))).astype(int).tolist(),
            'remaining_nonfinite_stress_rows': np.flatnonzero(~np.isfinite(Sfull).all(axis=(1,2))).astype(int).tolist(),
        })

    output.mkdir(parents=True)
    fallback_npz = output / 'mixed_reference_arrays.npz'
    np.savez_compressed(fallback_npz,
                        **{f'{name}_tangent': state_matrices[i] for i, name in enumerate(state_names)},
                        **{f'{name}_stress': state_stresses[i] for i, name in enumerate(state_names)},
                        **{f'mp_state_{k.replace(":", "_")}': v for k, v in mp_states.items()},
                        **{f'mp_tangent_{k.replace(":", "_")}': v for k, v in mp_tangents.items()},
                        **{f'mp_stress_{k.replace(":", "_")}': v for k, v in mp_stresses.items()})
    axes = {
        'native_projection': False, 'complete_native_tangent': False, 'full_assembled_action': False,
        'all_states_fp64_projection': False, 'finite_mixed_precision_reference_tangent_rows': all(x['material_tangent_all_1920_finite'] for x in state_finite_counts),
        'physical_or_calibration_qualification': False,
    }
    resolved_rows = [x for x in report_rows if x.get('fallback_status') == 'mp_gates_pass']
    center_rows = [x for x in resolved_rows if x['state_index'] == 0]
    max_fp64_mp_ift = max(float(x['mp_ift_vs_fp64_ift_at_same_rounded_mp_root']['relative_l2']) for x in resolved_rows)
    max_mp_90_110 = max(float(x['mp_ift_90_vs_110_same_equations']['relative_l2']) for x in resolved_rows)
    max_native_tangent = max(float(x['native_center_tangent_vs_mp_ift']['relative_l2']) for x in center_rows)
    max_native_stress_by_state = {name: max(float(x['native_captured_stress_vs_mp_stress']['relative_l2'])
        for x in resolved_rows if x['state'] == name) for name in state_names}
    max_native_vs_fp64_stress_at_native_state_by_state = {name: max(
        float(x['native_captured_stress_vs_fp64_stress_at_native_state']['relative_l2'])
        for x in resolved_rows if x['state'] == name) for name in state_names}
    branch_mismatches = [x for x in resolved_rows if not x['independent_branch_predicates']['native_capture_predicates_match']]
    report = {
        'schema': 'numi.cardboard.native469-explicit-mp-fallback.v1',
        'classification': 'optional MP-only reference completion for exact source166003 FP64-failed rows; not FP64 qualification, native projection, native tangent qualification, or physical validation',
        'bindings': bindings,
        'environment': {'python': platform.python_version(), 'mpmath': mp.__version__, 'numpy': np.__version__},
        'input_regression_manifest_sha256': sha(INPUT / 'manifest.json'),
        'source166003_manifest_sha256': sha(BASE / 'analysis/bilinear-capture-layer-audit-source166-003/manifest.json'),
        'source166003_reference_fix_sha256': sha(REFERENCE),
        'strict_gates': {'scaled_residual_max': '2e-12', 'dynamic_scaled_correction_max': '2e-11',
                         'mp_precision_root_spread_max': '1e-12', 'unchanged_from_native469-reader': True},
        'method': {'equations': 'authored unsimplified 15-state Hill/tensile-hardening return map with Fischer-Burmeister residual; mpmath arbitrary precision',
                   'seed': 'accepted history plus exact captured-F elastic trial stress',
                   'native_projected_state_used_as_seed': False,
                   'precision_dps': [70, 90, 110],
                   'line_search': 'strict decrease of original scaled residual, 112 halvings maximum',
                   'tangent': 'implicit function theorem differentiated at MP root with mpmath; compared against corrected FP64 IFT at the same rounded MP root'},
        'original_fp64_failures_preserved': [
            {'state': state_names[i], 'tet': int(tet), 'reason': reason}
            for i, failures in enumerate(failed_by_state) for tet, reason in sorted(failures.items())
        ],
        'mp_results': report_rows,
        'summary_metrics': {
            'failed_rows_fallback_count': len(report_rows),
            'strict_mp_gate_pass_count': len(resolved_rows),
            'max_root_spread_70_90_110_dynamic_scaled': max(float(max(x['mp_root_relative_spread_70_90_110'].values(), key=float)) for x in resolved_rows),
            'max_mp_ift_90_vs_110_relative_l2': max_mp_90_110,
            'max_fp64_ift_vs_mp_ift_relative_l2_at_same_rounded_mp_root': max_fp64_mp_ift,
            'max_center_native_tangent_vs_mp_ift_relative_l2': max_native_tangent,
            'max_native_captured_stress_vs_mp_relative_l2_by_state': max_native_stress_by_state,
            'max_native_captured_stress_vs_fp64_stress_at_native_state_relative_l2_by_state': max_native_vs_fp64_stress_at_native_state_by_state,
            'native_vs_mp_branch_predicate_mismatch_count': len(branch_mismatches),
            'native_vs_mp_branch_predicate_mismatches': [{'state': x['state'], 'tet': x['tet'],
                'native': x['native_branch_values'], 'mp': x['mp_branch_values']} for x in branch_mismatches],
        },
        'mixed_reference_finiteness': state_finite_counts,
        'qualification_axes': axes,
        'limits': [
            'MP roots are a separate high-precision reference result; original FP64 failure records remain unchanged.',
            'The output arrays mix source166003 FP64-success rows with explicit MP IFT rows for FP64 failures; this is not an all-FP64 tangent assembly.',
            'Native cached tangent comparisons are available only at the captured center state; side-state comparisons are independent CPU IFT comparisons, not native data.',
            'No finite-difference noise-floor or assembled element/global action claim is made by finite local arrays alone.',
            'All qualification axes remain false.'
        ],
        'mixed_reference_arrays_sha256': sha(fallback_npz),
        'elapsed_seconds': time.time() - t0,
        'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    r = run(args.output)
    print(json.dumps({'output': str(args.output), 'rows': len(r['mp_results']),
                      'fallback_mp_passes': sum(x.get('fallback_status') == 'mp_gates_pass' for x in r['mp_results']),
                      'state_finiteness': r['mixed_reference_finiteness'],
                      'elapsed_seconds': r['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
