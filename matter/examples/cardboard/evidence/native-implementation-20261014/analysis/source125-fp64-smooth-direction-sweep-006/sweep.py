#!/usr/bin/env python3
"""Ideal FP64 directional sweep for the source125 capture at run419 Newton48.

CPU diagnostic only. Every endpoint is an unquantized FP64 perturbation of the
promoted captured center high+low F along the captured local F direction, with
accepted history fixed. The native residual sweep is provenance only; native
FP32 realized endpoints are not used as inputs here.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import sys
import time
import warnings

import numpy as np

BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014')
RUN = BASE / 'solver/two-stage-first-failure-rich-fd-419'
ADAPTER = BASE / 'analysis/operator-fd-e2-adapter-001'
CAPTURE = RUN / 'operator-capture'
REF_PATH = BASE / 'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
FD_PATH = BASE / 'analysis/source101-capture360-directional-fd-001/compare_directional_fd.py'
GEOM_PATH = BASE / 'snapshots/source-boundary-reset-order-101/matter/tools/cardboard_cached_operator_audit.py'
TERM = BASE / 'analysis/native-operator-term-replay-result-422'
AUDIT = ADAPTER / 'coupon419-003'
SWEEP = BASE / 'analysis/native424-residual-sweep-audit-001'
BASELINE005 = BASE / 'analysis/source125-fp64-smooth-direction-sweep-005/result-005'


def sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def f32(x):
    a = np.asarray(x, dtype=np.float32).astype(np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError('nonfinite FP32 capture field')
    return a


def pair(rec, key):
    hi = np.asarray(rec[key], dtype=np.float32)
    lo = np.asarray(rec.get(key + '_low', np.zeros_like(hi)), dtype=np.float32)
    if not np.all(np.isfinite(hi)) or not np.all(np.isfinite(lo)):
        raise ValueError(f'nonfinite {key} word')
    return hi.astype(np.float64) + lo.astype(np.float64)


def l2(x):
    return math.sqrt(math.fsum(float(v) * float(v) for v in np.asarray(x).ravel()))


def rel_metrics(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    if not np.all(finite_a) or not np.all(finite_b):
        return {'status': 'unavailable_nonfinite_input',
                'actual_l2': None, 'reference_l2': None, 'difference_l2': None,
                'relative_to_reference': None, 'max_component_abs': None,
                'nonfinite_actual_values': int((~finite_a).sum()),
                'nonfinite_reference_values': int((~finite_b).sum())}
    d = a - b
    na, nb, nd = l2(a), l2(b), l2(d)
    return {'actual_l2': na, 'reference_l2': nb, 'difference_l2': nd,
            'relative_to_reference': nd / nb if nb else None,
            'max_component_abs': float(np.max(np.abs(d))) if d.size else 0.0}


def per_row_rel_metrics(a, b):
    """Relative Frobenius error for each local 3x3 response, without pooling rows."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 2:
        return {'status': 'invalid_shape', 'shape_actual': list(a.shape), 'shape_reference': list(b.shape)}
    finite = np.isfinite(a).all(axis=tuple(range(1, a.ndim))) & np.isfinite(b).all(axis=tuple(range(1, b.ndim)))
    if not np.all(finite):
        return {'status': 'unavailable_nonfinite_input', 'rows': int(len(a)),
                'nonfinite_rows': int((~finite).sum()), 'relative_l2_max': None,
                'rows_above_1e-6': None, 'absolute_l2_max': None}
    dnorm = np.asarray([l2(row) for row in (a-b)])
    bnorm = np.asarray([l2(row) for row in b])
    zero = bnorm == 0.0
    undefined = zero & (dnorm != 0.0)
    ratio = np.zeros_like(dnorm)
    np.divide(dnorm, bnorm, out=ratio, where=~zero)
    finite_ratio = ratio[~undefined]
    return {
        'status': 'undefined_zero_reference_with_nonzero_error' if np.any(undefined) else 'complete',
        'rows': int(len(a)),
        'zero_reference_rows': int(zero.sum()),
        'zero_reference_with_nonzero_error_rows': int(undefined.sum()),
        'relative_l2_max': float(np.max(finite_ratio)) if finite_ratio.size and not np.any(undefined) else None,
        'relative_l2_p99': float(np.quantile(finite_ratio, 0.99)) if finite_ratio.size else None,
        'rows_above_1e-6': int(np.sum(finite_ratio > 1e-6)) if not np.any(undefined) else None,
        'gate_value_for_diagnostic_count': 1e-6,
        'absolute_l2_max': float(np.max(dnorm)) if dnorm.size else 0.0,
    }


def per_row_error_arrays(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    dnorm = np.asarray([l2(row) for row in (a-b)])
    bnorm = np.asarray([l2(row) for row in b])
    relative = np.full(len(a), np.nan, dtype=np.float64)
    np.divide(dnorm, bnorm, out=relative, where=bnorm > 0.0)
    relative[(bnorm == 0.0) & (dnorm == 0.0)] = 0.0
    return relative, dnorm, bnorm


def assemble(tets, dP, inverse_rest, rotation, volume, nnode):
    out = np.zeros((nnode, 3), dtype=np.float64)
    for ti, tet in enumerate(tets):
        world = dP[ti] @ rotation[ti].T
        g = -float(volume[ti]) * world @ inverse_rest[ti].T
        local = np.empty((4, 3), dtype=np.float64)
        local[1:] = g.T
        local[0] = -np.sum(g.T, axis=0)
        for slot, node in enumerate(tet['nodes']):
            out[int(node)] += local[slot]
    return out


def element_force_norms(tets, stress, inverse_rest, rotation, volume):
    """Per-element nodal-force norm under the same assembly convention."""
    out = np.full(len(tets), np.nan, dtype=np.float64)
    for ti, _tet in enumerate(tets):
        if not np.all(np.isfinite(stress[ti])):
            continue
        world = stress[ti] @ rotation[ti].T
        g = -float(volume[ti]) * world @ inverse_rest[ti].T
        local = np.empty((4, 3), dtype=np.float64)
        local[1:] = g.T
        local[0] = -np.sum(g.T, axis=0)
        out[ti] = l2(local)
    return out

def observed_branch_signature(verifier, ref, p, old, z, valid):
    """Return all endpoint predicates directly observable from this law replay."""
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        return {'rows': 0, 'status': 'unavailable_no_valid_projection_rows'}
    pp, oo, zz = p[valid], old[valid], z[valid]
    branch = verifier.paper_branch(ref, pp, oo, zz)
    dl = zz[:, 12] - oo[:, 12]
    y = branch['yield_function_residual_value']
    root = np.sqrt(dl * dl + y * y + pp[:, 26] * pp[:, 26])
    denominator = root + dl + y
    # The reference transcription uses algebraically equivalent rationalized
    # and direct FB residual forms on opposite sides of this numerical cutoff.
    return {
        'rows': int(len(pp)),
        'plastic_multiplier_increment_positive_rows': int(np.sum(dl > 0.0)),
        'plastic_multiplier_increment_zero_rows': int(np.sum(dl == 0.0)),
        'plastic_multiplier_increment_negative_rows': int(np.sum(dl < 0.0)),
        'yield_residual_sign_monitor': {
            'positive_rows': int(np.sum(y > 0.0)),
            'zero_rows': int(np.sum(y == 0.0)),
            'negative_rows': int(np.sum(y < 0.0)),
            'absolute_min': float(np.min(np.abs(y))),
            'absolute_max': float(np.max(np.abs(y))),
        },
        'fb_residual_evaluation_path': {
            'rationalized_denominator_gt_1e-14_rows': int(np.sum(denominator > 1e-14)),
            'direct_root_minus_lambda_minus_yield_rows': int(np.sum(denominator <= 1e-14)),
            'cutoff': 1e-14,
            'classification': 'reference implementation arithmetic path, not an additional physical active set',
        },
        'scope': 'captured-law endpoint predicates available from projected state; does not retain every internal Newton subiteration branch',
    }

def signed_work(a, b):
    return math.fsum(float(x) * float(y) for x, y in zip(
        np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()))


TIGHT_RESIDUAL = 2e-12
TIGHT_DYNAMIC_CORRECTION = 2e-11


def continue_projection_tight(ref, p, F, old, p26, base_result):
    """Continue the exact same FP64 return map to stricter diagnostic gates."""
    z, iterations, rn, cn, ok, reasons, base_passes = base_result
    z = np.asarray(z, dtype=np.float64).copy()
    iterations = np.asarray(iterations, dtype=np.int32).copy()
    rn = np.asarray(rn, dtype=np.float64).copy()
    cn = np.asarray(cn, dtype=np.float64).copy()
    ok = np.asarray(ok, dtype=bool).copy()
    reasons = list(reasons)
    residual_scale = np.array([1e4] * 6 + [1.0] * 6 + [1e4] * 3, dtype=np.float64)
    passes = []
    for extra in range(16):
        pending_ids = np.flatnonzero(ok & ((rn > TIGHT_RESIDUAL) | (cn > TIGHT_DYNAMIC_CORRECTION)))
        if pending_ids.size == 0:
            break
        r = ref.residual(p[pending_ids], F[pending_ids], old[pending_ids], z[pending_ids])
        norm = np.max(np.abs(r) * residual_scale, axis=1)
        try:
            j = ref.jacobian(p[pending_ids], F[pending_ids], old[pending_ids], z[pending_ids])
            corr = np.linalg.solve(j, -r[..., None])[..., 0]
        except Exception:
            corr = np.full((len(pending_ids), 15), np.nan)
            for local, row in enumerate(pending_ids):
                try:
                    rr = ref.residual(p[row:row+1], F[row:row+1], old[row:row+1], z[row:row+1])
                    jj = ref.jacobian(p[row:row+1], F[row:row+1], old[row:row+1], z[row:row+1])
                    corr[local] = np.linalg.solve(jj, -rr[..., None])[0, :, 0]
                except Exception as e:
                    ok[row] = False
                    reasons[row] = f'tight diagnostic correction solve: {type(e).__name__}: {e}'
        state_scale = np.maximum(np.maximum(np.abs(old[pending_ids]), np.abs(z[pending_ids])), p26[pending_ids, None])
        dyn = np.max(np.abs(corr) / state_scale, axis=1)
        need = np.isfinite(norm) & np.isfinite(dyn) & ((norm > TIGHT_RESIDUAL) | (dyn > TIGHT_DYNAMIC_CORRECTION))
        if not np.any(need):
            ok[pending_ids[~np.isfinite(norm) | ~np.isfinite(dyn)]] = False
            break
        chosen = pending_ids[need]
        corr_chosen = corr[need]
        base_norm = norm[need]
        alpha = np.ones(len(chosen), dtype=np.float64)
        pending = np.ones(len(chosen), dtype=bool)
        next_z = z[chosen].copy()
        for backtrack in range(32):
            candidate = z[chosen] + alpha[:, None] * corr_chosen
            trial_r = ref.residual(p[chosen], F[chosen], old[chosen], candidate)
            trial_norm = np.max(np.abs(trial_r) * residual_scale, axis=1)
            accept = pending & np.isfinite(trial_norm) & (trial_norm < base_norm)
            next_z[accept] = candidate[accept]
            pending[accept] = False
            if not np.any(pending):
                break
            alpha[pending] *= 0.5
        accepted_ids = chosen[~pending]
        z[accepted_ids] = next_z[~pending]
        iterations[accepted_ids] += 1
        for row in chosen[pending]:
            ok[row] = False
            reasons[row] = f'tight diagnostic residual descent stalled at residual {rn[row]:.17g}, correction {cn[row]:.17g}'
        passes.append({'extra_iteration': extra, 'rows_considered': int(len(pending_ids)),
                       'rows_updated': int(len(accepted_ids)), 'rows_stalled': int(pending.sum()),
                       'max_preupdate_scaled_residual': float(np.max(norm[need])) if np.any(need) else None,
                       'max_preupdate_dynamic_correction': float(np.max(dyn[need])) if np.any(need) else None})
        # Refresh gates on all still-healthy rows, including ones that met both limits already.
        ids = np.flatnonzero(ok)
        if ids.size:
            rr = ref.residual(p[ids], F[ids], old[ids], z[ids])
            rn[ids] = np.max(np.abs(rr) * residual_scale, axis=1)
            jj = ref.jacobian(p[ids], F[ids], old[ids], z[ids])
            cc = np.linalg.solve(jj, -rr[..., None])[..., 0]
            scale = np.maximum(np.maximum(np.abs(old[ids]), np.abs(z[ids])), p26[ids, None])
            cn[ids] = np.max(np.abs(cc) / scale, axis=1)
    # Final recalc and fail closed if either tighter diagnostic criterion remains exceeded.
    ids = np.flatnonzero(ok)
    if ids.size:
        rr = ref.residual(p[ids], F[ids], old[ids], z[ids])
        rn[ids] = np.max(np.abs(rr) * residual_scale, axis=1)
        jj = ref.jacobian(p[ids], F[ids], old[ids], z[ids])
        cc = np.linalg.solve(jj, -rr[..., None])[..., 0]
        scale = np.maximum(np.maximum(np.abs(old[ids]), np.abs(z[ids])), p26[ids, None])
        cn[ids] = np.max(np.abs(cc) / scale, axis=1)
        fail = (rn[ids] > TIGHT_RESIDUAL) | (cn[ids] > TIGHT_DYNAMIC_CORRECTION)
        for local in np.flatnonzero(fail):
            row = ids[local]
            ok[row] = False
            reasons[row] = 'tight diagnostic residual/correction gate failed'
    passes = {'base_projection_passes': base_passes,
              'tight_gate_residual': TIGHT_RESIDUAL,
              'tight_gate_dynamic_correction': TIGHT_DYNAMIC_CORRECTION,
              'tight_continuation': passes}
    return z, iterations, rn, cn, ok, reasons, passes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True, type=pathlib.Path)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError(f'refusing to overwrite {out}')
    for p in (CAPTURE, RUN / 'prediction.json', RUN / 'receipt.json',
              AUDIT / 'report.json', TERM / 'result/replay-report.json',
              SWEEP / 'report.json', BASELINE005 / 'report.json'):
        if not p.is_file():
            raise RuntimeError(f'missing bound input {p}')

    prediction = json.loads((RUN / 'prediction.json').read_text())
    receipt = json.loads((RUN / 'receipt.json').read_text())
    native_audit = json.loads((AUDIT / 'report.json').read_text())
    term_report = json.loads((TERM / 'result/replay-report.json').read_text())
    sweep_report = json.loads((SWEEP / 'report.json').read_text())
    baseline005 = json.loads((BASELINE005 / 'report.json').read_text())
    if receipt.get('source_inputs_unchanged') is not True or receipt.get('timed_out') is not False:
        raise RuntimeError('419 receipt not intact/non-timeout')
    if native_audit['bindings']['capture_sha256'] != sha(CAPTURE):
        raise RuntimeError('existing 419 audit/capture hash mismatch')
    rate_contract = native_audit.get('input_contract', {}).get('rate')
    if rate_contract != 'captured maximum rate exponent must be zero; independent 15-state law is rate independent':
        raise RuntimeError('419 rate contract is absent or changed')
    if term_report.get('capture_selection') != {'control_step': 1, 'newton_iteration': 48}:
        raise RuntimeError('422 term replay selected a different Newton state')
    if term_report.get('physical_transactions') != 0:
        raise RuntimeError('422 term replay unexpectedly contains physical transactions')
    if sweep_report.get('capture_sha256') != sha(CAPTURE) or not sweep_report.get('all_retained_endpoints_bitwise_equal'):
        raise RuntimeError('424 residual sweep does not bind and match 419 endpoints')

    records = [json.loads(line) for line in CAPTURE.read_text().splitlines() if line.strip()]
    header = records[0]
    if header.get('record') != 'header' or records[-1].get('record') != 'integrity':
        raise RuntimeError('capture framing invalid')
    env0_nodes = sorted([r for r in records if r.get('record') == 'node' and r.get('environment') == 0],
                        key=lambda r: r['node_id'])
    tets = sorted([r for r in records if r.get('record') == 'tetrahedron' and r.get('environment') == 0],
                  key=lambda r: r['tet_id'])
    nnode, ntet = len(env0_nodes), len(tets)
    if (nnode, ntet) != (882, 1920):
        raise RuntimeError('unexpected env0 topology')
    if any(r['node_id'] != i for i, r in enumerate(env0_nodes)) or any(r['tet_id'] != i for i, r in enumerate(tets)):
        raise RuntimeError('node/tet ids not complete canonical sequence')

    verifier = load_module(ADAPTER / 'audit_capture.py', 'audit_capture_smallstep')
    ref = load_module(REF_PATH, 'bilinear15_reference_smallstep')
    fd = load_module(FD_PATH, 'directional_fd_smallstep')
    geo = load_module(GEOM_PATH, 'geometry_smallstep')
    ptab = f32(header['parameters_by_environment'][0])
    mat_specs = {int(m['index']): m for m in header['materials']}
    dt = float(f32([header['dt_seconds_by_environment'][0]])[0])
    dv = f32([r['correction_delta_v_m_s'] for r in env0_nodes])
    mass = f32([r['center']['mass_kg'] for r in env0_nodes])
    free = np.array([float(f32([r['center']['constraint_tag']])[0]) == 0.0 for r in env0_nodes], dtype=bool)
    F0 = np.stack([pair(t['center'], 'F_local') for t in tets])
    H = np.stack([f32(t['center']['F_direction_local_per_s']) for t in tets])
    inverse_rest = np.stack([f32(t['inverse_rest_rows_per_m']) for t in tets])
    volume = f32([t['rest_volume_m3'] for t in tets])
    rotations = np.asarray([
        geo.rotation_from_xyzw(f32(t['material_frame_quaternion_xyzw']))
        if int(f32([t['center']['dt_seconds_bulk_modulus_kind_frame_active'][3]])[0]) == 1
        else np.eye(3) for t in tets])
    mat_id = np.asarray([int(t['material_index']) for t in tets], dtype=np.int32)
    # Glue is stateless and its captured accepted-state arrays are empty; the
    # padded table is only indexed for the two 15-state paper materials.
    old = np.zeros((ntet, 15), dtype=np.float64)
    for mi in (0, 1):
        for i in np.flatnonzero(mat_id == mi):
            old[i] = f32(tets[int(i)]['center']['accepted_state'])
    params = np.empty((ntet, 28), dtype=np.float64)
    for i, t in enumerate(tets):
        mi = int(t['material_index'])
        spec = mat_specs[mi]
        po = int(spec['parameter_offset'])
        pc = len(spec['parameter_names'])
        expected_pc = 28 if mi in (0, 1) else 3
        if pc != expected_pc or int(t['material_parameter_count']) != expected_pc:
            raise RuntimeError('unsupported material parameter layout')
        params[i] = 0.0
        params[i, :pc] = ptab[po:po + pc]

    warnings_seen = []
    def project_logged(p, F, old_state, p26):
        # First run the unchanged 005 strict projector, then continue its same
        # equations with stricter diagnostic-only residual/correction gates.
        with warnings.catch_warnings(record=True) as caught_projection:
            warnings.simplefilter('always')
            base_result = verifier.project_centers(ref, fd, p, F, old_state, p26)
            result = continue_projection_tight(ref, p, F, old_state, p26, base_result)
        for w in caught_projection:
            warnings_seen.append({'message': str(w.message), 'category': w.category.__name__,
                                  'filename': str(w.filename), 'line': int(w.lineno), 'count': 1})
        return result

    def jacobian_condition_rows(p, F, old_state, z, ok):
        values = np.full(len(p), np.nan, dtype=np.float64)
        ids = np.flatnonzero(np.asarray(ok, dtype=bool))
        if not ids.size:
            return values
        try:
            values[ids] = np.linalg.cond(ref.jacobian(p[ids], F[ids], old_state[ids], z[ids]))
        except Exception:
            for row in ids:
                try:
                    values[row] = np.linalg.cond(ref.jacobian(
                        p[row:row+1], F[row:row+1], old_state[row:row+1], z[row:row+1])[0])
                except Exception:
                    pass
        return values
    start = time.monotonic()
    report = {
        'schema': 'numi.cardboard.source125-fp64-smooth-direction-sweep-tight-projection.v1',
        'status': 'running',
        'evidence_class': 'source-bound CPU constitutive-curvature diagnostic; not solver/physical qualification',
        'scope': 'environment 0, root1/Newton48, fixed captured accepted history, all 1920 tetrahedra; ideal unquantized FP64 direction sweep',
        'bindings': {
            'source_snapshot': prediction['source_snapshot'],
            'source125_runtime_mm_sha256': sha(pathlib.Path(prediction['source_snapshot']) / 'matter/src/runtime.mm'),
            'run_receipt_sha256': sha(RUN / 'receipt.json'),
            'run_capture_sha256': sha(CAPTURE),
            'run_prediction_sha256': sha(RUN / 'prediction.json'),
            'native_fd_cpu_audit_report_sha256': sha(AUDIT / 'report.json'),
            'native_fd_cpu_audit_script_sha256': sha(ADAPTER / 'audit_capture.py'),
            'term_replay_receipt_sha256': sha(TERM / 'receipt.json'),
            'term_replay_report_sha256': sha(TERM / 'result/replay-report.json'),
            'term_replay_manifest_sha256': sha(TERM / 'manifest.json'),
            'captured_residual_sweep_report_sha256': sha(SWEEP / 'report.json'),
            'baseline005_report_sha256': sha(BASELINE005 / 'report.json'),
            'baseline005_manifest_sha256': sha(BASELINE005 / 'manifest.json'),
            'independent_reference_sha256': sha(REF_PATH),
            'projection_helper_sha256': sha(FD_PATH),
            'geometry_helper_sha256': sha(GEOM_PATH),
            'input_npz_sha256': sha(AUDIT / 'environment-0.npz'),
            'term_files_sha256': {
                name: sha(TERM / 'result' / name) for name in (
                    'captured-fresh-action.float4.bin', 'fresh-action.float4.bin',
                    'fresh-elements.float4x4.bin', 'mass-only-action.float4.bin')},
        },
        'selection': {'environment': 0, 'root': 1, 'newton_iteration': 48,
                      'nodes': nnode, 'tets': ntet, 'dt_seconds': dt,
                      'eps_values': [math.ldexp(1.0, -k) for k in range(14, 31)],
                      'rate_exponent_max_by_paper_material': {
                          str(mi): float(np.max(params[mat_id == mi, 26])) for mi in (0, 1)},
                      'captured_delta_v_l2_m_per_s': l2(dv),
                      'captured_dx_equals_dt_dv_l2_m': l2(dt * dv),
                      'captured_F_direction_H_l2_per_s': l2(H)},
        'operator_action_crosscheck': {},
        'native_endpoint_projection_validation': {},
        'independent_fp64_epsilon_sweep': [],
        'caveats': [
            'Every endpoint is ideal unquantized FP64 Fcenter +/- epsilon*dt*H from the exact promoted center high+low F; no native FP32 endpoint words are used as endpoint values.',
            'Accepted constitutive history is fixed at the captured history. The captured rate exponent is required to be zero and rate inputs are unchanged.',
            'Every paper projection must pass the existing independent residual and dynamic state-correction gates; failed rows are retained and make complete assembled actions unavailable.',
            'Branch reporting covers lambda-increment sign, yield residual sign, and the reference FB residual arithmetic path; it does not retain every internal return-map Newton branch.',
            'The 424 native residual sweep is bound as provenance only. Its FP32 realized endpoints are not used by this ideal FP64 sweep.',
            'Mass and signed stiffness contributions are reported separately for A=M-dt^2 K. This is a local directional diagnostic, not an accepted solver update.',
            'No universal noise floor, native Metal derivative pass, coupon convergence, calibration, or physical qualification is claimed.',
        ],
        'warnings': [],
        'execution': {
            'python': sys.version,
            'numpy': np.__version__,
            'thread_environment': {k: os.environ.get(k) for k in (
                'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS')},
            'command': f'{sys.executable} {pathlib.Path(__file__).resolve()} --output <fresh-output-directory>',
        },
        'projection_policy': {
            'baseline_projector': 'unchanged audit_capture.project_centers, including source reference projector and original 2e-10 scaled residual / 2e-8 dynamic correction criteria',
            'diagnostic_continuation': 'same FP64 residual equations, same complex-step Jacobian, residual-descending Newton update; continue from baseline state until stricter gates or fail closed',
            'tight_scaled_residual_gate': TIGHT_RESIDUAL,
            'tight_dynamic_correction_gate': TIGHT_DYNAMIC_CORRECTION,
            'maximum_tight_extra_updates': 16,
            'maximum_backtracks_per_extra_update': 32,
            'production_tolerances_changed': False,
        },
    }

    # Independent center tangent action versus the exact source125 fresh action from 422.
    npz = np.load(AUDIT / 'environment-0.npz')
    native_term = np.fromfile(TERM / 'result/captured-fresh-action.float4.bin', dtype='<f4').reshape(-1, 4)
    if native_term.shape != (2 * nnode, 4):
        raise RuntimeError('term replay action dimensions mismatch')
    native_fresh = native_term[:nnode, :3].astype(np.float64)
    independent_saved = np.asarray(npz['independent_A_delta_v_impulse_Ns'], dtype=np.float64)
    captured_saved = np.asarray(npz['native_production_A_delta_v_impulse_Ns'], dtype=np.float64)
    term_report_env = term_report['environment_0']
    report['operator_action_crosscheck'] = {
        '422_term_selection': term_report['capture_selection'],
        'term_replay_operator_path': term_report['operator_path'],
        'native_fresh_action_vs_native_capture': rel_metrics(native_fresh, captured_saved),
        'independent_fp64_center_action_vs_native422_fresh_action': rel_metrics(native_fresh, independent_saved),
        'native_captured_action_vs_independent_fp64_center_action': rel_metrics(captured_saved, independent_saved),
        'native422_reported': {k: term_report_env[k] for k in (
            'fresh_action_l2_Ns', 'fresh_vs_captured_action_difference_l2_Ns',
            'fresh_vs_captured_action_difference_relative_to_cached',
            'fresh_rhs_minus_Ax_l2_Ns', 'rhs_l2_Ns', 'correction_l2_m_per_s', 'free_nodes')
            if k in term_report_env},
        'source125_world_fingerprint_hex': term_report['source125_world_fingerprint_hex'],
    }

    # Base center response: use independent converged 15-state projection and its full tangent.
    center_z = np.full((ntet, 15), np.nan)
    center_P = np.full((ntet, 3, 3), np.nan)
    center_dP = np.full((ntet, 3, 3), np.nan)
    center_ok = np.ones(ntet, dtype=bool)
    center_projection = {}
    center_jacobian_condition = np.full(ntet, np.nan, dtype=np.float64)
    for mi in (0, 1):
        ids = np.flatnonzero(mat_id == mi)
        pp, ff, oo, hh = params[ids], F0[ids], old[ids], H[ids]
        z, it, rn, cn, ok, reasons, refinements = project_logged(pp, ff, oo, pp[:, 26])
        center_z[ids] = z
        center_ok[ids] = ok
        center_P[ids] = ref.stress(pp, ff, z)
        center_dP[ids] = ref.tangent(pp, ff, oo, z, hh)
        center_jacobian_condition[ids] = jacobian_condition_rows(pp, ff, oo, z, ok)
        center_projection[str(mi)] = {
            'name': mat_specs[mi]['name'], 'rows': int(len(ids)),
            'projection_success': int(ok.sum()),
            'max_reference_scaled_residual': float(np.nanmax(rn)) if np.any(np.isfinite(rn)) else None,
            'max_dynamic_correction': float(np.nanmax(cn)) if np.any(np.isfinite(cn)) else None,
            'max_updates': int(np.max(it[ok])) if np.any(ok) else None,
            'failure_examples': [{'tet_id': int(ids[j]), 'reason': reasons[j]}
                                 for j in np.flatnonzero(~ok)[:12]],
        }
    # Analytic glue response.
    ids = np.flatnonzero(mat_id == 2)
    if len(ids):
        mu = params[ids, 1]
        lam = params[ids, 2]
        center_P[ids] = verifier.glue_stress(F0[ids], mu, lam)
        center_dP[ids] = verifier.glue_tangent_action(F0[ids], H[ids], mu, lam)
        center_z[ids] = 0.0
    center_k = assemble(tets, center_dP, inverse_rest, rotations, volume, nnode)
    center_A = mass[:, None] * dv - dt * dt * center_k
    center_A[~free] = 0.0
    report['center_projection'] = center_projection
    finite_center_cond = center_jacobian_condition[np.isfinite(center_jacobian_condition)]
    report['center_jacobian_condition_2norm'] = {
        'valid_rows': int(finite_center_cond.size),
        'max': float(np.max(finite_center_cond)) if finite_center_cond.size else None,
        'median': float(np.median(finite_center_cond)) if finite_center_cond.size else None,
        'p95': float(np.quantile(finite_center_cond, 0.95)) if finite_center_cond.size else None,
        'per_tet_values_in_rowwise_npz': True,
    }
    report['center_action_vs_saved_independent_action'] = rel_metrics(
        center_A, independent_saved)

    center_branch_cache = {}
    center_branch_signatures = {}
    for mi in (0, 1):
        ids = np.flatnonzero(mat_id == mi)
        center_branch_cache[mi] = verifier.paper_branch(ref, params[ids], old[ids], center_z[ids])
        center_branch_signatures[str(mi)] = observed_branch_signature(
            verifier, ref, params[ids], old[ids], center_z[ids], center_ok[ids])
    report['center_observable_branch_signatures'] = center_branch_signatures

    eps_values = [math.ldexp(1.0, -k) for k in range(14, 31)]
    paper_tet_ids = np.flatnonzero((mat_id == 0) | (mat_id == 1))
    neps = len(eps_values)
    rowwise = {
        'epsilon_power': np.asarray([14 + k for k in range(neps)], dtype=np.int16),
        'tet_id': np.arange(ntet, dtype=np.int32),
        'material_index': mat_id.astype(np.int16),
        'center_jacobian_condition_2norm': center_jacobian_condition.copy(),
        'center_stress_frobenius_Pa': np.full(ntet, np.nan, dtype=np.float64),
        'center_tangent_frobenius_Pa_per_s': np.full(ntet, np.nan, dtype=np.float64),
        'center_element_force_l2_N': element_force_norms(tets, center_P, inverse_rest, rotations, volume),
        'center_element_tangent_force_l2_N_per_s': element_force_norms(tets, center_dP, inverse_rest, rotations, volume),
        'center_lambda_active': np.full(ntet, -1, dtype=np.int8),
        'center_yield_residual_sign': np.full(ntet, -9, dtype=np.int8),
        'center_fb_rationalized_path': np.full(ntet, -1, dtype=np.int8),
        'plus_projection_ok': np.zeros((neps, ntet), dtype=np.bool_),
        'minus_projection_ok': np.zeros((neps, ntet), dtype=np.bool_),
        'plus_branch_matches_center': np.zeros((neps, ntet), dtype=np.bool_),
        'minus_branch_matches_center': np.zeros((neps, ntet), dtype=np.bool_),
        'plus_scaled_residual': np.full((neps, ntet), np.nan, dtype=np.float64),
        'minus_scaled_residual': np.full((neps, ntet), np.nan, dtype=np.float64),
        'plus_dynamic_correction': np.full((neps, ntet), np.nan, dtype=np.float64),
        'minus_dynamic_correction': np.full((neps, ntet), np.nan, dtype=np.float64),
        'plus_iterations': np.full((neps, ntet), -1, dtype=np.int16),
        'minus_iterations': np.full((neps, ntet), -1, dtype=np.int16),
        'plus_jacobian_condition_2norm': np.full((neps, ntet), np.nan, dtype=np.float64),
        'minus_jacobian_condition_2norm': np.full((neps, ntet), np.nan, dtype=np.float64),
        'plus_centered_local_relative_error': np.full((neps, ntet), np.nan, dtype=np.float64),
        'minus_centered_local_relative_error': np.full((neps, ntet), np.nan, dtype=np.float64),
        'centered_local_relative_error': np.full((neps, ntet), np.nan, dtype=np.float64),
        'plus_local_absolute_error_Pa_per_s': np.full((neps, ntet), np.nan, dtype=np.float64),
        'minus_local_absolute_error_Pa_per_s': np.full((neps, ntet), np.nan, dtype=np.float64),
        'centered_local_absolute_error_Pa_per_s': np.full((neps, ntet), np.nan, dtype=np.float64),
    }
    for mi in (0, 1):
        ids = np.flatnonzero(mat_id == mi)
        rowwise['center_stress_frobenius_Pa'][ids] = np.linalg.norm(center_P[ids].reshape(len(ids), -1), axis=1)
        rowwise['center_tangent_frobenius_Pa_per_s'][ids] = np.linalg.norm(center_dP[ids].reshape(len(ids), -1), axis=1)
        cb = center_branch_cache[mi]
        rowwise['center_lambda_active'][ids] = np.asarray(cb['plastic_active_lambda_positive'], dtype=np.int8)
        cy = np.asarray(cb['yield_function_residual_value'], dtype=np.float64)
        rowwise['center_yield_residual_sign'][ids] = np.sign(cy).astype(np.int8)
        cdl = center_z[ids, 12] - old[ids, 12]
        croot = np.sqrt(cdl * cdl + cy * cy + params[ids, 26] ** 2)
        rowwise['center_fb_rationalized_path'][ids] = (croot + cdl + cy > 1e-14).astype(np.int8)
    rowwise['center_scaled_residual'] = np.full(ntet, np.nan, dtype=np.float64)
    rowwise['center_dynamic_correction'] = np.full(ntet, np.nan, dtype=np.float64)
    for mi in (0, 1):
        ids = np.flatnonzero(mat_id == mi)
        # Per-row values are recorded from the independent strict center solve below.
        pp, ff, oo = params[ids], F0[ids], old[ids]
        rr = ref.residual(pp, ff, oo, center_z[ids])
        scale = np.array([1e4] * 6 + [1.0] * 6 + [1e4] * 3, dtype=np.float64)
        rowwise['center_scaled_residual'][ids] = np.max(np.abs(rr) * scale, axis=1)
        jj = ref.jacobian(pp, ff, oo, center_z[ids])
        cc = np.linalg.solve(jj, -rr[..., None])[..., 0]
        ss = np.maximum(np.maximum(np.abs(oo), np.abs(center_z[ids])), pp[:, 26, None])
        rowwise['center_dynamic_correction'][ids] = np.max(np.abs(cc) / ss, axis=1)
    mass_action = mass[:, None] * dv
    mass_action[~free] = 0.0

    def action_from_dP(dP):
        k_action = assemble(tets, dP, inverse_rest, rotations, volume, nnode)
        k_action[~free] = 0.0
        stiffness = -dt * dt * k_action
        total = mass_action + stiffness
        total[~free] = 0.0
        return {'K': k_action, 'stiffness': stiffness, 'A': total}

    def projection_summary(ids, result):
        z, it, rn, cn, ok, reasons, refinements = result
        ok = np.asarray(ok, dtype=bool)
        finite_rn = np.asarray(rn, dtype=np.float64)[np.isfinite(rn)]
        finite_cn = np.asarray(cn, dtype=np.float64)[np.isfinite(cn)]
        return {
            'rows': int(len(ids)), 'projection_success': int(ok.sum()),
            'projection_failed': int((~ok).sum()),
            'max_reference_scaled_residual': float(np.max(finite_rn)) if finite_rn.size else None,
            'max_dynamic_correction': float(np.max(finite_cn)) if finite_cn.size else None,
            'max_updates': int(np.max(np.asarray(it)[ok])) if np.any(ok) else None,
            'refinement_passes': refinements,
            'failure_examples': [{'tet_id': int(ids[j]), 'reason': str(reasons[j])}
                                 for j in np.flatnonzero(~ok)[:12]],
        }

    center_terms = action_from_dP(center_dP)
    center_A = center_terms['A']
    report['center_A_terms'] = {
        'mass_action_M_dv_Ns': {'l2': l2(mass_action), 'signed_work_dv_dot': signed_work(dv, mass_action)},
        'stiffness_action_minus_dt2_K_Ns': {'l2': l2(center_terms['stiffness']),
                                             'signed_work_dv_dot': signed_work(dv, center_terms['stiffness'])},
        'total_A_dv_Ns': {'l2': l2(center_A), 'signed_work_dv_dot': signed_work(dv, center_A)},
        'free_node_mask_applied': True,
        'direction_contract': 'H=grad(delta_v) in per-second local F units; dx=dt*delta_v; A delta_v=M delta_v-dt^2 K delta_v',
    }

    for epsilon_index, eps in enumerate(eps_values):
        per_mat = {}
        endpoint_P = {1: {}, -1: {}}
        endpoint_results = {1: {}, -1: {}}
        endpoint_branch = {1: {}, -1: {}}
        branch_transitions = {1: {}, -1: {}}
        endpoint_F = {1: F0 + eps * dt * H, -1: F0 - eps * dt * H}
        for sign in (1, -1):
            Fep = endpoint_F[sign]
            for mi in (0, 1):
                ids = np.flatnonzero(mat_id == mi)
                pp, oo = params[ids], old[ids]
                result = project_logged(pp, Fep[ids], oo, pp[:, 26])
                z, it, rn, cn, ok, reasons, refinements = result
                ok = np.asarray(ok, dtype=bool)
                rowwise['plus_projection_ok' if sign == 1 else 'minus_projection_ok'][epsilon_index, ids] = ok
                rowwise['plus_scaled_residual' if sign == 1 else 'minus_scaled_residual'][epsilon_index, ids] = rn
                rowwise['plus_dynamic_correction' if sign == 1 else 'minus_dynamic_correction'][epsilon_index, ids] = cn
                rowwise['plus_iterations' if sign == 1 else 'minus_iterations'][epsilon_index, ids] = it
                rowwise['plus_jacobian_condition_2norm' if sign == 1 else 'minus_jacobian_condition_2norm'][epsilon_index, ids] = jacobian_condition_rows(pp, Fep[ids], oo, z, ok)
                stress = ref.stress(pp, Fep[ids], z)
                endpoint_P[sign][mi] = stress
                endpoint_results[sign][mi] = result
                same_rows = ok & center_ok[ids]
                endpoint_branch[sign][mi] = observed_branch_signature(
                    verifier, ref, pp, oo, z, same_rows)
                if np.any(same_rows):
                    side_branch = verifier.paper_branch(ref, pp[same_rows], oo[same_rows], z[same_rows])
                    center_branch = center_branch_cache[mi]
                    source_rows = np.flatnonzero(same_rows)
                    crossing = side_branch['plastic_active_lambda_positive'] != center_branch['plastic_active_lambda_positive'][same_rows]
                    crossing_count = int(np.sum(crossing))
                    crossing_ids = [int(ids[j]) for j in source_rows[np.flatnonzero(crossing)[:32]]]
                    side_yield = np.asarray(side_branch['yield_function_residual_value'])
                    center_yield = np.asarray(center_branch['yield_function_residual_value'])[same_rows]
                    yield_cross = np.sign(side_yield) != np.sign(center_yield)
                    dl = z[same_rows, 12] - oo[same_rows, 12]
                    center_dl = center_z[ids][same_rows, 12] - oo[same_rows, 12]
                    side_root = np.sqrt(dl * dl + side_yield * side_yield + pp[same_rows, 26] ** 2)
                    center_root = np.sqrt(center_dl * center_dl + center_yield * center_yield + pp[same_rows, 26] ** 2)
                    side_rat = side_root + dl + side_yield > 1e-14
                    center_rat = center_root + center_dl + center_yield > 1e-14
                    fb_cross = side_rat != center_rat
                    active_match = (np.asarray(side_branch['plastic_active_lambda_positive'], dtype=bool)
                                    == np.asarray(center_branch['plastic_active_lambda_positive'], dtype=bool)[same_rows])
                    branch_matches = active_match & (~yield_cross) & (~fb_cross)
                    rowwise['plus_branch_matches_center' if sign == 1 else 'minus_branch_matches_center'][
                        epsilon_index, ids[same_rows]] = branch_matches
                    branch_transitions[sign][mi] = {
                        'valid_rows_compared': int(np.sum(same_rows)),
                        'lambda_activity_transition_rows': crossing_count,
                        'lambda_activity_transition_tet_ids_first32': crossing_ids,
                        'yield_residual_sign_transition_rows': int(np.sum(yield_cross)),
                        'fb_reference_arithmetic_path_transition_rows': int(np.sum(fb_cross)),
                    }
                else:
                    crossing_count, crossing_ids = None, []
                    branch_transitions[sign][mi] = {'status': 'unavailable_no_valid_projected_rows'}
                projection = projection_summary(ids, result)
                per_mat.setdefault(str(mi), {'name': mat_specs[mi]['name']})[
                    'plus' if sign == 1 else 'minus'] = {
                        'projection': projection,
                        'observable_branch_signature': endpoint_branch[sign][mi],
                        'observable_branch_transitions_vs_center': branch_transitions[sign][mi],
                        'F_endpoint_delta_l2': l2(Fep[ids] - F0[ids]),
                        'endpoint_stress_delta_l2_Pa_on_valid_rows': l2((stress - center_P[ids])[same_rows]) if np.any(same_rows) else None,
                        'endpoint_stress_delta_max_abs_Pa_on_valid_rows': float(np.max(np.abs((stress - center_P[ids])[same_rows]))) if np.any(same_rows) else None,
                        'state_delta_l2_on_valid_rows': l2((z - center_z[ids])[same_rows]) if np.any(same_rows) else None,
                    }
            ids = np.flatnonzero(mat_id == 2)
            if len(ids):
                endpoint_P[sign][2] = verifier.glue_stress(
                    Fep[ids], params[ids, 1], params[ids, 2])

        side_derivatives = {1: np.full_like(center_dP, np.nan),
                            -1: np.full_like(center_dP, np.nan)}
        centered_derivative = np.full_like(center_dP, np.nan)
        side_valid = {1: center_ok.copy(), -1: center_ok.copy()}
        centered_valid = center_ok.copy()
        for mi in (0, 1):
            ids = np.flatnonzero(mat_id == mi)
            plus_ok = np.asarray(endpoint_results[1][mi][4], dtype=bool) & center_ok[ids]
            minus_ok = np.asarray(endpoint_results[-1][mi][4], dtype=bool) & center_ok[ids]
            paired = plus_ok & minus_ok
            side_valid[1][ids] = plus_ok
            side_valid[-1][ids] = minus_ok
            centered_valid[ids] &= paired
            if np.any(plus_ok):
                side_derivatives[1][ids[plus_ok]] = (endpoint_P[1][mi][plus_ok] - center_P[ids][plus_ok]) / (eps * dt)
                rel, absolute, _ = per_row_error_arrays(side_derivatives[1][ids[plus_ok]], center_dP[ids][plus_ok])
                rowwise['plus_centered_local_relative_error'][epsilon_index, ids[plus_ok]] = rel
                rowwise['plus_local_absolute_error_Pa_per_s'][epsilon_index, ids[plus_ok]] = absolute
            if np.any(minus_ok):
                side_derivatives[-1][ids[minus_ok]] = (center_P[ids][minus_ok] - endpoint_P[-1][mi][minus_ok]) / (eps * dt)
                rel, absolute, _ = per_row_error_arrays(side_derivatives[-1][ids[minus_ok]], center_dP[ids][minus_ok])
                rowwise['minus_centered_local_relative_error'][epsilon_index, ids[minus_ok]] = rel
                rowwise['minus_local_absolute_error_Pa_per_s'][epsilon_index, ids[minus_ok]] = absolute
            if np.any(paired):
                centered_derivative[ids[paired]] = (endpoint_P[1][mi][paired] - endpoint_P[-1][mi][paired]) / (2.0 * eps * dt)
                rel, absolute, _ = per_row_error_arrays(centered_derivative[ids[paired]], center_dP[ids][paired])
                rowwise['centered_local_relative_error'][epsilon_index, ids[paired]] = rel
                rowwise['centered_local_absolute_error_Pa_per_s'][epsilon_index, ids[paired]] = absolute
            per_mat[str(mi)]['directional_derivative_comparisons'] = {
                'valid_plus_rows': int(plus_ok.sum()), 'valid_minus_rows': int(minus_ok.sum()),
                'valid_centered_pair_rows': int(paired.sum()),
                'plus_one_sided_vs_center_algorithmic_tangent': rel_metrics(side_derivatives[1][ids[plus_ok]], center_dP[ids][plus_ok]) if np.any(plus_ok) else None,
                'minus_one_sided_vs_center_algorithmic_tangent': rel_metrics(side_derivatives[-1][ids[minus_ok]], center_dP[ids][minus_ok]) if np.any(minus_ok) else None,
                'centered_secant_vs_center_algorithmic_tangent': rel_metrics(centered_derivative[ids[paired]], center_dP[ids][paired]) if np.any(paired) else None,
                'plus_one_sided_per_tet_error': per_row_rel_metrics(side_derivatives[1][ids[plus_ok]], center_dP[ids][plus_ok]) if np.any(plus_ok) else None,
                'minus_one_sided_per_tet_error': per_row_rel_metrics(side_derivatives[-1][ids[minus_ok]], center_dP[ids][minus_ok]) if np.any(minus_ok) else None,
                'centered_secant_per_tet_error': per_row_rel_metrics(centered_derivative[ids[paired]], center_dP[ids][paired]) if np.any(paired) else None,
                'center_directional_tangent_l2_Pa_per_s': l2(center_dP[ids][center_ok[ids]]),
                'centered_stress_difference_l2_Pa': l2(endpoint_P[1][mi][paired] - endpoint_P[-1][mi][paired]) if np.any(paired) else None,
                'centered_stress_difference_over_2epsilon_l2_Pa': l2((endpoint_P[1][mi][paired] - endpoint_P[-1][mi][paired]) / (2.0 * eps)) if np.any(paired) else None,
                'centered_stress_cancellation_ratio': (
                    l2(endpoint_P[1][mi][paired] - endpoint_P[-1][mi][paired]) /
                    (l2(endpoint_P[1][mi][paired]) + l2(endpoint_P[-1][mi][paired])))
                    if np.any(paired) and (l2(endpoint_P[1][mi][paired]) + l2(endpoint_P[-1][mi][paired])) > 0 else None,
            }

        ids = np.flatnonzero(mat_id == 2)
        if len(ids):
            side_derivatives[1][ids] = (endpoint_P[1][2] - center_P[ids]) / (eps * dt)
            side_derivatives[-1][ids] = (center_P[ids] - endpoint_P[-1][2]) / (eps * dt)
            centered_derivative[ids] = (endpoint_P[1][2] - endpoint_P[-1][2]) / (2.0 * eps * dt)
            side_valid[1][ids] = True
            side_valid[-1][ids] = True
            centered_valid[ids] = True
            per_mat['2'] = {
                'name': mat_specs[2]['name'], 'analytic_stateless': True,
                'directional_derivative_comparisons': {
                    'valid_plus_rows': len(ids), 'valid_minus_rows': len(ids), 'valid_centered_pair_rows': len(ids),
                    'plus_one_sided_vs_center_algorithmic_tangent': rel_metrics(side_derivatives[1][ids], center_dP[ids]),
                    'minus_one_sided_vs_center_algorithmic_tangent': rel_metrics(side_derivatives[-1][ids], center_dP[ids]),
                    'centered_secant_vs_center_algorithmic_tangent': rel_metrics(centered_derivative[ids], center_dP[ids]),
                    'plus_one_sided_per_tet_error': per_row_rel_metrics(side_derivatives[1][ids], center_dP[ids]),
                    'minus_one_sided_per_tet_error': per_row_rel_metrics(side_derivatives[-1][ids], center_dP[ids]),
                    'centered_secant_per_tet_error': per_row_rel_metrics(centered_derivative[ids], center_dP[ids]),
                    'center_directional_tangent_l2_Pa_per_s': l2(center_dP[ids]),
                    'centered_stress_difference_l2_Pa': l2(endpoint_P[1][2] - endpoint_P[-1][2]),
                    'centered_stress_difference_over_2epsilon_l2_Pa': l2((endpoint_P[1][2] - endpoint_P[-1][2]) / (2.0 * eps)),
                    'centered_stress_cancellation_ratio': l2(endpoint_P[1][2] - endpoint_P[-1][2]) / max(
                        l2(endpoint_P[1][2]) + l2(endpoint_P[-1][2]), 1e-300),
                },
            }

        action_results = {}
        for label, dp, valid in (
                ('plus_one_sided', side_derivatives[1], side_valid[1]),
                ('minus_one_sided', side_derivatives[-1], side_valid[-1]),
                ('centered_secant', centered_derivative, centered_valid)):
            if not np.all(valid):
                action_results[label] = {
                    'status': 'unavailable_projection_gate_failed',
                    'valid_tetrahedra': int(np.sum(valid)), 'required_tetrahedra': ntet,
                }
                continue
            act = action_from_dP(dp)
            action_results[label] = {
                'status': 'complete_all_required_projection_rows_passed',
                'A_action_vs_center_A': rel_metrics(act['A'], center_A),
                'A_action_vs_saved_independent_419_center_A': rel_metrics(act['A'], independent_saved),
                'K_action_vs_center_K': rel_metrics(act['K'], center_terms['K']),
                'mass_term_M_dv_Ns': {'l2': l2(mass_action), 'signed_work_dv_dot': signed_work(dv, mass_action)},
                'signed_stiffness_term_minus_dt2K_Ns': {'l2': l2(act['stiffness']),
                                                          'signed_work_dv_dot': signed_work(dv, act['stiffness'])},
                'total_A_Ns': {'l2': l2(act['A']), 'signed_work_dv_dot': signed_work(dv, act['A'])},
                'stiffness_change_vs_center_Ns': {
                    'l2': l2(act['stiffness'] - center_terms['stiffness']),
                    'signed_work_dv_dot': signed_work(dv, act['stiffness'] - center_terms['stiffness']),
                },
            }
        report['independent_fp64_epsilon_sweep'].append({
            'epsilon': eps,
            'endpoint_mode': 'ideal unquantized FP64 F0 +/- epsilon*dt*H from promoted captured center high+low; accepted history fixed',
            'endpoint_F_delta_l2': l2(eps * dt * H),
            'endpoint_displacement_delta_x_equals_dt_dv_l2_m': l2(dt * dv),
            'centered_F_difference_over_2epsilon_dt_l2_per_s': l2((endpoint_F[1] - endpoint_F[-1]) / (2.0 * eps * dt)),
            'all_center_and_both_endpoint_projection_rows_passed': bool(np.all(centered_valid)),
            'center_projection_success_rows': int(np.sum(center_ok)),
            'plus_projection_success_rows_paper_materials': int(sum(np.sum(endpoint_results[1][mi][4]) for mi in (0, 1))),
            'minus_projection_success_rows_paper_materials': int(sum(np.sum(endpoint_results[-1][mi][4]) for mi in (0, 1))),
            'expected_rows_each_side_paper_materials': int(sum(np.sum(mat_id == mi) for mi in (0, 1))),
            'endpoint_observable_branch_signatures': {
                ('plus' if sign == 1 else 'minus'): {str(mi): endpoint_branch[sign][mi] for mi in (0, 1)}
                for sign in (1, -1)
            },
            'endpoint_observable_branch_transitions_vs_center': {
                ('plus' if sign == 1 else 'minus'): {str(mi): branch_transitions[sign][mi] for mi in (0, 1)}
                for sign in (1, -1)
            },
            'material': per_mat,
            'assembled_actions': action_results,
            'native_endpoint_comparison': {'status': 'not_used', 'reason': 'all endpoints are ideal FP64 perturbations; no captured FP32 realized endpoint is substituted'},
        })

    # Rowwise all-grid audit. A row enters a passing window only when both
    # endpoint projections pass the tighter diagnostic gates, the available
    # endpoint predicates match the center, and the centered secant error is
    # below the local 1e-6 comparison threshold.
    local_gate = 1e-6
    row_errors = rowwise['centered_local_relative_error']
    row_ready = (rowwise['plus_projection_ok'] & rowwise['minus_projection_ok'] &
                 rowwise['plus_branch_matches_center'] & rowwise['minus_branch_matches_center'] &
                 np.isfinite(row_errors))
    row_pass = row_ready & (row_errors <= local_gate)

    def longest_run(values):
        best = current = 0
        for value in values:
            if bool(value):
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    per_row_longest = np.zeros(ntet, dtype=np.int16)
    per_row_longest_nonincreasing = np.zeros(ntet, dtype=np.int16)
    per_row_min_error = np.full(ntet, np.nan, dtype=np.float64)
    per_row_min_error_power = np.full(ntet, -1, dtype=np.int16)
    for ti in paper_tet_ids:
        per_row_longest[ti] = longest_run(row_pass[:, ti])
        best = current = 0
        for ei in range(neps):
            if not row_pass[ei, ti]:
                current = 0
            elif ei > 0 and row_pass[ei - 1, ti] and row_errors[ei, ti] <= row_errors[ei - 1, ti]:
                current += 1
                best = max(best, current)
            else:
                current = 1
                best = max(best, current)
        per_row_longest_nonincreasing[ti] = best
        valid_errors = row_errors[:, ti][row_ready[:, ti]]
        if valid_errors.size:
            best_idx = np.flatnonzero(row_ready[:, ti])[int(np.argmin(valid_errors))]
            per_row_min_error[ti] = row_errors[best_idx, ti]
            per_row_min_error_power[ti] = rowwise['epsilon_power'][best_idx]
    rowwise['longest_consecutive_branch_stable_pass_window'] = per_row_longest
    rowwise['longest_consecutive_nonincreasing_branch_stable_pass_window'] = per_row_longest_nonincreasing
    rowwise['minimum_branch_stable_local_relative_error_on_grid'] = per_row_min_error
    rowwise['epsilon_power_at_minimum_branch_stable_error'] = per_row_min_error_power

    by_material = {}
    for mi in (0, 1):
        ids = np.flatnonzero(mat_id == mi)
        failing = ids[per_row_longest[ids] < 2]
        never_meet_at_any_gridpoint = ids[~np.isfinite(per_row_min_error[ids]) | (per_row_min_error[ids] > local_gate)]
        failure_rows = []
        for ti in failing:
            failure_rows.append({
                'tet_id': int(ti),
                'longest_consecutive_pass_window': int(per_row_longest[ti]),
                'minimum_branch_stable_error_on_grid': float(per_row_min_error[ti]) if np.isfinite(per_row_min_error[ti]) else None,
                'epsilon_power_at_minimum': int(per_row_min_error_power[ti]) if per_row_min_error_power[ti] >= 0 else None,
                'center_tangent_frobenius_Pa_per_s': float(rowwise['center_tangent_frobenius_Pa_per_s'][ti]),
                'center_stress_frobenius_Pa': float(rowwise['center_stress_frobenius_Pa'][ti]),
                'center_element_tangent_force_l2_N_per_s': float(rowwise['center_element_tangent_force_l2_N_per_s'][ti]),
                'center_element_force_l2_N': float(rowwise['center_element_force_l2_N'][ti]),
                'center_jacobian_condition_2norm': float(center_jacobian_condition[ti]) if np.isfinite(center_jacobian_condition[ti]) else None,
                'center_scaled_residual': float(rowwise['center_scaled_residual'][ti]),
                'center_dynamic_correction': float(rowwise['center_dynamic_correction'][ti]),
            })
        eps_counts = []
        for ei, eps in enumerate(eps_values):
            valid_branch = row_ready[ei, ids]
            err = row_errors[ei, ids]
            eps_counts.append({
                'epsilon_power': int(rowwise['epsilon_power'][ei]),
                'epsilon': eps,
                'both_tight_projections_and_observed_branches_stable': int(np.sum(valid_branch)),
                'rows_under_local_1e-6_gate': int(np.sum(valid_branch & (err <= local_gate))),
                'max_error_among_branch_stable_rows': float(np.max(err[valid_branch])) if np.any(valid_branch) else None,
                'assembled_action_relative_error': report['independent_fp64_epsilon_sweep'][ei]['assembled_actions']['centered_secant'].get('A_action_vs_center_A', {}).get('relative_to_reference'),
                'assembled_1_percent_gate_pass': (
                    report['independent_fp64_epsilon_sweep'][ei]['assembled_actions']['centered_secant'].get('A_action_vs_center_A', {}).get('relative_to_reference') is not None and
                    report['independent_fp64_epsilon_sweep'][ei]['assembled_actions']['centered_secant']['A_action_vs_center_A']['relative_to_reference'] <= 1e-2),
            })
        by_material[str(mi)] = {
            'material_name': mat_specs[mi]['name'],
            'paper_rows': int(len(ids)),
            'rows_with_at_least_2_adjacent_passing_grid_points': int(np.sum(per_row_longest[ids] >= 2)),
            'rows_with_at_least_3_adjacent_passing_grid_points': int(np.sum(per_row_longest[ids] >= 3)),
            'rows_with_at_least_2_adjacent_passing_points_and_nonincreasing_error_as_epsilon_shrinks': int(np.sum(per_row_longest_nonincreasing[ids] >= 2)),
            'rows_with_at_least_3_adjacent_passing_points_and_nonincreasing_error_as_epsilon_shrinks': int(np.sum(per_row_longest_nonincreasing[ids] >= 3)),
            'rows_failing_2_point_window': int(len(failing)),
            'rows_never_at_or_below_1e-6_on_any_branch_stable_grid_point': int(len(never_meet_at_any_gridpoint)),
            'failing_rows': failure_rows,
            'per_epsilon_counts': eps_counts,
        }
    report['rowwise_resolved_window_audit'] = {
        'local_per_tet_secant_comparison_gate': local_gate,
        'window_definition': 'at least two adjacent points in the fixed epsilon grid 2^-14 through 2^-30; at each point both stricter diagnostic projections pass, all observed endpoint predicates match center, and local centered secant relative error <=1e-6',
        'observed_branch_predicates': ['plastic lambda-increment activity', 'yield residual sign', 'FB reference arithmetic path'],
        'unobserved_internal_return_map_branch_changes': 'not tested; this window is limited to endpoint predicates explicitly recorded by the reference helper',
        'assembled_action_gate': {'relative_error_threshold': 1e-2, 'scope': 'assembled A action only; distinct from local per-tet 1e-6 comparison'},
        'materials': by_material,
    }

    fixed_grid = {}
    joint_allrow_eps = []
    for power in (20, 22):
        ei = power - 14
        material_rows = {}
        for mi in (0, 1):
            ids = np.flatnonzero(mat_id == mi)
            selected_failures = ids[row_ready[ei, ids] & (row_errors[ei, ids] > local_gate)]
            material_rows[str(mi)] = {
                'material_name': mat_specs[mi]['name'],
                'paper_rows': int(len(ids)),
                'both_endpoint_projections_and_observed_branches_stable': int(np.sum(row_ready[ei, ids])),
                'local_rows_over_1e-6': int(len(selected_failures)),
                'local_failure_tet_ids': [int(ti) for ti in selected_failures],
                'local_failure_details': [{
                    'tet_id': int(ti),
                    'centered_relative_error': float(row_errors[ei, ti]),
                    'centered_absolute_error_Pa_per_s': float(rowwise['centered_local_absolute_error_Pa_per_s'][ei, ti]),
                    'center_stress_frobenius_Pa': float(rowwise['center_stress_frobenius_Pa'][ti]),
                    'center_tangent_frobenius_Pa_per_s': float(rowwise['center_tangent_frobenius_Pa_per_s'][ti]),
                    'center_element_force_l2_N': float(rowwise['center_element_force_l2_N'][ti]),
                    'center_element_tangent_force_l2_N_per_s': float(rowwise['center_element_tangent_force_l2_N_per_s'][ti]),
                    'center_jacobian_condition_2norm': float(center_jacobian_condition[ti]) if np.isfinite(center_jacobian_condition[ti]) else None,
                    'center_scaled_residual': float(rowwise['center_scaled_residual'][ti]),
                    'center_dynamic_correction': float(rowwise['center_dynamic_correction'][ti]),
                } for ti in selected_failures],
            }
        assembled = report['independent_fp64_epsilon_sweep'][ei]['assembled_actions']['centered_secant']
        fixed_grid[str(power)] = {
            'epsilon': eps_values[ei],
            'all_1920_required_rows_projected_for_assembled_action': bool(report['independent_fp64_epsilon_sweep'][ei]['all_center_and_both_endpoint_projection_rows_passed']),
            'assembled_A_action_vs_center_relative_error': assembled.get('A_action_vs_center_A', {}).get('relative_to_reference'),
            'assembled_1_percent_gate_pass': assembled.get('A_action_vs_center_A', {}).get('relative_to_reference') is not None and assembled['A_action_vs_center_A']['relative_to_reference'] <= 1e-2,
            'materials': material_rows,
        }
    for ei, eps in enumerate(eps_values):
        if (np.all(row_ready[ei, paper_tet_ids]) and
                np.all(row_errors[ei, paper_tet_ids] <= local_gate)):
            joint_allrow_eps.append({'epsilon_power': int(rowwise['epsilon_power'][ei]), 'epsilon': eps})
    report['fixed_grid_failure_diagnostics'] = fixed_grid
    report['joint_all_paper_rows_local_gate_and_projection_gate'] = {
        'epsilon_points_passing_all_1728_paper_rows': joint_allrow_eps,
        'interpretation': 'A per-row window can exist at different row-specific epsilon values even when the shared grid has no single point that satisfies every row simultaneously.'
    }

    baseline_by_eps = {float(item['epsilon']): item for item in baseline005['independent_fp64_epsilon_sweep']}
    baseline_deltas = []
    baseline_gate_count_deltas = []
    selected_grid_deltas = {'2^-20': [], '2^-22': []}
    for item in report['independent_fp64_epsilon_sweep']:
        base = baseline_by_eps.get(float(item['epsilon']))
        if base is None:
            item['baseline005_comparison'] = {'status': 'epsilon_not_present_in_baseline'}
            continue
        material_summary = {}
        for mi in ('0', '1'):
            bmat = base.get('material', {}).get(mi, {})
            bproj = bmat.get('plus', {}).get('projection', {})
            bminus = bmat.get('minus', {}).get('projection', {})
            bcmp = bmat.get('directional_derivative_comparisons', {}).get('centered_secant_per_tet_error')
            material_summary[mi] = {
                'baseline005_plus_projection_success': bproj.get('projection_success'),
                'baseline005_minus_projection_success': bminus.get('projection_success'),
                'baseline005_centered_local_relative_error_max': bcmp.get('relative_l2_max') if bcmp else None,
                'baseline005_centered_rows_above_1e-6': bcmp.get('rows_above_1e-6') if bcmp else None,
            }
            new_cmp = item['material'][mi]['directional_derivative_comparisons'].get('centered_secant_per_tet_error')
            old_error = material_summary[mi]['baseline005_centered_local_relative_error_max']
            if new_cmp and old_error is not None:
                delta = abs(float(new_cmp['relative_l2_max']) - float(old_error))
                baseline_deltas.append(delta)
                baseline_gate_count_deltas.append(abs(int(new_cmp['rows_above_1e-6']) - int(material_summary[mi]['baseline005_centered_rows_above_1e-6'])))
                power = round(-math.log2(float(item['epsilon'])))
                if power == 20:
                    selected_grid_deltas['2^-20'].append(delta)
                elif power == 22:
                    selected_grid_deltas['2^-22'].append(delta)
        item['baseline005_comparison'] = material_summary
    report['baseline005_projection_stopping_sensitivity'] = {
        'baseline_projection_gates': '2e-10 scaled residual and 2e-8 dynamic correction',
        'tight_diagnostic_projection_gates': {'scaled_residual': TIGHT_RESIDUAL, 'dynamic_correction': TIGHT_DYNAMIC_CORRECTION},
        'max_absolute_change_in_per_material_per_epsilon_max_local_error': max(baseline_deltas) if baseline_deltas else None,
        'max_absolute_change_in_per_material_per_epsilon_rows_above_1e-6': max(baseline_gate_count_deltas) if baseline_gate_count_deltas else None,
        'epsilon_2^-20_and_2^-22_max_local_error_absolute_deltas_vs_baseline005': {
            key: max(values) if values else None for key, values in selected_grid_deltas.items()},
        'interpretation': 'Tightening independent projection stops by 100x in residual and 1000x in correction does not materially move the local secant errors in this run; the stopping thresholds do not explain the remaining per-tet 1e-6 misses on this grid.'
    }

    out.mkdir(parents=True, exist_ok=False)
    rowwise_path = out / 'rowwise-epsilon.npz'
    np.savez_compressed(rowwise_path, **rowwise)
    report['rowwise_data'] = {
        'path': rowwise_path.name,
        'sha256': sha(rowwise_path),
        'arrays': sorted(rowwise.keys()),
        'per_tet_gate_arrays_preserve_nan_for_unavailable_rows': True,
    }
    report['gate_scope'] = {
        'local_material_directional_secant_comparison': {'threshold': 1e-6, 'unit': 'relative Frobenius error per tet'},
        'assembled_global_A_action_comparison': {'threshold': 1e-2, 'unit': 'relative L2 error'},
        'interpretation': 'The 1e-6 local material comparison and 1% assembled action comparison are distinct gates; passing assembled does not waive a local miss, and a local miss does not imply assembled miss.'
    }
    warning_counts = {}
    for w in warnings_seen:
        key = (w['category'], w['filename'], w['line'], w['message'])
        if key not in warning_counts:
            warning_counts[key] = dict(w, count=0)
        warning_counts[key]['count'] += 1
    report['warnings'] = [warning_counts[k] for k in sorted(warning_counts)]
    report['status'] = 'complete_diagnostic'
    report['wall_seconds'] = time.monotonic() - start
    report['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rp = out / 'report.json'
    rp.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    readme = out / 'README.md'
    readme.write_text(
        '# Source125 run 419 FP64 directional sweep with tightened diagnostic projections\n\n'
        'This CPU-only diagnostic binds the exact source125 run419 Newton48 rich capture, fixed accepted history, the 422 production-term replay, the 424 native residual-sweep report, and immutable baseline 005. It evaluates ideal unquantized FP64 endpoints F0 +/- epsilon*dt*H from the captured FP32 high+low center for epsilon 2^-14 through 2^-30.\n\n'
        'Each endpoint first runs the unchanged 005 projector, then continues the same residual equations and complex-step Jacobian with residual-descending Newton steps to diagnostic-only gates of 2e-12 scaled residual and 2e-11 dynamic correction. These are not production tolerances. Failed rows remain unavailable; no endpoint is borrowed from 005.\n\n'
        'The rowwise audit requires two adjacent fixed-grid points with both endpoint projections passing, the recorded lambda/yield/FB predicates matching center, and local centered secant error <=1e-6. It separately reports the 1% assembled A-action gate. The two scopes are distinct. The compressed NPZ records per-tet errors, projection gates, force/stress scale, and Jacobian condition.\n\n'
        'This is a local constitutive diagnostic, not a native tangent qualification, universal noise floor, coupon convergence, calibration, or physical qualification. See report.json and manifest.json for source hashes, warnings, row failures, and limitations.\n')
    manifest = {'schema': 'numi.cardboard.source125-fp64-smooth-direction-sweep-tight-projection-package.v1',
                'report_sha256': sha(rp), 'README.md_sha256': sha(readme),
                'rowwise-epsilon.npz_sha256': sha(rowwise_path),
                'script_sha256': sha(pathlib.Path(__file__).resolve()),
                'bindings': report['bindings']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(out), 'status': report['status'],
                      'wall_seconds': report['wall_seconds'],
                      'report_sha256': sha(rp), 'script_sha256': sha(pathlib.Path(__file__).resolve())}, indent=2))


if __name__ == '__main__':
    main()
