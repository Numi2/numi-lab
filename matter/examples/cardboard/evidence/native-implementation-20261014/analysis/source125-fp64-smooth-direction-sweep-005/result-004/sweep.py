#!/usr/bin/env python3
"""Independent FP64 material-response sweep for source125 capture 419.

CPU diagnostic only. The native 1/64 endpoints are replayed from their exact
captured high/low F words. Smaller epsilons are unquantized FP64 perturbations
of the captured center F along the captured local F direction, with accepted
history fixed. This distinguishes reference-law curvature from native FP32
endpoint/residual quantization; it is not a native endpoint replay.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True, type=pathlib.Path)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError(f'refusing to overwrite {out}')
    for p in (CAPTURE, RUN / 'prediction.json', RUN / 'receipt.json',
              AUDIT / 'report.json', TERM / 'result/replay-report.json',
              SWEEP / 'report.json'):
        if not p.is_file():
            raise RuntimeError(f'missing bound input {p}')

    prediction = json.loads((RUN / 'prediction.json').read_text())
    receipt = json.loads((RUN / 'receipt.json').read_text())
    native_audit = json.loads((AUDIT / 'report.json').read_text())
    term_report = json.loads((TERM / 'result/replay-report.json').read_text())
    sweep_report = json.loads((SWEEP / 'report.json').read_text())
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
        # The reference law can emit warnings from inactive numpy.where arms.
        # Count them in the output instead of treating the run as warning-free.
        with warnings.catch_warnings(record=True) as caught_projection:
            warnings.simplefilter('always')
            result = verifier.project_centers(ref, fd, p, F, old_state, p26)
        for w in caught_projection:
            warnings_seen.append({'message': str(w.message), 'category': w.category.__name__,
                                  'filename': str(w.filename), 'line': int(w.lineno), 'count': 1})
        return result
    start = time.monotonic()
    report = {
        'schema': 'numi.cardboard.source125-fp64-smooth-direction-sweep.v1',
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
                      'dt_seconds': dt,
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
    for mi in (0, 1):
        ids = np.flatnonzero(mat_id == mi)
        pp, ff, oo, hh = params[ids], F0[ids], old[ids], H[ids]
        z, it, rn, cn, ok, reasons, refinements = project_logged(pp, ff, oo, pp[:, 26])
        center_z[ids] = z
        center_ok[ids] = ok
        center_P[ids] = ref.stress(pp, ff, z)
        center_dP[ids] = ref.tangent(pp, ff, oo, z, hh)
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
    report['center_action_vs_saved_independent_action'] = rel_metrics(
        center_A, independent_saved)

    eps_values = [math.ldexp(1.0, -k) for k in range(14, 31)]
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

    center_branch_cache = {}
    center_branch_signatures = {}
    for mi in (0, 1):
        ids = np.flatnonzero(mat_id == mi)
        center_branch_cache[mi] = verifier.paper_branch(ref, params[ids], old[ids], center_z[ids])
        center_branch_signatures[str(mi)] = observed_branch_signature(
            verifier, ref, params[ids], old[ids], center_z[ids], center_ok[ids])
    report['center_observable_branch_signatures'] = center_branch_signatures

    for eps in eps_values:
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
            if np.any(minus_ok):
                side_derivatives[-1][ids[minus_ok]] = (center_P[ids][minus_ok] - endpoint_P[-1][mi][minus_ok]) / (eps * dt)
            if np.any(paired):
                centered_derivative[ids[paired]] = (endpoint_P[1][mi][paired] - endpoint_P[-1][mi][paired]) / (2.0 * eps * dt)
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
    out.mkdir(parents=True, exist_ok=False)
    rp = out / 'report.json'
    rp.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    readme = out / 'README.md'
    readme.write_text(
        '# Source125 run 419 ideal FP64 directional epsilon sweep\n\n'
        'This CPU-only diagnostic binds the exact source125 run419 Newton48 rich capture, fixed accepted history, the 422 production-term replay, and the 424 native residual-sweep report as provenance. It evaluates ideal unquantized FP64 endpoints F0 +/- epsilon*dt*H from the captured FP32 high+low center, for epsilon 2^-14 through 2^-30. It reports plus, minus, and centered local secants.\n\n'
        'Every independent constitutive projection must pass the existing reference residual and dynamic state-correction gates. Failed rows are retained, and a global assembled action is unavailable unless all required paper rows pass. The assembled action keeps M*dv and the signed -dt^2*K contribution separate. Branch summaries cover endpoint lambda-increment sign, yield-residual sign, and reference FB arithmetic path; they do not reconstruct every internal return-map Newton branch.\n\n'
        'All endpoints are ideal FP64 inputs; no captured FP32 side endpoints are substituted. The 424 result is bound as separate provenance only. This sweep is a local diagnostic, not a native tangent qualification, universal noise floor, coupon convergence, calibration, or physical qualification. See report.json and manifest.json for source hashes, projection outcomes, and limitations.\n')
    manifest = {'schema': 'numi.cardboard.source125-fp64-smooth-direction-sweep-package.v1',
                'report_sha256': sha(rp), 'README.md_sha256': sha(readme),
                'script_sha256': sha(pathlib.Path(__file__).resolve()),
                'bindings': report['bindings']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(out), 'status': report['status'],
                      'wall_seconds': report['wall_seconds'],
                      'report_sha256': sha(rp), 'script_sha256': sha(pathlib.Path(__file__).resolve())}, indent=2))


if __name__ == '__main__':
    main()
