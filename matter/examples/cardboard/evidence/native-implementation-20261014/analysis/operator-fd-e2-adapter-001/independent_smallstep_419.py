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
import pathlib
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
        'schema': 'numi.cardboard.operator-fd-independent-smallstep.v1',
        'status': 'running',
        'evidence_class': 'source-bound CPU constitutive-curvature diagnostic; not solver/physical qualification',
        'scope': 'environment 0, root1/Newton48, fixed captured accepted history, 1920 tetrahedra',
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
                      'eps_native_exact_endpoint': 1.0 / 64.0},
        'operator_action_crosscheck': {},
        'native_endpoint_projection_validation': {},
        'independent_fp64_epsilon_sweep': [],
        'caveats': [
            'Epsilon 1/64 uses the exact captured FP32 high+low side F endpoints; smaller epsilon F endpoints are unquantized FP64 center-F plus/minus epsilon*dt*captured H.',
            'All side projections use the same captured accepted history; independent projection must satisfy the existing residual and dynamic-correction gates.',
            'Only lambda-increment positivity and yield residual are summarized as branch indicators; this is not a complete active-set signature.',
            'Native 424 smaller-epsilon residual sweep is bound separately; no native per-tet material states were captured for those new epsilons.',
            'A finite secant far from the local tangent may reflect nonlinear response or branch/root changes; it is not by itself proof that the algorithmic tangent is wrong.',
            'No convergence, calibration, or physical qualification is claimed.',
        ],
        'warnings': [],
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

    eps_values = [1.0 / 64.0, 1.0 / 1024.0, 1.0 / 16384.0, 1.0 / 262144.0,
                  1.0 / 4194304.0, 1.0 / 16777216.0, 1.0 / 67108864.0,
                  1.0 / 268435456.0, 1.0 / 1073741824.0]
    for eps in eps_values:
        per_mat = {}
        endpoint_P = {1: {}, -1: {}}
        endpoint_info = {1: {}, -1: {}}
        reference_native_pair = math.isclose(eps, 1.0 / 64.0, rel_tol=0.0, abs_tol=0.0)
        for sign in (1, -1):
            Fep = np.empty_like(F0)
            if reference_native_pair:
                side_idx = 2 if sign == 1 else 3
                for i, tet in enumerate(tets):
                    Fep[i] = pair(tet['sides'][side_idx], 'F_local')
            else:
                Fep[:] = F0 + float(sign) * eps * dt * H
            for mi in (0, 1):
                ids = np.flatnonzero(mat_id == mi)
                pp, oo = params[ids], old[ids]
                z, it, rn, cn, ok, reasons, refinements = project_logged(
                    pp, Fep[ids], oo, pp[:, 26])
                stress = ref.stress(pp, Fep[ids], z)
                endpoint_P[sign][mi] = stress
                branch = verifier.paper_branch(ref, pp, oo, z)
                center_branch = verifier.paper_branch(ref, pp, oo, center_z[ids])
                valid = ok & center_ok[ids]
                crossing = valid & (branch['plastic_active_lambda_positive'] !=
                                    center_branch['plastic_active_lambda_positive'])
                # max absolute FB/yield residual is retained as proximity information only.
                per_mat.setdefault(str(mi), {'plus': None, 'minus': None})['plus' if sign == 1 else 'minus'] = {
                    'projection_success': int(ok.sum()), 'rows': int(len(ids)),
                    'max_reference_scaled_residual': float(np.nanmax(rn)) if np.any(np.isfinite(rn)) else None,
                    'max_dynamic_correction': float(np.nanmax(cn)) if np.any(np.isfinite(cn)) else None,
                    'max_updates': int(np.max(it[ok])) if np.any(ok) else None,
                    'failure_examples': [{'tet_id': int(ids[j]), 'reason': reasons[j]}
                                         for j in np.flatnonzero(~ok)[:8]],
                    'lambda_activity_crossing_rows_vs_center': int(crossing.sum()),
                    'lambda_activity_crossing_tet_ids': [int(ids[j]) for j in np.flatnonzero(crossing)[:32]],
                    'minimum_abs_yield_residual_on_valid_rows': float(np.min(np.abs(branch['yield_function_residual_value'][valid]))) if np.any(valid) else None,
                    'max_abs_yield_residual_on_valid_rows': float(np.max(np.abs(branch['yield_function_residual_value'][valid]))) if np.any(valid) else None,
                    'branch_scope': 'lambda increment > 0 and reported Hill yield residual only; not a complete active-set signature',
                }
                endpoint_info[sign][mi] = {'F': Fep[ids], 'z': z, 'ok': ok, 'iterations': it}
            # Glue is analytic and has no internal state.
            ids = np.flatnonzero(mat_id == 2)
            if len(ids):
                spec = mat_specs[2]
                mu = params[ids, 1]
                lam = params[ids, 2]
                endpoint_P[sign][2] = verifier.glue_stress(Fep[ids], mu, lam)

        for mi in (0, 1):
            ids = np.flatnonzero(mat_id == mi)
            for sign, label in ((1, 'plus'), (-1, 'minus')):
                info = endpoint_info[sign][mi]
                state_delta = np.max(np.abs(info['z'] - center_z[ids]), axis=1)
                state_delta = np.where(info['ok'] & center_ok[ids], state_delta, np.nan)
                finite_delta = np.isfinite(state_delta)
                per_mat[str(mi)][label]['max_abs_state_component_delta_vs_center'] = (
                    float(np.max(state_delta[finite_delta])) if np.any(finite_delta) else None)
                per_mat[str(mi)][label]['largest_state_change_tet_ids'] = [
                    int(ids[j]) for j in np.argsort(np.nan_to_num(state_delta, nan=-1.0))[-8:][::-1]
                    if np.isfinite(state_delta[j])]

        dP = np.empty_like(F0)
        all_valid = center_ok.copy()
        for mi in (0, 1):
            ids = np.flatnonzero(mat_id == mi)
            dP[ids] = (endpoint_P[1][mi] - endpoint_P[-1][mi]) / (2.0 * eps * dt)
            all_valid[ids] &= endpoint_info[1][mi]['ok'] & endpoint_info[-1][mi]['ok']
            per_mat[str(mi)]['fd_tangent_vs_center_tangent'] = rel_metrics(
                dP[ids], center_dP[ids])
        ids = np.flatnonzero(mat_id == 2)
        if len(ids):
            dP[ids] = (endpoint_P[1][2] - endpoint_P[-1][2]) / (2.0 * eps * dt)
            per_mat['2'] = {'name': mat_specs[2]['name'],
                            'fd_tangent_vs_center_tangent': rel_metrics(dP[ids], center_dP[ids]),
                            'analytic_stateless': True}
        k_eps = assemble(tets, dP, inverse_rest, rotations, volume, nnode)
        A_eps = mass[:, None] * dv - dt * dt * k_eps
        A_eps[~free] = 0.0
        run_endpoint = None
        if reference_native_pair:
            raw_plus = f32([n['epsilon_pair_residual_impulses_Ns'][2][:3] for n in env0_nodes])
            raw_minus = f32([n['epsilon_pair_residual_impulses_Ns'][3][:3] for n in env0_nodes])
            raw_plus[~free] = 0.0
            raw_minus[~free] = 0.0
            native_fd = -(raw_plus - raw_minus) / (2.0 * eps)
            run_endpoint = {
                'native_side_statuses_from_event': [0, 0],
                'native_fd_vs_independent_fp64_endpoint_action': rel_metrics(native_fd, A_eps),
                'native_fd_vs_center_tangent_action': rel_metrics(native_fd, center_A),
                'native_plus_residual_l2_Ns': l2(raw_plus),
                'native_minus_residual_l2_Ns': l2(raw_minus),
            }
            native_endpoint_stress = {}
            for mi in (0, 1):
                ids = np.flatnonzero(mat_id == mi)
                for s, idx in [('plus', 0), ('minus', 1)]:
                    capP = np.stack([f32(tets[int(i)]['sides'][2 + idx]['P_local_Pa']) for i in ids])
                    valid_side = endpoint_info[1 if s == 'plus' else -1][mi]['ok']
                    native_endpoint_stress[f'{s}_material_{mi}'] = {
                        'valid_projection_rows': int(valid_side.sum()),
                        'excluded_failed_projection_tet_ids': [int(ids[j]) for j in np.flatnonzero(~valid_side)],
                        'stress_metrics_on_valid_rows_Pa': rel_metrics(
                            capP[valid_side], endpoint_P[1 if s == 'plus' else -1][mi][valid_side]),
                    }
            run_endpoint['independent_stress_vs_native_captured_endpoint_Pa'] = native_endpoint_stress
        report['independent_fp64_epsilon_sweep'].append({
            'epsilon': eps,
            'endpoint_mode': 'exact captured FP32 high+low side F' if reference_native_pair else
                             'unquantized FP64 Fcenter +/- epsilon*dt*H along captured direction',
            'all_center_and_endpoint_projection_rows_passed': bool(np.all(all_valid)),
            'material': per_mat,
            'assembled_fp64_A_endpoint_action_vs_center_tangent_action': rel_metrics(A_eps, center_A),
            'assembled_fp64_A_endpoint_action_vs_419_independent_center_action': rel_metrics(A_eps, independent_saved),
            'native_captured_endpoint_comparison': run_endpoint,
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
        '# Source125 run 419 independent FP64 small-step audit\n\n'
        'This CPU-only diagnostic binds the exact run419 source125 Newton48 rich capture, the 422 fresh production operator replay, and the source125 residual endpoint sweep424. It recomputes the independent 15-state response at exact captured 1/64 high/low endpoints, then at smaller unquantized FP64 perturbations with the same accepted history.\n\n'
        'The native 1/64 endpoints provide a direct model check: captured stress is compared with independent projected stress and the endpoint residual secant is compared with a separately assembled FP64 response. Smaller-epsilon values are independent constitutive evaluations, not native outcomes. The separate 424 report is the source-bound native smaller-step record and explicitly does not resolve the native derivative/noise question.\n\n'
        'All per-environment status codes are treated as propagated stage status; they do not mean every tetrahedron independently failed. The private status5 record identifies one first failing tet at a different stage/iteration.\n\n'
        'No tolerance, material parameter, accepted history, or physical state is changed. This does not qualify the failed coupon trajectory or a global tangent. See report.json and manifest.json for hashes and limits.\n')
    manifest = {'schema': 'numi.cardboard.operator-fd-independent-smallstep-package.v1',
                'report_sha256': sha(rp), 'README.md_sha256': sha(readme),
                'script_sha256': sha(pathlib.Path(__file__).resolve()),
                'bindings': report['bindings']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(out), 'status': report['status'],
                      'wall_seconds': report['wall_seconds'],
                      'report_sha256': sha(rp), 'script_sha256': sha(pathlib.Path(__file__).resolve())}, indent=2))


if __name__ == '__main__':
    main()
