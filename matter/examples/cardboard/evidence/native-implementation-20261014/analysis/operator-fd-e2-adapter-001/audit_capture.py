#!/usr/bin/env python3
"""Independent CPU audit of one bounded FEM operator-FD rich capture.

The analyzer is deliberately scoped to the captured 15-state Haj-Ali
bilinear Hill paper programs plus the authored finite-glue neo-Hookean law.
It verifies the exact source-bound capture and computes per-environment
center projection/stress/tangent and assembled A*delta-v independently.
No native execution or production state mutation occurs here.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import pathlib
import sys
import time
import warnings

import numpy as np

BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014')
REF_PATH = BASE / 'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
FD_HELPER_PATH = BASE / 'analysis/source101-capture360-directional-fd-001/compare_directional_fd.py'
GEOM_HELPER_PATH = BASE / 'snapshots/source-boundary-reset-order-101/matter/tools/cardboard_cached_operator_audit.py'
SOURCE125 = BASE / 'snapshots/source-status5-vtccd-diagnostics-125'


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import helper {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f32(values):
    """Round capture decimal spelling back through its original FP32 words."""
    a = np.asarray(values, dtype=np.float32).astype(np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError('captured FP32 field is nonfinite or outside FP32 range')
    return a


def pair(record, key):
    high = np.asarray(record[key], dtype=np.float32)
    low = np.asarray(record.get(key + '_low', np.zeros_like(high)), dtype=np.float32)
    if not np.all(np.isfinite(high)) or not np.all(np.isfinite(low)):
        raise ValueError(f'{key} high/low contains nonfinite FP32 word')
    return high.astype(np.float64) + low.astype(np.float64)


def metrics(actual, reference, mask=None):
    a = np.asarray(actual, dtype=np.float64).reshape(len(actual), -1)
    r = np.asarray(reference, dtype=np.float64).reshape(len(reference), -1)
    if mask is None:
        mask = np.ones(len(a), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    mask &= np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(r), axis=1)
    if not np.any(mask):
        return {'rows': 0, 'absolute_l2_max': None, 'global_l2_relative': None,
                'per_row_relative_max': None, 'reference_l2_max': None}
    d = a[mask] - r[mask]
    an = np.linalg.norm(a[mask], axis=1)
    rn = np.linalg.norm(r[mask], axis=1)
    dn = np.linalg.norm(d, axis=1)
    rel = np.divide(dn, rn, out=np.full_like(dn, np.nan), where=rn > 0)
    finite_rel = rel[np.isfinite(rel)]
    return {
        'rows': int(mask.sum()),
        'absolute_l2_max': float(dn.max()),
        'absolute_l2_median': float(np.median(dn)),
        'maximum_component_absolute': float(np.max(np.abs(d))),
        'global_l2_relative': float(np.linalg.norm(d) / np.linalg.norm(r[mask]))
            if np.linalg.norm(r[mask]) > 0 else None,
        'per_row_relative_p50': float(np.percentile(finite_rel, 50)) if finite_rel.size else None,
        'per_row_relative_p95': float(np.percentile(finite_rel, 95)) if finite_rel.size else None,
        'per_row_relative_max': float(np.max(finite_rel)) if finite_rel.size else None,
        'reference_l2_max': float(rn.max()),
        'zero_reference_rows': int(np.count_nonzero(rn == 0)),
    }


def project_centers(ref, fd, p, F, old, p26):
    """Project, then enforce source-373's residual and dynamic correction gates."""
    z, iterations, scaled_residual, ok, reasons = fd.project_resilient(ref, p, F, old)
    refinement = []
    residual_scale = np.array([1e4] * 6 + [1.0] * 6 + [1e4] * 3, dtype=np.float64)
    dynamic_correction = np.full(len(p), np.nan)
    for refinement_iteration in range(5):
        ids = np.flatnonzero(ok)
        if ids.size == 0:
            break
        r = ref.residual(p[ids], F[ids], old[ids], z[ids])
        try:
            corr = np.linalg.solve(ref.jacobian(p[ids], F[ids], old[ids], z[ids]),
                                   -r[..., None])[..., 0]
        except Exception:
            # Preserve individual failures instead of losing a whole batch.
            valid_ids = []
            valid_corr = []
            valid_residual = []
            for original in ids:
                try:
                    rr = ref.residual(p[original:original + 1], F[original:original + 1],
                                      old[original:original + 1], z[original:original + 1])
                    jj = ref.jacobian(p[original:original + 1], F[original:original + 1],
                                      old[original:original + 1], z[original:original + 1])
                    cc = np.linalg.solve(jj, -rr[..., None])[0, :, 0]
                    valid_ids.append(original)
                    valid_corr.append(cc)
                    valid_residual.append(rr[0])
                except Exception as e:
                    ok[original] = False
                    reasons[original] = f'FP64 correction solve: {type(e).__name__}: {e}'
            ids = np.asarray(valid_ids, dtype=np.int64)
            corr = np.asarray(valid_corr, dtype=np.float64).reshape(-1, 15)
            r = np.asarray(valid_residual, dtype=np.float64).reshape(-1, 15)
            if ids.size == 0:
                break
        state_scale = np.maximum(np.maximum(np.abs(old[ids]), np.abs(z[ids])),
                                 p26[ids, None])
        corr_norm = np.max(np.abs(corr) / state_scale, axis=1)
        dynamic_correction[ids] = corr_norm
        selected_local = np.flatnonzero(corr_norm > 2e-8)
        refinement.append({
            'iteration': refinement_iteration,
            'rows_above_dynamic_correction_gate': int(selected_local.size),
            'maximum_dynamic_scaled_correction': float(corr_norm.max()) if corr_norm.size else None,
        })
        if selected_local.size == 0:
            break
        if refinement_iteration >= 4:
            for local in selected_local:
                original = ids[local]
                ok[original] = False
                reasons[original] = 'bounded FP64 dynamic-correction refinement exhausted'
            break
        selected = ids[selected_local]
        corr_selected = corr[selected_local]
        baseline = np.max(np.abs(r[selected_local]) * residual_scale, axis=1)
        pending = np.ones(len(selected), dtype=bool)
        alpha = np.ones(len(selected), dtype=np.float64)
        next_z = z[selected].copy()
        for backtrack in range(24):
            candidate = z[selected] + alpha[:, None] * corr_selected
            nr = np.max(np.abs(ref.residual(p[selected], F[selected], old[selected], candidate)) *
                        residual_scale, axis=1)
            accepted = pending & np.isfinite(nr) & (nr < baseline)
            next_z[accepted] = candidate[accepted]
            pending[accepted] = False
            if not np.any(pending):
                break
            alpha[pending] *= 0.5
        for local in np.flatnonzero(pending):
            original = selected[local]
            ok[original] = False
            reasons[original] = 'bounded FP64 residual-descent refinement stalled'
        z[selected[~pending]] = next_z[~pending]
        iterations[selected[~pending]] += 1
    # Re-evaluate both gates at the final returned states.
    ids = np.flatnonzero(ok)
    final_scaled_residual = np.full(len(p), np.nan)
    final_dynamic_correction = np.full(len(p), np.nan)
    if ids.size:
        rr = ref.residual(p[ids], F[ids], old[ids], z[ids])
        final_scaled_residual[ids] = np.max(np.abs(rr) * residual_scale, axis=1)
        jj = ref.jacobian(p[ids], F[ids], old[ids], z[ids])
        cc = np.linalg.solve(jj, -rr[..., None])[..., 0]
        ss = np.maximum(np.maximum(np.abs(old[ids]), np.abs(z[ids])), p26[ids, None])
        final_dynamic_correction[ids] = np.max(np.abs(cc) / ss, axis=1)
        bad = (final_scaled_residual[ids] > 2e-10) | (final_dynamic_correction[ids] > 2e-8)
        for local in np.flatnonzero(bad):
            original = ids[local]
            ok[original] = False
            reasons[original] = 'final FP64 residual/correction gate failed'
    return z, iterations, final_scaled_residual, final_dynamic_correction, ok, reasons, refinement


def paper_branch(ref, p, old, z):
    lam_increment = z[:, 12] - old[:, 12]
    q, _ = ref.hill(p, z[:, 6:12])
    yield_value = (1.0 + p[:, 17] * p[:, 19] * p[:, 21] * z[:, 13] / p[:, 10]
                   + p[:, 18] * p[:, 20] * p[:, 22] * z[:, 14] / p[:, 10] - q)
    return {
        'plastic_multiplier_increment': lam_increment,
        'plastic_active_lambda_positive': lam_increment > 0.0,
        'yield_function_residual_value': yield_value,
        'q_hill': q,
    }


def glue_stress(f, mu, lam):
    invt = np.swapaxes(np.linalg.inv(f), -1, -2)
    det = np.linalg.det(f)
    return mu[:, None, None] * (f - invt) + lam[:, None, None] * np.log(det)[..., None, None] * invt


def glue_tangent_action(f, h, mu, lam):
    inv = np.linalg.inv(f)
    invt = np.swapaxes(inv, -1, -2)
    logj = np.log(np.linalg.det(f))
    tr = np.einsum('nij,nji->n', inv, h)
    second = np.swapaxes(h, -1, -2)
    return (mu[:, None, None] * h + (mu - lam * logj)[:, None, None] * (invt @ second @ invt)
            + lam[:, None, None] * tr[:, None, None] * invt)


def glue_tangent_matrix(f, mu, lam):
    n = len(f)
    out = np.empty((n, 9, 9), dtype=np.float64)
    for column in range(9):
        h = np.zeros((n, 3, 3), dtype=np.float64)
        h[:, column // 3, column % 3] = 1.0
        out[:, :, column] = glue_tangent_action(f, h, mu, lam).reshape(n, 9)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-dir', required=True, type=pathlib.Path)
    parser.add_argument('--capture', type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    parser.add_argument('--expected-environments', type=int, default=2)
    parser.add_argument('--expected-source-snapshot', type=pathlib.Path)
    parser.add_argument('--run-sides', action='store_true',
                        help='also independently reproject captured plus/minus endpoints')
    args = parser.parse_args()
    case = args.case_dir.resolve()
    out = args.output.resolve()
    need(not out.exists(), f'refusing to overwrite {out}')
    prediction_path = case / 'prediction.json'
    receipt_path = case / 'receipt.json'
    stderr_path = case / 'stderr.log'
    for pth in (prediction_path, receipt_path, stderr_path):
        need(pth.is_file(), f'missing source-bound run file: {pth}')
    prediction = json.loads(prediction_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    source_root = pathlib.Path(prediction['source_snapshot']).resolve()
    if args.expected_source_snapshot is not None:
        need(source_root == args.expected_source_snapshot.resolve(), 'source snapshot mismatch')
    need(receipt.get('schema') == 'numi.cardboard.native-run-receipt.v2', 'unexpected receipt schema')
    need(receipt.get('execution_started') is True, 'native command never started')
    need(receipt.get('source_inputs_unchanged') is True, 'source inputs changed during native run')
    need(receipt.get('changed_inputs') == [], 'changed input list is nonempty')
    env = prediction.get('environment', {})
    selector = env.get('NM_FEM_OPERATOR_FD_SELECTOR')
    capture_path = args.capture.resolve() if args.capture else pathlib.Path(
        env.get('NM_FEM_OPERATOR_FD_CAPTURE_PATH', '')).resolve()
    need(selector and ':' in selector, 'prediction lacks root:iteration FD selector')
    try:
        selected_root, selected_iteration = (int(v) for v in selector.split(':', 1))
    except Exception as e:
        raise RuntimeError('malformed FD selector in source-bound prediction') from e
    need(capture_path.is_file(), f'missing FD capture: {capture_path}')
    need(env.get('NM_FEM_OPERATOR_FD_CAPTURE_PATH') is not None,
         'capture path is not bound in source prediction')
    if args.capture:
        need(capture_path == pathlib.Path(env['NM_FEM_OPERATOR_FD_CAPTURE_PATH']).resolve(),
             'explicit capture path differs from sealed environment')
    need(env.get('NM_FEM_COUPLED_CAPTURE_SELECTOR') is None,
         'coupled capture selector conflicts with bounded FD readback')
    need(env.get('NM_FEM_REGIONAL_RAP_CAPTURE_ONLY') in (None, '0'),
         'RAP-only capture cannot provide finite-difference payload')
    source_runtime = source_root / 'matter/src/runtime.mm'
    need(source_runtime.is_file(), 'source snapshot runtime.mm absent')

    stderr_lines = stderr_path.read_text(errors='replace').splitlines()
    events = []
    for line in stderr_lines:
        if line.startswith('fem_operator_fd_iteration='):
            events.append(json.loads(line.split('=', 1)[1]))
    selected_events = [e for e in events if (e.get('root'), e.get('newton_iteration')) ==
                       (selected_root, selected_iteration)]
    need(len(selected_events) == 1, 'source stderr must contain exactly one selected FD event')
    event = selected_events[0]
    need(event.get('production_inputs_bitwise_unchanged') is True,
         'native capture mutation guard failed')

    records = []
    with capture_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f'invalid JSONL at line {line_number}: {e}') from e
    need(records and records[0].get('record') == 'header', 'missing capture header')
    header = records[0]
    need(records[-1].get('record') == 'integrity', 'missing terminal integrity record')
    need(header.get('production_inputs_bitwise_unchanged') is True and
         records[-1].get('production_inputs_bitwise_unchanged') is True,
         'capture input mutation guard failed')
    need((header.get('root'), header.get('newton_iteration')) ==
         (selected_root, selected_iteration), 'capture selection differs from sealed selector')
    env_count = int(header['environment_count'])
    nnode = int(header['node_count_per_environment'])
    ntet = int(header['tetrahedron_count_per_environment'])
    need(env_count == args.expected_environments and env_count == len(event['environments']),
         f'expected E={args.expected_environments}, capture/event disagree')
    need(env_count <= 4 and nnode <= 4096 and ntet <= 8192,
         'capture dimensions exceed source125 FD scope')
    need(header['schema'] == 'numi.cardboard.fem-newton-capture.v1', 'unsupported capture schema')
    need(header['direction_contract']['operator'] ==
         'A*delta_v=M*delta_v-dt^2*K(delta_v)', 'operator contract mismatch')
    need(header['direction_contract']['centered_difference'] ==
         '-(R_plus-R_minus)/(2*epsilon)', 'FD sign/scale contract mismatch')
    need(int(receipt.get('exit_code', -1)) in (0, 1, 2), 'unexpected native exit status')
    need(receipt.get('timed_out') is False, 'native run timed out; capture cannot represent terminal run')
    need(prediction.get('executed_build_verified') is True, 'run is not fresh-build source bound')
    need(prediction.get('source_snapshot') == str(source_root), 'prediction source path mismatch')
    need(int(header['world_fingerprint_hex'], 16) == int(event['world_fingerprint']),
         'world fingerprint differs between capture and event')
    need(int(header['execution_fingerprint_hex'], 16) == int(event['execution_fingerprint']),
         'execution fingerprint differs between capture and event')
    need(header['dt_seconds_by_environment'] and len(header['dt_seconds_by_environment']) == env_count,
         'per-environment dt missing')
    need(len(header['parameters_by_environment']) == env_count,
         'parameter tables are not environment-major')
    material_specs = {int(m['index']): m for m in header['materials']}
    need(len(material_specs) == len(header['materials']), 'duplicate material index')
    for e in range(env_count):
        need(len(header['parameters_by_environment'][e]) ==
             len(header['parameters_by_environment'][0]), 'parameter table lengths differ by environment')

    node_groups = [[] for _ in range(env_count)]
    tet_groups = [[] for _ in range(env_count)]
    for record in records[1:-1]:
        if record.get('record') == 'node':
            e = int(record['environment'])
            need(0 <= e < env_count, 'node environment out of range')
            node_groups[e].append(record)
        elif record.get('record') == 'tetrahedron':
            e = int(record['environment'])
            need(0 <= e < env_count, 'tetrahedron environment out of range')
            tet_groups[e].append(record)
        else:
            raise RuntimeError(f'unknown rich capture record: {record.get("record")}')
    for e in range(env_count):
        node_groups[e].sort(key=lambda r: int(r['node_id']))
        tet_groups[e].sort(key=lambda r: int(r['tet_id']))
        need(len(node_groups[e]) == nnode and
             [int(r['node_id']) for r in node_groups[e]] == list(range(nnode)),
             f'environment {e} node coverage is incomplete/duplicated')
        need(len(tet_groups[e]) == ntet and
             [int(r['tet_id']) for r in tet_groups[e]] == list(range(ntet)),
             f'environment {e} tet coverage is incomplete/duplicated')

    # Every point-dependent field is rounded back to its native FP32 value
    # before widening into the independent FP64 equations.
    ref = load_module(REF_PATH, 'bilinear15_reference_e2')
    fd = load_module(FD_HELPER_PATH, 'capture_helpers_e2')
    geo = load_module(GEOM_HELPER_PATH, 'geometry_helpers_e2')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        'schema': 'numi.cardboard.operator-fd-e2-independent-audit.v1',
        'status': 'running',
        'evidence_class': 'source-bound CPU diagnostic; not convergence or physical qualification',
        'selection': {'root': selected_root, 'newton_iteration': selected_iteration,
                      'environment_count': env_count, 'node_count_per_environment': nnode,
                      'tetrahedron_count_per_environment': ntet},
        'bindings': {
            'capture_path': str(capture_path), 'capture_sha256': sha(capture_path),
            'case_dir': str(case), 'prediction_sha256': sha(prediction_path),
            'receipt_sha256': sha(receipt_path), 'stderr_sha256': sha(stderr_path),
            'source_snapshot': str(source_root), 'runtime_mm_sha256': sha(source_runtime),
            'independent_reference_sha256': sha(REF_PATH),
            'capture_helpers_sha256': sha(FD_HELPER_PATH),
            'geometry_helpers_sha256': sha(GEOM_HELPER_PATH),
            'source_build_provenance': prediction.get('build_provenance'),
            'native_exit_code': receipt.get('exit_code'),
        },
        'input_contract': {
            'FP32_restore': 'all captured parameters, states, F high/low, directions, masses, inverse-rest, frame quaternion, and volume round through float32 then widen',
            'F': 'center and side local F equals separately captured FP32 high plus FP32 low, widened before summation',
            'accepted_history': 'center and all side references hold the captured accepted state fixed',
            'rate': 'captured maximum rate exponent must be zero; independent 15-state law is rate independent',
            'FD_pairs': 'two captured epsilon pairs; requested and realized directions reported separately',
            'assembly': 'independent 15-state tangent + analytic finite-glue tangent, captured frame/rest geometry, mass and unchanged free-node mask',
        },
        'material_population': [],
        'environments': [],
        'warnings': [],
        'limits': [
            'The rich capture is a single selected Newton linearization, not a complete solve.',
            'The independent model is a handwritten FP64 transcription of the captured 15-state paper law and analytic finite-glue law.',
            'Branch crossing, near-zero directional actions, or failed independent projection are reported separately and are not passed by a relative metric.',
            'This CPU audit does not alter physical equations, source parameters, solver tolerances, or accepted history.',
        ],
    }
    warnings_seen = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for e in range(env_count):
            nodes = node_groups[e]
            tets = tet_groups[e]
            dv = f32([r['correction_delta_v_m_s'] for r in nodes])
            mass = f32([r['center']['mass_kg'] for r in nodes])
            free = np.array([float(f32([r['center']['constraint_tag']])[0]) == 0.0
                             for r in nodes], dtype=bool)
            dt = float(f32([header['dt_seconds_by_environment'][e]])[0])
            need(dt > 0 and math.isfinite(dt), f'environment {e} has invalid dt')
            ptab = f32(header['parameters_by_environment'][e])
            tet_ids = np.array([int(t['material_index']) for t in tets], dtype=np.int32)
            F = np.stack([pair(t['center'], 'F_local') for t in tets])
            H_reported = np.stack([f32(t['center']['F_direction_local_per_s']) for t in tets])
            inverse_rest = np.stack([f32(t['inverse_rest_rows_per_m']) for t in tets])
            volume = f32([t['rest_volume_m3'] for t in tets])
            frame_q = np.stack([f32(t['material_frame_quaternion_xyzw']) for t in tets])
            rotations = []
            for t, q in zip(tets, frame_q):
                frame_active = int(f32([t['center']['dt_seconds_bulk_modulus_kind_frame_active'][3]])[0]) == 1
                rotations.append(geo.rotation_from_xyzw(q) if frame_active else np.eye(3))
            rotations = np.asarray(rotations)
            H_from_nodes = np.empty((ntet, 3, 3), dtype=np.float64)
            for i, t in enumerate(tets):
                conn = np.asarray(t['nodes'], dtype=np.int64)
                need(np.all((0 <= conn) & (conn < nnode)), 'tet has invalid node index')
                local = np.column_stack((dv[conn[1]] - dv[conn[0]],
                                         dv[conn[2]] - dv[conn[0]],
                                         dv[conn[3]] - dv[conn[0]])) @ inverse_rest[i] @ rotations[i]
                H_from_nodes[i] = local
            h_metric = metrics(H_reported, H_from_nodes)
            need(h_metric['maximum_component_absolute'] <= 2e-5 *
                 max(1.0, float(np.max(np.abs(H_from_nodes)))),
                 f'environment {e} captured local direction disagrees with node gather')
            p_by_tet = []
            state_by_tet = []
            for t in tets:
                mi = int(t['material_index'])
                need(mi in material_specs, f'unknown material index {mi}')
                spec = material_specs[mi]
                po = int(spec['parameter_offset'])
                pc = len(spec['parameter_names'])
                need(po >= 0 and po + pc <= len(ptab), 'material parameter slice out of bounds')
                need(int(t['material_parameter_count']) == pc, 'tet parameter count differs from header')
                p_by_tet.append(ptab[po:po + pc])
                state_by_tet.append(int(t['material_state_count']))
            dP = np.full((ntet, 3, 3), np.nan)
            C = np.full((ntet, 9, 9), np.nan)
            Pref = np.full((ntet, 3, 3), np.nan)
            candidate_state_error = []
            projection_rows = []
            branch_rows = []
            material_summary = {}
            for mi in sorted(set(tet_ids.tolist())):
                ids = np.flatnonzero(tet_ids == mi)
                spec = material_specs[int(mi)]
                pname = spec['name']
                state_count = state_by_tet[int(ids[0])]
                need(all(state_by_tet[int(i)] == state_count for i in ids),
                     'one material index maps to multiple state strides')
                need(all(tets[int(i)]['material_name'] == pname for i in ids),
                     'material index/name mismatch across tets')
                pp = np.stack([p_by_tet[int(i)] for i in ids])
                old = np.stack([f32(tets[int(i)]['center']['accepted_state']) for i in ids])
                zcap = np.stack([pair(tets[int(i)]['center'], 'projected_state') for i in ids])
                pcap = np.stack([f32(tets[int(i)]['center']['P_local_Pa']) for i in ids])
                tcap = np.stack([f32(tets[int(i)]['center']['tangent_local_Pa_per_s']) for i in ids])
                H = H_reported[ids]
                if pname.startswith('hajali2009_'):
                    need(state_count == 15 and pp.shape[1] == 28,
                         'unsupported Haj-Ali 15-state parameter/state layout')
                    zref, it, rn, cn, ok, reasons, refinements = project_centers(
                        ref, fd, pp, F[ids], old, pp[:, 26])
                    pref_cap = ref.stress(pp, F[ids], zcap)
                    tangent_cap = ref.tangent(pp, F[ids], old, zcap, H)
                    tangent_ref, _, _ = ref.project(pp, F[ids], old, maxiter=80)
                    tangent_projected = ref.tangent(pp, F[ids], old, tangent_ref, H)
                    tangent_matrix = np.empty((len(ids), 9, 9), dtype=np.float64)
                    for col in range(9):
                        basis = np.zeros((len(ids), 3, 3), dtype=np.float64)
                        basis[:, col // 3, col % 3] = 1.0
                        tangent_matrix[:, :, col] = ref.tangent(pp, F[ids], old, zcap, basis).reshape(len(ids), 9)
                    branch = paper_branch(ref, pp, old, zcap)
                    captured_lambdas = np.array([
                        (tets[int(i)]['center'].get('hill_q_lambda_a_b') or [None, None])[1]
                        for i in ids], dtype=object)
                    captured_lambda_known = np.array([
                        isinstance(value, (int, float)) and math.isfinite(float(value))
                        for value in captured_lambdas], dtype=bool)
                    captured_lambda_active = np.array([
                        float(value) > 0.0 if known else False
                        for value, known in zip(captured_lambdas, captured_lambda_known)], dtype=bool)
                    candidate_state_error.append((ids, zcap - zref))
                    projection_rows.append((ids, rn, cn, ok, it, reasons))
                    branch_rows.append((ids, branch))
                    dP[ids] = tangent_cap
                    C[ids] = tangent_matrix
                    Pref[ids] = pref_cap
                    material_summary[str(mi)] = {
                        'name': pname, 'family': 'independent_15_state_hill_bilinear',
                        'projection_status': 'pass' if bool(np.all(ok)) else 'incomplete',
                        'rows': int(len(ids)), 'independent_projection_success': int(np.sum(ok)),
                        'independent_projection_failures': [
                            {'tet_id': int(tets[int(ids[j])]['tet_id']), 'reason': reasons[j]}
                            for j in np.flatnonzero(~ok)[:16]],
                        'reference_projected_residual_scaled_max': float(np.nanmax(rn)) if np.any(np.isfinite(rn)) else None,
                        'reference_dynamic_correction_scaled_max': float(np.nanmax(cn)) if np.any(np.isfinite(cn)) else None,
                        'maximum_reference_newton_updates': int(np.max(it[ok])) if np.any(ok) else None,
                        'center_capture_candidate_vs_independent_candidate': metrics(zcap, zref, ok),
                        'production_local_P_vs_independent_at_captured_candidate_Pa': metrics(pcap, pref_cap),
                        'production_local_tangent_action_vs_independent_at_captured_candidate_Pa_per_s': metrics(tcap, tangent_cap),
                        'production_local_tangent_action_vs_independent_at_reference_projected_candidate_Pa_per_s': metrics(tcap, tangent_projected),
                        'center_branch': {
                            'captured_lambda_available_rows': int(captured_lambda_known.sum()),
                            'captured_lambda_increment_positive': int(np.sum(captured_lambda_active)),
                            'reference_lambda_increment_positive': int(np.sum(branch['plastic_active_lambda_positive'])),
                            'captured_vs_reference_lambda_active_mismatch_on_available_rows': int(np.sum(
                                captured_lambda_known &
                                (captured_lambda_active != branch['plastic_active_lambda_positive']))),
                            'branch_scope': 'only the captured/reference plastic-multiplier activity predicate is adjudicated; smooth FB/yield proximity is reported numerically, not called a globally smooth branch',
                            'minimum_absolute_reference_yield_value': float(np.min(np.abs(branch['yield_function_residual_value']))),
                        },
                        'dynamic_correction_refinement': refinements,
                    }
                elif 'finite_glue_elastic' in pname:
                    need(state_count == 0 and pp.shape[1] == 3,
                         'unsupported finite-glue state/parameter layout')
                    mu, lam = pp[:, 1], pp[:, 2]
                    pref_cap = glue_stress(F[ids], mu, lam)
                    tangent_cap = glue_tangent_action(F[ids], H, mu, lam)
                    tangent_matrix = glue_tangent_matrix(F[ids], mu, lam)
                    dP[ids] = tangent_cap
                    C[ids] = tangent_matrix
                    Pref[ids] = pref_cap
                    material_summary[str(mi)] = {
                        'name': pname, 'family': 'independent_analytic_finite_glue_neo_hookean',
                        'projection_status': 'not_applicable_stateless_analytic',
                        'rows': int(len(ids)), 'production_local_P_vs_independent_Pa': metrics(
                            np.stack([f32(tets[int(i)]['center']['P_local_Pa']) for i in ids]), pref_cap),
                        'production_local_tangent_action_vs_independent_Pa_per_s': metrics(
                            np.stack([f32(tets[int(i)]['center']['tangent_local_Pa_per_s']) for i in ids]), tangent_cap),
                    }
                else:
                    raise RuntimeError(f'unsupported captured material family: {pname}')

            # Independent element force and assembled action; same canonical tet/node order.
            element_action = np.zeros((ntet, 4, 3), dtype=np.float64)
            assembled = np.zeros((nnode, 3), dtype=np.float64)
            nominal_h_from_C = np.einsum('nij,nj->ni', C, H_reported.reshape(ntet, 9)).reshape(ntet, 3, 3)
            for ti, t in enumerate(tets):
                world_dP = nominal_h_from_C[ti] @ rotations[ti].T
                g = -float(volume[ti]) * world_dP @ inverse_rest[ti].T
                element_action[ti, 1:] = g.T
                element_action[ti, 0] = -np.sum(g.T, axis=0)
                for slot, node_id in enumerate(t['nodes']):
                    assembled[int(node_id)] += element_action[ti, slot]
            independent_A_dv = mass[:, None] * dv - dt * dt * assembled
            independent_A_dv[~free] = 0.0
            native_A_dv = f32([r['production_A_delta_v_impulse_Ns'][:3] for r in nodes])
            native_A_dv[~free] = 0.0
            native_residual = f32([r['center_residual_impulse_Ns'][:3] for r in nodes])
            native_residual[~free] = 0.0
            captured_element = np.stack([
                f32(t['center']['operator_force_node_world_N_per_s']) for t in tets])
            captured_tangent_action = np.stack([
                f32(t['center']['tangent_local_Pa_per_s']) for t in tets])

            pair_reports = []
            for pair_id in range(2):
                epsilon = float(f32([header['epsilon_values'][pair_id]])[0])
                need(epsilon > 0 and math.isfinite(epsilon), 'invalid captured epsilon')
                plus_key = f'epsilon_pair_{pair_id}_plus'
                minus_key = f'epsilon_pair_{pair_id}_minus'
                rplus = f32([r['epsilon_pair_residual_impulses_Ns'][2 * pair_id][:3]
                             for r in nodes])
                rminus = f32([r['epsilon_pair_residual_impulses_Ns'][2 * pair_id + 1][:3]
                              for r in nodes])
                rplus[~free] = 0.0
                rminus[~free] = 0.0
                fd_action = -(rplus - rminus) / (2.0 * epsilon)
                realized_plus_v = f32([r['epsilon_pair_realized_velocity_delta_m_s'][2 * pair_id]
                                       for r in nodes])
                realized_minus_v = f32([r['epsilon_pair_realized_velocity_delta_m_s'][2 * pair_id + 1]
                                        for r in nodes])
                realized_dv = (realized_plus_v - realized_minus_v) / (2.0 * epsilon)
                realized_h = np.empty((ntet, 3, 3), dtype=np.float64)
                side_status = {'plus': [], 'minus': []}
                for ti, t in enumerate(tets):
                    plus = t['sides'][2 * pair_id]
                    minus = t['sides'][2 * pair_id + 1]
                    fp = pair(plus, 'F_local')
                    fm = pair(minus, 'F_local')
                    realized_h[ti] = (fp - fm) / (2.0 * epsilon * dt)
                    for label, side in (('plus', plus), ('minus', minus)):
                        diag = f32(side['projection_diagnostics'])
                        fs = f32(side['flags_status_determinant'])
                        side_status[label].append({'projection': diag.tolist(),
                                                   'status_determinant': fs.tolist()})
                actual_h = realized_h
                realized_dP = np.einsum('nij,nj->ni', C, actual_h.reshape(ntet, 9)).reshape(ntet, 3, 3)
                realized_element = np.zeros((ntet, 4, 3), dtype=np.float64)
                realized_gather = np.zeros((nnode, 3), dtype=np.float64)
                for ti, t in enumerate(tets):
                    world_dP = realized_dP[ti] @ rotations[ti].T
                    g = -float(volume[ti]) * world_dP @ inverse_rest[ti].T
                    realized_element[ti, 1:] = g.T
                    realized_element[ti, 0] = -np.sum(g.T, axis=0)
                    for slot, node_id in enumerate(t['nodes']):
                        realized_gather[int(node_id)] += realized_element[ti, slot]
                realized_A_dv = mass[:, None] * realized_dv - dt * dt * realized_gather
                realized_A_dv[~free] = 0.0
                # Optional re-projection uses the fixed accepted history and exact captured high/low side F.
                endpoint_projection = None
                if args.run_sides:
                    endpoint_projection = {}
                    for sign, side_index in (('plus', 2 * pair_id), ('minus', 2 * pair_id + 1)):
                        paper_ids = np.flatnonzero(tet_ids < 2)
                        for mi in (0, 1):
                            idx = paper_ids[tet_ids[paper_ids] == mi]
                            if idx.size == 0:
                                continue
                            spec = material_specs[mi]
                            po = int(spec['parameter_offset'])
                            pc = len(spec['parameter_names'])
                            pp = np.tile(ptab[po:po + pc], (len(idx), 1))
                            old = np.stack([f32(tets[int(i)]['center']['accepted_state']) for i in idx])
                            fs = np.stack([pair(tets[int(i)]['sides'][side_index], 'F_local') for i in idx])
                            zcap_side = np.stack([pair(tets[int(i)]['sides'][side_index], 'projected_state') for i in idx])
                            zref, it, rn, cn, ok, reasons, refinements = project_centers(
                                ref, fd, pp, fs, old, pp[:, 26])
                            pref_side = ref.stress(pp, fs, zref)
                            pcap_side = np.stack([f32(tets[int(i)]['sides'][side_index]['P_local_Pa']) for i in idx])
                            center_zcap = np.stack([pair(tets[int(i)]['center'], 'projected_state')
                                                    for i in idx])
                            center_branch = paper_branch(ref, pp, old, center_zcap)
                            endpoint_branch = paper_branch(ref, pp, old, zref)
                            branch_valid = ok & np.all(np.isfinite(zref), axis=1)
                            branch_summary = {
                                'center_lambda_positive_rows': int(np.sum(
                                    center_branch['plastic_active_lambda_positive'])),
                                'endpoint_lambda_positive_rows': int(np.sum(
                                    endpoint_branch['plastic_active_lambda_positive'] & branch_valid)),
                                'comparable_rows': int(branch_valid.sum()),
                                'branch_crossing_rows': int(np.sum(branch_valid & (
                                    center_branch['plastic_active_lambda_positive'] !=
                                    endpoint_branch['plastic_active_lambda_positive']))),
                                'branch_crossing_tet_ids': [
                                    int(tets[int(idx[j])]['tet_id'])
                                    for j in np.flatnonzero(branch_valid & (
                                        center_branch['plastic_active_lambda_positive'] !=
                                        endpoint_branch['plastic_active_lambda_positive']))[:32]],
                                'branch_predicate': 'reference plastic multiplier increment > 0; other branch predicates are not inferred',
                            }
                            endpoint_projection[f'{sign}_material_{mi}'] = {
                                'rows': int(len(idx)), 'reference_projection_success': int(ok.sum()),
                                'reference_projection_failure_examples': [
                                    {'tet_id': int(tets[int(idx[j])]['tet_id']), 'reason': reasons[j]}
                                    for j in np.flatnonzero(~ok)[:8]],
                                'projected_state_vs_capture': metrics(zcap_side, zref, ok),
                                'stress_vs_independent_reference_Pa': metrics(pcap_side, pref_side, ok),
                                'maximum_reference_scaled_residual': float(np.nanmax(rn)) if np.any(np.isfinite(rn)) else None,
                                'maximum_reference_dynamic_correction': float(np.nanmax(cn)) if np.any(np.isfinite(cn)) else None,
                                'dynamic_refinement': refinements,
                                'plastic_multiplier_branch_comparison': branch_summary,
                            }
                # Captured FD comparison uses nominal epsilon and exact production residual endpoints.
                stage_metrics = event['environments'][e]['epsilon_pairs'][pair_id]
                status_codes = stage_metrics.get('stage_status_codes_plus_minus')
                fd_gate = {
                    'stage_statuses_clear': status_codes == [0, 0],
                    'operator_status_clear': stage_metrics.get('operator_status_code') == 0,
                    'input_integrity': stage_metrics.get('input_integrity_flag') == 1,
                    'direction_nonzero': stage_metrics.get('direction_nonzero') is True,
                    'endpoint_residual_metrics_finite': all(
                        stage_metrics.get(k) is not None and math.isfinite(float(stage_metrics[k]))
                        for k in ('residual_plus_l2', 'residual_minus_l2')),
                }
                fd_gate['comparison_admissible'] = all(fd_gate.values())
                side_gate_counts = {}
                for side_name, rows in side_status.items():
                    # flags_status_determinant is [projection_flags,
                    # terminal_failure, diagnostic_status_code, det(F)].
                    # Keep the status code distinct from the terminal-failure
                    # enum; the latter is not a status code.
                    codes = [row['status_determinant'][2] for row in rows
                             if len(row['status_determinant']) > 2]
                    side_gate_counts[side_name] = {
                        'captured_side_rows': len(rows),
                        'status_code_counts': {str(code): codes.count(code) for code in sorted(set(codes))},
                        'first_nonzero_status_examples': [
                            {'tet_id': int(tets[i]['tet_id']), 'status_determinant': rows[i]['status_determinant']}
                            for i, code in enumerate(codes) if code != 0][:8],
                    }
                pair_reports.append({
                    'pair': pair_id, 'epsilon': epsilon,
                    'finite_difference_admission': fd_gate,
                    'native_fd_vs_independent_A_nominal': metrics(fd_action, independent_A_dv, free)
                        if fd_gate['comparison_admissible'] else None,
                    'native_fd_vs_independent_A_realized_direction': metrics(fd_action, realized_A_dv, free)
                        if fd_gate['comparison_admissible'] else None,
                    'native_fd_vs_captured_native_A_nominal': metrics(fd_action, native_A_dv, free)
                        if fd_gate['comparison_admissible'] else None,
                    'independent_A_realized_vs_nominal': metrics(realized_A_dv, independent_A_dv, free),
                    'realized_velocity_direction_vs_requested': metrics(
                        realized_dv, dv, free),
                    'realized_local_F_direction_vs_reported': metrics(actual_h, H_reported),
                    'realized_local_F_perturbation_roundoff_max': float(np.max(np.abs(
                        actual_h - H_reported))) if np.any(np.isfinite(actual_h)) else None,
                    'plus_minus_captured_projection_status': side_gate_counts,
                    'independent_side_projection': endpoint_projection,
                    'native_FD_stage_metrics': stage_metrics,
                    'branch_crossing': 'captured side branch auxiliaries may be null; centered FD is not admitted if native side status fails. Independent side projections are optional and are reported separately.',
                })

            # Branch labels and exact data arrays aid targeted independent review.
            np.savez_compressed(out / f'environment-{e}.npz',
                correction_delta_v_m_s=dv, mass_kg=mass, free_node_mask=free,
                center_residual_impulse_Ns=native_residual,
                native_production_A_delta_v_impulse_Ns=native_A_dv,
                independent_A_delta_v_impulse_Ns=independent_A_dv,
                native_center_element_tangent_action_local_Pa_per_s=captured_tangent_action,
                independent_element_tangent_action_local_Pa_per_s=dP,
                native_element_operator_force_node_world_N_per_s=captured_element,
                independent_element_operator_force_node_world_N_per_s=element_action,
                material_index=tet_ids, F_local_center=F, F_local_direction_reported_per_s=H_reported,
                F_local_direction_reconstructed_per_s=H_from_nodes,
                inverse_rest_rows_per_m=inverse_rest, material_frame_rotation=rotations,
                rest_volume_m3=volume)
            report['environments'].append({
                'environment': e, 'dt_seconds': dt, 'free_nodes': int(free.sum()),
                'tetrahedra': int(ntet), 'materials': material_summary,
                'captured_direction_consistency': h_metric,
                'independent_assembled_operator_action': {
                    'native_A_delta_v_vs_independent_A_delta_v': metrics(native_A_dv, independent_A_dv, free),
                    'native_captured_element_force_vs_independent_element_force_N_per_s': metrics(
                        captured_element, element_action),
                    'free_rhs_impulse_l2': float(np.linalg.norm(native_residual[free])),
                    'independent_A_delta_v_l2': float(np.linalg.norm(independent_A_dv[free])),
                    'native_A_delta_v_l2': float(np.linalg.norm(native_A_dv[free])),
                    'operator_sign_contract': 'M*delta_v - dt^2*K(delta_v)',
                },
                'finite_difference_pairs': pair_reports,
            })
        warnings_seen = sorted({
            (str(w.message), str(pathlib.Path(w.filename).resolve()), int(w.lineno))
            for w in caught
        })

    report['warnings'] = [
        {'message': message, 'filename': filename, 'line': line, 'count': sum(
            1 for w in caught if str(w.message) == message and
            str(pathlib.Path(w.filename).resolve()) == filename and int(w.lineno) == line)}
        for message, filename, line in warnings_seen]
    report['status'] = ('complete_diagnostic' if all(
        all(m.get('projection_status') in ('pass', 'not_applicable_stateless_analytic')
            for m in env['materials'].values()) for env in report['environments'])
        else 'complete_with_independent_projection_failures')
    report['wall_seconds'] = time.monotonic() - started
    report['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True,
                                               allow_nan=False) + '\n')
    (out / 'README.md').write_text(
        '# Two-environment operator-FD independent audit\n\n'
        f'Input capture SHA256: `{report["bindings"]["capture_sha256"]}`.\n\n'
        'This is a source-bound CPU diagnostic of one Newton linearization. It reconstructs '
        'the independent 15-state paper and finite-glue tangent actions, assembles the '
        'captured mass-minus-stiffness action separately for each environment, and compares '
        'it with captured native element/node actions and two captured residual-FD pairs. '
        'It is not a complete solve or physical qualification. See `report.json` for all '
        'gates, failures, branch qualifications, and source bindings.\n')
    files = {}
    for pth in sorted(out.iterdir()):
        if pth.is_file() and pth.name != 'manifest.json':
            files[pth.name] = sha(pth)
    manifest = {'schema': 'numi.cardboard.operator-fd-e2-audit-package.v1',
                'files_sha256': files, 'report_sha256': sha(out / 'report.json')}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(out), 'status': report['status'],
                      'environments': env_count, 'wall_seconds': report['wall_seconds'],
                      'warnings': warnings_seen}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'audit_failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
