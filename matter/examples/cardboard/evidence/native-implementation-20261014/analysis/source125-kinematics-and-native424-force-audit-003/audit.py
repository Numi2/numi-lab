#!/usr/bin/env python3
"""Source-bound kinematics and selected native endpoint force audit.

CPU only. Reconstructs source125 deformation inputs from exact FP32 node words
in native424 sample 6 (epsilon=2^-14), projects the unchanged accepted 15-state
history independently in FP64, and compares local response, element forces,
assembled residual increments, and the center tangent. Failed material rows
remain unavailable/null; this is not a solver or physical qualification.
"""
from __future__ import annotations

import datetime
import gzip
import hashlib
import importlib.util
import json
import math
import pathlib
import time
import warnings

import numpy as np

BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014')
SOURCE = BASE / 'snapshots/source-status5-vtccd-diagnostics-125'
CAPTURE = BASE / 'solver/two-stage-first-failure-rich-fd-419/operator-capture'
SWEEP = BASE / 'analysis/native-residual-fd-sweep-result-424'
SWEEP_AUDIT = BASE / 'analysis/native424-residual-sweep-audit-001'
E2 = BASE / 'analysis/operator-fd-e2-adapter-001'
CENTER_AUDIT = E2 / 'coupon419-003'
TERM = BASE / 'analysis/native-operator-term-replay-419-001'
REF_PATH = BASE / 'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
FD_PATH = BASE / 'analysis/source101-capture360-directional-fd-001/compare_directional_fd.py'
GEOM_PATH = BASE / 'snapshots/source-boundary-reset-order-101/matter/tools/cardboard_cached_operator_audit.py'
OUT = pathlib.Path(__file__).resolve().parent


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f32(x):
    value = np.asarray(x, dtype=np.float32).astype(np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError('nonfinite captured FP32 input')
    return value


def vecnorm(x) -> float:
    return math.sqrt(math.fsum(float(v) * float(v) for v in np.asarray(x).ravel()))


def metrics(actual, reference):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return {'status': 'unavailable_nonfinite_or_shape', 'actual_l2': None,
                'reference_l2': None, 'difference_l2': None,
                'relative_to_reference': None, 'max_component_abs': None}
    d = a - b
    na, nb, nd = vecnorm(a), vecnorm(b), vecnorm(d)
    return {'status': 'available', 'actual_l2': na, 'reference_l2': nb,
            'difference_l2': nd, 'relative_to_reference': nd / nb if nb else None,
            'max_component_abs': float(np.max(np.abs(d))) if d.size else 0.0}


def bits_equal(a, b) -> bool:
    aa = np.ascontiguousarray(a, dtype=np.float32)
    bb = np.ascontiguousarray(b, dtype=np.float32)
    return aa.shape == bb.shape and np.array_equal(aa.view(np.uint32), bb.view(np.uint32))


def _f32(x):
    return np.float32(x)


def _two_sum(a, b):
    a, b = _f32(a), _f32(b)
    high = _f32(a + b)
    virtual_b = _f32(high - a)
    low = _f32(_f32(a - _f32(high - virtual_b)) + _f32(b - virtual_b))
    return high, low


def _round_odd(pair):
    high, low = _f32(pair[0]), _f32(pair[1])
    bits = int(np.asarray(high).view(np.uint32))
    if low == 0 or bits & 1:
        return high
    direction = _f32(np.inf if low > 0 else -np.inf)
    return np.nextafter(high, direction, dtype=np.float32)


def _compensated_add(a, b):
    high = _two_sum(a[0], b[0])
    low = _two_sum(a[1], b[1])
    middle = _two_sum(high[1], low[0])
    merged = _two_sum(high[0], middle[0])
    tail0 = _two_sum(merged[1], middle[1])
    tail1 = _two_sum(tail0[0], low[1])
    sticky = _round_odd(_two_sum(tail1[1], tail0[1]))
    tail = _round_odd(_two_sum(tail1[0], sticky))
    return _two_sum(merged[0], tail)


def _compensated_product(a, b):
    a, b = _f32(a), _f32(b)
    high = _f32(a * b)
    # A binary32 x binary32 exact product fits in binary64's significand;
    # this computes the same rounded FMA residual for finite normal inputs.
    low = _f32(float(a) * float(b) - float(high))
    return high, low


def _compensated_multiply(a, b):
    return _compensated_add(_compensated_product(a[0], b[0]),
        _compensated_add(_compensated_product(a[0], b[1]),
            _compensated_add(_compensated_product(a[1], b[0]),
                             _compensated_product(a[1], b[1]))))


def frame_from_xyzw(quaternion):
    """Transcribe source125 femMaterialFrame FP32 operations."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float32)
    # Metal dot(float4,float4) source operands are four binary32 products and
    # a binary32 reduction. Both pairings are compared at center below; for
    # this captured source they produce the same quaternion/frame words.
    sq = [_f32(v * v) for v in (x, y, z, w)]
    norm2 = _f32(_f32(_f32(sq[0] + sq[1]) + sq[2]) + sq[3])
    invnorm = _f32(_f32(1.0) / _f32(np.sqrt(norm2, dtype=np.float32)))
    x, y, z, w = (_f32(v * invnorm) for v in (x, y, z, w))
    two = _f32(2.0)
    R = np.empty((3, 3), dtype=np.float32)
    R[0, 0] = _f32(1.0 - _f32(two * _f32(_f32(y*y) + _f32(z*z))))
    R[0, 1] = _f32(two * _f32(_f32(x*y) - _f32(z*w)))
    R[0, 2] = _f32(two * _f32(_f32(x*z) + _f32(y*w)))
    R[1, 0] = _f32(two * _f32(_f32(x*y) + _f32(z*w)))
    R[1, 1] = _f32(1.0 - _f32(two * _f32(_f32(x*x) + _f32(z*z))))
    R[1, 2] = _f32(two * _f32(_f32(y*z) - _f32(x*w)))
    R[2, 0] = _f32(two * _f32(_f32(x*z) - _f32(y*w)))
    R[2, 1] = _f32(two * _f32(_f32(y*z) + _f32(x*w)))
    R[2, 2] = _f32(1.0 - _f32(two * _f32(_f32(x*x) + _f32(y*y))))
    return R


def source_deformation_words(node_words, tets, invrest, frames, frame_active):
    """Transcribe nmFEMReferenceDeformationWords for exact FP32 node words."""
    u = np.asarray(node_words[:, 4, :3], dtype=np.float32)
    highs = np.empty((len(tets), 3, 3), dtype=np.float32)
    lows = np.empty_like(highs)
    for ti, tet in enumerate(tets):
        ids = np.asarray(tet['nodes'], dtype=np.int64)
        world = [[None] * 3 for _ in range(3)]
        for row in range(3):
            for col in range(3):
                value = (_f32(1.0 if row == col else 0.0), _f32(0.0))
                for k in range(3):
                    difference = _compensated_add((u[ids[k + 1], row], _f32(0.0)),
                                                   (-u[ids[0], row], _f32(0.0)))
                    value = _compensated_add(value,
                        _compensated_multiply(difference, (_f32(invrest[ti, k, col]), _f32(0.0))))
                world[row][col] = value
        frame = frames[ti] if frame_active[ti] else np.eye(3, dtype=np.float32)
        for row in range(3):
            for col in range(3):
                value = (_f32(0.0), _f32(0.0))
                for k in range(3):
                    value = _compensated_add(value,
                        _compensated_multiply(world[row][k], (_f32(frame[k, col]), _f32(0.0))))
                highs[ti, row, col], lows[ti, row, col] = value
    return highs, lows


def _fma32(a, b, c):
    # Exact binary32 product plus add rounded once to binary32. The product of
    # two finite binary32 values is exactly representable in binary64 here.
    return np.float32(float(np.float32(a)) * float(np.float32(b)) + float(np.float32(c)))


def source_velocity_gradient(node_words, tets, invrest, frames, frame_active):
    """Transcribe tetrahedronVelocityGradient then femMaterialInput in FP32."""
    v = np.asarray(node_words[:, 1, :3], dtype=np.float32)
    out = np.empty((len(tets), 3, 3), dtype=np.float32)
    for ti, tet in enumerate(tets):
        ids = np.asarray(tet['nodes'], dtype=np.int64)
        dv = np.stack([v[ids[k + 1]] - v[ids[0]] for k in range(3)], axis=1).astype(np.float32)
        world = np.empty((3, 3), dtype=np.float32)
        for row in range(3):
            for col in range(3):
                value = _f32(0.0)
                for k in range(3):
                    value = _fma32(dv[row, k], invrest[ti, k, col], value)
                world[row, col] = value
        if frame_active[ti]:
            for row in range(3):
                for col in range(3):
                    value = _f32(0.0)
                    for k in range(3):
                        value = _fma32(world[row, k], frames[ti, k, col], value)
                    out[ti, row, col] = value
        else:
            out[ti] = world
    return out


def reconstruct_kinematics(node_words, tets, invrest, frames, frame_active):
    """Return source-transcribed F high/low and diagnostic Fdot from state words."""
    u = np.asarray(node_words[:, 4, :3], dtype=np.float64)
    v = np.asarray(node_words[:, 1, :3], dtype=np.float64)
    count = len(tets)
    F_high, F_low = source_deformation_words(node_words, tets, invrest, frames, frame_active)
    F = F_high.astype(np.float64) + F_low.astype(np.float64)
    # Rate uses the production FP32 matrix path, unlike compensated F. The
    # retained 15-state reference constitutive law is rate-independent here.
    Fdot = source_velocity_gradient(node_words, tets, invrest, frames, frame_active).astype(np.float64)
    return F, Fdot, v, u, F_high, F_low


def elemental_force(P_local, invrest, volume, frames, frame_active):
    """Production-sign nodal internal force, float64 assembly per tet."""
    n = len(P_local)
    result = np.empty((n, 4, 3), dtype=np.float64)
    for i in range(n):
        R = frames[i] if frame_active[i] else np.eye(3)
        P_world = P_local[i] @ R.T
        h = -float(volume[i]) * (P_world @ invrest[i].T)
        result[i, 1] = h[:, 0]
        result[i, 2] = h[:, 1]
        result[i, 3] = h[:, 2]
        result[i, 0] = -(result[i, 1] + result[i, 2] + result[i, 3])
    return result


def scatter_elements(element, tets, nnode):
    # Canonical tet/slot order, FP64 serial sum. Avoid BLAS reductions and keep
    # a reproducible independent gather of the 4 element-vector records.
    result = np.zeros((nnode, 3), dtype=np.float64)
    for ti, tet in enumerate(tets):
        for slot, node in enumerate(tet['nodes']):
            result[int(node)] += element[ti, slot]
    return result


def state_scale_probe(old, z, p26):
    scale = np.maximum(np.maximum(np.abs(old), np.abs(z)), p26[:, None])
    return scale


def main():
    outdir = OUT
    if (outdir / 'report.json').exists() or (outdir / 'manifest.json').exists():
        raise RuntimeError('refusing to overwrite an existing audit output')

    receipt = json.loads((SWEEP / 'receipt.json').read_text())
    sweep_report = json.loads((SWEEP_AUDIT / 'report.json').read_text())
    center_report = json.loads((CENTER_AUDIT / 'report.json').read_text())
    prediction = json.loads((BASE / 'solver/two-stage-first-failure-rich-fd-419/prediction.json').read_text())
    if receipt.get('physical_transactions') != 0 or receipt.get('prepared_package_unchanged') is not True:
        raise RuntimeError('native424 receipt does not bind a read-only zero-transaction replay')
    if not sweep_report.get('all_retained_endpoints_bitwise_equal'):
        raise RuntimeError('native424 endpoint parity is not sealed')
    if center_report.get('bindings', {}).get('capture_sha256') != sha(CAPTURE):
        raise RuntimeError('center CPU audit is not bound to capture419')

    records = [json.loads(line) for line in CAPTURE.read_text().splitlines() if line.strip()]
    header = records[0]
    if header.get('schema') != 'numi.cardboard.fem-newton-capture.v1':
        raise RuntimeError('unsupported rich capture schema')
    if (header.get('root'), header.get('newton_iteration'), header.get('environment_count'),
        header.get('node_count_per_environment'), header.get('tetrahedron_count_per_environment')) != (1, 48, 2, 882, 1920):
        raise RuntimeError('selected capture dimensions/generation differ')
    nodes_json = sorted((r for r in records if r.get('record') == 'node' and r.get('environment') == 0),
                        key=lambda r: int(r['node_id']))
    tets = sorted((r for r in records if r.get('record') == 'tetrahedron' and r.get('environment') == 0),
                  key=lambda r: int(r['tet_id']))
    if len(nodes_json) != 882 or len(tets) != 1920:
        raise RuntimeError('incomplete captured environment-0 topology')
    if [int(n['node_id']) for n in nodes_json] != list(range(882)) or [int(t['tet_id']) for t in tets] != list(range(1920)):
        raise RuntimeError('noncanonical node/tet IDs')
    eps = float(np.float32(1.0 / 16384.0))
    eps_report = sweep_report['samples']['6']['epsilon'] if 'samples' in sweep_report else None
    if eps_report is None:
        eps_report = sweep_report['samples'][6]['epsilon']
    if float(eps_report) != eps:
        raise RuntimeError(f'wrong selected epsilon {eps_report}')
    dt = float(np.float32(header['dt_seconds_by_environment'][0]))
    selected = {
        'center': ('sample-0-0', 'center'),
        'plus': ('sample-6-0', 'epsilon_pair_1_plus'),
        'minus': ('sample-6-1', 'epsilon_pair_1_minus'),
    }
    node_states = {}
    node_binary_match = {}
    for label, (stem, capture_key) in selected.items():
        path = SWEEP / 'result' / f'{stem}-nodes.bin'
        raw = np.fromfile(path, dtype='<f4')
        if raw.size != 2 * 882 * 5 * 4:
            raise RuntimeError(f'{stem} node payload length mismatch')
        state = raw.reshape(2, 882, 5, 4)[0]
        node_states[label] = state
        if label == 'center':
            captured = np.asarray([n[capture_key]['raw_state_float4s'] for n in nodes_json], dtype=np.float32)
            node_binary_match[label] = {
                'rich_capture_parity': 'exact center comparison',
                'all_words_bitwise_equal_to_capture': bits_equal(state, captured),
                'word_mismatches': int(np.count_nonzero(state.view(np.uint32) != captured.view(np.uint32))),
                'binary_sha256': sha(path),
            }
            if not node_binary_match[label]['all_words_bitwise_equal_to_capture']:
                raise RuntimeError(f'{stem} exact center node words differ from capture419')
        else:
            # Epsilon=2^-14 was not one of capture419's two FD pairs. These
            # are exact native424 readback words and are the authoritative
            # endpoint inputs for this audit; do not substitute nominal F.
            node_binary_match[label] = {
                'rich_capture_parity': 'not captured at this epsilon; exact native424 binary readback',
                'all_words_bitwise_equal_to_capture': None,
                'word_mismatches': None,
                'binary_sha256': sha(path),
            }

    audit = load_module(E2 / 'audit_capture.py', 'e2_audit_selected424')
    ref = load_module(REF_PATH, 'bilinear15_reference_selected424')
    fd = load_module(FD_PATH, 'directional_fd_selected424')
    geo = load_module(GEOM_PATH, 'cardboard_geometry_selected424')
    npz = np.load(CENTER_AUDIT / 'environment-0.npz')

    nnode, ntet = 882, 1920
    invrest = np.asarray([[f32(t['inverse_rest_rows_per_m']) for t in tets]], dtype=np.float64)[0]
    volume = f32([t['rest_volume_m3'] for t in tets])
    frames32 = np.asarray([frame_from_xyzw(t['material_frame_quaternion_xyzw']) for t in tets],
                          dtype=np.float32)
    frames = frames32.astype(np.float64)
    frame_active = np.asarray([int(f32([t['center']['dt_seconds_bulk_modulus_kind_frame_active'][3]])[0]) == 1
                               for t in tets])
    ids = np.asarray([int(t['material_index']) for t in tets], dtype=np.int32)
    paper_ids = np.flatnonzero(np.isin(ids, [0, 1]))
    if len(paper_ids) != 1728 or np.count_nonzero(ids == 2) != 192:
        raise RuntimeError('paper/glue tetrahedron partition differs from 1728+192')
    ptab = f32(header['parameters_by_environment'][0])
    mats = {int(m['index']): m for m in header['materials']}
    params_all = np.zeros((ntet, 28), dtype=np.float64)
    for ti, tet in enumerate(tets):
        mat = mats[int(tet['material_index'])]
        off = int(mat['parameter_offset'])
        count = len(mat['parameter_names'])
        expected = 28 if int(tet['material_index']) in (0, 1) else 3
        if count != expected or int(tet['material_parameter_count']) != expected:
            raise RuntimeError(f'tet {ti} unsupported parameter layout')
        params_all[ti, :count] = ptab[off:off + count]
    accepted = np.zeros((ntet, 15), dtype=np.float64)
    for ti in paper_ids:
        accepted[ti] = f32(tets[int(ti)]['center']['accepted_state'])
    p26all = params_all[:, 26]
    mass = f32([n['center']['mass_kg'] for n in nodes_json])
    free = np.asarray([float(f32([n['center']['constraint_tag']])[0]) == 0.0 for n in nodes_json])
    accepted_v = np.asarray([f32(n['accepted']['v_m_s']) for n in nodes_json])
    if not np.array_equal(mass, f32([n['center']['raw_state_float4s'][0][3] for n in nodes_json])):
        raise RuntimeError('mass fields disagree in captured node record')

    kin = {}
    for label, state in node_states.items():
        F, Fdot, v, u, F_high, F_low = reconstruct_kinematics(
            state, tets, invrest, frames32, frame_active)
        kin[label] = {'F': F, 'Fdot': Fdot, 'v': v, 'u': u,
                      'F_high': F_high, 'F_low': F_low}

    # Validate the kinematic reconstruction against the captured high+low
    # center F and the independently saved center Fdot before using it for the
    # new endpoint material replay.
    Fc_high = np.stack([f32(t['center']['F_local']) for t in tets])
    Fc_low = np.stack([f32(t['center'].get('F_local_low', np.zeros((3, 3)))) for t in tets])
    Fc = Fc_high + Fc_low
    Fdotc = np.stack([f32(t['center']['Fdot_local_per_s']) for t in tets])
    F_recon_center = kin['center']['F']
    Fdot_recon_center = kin['center']['Fdot']
    f_center_delta = F_recon_center - Fc
    fdot_center_delta = Fdot_recon_center - Fdotc
    center_pair_high_exact = bits_equal(kin['center']['F_high'], Fc_high)
    center_pair_low_exact = bits_equal(kin['center']['F_low'], Fc_low)
    center_fdot_exact = bits_equal(kin['center']['Fdot'], np.stack(
        [f32(t['center']['Fdot_local_per_s']) for t in tets]))
    if not center_pair_high_exact or not center_pair_low_exact:
        raise RuntimeError('source-transcribed center F high/low does not bit-match capture419')
    if not center_fdot_exact:
        raise RuntimeError('source-transcribed center Fdot does not bit-match capture419')
    # Constitutive center replay is anchored to the actual captured production
    # input words. Side endpoints are source-transcribed from their exact raw
    # FP32 u words using those same recovered FP32 frames and inverse-rest data.
    kin['center']['F'] = Fc

    # Independent accepted-history projections. Keep failed values unavailable.
    paper_params = params_all[paper_ids]
    paper_old = accepted[paper_ids]
    p26 = paper_params[:, 26]
    paper_mat = ids[paper_ids]
    paper_F = {label: kin[label]['F'][paper_ids] for label in selected}
    projection = {}
    stress = {}
    warnings_seen = []
    residual_layer = {}
    for label in selected:
        projection[label] = {'ok': np.zeros(len(paper_ids), dtype=bool),
                             'z': np.full((len(paper_ids), 15), np.nan),
                             'scaled_residual': np.full(len(paper_ids), np.nan),
                             'scaled_correction': np.full(len(paper_ids), np.nan),
                             'iterations': np.full(len(paper_ids), -1, dtype=np.int32),
                             'reasons': [None] * len(paper_ids)}
        stress[label] = np.full((ntet, 3, 3), np.nan)
        for mi in (0, 1):
            local = np.flatnonzero(paper_mat == mi)
            ids_local = paper_ids[local]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                z, it, rn, cn, ok, reasons, _ = audit.project_centers(
                    ref, fd, paper_params[local], paper_F[label][local],
                    paper_old[local], paper_params[local, 26])
                for warning in caught:
                    warnings_seen.append({'endpoint': label, 'material_index': mi,
                        'category': warning.category.__name__, 'message': str(warning.message)})
            projection[label]['ok'][local] = ok
            projection[label]['z'][local] = z
            projection[label]['scaled_residual'][local] = rn
            projection[label]['scaled_correction'][local] = cn
            projection[label]['iterations'][local] = it
            for j, reason in enumerate(reasons):
                if not ok[j]:
                    projection[label]['reasons'][local[j]] = reason
            successful = np.flatnonzero(ok)
            if successful.size:
                stress[label][ids_local[successful]] = ref.stress(
                    paper_params[local[successful]], paper_F[label][local[successful]], z[successful])
        # Stateless glue has a direct analytic reference response.
        glue = np.flatnonzero(ids == 2)
        if glue.size:
            stress[label][glue] = audit.glue_stress(
                kin[label]['F'][glue], params_all[glue, 1], params_all[glue, 2])

    # Center response uses the same independent accepted-history projection.
    # Reuse center output above, but ensure captured baseline F agrees closely.
    forces_ind = {}
    for label in selected:
        valid = np.all(np.isfinite(stress[label]), axis=(1, 2))
        force = np.full((ntet, 4, 3), np.nan)
        if np.any(valid):
            force[valid] = elemental_force(stress[label][valid], invrest[valid],
                                           volume[valid], frames[valid], frame_active[valid])
        forces_ind[label] = force

    native_force = {}
    native_residual = {}
    for label, (stem, _) in selected.items():
        fa = np.fromfile(SWEEP / 'result' / f'{stem}-forces.bin', dtype='<f4')
        ra = np.fromfile(SWEEP / 'result' / f'{stem}-residual.bin', dtype='<f4')
        if fa.size != 2 * ntet * 4 * 4 or ra.size != 2 * nnode * 4:
            raise RuntimeError(f'{stem} force/residual payload size mismatch')
        native_force[label] = fa.reshape(2, ntet, 4, 4)[0, :, :, :3].astype(np.float64)
        native_residual[label] = ra.reshape(2, nnode, 4)[0, :, :3].astype(np.float64)

    # Native versus independent element-force error, and independent/global
    # residual assembled with the exact realized endpoint velocity words.
    for label in selected:
        valid_tet = np.all(np.isfinite(forces_ind[label]), axis=(1, 2))
        force_ind_global = np.zeros((nnode, 3), dtype=np.float64)
        force_native_global = np.zeros_like(force_ind_global)
        if valid_tet.all():
            force_ind_global = scatter_elements(forces_ind[label], tets, nnode)
        else:
            for ti in np.flatnonzero(valid_tet):
                tet = tets[int(ti)]
                for slot, node in enumerate(tet['nodes']):
                    force_ind_global[int(node)] += forces_ind[label][ti, slot]
        force_native_global = scatter_elements(native_force[label], tets, nnode)
        valid_ids = np.flatnonzero(valid_tet)
        force_native_valid_global = scatter_elements(
            native_force[label][valid_tet], [tets[int(i)] for i in valid_ids], nnode)
        force_native_center_valid_global = scatter_elements(
            native_force['center'][valid_tet], [tets[int(i)] for i in valid_ids], nnode)
        force_ind_center_valid_global = scatter_elements(
            forces_ind['center'][valid_tet], [tets[int(i)] for i in valid_ids], nnode)
        if valid_tet.all():
            force_metrics = metrics(native_force[label], forces_ind[label])
            valid_force_metrics = force_metrics
        else:
            force_metrics = {'status': 'partial_projection_failure', 'valid_tets': int(valid_tet.sum()),
                             'total_tets': ntet, 'actual_l2': None, 'reference_l2': None,
                             'difference_l2': None, 'relative_to_reference': None,
                             'max_component_abs': None}
            good = np.broadcast_to(valid_tet[:, None, None], forces_ind[label].shape)
            valid_force_metrics = metrics(native_force[label][good], forces_ind[label][good])
        # Residual changes cancel constant gravity/other constant loads. Source
        # mechanical residual is dt*(f_internal + constant external)-M*(v-vold).
        dv = kin[label]['v'] - kin['center']['v']
        dR_ind_partial = dt * (force_ind_global - (scatter_elements(forces_ind['center'], tets, nnode)
                              if np.all(np.isfinite(forces_ind['center'])) else np.zeros((nnode, 3)))) - mass[:, None] * dv
        dR_ind_valid_force = dt * (force_ind_global - force_ind_center_valid_global) - mass[:, None] * dv
        dR_native_force = dt * (force_native_global - scatter_elements(native_force['center'], tets, nnode)) - mass[:, None] * dv
        dR_native_valid_force = dt * (force_native_valid_global - force_native_center_valid_global) - mass[:, None] * dv
        dR_native_capture = native_residual[label] - native_residual['center']
        dR_ind_partial[~free] = 0.0
        dR_ind_valid_force[~free] = 0.0
        dR_native_force[~free] = 0.0
        dR_native_valid_force[~free] = 0.0
        dR_native_capture[~free] = 0.0
        dR_ind_complete = dR_ind_partial if valid_tet.all() else None
        failed_tets = np.flatnonzero(~valid_tet)
        failed_native_tet_delta = np.zeros((nnode, 3), dtype=np.float64)
        for ti in failed_tets:
            for slot, node in enumerate(tets[int(ti)]['nodes']):
                failed_native_tet_delta[int(node)] += dt * (
                    native_force[label][ti, slot] - native_force['center'][ti, slot])
        failed_native_tet_delta[~free] = 0.0
        residual_layer[label] = {
            'independent_force_mass_delta_R_Ns': dR_ind_complete,
            'independent_valid_rows_partial_delta_R_Ns': dR_ind_partial,
            'independent_valid_tet_force_mass_delta_R_Ns': dR_ind_valid_force,
            'independent_complete': bool(valid_tet.all()),
            'native_element_force_mass_delta_R_Ns': dR_native_force,
            'native_valid_tet_force_mass_delta_R_Ns': dR_native_valid_force,
            'native_captured_delta_R_Ns': dR_native_capture,
            'native_force_formula_vs_native_residual': metrics(dR_native_force[free], dR_native_capture[free]),
            'independent_valid_rows_vs_native_valid_tet_force_delta_R': metrics(
                dR_ind_valid_force[free], dR_native_valid_force[free]),
            'independent_valid_rows_vs_all_native_force_delta_R': (
                metrics(dR_ind_partial[free], dR_native_force[free]) if valid_tet.all() else None),
            'independent_complete_vs_native_force_delta_R': (
                metrics(dR_ind_complete[free], dR_native_force[free])
                if dR_ind_complete is not None else None),
            'failed_tet_ids': [int(i) for i in failed_tets],
            'failed_native_tet_delta_R_Ns_l2': vecnorm(failed_native_tet_delta[free]),
            'failed_native_tet_delta_R_Ns_max_abs': float(np.max(np.abs(failed_native_tet_delta[free])))
                if np.any(failed_tets) else 0.0,
            'native_element_force_vs_independent_valid_tets': valid_force_metrics,
            'independent_global_force': force_ind_global,
            'native_global_force': force_native_global,
            'realized_delta_velocity': dv,
        }

    # Recompute equation-level native and independent differences above. Keep
    # the centered residual/operator quantities in raw vectors for later exact
    # diagnostics, but serialize only norms/components in the JSON report.

    # Center A action from sealed independent FP64 audit and production replay.
    independent_A = np.asarray(npz['independent_A_delta_v_impulse_Ns'], dtype=np.float64)
    correction = np.asarray(npz['correction_delta_v_m_s'], dtype=np.float64)
    native_A = np.fromfile(SWEEP / 'result/fresh-action.float4.bin', dtype='<f4').reshape(2, nnode, 4)[0, :, :3].astype(np.float64)
    free3 = free.copy()
    for label in ('plus', 'minus'):
        residual_layer[label]['independent_center_A_nominal_direction'] = independent_A
        residual_layer[label]['native_center_A_nominal_direction'] = native_A

    # Metrics summary for actual per-tet force and projection differences.
    endpoint_summary = {}
    per_paper_rows = []
    center_local_z = projection['center']['z']
    center_local_ok = projection['center']['ok']
    # For exact tangent on realized directions, material input is dF/d(epsilon)
    # from endpoint node words, not epsilon*nominal correction alone.
    F_direction_realized = (kin['plus']['F'] - kin['minus']['F']) / (2.0 * eps)
    v_direction_realized = (kin['plus']['v'] - kin['minus']['v']) / (2.0 * eps)
    P_tangent_realized = np.full((ntet, 3, 3), np.nan)
    for mi in (0, 1):
        local = np.flatnonzero(ids[paper_ids] == mi)
        global_ids = paper_ids[local]
        ok = center_local_ok[local]
        valid_local = np.flatnonzero(ok)
        if valid_local.size:
            P_tangent_realized[global_ids[valid_local]] = ref.tangent(
                paper_params[local[valid_local]], paper_F['center'][local[valid_local]],
                paper_old[local[valid_local]], center_local_z[local[valid_local]],
                F_direction_realized[global_ids[valid_local]])
    glue = np.flatnonzero(ids == 2)
    if glue.size:
        P_tangent_realized[glue] = audit.glue_tangent_action(
            kin['center']['F'][glue], F_direction_realized[glue],
            params_all[glue, 1], params_all[glue, 2])
    realized_tangent_element = np.full((ntet, 4, 3), np.nan)
    tangent_valid_tets = np.all(np.isfinite(P_tangent_realized), axis=(1, 2))
    if np.any(tangent_valid_tets):
        realized_tangent_element[tangent_valid_tets] = elemental_force(
            P_tangent_realized[tangent_valid_tets], invrest[tangent_valid_tets],
            volume[tangent_valid_tets], frames[tangent_valid_tets], frame_active[tangent_valid_tets])
    tangent_global = np.zeros((nnode, 3), dtype=np.float64)
    if tangent_valid_tets.all():
        tangent_global = scatter_elements(realized_tangent_element, tets, nnode)
    else:
        for ti in np.flatnonzero(tangent_valid_tets):
            for slot, node in enumerate(tets[int(ti)]['nodes']):
                tangent_global[int(node)] += realized_tangent_element[ti, slot]
    action_realized_direction = mass[:, None] * v_direction_realized - dt * tangent_global
    action_realized_direction[~free] = 0.0

    # A common-row centered secant is admissible only on the same tet subset
    # at center, plus, and minus. It can diagnose the valid rows while keeping
    # the one failed plus projection explicitly outside the total secant.
    common_valid = np.ones(ntet, dtype=bool)
    for label in ('center', 'plus', 'minus'):
        common_valid &= np.all(np.isfinite(forces_ind[label]), axis=(1, 2))
    common_ids = np.flatnonzero(common_valid)
    common_tets = [tets[int(i)] for i in common_ids]
    common_center_ind = scatter_elements(forces_ind['center'][common_valid], common_tets, nnode)
    common_center_native = scatter_elements(native_force['center'][common_valid], common_tets, nnode)
    common_dR_ind = {}
    common_dR_native = {}
    for side in ('plus', 'minus'):
        common_current_ind = scatter_elements(forces_ind[side][common_valid], common_tets, nnode)
        common_current_native = scatter_elements(native_force[side][common_valid], common_tets, nnode)
        dv = kin[side]['v'] - kin['center']['v']
        common_dR_ind[side] = dt * (common_current_ind - common_center_ind) - mass[:, None] * dv
        common_dR_native[side] = dt * (common_current_native - common_center_native) - mass[:, None] * dv
        common_dR_ind[side][~free] = 0.0
        common_dR_native[side][~free] = 0.0
    common_secant_ind = -(common_dR_ind['plus'] - common_dR_ind['minus']) / (2.0 * eps)
    common_secant_native = -(common_dR_native['plus'] - common_dR_native['minus']) / (2.0 * eps)
    common_tangent_force = scatter_elements(realized_tangent_element[common_valid], common_tets, nnode)
    common_tangent_action = mass[:, None] * v_direction_realized - dt * common_tangent_force
    common_secant_ind[~free] = 0.0
    common_secant_native[~free] = 0.0
    common_tangent_action[~free] = 0.0

    native_secant = -(residual_layer['plus']['native_captured_delta_R_Ns'] -
                      residual_layer['minus']['native_captured_delta_R_Ns']) / (2.0 * eps)
    independent_complete = (residual_layer['plus']['independent_complete'] and
                           residual_layer['minus']['independent_complete'])
    independent_secant = None
    if independent_complete:
        independent_secant = -(residual_layer['plus']['independent_force_mass_delta_R_Ns'] -
                               residual_layer['minus']['independent_force_mass_delta_R_Ns']) / (2.0 * eps)
    nominal_action = independent_A
    native_nominal_action = native_A
    for x in (native_secant, action_realized_direction):
        x[~free] = 0.0
    if independent_secant is not None:
        independent_secant[~free] = 0.0
    unavailable_fd = {'status': 'unavailable_incomplete_independent_endpoint_projection',
                      'actual_l2': None, 'reference_l2': None, 'difference_l2': None,
                      'relative_to_reference': None, 'max_component_abs': None}
    endpoint_summary = {
        'selected_epsilon': eps,
        'native_centered_residual_secant_vs_native_center_A_nominal': metrics(native_secant[free], native_nominal_action[free]),
        'native_centered_residual_secant_vs_independent_center_A_nominal': metrics(native_secant[free], nominal_action[free]),
        'independent_endpoint_residual_secant_vs_independent_center_A_nominal':
            metrics(independent_secant[free], nominal_action[free]) if independent_secant is not None else unavailable_fd,
        'independent_endpoint_residual_secant_vs_center_tangent_on_realized_node_direction':
            metrics(independent_secant[free], action_realized_direction[free]) if independent_secant is not None else unavailable_fd,
        'native_endpoint_residual_secant_vs_center_tangent_on_realized_node_direction': metrics(
            native_secant[free], action_realized_direction[free]),
        'independent_vs_native_endpoint_residual_secant':
            metrics(independent_secant[free], native_secant[free]) if independent_secant is not None else unavailable_fd,
        'realized_dv_direction_vs_nominal_correction': metrics(v_direction_realized[free], correction[free]),
        'realized_dF_direction_vs_nominal_dt_times_captured_H': metrics(
            F_direction_realized[paper_ids],
            dt * np.stack([f32(t['center']['F_direction_local_per_s']) for t in tets])[paper_ids]),
        'realized_displacement_kinematics': {},
        'native_force_delta_R_formula_vs_captured': {
            side: residual_layer[side]['native_force_formula_vs_native_residual'] for side in ('plus', 'minus')},
        'native_vs_independent_delta_R': {
            side: residual_layer[side]['independent_valid_rows_vs_native_valid_tet_force_delta_R']
            for side in ('plus', 'minus')},
        'independent_valid_rows_vs_all_native_delta_R': {
            side: residual_layer[side]['independent_valid_rows_vs_all_native_force_delta_R']
            for side in ('plus', 'minus')},
        'common_valid_tet_centered_secant': {
            'status': 'partial_common_tet_subset_only',
            'valid_tet_count': int(common_valid.sum()),
            'total_tet_count': ntet,
            'omitted_tet_ids': [int(i) for i in np.flatnonzero(~common_valid)],
            'native_vs_independent': metrics(common_secant_native[free], common_secant_ind[free]),
            'independent_vs_full_native_center_A_incomplete_operator_diagnostic': {
                'status': 'diagnostic_only_full_A_includes_omitted_tet',
                'metrics': metrics(common_secant_ind[free], native_A[free])},
            'independent_vs_common_center_tangent': metrics(common_secant_ind[free], common_tangent_action[free]),
            'native_vs_common_center_tangent': metrics(common_secant_native[free], common_tangent_action[free]),
        },
        'omitted_tets_combined_native_plus_centered_secant_component_l2':
            residual_layer['plus']['failed_native_tet_delta_R_Ns_l2'] / (2.0 * eps),
    }
    for side in ('plus', 'minus'):
        du = kin[side]['u'] - kin['center']['u']
        dv = kin[side]['v'] - kin['center']['v']
        expected_du = dt * dv
        endpoint_summary['realized_displacement_kinematics'][side] = {
            'actual_du_vs_dt_dv': metrics(du[free], expected_du[free]),
            'unchanged_position_component_count': int(np.count_nonzero(
                (node_states[side][free, 0, :3].view(np.uint32) ==
                 node_states['center'][free, 0, :3].view(np.uint32)))),
            'free_position_components': int(3 * free.sum()),
            'u_endpoint_raw_words_match_capture': node_binary_match[side].get('all_words_bitwise_equal_to_capture'),
        }

    # Compare native and independent force increments on the identical set of
    # endpoint rows whose FP64 projections converged. These subset results are
    # diagnostic only; a failed row is never imputed into a complete force or
    # centered secant.
    endpoint_summary['valid_tet_force_increment_slopes'] = {}
    for side in ('plus', 'minus'):
        sign = 1.0 if side == 'plus' else -1.0
        valid = np.all(np.isfinite(forces_ind[side]), axis=(1, 2))
        valid &= np.all(np.isfinite(forces_ind['center']), axis=(1, 2))
        ids_valid = np.flatnonzero(valid)
        native_slope = (native_force[side][valid] - native_force['center'][valid]) / (sign * eps)
        independent_slope = (forces_ind[side][valid] - forces_ind['center'][valid]) / (sign * eps)
        tangent_slope = realized_tangent_element[valid]
        endpoint_summary['valid_tet_force_increment_slopes'][side] = {
            'eligible_tets': int(len(ids_valid)), 'omitted_failed_tet_ids': [int(i) for i in np.flatnonzero(~valid)],
            'native_vs_independent_same_tet_increment_per_epsilon': metrics(
                native_slope, independent_slope),
            'native_vs_center_tangent_same_tets': metrics(native_slope, tangent_slope),
            'independent_vs_center_tangent_same_tets': metrics(independent_slope, tangent_slope),
            'native_slope_l2_N_per_epsilon': vecnorm(native_slope),
            'independent_slope_l2_N_per_epsilon': vecnorm(independent_slope),
            'failed_tet_native_delta_R_l2_over_2epsilon_Ns_per_epsilon': (
                residual_layer[side]['failed_native_tet_delta_R_Ns_l2'] / (2.0 * eps)),
        }

    # Center force agreement and endpoint aggregate metrics.
    for label in selected:
        valid = np.all(np.isfinite(forces_ind[label]), axis=(1, 2))
        if valid.all():
            force_all = metrics(native_force[label], forces_ind[label])
            force_paper = metrics(native_force[label][paper_ids], forces_ind[label][paper_ids])
            force_glue = metrics(native_force[label][glue], forces_ind[label][glue])
        else:
            force_all = {'status': 'partial_projection_failure', 'valid_tets': int(valid.sum()),
                         'total_tets': ntet, 'difference_l2': None, 'relative_to_reference': None}
            force_paper = {'status': 'partial_projection_failure', 'valid_tets': int(np.count_nonzero(valid[paper_ids])),
                           'total_tets': len(paper_ids), 'difference_l2': None, 'relative_to_reference': None}
            force_glue = metrics(native_force[label][glue], forces_ind[label][glue])
        if label == 'center':
            force_tet_results = {'all_tets_native_vs_independent': force_all,
                                 'paper_tets_native_vs_independent': force_paper,
                                 'glue_tets_native_vs_independent': force_glue,
                                 'native_force_w_components_max_abs': float(np.max(np.abs(native_force[label])))}
        endpoint_summary.setdefault('native_vs_independent_element_force', {})[label] = {
            'all_tets': force_all, 'paper_tets': force_paper, 'glue_tets': force_glue,
            'projection_valid_tets_only': valid_force_metrics,
            'valid_independent_tets': int(valid.sum()), 'total_tets': ntet}

    for j, ti in enumerate(paper_ids):
        row = {'tet_id': int(ti), 'material_index': int(ids[ti]),
               'center_projection': {
                   'ok': bool(center_local_ok[j]),
                   'iterations': int(projection['center']['iterations'][j]),
                   'scaled_residual': (float(projection['center']['scaled_residual'][j])
                                       if center_local_ok[j] else None),
                   'scaled_correction': (float(projection['center']['scaled_correction'][j])
                                         if center_local_ok[j] else None),
                   'reason': projection['center']['reasons'][j]},
               'endpoints': {}}
        for side in ('plus', 'minus'):
            ok = bool(projection[side]['ok'][j])
            ti_force = native_force[side][ti]
            if ok and np.all(np.isfinite(forces_ind[side][ti])):
                row['endpoints'][side] = {
                    'projection_ok': True,
                    'iterations': int(projection[side]['iterations'][j]),
                    'scaled_residual': float(projection[side]['scaled_residual'][j]),
                    'scaled_correction': float(projection[side]['scaled_correction'][j]),
                    'lambda_active': bool(projection[side]['z'][j, 12] > paper_old[j, 12]),
                    'independent_stress_Pa': stress[side][ti].tolist(),
                    'native_vs_independent_element_force_N': metrics(ti_force, forces_ind[side][ti]),
                }
            else:
                row['endpoints'][side] = {'projection_ok': False,
                    'iterations': int(projection[side]['iterations'][j]),
                    'scaled_residual': None, 'scaled_correction': None,
                    'reason': projection[side]['reasons'][j],
                    'independent_stress_Pa': None,
                    'native_vs_independent_element_force_N': None}
        per_paper_rows.append(row)

    # Per-material projection gates and branch records.
    proj_summary = {}
    for label in selected:
        proj_summary[label] = {}
        for mi in (0, 1):
            local = np.flatnonzero(paper_mat == mi)
            ok = projection[label]['ok'][local]
            proj_summary[label][str(mi)] = {
                'name': mats[mi]['name'], 'rows': int(len(local)),
                'projection_success': int(ok.sum()), 'projection_failed': int((~ok).sum()),
                'max_scaled_residual_successful': (float(np.nanmax(projection[label]['scaled_residual'][local][ok]))
                                                    if np.any(ok) else None),
                'max_dynamic_scaled_correction_successful': (float(np.nanmax(projection[label]['scaled_correction'][local][ok]))
                                                              if np.any(ok) else None),
                'max_updates_successful': (int(np.max(projection[label]['iterations'][local][ok]))
                                           if np.any(ok) else None),
                'failure_examples': [{'tet_id': int(paper_ids[local[k]]),
                                      'reason': projection[label]['reasons'][local[k]]}
                                     for k in np.flatnonzero(~ok)[:12]],
            }
    lambda_branch = {'predicate': 'projected state[12] > accepted state[12]',
                     'scope': 'plastic-multiplier active/inactive only; not a complete constitutive branch signature',
                     'comparisons': {}}
    center_active = projection['center']['z'][:, 12] > paper_old[:, 12]
    for label in ('plus', 'minus'):
        valid = projection['center']['ok'] & projection[label]['ok']
        endpoint_active = projection[label]['z'][:, 12] > paper_old[:, 12]
        lambda_branch['comparisons'][label] = {
            'comparable_rows': int(valid.sum()),
            'omitted_rows': int((~valid).sum()),
            'center_active_rows': int(np.count_nonzero(center_active[valid])),
            'endpoint_active_rows': int(np.count_nonzero(endpoint_active[valid])),
            'active_to_inactive': int(np.count_nonzero(center_active[valid] & ~endpoint_active[valid])),
            'inactive_to_active': int(np.count_nonzero(~center_active[valid] & endpoint_active[valid])),
            'same_predicate_rows': int(np.count_nonzero(center_active[valid] == endpoint_active[valid])),
        }

    # Kinematics audit values from exact candidate words.
    correction = np.asarray(npz['correction_delta_v_m_s'], dtype=np.float64)
    realized = {}
    for side in ('plus', 'minus'):
        dv = kin[side]['v'] - kin['center']['v']
        du = kin[side]['u'] - kin['center']['u']
        dx = node_states[side][:, 0, :3].astype(np.float64) - node_states['center'][:, 0, :3].astype(np.float64)
        sign = 1.0 if side == 'plus' else -1.0
        nominal_dv = sign * eps * correction
        nominal_du = dt * (kin[side]['v']) - dt * kin['center']['v']
        realized[side] = {
            'node_words_match_rich_capture': node_binary_match[side],
            'realized_delta_v_vs_nominal_epsilon_correction': metrics(dv[free], nominal_dv[free]),
            'realized_delta_u_vs_dt_realized_delta_v': metrics(du[free], (dt * dv)[free]),
            'realized_position_delta_vs_realized_u_delta': metrics(dx[free], du[free]),
            'free_position_unchanged_components': int(np.count_nonzero(
                node_states[side][free, 0, :3].view(np.uint32) ==
                node_states['center'][free, 0, :3].view(np.uint32))),
            'free_u_unchanged_components': int(np.count_nonzero(
                node_states[side][free, 4, :3].view(np.uint32) ==
                node_states['center'][free, 4, :3].view(np.uint32))),
            'free_v_unchanged_components': int(np.count_nonzero(
                node_states[side][free, 1, :3].view(np.uint32) ==
                node_states['center'][free, 1, :3].view(np.uint32))),
            'free_xyz_component_count': int(3 * free.sum()),
            'nominal_vs_realized_delta_u': metrics(du[free], nominal_du[free]),
        }

    center_F_compare = metrics(F_recon_center, Fc)
    center_Fdot_compare = metrics(Fdot_recon_center, Fdotc)
    # Per tets choose worst reconstructed center discrepancy for provenance.
    paper_center_f_err = np.linalg.norm((F_recon_center - Fc).reshape(ntet, 9)[paper_ids], axis=1)
    paper_center_fdot_err = np.linalg.norm((Fdot_recon_center - Fdotc).reshape(ntet, 9)[paper_ids], axis=1)
    report = {
        'schema': 'numi.cardboard.source125-kinematics-native424-force-audit.v1',
        'status': 'diagnostic_inconclusive_one_endpoint_projection_failed',
        'evidence_class': 'source-bound CPU material/force/residual adjudication; no production or physical qualification',
        'scope': 'source125 control1/Newton48 environment0; center and exact native424 epsilon=2^-14 plus/minus endpoint node words; 1920 tets including all 1728 paper rows and 192 stateless glue rows',
        'bindings': {
            'source_snapshot': str(SOURCE), 'source125_runtime_mm_sha256': sha(SOURCE / 'matter/src/runtime.mm'),
            'source125_fem_metalinc_sha256': sha(SOURCE / 'matter/src/metal/fem.metalinc'),
            'source125_deformation_header_sha256': sha(SOURCE / 'matter/include/numi/matter/fem_compensated_deformation.h'),
            'source125_compensated_geometry_header_sha256': sha(SOURCE / 'include/metalrobo/compensated_geometry_gpu.h'),
            'source125_compensated_translation_header_sha256': sha(SOURCE / 'include/metalrobo/compensated_translation_gpu.h'),
            'source125_shared_node_state_sha256': sha(SOURCE / 'matter/include/numi/matter/shared.h'),
            'capture419_sha256': sha(CAPTURE), 'native424_receipt_sha256': sha(SWEEP / 'receipt.json'),
            'native424_report_sha256': sha(SWEEP_AUDIT / 'report.json'),
            'native424_sweep_manifest_sha256': sha(SWEEP / 'manifest.json'),
            'independent_center_report_sha256': sha(CENTER_AUDIT / 'report.json'),
            'independent_center_npz_sha256': sha(CENTER_AUDIT / 'environment-0.npz'),
            'bilinear15_reference_sha256': sha(REF_PATH),
            'projection_helper_sha256': sha(FD_PATH), 'geometry_helper_sha256': sha(GEOM_PATH),
            'native_endpoint_binary_sha256': {label: {
                'nodes': node_binary_match[label]['binary_sha256'],
                'forces': sha(SWEEP / 'result' / f'{stem}-forces.bin'),
                'residual': sha(SWEEP / 'result' / f'{stem}-residual.bin')}
                for label, (stem, _) in selected.items()},
        },
        'selection': {'environment': 0, 'root': 1, 'newton_iteration': 48,
                      'dt_seconds': dt, 'epsilon': eps, 'sample_index': 6,
                      'fd_admissibility': {'native_status_plus': 0, 'native_status_minus': 0,
                          'native_centered_residual_fd_vs_center_A_relative': 1.0948066880685303}},
        'source_update_path_audit': {
            'armijo_trial_candidate': {'file': 'matter/src/metal/fem.metalinc', 'lines': [4344, 4362],
                'contract': 'starts from the current candidate; increments its v by alpha*correction; computes deltaV relative to accepted v; in reference-displacement mode rebuilds u=accepted u+dt*new candidate v and x=rest+u'},
            'commit_candidate': {'file': 'matter/src/metal/fem.metalinc', 'lines': [3312, 3331],
                'contract': 'starts from the current candidate; increments its v by the same published alpha times the FGMRES solution; computes deltaV relative to accepted v; in reference-displacement mode rebuilds u=accepted u+dt*new candidate v and x=rest+u'},
            'alpha_publish': {'file': 'matter/src/metal/fem.metalinc', 'lines': [4491, 4539],
                'contract': 'tested alpha is published to FEM and coupled environment line-search state'},
            'runtime_order': {'file': 'matter/src/runtime.mm', 'lines': [22437, 22442, 23215, 23230, 23290],
                'contract': 'Armijo runs after geometry/contact alpha limiters; decision precedes normal FEM apply; no later alpha limiter found'},
            'compensated_F_input': {'file': 'matter/src/metal/fem.metalinc', 'lines': [212, 235],
                'contract': 'when paper compensated flag and reference-u tet flag are set, F is rebuilt as two-word I+grad(u) in local material frame'},
            'adjudication': 'Candidate velocity is incremented from the current Newton candidate, not reconstructed as accepted velocity plus one alpha-scaled correction. Displacement is then rebuilt from accepted displacement plus dt times that new candidate velocity. Thus the captured endpoint delta-u versus dt*delta-v relation is the relevant cross-sample check; alpha path and endpoint generation are distinct. No source evidence of a later alpha limiter was found.'
        },
        'node_word_kinematics': realized,
        'reconstructed_center_input_validation': {
            'F_local_source_transcription_from_center_node_u_vs_capture_high_plus_low': center_F_compare,
            'F_high_words_bitwise_equal_to_center_capture': bool(center_pair_high_exact),
            'F_low_words_bitwise_equal_to_center_capture': bool(center_pair_low_exact),
            'Fdot_words_bitwise_equal_to_center_capture': bool(center_fdot_exact),
            'center_pair_exact_tet_count': int(np.count_nonzero(np.all(
                (np.ascontiguousarray(kin['center']['F_high'], dtype=np.float32).view(np.uint32).reshape(ntet, 3, 3) ==
                 np.ascontiguousarray(Fc_high, dtype=np.float32).view(np.uint32).reshape(ntet, 3, 3)) &
                (np.ascontiguousarray(kin['center']['F_low'], dtype=np.float32).view(np.uint32).reshape(ntet, 3, 3) ==
                 np.ascontiguousarray(Fc_low, dtype=np.float32).view(np.uint32).reshape(ntet, 3, 3)), axis=(1, 2)))),
            'Fdot_local_reconstructed_from_center_node_v_vs_capture': center_Fdot_compare,
            'paper_rows': len(paper_ids),
            'max_center_F_difference_per_paper_tet': float(np.max(paper_center_f_err)),
            'max_center_Fdot_difference_per_paper_tet': float(np.max(paper_center_fdot_err)),
            'frame_active_tets': int(frame_active.sum()),
        },
        'independent_projection': proj_summary,
        'observed_lambda_branch_predicate': lambda_branch,
        'native_vs_independent_element_force': endpoint_summary.get('native_vs_independent_element_force', {}),
        'center_native_force_check': force_tet_results,
        'assembled_residual_and_fd': endpoint_summary,
        'realized_direction_center_tangent_action': {
            'projection_rows_valid_for_tangent': int(np.count_nonzero(tangent_valid_tets[paper_ids])),
            'all_tet_tangents_valid': bool(tangent_valid_tets.all()),
            'realized_direction_linear_action_vs_independent_endpoint_secant': metrics(
                action_realized_direction[free], independent_secant[free])
                if tangent_valid_tets.all() and independent_secant is not None else None,
        },
        'null_failure_policy': 'If any independent material projection fails, its stress/element-force fields are null in per-paper-tet.jsonl.gz; assembled independent totals use only valid rows and are explicitly marked partial, never represented as complete.',
        'warnings': warnings_seen,
        'conclusion': ('The FP32 source transcription of frame, compensated F high/low, and ordinary Fdot reproduces the captured center inputs bitwise for all 1,920 tetrahedra. On all plus/minus rows where the independent FP64 projection converges, native element-force increments differ from the independent FP64 increments by about 1.7e-5 relative in aggregate. One plus medium row (tet 1691) does not converge under the independent projector, so the complete independent centered force/residual secant is unavailable. Native force/residual bookkeeping itself matches the captured residual delta to about 1.6e-5 relative, but the native centered residual secant remains 1.0948 relative away from center A. This evidence does not establish that native stress/force quantization explains the full gap; the missing row and finite-step/branch effects leave that attribution unresolved.'),
        'limitations': [
            'The native FP32 frame quaternion, two-word deformation construction, and ordinary FP32 velocity-gradient path are transcribed from source125; reconstructed center F high/low and Fdot words match all 1,920 captured tets bitwise. Side F/Fdot words are produced by the same CPU transcription from exact native424 FP32 endpoint u/v words, but no separate side material-input capture exists at epsilon 2^-14.',
            'The constitutive response/projection and force assembly are independent FP64 equations evaluated from those source-transcribed side F pairs and exact captured material/history inputs; they are not a bitwise clone of native FP32 projection/force accumulation.',
            'Fdot is reconstructed and checked at center; the retained 15-state reference law is rate-independent, so Fdot is diagnostic and does not alter the projection equations used here.',
            'This one epsilon identifies endpoint force/residual layers only; it does not establish a universal numerical noise floor, a smooth full-history derivative gate, convergence, material calibration, or physical qualification.',
            'The run is a rejected Newton candidate with zero physical transactions.',
        ],
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    # Replace any internal arrays accidentally referenced in report summaries.
    # Construct compact exact serializable per-row evidence separately.
    # Validate no NaN/Inf survives strict output serialization.
    with gzip.open(outdir / 'per-paper-tet.jsonl.gz', 'xt', compresslevel=6, encoding='utf-8') as stream:
        for row in per_paper_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    (outdir / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    generated_bytes = sum(path.stat().st_size for path in outdir.iterdir() if path.is_file())
    if generated_bytes > 5 * 1024 * 1024:
        raise RuntimeError(f'output cap exceeded: {generated_bytes} bytes > 5 MiB')
    print(json.dumps({
        'report': str(outdir / 'report.json'),
        'per_tet_rows': str(outdir / 'per-paper-tet.jsonl.gz'),
        'rows': len(per_paper_rows),
        'generated_bytes': generated_bytes,
        'projections': proj_summary,
        'fd': endpoint_summary,
    }, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
