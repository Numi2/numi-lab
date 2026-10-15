#!/usr/bin/env python3
"""Measure cumulative private-Armijo residual stage deltas from strict captures.

This is a read-only observational audit, not an independent physics solver.
Every reported per-term vector is a promoted-FP64 difference between two
serialized FP32 cumulative residual checkpoints.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import struct
import sys
import numpy as np

READER_DIR = Path('/Users/home/cardboard-implementation-20261014/analysis/private-trial-reader-001')
sys.path.insert(0, str(READER_DIR))
from trial_capture_reader import Capture  # noqa: E402

SOURCE = Path('/Users/home/cardboard-implementation-20261014/snapshots/source-contact-capture-and-resolved-material-sweep-095')
SOURCE_FILES = {
    'runtime.mm': SOURCE / 'matter/src/runtime.mm',
    'fem.metalinc': SOURCE / 'matter/src/metal/fem.metalinc',
    'contact.metalinc': SOURCE / 'matter/src/metal/contact.metalinc',
    'fgmres.metalinc': SOURCE / 'matter/src/metal/fgmres.metalinc',
    'shared.h': SOURCE / 'matter/include/numi/matter/shared.h',
    'reader.py': READER_DIR / 'trial_capture_reader.py',
}
PHASES = {
    'pre': 'candidate_before_private_contact_rebuild',
    'bulk': 'after_bulk_mechanical_before_cohesive',
    'cohesive': 'after_cohesive_before_rigid_contact',
    'rigid_contact': 'after_rigid_contact_before_deformable_rebuild',
    'post': 'private_contact_rebuild_before_free_body_terms',
    'deformable': 'after_deformable_contact_before_free_body_terms',
    'full': 'full_residual_before_armijo_measure',
    'measurement': 'armijo_measurement_before_decision',
}
RESIDUAL_NAMES = {
    'bulk': 'residual_after_bulk_mechanical',
    'cohesive': 'residual_after_cohesive',
    'rigid_contact': 'residual_after_rigid_contact',
    'deformable': 'residual_after_deformable_contact',
    'full': 'residual_before_armijo_measure',
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def f32x4(capture: Capture, name: str, phase: str) -> np.ndarray:
    a = capture.arrays[(phase, name)]
    if a['dtype'] != 'float32x4' or a['element_bytes'] != 16:
        raise ValueError(f'{phase}/{name}: expected float32x4')
    values = capture.f32(name, phase)
    if values.size != 4 * a['count']:
        raise ValueError(f'{phase}/{name}: element count mismatch')
    return values.reshape((-1, 4))


def f32record(capture: Capture, name: str, phase: str, floats_per_record: int) -> np.ndarray:
    a = capture.arrays[(phase, name)]
    raw = capture.raw(name, phase)
    if a['element_bytes'] % 4 or a['element_bytes'] // 4 != floats_per_record:
        raise ValueError(f'{phase}/{name}: unexpected record float width')
    return np.frombuffer(raw, dtype='<f4').reshape((a['count'], floats_per_record))


def phase_by_token(capture: Capture, token: str) -> str:
    order = capture.manifest.get('producer_semantics', {}).get('phase_order', [])
    matches = [p for p in order if token == p or token in p]
    if len(matches) != 1:
        raise ValueError(f'expected one phase for {token!r}, got {matches}')
    return matches[0]


def safe_num(x: float):
    x = float(x)
    return x if np.isfinite(x) else None


_libm = ctypes.CDLL(None)
_fmaf = _libm.fmaf
_fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
_fmaf.restype = ctypes.c_float


def fma32(a, b, c) -> np.float32:
    """One correctly-rounded FP32 fused multiply-add, matching Metal fma."""
    return np.float32(_fmaf(float(a), float(b), float(c)))


def f32(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def incidence_ranges(capture: Capture, name: str, phase: str) -> np.ndarray:
    meta = capture.arrays[(phase, name)]
    raw = capture.raw(name, phase)
    if meta['dtype'] != 'record:NMIncidenceRangeGPU' or meta['element_bytes'] != 16:
        raise ValueError(f'{phase}/{name}: unexpected incidence range layout')
    return np.frombuffer(raw, dtype='<u4').reshape((-1, 4))


def f32_array(capture: Capture, name: str, phase: str) -> np.ndarray:
    meta = capture.arrays[(phase, name)]
    raw = capture.raw(name, phase)
    if meta['dtype'] != 'float32' or meta['element_bytes'] != 4:
        raise ValueError(f'{phase}/{name}: expected float32 array')
    return np.frombuffer(raw, dtype='<f4')


def u32_array(capture: Capture, name: str, phase: str) -> np.ndarray:
    meta = capture.arrays[(phase, name)]
    raw = capture.raw(name, phase)
    if meta['dtype'] not in ('uint32', 'uint32x4') or meta['element_bytes'] not in (4, 16):
        raise ValueError(f'{phase}/{name}: expected uint32 array')
    return np.frombuffer(raw, dtype='<u4')


def f32x4_array(capture: Capture, name: str, phase: str) -> np.ndarray:
    return f32x4(capture, name, phase)


def source_term_reconstruction(capture: Capture, dispatch: dict) -> dict:
    """Rebuild the three captured FEM contact/cohesive residual gathers.

    These use the exact captured production records and incidence order. The
    replay checks residual assembly, not narrowphase/constitutive correctness.
    """
    pre = phase_by_token(capture, PHASES['pre'])
    bulk_phase = phase_by_token(capture, PHASES['bulk'])
    cohesive_phase = phase_by_token(capture, PHASES['cohesive'])
    rigid_phase = phase_by_token(capture, PHASES['rigid_contact'])
    post = phase_by_token(capture, PHASES['post'])
    deformable_phase = phase_by_token(capture, PHASES['deformable'])
    full_phase = phase_by_token(capture, PHASES['full'])
    E, N = dispatch['environment_count'], dispatch['fem_node_count']
    if E != 1:
        raise ValueError('v1 exact gather replay supports one captured environment')

    def residual(name, phase):
        return f32x4(capture, name, phase).reshape((-1, 4))

    def compare(stage, previous_name, previous_phase, after_name, after_phase,
                predicted_term, predicted_after):
        before = residual(previous_name, previous_phase)[:N, :3]
        after = residual(after_name, after_phase)[:N, :3]
        promoted_delta = after.astype(np.float64) - before.astype(np.float64)
        predicted_delta = predicted_term.astype(np.float64)
        expected_after = predicted_after.astype(np.float32)
        mismatch = expected_after.view('<u4') != after.view('<u4')
        return {
            'comparison': 'captured cumulative residual after the production gather vs source-reconstructed gather applied to captured pre-stage residual',
            'node_count': int(N),
            'observed_stage_delta_fp64_from_fp32_checkpoints': vector_stats(promoted_delta),
            'reconstructed_stage_term_fp32': vector_stats(predicted_delta),
            'stage_delta_difference_fp64': vector_stats(promoted_delta - predicted_delta),
            'predicted_after_vs_captured_after': {
                'bitwise_equal_xyz': not bool(np.any(mismatch)),
                'mismatched_float_words_xyz': int(np.count_nonzero(mismatch)),
                'max_abs_fp32_difference': safe_num(np.max(np.abs(
                    expected_after.astype(np.float64) - after.astype(np.float64)))),
                'mismatch_locations_node_component': [
                    [int(i), int(j)] for i, j in np.argwhere(mismatch)[:32]
                ],
            },
        }

    # Cohesive gather: sum six-node face-force records by the captured stable
    # node incidence, multiply by the authored object timestep, then add to
    # the pre-stage FP32 residual using one fused multiply-add.
    coh_ranges = incidence_ranges(capture, 'cohesive_node_ranges', pre)
    coh_inc = u32_array(capture, 'cohesive_node_incidence', pre)
    coh_force = f32record(capture, 'fem_armijo_cohesive_face_forces', post, 24).reshape((-1, 6, 4))
    dispatch_raw = capture.raw('dispatch_buffer', pre)
    dispatch_u32 = np.frombuffer(dispatch_raw, dtype='<u4')
    dispatch_f32 = np.frombuffer(dispatch_raw, dtype='<f4')
    sched_raw = capture.raw('schedulers', pre)
    scheduler_words = np.frombuffer(sched_raw, dtype='<u4').reshape((-1, 20))
    rate_exp = min(int(scheduler_words[0, 1]), int(dispatch_u32[12]))
    dt = np.float32(dispatch_f32[43] / np.float32(1 << rate_exp))
    cohesive_term = np.zeros((N, 3), dtype=np.float32)
    cohesive_after = residual('residual_after_bulk_mechanical', bulk_phase)[:N, :3].copy()
    invalid_cohesive = []
    for node in range(N):
        first, count = map(int, coh_ranges[node, :2])
        force = np.zeros(3, dtype=np.float32)
        for encoded in coh_inc[first:first + count]:
            face_index, local = divmod(int(encoded), 6)
            if face_index >= len(coh_force) or local >= 6:
                invalid_cohesive.append([node, int(encoded)])
                continue
            force = f32(force + coh_force[face_index, local, :3])
        for axis in range(3):
            cohesive_term[node, axis] = np.float32(dt * force[axis])
            cohesive_after[node, axis] = fma32(
                dt, force[axis], cohesive_after[node, axis])

    cohesive_check = compare(
        'cohesive', 'residual_after_bulk_mechanical', bulk_phase,
        'residual_after_cohesive', cohesive_phase,
        cohesive_term, cohesive_after)
    cohesive_check.update({
        'dt_seconds_from_dispatch_and_scheduler': safe_num(dt),
        'cohesive_face_count': int(len(coh_force)),
        'cohesive_incidence_count': int(len(coh_inc)),
        'invalid_incidence_count': len(invalid_cohesive),
        'invalid_incidence_examples': invalid_cohesive[:16],
        'source_equation': 'node force is incidence-order sum(faceForces[face].node[slot].xyz); residual.xyz = fma(dt, force, residual.xyz)',
    })

    # Rigid-contact gather: each node sums negative proxy-side impulse from
    # valid samples referenced by its captured incidence list.
    rigid_ranges = incidence_ranges(capture, 'contact_node_ranges_accepted', pre)
    rigid_inc = u32_array(capture, 'contact_node_incidence_accepted', pre)
    rigid_samples_raw = capture.raw('contact_samples_private_trial', post)
    rigid_samples = np.frombuffer(rigid_samples_raw, dtype='<f4').reshape((-1, 40))
    rigid_sample_words = np.frombuffer(rigid_samples_raw, dtype='<u4').reshape((-1, 40))
    rigid_term = np.zeros((N, 3), dtype=np.float32)
    rigid_after = residual('residual_after_cohesive', cohesive_phase)[:N, :3].copy()
    bad_rigid = []
    valid_sample_count = int(np.count_nonzero(rigid_sample_words[:, 3] & 1))
    for node in range(N):
        first, count = map(int, rigid_ranges[node, :2])
        impulse = np.zeros(3, dtype=np.float32)
        for pair_index in rigid_inc[first:first + count]:
            pair_index = int(pair_index)
            if pair_index >= len(rigid_samples):
                bad_rigid.append([node, pair_index])
                continue
            if rigid_sample_words[pair_index, 3] & 1:
                impulse = f32(impulse - rigid_samples[pair_index, 16:19])
        rigid_term[node] = impulse
        for axis in range(3):
            rigid_after[node, axis] = fma32(1.0, impulse[axis], rigid_after[node, axis])
    rigid_check = compare(
        'rigid_contact', 'residual_after_cohesive', cohesive_phase,
        'residual_after_rigid_contact', rigid_phase,
        rigid_term, rigid_after)
    rigid_check.update({
        'valid_sample_count': valid_sample_count,
        'node_incidence_count': int(len(rigid_inc)),
        'invalid_incidence_count': len(bad_rigid),
        'invalid_incidence_examples': bad_rigid[:16],
        'source_equation': 'node impulse is incidence-order sum(-sample.impulseAndNormal.xyz) over NM_CONTACT_VALID samples; residual.xyz = residual.xyz + impulse',
    })

    # Deformable gather mirrors the production frame, six-node map, weighting,
    # timestep scaling, and mollifier-gradient subtraction. FMA is used where
    # the Metal expression contracts multiply-add; no contact law is recomputed.
    deform_ranges = incidence_ranges(
        capture, 'deformable_contact_node_ranges_private_trial', post)
    deform_inc = u32_array(
        capture, 'deformable_contact_node_incidence_private_trial', post)
    deform_raw = capture.raw('deformable_contacts_private_trial', post)
    deform = np.frombuffer(deform_raw, dtype='<f4').reshape((-1, 108))
    deform_words = np.frombuffer(deform_raw, dtype='<u4').reshape((-1, 108))
    active_indices = u32_array(capture, 'deformable_active_indices_private_trial', post)
    active_count = int(u32_array(capture, 'deformable_active_counts_private_trial', post)[0])
    active_set = set(int(v) for v in active_indices[:active_count])

    def cross32(a, b):
        return f32([
            fma32(a[1], b[2], -np.float32(a[2] * b[1])),
            fma32(a[2], b[0], -np.float32(a[0] * b[2])),
            fma32(a[0], b[1], -np.float32(a[1] * b[0])),
        ])

    def dot32(a, b):
        return fma32(a[0], b[0], fma32(a[1], b[1], np.float32(a[2] * b[2])))

    deform_term = np.zeros((N, 3), dtype=np.float32)
    deform_after = residual('residual_after_rigid_contact', rigid_phase)[:N, :3].copy()
    bad_deform = []
    incidence_visits = 0
    touched_slots = set()
    for node in range(N):
        first, count = map(int, deform_ranges[node, :2])
        impulse = np.zeros(3, dtype=np.float32)
        for encoded in deform_inc[first:first + count]:
            slot, local = int(encoded >> 3), int(encoded & 7)
            incidence_visits += 1
            if slot >= len(deform) or local >= 6:
                bad_deform.append([node, slot, local, 'slot_or_local_out_of_range'])
                continue
            if slot not in active_set:
                bad_deform.append([node, slot, local, 'incidence_slot_not_in_active_prefix'])
                continue
            words = deform_words[slot]
            if not (int(words[7]) & 1):
                continue
            touched_slots.add(slot)
            if local < 4:
                mapped_node = int(words[local])
                weight, time_scale = deform[slot, 8 + local], deform[slot, 12 + local]
                gradient = deform[slot, 52 + 4 * local:55 + 4 * local]
            elif local == 4:
                mapped_node = int(words[72])
                weight, time_scale = deform[slot, 76], deform[slot, 80]
                gradient = deform[slot, 84:87]
            else:
                mapped_node = int(words[73])
                weight, time_scale = deform[slot, 77], deform[slot, 81]
                gradient = deform[slot, 88:91]
            if mapped_node != node:
                bad_deform.append([node, slot, local, 'mapped_node_mismatch', mapped_node])
                continue
            normal = deform[slot, 16:19]
            if abs(float(normal[0])) < 0.57735026919:
                axis = f32([1, 0, 0])
            elif abs(float(normal[1])) < 0.57735026919:
                axis = f32([0, 1, 0])
            else:
                axis = f32([0, 0, 1])
            crossed = cross32(normal, axis)
            length = np.float32(np.sqrt(float(dot32(crossed, crossed))))
            tangent0 = f32(crossed / length)
            tangent1 = cross32(normal, tangent0)
            values = deform[slot, 32:35]
            contact_impulse = np.empty(3, dtype=np.float32)
            term = np.empty(3, dtype=np.float32)
            for axis_index in range(3):
                contact_impulse[axis_index] = fma32(
                    values[2], tangent1[axis_index],
                    fma32(values[1], tangent0[axis_index],
                          np.float32(values[0] * normal[axis_index])))
                term[axis_index] = fma32(
                    -deform[slot, 69], gradient[axis_index],
                    np.float32(weight * contact_impulse[axis_index]))
                term[axis_index] = fma32(time_scale, term[axis_index], impulse[axis_index])
            impulse = term
        deform_term[node] = impulse
        for axis in range(3):
            deform_after[node, axis] = fma32(1.0, impulse[axis], deform_after[node, axis])
    deform_check = compare(
        'deformable_contact', 'residual_after_rigid_contact', rigid_phase,
        'residual_after_deformable_contact', deformable_phase,
        deform_term, deform_after)
    deform_check.update({
        'active_contact_count': active_count,
        'active_prefix_unique_count': len(active_set),
        'valid_active_contact_count': int(sum(bool(int(deform_words[i, 7]) & 1) for i in active_set)),
        'incidence_visit_count': incidence_visits,
        'unique_contact_slots_touched': len(touched_slots),
        'invalid_incidence_count': len(bad_deform),
        'invalid_incidence_examples': bad_deform[:16],
        'source_equation': 'per-node incidence order; impulse += timeScale * (weight * (Jn*n + Jt0*t0 + Jt1*t1) - mollifier.y * mollifierGradient); residual.xyz += impulse',
        'scope': 'Verifies the captured contact records and FEM residual gather; it does not independently rerun broadphase, narrowphase, CCD, contact-history update, or constitutive response.',
    })

    # The free-body residual is first cleared/rebuilt after the deformable FEM
    # checkpoint. Report the final body rows, never their subtraction from the
    # stale pre-clear scratch at the preceding checkpoint.
    norm = measured_norm_report(capture, dispatch)
    row_start = 2 * E * N + E * dispatch['mpm_active_node_capacity']
    full = residual('residual_before_armijo_measure', full_phase)
    final_rigid = full[row_start:row_start + E * dispatch['rigid_generalized_capacity'], 0]
    return {
        'status': 'SOURCE_RECONSTRUCTED_ASSEMBLY_ONLY_NOT_NARROWPHASE_OR_PHYSICAL_VALIDATION',
        'dt_seconds_reconstructed': safe_num(dt),
        'cohesive': cohesive_check,
        'rigid_contact': rigid_check,
        'deformable_contact': deform_check,
        'final_rigid_generalized_x_rows': {
            'stats': vector_stats(final_rigid),
            'values': safe_list(final_rigid),
            'preceding_deformable_checkpoint_rigid_rows': 'stale pre-clear scratch; not a physical term or a valid baseline for subtraction',
            'source': 'runtime clears/rebuilds generalized body rows after deformable FEM residual assembly',
        },
        'armijo_norm_context': norm,
    }


def safe_list(x: np.ndarray):
    values = np.asarray(x, dtype=np.float64)
    return [[safe_num(v) for v in row] for row in values] if values.ndim == 2 else [safe_num(v) for v in values.reshape(-1)]


def vector_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    out = {'scalar_count': int(x.size), 'nonfinite_count': int(x.size - finite.sum())}
    if finite.any():
        y = x[finite]
        out.update(norm_l2=safe_num(np.linalg.norm(y)),
                   max_abs=safe_num(np.max(np.abs(y))),
                   nonzero_count=int(np.count_nonzero(y)))
    else:
        out.update(norm_l2=None, max_abs=None, nonzero_count=0)
    return out


def rows_stats(x: np.ndarray, node_rows: int, reserved_end: int,
               rigid_begin: int, rigid_end: int) -> dict:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    def section(a, b): return x[max(0, a):min(n, b)]
    node = section(0, node_rows)
    reserved = section(node_rows, reserved_end)
    mpm = section(reserved_end, rigid_begin)
    rigid = section(rigid_begin, rigid_end)
    tail = section(rigid_end, n)
    # Residual rows store force/impulse components in xyz; w is pressure or
    # reserved. Do not drop w from the raw report, even when current scope has
    # no mixed pressure solve.
    return {
        'fem_node_rows': {
            'all_float4': vector_stats(node),
            'xyz': vector_stats(node[:, :3]) if node.ndim == 2 else vector_stats([]),
            'w': vector_stats(node[:, 3]) if node.ndim == 2 else vector_stats([]),
        },
        'fem_middle_rows_unclassified': vector_stats(section(node_rows, reserved_end)),
        'mpm_rows': vector_stats(mpm),
        'rigid_generalized_rows': {
            'all_float4': vector_stats(rigid),
            'x_residual_component': vector_stats(rigid[:, 0]) if rigid.ndim == 2 else vector_stats([]),
        },
        'trailing_rows_unclassified': vector_stats(tail),
    }


def raw_identity(capture: Capture, name: str, phase_a: str, phase_b: str) -> dict:
    ka, kb = (phase_a, name), (phase_b, name)
    if ka not in capture.arrays or kb not in capture.arrays:
        return {'present_both': False}
    a, b = capture.raw(name, phase_a), capture.raw(name, phase_b)
    same = a == b
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return {
        'present_both': True,
        'same_extent': len(a) == len(b),
        'same_bytes': len(a) == len(b) and same,
        'pre_bytes': len(a), 'post_bytes': len(b),
        'pre_sha256': hashlib.sha256(a).hexdigest(),
        'post_sha256': hashlib.sha256(b).hexdigest(),
        'first_changed_byte': first,
        'changed_byte_count': (sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))),
    }


def get_dispatch_counts(capture: Capture, pre_phase: str) -> dict:
    name = 'dispatch_buffer'
    key = (pre_phase, name)
    if key not in capture.arrays:
        return {'available': False, 'reason': 'dispatch buffer absent'}
    meta = capture.arrays[key]
    raw = capture.raw(name, pre_phase)
    if meta['dtype'] != 'record:NMMatterDispatchGPU' or meta['count'] != 1 or len(raw) < 192:
        return {'available': False, 'reason': 'dispatch buffer has unexpected type or extent'}
    words = struct.unpack_from('<48I', raw)
    # Field indices are bound to the frozen shared.h NMMatterDispatchGPU.
    return {
        'available': True,
        'environment_count': words[2], 'fem_node_count': words[8],
        'tetrahedron_count': words[9], 'particle_count': words[6],
        'mpm_active_node_capacity': words[34],
        'rigid_generalized_capacity': words[38],
        'unknown_count_manifest': capture.manifest.get('unknown_count'),
        'expected_unknown_count': 2 * words[2] * words[8] +
            words[2] * words[34] + words[2] * words[38],
        'dispatch_sha256': hashlib.sha256(raw).hexdigest(),
    }


def residual_term_report(capture: Capture, phase_map: dict, dispatch: dict) -> dict:
    stage_names = RESIDUAL_NAMES
    stage_phase = {
        'bulk': phase_by_token(capture, PHASES['bulk']),
        'cohesive': phase_by_token(capture, PHASES['cohesive']),
        'rigid_contact': phase_by_token(capture, PHASES['rigid_contact']),
        'deformable': phase_by_token(capture, PHASES['deformable']),
        'full': phase_by_token(capture, PHASES['full']),
    }
    residuals = {}
    for stage, name in stage_names.items():
        phase = stage_phase[stage]
        residuals[stage] = f32x4(capture, name, phase).astype(np.float64)
    shapes = {k: list(v.shape) for k, v in residuals.items()}
    if len({tuple(s) for s in shapes.values()}) != 1:
        raise ValueError(f'residual checkpoint shapes differ: {shapes}')
    E = dispatch.get('environment_count')
    N = dispatch.get('fem_node_count')
    M = dispatch.get('mpm_active_node_capacity')
    R = dispatch.get('rigid_generalized_capacity')
    if None in (E, N, M, R):
        raise ValueError('dispatch layout unavailable for residual decomposition')
    node_rows = E * N
    reserved_end = 2 * E * N
    rigid_begin = reserved_end + E * M
    rigid_end = rigid_begin + E * R
    checkpoints = {k: rows_stats(v, node_rows, reserved_end, rigid_begin, rigid_end)
                   for k, v in residuals.items()}
    increments = {}
    ordered = [('bulk_from_zero', None, 'bulk'),
               ('cohesive_addition', 'bulk', 'cohesive'),
               ('rigid_contact_fem_addition', 'cohesive', 'rigid_contact'),
               ('deformable_contact_fem_addition', 'rigid_contact', 'deformable'),
               ('post_deformable_rigid_block_net', 'deformable', 'full')]
    for label, before, after in ordered:
        delta = residuals[after].copy() if before is None else residuals[after] - residuals[before]
        if after == 'full':
            increments[label] = {
                'meaning': 'Not reported as a term: the rigid generalized block is cleared and rebuilt after the preceding checkpoint, so subtracting the stale scratch is not a physical residual contribution.',
                'stale_preclear_rows_sha256': hashlib.sha256(
                    residuals[before][rigid_begin:rigid_end].astype('<f4').tobytes()
                ).hexdigest(),
                'final_rebuilt_rows_sha256': hashlib.sha256(
                    residuals[after][rigid_begin:rigid_end].astype('<f4').tobytes()
                ).hexdigest(),
            }
        else:
            increments[label] = {
                'meaning': ('FEM node rows at this stage only. The rigid/middle rows are not reported because they can contain stale scratch until the final rigid-block clear/rebuild.'),
                'fem_node_rows': rows_stats(
                    delta, node_rows, reserved_end, rigid_begin, rigid_end
                )['fem_node_rows'],
            }
    # The free-body kernels only write rows at rigidBase. Check whether adding
    # those terms changed the previously assembled FEM node rows.
    final_vs_pre_free_body = residuals['full'][:node_rows] - residuals['deformable'][:node_rows]
    increments['final_minus_deformable_fem_node_rows'] = {
        'meaning': 'should be zero if the later generalized-body assembly wrote only its rigid block',
        'stats': vector_stats(final_vs_pre_free_body),
        'max_abs_by_component': [safe_num(np.max(np.abs(final_vs_pre_free_body[:, i])))
                                 for i in range(4)],
        'bitwise_equal': capture.raw(stage_names['full'], stage_phase['full'])[:node_rows*16] ==
                         capture.raw(stage_names['deformable'], stage_phase['deformable'])[:node_rows*16],
    }
    return {
        'layout': {'fem_node_row_count': node_rows,
                   'middle_reserved_rows_end': reserved_end,
                   'mpm_rows_begin': reserved_end,
                   'rigid_rows_begin': rigid_begin,
                   'rigid_rows_end': rigid_end,
                   'residual_row_count': int(next(iter(residuals.values())).shape[0]),
                   'unclassified_ranges': [[node_rows, reserved_end], [rigid_end,
                       int(next(iter(residuals.values())).shape[0])]]},
        'phase_for_residual': stage_phase,
        'checkpoint_stats': checkpoints,
        'adjacent_checkpoint_deltas': increments,
    }


def alpha_and_norm_report(capture: Capture) -> dict:
    out = {}
    for label, token in PHASES.items():
        matches = [p for p in capture.manifest.get('producer_semantics', {}).get('phase_order', [])
                   if token == p or token in p]
        if len(matches) != 1:
            continue
        phase = matches[0]
        fields = {}
        for name in ('fem_armijo_metrics', 'fem_line_search', 'environment_line_search'):
            if (phase, name) in capture.arrays:
                vals = f32x4(capture, name, phase).astype(np.float64)
                fields[name] = safe_list(vals)
        out[phase] = fields
    return out


def candidate_identity_report(capture: Capture) -> dict:
    pre = phase_by_token(capture, PHASES['pre'])
    post = phase_by_token(capture, PHASES['post'])
    names = [
        'fem_armijo_candidate_nodes', 'fem_solution_correction',
        'tetrahedra_candidate', 'fem_fields_candidate',
        'learned_weights_candidate', 'rigid_states_trial',
        'rigid_candidate_trial', 'contact_histories_used_by_private_trial',
        'deformable_histories_used_by_private_trial',
        'fem_armijo_cohesive_faces_trial',
    ]
    input_history_names = (
        ('contact_histories_accepted', 'contact_histories_used_by_private_trial'),
        ('deformable_histories_accepted', 'deformable_histories_used_by_private_trial'),
    )
    pre_history_comparisons = {}
    for accepted_name, used_name in input_history_names:
        if (pre, accepted_name) in capture.arrays and (pre, used_name) in capture.arrays:
            a, b = capture.raw(accepted_name, pre), capture.raw(used_name, pre)
            pre_history_comparisons[used_name] = {
                'accepted_transaction_history_equals_private_input': a == b,
                'accepted_bytes': len(a), 'private_input_bytes': len(b),
                'accepted_sha256': hashlib.sha256(a).hexdigest(),
                'private_input_sha256': hashlib.sha256(b).hexdigest(),
                'interpretation': 'At Newton iterations after zero, private input may be the current Newton-accepted working history rather than transaction-accepted bytes; inequality alone is not a mutation finding.',
            }
    return {
        'pre_phase': pre, 'post_phase': post,
        'arrays': {name: raw_identity(capture, name, pre, post) for name in names},
        'pre_accepted_vs_private_history_inputs': pre_history_comparisons,
        'interpretation': 'The two contexts bracket private contact rebuild. Equality is a diagnostic identity check; differing contact candidate arrays are expected, while changed input histories or candidate geometry require investigation.',
    }


def geometry_report(capture: Capture) -> dict:
    pre = phase_by_token(capture, PHASES['pre'])
    post = phase_by_token(capture, PHASES['post'])
    n = f32record(capture, 'fem_armijo_candidate_nodes', pre, 20)
    p = f32record(capture, 'fem_armijo_candidate_nodes', post, 20)
    if n.shape != p.shape:
        raise ValueError('pre/post candidate node dimensions differ')
    # NMFEMNodeStateGPU is five float4s: position/mass [0:4], velocity/inverse
    # mass [4:8], rest/fixed [8:12], delta-v [12:16], reference displacement
    # and mode [16:20]. Preserve word-level comparisons before promotion.
    raw_pre = capture.raw('fem_armijo_candidate_nodes', pre)
    raw_post = capture.raw('fem_armijo_candidate_nodes', post)
    pos_delta = p[:, :3].astype(np.float64) - n[:, :3].astype(np.float64)
    vel_delta = p[:, 4:7].astype(np.float64) - n[:, 4:7].astype(np.float64)
    ref_delta = p[:, 16:19].astype(np.float64) - n[:, 16:19].astype(np.float64)
    accepted = f32record(capture, 'fem_accepted_nodes', pre, 20)
    linearization = f32record(capture, 'fem_linearization_nodes', pre, 20)
    correction = f32x4(capture, 'fem_solution_correction', pre)
    metrics = f32x4(capture, 'fem_armijo_metrics', pre)
    alpha = np.float32(metrics[0, 2])
    dt = np.float32(capture.manifest['dt_seconds'])
    # Re-evaluate the authored FEM candidate update in float32 operation order
    # as a consistency check. Report discrepancies; do not treat this as a
    # substitute for the production candidate bytes.
    pred_v = (linearization[:, 4:7] + alpha * correction[:len(linearization), :3]).astype('<f4')
    actual_v = n[:, 4:7]
    pred_ref = accepted[:, 16:19] + dt * pred_v
    ref_mode = accepted[:, 19] == np.float32(1.0)
    pred_pos_ref = accepted[:, 8:11] + pred_ref
    pred_pos_legacy = accepted[:, :3] + dt * pred_v
    pred_pos = np.where(ref_mode[:, None], pred_pos_ref, pred_pos_legacy).astype('<f4')
    v_diff = actual_v.astype(np.float64) - pred_v.astype(np.float64)
    pos_diff = n[:, :3].astype(np.float64) - pred_pos.astype(np.float64)
    accepted_pos = accepted[:, :3].astype(np.float64)
    trial_pos = n[:, :3].astype(np.float64)
    realized_dx = trial_pos - accepted_pos
    expected_dx = dt.astype(np.float64) * actual_v.astype(np.float64)
    ulp = np.abs(np.spacing(accepted[:, :3].astype('<f4')).astype(np.float64))
    changed_pos = n[:, :3].view('<u4') != accepted[:, :3].view('<u4')
    return {
        'trial_alpha_from_metrics_z': safe_num(alpha),
        'timestep_seconds': safe_num(dt),
        'candidate_velocity_update': {
            'expected_formula': 'fem_linearization_velocity + float32(alpha * fem_solution_correction.xyz)',
            'max_abs_actual_minus_recomputed_m_per_s': safe_num(np.max(np.abs(v_diff))),
            'changed_word_count_vs_formula': int(np.count_nonzero(actual_v.view('<u4') != pred_v.view('<u4'))),
        },
        'candidate_position_update': {
            'expected_formula': 'reference mode: rest + (accepted_u + dt * trial_v); legacy mode: accepted_x + dt * trial_v',
            'max_abs_actual_minus_recomputed_m': safe_num(np.max(np.abs(pos_diff))),
            'changed_word_count_vs_formula': int(np.count_nonzero(n[:, :3].view('<u4') != pred_pos.view('<u4'))),
        },
        'accepted_to_trial_geometry': {
            'position_components_changed': int(np.count_nonzero(changed_pos)),
            'position_components_total': int(changed_pos.size),
            'max_abs_realized_delta_x_m': safe_num(np.max(np.abs(realized_dx))),
            'max_abs_delta_x_minus_dt_v_m': safe_num(np.max(np.abs(realized_dx - expected_dx))),
            'changed_position_component_min_displacement_over_ulp': safe_num(np.min(np.abs(realized_dx[changed_pos]) / ulp[changed_pos])) if changed_pos.any() else None,
            'unchanged_position_component_max_abs_dt_v_m': safe_num(np.max(np.abs(expected_dx[~changed_pos]))) if (~changed_pos).any() else None,
        },
        'pre_post_rebuild': {
        'node_records': int(n.shape[0]),
        'pre_post_candidate_records_bitwise_equal': raw_pre == raw_post,
        'position_words_changed': int(np.count_nonzero(n[:, :3].view('<u4') != p[:, :3].view('<u4'))),
        'velocity_words_changed': int(np.count_nonzero(n[:, 4:7].view('<u4') != p[:, 4:7].view('<u4'))),
        'reference_displacement_words_changed': int(np.count_nonzero(n[:, 16:19].view('<u4') != p[:, 16:19].view('<u4'))),
        'position_pre_post_delta_m': vector_stats(pos_delta),
        'velocity_pre_post_delta_m_per_s': vector_stats(vel_delta),
        'reference_displacement_pre_post_delta_m': vector_stats(ref_delta),
        'candidate_position_max_abs_m': safe_num(np.max(np.abs(p[:, :3].astype(np.float64)))),
        'candidate_velocity_max_abs_m_per_s': safe_num(np.max(np.abs(p[:, 4:7].astype(np.float64)))),
        'reference_displacement_mode_values': sorted(set(float(x) for x in p[:, 19])),
        'meaning': 'These are pre/post private rebuild comparisons, not accepted-to-trial displacement. Accepted and linearization nodes are separately captured in the pre context.',
        },
    }


def status_report(capture: Capture) -> dict:
    result = {}
    for phase in capture.manifest.get('producer_semantics', {}).get('phase_order', []):
        key = (phase, 'fem_armijo_statuses')
        if key not in capture.arrays:
            continue
        raw = capture.raw('fem_armijo_statuses', phase)
        a = capture.arrays[key]
        if a['dtype'] != 'record:NMMatterStatusGPU' or a['element_bytes'] != 48:
            result[phase] = {'decode': 'unsupported status record layout', 'sha256': hashlib.sha256(raw).hexdigest()}
            continue
        words = struct.unpack_from('<8I', raw)
        diagnostics = struct.unpack_from('<4f', raw, 32)
        result[phase] = {
            'code': words[0], 'environment': words[1], 'object_index': words[2],
            'failing_index': words[3], 'completed_microsteps': words[4],
            'fgmres_iterations': words[5], 'contact_count': words[6],
            'event_count': words[7], 'diagnostics': [safe_num(v) for v in diagnostics],
            'sha256': hashlib.sha256(raw).hexdigest(),
        }
    return result


def measured_norm_report(capture: Capture, dispatch: dict) -> dict:
    if not dispatch.get('available'):
        return {'available': False, 'reason': 'dispatch layout unavailable'}
    phase = phase_by_token(capture, PHASES['post'])
    nodes = f32record(capture, 'fem_armijo_candidate_nodes', phase, 20)
    residual_phase = phase_by_token(capture, PHASES['full'])
    residual = f32x4(capture, 'residual_before_armijo_measure', residual_phase).astype(np.float64)
    E, N, M, R = (dispatch[k] for k in ('environment_count','fem_node_count','mpm_active_node_capacity','rigid_generalized_capacity'))
    node_rows = E * N
    rigid_begin = 2 * E * N + E * M
    fixed_or_massless = (nodes[:, 11] > 0.5) | (nodes[:, 3] <= 0.0)
    free = ~fixed_or_massless
    fem_xyz = residual[:node_rows, :3]
    rigid = residual[rigid_begin:rigid_begin + E * R, 0]
    fem_norm = float(np.linalg.norm(fem_xyz[free].reshape(-1)))
    rigid_norm = float(np.linalg.norm(rigid))
    total_norm = float(np.hypot(fem_norm, rigid_norm))
    metrics_phase = phase_by_token(capture, PHASES['measurement'])
    metrics = f32x4(capture, 'fem_armijo_metrics', metrics_phase)[0]
    return {
        'free_fem_node_count': int(free.sum()),
        'fem_free_xyz_norm_from_captured_final_residual': safe_num(fem_norm),
        'rigid_generalized_x_norm_from_captured_final_residual': safe_num(rigid_norm),
        'combined_norm_reconstructed_from_capture': safe_num(total_norm),
        'armijo_metrics_before_decision': [safe_num(v) for v in metrics],
        'trial_norm_difference_vs_metric_y': safe_num(total_norm - float(metrics[1])),
        'method': 'Matches nm_fem_armijo_measure_trial row selection, accumulated in FP64 from captured FP32 residual words; the GPU reduction is FP32 and can differ by rounding.',
    }


def derivative_data_inventory() -> dict:
    """State exactly which vectors are present for a same-generation test."""
    return {
        'schema': 'numi.cardboard.private_trial_derivative_availability.v1',
        'purpose': 'Read-only inventory of what the exact 129-array capture can support for a same-history Newton residual-update comparison.',
        'present': [
            {'array': 'fem_solution_correction', 'phase': 'candidate_before_private_contact_rebuild', 'dtype': 'float32x4', 'count': 78, 'bytes': 1248, 'meaning': 'Captured solution correction; trial velocity update is independently consistent with linearization velocity plus alpha times correction.'},
            {'array': 'fem_linearization_nodes', 'phase': 'candidate_before_private_contact_rebuild', 'dtype': 'record:NMFEMNodeStateGPU', 'count': 36, 'bytes': 2880, 'meaning': 'Newton linearization node state.'},
            {'array': 'fem_armijo_candidate_nodes', 'phase': 'candidate_before_private_contact_rebuild and private_contact_rebuild_before_free_body_terms', 'dtype': 'record:NMFEMNodeStateGPU', 'count': 36, 'bytes': 2880, 'meaning': 'Trial candidate before and after private contact rebuild; records are bitwise identical.'},
            {'array': 'residual_after_bulk_mechanical', 'phase': 'after_bulk_mechanical_before_cohesive', 'dtype': 'float32x4', 'count': 78, 'bytes': 1248, 'meaning': 'Trial-state cumulative residual checkpoint.'},
            {'array': 'residual_after_cohesive', 'phase': 'after_cohesive_before_rigid_contact', 'dtype': 'float32x4', 'count': 78, 'bytes': 1248, 'meaning': 'Trial-state cumulative residual checkpoint.'},
            {'array': 'residual_after_rigid_contact', 'phase': 'after_rigid_contact_before_deformable_rebuild', 'dtype': 'float32x4', 'count': 78, 'bytes': 1248, 'meaning': 'Trial-state cumulative residual checkpoint; generalized rigid slots here are stale scratch.'},
            {'array': 'residual_after_deformable_contact', 'phase': 'after_deformable_contact_before_free_body_terms', 'dtype': 'float32x4', 'count': 78, 'bytes': 1248, 'meaning': 'Trial-state cumulative residual checkpoint.'},
            {'array': 'residual_before_armijo_measure', 'phase': 'full_residual_before_armijo_measure', 'dtype': 'float32x4', 'count': 78, 'bytes': 1248, 'meaning': 'Full trial residual immediately before Armijo measurement.'},
            {'array': 'fem_armijo_metrics', 'phase': 'candidate and each trial stage', 'dtype': 'float32x4', 'count': 1, 'bytes': 16, 'meaning': 'Scalar norm/line-search metrics only; no residual vector at Newton center.'},
            {'array': 'rigid_states_trial and rigid_candidate_trial', 'phase': 'candidate_before_private_contact_rebuild and private_contact_rebuild_before_free_body_terms', 'meaning': 'Trial rigid state and candidate; pre/post copies match.'},
        ],
        'absent': [
            {'quantity': 'Newton-center residual vector R(x_k)', 'reason': 'Only scalar Armijo norm metrics are captured for the center; the full residual vectors are trial-state checkpoints.'},
            {'quantity': 'Production Jacobian action J(x_k) * delta', 'reason': 'No operator product/action output or same-generation linearization tangent action is among the 129 arrays.'},
            {'quantity': 'Explicit Newton-row map tying every correction/residual lane to constrained FEM DOFs and rigid generalized rows', 'reason': 'The capture exposes dispatch/counts and node records but not a dedicated serialized row-index map; preserve production mapping when extending the capture.'},
            {'quantity': 'Contact/operator generation token shared by the Newton-center tangent, correction, and trial residual', 'reason': 'Capture phases identify trial construction/rebuild ordering, but do not bind an exported J*delta to the current contact branch/history generation.'},
        ],
        'comparison_readiness': {
            'can_check_trial_residual_norm_and_trial_residual_stage_gathers': True,
            'can_compare_full_trial_residual_to_center_residual_plus_alpha_Jdelta': False,
            'reason': 'R(x_k) and J(x_k)*delta are both missing as full vectors. Correction and scalar norms alone are insufficient to identify a derivative mismatch or quantify its contribution.',
        },
        'required_followup_capture': [
            'Serialize the full Newton-center residual vector before any line-search trial, after the same production row masking/projection used by the solver.',
            'Serialize the production J*delta vector from the exact accepted Newton linearization and correction, before line-search backtracking; include operator/history/contact-generation identity.',
            'Serialize or hash the exact solver row map and constrained-DOF mask, including generalized rigid rows, and record alpha and the candidate velocity/position mapping.',
            'Retain trial residual checkpoints, accepted/current working histories, contact feature IDs/active branches, and both pre/post private-rebuild phases so changed branches are separately labeled.',
        ],
        'limits': 'This inventory only establishes whether the existing selected capture contains the vectors required for a derivative comparison; it does not identify a failing derivative layer.',
    }


def run(manifest: Path) -> dict:
    c = Capture(manifest)
    m = c.manifest
    phase_order = m.get('producer_semantics', {}).get('phase_order', [])
    observed = sorted({p for p, _ in c.arrays})
    missing = [p for p in PHASES.values() if p not in phase_order]
    if missing:
        raise ValueError(f'capture lacks expected producer phases: {missing}')
    dispatch = get_dispatch_counts(c, phase_by_token(c, PHASES['pre']))
    source_hashes = {name: sha256(path) for name, path in SOURCE_FILES.items()}
    return {
        'audit_schema': 'numi.matter.private_trial_term_audit.v1',
        'status': 'DIAGNOSTIC_ONLY_NOT_CHECKPOINT_OR_PHYSICAL_VALIDATION',
        'source_capture': {
            'manifest': str(c.path), 'manifest_sha256': sha256(c.path),
            **c.summary(),
        },
        'source_binding': {
            'snapshot_path': str(SOURCE), 'source_file_sha256': source_hashes,
            'strict_reader_path': str(READER_DIR / 'trial_capture_reader.py'),
        },
        'producer_phase_order': phase_order,
        'observed_array_phases': observed,
        'dispatch_layout': dispatch,
        'candidate_and_history_identity_pre_vs_post': candidate_identity_report(c),
        'candidate_geometry_pre_vs_post': geometry_report(c),
        'armijo_status_by_phase': status_report(c),
        'line_search_metrics_by_phase': alpha_and_norm_report(c),
        'source_reconstructed_contact_and_cohesive_assembly': source_term_reconstruction(c, dispatch),
        'residual_terms': residual_term_report(c, {}, dispatch),
        'measured_norm_crosscheck': measured_norm_report(c, dispatch),
        'newton_center_operator_data_inventory': derivative_data_inventory(),
        'causal_claim': False,
        'limitations': [
            'Residual checkpoints are cumulative FP32 buffers; per-term values are FP64 differences between adjacent serialized snapshots, not independent force evaluations.',
            'The rigid generalized residual is reported from the final rebuilt block; the preceding checkpoint contains stale pre-clear rows and is not a valid baseline for a residual-term subtraction.',
            'Cohesive and rigid/deformable FEM gathers are reconstructed from captured force/sample records and captured incidence order. Their narrowphase, CCD, constitutive response, and history evolution are not independently recomputed.',
            'The bulk mechanical residual is still a cumulative checkpoint. Its independent recomputation is approximate because the selected capture omits the FEM element-to-node incidence order.',
            'No standalone deformable TOI is exported; no TOI is inferred from endpoint samples.',
            'A single failed trial does not establish physical validity or on/off diagnostic invariance.',
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest', type=Path)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    report = run(args.manifest)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end='')

if __name__ == '__main__':
    main()
