#!/usr/bin/env python3
"""Bounded source-equation replay for native private-CCD captures 333/334.

This replays captured FP32 endpoint words in NumPy float32 arithmetic. It is
not Metal execution and deliberately reports the two common implementations
of the MSL `mix` arithmetic because last-bit rounding changes this near-grazing
case. The physical thickness and source roundoff formulas are unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

BASE = Path('/Users/home/cardboard-implementation-20261014')
READER = BASE / 'analysis/private-trial-reader-001'
sys.path.insert(0, str(READER))
from trial_capture_reader import Capture

F = np.float32
EPS = F(1.1920928955078125e-7)
GEOMETRY_FLOOR = F(1.0e-12)
EDGE_LIST = ((0, 1), (1, 2), (2, 0))
EXPECTED_NATIVE = {
    333: '8fe7d2943880bb9962b1745569fa4120d88a317fee43b00fbbe9d609cc5d07eb',
    334: 'bb645f01802e2d0cfcec1a15ec0769ff988e8f06b1ce4a58610bc82c8c7f6e38',
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32v(raw: bytes, count: int = 4) -> np.ndarray:
    return np.frombuffer(raw, dtype='<f4').reshape(-1, count)


def records(raw: bytes, stride: int) -> np.ndarray:
    if len(raw) % stride:
        raise ValueError('record payload is not stride aligned')
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, stride)


def sequential_dot(a: np.ndarray, b: np.ndarray) -> np.float32:
    value = F(0)
    for x, y in zip(a, b):
        value = F(value + F(F(x) * F(y)))
    return value


def norm3(v: np.ndarray) -> np.float32:
    return F(np.sqrt(sequential_dot(v, v), dtype=np.float32))


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray([
        F(F(a[1] * b[2]) - F(a[2] * b[1])),
        F(F(a[2] * b[0]) - F(a[0] * b[2])),
        F(F(a[0] * b[1]) - F(a[1] * b[0])),
    ], dtype=np.float32)


def mix_difference(a: np.ndarray, b: np.ndarray, t: np.float32) -> np.ndarray:
    # Algebraic affine form x + t*(y-x), evaluated with FP32 intermediates.
    return np.asarray([F(a[k] + F(t * F(b[k] - a[k]))) for k in range(3)], dtype=np.float32)


def mix_weighted(a: np.ndarray, b: np.ndarray, t: np.float32) -> np.ndarray:
    # MSL mix's weighted form x*(1-t) + y*t, with FP32 intermediates.
    one_minus_t = F(F(1) - t)
    return np.asarray([F(F(a[k] * one_minus_t) + F(b[k] * t)) for k in range(3)], dtype=np.float32)


def maxabs(v: np.ndarray) -> np.float32:
    return F(max(abs(float(x)) for x in v))


def closest_segments(p0, p1, q0, q1):
    # Source-order FP32 form from closestPointsOnSegments.
    d_p = p1 - p0
    d_q = q1 - q0
    r = p0 - q0
    a = sequential_dot(d_p, d_p)
    e = sequential_dot(d_q, d_q)
    f = sequential_dot(d_q, r)
    if a <= GEOMETRY_FLOOR and e <= GEOMETRY_FLOOR:
        s = tt = F(0)
    elif a <= GEOMETRY_FLOOR:
        s = F(0)
        tt = F(np.clip(F(f / max(e, GEOMETRY_FLOOR)), F(0), F(1)))
    else:
        c = sequential_dot(d_p, r)
        if e <= GEOMETRY_FLOOR:
            tt = F(0)
            s = F(np.clip(F(-c / max(a, GEOMETRY_FLOOR)), F(0), F(1)))
        else:
            b = sequential_dot(d_p, d_q)
            denominator = F(F(a * e) - F(b * b))
            if denominator > GEOMETRY_FLOOR:
                s = F(np.clip(F(F(F(b * f) - F(c * e)) / denominator), F(0), F(1)))
            else:
                s = F(0)
            tt = F(F(F(b * s) + f) / e)
            if tt < 0:
                tt = F(0)
                s = F(np.clip(F(-c / a), F(0), F(1)))
            elif tt > 1:
                tt = F(1)
                s = F(np.clip(F(F(b - c) / a), F(0), F(1)))
    point_p = np.asarray([F(p0[k] + F(s * F(p1[k] - p0[k]))) for k in range(3)], dtype=np.float32)
    point_q = np.asarray([F(q0[k] + F(tt * F(q1[k] - q0[k]))) for k in range(3)], dtype=np.float32)
    return s, tt, point_p, point_q


def swept_features_separated(first_start, first_finish, second_start, second_finish, axis, effective):
    axis_length = norm3(axis)
    if not (axis_length > GEOMETRY_FLOOR and np.isfinite(axis_length)):
        return False
    n = np.asarray([F(x / axis_length) for x in axis], dtype=np.float32)
    minimum = F(np.inf)
    scale = F(effective)
    for a0, a1 in zip(first_start, first_finish):
        for b0, b1 in zip(second_start, second_finish):
            begin = a0 - b0
            end = a1 - b1
            if not np.all(np.isfinite(begin)) or not np.all(np.isfinite(end)):
                return False
            minimum = min(minimum, sequential_dot(n, begin), sequential_dot(n, end))
            scale = max(scale, maxabs(begin), maxabs(end))
    projection_roundoff = F(F(16) * EPS * scale)
    return bool(np.isfinite(minimum) and minimum > F(effective + projection_roundoff))


def closest_point_triangle(point, a, b, c):
    """FP32 closest-point regions in source order."""
    ab, ac, ap = b-a, c-a, point-a
    d1, d2 = sequential_dot(ab, ap), sequential_dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a.copy(), 'a'
    bp = point-b
    d3, d4 = sequential_dot(ab, bp), sequential_dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b.copy(), 'b'
    vc = F(F(d1*d4)-F(d3*d2))
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = F(d1/max(F(d1-d3), GEOMETRY_FLOOR))
        return a+F(v*ab), 'ab'
    cp = point-c
    d5, d6 = sequential_dot(ab, cp), sequential_dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c.copy(), 'c'
    vb = F(F(d5*d2)-F(d1*d6))
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = F(d2/max(F(d2-d6), GEOMETRY_FLOOR))
        return a+F(w*ac), 'ac'
    va = F(F(d3*d6)-F(d5*d4))
    if va <= 0 and F(d4-d3) >= 0 and F(d5-d6) >= 0:
        w = F(F(d4-d3)/max(F(F(d4-d3)+F(d5-d6)), GEOMETRY_FLOOR))
        return b+F(w*(c-b)), 'bc'
    denominator = max(F(F(va+vb)+vc), GEOMETRY_FLOOR)
    v, w = F(vb/denominator), F(vc/denominator)
    return a+F(v*ab)+F(w*ac), 'face'


def vt_result(vertex_start, vertex_finish, tri_start, tri_finish, thickness, mixfn, budget=16):
    points = [vertex_start, vertex_finish, *tri_start, *tri_finish]
    coordinate_scale = max(maxabs(p) for p in points)
    effective = F(thickness + F(F(8)*EPS*max(F(thickness), coordinate_scale, GEOMETRY_FLOOR)))
    speed = F(norm3(vertex_finish-vertex_start) + max(
        norm3(tri_finish[i]-tri_start[i]) for i in range(3)))
    time = F(0)
    trace = []
    for iteration in range(budget):
        p = mixfn(vertex_start, vertex_finish, time)
        tri = [mixfn(tri_start[i], tri_finish[i], time) for i in range(3)]
        closest, region = closest_point_triangle(p, *tri)
        delta = p-closest
        distance = norm3(delta)
        trace.append({'iteration': iteration, 'time': float(time), 'distance_m': float(distance), 'region': region})
        if distance <= effective:
            return {'certified': True, 'hit': True, 'reason': 'thickness_hit', 'iterations': iteration+1,
                    'time': float(time), 'effective_thickness_m': float(effective), 'trace': trace}
        if swept_features_separated([vertex_start], [vertex_finish], tri_start, tri_finish, delta, effective):
            return {'certified': True, 'hit': False, 'reason': 'fixed_axis_separation', 'iterations': iteration+1,
                    'time': float(time), 'effective_thickness_m': float(effective), 'trace': trace}
        if iteration == 0:
            for axis in (cross3(tri_start[1]-tri_start[0], tri_start[2]-tri_start[0]),
                         cross3(tri_finish[1]-tri_finish[0], tri_finish[2]-tri_finish[0])):
                if sequential_dot(axis, vertex_start-tri_start[0]) < 0:
                    axis = -axis
                if swept_features_separated([vertex_start], [vertex_finish], tri_start, tri_finish, axis, effective):
                    return {'certified': True, 'hit': False, 'reason': 'triangle_axis_separation', 'iterations': iteration+1,
                            'time': float(time), 'effective_thickness_m': float(effective), 'trace': trace}
        if not (speed > GEOMETRY_FLOOR):
            return {'certified': True, 'hit': False, 'reason': 'zero_speed', 'iterations': iteration+1,
                    'time': float(time), 'effective_thickness_m': float(effective), 'trace': trace}
        time = F(time + max(F(F(distance-effective)/speed), F(1e-6)))
        if time > F(F(1)+F(F(8)*EPS)):
            return {'certified': True, 'hit': False, 'reason': 'conservative_step_passes_endpoint', 'iterations': iteration+1,
                    'time': float(time), 'effective_thickness_m': float(effective), 'trace': trace}
        time = min(time, F(1))
    return {'certified': False, 'hit': False, 'reason': 'iteration_budget_exhausted', 'iterations': budget,
            'time': float(time), 'effective_thickness_m': float(effective), 'trace': trace}


def ca_result(kind, start_a, finish_a, start_b, finish_b, thickness, mixfn, budget):
    all_points = [*start_a, *finish_a, *start_b, *finish_b]
    coordinate_scale = F(0)
    for point in all_points:
        coordinate_scale = max(coordinate_scale, maxabs(point))
    distance_roundoff = F(F(8) * EPS * max(F(thickness), coordinate_scale, GEOMETRY_FLOOR))
    effective = F(thickness + distance_roundoff)
    if kind == 'ee':
        speed = F(max(norm3(finish_a[0]-start_a[0]), norm3(finish_a[1]-start_a[1])) +
                  max(norm3(finish_b[0]-start_b[0]), norm3(finish_b[1]-start_b[1])))
    else:
        speed = F(norm3(finish_a[0]-start_a[0]) + max(
            norm3(finish_b[0]-start_b[0]), norm3(finish_b[1]-start_b[1]), norm3(finish_b[2]-start_b[2])))
    time = F(0)
    trace = []
    for iteration in range(budget):
        if kind == 'ee':
            p0, p1 = mixfn(start_a[0], finish_a[0], time), mixfn(start_a[1], finish_a[1], time)
            q0, q1 = mixfn(start_b[0], finish_b[0], time), mixfn(start_b[1], finish_b[1], time)
            s, t, cp, cq = closest_segments(p0, p1, q0, q1)
            separation = norm3(cp-cq)
            trace.append({'iteration': iteration, 'time': float(time), 'distance_m': float(separation),
                          'segment_parameters': [float(s), float(t)]})
            if separation <= effective:
                return {'certified': True, 'hit': True, 'reason': 'thickness_hit', 'iterations': iteration+1,
                        'time': float(time), 'effective_thickness_m': float(effective), 'speed_bound_m': float(speed), 'trace': trace}
            if swept_features_separated(start_a, finish_a, start_b, finish_b, cp-cq, effective):
                return {'certified': True, 'hit': False, 'reason': 'fixed_axis_separation', 'iterations': iteration+1,
                        'time': float(time), 'effective_thickness_m': float(effective), 'speed_bound_m': float(speed), 'trace': trace}
            if iteration == 0:
                axes = [cross3(start_a[1]-start_a[0], start_b[1]-start_b[0]),
                        cross3(finish_a[1]-finish_a[0], finish_b[1]-finish_b[0])]
                for axis in axes:
                    if sequential_dot(axis, start_a[0]-start_b[0]) < 0:
                        axis = -axis
                    if swept_features_separated(start_a, finish_a, start_b, finish_b, axis, effective):
                        return {'certified': True, 'hit': False, 'reason': 'edge_axis_separation', 'iterations': iteration+1,
                                'time': float(time), 'effective_thickness_m': float(effective), 'speed_bound_m': float(speed), 'trace': trace}
        else:
            raise NotImplementedError('this audit replays only the captured edge-edge feature')
        if not (speed > GEOMETRY_FLOOR):
            return {'certified': True, 'hit': False, 'reason': 'zero_speed', 'iterations': iteration+1,
                    'time': float(time), 'effective_thickness_m': float(effective), 'speed_bound_m': float(speed), 'trace': trace}
        distance = F(trace[-1]['distance_m'])
        step = max(F(F(distance-effective)/speed), F(1.0e-6))
        time = F(time + step)
        if time > F(F(1) + F(F(8)*EPS)):
            return {'certified': True, 'hit': False, 'reason': 'conservative_step_passes_endpoint', 'iterations': iteration+1,
                    'time': float(time), 'effective_thickness_m': float(effective), 'speed_bound_m': float(speed), 'trace': trace}
        time = min(time, F(1))
    return {'certified': False, 'hit': False, 'reason': 'iteration_budget_exhausted', 'iterations': budget,
            'time': float(time), 'effective_thickness_m': float(effective), 'speed_bound_m': float(speed), 'trace': trace}


def closest_segments64(p0, p1, q0, q1):
    """Independent double-precision segment-distance calculation."""
    p0, p1, q0, q1 = (np.asarray(v, dtype=np.float64) for v in (p0, p1, q0, q1))
    dp, dq, r = p1-p0, q1-q0, p0-q0
    a, e, f = float(np.dot(dp, dp)), float(np.dot(dq, dq)), float(np.dot(dq, r))
    if a <= 1e-30 and e <= 1e-30:
        s = t = 0.0
    elif a <= 1e-30:
        s, t = 0.0, float(np.clip(f/e, 0.0, 1.0))
    else:
        c = float(np.dot(dp, r))
        if e <= 1e-30:
            t, s = 0.0, float(np.clip(-c/a, 0.0, 1.0))
        else:
            b = float(np.dot(dp, dq))
            den = a*e - b*b
            s = float(np.clip((b*f-c*e)/den, 0.0, 1.0)) if den > 1e-30 else 0.0
            t = (b*s+f)/e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c/a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b-c)/a, 0.0, 1.0))
    delta = (p0+s*dp) - (q0+t*dq)
    return s, t, float(np.linalg.norm(delta))


def fp64_sweep_diagnostic(start_a, finish_a, start_b, finish_b, sample_count=10001):
    """Numerically scan continuous double geometry; descriptive, not a certificate."""
    ts = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    best = None
    for t in ts:
        a0 = np.asarray(start_a[0], np.float64) + t*(np.asarray(finish_a[0], np.float64)-start_a[0])
        a1 = np.asarray(start_a[1], np.float64) + t*(np.asarray(finish_a[1], np.float64)-start_a[1])
        b0 = np.asarray(start_b[0], np.float64) + t*(np.asarray(finish_b[0], np.float64)-start_b[0])
        b1 = np.asarray(start_b[1], np.float64) + t*(np.asarray(finish_b[1], np.float64)-start_b[1])
        s, q, distance = closest_segments64(a0, a1, b0, b1)
        if best is None or distance < best['distance_m']:
            best = {'time': float(t), 'distance_m': distance, 'segment_parameters': [s, q]}
    return best


def capture_feature(manifest_path: Path):
    cap = Capture(manifest_path)
    m = cap.manifest
    case_id = int(manifest_path.resolve().parent.name.rsplit('-', 1)[-1])
    if case_id not in EXPECTED_NATIVE:
        raise ValueError(f'unexpected native case directory {case_id}')
    payload_hash = sha(cap.payload_path)
    if payload_hash != EXPECTED_NATIVE[case_id]:
        raise ValueError(f'capture {case_id} payload hash changed')
    expected_trial = 20 if case_id == 333 else 21
    if (m['root'], m['iteration'], m['trial'], m['selected_phase']) != (1201, 12, expected_trial, 'both'):
        raise ValueError(f'unexpected capture identity for native case {case_id}')
    phase = 'private_contact_rebuild_before_free_body_terms'
    candidates = records(cap.raw('deformable_candidates_private_trial', phase), 16)
    candidate = np.frombuffer(candidates[331], dtype='<u4')
    if tuple(int(x) for x in candidate) != (88, 127, 331, 1):
        raise ValueError(f'unexpected candidate identity {candidate.tolist()}')
    primitives = records(cap.raw('surface_primitives_private_trial', phase), 64)
    primitive_nodes = [np.frombuffer(primitives[index], dtype='<u4')[:3].astype(int).tolist() for index in (88, 127)]
    if primitive_nodes != [[13, 14, 17], [28, 32, 31]]:
        raise ValueError(f'unexpected primitive node mapping {primitive_nodes}')
    states = records(cap.raw('fem_armijo_candidate_nodes', phase), 80)
    dispatch = f32v(cap.raw('dispatch_buffer', 'candidate_before_private_contact_rebuild'))
    dispatch_u32 = np.frombuffer(cap.raw('dispatch_buffer', 'candidate_before_private_contact_rebuild'), dtype='<u4')
    schedulers = cap.raw('schedulers', 'candidate_before_private_contact_rebuild')
    scheduler_u32 = np.frombuffer(schedulers, dtype='<u4')
    base_dt = F(dispatch.reshape(-1)[43])
    exponent = int(scheduler_u32[1])
    dt = F(base_dt / F(2 ** exponent))
    thickness = F(dispatch.reshape(-1)[44])
    if not (dt > 0 and thickness > 0 and np.isfinite(dt) and np.isfinite(thickness)):
        raise ValueError('invalid captured dt/thickness')
    start_by_id, finish_by_id = {}, {}
    for node_id in primitive_nodes[0] + primitive_nodes[1]:
        state = np.frombuffer(states[node_id], dtype='<f4')
        finish = np.asarray(state[0:3], dtype=np.float32).copy()
        velocity = np.asarray(state[4:7], dtype=np.float32).copy()
        start = np.asarray([F(finish[k] - F(dt * velocity[k])) for k in range(3)], dtype=np.float32)
        start_by_id[node_id] = start
        finish_by_id[node_id] = finish
    pair_a, pair_b = [13, 14], [28, 32]
    all_ids = primitive_nodes[0] + primitive_nodes[1]
    all_start = [start_by_id[n] for n in all_ids]
    all_finish = [finish_by_id[n] for n in all_ids]
    start_a = [start_by_id[n] for n in pair_a]
    finish_a = [finish_by_id[n] for n in pair_a]
    start_b = [start_by_id[n] for n in pair_b]
    finish_b = [finish_by_id[n] for n in pair_b]
    return cap, dict(native_case=case_id, root=int(m['root']), iteration=int(m['iteration']), trial=int(m['trial']),
        selected_phase=m['selected_phase'], payload_sha256=payload_hash,
        candidate=candidate.tolist(), primitive_nodes=primitive_nodes,
        all_node_ids=all_ids, all_start_f32=[v.tolist() for v in all_start],
        all_finish_f32=[v.tolist() for v in all_finish],
        edge_node_ids=[pair_a, pair_b], base_dt_seconds=float(base_dt),
        scheduler_exponent=exponent, object_dt_seconds=float(dt),
        thickness_m=float(thickness), start_a=[v.tolist() for v in start_a],
        finish_a=[v.tolist() for v in finish_a], start_b=[v.tolist() for v in start_b],
        finish_b=[v.tolist() for v in finish_b], dispatch_buffer_sha256=hashlib.sha256(cap.raw('dispatch_buffer', 'candidate_before_private_contact_rebuild')).hexdigest()), start_a, finish_a, start_b, finish_b, thickness


def build_report(capture_paths):
    captured = []
    for path in capture_paths:
        cap, item, sa, fa, sb, fb, thickness = capture_feature(Path(path))
        replays = {}
        for name, mixfn in (('difference_form_fp32', mix_difference), ('weighted_form_fp32', mix_weighted)):
            budgets = {}
            for budget in (16, 17, 32):
                budgets[str(budget)] = ca_result('ee', sa, fa, sb, fb, thickness, mixfn, budget)
            replays[name] = budgets
        vt_replays = {}
        for name, mixfn in (('difference_form_fp32', mix_difference), ('weighted_form_fp32', mix_weighted)):
            checks = []
            all_start = [np.asarray(v, dtype=np.float32) for v in item['all_start_f32']]
            all_finish = [np.asarray(v, dtype=np.float32) for v in item['all_finish_f32']]
            for direction in range(2):
                vertex_base, triangle_base = ((0, 3) if direction == 0 else (3, 0))
                for vertex_index in range(3):
                    checks.append({'direction': direction, 'vertex_index': vertex_index,
                        **vt_result(all_start[vertex_base+vertex_index], all_finish[vertex_base+vertex_index],
                                    [all_start[triangle_base+i] for i in range(3)],
                                    [all_finish[triangle_base+i] for i in range(3)],
                                    thickness, mixfn, 16)})
            vt_replays[name] = checks
        item['vertex_triangle_source_order'] = vt_replays
        # A synthetic, explicitly non-native true-hit control: shift both B-edge endpoints
        # toward A by 50 nm throughout the same segment; do not alter thickness.
        make_shifted = lambda amount: (
            [np.asarray([F(x - (F(amount) if k == 2 else F(0))) for k, x in enumerate(p)], dtype=np.float32) for p in sb],
            [np.asarray([F(x - (F(amount) if k == 2 else F(0))) for k, x in enumerate(p)], dtype=np.float32) for p in fb])
        hit_start_50, hit_finish_50 = make_shifted(50e-9)
        hit_start_60, hit_finish_60 = make_shifted(60e-9)
        hit_controls = {
          'swept_hit_shift_50nm_budget32': {name: ca_result('ee', sa, fa, hit_start_50, hit_finish_50, thickness, mixfn, 32)
            for name, mixfn in (('difference_form_fp32', mix_difference), ('weighted_form_fp32', mix_weighted))},
          'initial_hit_shift_60nm_budget17': {name: ca_result('ee', sa, fa, hit_start_60, hit_finish_60, thickness, mixfn, 17)
            for name, mixfn in (('difference_form_fp32', mix_difference), ('weighted_form_fp32', mix_weighted))}
        }
        item['replays'] = replays
        item['synthetic_true_hit_controls_are_non_native'] = True
        item['synthetic_true_hit_controls'] = hit_controls
        item['fp64_geometry_sweep'] = {
          'method': '10001-point double-precision sweep of captured FP32 start/finish endpoint words; descriptive only, not a proof',
          'minimum': fp64_sweep_diagnostic(sa, fa, sb, fb),
          'start_distance_m': closest_segments64(sa[0], sa[1], sb[0], sb[1])[2],
          'end_distance_m': closest_segments64(fa[0], fa[1], fb[0], fb[1])[2],
          'source_effective_thickness_m': replays['difference_form_fp32']['17']['effective_thickness_m']
        }
        for key, vectors in (('start_a', sa), ('finish_a', fa), ('start_b', sb), ('finish_b', fb)):
            item[key + '_f32_bits_u32'] = [[int(x) for x in np.asarray(v, dtype='<f4').view('<u4')] for v in vectors]
        item['native_case_directory'] = str(Path(path).resolve().parent)
        captured.append(item)
    report = {
      'schema': 'numi.cardboard.private_ccd_captured_near_miss_replay.v1',
      'evidence_class': 'CPU source-equation replay; not bit-exact Metal and not native qualification',
      'source_snapshot': str(BASE / 'snapshots/source-center-action-twoenv-restart-099'),
      'source_contact_metalinc_sha256': sha(BASE / 'snapshots/source-center-action-twoenv-restart-099/matter/src/metal/contact.metalinc'),
      'reader_sha256': sha(READER / 'trial_capture_reader.py'),
      'captures': captured,
      'interpretation': {
        'failure_feature': 'CPU source-order replay identifies edge-edge feature (edgeA=0, edgeB=0), nodes [13,14] vs [28,32], as first uncertified after all six preceding vertex-triangle tests certify.',
        'native_diagnostic_ambiguity': 'The native status diagnostic [primitiveA=88, primitiveB=127, z=0, w=0] aliases vertex-triangle vertex 0 and edge-edge (0,0); native status alone does not identify feature kind. The CPU source-order replay localizes the first uncaught branch, but is not a Metal instruction trace.',
        'rounding_sensitivity': 'Two FP32 evaluation forms for the MSL mix operation can differ at the final conservative step. The difference-form replay exhausts at 16 and certifies the miss at 17; weighted-form replay may certify by 16. This is why the one-step extension is a bounded robustness candidate, not proof that a native rerun will pass.',
        'effective_thickness': 'Computed by the frozen source formula: thickness + ccdDistanceRoundoff(thickness, coordinateScale). No gap, radius, epsilon, or feasibility threshold is changed.',
        'control_limit': 'The 50 nm translated-edge true-hit case is a synthetic replay control, not a native fixture or physical measurement.'
      },
      'physical_validation': False,
      'accepted_checkpoint': False
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', type=Path, action='append', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    report = build_report(args.capture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'captures': len(report['captures']),
                      'source_sha256': report['source_contact_metalinc_sha256'],
                      'payload_sha256': [x['payload_sha256'] for x in report['captures']]}, indent=2))

if __name__ == '__main__':
    main()
