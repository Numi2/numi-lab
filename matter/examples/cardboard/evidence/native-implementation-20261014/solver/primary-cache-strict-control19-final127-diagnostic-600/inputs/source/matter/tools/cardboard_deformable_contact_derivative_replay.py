#!/usr/bin/env python3
"""CPU-only replay of a bounded selected FEM/deformable-contact capture.

The tool reconstructs IPC contact residuals from captured active contacts and
candidate node states. It compares smooth and fixed-center-feature directional
actions with one-sided finite differences and the captured native action. It
does not re-run native contact generation, CCD, history updates, or a complete
nonlinear solve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

PREFIX = "fem_deformable_contact_capture="
MAX_CAPTURE_BYTES = 4 * 1024 * 1024
NODE_STRIDE = 80
CONTACT_STRIDE = 432
FLOAT4_STRIDE = 16
INVALID_NODE = 0xFFFFFFFF
CONTACT_VALID = 1 << 0
CONTACT_MAPPED_FEATURE = 1 << 3
K_GEOMETRY_FLOOR = 1.0e-12
DEFAULT_EPSILONS = (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6)


def _f4(raw: bytes, offset: int):
    return struct.unpack_from("<4f", raw, offset)


def _u4(raw: bytes, offset: int):
    return struct.unpack_from("<4I", raw, offset)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def scale(s, a):
    return [s * x for x in a]


def norm(a):
    return math.sqrt(dot(a, a))


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def decode_contact(raw: bytes) -> dict:
    if len(raw) != CONTACT_STRIDE:
        raise ValueError(f"contact record is {len(raw)} bytes, expected {CONTACT_STRIDE}")
    bary_x, bary_y, bary_z = _f4(raw, 384), _f4(raw, 400), _f4(raw, 416)
    return {
        "nodes": _u4(raw, 0) + _u4(raw, 288)[:2],
        "identity": _u4(raw, 16),
        "source": _u4(raw, 112),
        "weights": _f4(raw, 32) + _f4(raw, 304)[:2],
        "time_scales": _f4(raw, 48) + _f4(raw, 320)[:2],
        "normal_toi": _f4(raw, 64),
        "point_and_separation": _f4(raw, 80),
        "relative_velocity_response": _f4(raw, 96),
        "impulse_friction": _f4(raw, 128),
        "barrier": _f4(raw, 144),
        "hessian_rows": [_f4(raw, offset) for offset in (160, 176, 192)],
        "mollifier_gradients": [
            _f4(raw, offset)[:3] for offset in (208, 224, 240, 256, 336, 352)
        ],
        "mollifier": _f4(raw, 272),
        "feature_sides": _u4(raw, 368),
        "feature_barycentric": [
            [bary_x[i], bary_y[i], bary_z[i]] for i in range(4)
        ],
    }


def parse_capture_line(text: str, root: int | None = None,
                       iteration: int | None = None) -> dict:
    rows = [json.loads(line[len(PREFIX):]) for line in text.splitlines()
            if line.startswith(PREFIX)]
    if not rows:
        skipped = [line for line in text.splitlines()
                   if line.startswith("fem_deformable_contact_capture_skipped=")]
        raise ValueError(skipped[-1] if skipped else "no selected deformable-contact capture")
    matches = [row for row in rows
               if (root is None or row.get("root") == root) and
               (iteration is None or row.get("iteration") == iteration)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one matching capture; found {len(matches)}")
    row = matches[0]
    if row.get("schema") != 1 or row.get("stage") != "after_deformable_limit":
        raise ValueError("unsupported capture schema or stage")
    if row.get("microtick") != 0:
        raise ValueError("capture is not from owner microtick zero")
    if not (math.isfinite(row.get("dt_s", math.nan)) and row["dt_s"] > 0.0):
        raise ValueError("capture has invalid dt_s")
    if row.get("capture_bytes", MAX_CAPTURE_BYTES + 1) > MAX_CAPTURE_BYTES:
        raise ValueError("capture exceeds 4 MiB readback bound")
    exact_action_capture = row.get("production_exact_action_capture", False)
    if type(exact_action_capture) is not bool:
        raise ValueError("malformed production exact-action capture marker")
    if exact_action_capture and (
            row.get("production_exact_action_source") !=
                "deformableExactVertexTriangleDirectionalForce" or
            row.get("production_exact_action_direction") != "femSolution" or
            row.get("production_exact_action_support_order") !=
                "contact.nodes[0..3]" or
            row.get("production_exact_action_quantity") !=
                "native_fgmres_operator_action"):
        raise ValueError("unsupported production exact-action provenance")
    row["production_exact_action_capture"] = exact_action_capture
    feature_policy = row.get("production_exact_action_feature_policy",
                             "strict_smooth_feature")
    if feature_policy not in ("strict_smooth_feature", "fixed_candidate_feature"):
        raise ValueError("unsupported production exact-action feature policy")
    row["_production_exact_action_feature_policy"] = feature_policy
    envs, node_count, grid_count = (
        row.get("environment_count"), row.get("fem_node_count"),
        row.get("grid_node_count"))
    slots, object_count = row.get("trace_slots_per_environment"), row.get("object_count")
    if not all(isinstance(x, int) and x > 0 for x in
               (envs, node_count, slots, object_count)):
        raise ValueError("invalid capture dimensions")
    if not isinstance(grid_count, int) or grid_count < 0:
        raise ValueError("invalid grid_node_count")
    counts, truncated = row.get("active_counts"), row.get("truncated")
    if not (isinstance(counts, list) and isinstance(truncated, list) and
            len(counts) == envs and len(truncated) == envs):
        raise ValueError("active counts/truncation do not match environment count")
    if any(type(c) is not int or c < 0 for c in counts):
        raise ValueError("malformed active count")
    if any(type(t) is not bool for t in truncated):
        raise ValueError("malformed truncation marker")
    if any(truncated[e] != (counts[e] > slots) for e in range(envs)):
        raise ValueError("truncation marker disagrees with active count")
    if (row.get("node_state_stride_bytes"), row.get("contact_stride_bytes"),
        row.get("solution_stride_bytes")) != (NODE_STRIDE, CONTACT_STRIDE, FLOAT4_STRIDE):
        raise ValueError("native ABI stride mismatch")
    record_stride = row.get("record_stride_bytes")
    if not isinstance(record_stride, int) or record_stride <= CONTACT_STRIDE:
        raise ValueError("invalid trace record stride")
    nodes_raw = bytes.fromhex(row.get("candidate_nodes_hex", ""))
    solution_raw = bytes.fromhex(row.get("solution_hex", ""))
    if len(nodes_raw) != envs * node_count * NODE_STRIDE:
        raise ValueError("candidate node arena byte count mismatch")
    if len(solution_raw) != envs * node_count * FLOAT4_STRIDE:
        raise ValueError("solution arena byte count mismatch")
    if row.get("candidate_node_bytes") != len(nodes_raw) or \
       row.get("solution_bytes") != len(solution_raw):
        raise ValueError("captured arena metadata mismatch")
    records = row.get("contacts")
    if not isinstance(records, list) or len(records) != row.get("recorded_count"):
        raise ValueError("contact record count mismatch")
    seen = set()
    for record in records:
        env, active_slot, slot = (record.get("environment"), record.get("active_slot"),
                                  record.get("contact_slot"))
        if not (isinstance(env, int) and 0 <= env < envs and
                isinstance(active_slot, int) and 0 <= active_slot < slots and
                isinstance(slot, int) and slot >= 0):
            raise ValueError("malformed active contact identity")
        if active_slot >= counts[env] or (env, active_slot) in seen:
            raise ValueError("duplicate or out-of-count active contact")
        seen.add((env, active_slot))
        raw = bytes.fromhex(record.get("record_hex", ""))
        contact = decode_contact(raw)
        if list(contact["identity"]) != record.get("identity"):
            raise ValueError("raw contact identity disagrees with trace metadata")
        if exact_action_capture:
            markers = record.get("production_exact_action_supported")
            actions = record.get("production_exact_action")
            action_dt = record.get("production_exact_action_dt_s")
            if (not isinstance(markers, list) or len(markers) != 4 or
                    any(type(value) is not int or value not in (0, 1)
                        for value in markers)):
                raise ValueError("malformed production exact-action support markers")
            if not (type(action_dt) in (int, float) and
                    math.isfinite(action_dt) and action_dt > 0.0):
                raise ValueError("malformed production exact-action timestep")
            if (not isinstance(actions, list) or len(actions) != 4 or
                    any(not isinstance(vector, list) or len(vector) != 3 or
                        any(type(value) not in (int, float) or
                            not math.isfinite(value) for value in vector)
                        for vector in actions)):
                raise ValueError("malformed production exact-action vectors")
            record["_production_exact_action_supported"] = markers
            record["_production_exact_action"] = actions
            record["_production_exact_action_dt_s"] = action_dt
        record["_decoded"] = contact
    if len(records) > sum(min(count, slots) for count in counts):
        raise ValueError("capture has more records than bounded active slots")
    record_bytes = envs * slots * record_stride
    count_pad = (envs * 4 + 15) & ~15
    line_bytes = envs * object_count * FLOAT4_STRIDE
    expected = record_bytes + count_pad + line_bytes + len(nodes_raw) + len(solution_raw)
    if row.get("capture_bytes") != expected:
        raise ValueError("total capture bytes disagree with declared arenas")
    row["_candidate_nodes"] = nodes_raw
    row["_solution"] = solution_raw
    row["_complete_active_coverage"] = (
        not any(truncated) and all(
            sum(r["environment"] == env for r in records) == counts[env]
            for env in range(envs)))
    return row


def _node_state(raw: bytes, env: int, node: int, count: int):
    values = struct.unpack_from("<20f", raw, (env * count + node) * NODE_STRIDE)
    return {"position": list(values[:3]), "mass": values[3],
            "velocity": list(values[4:7]), "inverse_mass": values[7],
            "rest": list(values[8:11]), "fixed": values[11],
            "delta_velocity": list(values[12:15]), "u": list(values[16:19]),
            "reference_mode": values[19]}


def _direction(raw: bytes, env: int, node: int, count: int):
    return list(struct.unpack_from("<3f", raw, (env * count + node) * FLOAT4_STRIDE))


def _triangle_closest(p, a, b, c):
    ab, ac, ap = sub(b, a), sub(c, a), sub(p, a)
    d1, d2 = dot(ab, ap), dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, [1.0, 0.0, 0.0], "vertex0"
    bp = sub(p, b)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, [0.0, 1.0, 0.0], "vertex1"
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / max(d1 - d3, K_GEOMETRY_FLOOR)
        return add(a, scale(v, ab)), [1.0 - v, v, 0.0], "edge01"
    cp = sub(p, c)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, [0.0, 0.0, 1.0], "vertex2"
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / max(d2 - d6, K_GEOMETRY_FLOOR)
        return add(a, scale(w, ac)), [1.0 - w, 0.0, w], "edge02"
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        w = (d4 - d3) / max((d4 - d3) + (d5 - d6), K_GEOMETRY_FLOOR)
        return add(b, scale(w, sub(c, b))), [0.0, 1.0 - w, w], "edge12"
    den = max(va + vb + vc, K_GEOMETRY_FLOOR)
    v, w = vb / den, vc / den
    return add(add(a, scale(v, ab)), scale(w, ac)), [1.0 - v - w, v, w], "face"


def _triangle_closest_directional(point, a, b, c, dp, da, db, dc):
    """Return closest-feature data and its exact fixed-branch derivative."""
    ab, ac, ap = sub(b, a), sub(c, a), sub(point, a)
    d1, d2 = dot(ab, ap), dot(ac, ap)
    bp = sub(point, b)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    cp = sub(point, c)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    scale2 = max(dot(ab, ab), dot(ac, ac), dot(sub(c, b), sub(c, b)), 1.0e-30)
    dot_guard = 64.0 * (2.0 ** -23) * scale2
    det_guard = 64.0 * (2.0 ** -23) * scale2 * scale2

    if d1 <= 0.0 and d2 <= 0.0:
        bary, branch = [1.0, 0.0, 0.0], "vertex0"
        q, dq, dbary = a, da, [0.0, 0.0, 0.0]
        smooth = min(-d1, -d2) > dot_guard
    elif d3 >= 0.0 and d4 <= d3:
        bary, branch = [0.0, 1.0, 0.0], "vertex1"
        q, dq, dbary = b, db, [0.0, 0.0, 0.0]
        smooth = min(d3, d3 - d4) > dot_guard
    elif vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        den = d1 - d3
        if den <= 1.0e-12:
            return None
        t = d1 / den
        e, de = ab, sub(db, da)
        r, dr = ap, sub(dp, da)
        dnum = dot(dr, e) + dot(r, de)
        dden = 2.0 * dot(e, de)
        dt = (dnum - t * dden) / den
        q = add(a, scale(t, e))
        dq = add(add(da, scale(dt, e)), scale(t, de))
        bary, dbary, branch = [1.0 - t, t, 0.0], [-dt, dt, 0.0], "edge01"
        smooth = (-vc > det_guard and d1 > dot_guard and -d3 > dot_guard)
    elif d6 >= 0.0 and d5 <= d6:
        bary, branch = [0.0, 0.0, 1.0], "vertex2"
        q, dq, dbary = c, dc, [0.0, 0.0, 0.0]
        smooth = min(d6, d6 - d5) > dot_guard
    elif vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        den = d2 - d6
        if den <= 1.0e-12:
            return None
        t = d2 / den
        e, de = ac, sub(dc, da)
        r, dr = ap, sub(dp, da)
        dnum = dot(dr, e) + dot(r, de)
        dden = 2.0 * dot(e, de)
        dt = (dnum - t * dden) / den
        q = add(a, scale(t, e))
        dq = add(add(da, scale(dt, e)), scale(t, de))
        bary, dbary, branch = [1.0 - t, 0.0, t], [-dt, 0.0, dt], "edge02"
        smooth = (-vb > det_guard and d2 > dot_guard and -d6 > dot_guard)
    elif va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        e, de = sub(c, b), sub(dc, db)
        r, dr = sub(point, b), sub(dp, db)
        den = dot(e, e)
        if den <= 1.0e-12:
            return None
        t = dot(r, e) / den
        dt = (dot(dr, e) + dot(r, de) -
              2.0 * t * dot(e, de)) / den
        q = add(b, scale(t, e))
        dq = add(add(db, scale(dt, e)), scale(t, de))
        bary, dbary, branch = [0.0, 1.0 - t, t], [0.0, -dt, dt], "edge12"
        smooth = (-va > det_guard and d4 - d3 > dot_guard and
                  d5 - d6 > dot_guard)
    else:
        denominator = va + vb + vc
        gram11, gram12, gram22 = dot(ab, ab), dot(ab, ac), dot(ac, ac)
        gramdet = gram11 * gram22 - gram12 * gram12
        if denominator <= 1.0e-30 or gramdet <= det_guard:
            return None
        v, w = vb / denominator, vc / denominator
        q = add(add(a, scale(v, ab)), scale(w, ac))
        bary, branch = [1.0 - v - w, v, w], "face"
        delta = sub(point, q)
        deb, dec = sub(db, da), sub(dc, da)
        fixed = sub(sub(dp, da), add(scale(v, deb), scale(w, dec)))
        rhs0 = dot(ab, fixed) + dot(deb, delta)
        rhs1 = dot(ac, fixed) + dot(dec, delta)
        dv = (rhs0 * gram22 - rhs1 * gram12) / gramdet
        dw = (rhs1 * gram11 - rhs0 * gram12) / gramdet
        dq = add(add(add(da, scale(dv, ab)), scale(v, deb)),
                 add(scale(dw, ac), scale(w, dec)))
        dbary = [-dv - dw, dv, dw]
        smooth = min(va, vb, vc) > det_guard and min(bary) > 64.0 * (2.0 ** -23)

    return {"closest": q, "bary": bary, "branch": branch,
            "closest_direction": dq, "bary_direction": dbary,
            "smooth": smooth}


def _segments(p0, p1, q0, q1):
    dp, dq, r = sub(p1, p0), sub(q1, q0), sub(p0, q0)
    a, e, f = dot(dp, dp), dot(dq, dq), dot(dq, r)
    if a <= K_GEOMETRY_FLOOR and e <= K_GEOMETRY_FLOOR:
        s = t = 0.0
        branch = "both_degenerate"
    elif a <= K_GEOMETRY_FLOOR:
        s = 0.0
        t = min(1.0, max(0.0, f / max(e, K_GEOMETRY_FLOOR)))
        branch = "first_degenerate"
    else:
        c = dot(dp, r)
        if e <= K_GEOMETRY_FLOOR:
            t = 0.0
            s = min(1.0, max(0.0, -c / max(a, K_GEOMETRY_FLOOR)))
            branch = "second_degenerate"
        else:
            b = dot(dp, dq)
            den = a * e - b * b
            s = min(1.0, max(0.0, (b * f - c * e) / den)) \
                if den > K_GEOMETRY_FLOOR else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = min(1.0, max(0.0, -c / a))
                branch = "second_start"
            elif t > 1.0:
                t = 1.0
                s = min(1.0, max(0.0, (b - c) / a))
                branch = "second_end"
            else:
                branch = "interior" if 0.0 < s < 1.0 else "first_endpoint"
    return add(p0, scale(s, dp)), add(q0, scale(t, dq)), s, t, branch


def _mollifier(first, second):
    f2, s2 = dot(first, first), dot(second, second)
    product = f2 * s2
    if not product > K_GEOMETRY_FLOOR:
        return 0.0, [0.0] * 3, [0.0] * 3, "degenerate"
    relative = 1.0e-3
    threshold = max(relative * product, K_GEOMETRY_FLOOR)
    inner = dot(first, second)
    measure = max(product - inner * inner, 0.0)
    ratio = min(1.0, max(0.0, measure / threshold))
    value = ratio * (2.0 - ratio)
    if not 0.0 < ratio < 1.0:
        return value, [0.0] * 3, [0.0] * 3, "zero" if ratio == 0.0 else "one"
    mf = scale(2.0, sub(scale(s2, first), scale(inner, second)))
    ms = scale(2.0, sub(scale(f2, second), scale(inner, first)))
    if relative * product >= K_GEOMETRY_FLOOR:
        sf, ss = scale(2.0 * s2, first), scale(2.0 * f2, second)
        inv = 1.0 / max(threshold * threshold, K_GEOMETRY_FLOOR ** 2)
        rf = scale(relative * inv, sub(scale(product, mf), scale(measure, sf)))
        rs = scale(relative * inv, sub(scale(product, ms), scale(measure, ss)))
    else:
        rf, rs = scale(1.0 / threshold, mf), scale(1.0 / threshold, ms)
    derivative = 2.0 * (1.0 - ratio)
    return value, scale(derivative, rf), scale(derivative, rs), "interior"


def _barrier(separation, thickness, stiffness):
    if not (thickness > 0.0 and stiffness > 0.0) or separation >= thickness:
        return 0.0, 0.0, "inactive"
    floor = 1.0e-4 * thickness
    distance = max(separation, floor)
    d2, h2 = distance * distance, thickness * thickness
    offset = d2 - h2
    logterm = math.log(d2 / h2)
    value = max(-stiffness * offset * offset * logterm, 0.0)
    first = -stiffness * (2.0 * offset * logterm + offset * offset / d2)
    return value, 2.0 * distance * first, "floor" if separation <= floor else "active"


def _barrier_exact_terms(separation, thickness, stiffness):
    if not (thickness > 0.0 and stiffness > 0.0) or separation >= thickness:
        return 0.0, 0.0, 0.0, "inactive"
    floor = 1.0e-4 * thickness
    distance = max(separation, floor)
    d2, h2 = distance * distance, thickness * thickness
    offset = d2 - h2
    logterm = math.log(d2 / h2)
    first_squared = -stiffness * (
        2.0 * offset * logterm + offset * offset / d2)
    second_squared = -stiffness * (
        2.0 * logterm + 4.0 * offset / d2 - offset * offset / (d2 * d2))
    energy = max(-stiffness * offset * offset * logterm, 0.0)
    first = 2.0 * distance * first_squared
    second = 2.0 * first_squared + 4.0 * d2 * second_squared
    return energy, first, second, "floor" if separation <= floor else "active"


def _stiffness_from_state(contact, normal, separation, inverse_response,
                          relative_velocity, dt):
    thickness = contact["barrier"][2]
    if not (inverse_response > 0.0 and thickness > 0.0 and dt > 0.0):
        return None, "invalid_response"
    distance = max(separation, 1.0e-4 * thickness)
    remaining = max(thickness - distance, 1.0e-3 * thickness)
    denominator = inverse_response * dt * dt * thickness * thickness
    if denominator <= 1.0e-20:
        return None, "inertial_denominator_floor"
    inertial = 10.0 / denominator
    closing = max(-dot(relative_velocity, normal), 0.0)
    impact = closing / max(inverse_response * dt * remaining * thickness * thickness,
                           1.0e-20)
    if not (math.isfinite(inertial) and math.isfinite(impact)) or \
            inertial <= impact * (1.0 + 128.0 * (2.0 ** -23)):
        return max(inertial, impact), "impact_or_stiffness_switch"
    return inertial, "inertial"


def _exact_vt_operator_action(contact, positions, velocities, direction,
                              inverse_masses, dt,
                              allow_fixed_feature=False):
    """Differentiate one supported frictionless, inertial VT contact row.

    The production policy accepts only a smooth feature interior. The optional
    diagnostic mode drops just that feature-margin guard and differentiates
    the deterministic center-selected feature. It still retains the
    nondegenerate triangle/mollifier, barrier, stiffness, and constitutive
    guards; callers must compare only sides that keep the center feature.
    """
    identity = contact["identity"]
    if identity[2] != 0 or identity[3] & CONTACT_MAPPED_FEATURE:
        return None, "not_unmapped_vertex_triangle"
    if any(abs(x) > 1.0e-12 for x in contact["impulse_friction"][1:4]):
        return None, "friction_branch"
    if abs(contact["mollifier"][0] - 1.0) > 1.0e-7 or any(
            abs(x) > 1.0e-12 for row in contact["mollifier_gradients"] for x in row):
        return None, "nonunit_mollifier"
    if any(not math.isfinite(x) or x < 0.0 for x in inverse_masses[:4]):
        return None, "invalid_inverse_mass"
    if len(positions) < 4 or len(direction) < 4:
        return None, "incomplete_support"

    time_scales = contact["time_scales"]
    dx = [scale(dt * time_scales[i], direction[i]) for i in range(4)]
    feature = _triangle_closest_directional(
        positions[0], positions[1], positions[2], positions[3],
        dx[0], dx[1], dx[2], dx[3])
    if feature is None:
        return None, "feature_boundary_or_degenerate"
    if not feature["smooth"] and not allow_fixed_feature:
        return None, "feature_boundary_or_degenerate"
    bary = feature["bary"]
    dbary = feature["bary_direction"]
    weights = [1.0, -bary[0], -bary[1], -bary[2]]
    dweights = [0.0, -dbary[0], -dbary[1], -dbary[2]]
    if max(abs(weights[i] - contact["weights"][i]) for i in range(4)) > 2.0e-5:
        return None, "stored_weights_do_not_match_candidate_geometry"

    delta = sub(positions[0], feature["closest"])
    separation = norm(delta)
    thickness = contact["barrier"][2]
    if not (separation > 1.0e-4 * thickness and separation < thickness):
        return None, "barrier_floor_or_inactive"
    captured_separation = contact["point_and_separation"][3]
    if abs(separation - captured_separation) > max(1.0e-3 * thickness, 1.0e-9):
        return None, "captured_separation_mismatch"
    normal = scale(1.0 / separation, delta)
    if norm(sub(normal, contact["normal_toi"][:3])) > 2.0e-4:
        return None, "captured_normal_mismatch"

    inverse_response = sum(weights[i] * weights[i] * inverse_masses[i]
                           for i in range(4))
    if not (inverse_response > 0.0 and dt > 0.0):
        return None, "invalid_inverse_response"
    if abs(inverse_response - contact["relative_velocity_response"][3]) > \
            2.0e-5 * max(inverse_response, 1.0e-20):
        return None, "captured_inverse_response_mismatch"
    relative_velocity = [sum(weights[i] * velocities[i][axis]
                             for i in range(4)) for axis in range(3)]
    stiffness, stiffness_branch = _stiffness_from_state(
        contact, normal, separation, inverse_response, relative_velocity, dt)
    if stiffness_branch != "inertial" or stiffness is None:
        return None, stiffness_branch
    if abs(stiffness - contact["mollifier"][3]) > \
            2.0e-5 * max(stiffness, 1.0):
        return None, "captured_stiffness_mismatch"

    _, energy_prime, energy_second, barrier_branch = _barrier_exact_terms(
        separation, thickness, stiffness)
    if barrier_branch != "active" or not energy_prime < 0.0:
        return None, "barrier_branch_or_force_switch"
    impulse_magnitude = -dt * energy_prime
    if abs(impulse_magnitude - contact["impulse_friction"][0]) > \
            2.0e-4 * max(abs(impulse_magnitude), 1.0e-20):
        return None, "captured_impulse_mismatch"
    if abs(contact["barrier"][1] - max(dt * dt * energy_second, 0.0)) > \
            3.0e-4 * max(abs(contact["barrier"][1]), 1.0e-20):
        return None, "captured_projected_curvature_mismatch"

    ddelta = sub(dx[0], feature["closest_direction"])
    ddistance_position = dot(normal, ddelta)
    dnormal = scale(1.0 / separation,
                    sub(ddelta, scale(ddistance_position, normal)))
    dresponse = sum(2.0 * weights[i] * dweights[i] * inverse_masses[i]
                    for i in range(4))
    ddistance_velocity = ddistance_position / dt
    barrier_curvature = dt * dt * energy_second
    out = [[0.0] * 3 for _ in range(6)]
    for local in range(4):
        scalar_normal = (
            weights[local] * (
                barrier_curvature * ddistance_velocity +
                impulse_magnitude * dresponse / inverse_response) -
            impulse_magnitude * dweights[local])
        value = sub(scale(scalar_normal, normal),
                    scale(weights[local] * impulse_magnitude, dnormal))
        out[local] = scale(time_scales[local], value)
    return out, None


def _feature_state(contact, support_positions):
    kind = contact["identity"][2]
    mapped = bool(contact["identity"][3] & CONTACT_MAPPED_FEATURE)
    if mapped:
        maps, features = [], []
        for feature in range(4):
            side = contact["feature_sides"][feature]
            if side not in (0, 1):
                raise ValueError("invalid_mapped_feature_side")
            bary = contact["feature_barycentric"][feature]
            mapping = [0.0] * 6
            for corner in range(3):
                mapping[3 * side + corner] = bary[corner]
            maps.append(mapping)
            features.append([sum(mapping[i] * support_positions[i][axis]
                                 for i in range(6)) for axis in range(3)])
    else:
        used_features = 2 if kind == 2 else 4
        maps = [[float(i == j) for i in range(6)] if j < used_features
                else [0.0] * 6 for j in range(4)]
        features = support_positions[:4]
    gs = [[0.0] * 3 for _ in range(4)]
    gm = [[0.0] * 3 for _ in range(4)]
    if kind == 2:  # point-point
        delta = sub(features[0], features[1])
        separation = norm(delta)
        if separation == 0.0:
            raise ValueError("zero_separation")
        normal = scale(1.0 / separation, delta)
        gs[0], gs[1] = normal, scale(-1.0, normal)
        feature_branch, mollifier, mollifier_branch = "point_point", 1.0, "unit"
    elif kind == 0:  # vertex-triangle
        closest, bary, feature_branch = _triangle_closest(
            features[0], features[1], features[2], features[3])
        delta = sub(features[0], closest)
        separation = norm(delta)
        if separation == 0.0:
            raise ValueError("zero_separation")
        normal = scale(1.0 / separation, delta)
        gs[0] = normal
        for i in range(3):
            gs[i + 1] = scale(-bary[i], normal)
        mollifier, ma, mb, mollifier_branch = _mollifier(
            sub(features[2], features[1]), sub(features[3], features[1]))
        gm[1] = scale(-1.0, add(ma, mb))
        gm[2], gm[3] = ma, mb
    elif kind == 1:  # edge-edge
        p, q, s, t, feature_branch = _segments(*features)
        delta = sub(p, q)
        separation = norm(delta)
        if separation == 0.0:
            raise ValueError("zero_separation")
        normal = scale(1.0 / separation, delta)
        gs[0], gs[1] = scale(1.0 - s, normal), scale(s, normal)
        gs[2], gs[3] = scale(-(1.0 - t), normal), scale(-t, normal)
        mollifier, ma, mb, mollifier_branch = _mollifier(
            sub(features[1], features[0]), sub(features[3], features[2]))
        gm[0], gm[1] = scale(-1.0, ma), ma
        gm[2], gm[3] = scale(-1.0, mb), mb
    else:
        raise ValueError("unsupported_contact_kind")
    local_gs, local_gm = [[0.0] * 3 for _ in range(6)], [[0.0] * 3 for _ in range(6)]
    for feature in range(4):
        for local in range(6):
            for axis in range(3):
                local_gs[local][axis] += maps[feature][local] * gs[feature][axis]
                local_gm[local][axis] += maps[feature][local] * gm[feature][axis]
    return {"separation": separation, "normal": normal, "mollifier": mollifier,
            "feature_branch": feature_branch, "mollifier_branch": mollifier_branch,
            "grad_sep_local": local_gs, "grad_m_local": local_gm,
            "barycentric": bary if kind == 0 else None}


def _residual_impulse(contact, positions, dt, inverse_masses=None,
                      velocities=None):
    state = _feature_state(contact, positions)
    stiffness = contact["barrier"][3]
    if contact["identity"][2] == 0 and inverse_masses is not None:
        bary = state["barycentric"]
        weights = [1.0, -bary[0], -bary[1], -bary[2]]
        inverse_response = sum(weights[i] * weights[i] * inverse_masses[i]
                               for i in range(4))
        relative_velocity = ([sum(weights[i] * velocities[i][axis]
                                  for i in range(4)) for axis in range(3)]
                             if velocities is not None else
                             contact["relative_velocity_response"][:3])
        stiffness, _ = _stiffness_from_state(
            contact, state["normal"], state["separation"],
            inverse_response, relative_velocity, dt)
        if stiffness is None:
            stiffness = 0.0
    barrier, barrier_prime, barrier_branch = _barrier(
        state["separation"], contact["barrier"][2], stiffness)
    residual = []
    for local in range(6):
        gradient = add(
            scale(state["mollifier"] * barrier_prime,
                  state["grad_sep_local"][local]),
            scale(barrier, state["grad_m_local"][local]))
        residual.append(scale(-dt * contact["time_scales"][local], gradient))
    result_state = {k: state[k] for k in
        ("separation", "normal", "feature_branch", "mollifier_branch")} | {
            "barrier_branch": barrier_branch, "stiffness": stiffness}
    if state["barycentric"] is not None:
        result_state["barycentric"] = state["barycentric"]
    return residual, result_state


def _stiffness_branch(contact, state, velocities, inverse_masses, dt):
    if contact["identity"][2] == 0 and state.get("barycentric") is not None:
        weights = [1.0, *[-x for x in state["barycentric"]]]
        inverse_response = sum(weights[i] * weights[i] * inverse_masses[i]
                               for i in range(4))
        relative_velocity = [sum(weights[i] * velocities[i][axis]
                                 for i in range(4)) for axis in range(3)]
    else:
        inverse_response = contact["relative_velocity_response"][3]
        relative_velocity = contact["relative_velocity_response"][:3]
    _, branch = _stiffness_from_state(
        contact, state["normal"], state["separation"], inverse_response,
        relative_velocity, dt)
    return branch


def _frozen_action(contact, direction):
    weights, times = contact["weights"], contact["time_scales"]
    relative = [sum(weights[i] * times[i] * direction[i][axis] for i in range(6))
                for axis in range(3)]
    h = [row[:3] for row in contact["hessian_rows"]]
    hrel = [sum(h[r][c] * relative[c] for c in range(3)) for r in range(3)]
    grads = contact["mollifier_gradients"]
    mdirection = sum(dot(grads[i], scale(times[i], direction[i])) for i in range(6))
    return [scale(times[i], add(
        scale(weights[i], hrel),
        scale(contact["mollifier"][2] * mdirection, grads[i])))
        for i in range(6)]


def _norm_nested(rows):
    return math.sqrt(sum(x * x for row in rows for x in row))


def _relative_error(actual, expected):
    denominator = _norm_nested(expected)
    return (_norm_nested([[x - y for x, y in zip(a, b)]
                          for a, b in zip(actual, expected)]) / denominator
            if denominator else None)


def _compare_production_exact_action(capture, record, exact_action,
                                    exact_reason):
    """Compare captured Metal production action with independent CPU formula.

    The GPU helper emits one vector for each of the four VT support nodes.
    Results are comparable only when the exact CPU branch and all four Metal
    support evaluations are supported. A missing/partial marker is never
    interpreted as a zero action.
    """
    if not capture.get("production_exact_action_capture", False):
        return {"status": "not_captured"}
    markers = record["_production_exact_action_supported"]
    actions = record["_production_exact_action"]
    result = {
        "supported_support_nodes": sum(markers),
        "support_node_count": 4,
        "source": capture["production_exact_action_source"],
        "direction": capture["production_exact_action_direction"],
        "support_order": capture["production_exact_action_support_order"],
        "quantity": capture["production_exact_action_quantity"],
        "timestep_s": record["_production_exact_action_dt_s"],
    }
    if exact_action is None:
        result.update(status="cpu_reference_unsupported",
                      reason=exact_reason)
        return result
    if not all(markers):
        result.update(status="metal_branch_unsupported",
                      reason="one_or_more_support_nodes_not_supported")
        return result
    expected = exact_action[:4]
    result.update(
        status="compared",
        relative_error=_relative_error(actions, expected),
        cosine=_cosine(actions, expected),
        metal_action_norm=_norm_nested(actions),
        cpu_action_norm=_norm_nested(expected),
    )
    return result


def _saturated_unit_mollifier_is_interior(contact, positions):
    if contact["identity"][2] != 0:
        return contact["mollifier"][0] == 1.0
    first = sub(positions[2], positions[1])
    second = sub(positions[3], positions[1])
    product = dot(first, first) * dot(second, second)
    if product <= K_GEOMETRY_FLOOR:
        return False
    measure = max(product - dot(first, second) ** 2, 0.0)
    threshold = max(1.0e-3 * product, K_GEOMETRY_FLOOR)
    return measure > threshold * (1.0 + 128.0 * (2.0 ** -23))


def _cosine(a, b):
    av, bv = [x for row in a for x in row], [x for row in b for x in row]
    denominator = norm(av) * norm(bv)
    return dot(av, bv) / denominator if denominator else None


def _triangle_geometry_condition(positions):
    """Report the FP32 degeneracy guard and scale-free area of a VT triangle."""
    ab, ac, bc = sub(positions[2], positions[1]), \
        sub(positions[3], positions[1]), sub(positions[3], positions[2])
    edge2 = max(dot(ab, ab), dot(ac, ac), dot(bc, bc), 1.0e-30)
    gram_det = max(dot(ab, ab) * dot(ac, ac) - dot(ab, ac) ** 2, 0.0)
    guard = 64.0 * (2.0 ** -23) * edge2 * edge2
    return {
        "classification": "diagnostic_condition_only",
        "max_edge_length_squared_m2": edge2,
        "gram_determinant_m4": gram_det,
        "float32_degeneracy_guard_m4": guard,
        "gram_determinant_over_guard": gram_det / guard if guard > 0.0 else None,
        "above_replay_float32_degeneracy_guard": gram_det > guard,
        "twice_area_over_max_edge_squared": math.sqrt(gram_det) / edge2,
    }


def _one_sided_description(record, positions, direction, center_residual,
                            side_residual, side_state, center_state,
                            side_stiffness, center_stiffness, dt, epsilon,
                            sign, captured_fallback_action=None,
                            fixed_feature_action=None):
    """Describe a one-sided residual slope without treating it as qualification.

    ``sign`` is +1 for the positive perturbation and -1 for the negative one.
    The side is comparable to the captured production action only on nodes for
    which the native kernel explicitly marked its action supported.
    """
    action = [[-(side_residual[i][axis] - center_residual[i][axis]) /
               (sign * epsilon) for axis in range(3)] for i in range(6)]
    branch_keys = ("feature_branch", "mollifier_branch", "barrier_branch")
    same_branch = all(side_state.get(key) == center_state.get(key)
                      for key in branch_keys)
    same_stiffness = side_stiffness == center_stiffness
    moving = [i for i in range(4) if norm(direction[i]) > 0.0]
    realized = [i for i in moving if any(
        _f32(positions[i][axis] + sign * dt * epsilon * direction[i][axis]) !=
        _f32(positions[i][axis]) for axis in range(3))]
    result = {
        "status": "same_discrete_branch" if same_branch and same_stiffness
                  else "branch_or_stiffness_changed",
        "classification": "descriptive_one_sided_replay_not_qualification",
        "epsilon_velocity": epsilon,
        "direction_sign": sign,
        "feature_branch": side_state.get("feature_branch"),
        "mollifier_branch": side_state.get("mollifier_branch"),
        "barrier_branch": side_state.get("barrier_branch"),
        "stiffness_branch": side_stiffness,
        "same_feature_mollifier_barrier_branch_as_center": same_branch,
        "same_stiffness_branch_as_center": same_stiffness,
        "barycentric": side_state.get("barycentric"),
        "minimum_positive_barycentric": (
            min((value for value in side_state.get("barycentric", [])
                 if value > 0.0), default=None)),
        "fp32_realized_moving_support_nodes": len(realized),
        "fp32_moving_support_nodes": len(moving),
        "fp32_realized_position_change_node_ids": realized,
        "fp32_max_realized_position_change_m": max((
            norm(sub(
                [_f32(positions[i][axis] + sign * dt * epsilon * direction[i][axis])
                 for axis in range(3)],
                [_f32(positions[i][axis]) for axis in range(3)]))
            for i in moving), default=0.0),
        "fp64_replayed_action_norm": _norm_nested(action),
    }
    comparable_center_branch = same_branch and same_stiffness
    if captured_fallback_action is not None:
        result["captured_fallback_action_comparison"] = {
            "status": "same_discrete_branch" if comparable_center_branch
                      else "branch_or_stiffness_changed",
            "relative_error": (_relative_error(captured_fallback_action, action)
                               if comparable_center_branch else None),
            "per_support_node_relative_error": (
                [_relative_error([captured_fallback_action[i]], [action[i]])
                 for i in range(len(action))]
                if comparable_center_branch else [None] * len(action)),
            "fallback_action_norm": _norm_nested(captured_fallback_action),
            "one_sided_action_norm": _norm_nested(action),
        }
    if fixed_feature_action is not None:
        result["fixed_center_feature_action_comparison"] = {
            "status": "same_discrete_branch" if comparable_center_branch
                      else "branch_or_stiffness_changed",
            "relative_error": (_relative_error(fixed_feature_action, action)
                               if comparable_center_branch else None),
            "per_support_node_relative_error": (
                [_relative_error([fixed_feature_action[i]], [action[i]])
                 for i in range(len(action))]
                if comparable_center_branch else [None] * len(action)),
            "action_norm": _norm_nested(fixed_feature_action),
            "one_sided_action_norm": _norm_nested(action),
        }
    markers = record.get("_production_exact_action_supported", [])
    metal = record.get("_production_exact_action", [])
    supported = [i for i, value in enumerate(markers[:4]) if value == 1]
    if not supported:
        result["production_action_comparison"] = {
            "status": "native_support_unavailable_for_this_branch",
            "supported_support_nodes": 0,
        }
    else:
        native_subset = [metal[i] for i in supported]
        replay_subset = [action[i] for i in supported]
        result["production_action_comparison"] = {
            "status": "partial_support" if len(supported) < 4 else "compared",
            "supported_support_nodes": len(supported),
            "support_node_indices": supported,
            "relative_error": _relative_error(native_subset, replay_subset),
            "cosine": _cosine(native_subset, replay_subset),
            "native_action_norm": _norm_nested(native_subset),
            "fp64_replayed_action_norm": _norm_nested(replay_subset),
        }
    return result


def _analyze_record(capture, record, epsilons):
    env, contact = record["environment"], record["_decoded"]
    count, grid = capture["fem_node_count"], capture["grid_node_count"]
    positions, velocities, inverse_masses, direction = [], [], [], []
    required_locals = (6 if contact["identity"][3] & CONTACT_MAPPED_FEATURE
                       else 2 if contact["identity"][2] == 2 else 4)
    for local_index, unified in enumerate(contact["nodes"]):
        if unified == INVALID_NODE or unified < grid or unified >= grid + count:
            if local_index < required_locals:
                positions.append([math.nan] * 3)
                velocities.append([math.nan] * 3)
                inverse_masses.append(math.nan)
                direction.append([math.nan] * 3)
            else:
                positions.append([0.0] * 3)
                velocities.append([0.0] * 3)
                inverse_masses.append(0.0)
                direction.append([0.0] * 3)
            continue
        local = unified - grid
        node = _node_state(capture["_candidate_nodes"], env, local, count)
        positions.append(node["position"])
        velocities.append(node["velocity"])
        inverse_masses.append(node["inverse_mass"])
        direction.append(_direction(capture["_solution"], env, local, count))
    reason = None
    if grid != 0:
        reason = "grid_nodes_not_captured"
    elif not (contact["identity"][3] & CONTACT_VALID):
        reason = "invalid_contact_record"
    elif contact["impulse_friction"][3] > 1.0e-12 or any(
            abs(x) > 1.0e-12 for x in contact["impulse_friction"][1:3]):
        reason = "friction_or_lagged_tangent_not_replayed"
    elif any(not math.isfinite(x) or x < 0.0 for x in contact["time_scales"]):
        reason = "invalid_multirate_time_scale"
    elif any(not math.isfinite(x) for row in positions[:required_locals] for x in row):
        reason = "contact_node_outside_captured_fem_arena"
    result = {"environment": env, "active_slot": record["active_slot"],
              "contact_slot": record["contact_slot"], "identity": contact["identity"],
              "source": contact["source"], "node_ids_unified": contact["nodes"],
              "status": "unsupported" if reason else "analyzed",
              "unsupported_reason": reason}
    if reason:
        result["production_exact_vt_action"] = (
            _compare_production_exact_action(capture, record, None, reason))
        return result
    dt = record.get("_production_exact_action_dt_s", capture["dt_s"])
    result["timestep_s"] = dt
    try:
        dynamic_inertial = contact["identity"][2] == 0
        center, center_state = _residual_impulse(
            contact, positions, dt,
            inverse_masses if dynamic_inertial else None, velocities)
    except ValueError as exc:
        result["status"], result["unsupported_reason"] = "unsupported", str(exc)
        result["production_exact_vt_action"] = (
            _compare_production_exact_action(capture, record, None, str(exc)))
        return result
    frozen = _frozen_action(contact, direction)
    center_stiffness = _stiffness_branch(
        contact, center_state, velocities, inverse_masses, dt)
    normal, impulse = contact["normal_toi"][:3], contact["impulse_friction"]
    expected_center = []
    for local in range(6):
        expected_center.append(scale(contact["time_scales"][local], sub(
            scale(contact["weights"][local] * impulse[0], normal),
            scale(contact["mollifier"][1],
                  contact["mollifier_gradients"][local]))))
    result["center_feature_state"] = center_state
    if contact["identity"][2] == 0:
        result["triangle_geometry_condition"] = _triangle_geometry_condition(positions)
    result["center_inverse_mass_response"] = (
        sum(contact["weights"][i] ** 2 * inverse_masses[i]
            for i in range(4)) if dynamic_inertial else
        contact["relative_velocity_response"][3])
    result["captured_stiffness"] = contact["barrier"][3]
    result["recomputed_center_stiffness"] = center_state["stiffness"]
    result["center_residual_reconstruction_relative_error"] = _relative_error(
        center, expected_center)
    result["frozen_action_norm"] = _norm_nested(frozen)
    strict_action, strict_reason = _exact_vt_operator_action(
        contact, positions, velocities, direction, inverse_masses, dt)
    feature_policy = capture.get("_production_exact_action_feature_policy",
                                 "strict_smooth_feature")
    if feature_policy == "fixed_candidate_feature":
        exact_action, exact_reason = _exact_vt_operator_action(
            contact, positions, velocities, direction, inverse_masses, dt,
            allow_fixed_feature=True)
    else:
        exact_action, exact_reason = strict_action, strict_reason
    result["exact_vt_operator_status"] = (
        "supported" if exact_action is not None else "unsupported")
    result["exact_vt_operator_reason"] = exact_reason
    result["strict_smooth_feature_operator_status"] = (
        "supported" if strict_action is not None else "unsupported")
    result["strict_smooth_feature_operator_reason"] = strict_reason
    result["production_exact_action_feature_policy"] = feature_policy
    result["exact_vt_operator_action_norm"] = (
        _norm_nested(exact_action) if exact_action is not None else None)
    result["production_exact_vt_action"] = _compare_production_exact_action(
        capture, record, exact_action, exact_reason)
    fixed_feature_action = None
    fixed_feature_reason = None
    if strict_action is None and strict_reason == "feature_boundary_or_degenerate":
        # Diagnostic-only generalized Jacobian candidate. The feature is
        # selected once from center geometry by the same deterministic
        # closest-feature ordering as production; the action formula is then
        # linear in every input direction. We still require unchanged
        # one-sided feature/stiffness labels below before comparing to FD.
        fixed_feature_action, fixed_feature_reason = _exact_vt_operator_action(
            contact, positions, velocities, direction, inverse_masses, dt,
            allow_fixed_feature=True)
    result["fixed_center_feature_diagnostic"] = {
        "classification": (
            "captured_production_fixed_candidate_feature_policy"
            if feature_policy == "fixed_candidate_feature" else
            "diagnostic_generalized_jacobian_candidate_not_production"),
        "status": "available" if fixed_feature_action is not None else "unavailable",
        "reason": fixed_feature_reason if fixed_feature_action is None else None,
        "center_feature_branch": center_state["feature_branch"],
        "action_norm": (_norm_nested(fixed_feature_action)
                         if fixed_feature_action is not None else None),
    }
    result["sweep"] = []
    for epsilon in epsilons:
        plus, minus, plus32, minus32 = [], [], [], []
        for i in range(6):
            ts = contact["time_scales"][i]
            p = [positions[i][k] + dt * ts * epsilon * direction[i][k]
                 for k in range(3)]
            m = [positions[i][k] - dt * ts * epsilon * direction[i][k]
                 for k in range(3)]
            plus.append(p)
            minus.append(m)
            plus32.append([_f32(x) for x in p])
            minus32.append([_f32(x) for x in m])
        plus_velocity = [add(velocities[i], scale(epsilon, direction[i]))
                         for i in range(6)]
        minus_velocity = [sub(velocities[i], scale(epsilon, direction[i]))
                          for i in range(6)]
        try:
            rp, bp = _residual_impulse(
                contact, plus, dt,
                inverse_masses if dynamic_inertial else None, plus_velocity)
            rm, bm = _residual_impulse(
                contact, minus, dt,
                inverse_masses if dynamic_inertial else None, minus_velocity)
            rp32, bp32 = _residual_impulse(
                contact, plus32, dt,
                inverse_masses if dynamic_inertial else None, plus_velocity)
            rm32, bm32 = _residual_impulse(
                contact, minus32, dt,
                inverse_masses if dynamic_inertial else None, minus_velocity)
        except ValueError:
            result["sweep"].append({"epsilon_velocity": epsilon,
                                    "status": "invalid_geometry"})
            continue
        action = [[-(x - y) / (2.0 * epsilon) for x, y in zip(a, b)]
                  for a, b in zip(rp, rm)]
        action32 = [[-(x - y) / (2.0 * epsilon) for x, y in zip(a, b)]
                    for a, b in zip(rp32, rm32)]
        stiffness_plus = _stiffness_branch(
            contact, bp, plus_velocity, inverse_masses, dt)
        stiffness_minus = _stiffness_branch(
            contact, bm, minus_velocity, inverse_masses, dt)
        center_stiffness = _stiffness_branch(
            contact, center_state, velocities, inverse_masses, dt)
        branches_match = all(
            bp[k] == bm[k] == center_state[k] for k in
            ("feature_branch", "mollifier_branch", "barrier_branch"))
        if contact["identity"][2] == 0:
            feature_direction = _triangle_closest_directional(
                positions[0], positions[1], positions[2], positions[3],
                *[scale(dt * contact["time_scales"][i], direction[i])
                  for i in range(4)])
            interior_feature = feature_direction is not None and feature_direction["smooth"]
        else:
            interior_feature = (
                center_state["feature_branch"] == "point_point"
                if contact["identity"][2] == 2 else
                center_state["feature_branch"] == "interior")
        interior_mollifier = (
            center_state["mollifier_branch"] == "interior" or
            _saturated_unit_mollifier_is_interior(contact, positions))
        stiffness_match = (
            stiffness_plus == stiffness_minus == center_stiffness and
            center_stiffness in ("inertial", "impact"))
        smooth = branches_match and interior_feature and interior_mollifier and stiffness_match
        plus_one_sided = _one_sided_description(
            record, positions, direction, center, rp, bp, center_state,
            stiffness_plus, center_stiffness, dt, epsilon, +1.0,
            captured_fallback_action=frozen,
            fixed_feature_action=fixed_feature_action)
        minus_one_sided = _one_sided_description(
            record, positions, direction, center, rm, bm, center_state,
            stiffness_minus, center_stiffness, dt, epsilon, -1.0,
            captured_fallback_action=frozen,
            fixed_feature_action=fixed_feature_action)
        moving_locals = [i for i in range(required_locals)
                         if contact["time_scales"][i] > 0.0 and
                         norm(direction[i]) > 0.0]
        realized_locals = [i for i in moving_locals
                           if any(plus32[i][k] != minus32[i][k]
                                  for k in range(3))]
        distinct_positions = (None if not moving_locals else
                              len(realized_locals) == len(moving_locals))
        max_half_delta = max((0.5 * norm(sub(plus32[i], minus32[i]))
                              for i in range(required_locals)), default=0.0)
        result["sweep"].append({
            "epsilon_velocity": epsilon,
            "status": "smooth_branch" if smooth else "branch_changed_or_boundary",
            "captured_action_relative_error_fp64": _relative_error(action, frozen),
            "captured_action_cosine_fp64": _cosine(action, frozen),
            "captured_action_relative_error_fp32_emulated": _relative_error(action32, frozen),
            "centered_action_norm_fp64": _norm_nested(action),
            "dynamic_stiffness_center": center_state["stiffness"],
            "dynamic_stiffness_plus": bp["stiffness"],
            "dynamic_stiffness_minus": bm["stiffness"],
            "dynamic_stiffness_fd_action_relative_error": (
                _relative_error(action, exact_action)
                if exact_action is not None else None),
            "dynamic_stiffness_fd_action_relative_error_fp32_emulated": (
                _relative_error(action32, exact_action)
                if exact_action is not None else None),
            "branch_center": center_state, "branch_plus": bp, "branch_minus": bm,
            "one_sided": {"plus": plus_one_sided, "minus": minus_one_sided},
            "stiffness_branch_center": center_stiffness,
            "stiffness_branch_plus": stiffness_plus,
            "stiffness_branch_minus": stiffness_minus,
            "fp32_position_sides_distinct": distinct_positions,
            "fp32_moving_nodes": len(moving_locals),
            "fp32_realized_moving_nodes": len(realized_locals),
            "fp32_max_realized_position_half_delta_m": max_half_delta,
        })
    return result


def analyze_capture(capture: dict, epsilons=DEFAULT_EPSILONS) -> dict:
    contacts = [_analyze_record(capture, row, epsilons)
                for row in capture["contacts"]]
    return {
        "schema": 1,
        "classification": "diagnostic_only_not_solver_or_physical_qualification",
        "capture": {
            **{k: capture[k] for k in (
                "root", "iteration", "dt_s", "environment_count", "fem_node_count",
                "grid_node_count", "deformable_contact_capacity",
                "trace_slots_per_environment", "capture_bytes", "active_counts",
                "truncated", "recorded_count", "production_exact_action_capture")},
            "production_exact_action_feature_policy": capture.get(
                "_production_exact_action_feature_policy", "strict_smooth_feature"),
        },
        "active_coverage_complete": capture["_complete_active_coverage"],
        "epsilon_schedule_velocity": list(epsilons),
        "contacts": contacts,
        "counts": {
            "analyzed": sum(x["status"] == "analyzed" for x in contacts),
            "unsupported": sum(x["status"] == "unsupported" for x in contacts),
            "smooth_comparisons": sum(
                sample.get("status") == "smooth_branch"
                for x in contacts for sample in x.get("sweep", [])),
            "branch_or_boundary_comparisons": sum(
                sample.get("status") == "branch_changed_or_boundary"
                for x in contacts for sample in x.get("sweep", [])),
            "production_action_compared": sum(
                x.get("production_exact_vt_action", {}).get("status") == "compared"
                for x in contacts),
            "production_action_unsupported": sum(
                x.get("production_exact_vt_action", {}).get("status") in
                    ("cpu_reference_unsupported", "metal_branch_unsupported")
                for x in contacts),
            "fixed_center_feature_diagnostic_available": sum(
                x.get("fixed_center_feature_diagnostic", {}).get("status") == "available"
                for x in contacts),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--root", type=int)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--epsilon", type=float, action="append")
    parser.add_argument("--output", type=Path,
                        help="optional output JSON; created exclusively")
    args = parser.parse_args(argv)
    epsilons = tuple(args.epsilon) if args.epsilon else DEFAULT_EPSILONS
    if not epsilons or any(not math.isfinite(x) or x <= 0.0 for x in epsilons):
        parser.error("epsilon values must be finite and positive")
    try:
        text = args.capture.read_text()
        capture = parse_capture_line(text, args.root, args.iteration)
        report = analyze_capture(capture, epsilons)
        report["input"] = {
            "path": str(args.capture.resolve()),
            "sha256": hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        }
        encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        else:
            sys.stdout.write(encoded)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"capture replay failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
