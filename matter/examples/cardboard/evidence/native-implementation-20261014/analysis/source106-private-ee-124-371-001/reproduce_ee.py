#!/usr/bin/env python3
"""Source-bound CPU reconstruction of source106 private EE candidate 124."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path("/Users/home/cardboard-implementation-20261014")
S106 = ROOT / "snapshots/source-capture-callback-lifetime-106"
S103 = ROOT / "snapshots/source-whole-mortar-proof-103"
C371 = ROOT / "native/capture-lifetime-default-parity-371/center-capture.coupled.json"
C358 = ROOT / "native/first-fatal-private-trial-358/private-trial.coupled.json"
L371 = ROOT / "native/capture-lifetime-default-parity-371/diagnostic-state/dense/fixture-free-body-control-1201.log"
L358 = ROOT / "native/first-fatal-private-trial-358/stderr.log"
OUT = Path(__file__).resolve().parent
PAIR = (24, 127, 124, 1)
EPS = np.float32(1.1920928955078125e-7)
FLOOR = np.float32(1.0e-12)
EDGES = ((0, 1), (1, 2), (2, 0))
EDGE_A = (1, 2)
EDGE_B = (0, 1)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return sha_bytes(path.read_bytes())


def read_capture(path):
    manifest = json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    payload = path.parent / Path(manifest["payload"]["file"]).name
    raw = payload.read_bytes()
    if len(raw) != manifest["payload"]["byte_count"] or sha(payload) != manifest["payload"]["sha256"]:
        raise ValueError("capture payload size/hash mismatch")
    arrays = {}
    for row in manifest["arrays"]:
        key = (row["name"], row.get("phase"))
        if key in arrays:
            raise ValueError("ambiguous duplicate array key " + repr(key))
        start, size = row["offset"], row["byte_count"]
        data = raw[start:start + size]
        if len(data) != size or sha_bytes(data) != row["sha256"]:
            raise ValueError("capture array mismatch " + repr(key))
        arrays[key] = (row, data)
    return manifest, payload, arrays


def get_array(arrays, name, dtype, shape, phase=None):
    row, data = arrays[(name, phase)]
    value = np.frombuffer(data, dtype=dtype).copy()
    if value.size != int(np.prod(shape)):
        raise ValueError("count mismatch " + name)
    return value.reshape(shape), row


def u32_ulp_distance(a, b):
    """Monotone IEEE-754 float ordering distance for finite FP32 values."""
    ua = np.asarray(a, dtype="<f4").view("<u4").astype(np.int64)
    ub = np.asarray(b, dtype="<f4").view("<u4").astype(np.int64)
    sa = np.where((ua & 0x80000000) != 0, 0x80000000 - (ua & 0x7fffffff), 0x80000000 + ua)
    sb = np.where((ub & 0x80000000) != 0, 0x80000000 - (ub & 0x7fffffff), 0x80000000 + ub)
    return np.abs(sa - sb)


def selected_native_failure(path):
    selected = None
    for line in path.read_text().splitlines():
        if not line.startswith("fem_residual_armijo_iteration="):
            continue
        record = json.loads(line.split("=", 1)[1])
        if record.get("newton_iteration") == 7:
            selected = record["environments"][0]
    if selected is None:
        raise ValueError("source106 native log lacks N7 record")
    status = selected["trial_status"]
    if status.get("code") != 6 or status.get("failing_index") != 124 or status.get("diagnostics") != [24, 127, 1, 0]:
        raise ValueError("source106 N7 failure does not identify expected primitive/edge pair")
    return selected


def selected_n7_trial_status(path):
    selected = None
    for line in path.read_text().splitlines():
        if not line.startswith("fem_residual_armijo_iteration="):
            continue
        record = json.loads(line.split("=", 1)[1])
        if record.get("newton_iteration") == 7:
            selected = record["environments"][0]["trial_status"]
    if selected is None:
        raise ValueError("source103 trial358 log lacks N7 record")
    return selected


def f32(x):
    return np.asarray(x, dtype="<f4")


def add(a, b):
    return f32(f32(a) + f32(b))


def sub(a, b):
    return f32(f32(a) - f32(b))


def mul(a, b):
    return f32(f32(a) * f32(b))


def div(a, b):
    return np.float32(np.float32(a) / np.float32(b))


def dot3(a, b):
    p = mul(a, b)
    return np.float32(np.float32(np.float32(p[0] + p[1]) + p[2]))


def norm3(a):
    return np.float32(np.sqrt(dot3(a, a), dtype=np.float32))


def mix3(a, b, t):
    return add(mul(a, np.float32(1) - np.float32(t)), mul(b, t))


def clamp(x, lo, hi):
    return np.float32(min(max(float(x), float(lo)), float(hi)))


def closest_segments32(p0, p1, q0, q1):
    dp, dq, r = sub(p1, p0), sub(q1, q0), sub(p0, q0)
    a, e, f = dot3(dp, dp), dot3(dq, dq), dot3(dq, r)
    if a <= FLOOR and e <= FLOOR:
        s = t = np.float32(0)
    elif a <= FLOOR:
        s, t = np.float32(0), clamp(div(f, max(e, FLOOR)), 0, 1)
    else:
        c = dot3(dp, r)
        if e <= FLOOR:
            t, s = np.float32(0), clamp(div(-c, max(a, FLOOR)), 0, 1)
        else:
            b = dot3(dp, dq)
            den = np.float32(np.float32(a * e) - np.float32(b * b))
            s = clamp(div(np.float32(np.float32(b * f) - np.float32(c * e)), den), 0, 1) if den > FLOOR else np.float32(0)
            t = div(np.float32(np.float32(b * s) + f), e)
            if t < 0:
                t, s = np.float32(0), clamp(div(-c, a), 0, 1)
            elif t > 1:
                t, s = np.float32(1), clamp(div(np.float32(b - c), a), 0, 1)
    return s, t, mix3(p0, p1, s), mix3(q0, q1, t)


def separated(first_start, first_finish, second_start, second_finish, axis, effective):
    axis_length = norm3(axis)
    if not axis_length > FLOOR or not np.isfinite(axis_length):
        return False, None
    unit = f32([div(x, axis_length) for x in axis])
    minimum, scale = np.float32(np.inf), np.float32(effective)
    for a in range(2):
        for b in range(2):
            ds, df = sub(first_start[a], second_start[b]), sub(first_finish[a], second_finish[b])
            if not np.isfinite(ds).all() or not np.isfinite(df).all():
                return False, None
            minimum = np.float32(min(float(minimum), float(dot3(unit, ds)), float(dot3(unit, df))))
            scale = np.float32(max(float(scale), float(np.max(np.abs(ds))), float(np.max(np.abs(df)))))
    margin = np.float32(np.float32(16) * EPS * scale)
    threshold = np.float32(effective + margin)
    return bool(np.isfinite(minimum) and minimum > threshold), {
        "min_projection": float(minimum), "margin": float(margin), "threshold": float(threshold)
    }


def trace32(starts, finishes, thickness, limit):
    """Source-order scalar FP32 transcription; not Metal-native arithmetic."""
    x0, x1 = f32(starts), f32(finishes)
    coordinate_scale = np.float32(np.max(np.abs(np.concatenate((x0, x1)))))
    distance_roundoff = np.float32(np.float32(8) * EPS * max(float(thickness), float(coordinate_scale), 1.0e-12))
    effective = np.float32(thickness + distance_roundoff)
    p00, p01, p10, p11 = x0[0], x1[0], x0[1], x1[1]
    q00, q01, q10, q11 = x0[2], x1[2], x0[3], x1[3]
    speed = np.float32(max(float(norm3(sub(p01, p00))), float(norm3(sub(p11, p10)))) +
                        max(float(norm3(sub(q01, q00))), float(norm3(sub(q11, q10)))))
    first_start, first_finish = f32([p00, p10]), f32([p01, p11])
    second_start, second_finish = f32([q00, q10]), f32([q01, q11])
    time, rows = np.float32(0), []
    certified, hit, terminal = False, False, "iteration_exhausted"
    for iteration in range(limit):
        p0, p1 = mix3(p00, p01, time), mix3(p10, p11, time)
        q0, q1 = mix3(q00, q01, time), mix3(q10, q11, time)
        s, t, pp, pq = closest_segments32(p0, p1, q0, q1)
        delta, distance = sub(pp, pq), norm3(sub(pp, pq))
        row = {"iteration": iteration, "time": float(time), "s": float(s), "t": float(t),
               "pointP": pp.tolist(), "pointQ": pq.tolist(), "delta": delta.tolist(),
               "distance": float(distance), "gap": float(np.float32(distance - effective))}
        if distance <= effective:
            terminal, certified, hit = "distance_within_effective_thickness", True, True
            row["terminal"] = terminal
            rows.append(row)
            break
        miss, details = separated(first_start, first_finish, second_start, second_finish, delta, effective)
        row["delta_axis_separated"] = details
        if miss:
            terminal, certified = "delta_axis_separated", True
            row["terminal"] = terminal
            rows.append(row)
            break
        if iteration == 0:
            cross_axes = [np.cross(sub(p10, p00), sub(q10, q00)).astype("<f4"),
                          np.cross(sub(p11, p01), sub(q11, q01)).astype("<f4")]
            row["initial_cross_axis_checks"] = []
            axis_miss = False
            for index, axis in enumerate(cross_axes):
                orient = dot3(axis, sub(first_start[0], second_start[0]))
                if orient < 0:
                    axis = mul(axis, np.float32(-1))
                ok, diag = separated(first_start, first_finish, second_start, second_finish, axis, effective)
                row["initial_cross_axis_checks"].append({"axis": axis.tolist(), "orientation_dot": float(orient),
                                                          "certified": ok, "details": diag})
                if ok:
                    terminal, certified, axis_miss = "initial_cross_axis_separated", True, True
                    row["terminal"] = terminal
                    rows.append(row)
                    break
            if axis_miss:
                break
        if not speed > FLOOR:
            terminal, certified = "stationary_certified_miss", True
            row["terminal"] = terminal
            rows.append(row)
            break
        step = np.float32(max(float(div(np.float32(distance - effective), speed)), 1.0e-6))
        row["step"] = float(step)
        time = np.float32(time + step)
        if time > np.float32(1.0 + np.float32(8) * EPS):
            terminal, certified = "advanced_past_end", True
            row["terminal"] = terminal
            rows.append(row)
            break
        time = np.float32(min(float(time), 1.0))
        rows.append(row)
    return {"limit": limit, "certified": certified, "hit": hit, "terminal": terminal,
            "effective_thickness": float(effective), "distance_roundoff": float(distance_roundoff),
            "speed_bound": float(speed), "iterations": rows}


def closest64(x):
    p0, p1, q0, q1 = x
    u, v, w = p1-p0, q1-q0, p0-q0
    a, e, f = float(u@u), float(v@v), float(v@w)
    if a == 0 and e == 0: s = t = 0.
    elif a == 0: s, t = 0., min(1., max(0., f/e))
    else:
        c = float(u@w)
        if e == 0: t, s = 0., min(1., max(0., -c/a))
        else:
            b = float(u@v); den = a*e-b*b
            s = min(1., max(0., (b*f-c*e)/den)) if den > 0 else 0.
            t = (b*s+f)/e
            if t < 0: t, s = 0., min(1., max(0., -c/a))
            elif t > 1: t, s = 1., min(1., max(0., (b-c)/a))
    return p0+s*u, q0+t*v


def fp64_reference(starts, finishes, thickness):
    a, b = np.asarray(starts, np.float64), np.asarray(finishes, np.float64)
    scale = max(float(np.max(np.abs(np.concatenate((a, b))))), float(thickness), 1e-12)
    eff = float(thickness) + 8 * float(EPS) * scale
    def distance(t):
        p, q = closest64(a + t*(b-a))
        return float(np.linalg.norm(p-q))
    count = 20001
    ts = np.linspace(0, 1, count)
    ds = np.asarray([distance(float(t)) for t in ts])
    idx = int(np.argmin(ds)); lo, hi = ts[max(0, idx-1)], ts[min(count-1, idx+1)]
    phi = (math.sqrt(5)-1)/2
    c, d = hi-phi*(hi-lo), lo+phi*(hi-lo)
    fc, fd = distance(c), distance(d)
    for _ in range(100):
        if fc <= fd:
            hi, d, fd = d, c, fc; c = hi-phi*(hi-lo); fc = distance(c)
        else:
            lo, c, fc = c, d, fd; d = lo+phi*(hi-lo); fd = distance(d)
    tm = (lo+hi)/2
    return {"method": "FP64 promoted captured FP32 endpoints; dense sampling plus local golden-section refinement; not a formal CCD certificate",
            "sample_count": count, "minimum_distance": distance(tm), "minimum_time": tm,
            "effective_thickness": eff, "clearance": distance(tm)-eff}


def extract():
    m371, p371, a371 = read_capture(C371)
    m358, p358, a358 = read_capture(C358)
    if (m371["root"], m371["iteration"]) != (1201, 7) or (m358["root"], m358["iteration"], m358["trial"]) != (1201, 7, 23):
        raise ValueError("wrong root/Newton/trial selection")
    if m371["source_physics_fingerprint"] != m358["source_physics_fingerprint"]:
        raise ValueError("physics fingerprint mismatch")
    cand371, cand371_meta = get_array(a371, "deformable_contact_candidates", "<u4", (18336, 4))
    cand358, cand358_meta = get_array(a358, "deformable_candidates_private_trial", "<u4", (18336, 4), "private_contact_rebuild_before_free_body_terms")
    if tuple(cand371[124]) != PAIR or tuple(cand358[124]) != PAIR:
        raise ValueError("captured candidate identity mismatch")
    acc371, acc371_meta = get_array(a371, "fem_accepted_nodes", "<u4", (36,20))
    acc358, acc358_meta = get_array(a358, "fem_accepted_nodes", "<u4", (36,20), "candidate_before_private_contact_rebuild")
    cor371, cor371_meta = get_array(a371, "fem_solution_correction", "<u4", (78,4))
    cor358, cor358_meta = get_array(a358, "fem_solution_correction", "<u4", (78,4), "candidate_before_private_contact_rebuild")
    if not np.array_equal(acc371, acc358) or not np.array_equal(cor371, cor358):
        raise ValueError("N7 base or Newton correction mismatch")
    base371, base371_meta = get_array(a371, "fem_candidate_nodes", "<f4", (36,20))
    trial358_check, trial358_check_meta = get_array(a358, "fem_armijo_candidate_nodes", "<f4", (36,20), "private_contact_rebuild_before_free_body_terms")
    native_failure = selected_native_failure(L371)
    source103_trial_status = selected_n7_trial_status(L358)
    alpha = np.float32(native_failure["trial_alpha"])
    dt = np.float32(m371["dt_seconds"])
    # Reconstruct source106's actual N7 trial geometry using the captured
    # center/correction and the alpha recorded by its native Armijo trace.
    # Source103 trial358 is retained only as a cross-build check.
    accepted371_f = acc371.view("<f4").reshape(36,20)
    correction371_f = cor371.view("<f4").reshape(78,4)
    nodes106_trial = base371.copy()
    free = nodes106_trial[:,11] <= np.float32(0.5)  # restAndFixed.w
    velocity_new = add(base371[:,4:7], mul(correction371_f[:36,:3], alpha))
    reference_new = add(accepted371_f[:,16:19], mul(dt, velocity_new))
    nodes106_trial[free,4:7] = velocity_new[free]
    nodes106_trial[free,16:19] = reference_new[free]
    reference_mode = nodes106_trial[:,19] == np.float32(1.0)
    position_reference = add(base371[:,8:11], reference_new)
    position_legacy = add(accepted371_f[:,:3], mul(dt, velocity_new))
    nodes106_trial[free & reference_mode,:3] = position_reference[free & reference_mode]
    nodes106_trial[free & ~reference_mode,:3] = position_legacy[free & ~reference_mode]
    if not np.isfinite(nodes106_trial).all():
        raise ValueError("reconstructed source106 trial candidate is nonfinite")
    position_ulp = u32_ulp_distance(nodes106_trial[:,:3], trial358_check[:,:3])
    velocity_ulp = u32_ulp_distance(nodes106_trial[:,4:7], trial358_check[:,4:7])
    prim, prim_meta = get_array(a358, "surface_primitives_private_trial", "<u4", (192,16), "private_contact_rebuild_before_free_body_terms")
    nodes358_meta = trial358_check_meta
    dispatch, dispatch_meta = get_array(a358, "dispatch_buffer", "<u4", (48,), "candidate_before_private_contact_rebuild")
    ranges, ranges_meta = get_array(a358, "cohesive_surface_face_ranges", "<u4", (192,4), "candidate_before_private_contact_rebuild")
    refs, refs_meta = get_array(a358, "cohesive_surface_face_refs", "<u4", (8,), "candidate_before_private_contact_rebuild")
    left, right = prim[24], prim[127]
    if left[4] != 24 or right[4] != 127 or left[5] != 0 or right[5] != 0:
        raise ValueError("pair primitive identity/kind mismatch")
    if ranges[24,1] != 0 or ranges[127,1] != 0:
        raise ValueError("pair unexpectedly has cohesive patch references")
    grid_count, fem_count = int(dispatch[7]), int(dispatch[8])
    thickness = dispatch.view("<f4")[44]
    points0, points1 = [], []
    for primitive, local_corner in ((left,1),(left,2),(right,0),(right,1)):
        global_node = int(primitive[local_corner])
        if global_node < grid_count or global_node >= grid_count + fem_count:
            raise ValueError("pair node not a captured FEM node")
        row = nodes106_trial[global_node-grid_count]
        finish = row[:3].copy()
        start = sub(finish, mul(dt, row[4:7]))
        points0.append(start); points1.append(finish)
    return {
        "manifests": (m371,m358), "payloads": (p371,p358), "arrays": (a371,a358),
        "pair": PAIR, "starts": f32(points0), "finishes": f32(points1),
        "thickness": float(thickness),
        "metadata": {
            "pair_candidate_371": cand371_meta, "pair_candidate_358": cand358_meta,
            "accepted_371": acc371_meta, "accepted_358": acc358_meta,
            "correction_371": cor371_meta, "correction_358": cor358_meta,
            "base_candidate_371": base371_meta, "trial_candidate_358": nodes358_meta,
            "primitive_358": prim_meta,
            "dispatch_358": dispatch_meta, "ranges_358": ranges_meta, "refs_358": refs_meta,
            "ranges_24": ranges[24].tolist(), "ranges_127": ranges[127].tolist(),
            "refs": refs.tolist(), "primitive24": left.tolist(), "primitive127": right.tolist(),
            "dt_seconds_f32": float(dt), "dt_bits": f"{int(dt.view('<u4')):08x}",
            "thickness_f32": float(thickness), "thickness_bits": f"{int(np.asarray(thickness).view('<u4')):08x}",
            "source_start_expression": "finish - dt * velocity, componentwise FP32",
            "source106_native_trial_alpha": float(alpha),
            "source106_native_trial_alpha_bits": f"{int(alpha.view('<u4')):08x}",
            "source106_trial_status": native_failure["trial_status"],
            "source103_trial358_status": source103_trial_status,
            "source106_trial_candidate_vs_source103_trial358": {
                "position_max_ulp": int(position_ulp.max()),
                "velocity_max_ulp": int(velocity_ulp.max()),
                "position_exact_components": int(np.count_nonzero(position_ulp == 0)),
                "velocity_exact_components": int(np.count_nonzero(velocity_ulp == 0)),
            },
        }
    }


def source_function(path):
    text = path.read_text()
    begin = text.index("inline bool edgeEdgeCCD(\n    const float3 p00,")
    end = text.index("// Direct production-CCD regression entry point.", begin)
    return text[begin:end]


def main():
    e = extract()
    f106, f103 = source_function(S106/"matter/src/metal/contact.metalinc"), source_function(S103/"matter/src/metal/contact.metalinc")
    if f106 != f103:
        raise ValueError("edgeEdgeCCD function changed between source103 and source106")
    traces = {str(n): trace32(e["starts"], e["finishes"], e["thickness"], n) for n in (16,17,33)}
    geometry = fp64_reference(e["starts"], e["finishes"], e["thickness"])
    results = {
      "schema": "numi.cardboard.source106-private-ee-reproduction.v1",
      "scope": "CPU-only exact-input reconstruction and source-transcribed FP32 EE trace; no GPU dispatch and no production edits",
      "native_observation": {
        "run371_manifest": str(C371), "run371_manifest_sha256": sha(C371),
        "run371_payload_sha256": sha(e["payloads"][0]), "trial358_manifest": str(C358),
        "trial358_manifest_sha256": sha(C358), "trial358_payload_sha256": sha(e["payloads"][1]),
        "root": 1201, "newton_iteration": 7, "trial358": 23,
        "source_physics_fingerprint": e["manifests"][0]["source_physics_fingerprint"],
        "source106_device_program_fingerprint": e["manifests"][0]["device_program_fingerprint"],
        "source103_device_program_fingerprint": e["manifests"][1]["device_program_fingerprint"],
        "candidate_identity": list(PAIR), "candidate_count_center371": 400,
        "diagnostic_words": [24,127,1,0],
        "diagnostic_decoding": "source106 EE failure writes [candidate.x,candidate.y,edgeA,edgeB]; this is primitive pair 24/127, edgeA=1, edgeB=0. The source106 generic VT failure marker stores diagnostic.w=3+direction. Pair incidence ranges are empty, so the mortar complement path is not entered.",
        "native_n7_trial_status": e["metadata"]["source106_trial_status"],
        "native_n7_trial_alpha": e["metadata"]["source106_native_trial_alpha"],
        "native_n7_trial_log_sha256": sha(L371),
        "source103_trial358_log_sha256": sha(L358),
        "source103_trial358_actual_failure": e["metadata"]["source103_trial358_status"],
      },
      "lineage": {
        "accepted_nodes_371_sha256": e["metadata"]["accepted_371"]["sha256"],
        "accepted_nodes_358_sha256": e["metadata"]["accepted_358"]["sha256"],
        "accepted_nodes_equal_bitwise": True,
        "solution_correction_371_sha256": e["metadata"]["correction_371"]["sha256"],
        "solution_correction_358_sha256": e["metadata"]["correction_358"]["sha256"],
        "solution_correction_equal_bitwise": True,
        "pair_candidate_hash_371": e["metadata"]["pair_candidate_371"]["sha256"],
        "pair_candidate_hash_358": e["metadata"]["pair_candidate_358"]["sha256"],
        "pair_identity_equal": True,
        "source106_trial_reconstruction_vs_source103_trial358": e["metadata"]["source106_trial_candidate_vs_source103_trial358"],
        "device_programs_differ": True,
        "geometry_basis": "source106 N7 trial positions/velocities are reconstructed from its center nodes, accepted state, correction, and recorded Armijo alpha using nm_fem_apply_solution semantics. Source103 trial358 is only a cross-build geometry comparison; node position/velocity ULP distances are reported and device fingerprints differ. The edgeEdgeCCD bodies match byte-for-byte.",
      },
      "input": {
        "edgeA_local_corners": list(EDGE_A), "edgeB_local_corners": list(EDGE_B),
        "points_start_f32": e["starts"].tolist(), "points_start_f32_bits": [f"{x:08x}" for x in e["starts"].view("<u4").reshape(-1)],
        "points_finish_f32": e["finishes"].tolist(), "points_finish_f32_bits": [f"{x:08x}" for x in e["finishes"].view("<u4").reshape(-1)],
        "thickness_m_f32": e["thickness"], "other_inputs": e["metadata"],
      },
      "source_bindings": {
        "source106_snapshot": str(S106), "source106_contact_sha256": sha(S106/"matter/src/metal/contact.metalinc"),
        "source106_common_sha256": sha(S106/"matter/src/metal/common.metalinc"),
        "source106_shared_sha256": sha(S106/"matter/include/numi/matter/shared.h"),
        "source103_snapshot": str(S103), "source103_contact_sha256": sha(S103/"matter/src/metal/contact.metalinc"),
        "edgeEdgeCCD_function_sha256": sha_bytes(f106.encode()),
        "source106_failure_marker": "nm_contact_narrowphase_deformable triangle/triangle edge loop: diagnostic(candidate.identity.x,candidate.identity.y,edgeA,edgeB)",
      },
      "results": {"source_transcribed_fp32": traces, "independent_fp64_geometric_minimum": geometry},
      "interpretation": {
        "failure_type": "source106 native N7 last-trial generic EE call returned uncertified; the status is not proof of physical interpenetration",
        "vt_hypothesis": "ruled out for diagnostic [24,127,1,0]; separate VT replay is an unrelated certified miss",
        "mortar_hypothesis": "ruled out for pair24/127: both incidence ranges have count zero",
        "16_17_33": "17 is the source106 production edgeEdgeCCD loop bound; 16 is a comparison cap; 33 is diagnostic extension only. The CPU transcription hits on call 17 while native source106 returns uncertified, so this does not establish that a larger production bound would resolve the native case.",
        "cross_build_status": "source103 trial358 uses the same accepted state and correction and its reconstructed geometry is within one position ULP of source106, but its recorded failure is a different pair (92/127, edge0/edge0). It is not the source106 native failure.",
        "precision": "CPU FP32 transcription follows source expression ordering but is not bitwise Metal evidence. The hit-vs-uncertified margin is only about 0.05–0.06 nm; the independent FP64 minimum lies within the effective thickness band but is not a formal production CCD certificate.",
      },
      "limitations": [
        "No GPU execution or native replay in this package.",
        "Trial358 binary was source103; source106 371 supplies the authoritative N7 status, pair, and line-search alpha. Its candidate geometry is reconstructed from source106 arrays and cross-checked against the source103 candidate without claiming that the source103 device capture is source106.",
        "A later 33-iteration CPU extension cannot qualify a larger production bound.",
        "The FP64 geometric minimum does not prove continuous separation under the production rounding/certificate rules."
      ]
    }
    out = OUT/"reproduction.json"
    out.write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    print("source106_private_ee_reproduction=PASS")
    print("pair=(24,127) edgeA=1 edgeB=0 candidate=124 ranges=empty")
    for limit, trace in traces.items():
        print("fp32_transcription",limit,trace["terminal"],trace["certified"],len(trace["iterations"]))
    print("fp64_min",geometry["minimum_distance"],"at",geometry["minimum_time"],"clearance",geometry["clearance"])
    print("reproduction_sha256",sha(out))


if __name__ == "__main__":
    main()
