#!/usr/bin/env python3
"""Rebind existing NHSKIN v5 weights without changing its reference surface.

This is source preparation, not physical admission. Stored normals stay byte
identical; regenerated coarse tissue frames, mass and pose gates remain required.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


class RebindError(ValueError):
    pass


def decode(raw):
    if len(raw) < 60:
        raise RebindError("truncated NHSKIN header")
    magic, abi, nb, nv, ni, fingerprint = struct.unpack_from("<8s5I", raw)
    if magic != b"NHSKIN1\0" or abi != 5 or nb == 0 or nv == 0 or ni % 3:
        raise RebindError("requires an existing NHSKIN v5 triangle surface")
    vo = 60 + nb * 36
    io = vo + nv * 56
    wo = io + ni * 4
    if len(raw) != wo + nv * nb * 4:
        raise RebindError("NHSKIN byte count does not match its header")
    bindings = np.frombuffer(raw, dtype=[("body", "<u4"), ("v", "<f4", (8,))],
                             count=nb, offset=60)
    vertices = np.frombuffer(raw, dtype="<f4", count=nv * 14, offset=vo).reshape(nv, 14)
    faces = np.frombuffer(raw, dtype="<u4", count=ni, offset=io).reshape(-1, 3)
    weights = np.frombuffer(raw, dtype="<f4", count=nv * nb, offset=wo).reshape(nv, nb)
    if len(set(map(int, bindings["body"]))) != nb:
        raise RebindError("duplicate binding body identity")
    if faces.size and int(faces.max()) >= nv:
        raise RebindError("face index out of range")
    if not np.isfinite(vertices[:, :6]).all() or not np.isfinite(bindings["v"]).all():
        raise RebindError("nonfinite geometry or binding")
    validate_weights(weights)
    return dict(raw=raw, nb=nb, nv=nv, ni=ni, fingerprint=fingerprint,
                source_sha256=raw[28:60], vertoff=vo, indexoff=io,
                weightoff=wo, binds=bindings, verts=vertices, inds=faces, weights=weights)


def remap_unchanged_source_faces(source, target, source_face_ids):
    """Resolve retained face witnesses after reordering, without trusting ordinals.

    Only oriented, unique triangles on unchanged original vertex coordinates
    may transfer. Regional edits elsewhere do not invalidate this local mapping.
    Callers still pin the two payload hashes and the matched pose separately.
    """
    if (source["source_sha256"] != target["source_sha256"] or
            source["fingerprint"] != target["fingerprint"] or
            source["binds"].tobytes() != target["binds"].tobytes()):
        raise RebindError("source face registration or ancestry differs")
    ids = np.asarray(source_face_ids)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or
            np.any(ids < 0) or np.any(ids >= len(source["inds"]))):
        raise RebindError("invalid retained source face identity")
    def oriented(tri):
        a, b, c = map(int, tri)
        return min((a, b, c), (b, c, a), (c, a, b))
    wanted = {oriented(source["inds"][i]) for i in ids}
    matches = {}
    for i, tri in enumerate(target["inds"]):
        key = oriented(tri)
        if key in wanted:
            matches.setdefault(key, []).append(i)
    result = []
    for i in ids:
        tri = source["inds"][i]
        candidates = matches.get(oriented(tri), [])
        if len(candidates) != 1:
            raise RebindError("retained source face missing or ambiguous: " + str(i))
        if (int(np.max(tri)) >= target["nv"] or
                not np.array_equal(source["verts"][tri, :3], target["verts"][tri, :3])):
            raise RebindError("retained source face coordinates changed: " + str(i))
        result.append(candidates[0])
    return np.asarray(result, dtype=np.int64)


def validate_weights(w):
    if not np.isfinite(w).all() or np.any(w < 0):
        raise RebindError("weights must be finite and nonnegative")
    if np.max(np.abs(w.astype(float).sum(axis=1) - 1)) > 5e-7:
        raise RebindError("weight partition is not unity")


def rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.dot(q, q) - 1) > 1e-5:
        raise RebindError("invalid rigid quaternion")
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def reference_transforms(payload, pose, root_lift_m):
    lift = np.asarray(root_lift_m, dtype=float)
    if lift.shape != (3,) or not np.isfinite(lift).all():
        raise RebindError("root lift requires three finite metres")
    rows = pose["source_poses"]
    by_body = {int(p["core_body"]): p for p in rows}
    if len(rows) != len(by_body):
        raise RebindError("duplicate pose body identity")
    matrices, offsets = [], []
    for rec in payload["binds"]:
        body = int(rec["body"])
        if body not in by_body:
            raise RebindError("missing reference pose body " + str(body))
        p = by_body[body]
        vals = rec["v"].astype(float)
        r = rotation(p["quaternion_xyzw"])
        t = np.asarray(p["position"], dtype=float) + lift
        if not np.isfinite(t).all() or t.shape != (3,) or vals[7] <= 0:
            raise RebindError("invalid binding scale or reference translation")
        matrices.append(r @ (rotation(vals[3:7]) * vals[7]))
        offsets.append(t + r @ vals[:3])
    return np.asarray(matrices), np.asarray(offsets)


def registered_positions_float32(payload, snapshot, *, frame):
    """Replay the native ABI5 binding order in an explicitly named body frame.

    This is source geometry only, before accepted FEM deformation. No root lift
    or placement transform is inferred. Snapshot coordinates already use metres.
    """
    if frame not in ("rest", "current"):
        raise RebindError("registered frame must explicitly be rest or current")
    for key, expected in (("registration_fingerprint32", payload["fingerprint"]),
                          ("binding_count", payload["nb"]),
                          ("vertex_count", payload["nv"]), ("payload_abi", 5)):
        if snapshot.get(key) != expected:
            raise RebindError("registered snapshot identity mismatch: " + key)
    rows = snapshot.get("all_bodies", snapshot.get("bodies"))
    if not isinstance(rows, list):
        raise RebindError("registered snapshot lacks body poses")
    by_body = {}
    for row in rows:
        body = row.get("body_index")
        if isinstance(body, bool) or not isinstance(body, int) or body < 0 or body in by_body:
            raise RebindError("invalid or duplicate registered body identity")
        by_body[body] = row
    xyz = np.asarray(payload["verts"][:, :3], dtype=np.float32)
    weights = np.asarray(payload["weights"], dtype=np.float32)
    validate_weights(weights)
    if xyz.shape != (payload["nv"], 3) or weights.shape != (payload["nv"], payload["nb"]):
        raise RebindError("registered source dimensions differ")
    def rotate(q, points):
        # Keep the native cross-product order and Float32 intermediate rounding.
        x, y, z = points.T
        a, b, c, w = q
        u = np.float32(2) * (b*z - c*y)
        v = np.float32(2) * (c*x - a*z)
        t = np.float32(2) * (a*y - b*x)
        return np.stack((x + ((w*u + b*t) - c*v),
                         y + ((w*v + c*u) - a*t),
                         z + ((w*t + a*v) - b*u)), axis=1)
    out = np.zeros_like(xyz)
    for lane, binding in enumerate(payload["binds"]):
        body = int(binding["body"])
        row = by_body.get(body, {}).get(frame)
        if not isinstance(row, dict):
            raise RebindError("missing registered " + frame + " body " + str(body))
        p = np.asarray(row.get("position_world_m"), dtype=np.float32)
        q = np.asarray(row.get("orientation_world_xyzw"), dtype=np.float32)
        vals = binding["v"]
        if (p.shape != (3,) or q.shape != (4,) or
                not np.isfinite(p).all() or not np.isfinite(q).all() or
                not np.isfinite(vals).all() or vals[7] <= 0 or
                abs(float(q.astype(float) @ q.astype(float)) - 1) > 1e-5 or
                abs(float(vals[3:7].astype(float) @ vals[3:7].astype(float)) - 1) > 1e-5):
            raise RebindError("invalid registered rigid pose or binding")
        local = vals[:3] + rotate(vals[3:7], xyz) * vals[7]
        world = p + rotate(q, local)
        out = out + weights[:, lane, None] * world
    if not np.isfinite(out).all():
        raise RebindError("nonfinite registered emitted coordinates")
    return out


def registered_rest_vertex_normals_float32(payload, snapshot, vertex_rows):
    """Construct selected ABI5 normals from emitted geometry in native rest.

    Include every incident face, including faces outside the changed patch.
    Return only requested rows; never replace unrelated authored normals.
    This prepares surface records, not physical or collision admission.
    """
    rows = np.asarray(vertex_rows)
    if rows.ndim != 1 or rows.dtype.kind not in "iu":
        raise RebindError("normal rows must be an integer vector")
    rows = rows.astype(np.int64)
    if (len(np.unique(rows)) != len(rows) or
            (len(rows) and (rows.min() < 0 or rows.max() >= payload["nv"]))):
        raise RebindError("normal rows are duplicate or out of range")
    faces = np.asarray(payload["inds"])
    if (faces.ndim != 2 or faces.shape[1] != 3 or
            faces.dtype.kind not in "iu" or
            (faces.size and (faces.min() < 0 or faces.max() >= payload["nv"]))):
        raise RebindError("normal topology contains invalid source indices")
    xyz = registered_positions_float32(payload, snapshot, frame="rest").astype(float)
    if not len(rows):
        return np.empty((0, 3), dtype=np.float32)
    selected = np.zeros(payload["nv"], dtype=bool)
    selected[rows] = True
    # Invalid geometry elsewhere cannot alter a requested patch, but every
    # incident face must be finite and nondegenerate.
    incident = faces[np.any(selected[faces], axis=1)]
    tri = xyz[incident]
    cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    if not np.isfinite(cross).all() or np.any(lengths <= 1e-20):
        raise RebindError("degenerate incident face in rest-normal construction")
    accum = np.zeros_like(xyz)
    for corner in range(3):
        np.add.at(accum, incident[:, corner], cross)
    normals = accum[rows]
    length = np.linalg.norm(normals, axis=1)
    if not np.isfinite(length).all() or np.any(length <= 1e-20):
        raise RebindError("unreferenced or cancelling selected rest normal")
    return (normals/length[:, None]).astype(np.float32)


def positions(payload, a, b):
    w = payload["weights"].astype(float)
    return np.einsum("nk,kij,nj->ni", w, a, payload["verts"][:, :3].astype(float)) + w @ b


def replacement_weights(payload, other):
    if (payload["nb"], payload["nv"]) != (other["nb"], other["nv"]):
        raise RebindError("replacement weight dimensions differ")
    if (payload["fingerprint"] != other["fingerprint"] or
            payload["source_sha256"] != other["source_sha256"]):
        raise RebindError("replacement source ancestry or registration differs")
    if payload["binds"].tobytes() != other["binds"].tobytes():
        raise RebindError("replacement binding identities/transforms differ")
    if payload["ni"] != other["ni"] or not np.array_equal(payload["inds"], other["inds"]):
        raise RebindError("weight-only replacement triangle topology differs")
    return other["weights"].copy()



def merge_weight_repairs(payload, repairs):
    """Compose weight-only repairs against one pinned ancestor, never last-wins.

    Rest coordinates are regenerated once by rebind, so disjoint regions keep
    the same reference surface. Topology/geometry changes are not imported here.
    """
    merged = payload["weights"].copy()
    occupied = np.zeros(payload["nv"], dtype=bool)
    counts = []
    for other in repairs:
        weights = replacement_weights(payload, other)
        changed = np.any(weights != payload["weights"], axis=1)
        conflict = occupied & changed & np.any(weights != merged, axis=1)
        if np.any(conflict):
            row = int(np.flatnonzero(conflict)[0])
            raise RebindError("conflicting regional weight repairs at vertex " + str(row))
        merged[changed] = weights[changed]
        occupied |= changed
        counts.append(int(np.count_nonzero(changed)))
    return merged, counts


def merge_geometry_repairs(payload, repairs):
    """Compose compatible XYZ/normal edits against one pinned NHSKIN ancestor.

    Bindings, topology, source identities and both weight representations must
    remain byte-identical. Conflicting rows fail rather than silently choosing
    the last candidate. No interpolation or normal regeneration is performed;
    callers must gate the combined surface and physical mapping after merging.
    """
    baseline = payload['verts'][:, :6].copy()
    merged = baseline.copy()
    occupied = np.zeros(payload['nv'], dtype=bool)
    counts = []
    for other in repairs:
        replacement_weights(payload, other)
        if (other['raw'][:other['vertoff']] != payload['raw'][:payload['vertoff']]
                or other['inds'].tobytes() != payload['inds'].tobytes()
                or other['weights'].tobytes() != payload['weights'].tobytes()
                or other['verts'][:, 6:].tobytes() != payload['verts'][:, 6:].tobytes()):
            raise RebindError('geometry repair changed source, bindings, topology or weights')
        values = other['verts'][:, :6]
        changed = np.any(values.view('<u4') != baseline.view('<u4'), axis=1)
        conflict = occupied & changed & np.any(values.view('<u4') != merged.view('<u4'), axis=1)
        if np.any(conflict):
            raise RebindError('conflicting regional geometry repairs at vertex ' + str(int(np.flatnonzero(conflict)[0])))
        merged[changed] = values[changed]
        occupied |= changed
        counts.append(int(np.count_nonzero(changed)))
    blob = bytearray(payload['raw'])
    rows = np.flatnonzero(occupied)
    for i in rows:
        start = payload['vertoff'] + int(i)*56
        blob[start:start+24] = merged[i].astype('<f4').tobytes()
    # Revalidate Float32 geometry and the original source contract on emission.
    decode(bytes(blob))
    return bytes(blob), dict(changed_rows=list(map(int, rows)), regional_row_counts=counts)


def local_rigid_weights(payload, target_body, seeds, shell_gates):
    gates = np.asarray(shell_gates, dtype=float)
    if (not len(gates) or not np.isfinite(gates).all() or
            np.any(gates < 0) or np.any(gates > 1)):
        raise RebindError("shell gates must be finite fractions")
    lanes = {int(b["body"]): i for i, b in enumerate(payload["binds"])}
    if target_body not in lanes:
        raise RebindError("target body has no skin binding")
    seeds = set(map(int, seeds))
    if not seeds or min(seeds) < 0 or max(seeds) >= payload["nv"]:
        raise RebindError("invalid local seed vertex")
    adjacency = [set() for _ in range(payload["nv"])]
    for a, b, c in payload["inds"]:
        for i, j in [(int(a), int(b)), (int(b), int(c)), (int(c), int(a))]:
            adjacency[i].add(j)
            adjacency[j].add(i)
    distance = {i: 0 for i in seeds}
    front = seeds
    for ring in range(1, len(gates)):
        front = {j for i in front for j in adjacency[i] if j not in distance}
        distance.update((j, ring) for j in front)
    w = payload["weights"].copy()
    rows = np.array(sorted(distance), dtype=int)
    gate = np.array([gates[distance[int(i)]] for i in rows])
    target = np.zeros((len(rows), payload["nb"]))
    target[:, lanes[target_body]] = 1
    w[rows] = (w[rows].astype(float) * (1 - gate[:, None]) +
               target * gate[:, None]).astype("<f4")
    return w


def rebind(payload, weights, a, b, maximum_error_m=2e-6):
    weights = np.asarray(weights, dtype="<f4")
    if weights.shape != payload["weights"].shape:
        raise RebindError("weight dimensions differ")
    validate_weights(weights)
    if (a.shape != (payload["nb"], 3, 3) or b.shape != (payload["nb"], 3)
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise RebindError("invalid reference transforms")
    if not np.isfinite(maximum_error_m) or maximum_error_m <= 0:
        raise RebindError("reference error limit must be finite and positive")
    rows = np.flatnonzero(np.any(weights != payload["weights"], axis=1))
    if not len(rows):
        return payload["raw"], dict(changed_rows=[], reference_error_max_m=0.0)
    target = positions(payload, a, b)
    aa = np.einsum("nk,kij->nij", weights[rows].astype(float), a)
    bb = weights[rows].astype(float) @ b
    condition = np.linalg.cond(aa)
    if not np.isfinite(condition).all() or condition.max() >= 10:
        raise RebindError("singular or ill-conditioned inverse skinning")
    xyz = np.linalg.solve(aa, (target[rows] - bb)[..., None])[..., 0].astype("<f4")
    blob = bytearray(payload["raw"])
    for k, i in enumerate(rows):
        struct.pack_into("<3f", blob, payload["vertoff"] + int(i)*56, *map(float, xyz[k]))
        order = np.lexsort((np.arange(payload["nb"]), -weights[i].astype(float)))[:4]
        # The format always carries four lanes, even for tiny synthetic fixtures.
        if len(order) < 4:
            order = np.pad(order, (0, 4-len(order)), mode="constant")
            top = np.r_[weights[i, order[:payload["nb"]]], np.zeros(4-payload["nb"])]
        else:
            top = weights[i, order].astype(float)
        top = top.astype(float) / float(top.sum())
        struct.pack_into("<4I4f", blob, payload["vertoff"] + int(i)*56 + 24,
                         *map(int, order), *map(float, top))
        struct.pack_into("<" + str(payload["nb"]) + "f", blob,
                         payload["weightoff"] + int(i)*payload["nb"]*4,
                         *map(float, weights[i]))
    result = decode(bytes(blob))
    error = np.linalg.norm(positions(result, a, b) - target, axis=1)
    if not np.isfinite(error).all() or error.max() > maximum_error_m:
        raise RebindError("Float32 emission exceeds reference surface tolerance")
    return bytes(blob), dict(changed_rows=list(map(int, rows)),
                             reference_error_max_m=float(error.max()),
                             inverse_condition_max=float(condition.max()))



def localize_weight_support(payload, target_body, excluded_bodies, thresholds, strength=1.0):
    """Remove distant influence continuously, returning it to a declared owner.

    Body identities and support thresholds are explicit source-preparation inputs.
    This does not infer anatomy, add physical constraints, or admit a tissue model.
    Rebind the returned weights once to preserve the registered reference surface.
    """
    ids = list(map(int, excluded_bodies))
    t = np.asarray(thresholds, dtype=float)
    if (t.shape != (2,) or not np.isfinite(t).all() or
            not 0 <= t[0] < t[1] <= 1 or not np.isfinite(strength) or
            not 0 <= strength <= 1):
        raise RebindError("support requires ordered finite fractional thresholds and strength")
    lanes = {int(b["body"]): i for i, b in enumerate(payload["binds"])}
    if (not ids or len(set(ids)) != len(ids) or target_body in ids or
            target_body not in lanes or any(i not in lanes for i in ids)):
        raise RebindError("support body identities must be distinct existing bindings")
    source = payload["weights"]
    w = source.astype(float).copy()
    gate = np.clip((w[:, lanes[target_body]] - t[0]) / (t[1] - t[0]), 0, 1)
    gate = gate * gate * (3 - 2 * gate) * strength
    removed_lanes = [lanes[i] for i in ids]
    removed = w[:, removed_lanes] * gate[:, None]
    w[:, removed_lanes] -= removed
    w[:, lanes[target_body]] += removed.sum(axis=1)
    result = w.astype("<f4")
    validate_weights(result)
    return result



def validate_protected_source_faces(payload, candidate_faces, protected_source_face_ids):
    """Require exact oriented protected triangles, including multiplicity."""
    from collections import Counter
    ids = np.asarray(protected_source_face_ids)
    f = np.asarray(candidate_faces)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids)
            or (len(ids) and (ids.min() < 0 or ids.max() >= len(payload["inds"])))
            or f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in "iu"):
        raise RebindError("invalid protected source face identities")
    def canonical(t):
        t = tuple(map(int, t))
        return min(t, t[1:]+t[:1], t[2:]+t[:2])
    candidate = Counter(map(canonical, f))
    baseline = Counter(map(canonical, payload["inds"]))
    for i in ids:
        key = canonical(payload["inds"][int(i)])
        if candidate[key] != baseline[key]:
            raise RebindError("protected source face " + str(int(i)) +
                              " is missing, reversed or duplicated")
    return int(len(ids))

def append_registered_surface_patch(payload, pose, rest_pose, root_lift_m,
                                    vertices_world_m, faces, ancestry_offsets,
                                    source_vertex_ids, source_weights,
                                    maximum_error_m=2e-6,
                                    protected_source_face_ids=None):
    """Append an inferred surface patch with explicit existing-source ancestry.

    Original vertex/binding/weight records remain exact. Constructed coordinates
    need not lie on an old face; affine ancestry transfers binding identity only,
    and its geometric displacement is reported, never labeled a measurement.
    This prepares NHSKIN, not a mechanical mapping or physical admission.
    """
    x = np.asarray(vertices_world_m, dtype=float)
    f0 = np.asarray(faces)
    off0, sid0 = np.asarray(ancestry_offsets), np.asarray(source_vertex_ids)
    sw = np.asarray(source_weights, dtype=float)
    old = payload["nv"]
    if (x.ndim != 2 or x.shape[1] != 3 or len(x) <= old
            or not np.isfinite(x).all() or f0.ndim != 2 or f0.shape[1] != 3
            or f0.dtype.kind not in "iu" or not len(f0)
            or off0.ndim != 1 or off0.dtype.kind not in "iu"
            or sid0.ndim != 1 or sid0.dtype.kind not in "iu"):
        raise RebindError("invalid patch geometry or ancestry arrays")
    f, off, sid = f0.astype(np.int64), off0.astype(np.int64), sid0.astype(np.int64)
    if (len(off) != len(x)+1 or off[0] != 0 or np.any(np.diff(off) <= 0)
            or off[-1] != len(sid) or sw.shape != sid.shape
            or not np.isfinite(sw).all() or np.any(sw < 0)
            or sid.min() < 0 or sid.max() >= old
            or f.min() < 0 or f.max() >= len(x)):
        raise RebindError("invalid patch ancestry partition or face identity")
    if not np.isfinite(maximum_error_m) or maximum_error_m <= 0:
        raise RebindError("reference error limit must be finite and positive")
    row = np.repeat(np.arange(len(x)), np.diff(off))
    sums = np.bincount(row, weights=sw, minlength=len(x))
    if np.max(np.abs(sums-1)) > 1e-10:
        raise RebindError("patch ancestry is not an affine partition")
    if (not np.array_equal(off[:old+1], np.arange(old+1))
            or not np.array_equal(sid[:old], np.arange(old))
            or not np.array_equal(sw[:old], np.ones(old))):
        raise RebindError("existing vertex ancestry must remain identity")
    a, b = reference_transforms(payload, pose, root_lift_m)
    original = positions(payload, a, b)
    if not np.array_equal(x[:old], original):
        raise RebindError("existing registered vertex coordinates must remain exact")
    protected_count = None
    if protected_source_face_ids is not None:
        protected_count = validate_protected_source_faces(payload, f, protected_source_face_ids)
    extra = len(x)-old
    w = np.zeros((extra, payload["nb"]), dtype=float)
    np.add.at(w, row[old:]-old,
              sw[old:, None]*payload["weights"][sid[old:]].astype(float))
    w /= w.sum(1)[:, None]
    w = w.astype("<f4")
    # Correct the representable partition residual without changing donors.
    dominant = np.argmax(w, axis=1)
    w[np.arange(extra), dominant] += (1-w.astype(float).sum(1)).astype("<f4")
    validate_weights(w)
    aa = np.einsum("nk,kij->nij", w.astype(float), a)
    bb = w.astype(float) @ b
    cond = np.linalg.cond(aa)
    if not np.isfinite(cond).all() or cond.max() >= 10:
        raise RebindError("ill-conditioned patch inverse skinning")
    xyz = np.linalg.solve(aa, (x[old:]-bb)[..., None])[..., 0].astype("<f4")
    # ABI5 stores normals in the default articulated world frame.
    current = {int(v["core_body"]): rotation(v["quaternion_xyzw"])
               for v in pose["source_poses"]}
    default = {int(v["core_body"]): rotation(v["quaternion_xyzw"])
               for v in rest_pose["source_poses"]}
    if len(current) != len(pose["source_poses"]) or len(default) != len(rest_pose["source_poses"]):
        raise RebindError("duplicate normal-frame body identity")
    try:
        maps = np.asarray([current[int(v["body"])] @ default[int(v["body"])].T
                           for v in payload["binds"]])
    except KeyError as exc:
        raise RebindError("missing default normal-frame body identity") from exc
    normal_map = np.einsum("nk,kij->nij", w.astype(float), maps)
    nc = np.linalg.cond(normal_map)
    if not np.isfinite(nc).all() or nc.max() >= 10:
        raise RebindError("ill-conditioned patch normal transform")
    tri = x[f]
    cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    if np.any(np.linalg.norm(cross, axis=1) <= 1e-20):
        raise RebindError("degenerate patch surface face")
    normal = np.zeros_like(x)
    np.add.at(normal, f.ravel(), np.repeat(cross, 3, axis=0))
    length = np.linalg.norm(normal[old:], axis=1)
    if np.any(length <= 1e-20):
        raise RebindError("unreferenced or cancelling patch vertex normal")
    target_n = normal[old:]/length[:, None]
    local_n = np.linalg.solve(normal_map, target_n[..., None])[..., 0]
    local_n /= np.linalg.norm(local_n, axis=1)[:, None]
    header = bytearray(payload["raw"][:60])
    struct.pack_into("<2I", header, 16, len(x), 3*len(f))
    records = bytearray(extra*56)
    for k in range(extra):
        order = np.lexsort((np.arange(payload["nb"]), -w[k].astype(float)))[:4]
        if len(order) != 4:
            raise RebindError("patch emission requires at least four bindings")
        top = w[k, order].astype(float)
        top /= top.sum()
        struct.pack_into("<6f4I4f", records, k*56,
                         *xyz[k], *local_n[k].astype("<f4"),
                         *map(int, order), *map(float, top))
    data = (bytes(header)+payload["raw"][60:payload["vertoff"]]
            +payload["raw"][payload["vertoff"]:payload["indexoff"]]
            +bytes(records)+f.astype("<u4").tobytes()
            +payload["weights"].tobytes()+w.tobytes())
    result = decode(data)
    error = np.linalg.norm(positions(result, a, b)-x, axis=1)
    if not np.isfinite(error).all() or error.max() > maximum_error_m:
        raise RebindError("Float32 patch emission exceeds reference tolerance")
    projected = np.zeros((extra, 3))
    np.add.at(projected, row[old:]-old, sw[old:, None]*original[sid[old:]])
    displacement = np.linalg.norm(x[old:]-projected, axis=1)
    return data, dict(added_vertices=extra, faces=len(f),
                     original_vertex_records_exact=True,
                     protected_source_faces_checked=protected_count,
                     reference_error_max_m=float(error.max()),
                     inverse_condition_max=float(cond.max()),
                     inferred_displacement_from_ancestry_max_m=float(displacement.max()),
                     physical_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--reference-pose", type=Path, required=True)
    parser.add_argument("--root-lift-m", type=float, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--weights-from", type=Path, nargs="+")
    mode.add_argument("--local-body", type=int)
    mode.add_argument("--support-body", type=int)
    parser.add_argument("--excluded-bodies", type=int, nargs="+")
    parser.add_argument("--support-thresholds", type=float, nargs=2)
    parser.add_argument("--support-strength", type=float, default=1.0)
    parser.add_argument("--seed-vertices", type=int, nargs="+")
    parser.add_argument("--shell-gates", type=float, nargs="+")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.input_sha256:
        raise RebindError("input hash mismatch")
    if args.output.exists() or args.receipt.exists():
        raise RebindError("refusing to overwrite source or evidence")
    p = decode(raw)
    repair_inputs = []
    if args.weights_from:
        repairs = []
        for path in args.weights_from:
            repair_raw = path.read_bytes()
            repairs.append(decode(repair_raw))
            repair_inputs.append(dict(path=str(path.resolve()),
                sha256=hashlib.sha256(repair_raw).hexdigest()))
        w, repair_counts = merge_weight_repairs(p, repairs)
        for source, count in zip(repair_inputs, repair_counts):
            source["changed_weight_rows"] = count
    elif args.support_body is not None:
        if args.excluded_bodies is None or args.support_thresholds is None:
            parser.error("--support-body requires --excluded-bodies and --support-thresholds")
        w = localize_weight_support(p, args.support_body, args.excluded_bodies,
                                    args.support_thresholds, args.support_strength)
    else:
        if args.seed_vertices is None or args.shell_gates is None:
            parser.error("--local-body requires --seed-vertices and --shell-gates")
        w = local_rigid_weights(p, args.local_body, args.seed_vertices, args.shell_gates)
    a, b = reference_transforms(p, json.loads(args.reference_pose.read_text()), args.root_lift_m)
    result, report = rebind(p, w, a, b)
    args.output.write_bytes(result)
    report.update(input_sha256=args.input_sha256,
                  output_sha256=hashlib.sha256(result).hexdigest(),
                  reference_pose_sha256=hashlib.sha256(args.reference_pose.read_bytes()).hexdigest(),
                  root_lift_m=args.root_lift_m,
                  admission="SOURCE_PREPARATION_ONLY; physical mappings, mass and pose gates required")
    if args.weights_from:
        report["weight_repairs"] = repair_inputs
        report["composition"] = "common ancestor; disjoint or identical overlapping rows; one reference rebind"
    if args.support_body is not None:
        report["support_repair"] = dict(target_body=args.support_body,
            excluded_bodies=args.excluded_bodies, thresholds=args.support_thresholds,
            strength=args.support_strength, gate="cubic smoothstep of original target weight",
            reference_rebind_once=True)
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
