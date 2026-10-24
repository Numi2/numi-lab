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


def merge_edge_refinement_repairs(payload, repairs):
    """Compose disjoint conforming edge refinements of one pinned NHSKIN.

    Each repair is (decoded_payload, parent_face_rows, new_vertex_edges,
    new_vertex_weights). Ancestry refers only to original source rows; no
    pose, interpolation, normal or physical admission is inferred. Conflicting
    old-row edits and overlapping refined parents fail. Combined geometry,
    mechanics and motion still require their normal acceptance checks.
    """
    n, nf = payload['nv'], len(payload['inds'])
    source_edge_parents = {}
    for parent, tri in enumerate(payload['inds']):
        for a,b in zip(tri,np.roll(tri,-1)):
            source_edge_parents.setdefault(tuple(sorted((int(a),int(b)))),[]).append(parent)
    merged = payload['verts'].copy()
    occupied = np.zeros(n, dtype=bool)
    replaced = set()
    by_parent = {}
    extra_v, extra_w, provenance = [], [], []
    next_row = n
    for other, parent_rows, edge_rows, edge_weights in repairs:
        if (other['nb'] != payload['nb'] or other['nv'] < n or
                other['source_sha256'] != payload['source_sha256'] or
                other['fingerprint'] != payload['fingerprint'] or
                other['binds'].tobytes() != payload['binds'].tobytes()):
            raise RebindError('refinement registration or source differs')
        if (other['verts'][:n, 6:].tobytes() != payload['verts'][:, 6:].tobytes() or
                other['weights'][:n].tobytes() != payload['weights'].tobytes()):
            raise RebindError('refinement changed original binding weights')
        values = other['verts'][:n, :6]
        changed = np.any(values.view('<u4') != payload['verts'][:, :6].view('<u4'), axis=1)
        conflict = occupied & changed & np.any(values.view('<u4') != merged[:, :6].view('<u4'), axis=1)
        if np.any(conflict):
            raise RebindError('conflicting regional geometry repairs at vertex ' + str(int(np.flatnonzero(conflict)[0])))
        merged[changed, :6] = values[changed]
        occupied |= changed
        parents, edges, weights = map(np.asarray, (parent_rows, edge_rows, edge_weights))
        extra = other['nv']-n
        if (parents.shape != (len(other['inds']),) or parents.dtype.kind not in 'iu' or
                np.any(parents < 0) or np.any(parents >= nf) or
                edges.shape != (extra, 2) or edges.dtype.kind not in 'iu' or
                np.any(edges < 0) or np.any(edges >= n) or
                np.any(edges[:, 0] == edges[:, 1]) or
                weights.shape != (extra, 2) or not np.isfinite(weights).all() or
                np.any(weights <= 0) or np.max(np.abs(weights.sum(axis=1)-1), initial=0) > 1e-12):
            raise RebindError('invalid original-edge refinement ancestry')
        expected = np.sum(payload['weights'][edges].astype(float)*weights[:, :, None], axis=1)
        if np.max(np.abs(expected-other['weights'][n:]), initial=0) > 5e-7:
            raise RebindError('new binding weights disagree with edge ancestry')
        new_ids = set(map(int, other['inds'][other['inds'] >= n]))
        if new_ids != set(range(n, other['nv'])):
            raise RebindError('unreferenced appended refinement vertex')
        groups = [[] for _ in range(nf)]
        for fi, parent in enumerate(parents):
            groups[int(parent)].append(fi)
        seen_edges = set()
        for k, edge in enumerate(edges):
            key = tuple(sorted(map(int,edge)))
            if key in seen_edges or key not in source_edge_parents:
                raise RebindError('duplicate or absent original split edge')
            seen_edges.add(key)
            for parent in source_edge_parents[key]:
                if not np.any(other['inds'][groups[parent]] == n+k):
                    raise RebindError('refinement leaves a source-edge T junction')
        local_replaced = set()
        for parent, rows in enumerate(groups):
            if not rows:
                raise RebindError('source face ancestry is missing')
            triangles = other['inds'][rows]
            original = payload['inds'][parent]
            if len(rows) == 1 and np.array_equal(triangles[0], original):
                continue
            local_replaced.add(parent)
            # Expected oriented boundary is the original three edges, split at
            # every declared new row used by these children. Interior edges
            # must occur exactly twice with opposite orientation.
            vertices = set(map(int, triangles.ravel()))
            if not set(map(int, original)).issubset(vertices):
                raise RebindError('refined parent lost an original corner')
            expected_boundary = set()
            for a, b in zip(original, np.roll(original, -1)):
                a,b = int(a),int(b)
                points = [(0.,a),(1.,b)]
                for row in vertices:
                    if row < n:
                        if row not in original:
                            raise RebindError('refined parent uses unrelated source corner')
                        continue
                    edge, weight = edges[row-n], weights[row-n]
                    if set(map(int, edge)) == {a,b}:
                        points.append((float(weight[1] if int(edge[0]) == a else weight[0]),row))
                    elif not set(map(int, edge)).issubset(set(map(int, original))):
                        raise RebindError('refined child has unrelated edge ancestry')
                points.sort()
                if any(points[i][0] == points[i+1][0] for i in range(len(points)-1)):
                    raise RebindError('duplicate refinement point on source edge')
                expected_boundary.update((points[i][1],points[i+1][1]) for i in range(len(points)-1))
            incidence = {}
            for tri in triangles:
                if len(set(map(int,tri))) != 3:
                    raise RebindError('refined face repeats a corner')
                for a,b in zip(tri,np.roll(tri,-1)):
                    a,b=int(a),int(b)
                    incidence.setdefault(tuple(sorted((a,b))),[]).append((a,b))
            boundary = set()
            for pair in incidence.values():
                if len(pair) == 1: boundary.add(pair[0])
                elif len(pair) != 2 or pair[0] != pair[1][::-1]:
                    raise RebindError('refined parent is nonmanifold or inconsistently oriented')
            adjacency = [set() for _ in triangles]
            edge_faces = {}
            for k,tri in enumerate(triangles):
                for a,b in zip(tri,np.roll(tri,-1)):
                    key = tuple(sorted((int(a),int(b))))
                    for j in edge_faces.get(key,[]):
                        adjacency[k].add(j); adjacency[j].add(k)
                    edge_faces.setdefault(key,[]).append(k)
            reached, pending = set(), [0]
            while pending:
                k = pending.pop()
                if k not in reached:
                    reached.add(k); pending.extend(adjacency[k]-reached)
            if len(reached) != len(triangles):
                raise RebindError('disconnected refined parent')
            if boundary != expected_boundary or len(vertices)-len(incidence)+len(triangles) != 1:
                raise RebindError('refined parent is not its original oriented disk')
            mapped = triangles.astype(np.int64).copy()
            mapped[mapped >= n] += next_row-n
            by_parent[parent] = mapped
        if replaced & local_replaced:
            raise RebindError('overlapping regional refined source faces')
        replaced |= local_replaced
        provenance.append(dict(added_vertex_offset=next_row, added_vertices=extra,
                               replaced_source_faces=sorted(local_replaced)))
        extra_v.append(other['verts'][n:].copy()); extra_w.append(other['weights'][n:].copy())
        next_row += extra
    out_faces, parent_ids = [], []
    for parent, original in enumerate(payload['inds']):
        children = by_parent.get(parent, original[None,:])
        out_faces.extend(children); parent_ids.extend([parent]*len(children))
    verts = np.concatenate([merged]+extra_v)
    weights = np.concatenate([payload['weights']]+extra_w)
    faces = np.asarray(out_faces,dtype='<u4').reshape(-1,3)
    if len(verts) > np.iinfo(np.uint32).max or faces.size > np.iinfo(np.uint32).max:
        raise RebindError('merged refinement exceeds NHSKIN capacity')
    header = bytearray(payload['raw'][:60]); struct.pack_into('<2I',header,16,len(verts),faces.size)
    raw = bytes(header)+payload['binds'].tobytes()+verts.astype('<f4').tobytes()+faces.tobytes()+weights.astype('<f4').tobytes()
    decode(raw)
    return raw, dict(changed_original_rows=list(map(int,np.flatnonzero(occupied))),
                     face_parent_source_rows=parent_ids, regions=provenance,
                     physical_admission=False)


def _validate_duplicate_source_repair(payload, repair):
    """Validate a same-face-count patch that duplicates existing source rows.

    This ancestry kind is intentionally distinct from edge interpolation: a
    duplicated row may move in XYZ and normal space, but its source/binding
    identity is one existing source row. No zero-length source edge is
    fabricated and no geometric-overlap claim is made.
    """
    other = repair["payload"]
    n, nf = payload["nv"], len(payload["inds"])
    if (other["nv"] <= n or len(other["inds"]) != nf or
            other["nb"] != payload["nb"] or
            other["source_sha256"] != payload["source_sha256"] or
            other["fingerprint"] != payload["fingerprint"] or
            other["binds"].tobytes() != payload["binds"].tobytes()):
        raise RebindError("duplicate-source patch registration or topology domain differs")
    if (other["weights"][:n].tobytes() != payload["weights"].tobytes() or
            other["raw"][other["vertoff"]+24:other["vertoff"]+n*56] !=
            payload["raw"][payload["vertoff"]+24:payload["vertoff"]+n*56]):
        raise RebindError("duplicate-source patch changed original binding records")

    parent = np.asarray(repair["face_parent_source_rows"])
    protected = np.asarray(repair["protected_source_face_ids"])
    replaced = np.asarray(repair["replaced_source_face_ids"])
    dup_out = np.asarray(repair["duplicated_output_vertex_ids"])
    dup_src = np.asarray(repair["duplicated_source_vertex_ids"])
    offsets = np.asarray(repair["ancestry_offsets"])
    source_ids = np.asarray(repair["source_vertex_ids"])
    source_weights = np.asarray(repair["source_weights"], dtype=float)
    integer_arrays = (parent, protected, replaced, dup_out, dup_src,
                      offsets, source_ids)
    if any(a.ndim != 1 or a.dtype.kind not in "iu" for a in integer_arrays):
        raise RebindError("duplicate-source identities must be integer vectors")
    if (parent.shape != (nf,) or np.any(parent < 0) or np.any(parent >= nf) or
            len(np.unique(parent)) != nf or not np.array_equal(np.sort(parent), np.arange(nf))):
        raise RebindError("duplicate-source face parent map must cover each base face once")
    if (len(np.unique(protected)) != len(protected) or
            len(np.unique(replaced)) != len(replaced) or
            np.any(protected < 0) or np.any(protected >= nf) or
            np.any(replaced < 0) or np.any(replaced >= nf) or
            np.intersect1d(protected, replaced).size or
            not np.array_equal(np.sort(np.concatenate((protected, replaced))), np.arange(nf))):
        raise RebindError("protected and replaced source faces must partition the base")
    extra = other["nv"] - n
    if (dup_out.shape != (extra,) or dup_src.shape != (extra,) or
            not np.array_equal(np.sort(dup_out), np.arange(n, other["nv"])) or
            np.any(dup_src < 0) or np.any(dup_src >= n)):
        raise RebindError("duplicate source/output rows do not cover appended vertices")
    if (offsets.shape != (other["nv"]+1,) or offsets[0] != 0 or
            np.any(np.diff(offsets) <= 0) or offsets[-1] != len(source_ids) or
            source_weights.shape != source_ids.shape or
            not np.isfinite(source_weights).all() or np.any(source_weights < 0) or
            np.any(source_ids < 0) or np.any(source_ids >= n)):
        raise RebindError("invalid duplicate-source CSR partition")
    if (not np.array_equal(offsets[:n+1], np.arange(n+1)) or
            not np.array_equal(source_ids[:n], np.arange(n)) or
            not np.array_equal(source_weights[:n], np.ones(n))):
        raise RebindError("original source ancestry must remain identity")
    for output_row, source_row in zip(dup_out, dup_src):
        lo, hi = int(offsets[output_row]), int(offsets[output_row+1])
        if (hi-lo != 1 or int(source_ids[lo]) != int(source_row) or
                source_weights[lo] != 1.0):
            raise RebindError("appended row is not a direct unit-weight source duplicate")
        if other["weights"][output_row].tobytes() != payload["weights"][source_row].tobytes():
            raise RebindError("duplicated row dense binding differs from its source row")
        src0 = payload["vertoff"] + int(source_row)*56 + 24
        dup0 = other["vertoff"] + int(output_row)*56 + 24
        src_lanes = struct.unpack_from("<4I4f", payload["raw"], src0)
        dup_lanes = struct.unpack_from("<4I4f", other["raw"], dup0)
        src_index, src_weight = src_lanes[:4], src_lanes[4:]
        dup_index, dup_weight = dup_lanes[:4], dup_lanes[4:]
        if src_index != dup_index:
            raise RebindError("duplicated row compact binding IDs differ from its source row")
        src_weight = np.asarray(src_weight, dtype="<f4")
        dup_weight = np.asarray(dup_weight, dtype="<f4")
        # Compact ABI lanes are separately rounded from the dense influence
        # table. Permit only a one-ULP representation change for nonnegative
        # Float32 weights; donor identities and full dense rows remain exact.
        if (not np.isfinite(dup_weight).all() or np.any(dup_weight < 0) or
                np.any(np.abs(src_weight.view("<u4").astype(np.int64) -
                              dup_weight.view("<u4").astype(np.int64)) > 1)):
            raise RebindError("duplicated row compact binding weights differ from its source row")

    # Every candidate triangle has one declared source-face identity. Mapping
    # duplicated vertices back to their source rows must recover that parent's
    # oriented source triangle exactly. This proves lineage, not overlap.
    output_to_source = np.arange(other["nv"], dtype=np.int64)
    output_to_source[dup_out] = dup_src
    projected = output_to_source[other["inds"]]
    if not np.array_equal(projected, payload["inds"][parent]):
        raise RebindError("face parent identity disagrees with duplicated source rows")
    raw_faces_by_parent = np.empty_like(payload["inds"])
    raw_faces_by_parent[parent] = other["inds"]
    touched = set(map(int, np.flatnonzero(np.any(raw_faces_by_parent != payload["inds"], axis=1))))
    if set(map(int, other["inds"].ravel()[other["inds"].ravel() >= n])) != set(map(int, dup_out)):
        raise RebindError("duplicate-source patch contains unused or undeclared appended rows")
    return dict(other=other, parent=parent.astype(np.int64),
                protected=set(map(int, protected)), replaced=set(map(int, replaced)),
                touched=touched,
                duplicate_output=dup_out.astype(np.int64),
                duplicate_source=dup_src.astype(np.int64),
                ancestry_source_ids=source_ids.astype(np.int64),
                ancestry_weights=source_weights)


def merge_source_topology_repairs(payload, *, edge_repairs=(), duplicate_repairs=()):
    """Compose disjoint edge refinements and explicit source-row duplicates.

    Edge repairs use the existing conforming-disk validator. Duplicate repairs
    may relabel vertices in a full same-face-count candidate, with each appended
    row carrying an explicit direct source-row CSR entry. All repairs are
    compared to one original payload and overlapping parent faces fail.
    The result is source-preparation output only; it does not certify pose,
    mechanics, tissue mass, physical admission, or geometric overlap ancestry.
    """
    if edge_repairs:
        edge_raw, edge_report = merge_edge_refinement_repairs(payload, edge_repairs)
    else:
        edge_raw = payload["raw"]
        edge_report = dict(changed_original_rows=[],
                           face_parent_source_rows=list(range(len(payload["inds"]))),
                           regions=[], physical_admission=False)
    current = decode(edge_raw)
    n, nf = payload["nv"], len(payload["inds"])
    parent_rows = np.asarray(edge_report["face_parent_source_rows"], dtype=np.int64)
    edge_replaced = {int(p) for region in edge_report.get("regions", [])
                     for p in region.get("replaced_source_faces", [])}
    current_vertices = current["verts"].copy()
    current_weights = current["weights"].copy()
    current_faces = current["inds"].astype(np.int64).copy()
    changed_original = set(map(int, edge_report["changed_original_rows"]))
    duplicate_used = set(edge_replaced)
    duplicate_regions = []
    next_row = current["nv"]

    for repair in duplicate_repairs:
        validated = _validate_duplicate_source_repair(payload, repair)
        other = validated["other"]
        overlap = duplicate_used & validated["touched"]
        if overlap:
            raise RebindError("overlapping regional source-face patches: " + str(min(overlap)))

        # Merge permitted original XYZ/normal edits by row. Binds, dense
        # weights and the compact four-lane binding representation are pinned.
        values = other["verts"][:n, :6]
        baseline = payload["verts"][:, :6]
        changed = np.any(values.view("<u4") != baseline.view("<u4"), axis=1)
        for row in np.flatnonzero(changed):
            row = int(row)
            if row in changed_original and not np.array_equal(
                    values[row].view("<u4"), current_vertices[row, :6].view("<u4")):
                raise RebindError("conflicting original vertex repair at vertex " + str(row))
            current_vertices[row, :6] = values[row]
            changed_original.add(row)

        remap = {}
        for output_row, source_row in zip(validated["duplicate_output"],
                                          validated["duplicate_source"]):
            remap[int(output_row)] = next_row
            current_vertices = np.concatenate((current_vertices,
                                                other["verts"][int(output_row):int(output_row)+1]))
            current_weights = np.concatenate((current_weights,
                                               other["weights"][int(output_row):int(output_row)+1]))
            next_row += 1
        # Each source parent currently has exactly one unsplit face because
        # duplicate patches are disjoint from edge-refined parent faces.
        for parent in sorted(validated["touched"]):
            rows = np.flatnonzero(parent_rows == parent)
            if len(rows) != 1 or parent in edge_replaced:
                raise RebindError("duplicate-source patch overlaps a refined/missing parent face")
            candidate_row = int(np.flatnonzero(validated["parent"] == parent)[0])
            tri = other["inds"][candidate_row].astype(np.int64).copy()
            for i, vertex in enumerate(tri):
                if int(vertex) >= n:
                    tri[i] = remap[int(vertex)]
            current_faces[int(rows[0])] = tri
        duplicate_used |= validated["touched"]
        duplicate_regions.append(dict(
            added_vertices=len(remap),
            replaced_source_faces=sorted(validated["replaced"]),
            touched_source_faces=sorted(validated["touched"]),
            duplicate_source_vertex_ids=validated["duplicate_source"].tolist(),
            source_face_lineage="declared parent row and direct source-row projection; not geometric overlap"))

    if not duplicate_regions:
        return edge_raw, edge_report
    header = bytearray(payload["raw"][:60])
    struct.pack_into("<2I", header, 16, len(current_vertices), 3*len(current_faces))
    raw = (bytes(header) + payload["raw"][60:payload["vertoff"]] +
           current_vertices.astype("<f4").tobytes() +
           current_faces.astype("<u4").tobytes() + current_weights.astype("<f4").tobytes())
    result = decode(raw)
    report = dict(
        changed_original_rows=sorted(changed_original),
        face_parent_source_rows=parent_rows.tolist(),
        regions=list(edge_report.get("regions", [])) + duplicate_regions,
        added_duplicate_vertices=sum(r["added_vertices"] for r in duplicate_regions),
        ancestry_semantics="edge interpolation for edge regions; direct source-row identity for duplicate regions",
        geometric_overlap_certified=False,
        physical_admission=False,
        native_admission=False)
    return result["raw"], report


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
