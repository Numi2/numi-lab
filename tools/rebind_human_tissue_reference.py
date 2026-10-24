#!/usr/bin/env python3
"""Build a hash-pinned NHTISS ABI-5 common-reference candidate.

This is an internal source-preparation tool. It keeps the existing NHTISS ABI,
uses an explicitly pinned native emitter for the current-pose replay, and never
claims physical admission. Per-record reference coordinates are inferred from
that replay; they are not measured source OBJ coordinates.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np


class ReferenceError(ValueError):
    """Invalid pin, source payload, pose, or reference construction."""


HEADER = struct.Struct("<8s6I32s")
BIND_DTYPE = np.dtype([("body", "<u4"), ("transform", "<f4", (8,))])
VERT_DTYPE = np.dtype([
    ("p", "<f4", (3,)), ("n", "<f4", (3,)),
    ("binding", "<u4", (4,)), ("w", "<f4", (4,)),
])


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(path: str | Path, expected: str, label: str) -> str:
    if len(expected) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in expected):
        raise ReferenceError(f"{label} SHA-256 pin must be 64 hexadecimal characters")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ReferenceError(f"{label} SHA-256 mismatch: expected {expected.lower()}, got {actual}")
    return actual


def decode_nhtiss(raw: bytes) -> dict[str, Any]:
    if len(raw) < HEADER.size:
        raise ReferenceError("truncated NHTISS header")
    magic, abi, nr, nb, nv, ni, registration, source_id = HEADER.unpack_from(raw)
    if magic != b"NHTISS4\0" or abi != 5:
        raise ReferenceError("requires existing NHTISS4 ABI 5")
    if nr == 0 or nb == 0 or nv == 0 or ni == 0 or ni % 3:
        raise ReferenceError("invalid NHTISS record/binding/vertex/index counts")
    rec_off = HEADER.size
    bind_off = rec_off + nr * 32
    vert_off = bind_off + nb * 36
    index_off = vert_off + nv * 56
    expected_bytes = index_off + ni * 4
    if len(raw) != expected_bytes:
        raise ReferenceError(f"NHTISS byte count mismatch: expected {expected_bytes}, got {len(raw)}")
    records = np.frombuffer(raw, dtype="<u4", count=nr * 8, offset=rec_off).reshape(nr, 8).copy()
    bindings = np.frombuffer(raw, dtype=BIND_DTYPE, count=nb, offset=bind_off).copy()
    vertices = np.frombuffer(raw, dtype=VERT_DTYPE, count=nv, offset=vert_off).copy()
    indices = np.frombuffer(raw, dtype="<u4", count=ni, offset=index_off).copy()
    parsed = {
        "raw": raw, "magic": magic, "abi": abi, "nr": nr, "nb": nb,
        "nv": nv, "ni": ni, "registration": registration, "source_id": source_id,
        "records": records, "bindings": bindings, "vertices": vertices, "indices": indices,
        "record_offset": rec_off, "binding_offset": bind_off,
        "vertex_offset": vert_off, "index_offset": index_off,
    }
    validate_nhtiss(parsed)
    return parsed


def _integer_array(value: Any, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ReferenceError(f"{name} must be a {ndim}D integer array")
    return array.astype(np.int64, copy=False)


def _oriented_patch_boundary(faces: np.ndarray, vertex_count: int, label: str) -> list[tuple[int, int]]:
    """Validate an oriented triangular disk patch and return its directed boundary."""
    faces = _integer_array(faces, f"{label} faces", 2)
    if faces.shape[1:] != (3,) or len(faces) == 0:
        raise ReferenceError(f"{label} faces must be a nonempty Nx3 array")
    if np.any(faces < 0) or np.any(faces >= vertex_count):
        raise ReferenceError(f"{label} face vertex escapes the surface vertex range")
    edge_uses: dict[tuple[int, int], list[tuple[int, int]]] = {}
    triangle_keys: set[tuple[int, int, int]] = set()
    for face in faces:
        a, b, c = map(int, face)
        if len({a, b, c}) != 3:
            raise ReferenceError(f"{label} contains a repeated-index triangle")
        key = tuple(sorted((a, b, c)))
        if key in triangle_keys:
            raise ReferenceError(f"{label} contains duplicate triangles")
        triangle_keys.add(key)
        for u, v in ((a, b), (b, c), (c, a)):
            edge_uses.setdefault((min(u, v), max(u, v)), []).append((u, v))
    boundary: list[tuple[int, int]] = []
    face_adjacency = [set() for _ in range(len(faces))]
    edge_face_uses: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        a, b, c = map(int, face)
        for u, v in ((a, b), (b, c), (c, a)):
            edge_face_uses.setdefault((min(u, v), max(u, v)), []).append(face_index)
    for edge, uses in edge_uses.items():
        if len(uses) == 1:
            boundary.append(uses[0])
        elif len(uses) == 2:
            if uses[0] != (uses[1][1], uses[1][0]):
                raise ReferenceError(f"{label} has an inconsistently oriented internal edge {edge}")
            left, right = edge_face_uses[edge]
            face_adjacency[left].add(right)
            face_adjacency[right].add(left)
        else:
            raise ReferenceError(f"{label} has a nonmanifold edge {edge}")
    reached = {0}
    stack = [0]
    while stack:
        current = stack.pop()
        for neighbor in face_adjacency[current] - reached:
            reached.add(neighbor)
            stack.append(neighbor)
    if len(reached) != len(faces):
        raise ReferenceError(f"{label} faces are not one edge-connected patch")
    used_vertices = set(map(int, faces.reshape(-1)))
    if len(used_vertices) - len(edge_uses) + len(faces) != 1:
        raise ReferenceError(f"{label} patch is not topologically a disk")
    if len(boundary) < 3:
        raise ReferenceError(f"{label} has no simple disk boundary")
    outgoing: dict[int, int] = {}
    incoming: dict[int, int] = {}
    for u, v in boundary:
        if u in outgoing or v in incoming:
            raise ReferenceError(f"{label} boundary is not one directed simple loop")
        outgoing[u] = v
        incoming[v] = u
    if set(outgoing) != set(incoming):
        raise ReferenceError(f"{label} boundary has open or branched vertices")
    start = min(outgoing)
    loop = [start]
    current = start
    for _ in range(len(boundary)):
        current = outgoing[current]
        if current == start:
            break
        loop.append(current)
    if current != start or len(loop) != len(boundary) or len(set(loop)) != len(loop):
        raise ReferenceError(f"{label} boundary is not a single simple cycle")
    return boundary


def replace_surface_topology_patch(
        payload_raw: bytes, *, stable_id: int,
        removed_face_rows: Any, replacement_faces_local: Any,
        boundary_cycle_local: Any, ancestry_offsets: Any, ancestry_face_rows: Any,
        construction_ancestry_label: str) -> tuple[bytes, dict[str, Any]]:
    """Replace one oriented disk patch in an existing NHTISS ABI-5 surface.

    Face ancestry is construction lineage only. It is not a geometric overlap,
    mass transfer, or FEM face-map certificate. The operation does not add
    vertices and preserves all position, normal, binding, and route-weight
    payload bytes.
    """
    label = str(construction_ancestry_label).strip()
    if not label or "construction" not in label.lower():
        raise ReferenceError("construction ancestry label must explicitly identify construction lineage")
    source = decode_nhtiss(payload_raw)
    matches = np.flatnonzero(source["records"][:, 6] == int(stable_id))
    if len(matches) != 1:
        raise ReferenceError(f"stable surface {stable_id} is absent or ambiguous")
    record_index = int(matches[0])
    record = source["records"][record_index].copy()
    fb, bc, fv, vc, fi, ic, sid, layer = map(int, record)
    source_faces_global = source["indices"][fi:fi + ic].reshape(-1, 3)
    source_faces_local = source_faces_global.astype(np.int64) - fv
    face_count = len(source_faces_local)
    removed = _integer_array(removed_face_rows, "removed source face rows", 1)
    if len(removed) == 0 or len(np.unique(removed)) != len(removed):
        raise ReferenceError("removed source face rows must be nonempty and unique")
    if np.any(removed < 0) or np.any(removed >= face_count):
        raise ReferenceError("removed source face row is out of range")
    removed_sorted = np.sort(removed)
    replacement = _integer_array(replacement_faces_local, "replacement faces", 2)
    if replacement.shape[1:] != (3,) or len(replacement) == 0:
        raise ReferenceError("replacement faces must be a nonempty Nx3 array")
    cycle = _integer_array(boundary_cycle_local, "boundary cycle", 1)
    if len(cycle) < 3 or len(np.unique(cycle)) != len(cycle):
        raise ReferenceError("boundary cycle must contain at least three unique local vertices")
    if np.any(cycle < 0) or np.any(cycle >= vc):
        raise ReferenceError("boundary cycle vertex is out of range")
    cycle_edges = [(int(cycle[i]), int(cycle[(i + 1) % len(cycle)])) for i in range(len(cycle))]
    removed_faces = source_faces_local[removed_sorted]
    replacement_boundary = _oriented_patch_boundary(replacement, vc, "replacement")
    removed_boundary = _oriented_patch_boundary(removed_faces, vc, "removed source")
    if set(removed_boundary) != set(cycle_edges):
        raise ReferenceError("declared boundary cycle does not match removed source patch orientation")
    if set(replacement_boundary) != set(cycle_edges):
        raise ReferenceError("replacement patch does not preserve the exact oriented boundary cycle")
    # Geometry is deliberately not certified here, but zero-area Float32 faces
    # are rejected as invalid topology output before serialization.
    local_points = source["vertices"][fv:fv + vc]["p"].astype(np.float64)
    replaced_points = local_points[replacement]
    area_twice = np.linalg.norm(np.cross(replaced_points[:, 1] - replaced_points[:, 0],
                                         replaced_points[:, 2] - replaced_points[:, 0]), axis=1)
    if not np.isfinite(area_twice).all() or np.any(area_twice <= 0):
        raise ReferenceError("replacement contains a zero-area or nonfinite Float32 face")

    offsets = _integer_array(ancestry_offsets, "construction ancestry offsets", 1)
    parents = _integer_array(ancestry_face_rows, "construction ancestry face rows", 1)
    if len(offsets) != len(replacement) + 1 or offsets[0] != 0 or offsets[-1] != len(parents):
        raise ReferenceError("construction ancestry CSR offsets do not match replacement face count")
    if np.any(np.diff(offsets) < 0):
        raise ReferenceError("construction ancestry CSR offsets are not monotone")
    if np.any(parents < 0) or np.any(parents >= face_count) or not np.isin(parents, removed_sorted).all():
        raise ReferenceError("construction ancestry parent must be a removed source face row")
    for child in range(len(replacement)):
        row_parents = parents[offsets[child]:offsets[child + 1]]
        if len(row_parents) == 0 or len(np.unique(row_parents)) != len(row_parents):
            raise ReferenceError(f"replacement face {child} needs unique removed-face construction ancestry")

    old_to_new: list[int | None] = [None] * face_count
    insert_at = int(removed_sorted[0])
    removed_set = set(map(int, removed_sorted))
    new_local_faces: list[np.ndarray] = []
    insertion_written = False
    for old_row, face in enumerate(source_faces_local):
        if old_row == insert_at:
            first_child = len(new_local_faces)
            new_local_faces.extend(row.copy() for row in replacement)
            insertion_written = True
        if old_row in removed_set:
            continue
        new_row = len(new_local_faces)
        new_local_faces.append(face.copy())
        old_to_new[old_row] = new_row
    if not insertion_written:
        raise ReferenceError("failed to place replacement patch in source face sequence")
    if len(new_local_faces) != face_count - len(removed_sorted) + len(replacement):
        raise ReferenceError("replacement face accounting mismatch")

    # Require original record index spans to partition the source index array;
    # this prevents silently dropping unowned/reserved index bytes.
    records = source["records"].copy()
    by_old_start = sorted((int(row[4]), int(row[5]), i) for i, row in enumerate(records))
    cursor = 0
    for start, count, _row in by_old_start:
        if start != cursor:
            raise ReferenceError("surface index ranges do not exactly partition the source index array")
        cursor += count
    if cursor != source["ni"]:
        raise ReferenceError("surface index ranges leave unowned source indices")

    output_faces: list[np.ndarray] = []
    candidate_record_rows: dict[int, list[np.ndarray]] = {}
    new_index_cursor = 0
    for _old_start, _old_count, row_index in by_old_start:
        row = records[row_index]
        rfb, rbc, rfv, rvc, rfi, ric, rsid, _rlayer = map(int, source["records"][row_index])
        if rsid == int(stable_id):
            rows = np.asarray(new_local_faces, dtype=np.int64) + rfv
            records[row_index, 4] = new_index_cursor
            records[row_index, 5] = len(rows) * 3
            candidate_record_rows[rsid] = rows.copy()
        else:
            rows = source["indices"][rfi:rfi + ric].reshape(-1, 3).copy()
            records[row_index, 4] = new_index_cursor
            records[row_index, 5] = ric
            candidate_record_rows[rsid] = rows.copy()
        output_faces.append(rows.astype("<u4", copy=False).reshape(-1))
        new_index_cursor += rows.size
    new_indices = np.concatenate(output_faces).astype("<u4", copy=False)
    if new_indices.size > 0xFFFFFFFF:
        raise ReferenceError("replacement exceeds NHTISS uint32 index capacity")
    rebuilt_header = HEADER.pack(source["magic"], source["abi"], source["nr"], source["nb"],
                                source["nv"], int(new_indices.size), source["registration"], source["source_id"])
    result_raw = (rebuilt_header + records.astype("<u4", copy=False).tobytes() +
                  source["bindings"].tobytes() + source["vertices"].tobytes() + new_indices.tobytes())
    readback = decode_nhtiss(result_raw)
    if not np.array_equal(readback["bindings"], source["bindings"]):
        raise ReferenceError("topology replacement changed binding bytes")
    if not np.array_equal(readback["vertices"], source["vertices"]):
        raise ReferenceError("topology replacement changed vertex payload bytes")
    for row_index, source_record in enumerate(source["records"]):
        rfb, rbc, rfv, rvc, rfi, ric, rsid, rlayer = map(int, source_record)
        out_row = readback["records"][row_index]
        if tuple(map(int, out_row[[0, 1, 2, 3, 6, 7]])) != (rfb, rbc, rfv, rvc, rsid, rlayer):
            raise ReferenceError("topology replacement changed record identity or non-index ranges")
        if rsid != int(stable_id):
            if not np.array_equal(readback["indices"][int(out_row[4]):int(out_row[4] + out_row[5])],
                                  source["indices"][rfi:rfi + ric]):
                raise ReferenceError(f"topology replacement changed unrelated surface {rsid}")
        elif not np.array_equal(readback["indices"][int(out_row[4]):int(out_row[4] + out_row[5])],
                                candidate_record_rows[rsid].reshape(-1)):
            raise ReferenceError("serialized replacement face sequence mismatch")

    insertion_candidate_row = insert_at - int(np.count_nonzero(removed_sorted < insert_at))
    # Since rows before insert_at are all retained, this is simply insert_at
    # minus removed rows before the insertion point (zero for min(removed)).
    source_to_candidate = [None if old in removed_set else old - int(np.count_nonzero(removed_sorted < old)) + len(replacement)
                           if old > insert_at else old for old in range(face_count)]
    for old in range(face_count):
        if old not in removed_set and old < insert_at:
            source_to_candidate[old] = old
        elif old not in removed_set and old >= insert_at:
            source_to_candidate[old] = old - len(removed_sorted) + len(replacement)
    child_rows = list(range(insertion_candidate_row, insertion_candidate_row + len(replacement)))
    ancestry = [[int(x) for x in parents[offsets[i]:offsets[i + 1]]] for i in range(len(replacement))]
    report = {
        "schema": "numi.human.nhtiss.surface-topology-patch-replacement.v1",
        "source_payload_sha256": sha256_bytes(payload_raw),
        "candidate_payload_sha256": sha256_bytes(result_raw),
        "stable_id": int(stable_id),
        "record_index": record_index,
        "source_face_count": face_count,
        "candidate_face_count": len(new_local_faces),
        "source_index_count": source["ni"],
        "candidate_index_count": int(new_indices.size),
        "removed_source_face_rows": [int(x) for x in removed_sorted],
        "new_candidate_face_rows": child_rows,
        "source_face_row_to_candidate_face_row": source_to_candidate,
        "construction_ancestry": {
            "label": label,
            "semantics": "construction lineage only; not geometric overlap, material transfer, or a mechanical face-map certificate",
            "parent_face_rows_by_new_candidate_face": ancestry,
        },
        "preservation": {
            "all_vertex_position_normal_and_route_weight_bytes_exact": True,
            "all_binding_bytes_exact": True,
            "all_unselected_surface_face_sequences_exact": True,
            "target_nonpatch_face_sequence_exact": True,
        },
        "mechanical_face_map": {
            "previous_map_valid": False,
            "requires_regeneration": True,
            "candidate_face_rows_must_not_be_joined_by_stale_flat_offsets": True,
        },
        "gate": {"geometry_gate": False, "physical_admission": False, "native_admission": False},
    }
    return result_raw, report


def validate_nhtiss(payload: dict[str, Any]) -> None:
    records = payload["records"]
    bindings = payload["bindings"]
    vertices = payload["vertices"]
    indices = payload["indices"]
    if not np.isfinite(bindings["transform"]).all():
        raise ReferenceError("nonfinite binding transform")
    if not np.isfinite(vertices["p"]).all() or not np.isfinite(vertices["n"]).all():
        raise ReferenceError("nonfinite NHTISS position/normal")
    if not np.isfinite(vertices["w"]).all() or np.any(vertices["w"] < 0):
        raise ReferenceError("invalid NHTISS route weights")
    stable = list(map(int, records[:, 6]))
    if len(set(stable)) != len(stable):
        raise ReferenceError("duplicate stable surface identity")
    seen_binding_ranges: list[tuple[int, int, int]] = []
    seen_vertex_ranges: list[tuple[int, int, int]] = []
    seen_index_ranges: list[tuple[int, int, int]] = []
    for row in records:
        fb, bc, fv, vc, fi, ic, sid, _layer = map(int, row)
        if bc == 0 or vc == 0 or ic == 0 or ic % 3:
            raise ReferenceError(f"invalid record sizes for stable surface {sid}")
        if fb + bc > payload["nb"] or fv + vc > payload["nv"] or fi + ic > payload["ni"]:
            raise ReferenceError(f"record range out of bounds for stable surface {sid}")
        for ranges, pair in ((seen_binding_ranges, (fb, fb + bc, sid)),
                             (seen_vertex_ranges, (fv, fv + vc, sid)),
                             (seen_index_ranges, (fi, fi + ic, sid))):
            if any(max(pair[0], lo) < min(pair[1], hi) for lo, hi, _ in ranges):
                raise ReferenceError(f"overlapping surface record range for stable surface {sid}")
            ranges.append(pair)
        faces = indices[fi:fi + ic].reshape(-1, 3)
        if np.any(faces < fv) or np.any(faces >= fv + vc):
            raise ReferenceError(f"face index escapes stable surface {sid} vertex range")
        local_vertices = vertices[fv:fv + vc]
        active = local_vertices["w"] > np.float32(2e-6)
        active_sums = np.where(active, local_vertices["w"], 0).sum(axis=1)
        if np.any(np.abs(active_sums.astype(np.float64) - 1.0) > 2e-3):
            raise ReferenceError(f"active route weights do not sum to one for stable surface {sid}")
        if np.any(local_vertices["binding"][active] >= bc):
            raise ReferenceError(f"active local binding index is out of range for stable surface {sid}")
        if len(set(map(int, bindings[fb:fb + bc]["body"]))) != bc:
            raise ReferenceError(f"duplicate body binding within stable surface {sid}")
    # Header count arrays are authoritative only after every record range is safe.
    if int(np.max(indices)) >= payload["nv"]:
        raise ReferenceError("face index out of range")


def load_pose_map(pose_data: dict[str, Any], registration: int) -> dict[int, dict[str, Any]]:
    if int(pose_data.get("registration_fingerprint32", -1)) != int(registration):
        raise ReferenceError("pose registration fingerprint does not match NHTISS")
    rows = pose_data.get("all_bodies")
    if not isinstance(rows, list) or not rows:
        raise ReferenceError("pose snapshot must include all_bodies")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or "body_index" not in row or "current" not in row:
            raise ReferenceError("malformed all_bodies pose row")
        body = int(row["body_index"])
        if body in result:
            raise ReferenceError(f"duplicate pose body identity {body}")
        state = row["current"]
        pos = np.asarray(state.get("position_world_m"), dtype=np.float32)
        quat = np.asarray(state.get("orientation_world_xyzw"), dtype=np.float32)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ReferenceError(f"invalid current position for body {body}")
        if quat.shape != (4,) or not np.isfinite(quat).all():
            raise ReferenceError(f"invalid current quaternion for body {body}")
        norm = float(np.linalg.norm(quat.astype(np.float64)))
        if norm <= 0 or abs(norm - 1.0) > 2e-5:
            raise ReferenceError(f"zero or invalid current quaternion for body {body}: norm={norm}")
        result[body] = state
    return result


def load_exact_191_helpers(emitter_path: str | Path, expected_sha256: str,
                           pose_map: dict[int, dict[str, Any]]) -> tuple[Callable, Callable]:
    require_sha256(emitter_path, expected_sha256, "exact 191 emitter")
    source = Path(emitter_path).read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(emitter_path))
    except SyntaxError as exc:
        raise ReferenceError(f"pinned 191 emitter is not valid Python: {exc}") from exc
    wanted = {"rotate", "bodyworld"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise ReferenceError("pinned 191 emitter must define rotate and bodyworld")
    helper_module = ast.Module(body=nodes, type_ignores=[])
    namespace: dict[str, Any] = {"np": np, "poses": pose_map}
    exec(compile(helper_module, str(emitter_path), "exec"), namespace)
    return namespace["rotate"], namespace["bodyworld"]


def emit_current(payload: dict[str, Any], record: np.ndarray,
                 pose_map: dict[int, dict[str, Any]], rotate: Callable,
                 bodyworld: Callable) -> tuple[np.ndarray, np.ndarray]:
    fb, bc, fv, vc, _fi, _ic, stable_id, _layer = map(int, record)
    bind = payload["bindings"][fb:fb + bc]
    vertices = payload["vertices"][fv:fv + vc]
    active = vertices["w"] > np.float32(2e-6)
    if np.any(vertices["binding"][active] >= bc):
        raise ReferenceError(f"active binding index out of range in stable surface {stable_id}")
    p = np.zeros((vc, 3), dtype=np.float32)
    n = np.zeros((vc, 3), dtype=np.float32)
    for slot in range(4):
        enabled = active[:, slot]
        for local_binding in np.unique(vertices["binding"][enabled, slot]):
            local_binding = int(local_binding)
            rows = np.flatnonzero(enabled & (vertices["binding"][:, slot] == local_binding))
            binding = bind[local_binding]
            body = int(binding["body"])
            if body not in pose_map:
                raise ReferenceError(f"missing current pose for body {body}")
            transform = binding["transform"]
            local = (transform[:3] + rotate(transform[3:7], vertices["p"][rows]) * transform[7]).astype(np.float32)
            world = bodyworld(body, local)
            p[rows] = (p[rows] + vertices["w"][rows, slot, None] * world).astype(np.float32)
            current_q = pose_map[body]["orientation_world_xyzw"]
            nw = rotate(current_q, rotate(transform[3:7], vertices["n"][rows]))
            n[rows] = (n[rows] + vertices["w"][rows, slot, None] * nw).astype(np.float32)
    lengths = np.sqrt(np.sum(n * n, axis=1, dtype=np.float32), dtype=np.float32)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= np.float32(1e-6)):
        raise ReferenceError(f"degenerate blended normal in stable surface {stable_id}")
    return p, (n / lengths[:, None]).astype(np.float32)


def _inverse_current_transform(body: int, pose_map: dict[int, dict[str, Any]], rotate: Callable):
    state = pose_map.get(int(body))
    if state is None:
        raise ReferenceError(f"body {body} absent from pinned current pose")
    q = np.asarray(state["orientation_world_xyzw"], dtype=np.float32)
    qnorm = float(np.linalg.norm(q.astype(np.float64)))
    if not np.isfinite(qnorm) or qnorm <= 0 or abs(qnorm - 1.0) > 2e-5:
        raise ReferenceError(f"body {body} has zero or invalid quaternion norm {qnorm}")
    q = (q / np.float32(qnorm)).astype(np.float32)
    qinv = np.asarray([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)
    pos = np.asarray(state["position_world_m"], dtype=np.float32)
    return pos, q, qinv


def validate_inventory(inventory: dict[str, Any], base_sha: str,
                       pose_sha: str, cache_sha: str, record_count: int) -> dict[int, dict[str, Any]]:
    if str(inventory.get("pose_sha256", "")).lower() != pose_sha.lower():
        raise ReferenceError("inventory pose hash does not match pinned pose")
    if str(inventory.get("cache_sha256", "")).lower() != cache_sha.lower():
        raise ReferenceError("inventory cache hash does not match pinned cache")
    source_rows = [row for row in inventory.get("inputs", [])
                   if row.get("kind") == "soft_tissue" and str(row.get("sha256", "")).lower() == base_sha.lower()]
    if len(source_rows) != 1 or int(source_rows[0].get("record_count", -1)) != record_count:
        raise ReferenceError("inventory does not uniquely pin the baseline NHTISS source")
    result: dict[int, dict[str, Any]] = {}
    for bank in inventory.get("banks", []):
        if bank.get("kind") != "soft_tissue":
            continue
        sid = int(bank["stable_id"])
        if sid in result:
            raise ReferenceError(f"duplicate soft-tissue inventory stable ID {sid}")
        result[sid] = bank
    return result


def _validate_selected_ranges(source: dict[str, Any], candidate: dict[str, Any],
                              selected_ids: list[int]) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    if (source["magic"], source["abi"], source["nr"], source["nb"], source["nv"], source["ni"],
        source["registration"], source["source_id"]) != (
        candidate["magic"], candidate["abi"], candidate["nr"], candidate["nb"], candidate["nv"], candidate["ni"],
        candidate["registration"], candidate["source_id"]):
        raise ReferenceError("baseline and weight candidate NHTISS headers differ")
    if not np.array_equal(source["records"], candidate["records"]):
        raise ReferenceError("baseline and weight candidate record identities/ranges differ")
    if not np.array_equal(source["indices"], candidate["indices"]):
        raise ReferenceError("baseline and weight candidate surface topology differs")
    if not np.array_equal(source["bindings"]["body"], candidate["bindings"]["body"]):
        raise ReferenceError("baseline and weight candidate binding body identities differ")
    if not np.array_equal(source["bindings"]["transform"], candidate["bindings"]["transform"]):
        raise ReferenceError("weight candidate unexpectedly changed bind transforms")
    if not np.array_equal(source["vertices"]["p"], candidate["vertices"]["p"]):
        raise ReferenceError("weight candidate unexpectedly changed reference positions")
    if not np.array_equal(source["vertices"]["n"], candidate["vertices"]["n"]):
        raise ReferenceError("weight candidate unexpectedly changed reference normals")
    ids = [int(x) for x in selected_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ReferenceError("stable IDs must be a nonempty unique list")
    record_by_id = {int(row[6]): row.copy() for row in candidate["records"]}
    if any(sid not in record_by_id for sid in ids):
        missing = sorted(set(ids) - set(record_by_id))
        raise ReferenceError(f"selected stable surface IDs missing from payload: {missing}")
    selected_ranges: dict[int, np.ndarray] = {}
    selected_vertices = np.zeros(candidate["nv"], dtype=bool)
    selected_bindings = np.zeros(candidate["nb"], dtype=bool)
    for sid in ids:
        row = record_by_id[sid]
        fb, bc, fv, vc = map(int, row[:4])
        selected_ranges[sid] = row
        if np.any(selected_vertices[fv:fv + vc]) or np.any(selected_bindings[fb:fb + bc]):
            raise ReferenceError(f"selected records alias each other at stable surface {sid}")
        selected_vertices[fv:fv + vc] = True
        selected_bindings[fb:fb + bc] = True
        active = candidate["vertices"][fv:fv + vc]["w"] > np.float32(2e-6)
        if np.any(candidate["vertices"][fv:fv + vc]["binding"][active] >= bc):
            raise ReferenceError(f"selected surface {sid} has invalid active local binding indices")
    # No selected storage region may alias an unselected record. This protects
    # byte preservation when the selected vertices/bindings are rewritten.
    for row in candidate["records"]:
        fb, bc, fv, vc, _fi, _ic, sid, _layer = map(int, row)
        if sid in ids:
            continue
        if np.any(selected_bindings[fb:fb + bc]) or np.any(selected_vertices[fv:fv + vc]):
            raise ReferenceError(f"selected storage aliases unselected stable surface {sid}")
    return selected_ranges, selected_vertices, selected_bindings


def construct_candidate(baseline_raw: bytes, weight_raw: bytes, *, stable_ids: list[int],
                        pose_data: dict[str, Any], inventory: dict[str, Any], cache: Any,
                        baseline_sha256: str, pose_sha256: str, inventory_sha256: str,
                        cache_sha256: str, emitter_sha256: str,
                        reference_label: str, rotate: Callable, bodyworld: Callable,
                        position_tolerance_m: float = 1e-6,
                        normal_tolerance_rad: float = 1e-3) -> tuple[bytes, dict[str, Any]]:
    label = str(reference_label).strip()
    if not label or "inferred" not in label.lower():
        raise ReferenceError("reference label is required and must explicitly say inferred")
    if not np.isfinite(position_tolerance_m) or position_tolerance_m <= 0:
        raise ReferenceError("position tolerance must be finite and positive")
    if not np.isfinite(normal_tolerance_rad) or normal_tolerance_rad <= 0:
        raise ReferenceError("normal tolerance must be finite and positive")
    source = decode_nhtiss(baseline_raw)
    candidate = decode_nhtiss(weight_raw)
    selected, selected_vertices, selected_bindings = _validate_selected_ranges(source, candidate, stable_ids)
    if int(pose_data.get("registration_fingerprint32", -1)) != source["registration"]:
        raise ReferenceError("pose registration fingerprint does not match baseline NHTISS")
    pose_map = load_pose_map(pose_data, source["registration"])
    inventory_banks = validate_inventory(inventory, baseline_sha256, pose_sha256,
                                          cache_sha256, source["nr"])
    for sid in selected:
        if sid not in inventory_banks:
            raise ReferenceError(f"pinned inventory lacks soft-tissue stable surface {sid}")
    if cache is None:
        raise ReferenceError("current-pose cache is required")

    out_raw = bytearray(weight_raw)
    out = decode_nhtiss(bytes(out_raw))
    metrics: dict[str, Any] = {}
    for sid, record in selected.items():
        fb, bc, fv, vc, fi, ic, _stable, _layer = map(int, record)
        target_p, target_n = emit_current(source, source["records"][np.flatnonzero(source["records"][:, 6] == sid)[0]],
                                          pose_map, rotate, bodyworld)
        bank = inventory_banks[sid]
        vertex_key = str(bank["key"]) + "_vertices"
        face_key = str(bank["key"]) + "_faces"
        if vertex_key not in cache or face_key not in cache:
            raise ReferenceError(f"cache lacks exact current surface arrays for stable ID {sid}")
        cached_p = np.asarray(cache[vertex_key])
        cached_faces = np.asarray(cache[face_key])
        if cached_p.dtype != np.float32 or cached_p.shape != (vc, 3):
            raise ReferenceError(f"cache vertex shape/dtype mismatch for stable ID {sid}")
        source_faces = source["indices"][fi:fi + ic].reshape(-1, 3).astype(np.int64) - fv
        if cached_faces.dtype != np.uint32 or cached_faces.shape != source_faces.shape or not np.array_equal(cached_faces, source_faces):
            raise ReferenceError(f"cache face topology mismatch for stable ID {sid}")
        if int(bank.get("vertices", -1)) != vc or int(bank.get("triangles", -1)) != ic // 3:
            raise ReferenceError(f"inventory dimensions mismatch for stable ID {sid}")
        if not np.array_equal(target_p, cached_p):
            delta = float(np.linalg.norm((target_p - cached_p).astype(np.float64), axis=1).max())
            raise ReferenceError(f"{sid}: baseline current cache is not exact191 parity ({delta} m)")

        # One shared inferred coordinate frame per surface; origin is persisted
        # into each inverse bind so a rotation cannot erase the centering shift.
        origin = target_p.astype(np.float64).mean(axis=0).astype(np.float32)
        out["vertices"]["p"][fv:fv + vc] = ((target_p.astype(np.float64) - origin.astype(np.float64)) * 1000.0).astype(np.float32)
        out["vertices"]["n"][fv:fv + vc] = target_n.copy()
        body_rows = []
        for bi in range(fb, fb + bc):
            body = int(out["bindings"][bi]["body"])
            state = pose_map.get(body)
            if state is None:
                raise ReferenceError(f"missing current pose for selected binding body {body}")
            pos, q, qinv = _inverse_current_transform(body, pose_map, rotate)
            bind_t = rotate(qinv, (origin - pos).astype(np.float32)).astype(np.float32)
            out["bindings"][bi]["transform"][:] = np.asarray([*bind_t, *qinv, 0.001], dtype=np.float32)
            body_rows.append((bi, body))

        pos_iterations = 0
        normal_iterations = 0
        for iteration in range(8):
            got_p, _ = emit_current(out, record, pose_map, rotate, bodyworld)
            delta = target_p.astype(np.float64) - got_p.astype(np.float64)
            if float(np.linalg.norm(delta, axis=1).max()) <= 2e-8:
                break
            out["vertices"]["p"][fv:fv + vc] = (
                out["vertices"]["p"][fv:fv + vc] + (delta * 1000.0).astype(np.float32)
            ).astype(np.float32)
            pos_iterations = iteration + 1
        for iteration in range(4):
            _, got_n = emit_current(out, record, pose_map, rotate, bodyworld)
            delta_n = target_n.astype(np.float64) - got_n.astype(np.float64)
            if float(np.linalg.norm(delta_n, axis=1).max()) <= 1e-7:
                break
            normal = out["vertices"]["n"][fv:fv + vc].astype(np.float64) + delta_n
            normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-30)
            out["vertices"]["n"][fv:fv + vc] = normal.astype(np.float32)
            normal_iterations = iteration + 1

        final_p, final_n = emit_current(out, record, pose_map, rotate, bodyworld)
        position_delta = np.linalg.norm((final_p - target_p).astype(np.float64), axis=1)
        normal_dot = np.sum(final_n.astype(np.float64) * target_n.astype(np.float64), axis=1)
        normal_angle = np.arccos(np.clip(normal_dot, -1.0, 1.0))
        if not np.isfinite(position_delta).all() or float(position_delta.max()) > position_tolerance_m:
            raise ReferenceError(f"stable surface {sid} current position error exceeds tolerance")
        if not np.isfinite(normal_angle).all() or float(normal_angle.max()) > normal_tolerance_rad:
            raise ReferenceError(f"stable surface {sid} current normal error exceeds tolerance")

        # Verify the common map independently for every bound body, not only the
        # route-weighted blend. This is the coherent inverse-bind invariant.
        bind_position_spread = 0.0
        bind_normal_spread = 0.0
        for bi, body in body_rows:
            transform = out["bindings"][bi]["transform"]
            local = (transform[:3] + rotate(transform[3:7], out["vertices"]["p"][fv:fv + vc]) * transform[7]).astype(np.float32)
            one_position = bodyworld(body, local)
            bind_position_spread = max(bind_position_spread, float(np.linalg.norm((one_position - final_p).astype(np.float64), axis=1).max()))
            current_q = pose_map[body]["orientation_world_xyzw"]
            one_normal = rotate(current_q, rotate(transform[3:7], out["vertices"]["n"][fv:fv + vc]))
            one_length = np.sqrt(np.sum(one_normal * one_normal, axis=1, dtype=np.float32), dtype=np.float32)
            if np.any(one_length <= 1e-6) or not np.isfinite(one_length).all():
                raise ReferenceError(f"degenerate per-binding normal for stable surface {sid}")
            one_normal = (one_normal / one_length[:, None]).astype(np.float32)
            angle = np.arccos(np.clip(np.sum(one_normal.astype(np.float64) * final_n.astype(np.float64), axis=1), -1.0, 1.0))
            bind_normal_spread = max(bind_normal_spread, float(angle.max()))
        if bind_position_spread > position_tolerance_m or bind_normal_spread > normal_tolerance_rad:
            raise ReferenceError(f"stable surface {sid} per-binding current spread exceeds tolerance")
        metrics[str(sid)] = {
            "vertex_count": vc, "face_count": ic // 3,
            "inferred_reference_origin_world_m": [float(x) for x in origin],
            "reference_position_abs_max_mm": float(np.abs(out["vertices"]["p"][fv:fv + vc]).max()),
            "current_position_delta_max_m": float(position_delta.max()),
            "current_normal_delta_max_rad": float(normal_angle.max()),
            "per_binding_position_spread_max_m": bind_position_spread,
            "per_binding_normal_spread_max_rad": bind_normal_spread,
            "position_residual_iterations": pos_iterations,
            "normal_residual_iterations": normal_iterations,
            "target_current_sha256": sha256_bytes(target_p.tobytes()),
            "emitted_current_sha256": sha256_bytes(final_p.tobytes()),
        }

    # Serialize only the selected records' transforms and p/n fields into the
    # existing NHTISS ABI. Every weight, owner lane, index, and other byte stays.
    for sid, record in selected.items():
        fb, bc, fv, vc = map(int, record[:4])
        for bi in range(fb, fb + bc):
            struct.pack_into("<8f", out_raw, candidate["binding_offset"] + bi * 36 + 4,
                             *map(float, out["bindings"][bi]["transform"]))
        for vertex in range(fv, fv + vc):
            struct.pack_into("<6f", out_raw, candidate["vertex_offset"] + vertex * 56,
                             *map(float, out["vertices"]["p"][vertex]),
                             *map(float, out["vertices"]["n"][vertex]))

    result_raw = bytes(out_raw)
    readback = decode_nhtiss(result_raw)
    if (readback["magic"], readback["abi"], readback["nr"], readback["nb"], readback["nv"], readback["ni"],
        readback["registration"], readback["source_id"]) != (
        candidate["magic"], candidate["abi"], candidate["nr"], candidate["nb"], candidate["nv"], candidate["ni"],
        candidate["registration"], candidate["source_id"]):
        raise ReferenceError("serialized candidate header readback mismatch")
    if not np.array_equal(readback["records"], candidate["records"]) or not np.array_equal(readback["indices"], candidate["indices"]):
        raise ReferenceError("serialized candidate changed stable IDs or topology")
    if not np.array_equal(readback["vertices"]["w"], candidate["vertices"]["w"]):
        raise ReferenceError("serialized candidate changed route weights")
    if not np.array_equal(readback["vertices"]["binding"], candidate["vertices"]["binding"]):
        raise ReferenceError("serialized candidate changed vertex owner indices")
    if not np.array_equal(readback["bindings"]["body"], candidate["bindings"]["body"]):
        raise ReferenceError("serialized candidate changed binding body identities")
    if not np.array_equal(readback["vertices"]["p"][~selected_vertices], candidate["vertices"]["p"][~selected_vertices]):
        raise ReferenceError("serialized candidate changed unselected vertex positions")
    if not np.array_equal(readback["vertices"]["n"][~selected_vertices], candidate["vertices"]["n"][~selected_vertices]):
        raise ReferenceError("serialized candidate changed unselected vertex normals")
    if not np.array_equal(readback["bindings"]["transform"][~selected_bindings], candidate["bindings"]["transform"][~selected_bindings]):
        raise ReferenceError("serialized candidate changed unselected binding transforms")
    # Re-run exact191 on the serialized bytes, not just the in-memory candidate.
    for sid, record in selected.items():
        got_p, _ = emit_current(readback, record, pose_map, rotate, bodyworld)
        bank = inventory_banks[sid]
        cache_p = np.asarray(cache[str(bank["key"]) + "_vertices"], dtype=np.float32)
        if not np.array_equal(got_p, cache_p):
            delta = float(np.linalg.norm((got_p - cache_p).astype(np.float64), axis=1).max())
            if delta > position_tolerance_m:
                raise ReferenceError(f"serialized stable surface {sid} diverges from exact191 current cache: {delta} m")
            metrics[str(sid)]["serialized_vs_exact191_max_m"] = delta
        else:
            metrics[str(sid)]["serialized_vs_exact191_max_m"] = 0.0

    report = {
        "schema": "numi.human.nhtiss.coherent-inverse-current-reference-candidate.v1",
        "inputs": {
            "baseline_nhtiss_sha256": baseline_sha256,
            "weight_candidate_nhtiss_sha256": sha256_bytes(weight_raw),
            "pose_snapshot_sha256": pose_sha256,
            "inventory_sha256": inventory_sha256,
            "current_pose_cache_sha256": cache_sha256,
            "exact_191_emitter_sha256": emitter_sha256,
        },
        "scope": {
            "stable_surface_ids": [int(x) for x in stable_ids],
            "existing_abi": "NHTISS4 ABI 5",
            "preserved": "all topology, stable record identities, route weights, owner lanes, unselected positions/normals, and unselected binding transforms",
            "rewritten": "selected surfaces common inferred reference positions/normals and selected binding inverse-current transforms",
        },
        "method": {
            "reference_coordinates": "inferred from exact 191 current-pose registered-surface emission; per-record p_mm = 1000 * (target_world_m - origin_world_m)",
            "binding_translation": "R_current_inverse * (origin_world - body_position_world), including the centering origin",
            "binding_rotation_and_scale": "inverse current quaternion with uniform 0.001 scale so every selected binding maps the common reference point to one registered current point",
            "emitter": "rotate/bodyworld AST-loaded from the explicitly pinned 191 source file",
        },
        "per_surface": metrics,
        "gate": {
            "baseline_cache_replay": "bitwise equal for selected exact191 vertex arrays and face arrays",
            "serialized_candidate_replay": "exact191 current replay within declared position tolerance",
            "physical_admission": False,
            "status": "candidate-only; requires independent exact contact, pose, and owner review",
        },
        "provenance": {
            "reference_kind": "inferred",
            "inferred_reference_label": label,
            "measured_original_source_coordinates": False,
            "physical_asset_promoted": False,
        },
    }
    return result_raw, report


def _load_cli_inputs(args: argparse.Namespace):
    paths = {
        "baseline": args.baseline, "weight candidate": args.weights,
        "pose": args.pose, "inventory": args.inventory,
        "current-pose cache": args.cache, "exact 191 emitter": args.emitter,
    }
    pins = {
        "baseline": args.baseline_sha256, "weight candidate": args.weights_sha256,
        "pose": args.pose_sha256, "inventory": args.inventory_sha256,
        "current-pose cache": args.cache_sha256, "exact 191 emitter": args.emitter_sha256,
    }
    actual = {name: require_sha256(paths[name], pins[name], name) for name in paths}
    baseline_raw = Path(args.baseline).read_bytes()
    weight_raw = Path(args.weights).read_bytes()
    pose_data = json.loads(Path(args.pose).read_text(encoding="utf-8"))
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    cache = np.load(args.cache, allow_pickle=False)
    pose_map = load_pose_map(pose_data, decode_nhtiss(baseline_raw)["registration"])
    rotate, bodyworld = load_exact_191_helpers(args.emitter, actual["exact 191 emitter"], pose_map)
    return baseline_raw, weight_raw, pose_data, inventory, cache, actual, rotate, bodyworld


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="pinned baseline NHTISS ABI-5 payload")
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--weights", required=True, help="same-structure NHTISS payload with proposed weights")
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--pose", required=True, help="pinned registered all_bodies current/rest pose JSON")
    parser.add_argument("--pose-sha256", required=True)
    parser.add_argument("--inventory", required=True, help="pinned exact191 source/cache inventory JSON")
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--cache", required=True, help="pinned exact191 current-pose NPZ")
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--emitter", required=True, help="pinned exact191 emitter source")
    parser.add_argument("--emitter-sha256", required=True)
    parser.add_argument("--stable-id", type=int, action="append", required=True,
                        help="selected stable surface ID; repeat for each surface")
    parser.add_argument("--inferred-reference-label", required=True,
                        help="mandatory provenance text explicitly identifying inferred coordinates")
    parser.add_argument("--output", required=True, help="new NHTISS output path; existing files are never overwritten")
    parser.add_argument("--report", required=True, help="new JSON receipt path; existing files are never overwritten")
    parser.add_argument("--position-tolerance-m", type=float, default=1e-6)
    parser.add_argument("--normal-tolerance-rad", type=float, default=1e-3)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    output = Path(args.output)
    report_path = Path(args.report)
    if output.resolve() == report_path.resolve():
        parser.error("--output and --report must be different paths")
    if output.exists() or report_path.exists():
        parser.error("refusing to overwrite an existing output or report")
    temp_paths: list[Path] = []
    try:
        base, weights, pose, inventory, cache, actual, rotate, bodyworld = _load_cli_inputs(args)
        result, report = construct_candidate(
            base, weights, stable_ids=args.stable_id, pose_data=pose, inventory=inventory,
            cache=cache, baseline_sha256=actual["baseline"], pose_sha256=actual["pose"],
            inventory_sha256=actual["inventory"], cache_sha256=actual["current-pose cache"],
            emitter_sha256=actual["exact 191 emitter"], reference_label=args.inferred_reference_label,
            rotate=rotate, bodyworld=bodyworld, position_tolerance_m=args.position_tolerance_m,
            normal_tolerance_rad=args.normal_tolerance_rad)
        output_parent = output.parent
        report_parent = report_path.parent
        output_parent.mkdir(parents=True, exist_ok=True)
        report_parent.mkdir(parents=True, exist_ok=True)
        if output.exists() or report_path.exists():
            raise ReferenceError("output or report appeared during validation; refusing to overwrite")
        report["output"] = {"path": str(output), "sha256": sha256_bytes(result), "bytes": len(result)}
        report_raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
        # Stage both validated files in their destination directories. The final
        # NHTISS path is not created until all pins/replay/readback checks passed.
        with tempfile.NamedTemporaryFile(prefix=".nhtiss-candidate-", dir=output_parent, delete=False) as stream:
            stream.write(result); stream.flush(); os.fsync(stream.fileno()); out_tmp = Path(stream.name)
        temp_paths.append(out_tmp)
        with tempfile.NamedTemporaryFile(prefix=".nhtiss-report-", dir=report_parent, delete=False) as stream:
            stream.write(report_raw); stream.flush(); os.fsync(stream.fileno()); report_tmp = Path(stream.name)
        temp_paths.append(report_tmp)
        if output.exists() or report_path.exists():
            raise ReferenceError("output or report appeared before publish; refusing to overwrite")
        os.replace(report_tmp, report_path); temp_paths.remove(report_tmp)
        os.replace(out_tmp, output); temp_paths.remove(out_tmp)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ReferenceError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        for path in temp_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        print(f"rebind_human_tissue_reference: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
