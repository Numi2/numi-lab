#!/usr/bin/env python3
"""Candidate-parameterized exact O-I/I-D route audit.

This owner emits the established route-report schema. It never derives or emits
construction planes. Optional matched-query inputs are accepted only when their
world nodes, body poses, and local query rows match the candidate and pinned 039
bank/body-frame export.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import pathlib
import runpy
import time
from typing import Any

import numpy as np

PINNED_039 = {
    "exact-emitted.triangles.csv": "0d3418064af1d4f34d19a1ec91992afceacaaa70b6c982d4a2637ded1d829eac",
    "exact-emitted.nodes.csv": "99e58d6afb68e0939670f69d4132bfa730cf3e988a7f70a179a41a89eac3802c",
    "exact-emitted.bodies.csv": "dae52ab190e02586421b56381b03a4453aff1058edca7b6a42afcb9051b50b90",
}
PINNED_SEGMENT_HELPER = "d6a2efe451e7743b659d74c274d1372a5ed2a765511a8671b2c1a2370b7d03a2"
EXPECTED_NODES = 7233
ROUTE_ROWS = 2411


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_pin(path: pathlib.Path, expected: str, what: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {what}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{what} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def load_candidate_points(candidate_npz: pathlib.Path, expected_sha: str,
                          node_ids_npz: pathlib.Path | None,
                          node_ids_sha: str | None) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    pins = {"candidate_npz": sha256(candidate_npz)}
    id_source = None
    candidate_tets = None
    candidate_materials = None
    if pins["candidate_npz"] != expected_sha:
        raise ValueError("candidate NPZ SHA-256 does not match explicit pin")
    with np.load(candidate_npz, allow_pickle=False) as data:
        if "points_world_f32" in data.files:
            points = np.asarray(data["points_world_f32"], dtype=np.float32)
        elif "points" in data.files:
            points = np.asarray(data["points"], dtype=np.float32)
        else:
            raise ValueError("candidate NPZ must contain points_world_f32 or points")
        candidate_tets = np.asarray(data["tetrahedra"], dtype=np.int64) if "tetrahedra" in data.files else None
        candidate_materials = np.asarray(data["material_ids"], dtype=np.int64) if "material_ids" in data.files else None
        if "global_node_ids" in data.files:
            gids = np.asarray(data["global_node_ids"], dtype=np.int64)
            id_source = "candidate_npz:global_node_ids"
        elif "global_nodes" in data.files:
            gids = np.asarray(data["global_nodes"], dtype=np.int64)
            id_source = "candidate_npz:global_nodes"
        else:
            gids = None
    if gids is None:
        if node_ids_npz is None or node_ids_sha is None:
            raise ValueError("candidate lacks IDs; an explicitly pinned node-ID NPZ is required")
        pins["node_ids_npz"] = require_pin(node_ids_npz, node_ids_sha, "node-ID NPZ")
        with np.load(node_ids_npz, allow_pickle=False) as ids_data:
            key = "global_node_ids" if "global_node_ids" in ids_data.files else "global_nodes"
            if key not in ids_data.files:
                raise ValueError("node-ID NPZ lacks global_node_ids/global_nodes")
            gids = np.asarray(ids_data[key], dtype=np.int64)
            if candidate_tets is None or candidate_materials is None:
                raise ValueError("external node IDs require candidate tetrahedra and material IDs to prove row identity")
            if "tetrahedra" not in ids_data.files or "material_ids" not in ids_data.files:
                raise ValueError("node-ID source lacks topology/material arrays needed to prove row identity")
            source_tets = np.asarray(ids_data["tetrahedra"], dtype=np.int64)
            source_materials = np.asarray(ids_data["material_ids"], dtype=np.int64)
            if not np.array_equal(candidate_tets, source_tets):
                raise ValueError("candidate tetrahedra differ from node-ID source; row mapping is unproven")
            if not np.array_equal(candidate_materials, source_materials):
                raise ValueError("candidate material rows differ from node-ID source; row mapping is unproven")
            id_source = f"node_ids_npz:{key}; row identity verified by exact tetrahedra/material arrays"
    elif node_ids_npz is not None or node_ids_sha is not None:
        raise ValueError("do not supply a second ID source when candidate already contains IDs")
    if points.shape != (EXPECTED_NODES, 3) or gids.shape != (EXPECTED_NODES,):
        raise ValueError(f"unexpected candidate shapes: points={points.shape}, ids={gids.shape}")
    if len(np.unique(gids)) != EXPECTED_NODES or not np.isfinite(points).all():
        raise ValueError("candidate points/IDs must be finite and IDs unique")
    pins["candidate_id_source"] = id_source
    return points, gids, pins


def validate_node_query_parity(candidate_ids: np.ndarray, candidate_points: np.ndarray,
                               query_rows: list[dict[str, str]]) -> None:
    if len(query_rows) != len(candidate_ids):
        raise ValueError("matched query node row count differs from candidate")
    qids = np.asarray([int(r["global_node"]) for r in query_rows], dtype=np.int64)
    if not np.array_equal(qids, candidate_ids):
        raise ValueError("matched query global-node sequence differs from candidate")
    qpoints = np.asarray([[float(r[k]) for k in ("x_m", "y_m", "z_m")] for r in query_rows], dtype=np.float32)
    if not np.array_equal(qpoints.view(np.uint32), candidate_points.view(np.uint32)):
        raise ValueError("matched query world positions differ from candidate Float32 positions")
    for i, r in enumerate(query_rows):
        if int(r["layer"]) != i // ROUTE_ROWS or int(r["row"]) != i % ROUTE_ROWS:
            raise ValueError("matched query layer/row order differs from candidate route layout")


def body_pose_map(pose_path: pathlib.Path) -> dict[int, dict[str, Any]]:
    data = json.loads(pose_path.read_text())
    if data.get("schema") != "numi.human.native-skin-pose-snapshot.v1":
        raise ValueError("matched query pose snapshot has unexpected schema")
    records = data.get("all_bodies", data.get("bodies", []))
    out: dict[int, dict[str, Any]] = {}
    for rec in records:
        bi = int(rec["body_index"])
        if bi in out:
            raise ValueError(f"duplicate body pose {bi}")
        out[bi] = rec
    return out


def validate_body_pose_parity(bank_bodies: dict[int, dict[str, str]],
                              query_poses: dict[int, dict[str, Any]],
                              used_bodies: np.ndarray) -> None:
    for bi0 in np.unique(used_bodies):
        bi = int(bi0)
        if bi not in bank_bodies or bi not in query_poses:
            raise ValueError(f"matched query missing required body pose {bi}")
        bank = bank_bodies[bi]
        current = query_poses[bi].get("current")
        if not isinstance(current, dict):
            raise ValueError(f"matched query body {bi} lacks current pose")
        qpos = np.asarray(current["position_world_m"], dtype=np.float32)
        qquat = np.asarray(current["orientation_world_xyzw"], dtype=np.float32)
        bpos = np.asarray([bank[f"position_{c}_m"] for c in "xyz"], dtype=np.float32)
        bquat = np.asarray([bank[f"orientation_q{c}"] for c in "xyzw"], dtype=np.float32)
        if not np.array_equal(qpos.view(np.uint32), bpos.view(np.uint32)):
            raise ValueError(f"matched query body {bi} position frame differs from pinned 039")
        if not np.array_equal(qquat.view(np.uint32), bquat.view(np.uint32)):
            raise ValueError(f"matched query body {bi} orientation frame differs from pinned 039")


def read_bank_inputs(bank_dir: pathlib.Path, evidence_root: pathlib.Path):
    paths = {name: bank_dir / name for name in PINNED_039}
    pins = {name: require_pin(path, PINNED_039[name], f"039 {name}") for name, path in paths.items()}
    tr_path = paths["exact-emitted.triangles.csv"]
    nodes_path = paths["exact-emitted.nodes.csv"]
    bodies_path = paths["exact-emitted.bodies.csv"]
    raw = np.loadtxt(tr_path, delimiter=",", skiprows=1)
    if raw.shape != (469136, 23):
        raise ValueError(f"039 triangle row shape changed: {raw.shape}")
    tri_f32 = raw[:, 5:14].astype(np.float32)
    if not np.array_equal(tri_f32.view(np.uint32), raw[:, 14:23].astype(np.uint32)):
        raise ValueError("039 exact emitted Float32 triangle replay mismatch")
    triangles = tri_f32.astype(np.float64).reshape(-1, 3, 3)
    ids = raw[:, 1:5].astype(np.int64)
    if not np.array_equal(ids.astype(float), raw[:, 1:5]):
        raise ValueError("039 stable triangle identity columns are not exact integers")
    nodes = np.loadtxt(nodes_path, delimiter=",", skiprows=1)
    if nodes.shape != (EXPECTED_NODES, 13):
        raise ValueError(f"039 node-row shape changed: {nodes.shape}")
    if not np.array_equal(nodes[:, 5:8].astype(np.float32).view(np.uint32),
                          nodes[:, 9:12].astype(np.uint32)):
        raise ValueError("039 node position Float32 bit columns mismatch")
    if not np.array_equal(nodes[:, 0], np.arange(EXPECTED_NODES)):
        raise ValueError("039 local-node row order changed")
    gids = nodes[:, 1].astype(np.int64)
    if not np.array_equal(gids.astype(float), nodes[:, 1]):
        raise ValueError("039 global node IDs are not exact integers")
    for layer in range(3):
        block = nodes[layer * ROUTE_ROWS:(layer + 1) * ROUTE_ROWS]
        if not np.array_equal(block[:, 2], np.full(ROUTE_ROWS, layer)):
            raise ValueError("039 layer ordering changed")
        if not np.array_equal(block[:, 3], np.arange(ROUTE_ROWS)):
            raise ValueError("039 route row ordering changed")
    bodies = {int(r["body_index"]): r for r in csv.DictReader(bodies_path.open(newline=""))}
    matrices: dict[int, np.ndarray] = {}
    offsets: dict[int, np.ndarray] = {}
    positions: dict[int, np.ndarray] = {}
    locations: dict[int, np.ndarray] = {}
    for bi in np.unique(ids[:, 0]):
        body = int(bi)
        rec = bodies[body]
        position = np.asarray([rec[f"position_{c}_m"] for c in "xyz"], dtype=np.float32).astype(np.float64)
        quat = np.asarray([rec[f"orientation_q{c}"] for c in "xyzw"], dtype=np.float32).astype(np.float64)
        quat[:3] *= -1.0
        matrix = np.asarray([float(rec[f"world_to_body_m{i}{j}"])
                             for i in range(3) for j in range(3)]).reshape(3, 3)
        def rotate(v):
            return v + 2.0 * np.cross(quat[:3], np.cross(quat[:3], v) + quat[3] * v)
        expected = np.column_stack([rotate(v) for v in np.eye(3)])
        if np.max(np.abs(matrix - expected)) >= 1e-12:
            raise ValueError(f"039 body {body} matrix/quaternion parity failure")
        offset = np.asarray([rec[f"com_offset_{c}_m"] for c in "xyz"], dtype=np.float64)
        matrices[body] = matrix
        offsets[body] = offset
        positions[body] = position
    bankkeys = np.unique(ids[:, :2], axis=0)
    if len(bankkeys) != 185:
        raise ValueError(f"expected 185 typed banks, found {len(bankkeys)}")
    inventory_path = bank_dir / "all-bank-containment.csv.banks.csv"
    inventory_pin = sha256(inventory_path)
    inventory = {(int(r["body"]), int(r["stable_bone_id"])): int(r["triangle_count"])
                 for r in csv.DictReader(inventory_path.open(newline=""))}
    counts = {}
    for body, bone in bankkeys:
        key = (int(body), int(bone))
        counts[key] = int(np.count_nonzero((ids[:, 0] == key[0]) & (ids[:, 1] == key[1])))
    if counts != inventory:
        raise ValueError("039 triangle rows do not match typed-bank inventory")
    pins["all-bank-containment.csv.banks.csv"] = inventory_pin
    exact_path = evidence_root / "root_exact_segment_witness.py"
    require_pin(exact_path, PINNED_SEGMENT_HELPER, "exact segment helper")
    exact = runpy.run_path(str(exact_path))
    return triangles, ids, gids, bodies, matrices, offsets, positions, bankkeys, pins, exact


def matched_query_node_path(query_dir: pathlib.Path, requested_name: str | None = None) -> pathlib.Path:
    """Select a pinned 023 node export; exact emitted points are the safe default."""
    allowed = {"exact-emitted.nodes.csv", "initial-skin-fem-nodes.csv"}
    if requested_name is not None:
        if requested_name not in allowed:
            raise ValueError("matched query node file must be exact-emitted.nodes.csv or initial-skin-fem-nodes.csv")
        return query_dir / requested_name
    exact_nodes = query_dir / "exact-emitted.nodes.csv"
    if exact_nodes.is_file():
        return exact_nodes
    return query_dir / "initial-skin-fem-nodes.csv"


def load_matched_query(query_dir: pathlib.Path, candidate_ids: np.ndarray,
                       candidate_points: np.ndarray, bank_bodies: dict[int, dict[str, str]],
                       matrices: dict[int, np.ndarray], offsets: dict[int, np.ndarray],
                       positions: dict[int, np.ndarray], bank_ids: np.ndarray,
                       node_file: str | None = None):
    node_path = matched_query_node_path(query_dir, node_file)
    query_path = query_dir / "all-bank-containment.csv"
    inventory_path = query_dir / "all-bank-containment.csv.banks.csv"
    poses_path = query_dir / "registered-skin-poses.json"
    for p in (node_path, query_path, inventory_path, poses_path):
        if not p.is_file():
            raise FileNotFoundError(f"matched query is missing {p.name}: {p}")
    node_rows = list(csv.DictReader(node_path.open(newline="")))
    validate_node_query_parity(candidate_ids, candidate_points, node_rows)
    pose_map = body_pose_map(poses_path)
    validate_body_pose_parity(bank_bodies, pose_map, bank_ids)
    query_rows = list(csv.DictReader(query_path.open(newline="")))
    qkeys = set()
    qmap = {}
    maxlocal = 0.0
    candidate_id_to_index = {int(g): i for i, g in enumerate(candidate_ids)}
    for r in query_rows:
        body = int(r["bank_body"])
        gid = int(r["global_node"])
        bone = int(r["bank_stable_id"])
        key = (gid, body, bone)
        if key in qkeys:
            raise ValueError(f"duplicate matched query row {key}")
        qkeys.add(key)
        if gid not in candidate_id_to_index:
            raise ValueError(f"matched query references unknown candidate node {gid}")
        row = candidate_id_to_index[gid]
        xyz = candidate_points[row].astype(np.float64)
        expected_local = (xyz.astype(np.float64) - positions[body]) @ matrices[body].T - offsets[body]
        actual_local = np.asarray([float(r[f"query_local_{c}"]) for c in "xyz"])
        local_error = float(np.linalg.norm(expected_local - actual_local))
        if local_error >= 1e-12:
            raise ValueError(f"matched query body-local point mismatch for {key}")
        maxlocal = max(maxlocal, local_error)
        qmap[key] = r["classification"]
    return qkeys, qmap, maxlocal, {
        node_path.name: sha256(node_path),
        "all-bank-containment.csv": sha256(query_path),
        "all-bank-containment.csv.banks.csv": sha256(inventory_path),
        "registered-skin-poses.json": sha256(poses_path),
    }


def unique_undirected_tet_edges(tetrahedra: np.ndarray) -> np.ndarray:
    """Return deterministic lexicographic unique local-node edges from tetrahedra."""
    tets = np.asarray(tetrahedra, dtype=np.int64)
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError("tetrahedra must have shape (n, 4)")
    if not len(tets):
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.concatenate([tets[:, pair] for pair in itertools.combinations(range(4), 2)], axis=0)
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0)


def exact_segment_triangle_hits(segment_start: np.ndarray, segment_end: np.ndarray,
                                bank_triangles: np.ndarray, stable_ids: np.ndarray,
                                exact_classifier) -> dict[str, Any]:
    """AABB and floating filter followed by exact classification in one bank."""
    a = np.asarray(segment_start, dtype=np.float64)
    b = np.asarray(segment_end, dtype=np.float64)
    tris = np.asarray(bank_triangles, dtype=np.float64)
    stable_ids = np.asarray(stable_ids, dtype=np.int64)
    if a.shape != (3,) or b.shape != (3,) or tris.ndim != 3 or tris.shape[1:] != (3, 3):
        raise ValueError("segment/triangle shapes are invalid")
    if stable_ids.shape != (len(tris),):
        raise ValueError("stable IDs must align with bank-local triangles")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(tris).all():
        raise ValueError("segment and bank triangles must be finite")
    seg_low, seg_high = np.minimum(a, b), np.maximum(a, b)
    tri_low, tri_high = tris.min(axis=1), tris.max(axis=1)
    aabb = np.flatnonzero(np.all(tri_high >= seg_low, axis=1) & np.all(tri_low <= seg_high, axis=1))
    if not len(aabb):
        return {"triangle_aabb_candidates": 0, "exact_candidates": 0, "hits": [], "uncertain": []}
    candidate = tris[aabb]
    d = b - a
    e1, e2 = candidate[:, 1] - candidate[:, 0], candidate[:, 2] - candidate[:, 0]
    pvec = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    scale = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(d)
    near = np.abs(det) <= 64 * np.finfo(float).eps * scale
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=det != 0)
    tvec = a - candidate[:, 0]
    u = inv * np.einsum("ij,ij->i", tvec, pvec)
    qvec = np.cross(tvec, e1)
    v = inv * (qvec @ d)
    t = inv * np.einsum("ij,ij->i", e2, qvec)
    possible = near | ((u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10) &
                       (t >= -1e-10) & (t <= 1 + 1e-10))
    hits, uncertain = [], []
    for ci in np.flatnonzero(possible):
        result = exact_classifier(a, b, candidate[ci])
        kind = result["classification"]
        record = {"bank_local_index": int(aabb[ci]),
                  "triangle_stable_id": int(stable_ids[aabb[ci]]),
                  "classification": kind, "exact_result": result}
        if kind in ("proper_interior_hit", "boundary_hit"):
            error = validate_hit_reconstruction(candidate[ci], np.asarray(result["barycentric"]),
                                                a, b, float(result["t"]))
            record["hit_reconstruction_error_m"] = error
            hits.append(record)
        elif kind in ("coplanar_requires_2d_test", "invalid_degenerate_triangle"):
            uncertain.append(record)
    return {"triangle_aabb_candidates": len(aabb), "exact_candidates": int(np.count_nonzero(possible)),
            "hits": hits, "uncertain": uncertain}


def audit_unique_tet_edges(global_node_ids: np.ndarray, tetrahedra: np.ndarray,
                           triangle_ids: np.ndarray, bank_triangles: np.ndarray,
                           body_local_points: dict[int, np.ndarray], exact_classifier,
                           *, node_identity_domain: str = "global_node") -> dict[str, Any]:
    """Screen unique FEM edges against typed bone banks, preserving ID semantics.

    global_node is the historical output contract. query_row is for point sets
    without compiled global IDs: the supplied identity vector must be exactly
    the contiguous array row sequence, and witnesses emit endpoint_query_rows
    instead of global-node/layer/column fields.
    """
    if node_identity_domain not in ("global_node", "query_row"):
        raise ValueError("node_identity_domain must be 'global_node' or 'query_row'")
    gids = np.asarray(global_node_ids, dtype=np.int64)
    if gids.ndim != 1:
        raise ValueError("node identity vector must be one-dimensional")
    if node_identity_domain == "query_row" and not np.array_equal(
            gids, np.arange(len(gids), dtype=np.int64)):
        raise ValueError("query_row identity requires contiguous zero-based array rows")
    tets = np.asarray(tetrahedra, dtype=np.int64)
    ids = np.asarray(triangle_ids, dtype=np.int64)
    tris = np.asarray(bank_triangles, dtype=np.float64)
    if tets.ndim != 2 or tets.shape[1] != 4 or tets.min(initial=0) < 0 or tets.max(initial=-1) >= len(gids):
        raise ValueError("candidate tetrahedra are invalid for supplied node rows")
    if ids.ndim != 2 or ids.shape[1] != 4 or tris.shape != (len(ids), 3, 3):
        raise ValueError("typed triangle arrays do not align")
    edges = unique_undirected_tet_edges(tets)
    counts = dict(unique_tetrahedral_edges=len(edges), typed_banks=0, broadphase_edge_bank_pairs=0,
                  triangle_aabb_pairs=0, exact_narrowphase_tests=0, proper_hits=0,
                  boundary_hits=0, ambiguous=0)
    records, uncertain = [], []
    bank_keys = np.unique(ids[:, :2], axis=0)
    counts["typed_banks"] = len(bank_keys)
    for body0, bone0 in bank_keys:
        body, bone = int(body0), int(bone0)
        if body not in body_local_points:
            raise ValueError(f"missing body-local FEM points for body {body}")
        ix = np.flatnonzero((ids[:, 0] == body) & (ids[:, 1] == bone))
        bank = tris[ix]
        stable = ids[ix, 3]
        source = int(ids[ix[0], 2])
        tri_low, tri_high = bank.min(axis=1), bank.max(axis=1)
        bank_low, bank_high = tri_low.min(axis=0), tri_high.max(axis=0)
        local = body_local_points[body]
        a_all, b_all = local[edges[:, 0]], local[edges[:, 1]]
        edge_low, edge_high = np.minimum(a_all, b_all), np.maximum(a_all, b_all)
        active = np.flatnonzero(np.all(edge_high >= bank_low, axis=1) & np.all(edge_low <= bank_high, axis=1))
        counts["broadphase_edge_bank_pairs"] += len(active)
        for edge_index in active:
            candidate_mask = np.all(tri_high >= edge_low[edge_index], axis=1) & np.all(tri_low <= edge_high[edge_index], axis=1)
            candidate_rows = np.flatnonzero(candidate_mask)
            counts["triangle_aabb_pairs"] += len(candidate_rows)
            if not len(candidate_rows):
                continue
            result = exact_segment_triangle_hits(a_all[edge_index], b_all[edge_index],
                                                 bank[candidate_rows], stable[candidate_rows], exact_classifier)
            counts["exact_narrowphase_tests"] += result["exact_candidates"]
            # Convert candidate-subset index back to the full typed-bank-local index.
            for hit in result["hits"]:
                local_index = int(candidate_rows[hit["bank_local_index"]])
                record = {"edge_index": int(edge_index),
                          "endpoint_local_nodes": edges[edge_index].tolist(),
                          "body": body, "bone_stable_id": bone,
                          "source_record_index": source,
                          "triangle_bank_local_index": local_index,
                          "triangle_stable_id": int(stable[local_index]),
                          "classification": hit["classification"],
                          "exact_result": hit["exact_result"],
                          "hit_reconstruction_error_m": hit["hit_reconstruction_error_m"]}
                if node_identity_domain == "global_node":
                    record.update({"endpoint_global_nodes": gids[edges[edge_index]].tolist(),
                                   "endpoint_layers": (edges[edge_index] // ROUTE_ROWS).tolist(),
                                   "endpoint_columns": (edges[edge_index] % ROUTE_ROWS).tolist()})
                else:
                    record["endpoint_query_rows"] = edges[edge_index].tolist()
                records.append(record)
                counts["proper_hits"] += int(hit["classification"] == "proper_interior_hit")
                counts["boundary_hits"] += int(hit["classification"] == "boundary_hit")
            for bad in result["uncertain"]:
                local_index = int(candidate_rows[bad["bank_local_index"]])
                uncertain.append({"edge_index": int(edge_index),
                                  "endpoint_local_nodes": edges[edge_index].tolist(),
                                  "body": body, "bone_stable_id": bone,
                                  "source_record_index": source,
                                  "triangle_bank_local_index": local_index,
                                  "triangle_stable_id": int(stable[local_index]),
                                  "classification": bad["classification"],
                                  "exact_result": bad["exact_result"]})
                counts["ambiguous"] += 1
    return {"status": "EDGE_WITNESSES_NOT_FULL_SOLID_CERTIFICATION", "counts": counts,
            "edges_with_hits": len({r["edge_index"] for r in records}),
            "edge_bank_groups_with_hits": len({(r["edge_index"], r["body"], r["bone_stable_id"]) for r in records}),
            "witnesses": records, "uncertain_witnesses": uncertain,
            "limitations": ["Only unique tetrahedral edges are screened against typed bone triangles.",
                            "Node containment and face/tet solid intersection require separate checks.",
                            "Floating broadphase filters exact rational narrowphase candidates.",
                            "No native physical or anatomical acceptance."]}


def bank_local_hit_triangle(bank_triangles: np.ndarray, stable_ids: np.ndarray,
                            bank_local_index: int, expected_stable_id: int) -> np.ndarray:
    """Resolve a hit index only within its typed bank and verify stable identity."""
    if bank_triangles.ndim != 3 or bank_triangles.shape[1:] != (3, 3):
        raise ValueError("bank triangles must have shape (n, 3, 3)")
    if stable_ids.ndim != 1 or len(stable_ids) != len(bank_triangles):
        raise ValueError("bank-local stable-ID array does not match triangle bank")
    if not 0 <= bank_local_index < len(bank_triangles):
        raise IndexError("bank-local hit triangle index is out of range")
    if int(stable_ids[bank_local_index]) != int(expected_stable_id):
        raise ValueError("bank-local hit index and stable triangle ID disagree")
    return bank_triangles[bank_local_index]


def validate_hit_reconstruction(triangle_body_local: np.ndarray, barycentric: np.ndarray,
                                segment_start_local: np.ndarray, segment_end_local: np.ndarray,
                                segment_parameter: float, tolerance_m: float = 2e-12) -> float:
    """Fail closed unless reported bank triangle reproduces the segment hit."""
    barycentric = np.asarray(barycentric, dtype=np.float64)
    if barycentric.shape != (3,) or not np.isfinite(barycentric).all():
        raise ValueError("hit barycentric coordinates must be three finite values")
    if abs(float(np.sum(barycentric)) - 1.0) > 1e-12:
        raise ValueError("hit barycentric coordinates do not sum to one")
    reconstructed = barycentric @ triangle_body_local
    on_segment = segment_start_local + float(segment_parameter) * (segment_end_local - segment_start_local)
    error = float(np.linalg.norm(reconstructed - on_segment))
    if not np.isfinite(error) or error > tolerance_m:
        raise ValueError(f"reported bank-local hit triangle misses segment witness by {error:.6g} m")
    return error


def orient_hit_normal(triangle_body_local: np.ndarray, world_to_body: np.ndarray,
                      segment_start_world: np.ndarray, segment_end_world: np.ndarray,
                      hit_world: np.ndarray, start_classification: str,
                      end_classification: str) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Keep start-side orientation separate from endpoint-certified exterior."""
    normal = np.cross(triangle_body_local[1] - triangle_body_local[0],
                      triangle_body_local[2] - triangle_body_local[0])
    normal = world_to_body.T @ normal
    length = float(np.linalg.norm(normal))
    if not np.isfinite(length) or length == 0:
        raise ValueError("hit triangle has invalid transformed normal")
    normal = normal / length
    if float(np.dot(normal, segment_start_world - hit_world)) < 0:
        normal = -normal
    if start_classification == "outside":
        return normal, normal.copy(), "exterior_toward_outside_start"
    if start_classification == "inside" and end_classification in ("outside", "inside"):
        away_start = -normal
        basis = ("exterior_toward_outside_end" if end_classification == "outside"
                 else "first_exit_toward_inside_end")
        return normal, away_start, basis
    if start_classification == "inside":
        return normal, None, "inside_start_end_unclassified_exterior_unknown"
    return normal, None, "start_endpoint_not_classified_exterior_unknown"


def geometry_only_endpoint_classification(query_map: dict | None, gid: int, body: int, bone: int) -> str:
    if query_map is None:
        return "not_queried"
    return query_map.get((gid, body, bone), "outside_AABB_proven")


def geometry_only_plane_fields() -> dict[str, Any]:
    """Construction planes are intentionally absent from this exact audit owner."""
    return {"construction_planes": 0, "construction_planes_world": []}


def audit(candidate_npz: pathlib.Path, candidate_sha: str, output_dir: pathlib.Path,
          evidence_root: pathlib.Path, bank_dir: pathlib.Path,
          candidate_report: pathlib.Path | None = None, candidate_report_sha: str | None = None,
          node_ids_npz: pathlib.Path | None = None, node_ids_sha: str | None = None,
          query_dir: pathlib.Path | None = None,
          query_node_file: str | None = None,
          all_tet_edges: bool = False) -> dict[str, Any]:
    t0 = time.monotonic()
    points, gids, pins = load_candidate_points(candidate_npz, candidate_sha, node_ids_npz, node_ids_sha)
    if candidate_report is not None:
        if candidate_report_sha is None:
            raise ValueError("candidate report requires an explicit SHA-256 pin")
        pins["candidate_report"] = require_pin(candidate_report, candidate_report_sha, "candidate report")
    elif candidate_report_sha is not None:
        raise ValueError("candidate report SHA supplied without a report path")
    triangles, ids, bank_gids, bodies, matrices, offsets, positions, bankkeys, bank_pins, exact = read_bank_inputs(bank_dir, evidence_root)
    pins.update({f"039/{k}": v for k, v in bank_pins.items()})
    if not np.array_equal(gids, bank_gids):
        raise ValueError("candidate global node IDs/order differ from frozen 039 route rows")
    # Local coordinates are computed using the same frozen native body transform
    # and COM offset used by the exact 039 bank export.
    locations = {}
    for bi in np.unique(ids[:, 0]):
        body = int(bi)
        locations[body] = (points.astype(np.float64) - positions[body]) @ matrices[body].T - offsets[body]
    query_map = None
    if query_dir is not None:
        qkeys, query_map, query_frame_max_error, query_pins = load_matched_query(
            query_dir, gids, points, bodies, matrices, offsets, positions, ids[:, 0], query_node_file)
        pins.update({f"matched_query/{k}": v for k, v in query_pins.items()})
        # Reproduce the all-bank broadphase coverage exactly from this candidate.
        actual = set()
        for body, bone in bankkeys:
            mask = (ids[:, 0] == int(body)) & (ids[:, 1] == int(bone))
            tri = triangles[mask]
            low, high = tri.min(axis=(0, 1)), tri.max(axis=(0, 1))
            inside = np.all(locations[int(body)] >= low, axis=1) & np.all(locations[int(body)] <= high, axis=1)
            actual.update((int(g), int(body), int(bone)) for g in gids[inside])
        if actual != qkeys:
            raise ValueError(f"matched query AABB coverage differs: extra={list(actual-qkeys)[:4]}, missing={list(qkeys-actual)[:4]}")
    gmap = {int(g): i for i, g in enumerate(gids)}
    counts = dict(total_column_bank_pairs=ROUTE_ROWS * 185,
                  total_segment_bank_pairs=2 * ROUTE_ROWS * 185,
                  AABB_candidate_segment_bank_pairs=0,
                  triangle_AABB_candidates=0, exact_candidates=0, proper_hits=0,
                  boundary_hits=0, coplanar_candidates=0, hit_segment_bank_pairs=0)
    records = []
    uncertain = []
    for body0, bone0 in bankkeys:
        body, bone = int(body0), int(bone0)
        ix = np.flatnonzero((ids[:, 0] == body) & (ids[:, 1] == bone))
        tris = triangles[ix]
        stable = ids[ix, 3]
        source = int(ids[ix[0], 2])
        tri_low = tris.min(axis=1)
        tri_high = tris.max(axis=1)
        bank_low, bank_high = tri_low.min(axis=0), tri_high.max(axis=0)
        local = locations[body].reshape(3, ROUTE_ROWS, 3)
        for seg in range(2):
            aa, bb = local[seg], local[seg + 1]
            route_low, route_high = np.minimum(aa, bb), np.maximum(aa, bb)
            active = np.flatnonzero(np.all(route_high >= bank_low, axis=1) & np.all(route_low <= bank_high, axis=1))
            counts["AABB_candidate_segment_bank_pairs"] += len(active)
            for row in active:
                candidate_tris = np.flatnonzero(np.all(tri_high >= route_low[row], axis=1) &
                                                  np.all(tri_low <= route_high[row], axis=1))
                counts["triangle_AABB_candidates"] += len(candidate_tris)
                if not len(candidate_tris):
                    continue
                candidate = tris[candidate_tris]
                direction = bb[row] - aa[row]
                e1, e2 = candidate[:, 1] - candidate[:, 0], candidate[:, 2] - candidate[:, 0]
                pvec = np.cross(direction, e2)
                det = np.einsum("ij,ij->i", e1, pvec)
                scale = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(direction)
                near = np.abs(det) <= 64 * np.finfo(float).eps * scale
                inv = np.divide(1.0, det, out=np.zeros_like(det), where=det != 0)
                tvec = aa[row] - candidate[:, 0]
                u = inv * np.einsum("ij,ij->i", tvec, pvec)
                qvec = np.cross(tvec, e1)
                v = inv * (qvec @ direction)
                t = inv * np.einsum("ij,ij->i", e2, qvec)
                possible = near | ((u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10) &
                                   (t >= -1e-10) & (t <= 1 + 1e-10))
                hits = []
                for ci in np.flatnonzero(possible):
                    counts["exact_candidates"] += 1
                    result = exact["classify_segment_triangle"](aa[row], bb[row], candidate[ci])
                    classification = result["classification"]
                    tid = int(stable[candidate_tris[ci]])
                    if classification in ("coplanar_requires_2d_test", "invalid_degenerate_triangle"):
                        counts["coplanar_candidates"] += classification == "coplanar_requires_2d_test"
                        uncertain.append(dict(row=int(row), segment=seg, body=body,
                                              bone_stable_id=bone, triangle_stable_id=tid,
                                              classification=classification))
                        continue
                    if classification not in ("proper_interior_hit", "boundary_hit"):
                        continue
                    counts["proper_hits" if classification == "proper_interior_hit" else "boundary_hits"] += 1
                    hits.append((result["t"], tid, int(candidate_tris[ci]), result))
                if not hits:
                    continue
                counts["hit_segment_bank_pairs"] += 1
                hits.sort(key=lambda z: (z[0], z[1]))
                first_t, tid, trirow, result = hits[0]
                world_start = points[seg * ROUTE_ROWS + row].astype(np.float64)
                world_end = points[(seg + 1) * ROUTE_ROWS + row].astype(np.float64)
                point = world_start + first_t * (world_end - world_start)
                gid_start = int(gids[seg * ROUTE_ROWS + row])
                gid_end = int(gids[(seg + 1) * ROUTE_ROWS + row])
                start_class = geometry_only_endpoint_classification(query_map, gid_start, body, bone)
                end_class = geometry_only_endpoint_classification(query_map, gid_end, body, bone)
                hit_triangle = bank_local_hit_triangle(tris, stable, trirow, tid)
                hit_reconstruction_error = validate_hit_reconstruction(
                    hit_triangle, np.asarray(result["barycentric"]), aa[row], bb[row], first_t)
                end_world = points[(seg + 1) * ROUTE_ROWS + row].astype(np.float64)
                start_side_normal, exterior_normal, normal_orientation_basis = orient_hit_normal(
                    hit_triangle, matrices[body], world_start, end_world, point, start_class, end_class)
                firstboundary = (result["classification"] != "proper_interior_hit" or
                                 (len(hits) > 1 and abs(hits[1][0] - first_t) < 1e-10))
                records.append(dict(
                    row=int(row), segment=seg, segment_name="O-I" if seg == 0 else "I-D",
                    body=body, bone_stable_id=bone, source_record_index=source,
                    triangle_stable_id=tid, t=first_t, start_classification=start_class,
                    end_classification=end_class, hit_triangle_count=len(hits),
                    first_hit_ambiguous=firstboundary, exact_first_hit=result,
                    hit_reconstruction_error_m=hit_reconstruction_error,
                    plane_point_world_m=point.tolist(),
                    unit_start_side_normal_world=start_side_normal.tolist(),
                    unit_exterior_normal_world=(exterior_normal.tolist() if exterior_normal is not None else None),
                    normal_orientation_basis=normal_orientation_basis,
                    segment_start_world_f32_m=world_start.tolist(),
                    segment_end_world_f32_m=world_end.tolist(),
                    segment_start_query_local_m=aa[row].tolist(),
                    segment_end_query_local_m=bb[row].tolist(),
                    triangle_body_local_f32_m=hit_triangle.tolist(),
                    all_hits=[dict(t=h[0], triangle_stable_id=h[1], classification=h[3]["classification"])
                              for h in hits],
                ))
    groups = {}
    for record in records:
        groups.setdefault((record["row"], record["body"], record["bone_stable_id"]), []).append(record)
    for route_records in groups.values():
        route_records.sort(key=lambda r: (r["segment"], r["t"], r["triangle_stable_id"]))
        route_records[0]["is_first_route_hit"] = True
    # This audit is geometry-only: construction planes are deliberately never
    # synthesized, even when matched endpoint classifications are available.
    construction_planes = []
    input_pins = {str(candidate_npz): pins["candidate_npz"]}
    if candidate_report is not None:
        input_pins[str(candidate_report)] = pins["candidate_report"]
    if node_ids_npz is not None:
        input_pins[str(node_ids_npz)] = pins["node_ids_npz"]
    for key, value in bank_pins.items():
        path = bank_dir / key
        input_pins[str(path)] = value
    input_pins[str(evidence_root / "root_exact_segment_witness.py")] = PINNED_SEGMENT_HELPER
    audit_script = pathlib.Path(__file__).resolve()
    input_pins[str(audit_script)] = sha256(audit_script)
    for key, value in pins.items():
        if key.startswith("matched_query/"):
            input_pins[str(query_dir / key.split("/", 1)[1])] = value
    edge_report = None
    if all_tet_edges:
        with np.load(candidate_npz, allow_pickle=False) as candidate_data:
            if "tetrahedra" not in candidate_data.files:
                raise ValueError("--all-tet-edges requires candidate tetrahedra")
            candidate_tets = np.asarray(candidate_data["tetrahedra"], dtype=np.int64)
        edge_report = audit_unique_tet_edges(gids, candidate_tets, ids, triangles, locations,
                                             exact["classify_segment_triangle"])
        edge_report.update({"candidate_npz": str(candidate_npz),
                            "candidate_sha256": pins["candidate_npz"],
                            "bank_triangle_export_sha256": bank_pins["exact-emitted.triangles.csv"],
                            "exact_segment_helper_sha256": PINNED_SEGMENT_HELPER,
                            "audit_script_sha256": sha256(pathlib.Path(__file__).resolve())})
    summary = dict(
        status="EXACT_NATIVE_GEOMETRY_ROUTE_WITNESSES_NOT_FULL_TISSUE_CLEARANCE",
        counts=counts,
        columns_with_hits=len({r["row"] for r in records}),
        column_bank_groups_with_hits=len(groups),
        **geometry_only_plane_fields(),
        uncertainties=len(uncertain),
        native_query_frame_max_error_m=query_frame_max_error if query_dir is not None else None,
        inputs=input_pins,
        elapsed_seconds=time.monotonic() - t0,
        limitations=[
            "Exact rational signs certify narrowphase candidates; this is not a layer-triangle/tet clearance test.",
            "Coplanar and first-boundary feature hits remain explicit unresolved witnesses.",
            *( ["Endpoint parity was not queried; endpoint classifications are not_queried."]
               if query_dir is None else
               ["Endpoint classifications are copied from the hash-pinned matched query; this audit does not rerun endpoint parity."] ),
            "This owner never emits construction planes.",
            "No native physical step or anatomy acceptance.",
        ],
        route_witnesses=records,
        uncertain_candidates=uncertain,
        tetrahedral_edge_audit=(None if edge_report is None else {
            "status": edge_report["status"], "counts": edge_report["counts"],
            "edges_with_hits": edge_report["edges_with_hits"],
            "edge_bank_groups_with_hits": edge_report["edge_bank_groups_with_hits"],
            "companion_report": "tet-edge-report.json"}),
    )
    # Preserve the established route-report structure and fail closed if a
    # future edit accidentally reintroduces plane output.
    if summary["construction_planes"] != 0 or summary["construction_planes_world"]:
        raise AssertionError("geometry-only route audit must never emit construction planes")
    if query_dir is None:
        if any(r["start_classification"] != "not_queried" or
               r["end_classification"] != "not_queried" for r in records):
            raise AssertionError("geometry-only endpoints must remain not_queried")
    output_dir.mkdir(parents=True, exist_ok=False)
    edge_report_path = None
    if edge_report is not None:
        edge_report_path = output_dir / "tet-edge-report.json"
        edge_report_path.write_text(json.dumps(edge_report, indent=2) + "\n")
        summary["tetrahedral_edge_audit"]["companion_report_sha256"] = sha256(edge_report_path)
    (output_dir / "route-report.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "source-pins.json").write_text(json.dumps({
        "schema": "numi.human.route-audit-consolidation-019.source-pins.v1",
        "candidate_npz": {"path": str(candidate_npz), "sha256": pins["candidate_npz"],
                          "global_node_id_source": pins["candidate_id_source"],
                          "row_identity_proof": pins["candidate_id_source"]},
        "candidate_report": ({"path": str(candidate_report), "sha256": pins["candidate_report"]}
                             if candidate_report is not None else None),
        "node_ids_npz": ({"path": str(node_ids_npz), "sha256": pins["node_ids_npz"]}
                         if node_ids_npz is not None else None),
        "bank_export_039": {str(bank_dir / key): value for key, value in bank_pins.items()},
        "exact_segment_helper": {"path": str(evidence_root / "root_exact_segment_witness.py"),
                                 "sha256": PINNED_SEGMENT_HELPER},
        "audit_script": {"path": str(audit_script), "sha256": sha256(audit_script)},
        "matched_query": ({str(query_dir / key.split("/", 1)[1]): value
                           for key, value in pins.items() if key.startswith("matched_query/")}
                          if query_dir is not None else None),
        "geometry_only": True,
        "construction_planes_emitted": False,
        "endpoint_parity_queried": query_dir is not None,
        "tetrahedral_edge_audit": ({"enabled": True,
                                     "path": str(edge_report_path),
                                     "sha256": sha256(edge_report_path)}
                                    if edge_report_path is not None else {"enabled": False}),
    }, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", required=True, type=pathlib.Path)
    parser.add_argument("--bank-export-dir", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-npz", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--candidate-report", type=pathlib.Path)
    parser.add_argument("--candidate-report-sha256")
    parser.add_argument("--node-ids-npz", type=pathlib.Path)
    parser.add_argument("--node-ids-sha256")
    parser.add_argument("--matched-query-run-dir", type=pathlib.Path)
    parser.add_argument("--matched-query-node-file", choices=("exact-emitted.nodes.csv", "initial-skin-fem-nodes.csv"),
                        help="Explicit 023 node export for parity check; default prefers exact-emitted.nodes.csv")
    parser.add_argument("--all-tet-edges", action="store_true",
                        help="Also audit deterministic unique candidate tetrahedral edges against all typed bone banks")
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = audit(args.candidate_npz, args.candidate_sha256, args.output_dir,
                   args.evidence_root, args.bank_export_dir,
                   args.candidate_report, args.candidate_report_sha256,
                   args.node_ids_npz, args.node_ids_sha256,
                   args.matched_query_run_dir, args.matched_query_node_file,
                   args.all_tet_edges)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("route_witnesses", "construction_planes_world", "uncertain_candidates")},
                     indent=2))



def world_clearance_plane(world,query_local,closest_local,M,offset,position):
    """Build a local proposal plane from an outside closest-point query.

    Raw registered frames may be slightly nonorthogonal. Transform normals as
    covectors and invert the matrix for positions; do not renormalize the source
    quaternion. This plane is not a global solid-clearance certificate.
    """
    world=np.asarray(world,dtype=float);ql=np.asarray(query_local,dtype=float);cl=np.asarray(closest_local,dtype=float)
    M=np.asarray(M,dtype=float);offset=np.asarray(offset,dtype=float);position=np.asarray(position,dtype=float)
    if M.shape!=(3,3) or any(v.shape!=(3,) for v in [world,ql,cl,offset,position]) or not all(np.isfinite(v).all()for v in [world,ql,cl,M,offset,position]):
        raise ValueError("nonfinite or malformed pinned transform")
    expected=M@(world-position)-offset
    error=float(np.linalg.norm(expected-ql))
    if error>1.e-12: raise ValueError(f"query position does not match pinned source: {error}")
    local_delta=ql-cl
    norm=float(np.linalg.norm(local_delta))
    if norm<=0 or not np.isfinite(norm):raise ValueError("undefined outside distance normal")
    local_normal=local_delta/norm
    covector=M.T@local_normal
    scale=float(np.linalg.norm(covector))
    if scale<=0 or not np.isfinite(scale):raise ValueError("singular plane transform")
    n=covector/scale
    closest_world=np.linalg.solve(M,cl+offset)+position
    distance=float(n@(world-closest_world))
    if distance<=0:raise ValueError("baseline point is not on positive plane side")
    if not np.isclose(distance,norm/scale,rtol=1.e-9,atol=2.e-15):raise ValueError("plane transform distance mismatch")
    margin=min(50.e-6,.25*distance)
    return n,margin-distance,closest_world,distance,error


def summarize_surface_crossing_report(report, expected_banks=185, *, expected_bank_keys=None):
    if not isinstance(report,dict) or report.get("complete") is not True:
        raise ValueError("159 report must be a complete object")
    banks=report.get("banks")
    if not isinstance(banks,list) or len(banks)!=expected_banks:
        raise ValueError("159 bank inventory is incomplete")
    if int(report.get("bank_count",-1))!=expected_banks or int(report.get("expected_bank_count",-1))!=expected_banks:
        raise ValueError("159 bank-count receipts disagree")
    if not isinstance(expected_banks,int) or isinstance(expected_banks,bool) or expected_banks<=0:
        raise ValueError("expected_banks must be a positive integer")
    total=0; per_bank=[]; seen=set()
    for i,bank in enumerate(banks):
        if not isinstance(bank,dict):
            raise ValueError(f"invalid bank row {i}")
        body=bank.get("body");bone=bank.get("stable_bone_id")
        if type(body) is not int or type(bone) is not int:
            raise ValueError(f"missing typed bank identity at row {i}")
        key=(body,bone)
        if key in seen:
            raise ValueError(f"duplicate typed bank identity: {key}")
        seen.add(key)
        if "result" not in bank:
            raise ValueError(f"missing explicit result at bank row {i}")
        result=bank["result"]
        if result is None:
            # The only admissible zero without narrowphase is a recorded complete
            # skin-vs-bank AABB exclusion. Missing counts never silently become zero.
            if (bank.get("skin_aabb_active") != 0 or bank.get("bank_aabb_active") != 0
                or not isinstance(bank.get("skin_faces"),int) or bank["skin_faces"]<=0
                or not isinstance(bank.get("bank_faces"),int) or bank["bank_faces"]<=0):
                raise ValueError(f"missing result/count without explicit AABB exclusion at bank row {i}")
            count=0; source="aabb_disjoint_no_narrowphase"
        else:
            if not isinstance(result,dict):
                raise ValueError(f"invalid result object at bank row {i}")
            count=result.get("count")
            pairs=result.get("triangle_pairs")
            if not isinstance(count,int) or isinstance(count,bool) or count<0:
                raise ValueError(f"missing or invalid explicit count at bank row {i}")
            if not isinstance(pairs,list) or len(pairs)!=count:
                raise ValueError(f"pair-array/count mismatch at bank row {i}")
            source="exact_narrowphase_count"
        total+=count
        per_bank.append({"body":body,"stable_bone_id":bone,"count":count,"count_source":source})
    if expected_bank_keys is not None:
        expected=set(expected_bank_keys)
        if len(expected)!=expected_banks or seen!=expected:
            raise ValueError("typed bank inventory differs from pinned expected keys")
    return {"status":"COMPLETE_EXPLICIT_PER_BANK_COUNT_SUM","bank_count":len(banks),
            "expected_bank_count":expected_banks,"total_crossing_pairs":total,
            "nonzero_bank_count":sum(x["count"]>0 for x in per_bank),
            "per_bank":per_bank}


def _read_point_clearance_report(report):
    """Parse P089's complete point witnesses and return typed pair depths."""
    import math

    if not isinstance(report, dict) or report.get("complete_surface_audit") is not True:
        raise ValueError("incomplete exact point/surface audit")
    self_pairs = report.get("exact_PLC_surface_pairs")
    if type(self_pairs) is not int or self_pairs != 0:
        raise ValueError("proposal has unresolved surface self intersections")
    for name in ("points", "faces", "inside_node_bank_pairs", "inside_nodes"):
        if type(report.get(name)) is not int or report[name] < 0:
            raise ValueError("missing or invalid audit count: " + name)
    rows = report.get("inside_witnesses")
    if not isinstance(rows, list):
        raise ValueError("missing typed inside witnesses")
    values = {}

    def identity(value):
        # Existing query reports retain CSV identities as decimal strings.
        if type(value) is int and value >= 0:
            return value
        if isinstance(value, str) and value.isascii() and value.isdecimal():
            return int(value)
        raise ValueError("invalid typed point/bank identity")

    for row in rows:
        if not isinstance(row, dict) or row.get("classification") != "inside":
            raise ValueError("witness is not an explicit inside classification")
        key = tuple(identity(row.get(k)) for k in ("query_row", "bank_body", "bank_stable_id"))
        if key[0] >= report["points"] or key in values:
            raise ValueError("duplicate or out of range typed inside witness")
        if isinstance(row.get("nearest_distance_m"), bool):
            raise ValueError("invalid penetration distance")
        try:
            distance = float(row["nearest_distance_m"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("missing penetration distance") from None
        if not math.isfinite(distance) or distance < 0 or not math.isfinite(distance * distance):
            raise ValueError("nonfinite, negative or unrepresentable penetration distance")
        values[key] = distance
    if len(values) != report["inside_node_bank_pairs"] or len({k[0] for k in values}) != report["inside_nodes"]:
        raise ValueError("typed inside counts disagree")
    return values


def assess_point_clearance_construction_progress(before, after):
    """Compare complete, same-topology point audits during mesh construction.

    This preserves the P089 point-only merit. It never grants geometry or
    native admission; the joint helper below handles the combined count rule.
    """
    import math

    old, new = _read_point_clearance_report(before), _read_point_clearance_report(after)
    if (before["points"], before["faces"]) != (after["points"], after["faces"]):
        raise ValueError("point merit requires matching topology")
    maximum_old, maximum_new = max(old.values(), default=0.0), max(new.values(), default=0.0)
    squared_old = math.fsum(v * v for v in old.values())
    squared_new = math.fsum(v * v for v in new.values())
    introduced = sorted(new.keys() - old.keys())
    removed = sorted(old.keys() - new.keys())
    count_improved = len(new) < len(old)
    return {
        "construction_progress": count_improved and maximum_new <= maximum_old and squared_new <= squared_old,
        "strict_pair_count_reduction": count_improved,
        "maximum_depth_nonincreasing": maximum_new <= maximum_old,
        "squared_depth_nonincreasing": squared_new <= squared_old,
        "before_inside_pairs": len(old), "after_inside_pairs": len(new),
        "before_maximum_depth_m": maximum_old, "after_maximum_depth_m": maximum_new,
        "before_squared_depth_m2": squared_old, "after_squared_depth_m2": squared_new,
        "introduced_typed_pairs": [list(k) for k in introduced],
        "introduced_depths_m": [new[k] for k in introduced],
        "removed_typed_pairs": [list(k) for k in removed],
        "geometry_admitted": False, "accepted_physical_steps": 0,
    }


def assess_joint_clearance_construction_progress(before_points, after_points,
                                                  before_surface, after_surface):
    """Compare matching complete point and strict-185 surface audits.

    A proposal advances construction only when the bone-surface crossing count
    and combined point-plus-surface count both fall strictly, while maximum and
    squared point penetration do not rise. This is not geometric, anatomical,
    or native admission.
    """
    import math
    import re

    def candidate_hash(report, label):
        if not isinstance(report, dict):
            raise ValueError(label + " report must be an object")
        value = report.get("candidate_sha256")
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(label + " report lacks a valid candidate SHA-256")
        return value

    def topology(point_report, surface_report, label):
        pairs = (("points", "surface_point_count"), ("faces", "surface_face_count"))
        out = []
        for point_key, surface_key in pairs:
            value = point_report.get(point_key)
            surface_value = surface_report.get(surface_key)
            if type(value) is not int or value < 0 or type(surface_value) is not int or surface_value < 0:
                raise ValueError(label + " report has missing or invalid topology dimensions")
            if value != surface_value:
                raise ValueError(label + " point/surface topology dimensions disagree")
            out.append(value)
        return tuple(out)

    before_candidate = candidate_hash(before_points, "before point")
    after_candidate = candidate_hash(after_points, "after point")
    if candidate_hash(before_surface, "before surface") != before_candidate:
        raise ValueError("before point/surface candidate SHA mismatch")
    if candidate_hash(after_surface, "after surface") != after_candidate:
        raise ValueError("after point/surface candidate SHA mismatch")

    before_point_values = _read_point_clearance_report(before_points)
    after_point_values = _read_point_clearance_report(after_points)
    for report, label in ((before_points, "before"), (after_points, "after")):
        if type(report.get("query_pair_count")) is not int or report["query_pair_count"] < 0:
            raise ValueError(label + " point report has missing or invalid query-pair count")
    before_topology = topology(before_points, before_surface, "before")
    after_topology = topology(after_points, after_surface, "after")
    if before_topology != after_topology:
        raise ValueError("joint merit requires matching topology dimensions")

    before_surface_summary = summarize_surface_crossing_report(before_surface, expected_banks=185)
    expected_keys = [(row["body"], row["stable_bone_id"])
                     for row in before_surface_summary["per_bank"]]
    after_surface_summary = summarize_surface_crossing_report(
        after_surface, expected_banks=185, expected_bank_keys=expected_keys)
    before_keys = {(row["body"], row["stable_bone_id"]) for row in before_surface_summary["per_bank"]}
    after_keys = {(row["body"], row["stable_bone_id"]) for row in after_surface_summary["per_bank"]}
    if before_keys != after_keys or len(before_keys) != 185:
        raise ValueError("joint merit requires the same 185 typed banks")

    before_point_count, after_point_count = len(before_point_values), len(after_point_values)
    before_surface_count = before_surface_summary["total_crossing_pairs"]
    after_surface_count = after_surface_summary["total_crossing_pairs"]
    before_total = before_point_count + before_surface_count
    after_total = after_point_count + after_surface_count
    before_max = max(before_point_values.values(), default=0.0)
    after_max = max(after_point_values.values(), default=0.0)
    before_squared = math.fsum(value * value for value in before_point_values.values())
    after_squared = math.fsum(value * value for value in after_point_values.values())
    strict_surface_reduction = after_surface_count < before_surface_count
    strict_total_reduction = after_total < before_total
    max_nonincreasing = after_max <= before_max
    squared_nonincreasing = after_squared <= before_squared
    progress = (strict_surface_reduction and strict_total_reduction
                and max_nonincreasing and squared_nonincreasing)
    final_zero_unchanged = before_total == 0 and after_total == 0
    return {
        "classification": ("JOINT_CLEARANCE_CONSTRUCTION_PROGRESS" if progress
                           else "NO_JOINT_CLEARANCE_CONSTRUCTION_PROGRESS"),
        "construction_progress": progress,
        "strict_surface_pair_reduction": strict_surface_reduction,
        "strict_total_pair_reduction": strict_total_reduction,
        "maximum_depth_nonincreasing": max_nonincreasing,
        "squared_depth_nonincreasing": squared_nonincreasing,
        "topology_points": before_topology[0],
        "topology_faces": before_topology[1],
        "typed_bank_count": len(before_keys),
        "before_candidate_sha256": before_candidate,
        "after_candidate_sha256": after_candidate,
        "before_point_pairs": before_point_count,
        "after_point_pairs": after_point_count,
        "before_surface_pairs": before_surface_count,
        "after_surface_pairs": after_surface_count,
        "before_total_pairs": before_total,
        "after_total_pairs": after_total,
        "before_maximum_point_depth_m": before_max,
        "after_maximum_point_depth_m": after_max,
        "before_squared_point_depth_m2": before_squared,
        "after_squared_point_depth_m2": after_squared,
        "final_zero_clearance_unchanged": final_zero_unchanged,
        "geometry_admitted": False,
        "accepted_physical_steps": 0,
        "native_admission": False,
    }



from collections import Counter

def assess_construction_depth_progress(before_points, after_points,
                                      before_surface, after_surface, *,
                                      absolute_roundoff_m2=1.0e-24,
                                      relative_roundoff=64.0 * 2.220446049250313e-16):
    """Classify strict point-depth progress without requiring pair-count reduction.

    The existing joint assessor is called first, so all complete-report,
    candidate-hash, topology, zero-self-pair, and matched-185-bank checks remain
    mandatory. This additive construction classification permits unchanged
    counts only when no typed point or surface witness is introduced, maximum
    point depth does not increase, and squared point depth decreases beyond a
    conservative floating-point roundoff threshold. It never admits geometry or
    native execution.
    """
    import json
    import math

    for name, value in (("absolute_roundoff_m2", absolute_roundoff_m2),
                        ("relative_roundoff", relative_roundoff)):
        if isinstance(value, bool):
            raise ValueError(name + " must be a finite nonnegative number")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(name + " must be a finite nonnegative number") from None
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(name + " must be a finite nonnegative number")
        if name == "absolute_roundoff_m2":
            absolute_roundoff_m2 = value
        else:
            relative_roundoff = value

    # Reuse the owner validator; this must not weaken the default joint merit.
    base = assess_joint_clearance_construction_progress(
        before_points, after_points, before_surface, after_surface)

    old_points = _read_point_clearance_report(before_points)
    new_points = _read_point_clearance_report(after_points)
    old_max = max(old_points.values(), default=0.0)
    new_max = max(new_points.values(), default=0.0)
    old_squared = math.fsum(v * v for v in old_points.values())
    new_squared = math.fsum(v * v for v in new_points.values())

    old_surface = summarize_surface_crossing_report(before_surface, expected_banks=185)
    new_surface = summarize_surface_crossing_report(
        after_surface, expected_banks=185,
        expected_bank_keys=[(row["body"], row["stable_bone_id"])
                            for row in old_surface["per_bank"]])
    old_bank_counts = {(row["body"], row["stable_bone_id"]): row["count"]
                       for row in old_surface["per_bank"]}
    new_bank_counts = {(row["body"], row["stable_bone_id"]): row["count"]
                       for row in new_surface["per_bank"]}
    if old_bank_counts.keys() != new_bank_counts.keys() or len(old_bank_counts) != 185:
        raise ValueError("depth progress requires the same 185 typed banks")

    # Surface witnesses are typed by bank plus the exact triangle-pair identity.
    def surface_pairs(report):
        typed = Counter()
        for bank in report["banks"]:
            result = bank["result"]
            if result is None:
                continue
            for pair in result["triangle_pairs"]:
                try:
                    identity = json.dumps(pair, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False)
                except (TypeError, ValueError):
                    raise ValueError("surface pair identity is not finite JSON data") from None
                typed[(bank["body"], bank["stable_bone_id"], identity)] += 1
        return typed

    old_surface_pairs = surface_pairs(before_surface)
    new_surface_pairs = surface_pairs(after_surface)
    introduced_point_pairs = sorted(new_points.keys() - old_points.keys())
    introduced_surface_pairs = sorted(
        key for key, count in new_surface_pairs.items()
        if count > old_surface_pairs.get(key, 0)
    )
    no_introduced_typed_pairs = not introduced_point_pairs and not introduced_surface_pairs
    point_count_nonincreasing = len(new_points) <= len(old_points)
    surface_count_nonincreasing = (
        new_surface["total_crossing_pairs"] <= old_surface["total_crossing_pairs"]
        and all(new_bank_counts[key] <= old_bank_counts[key] for key in old_bank_counts)
    )
    same_query_coverage = before_points["query_pair_count"] == after_points["query_pair_count"]
    maximum_nonincreasing = new_max <= old_max
    roundoff_threshold = max(
        absolute_roundoff_m2,
        relative_roundoff * max(abs(old_squared), abs(new_squared)),
        8.0 * max(math.ulp(old_squared), math.ulp(new_squared)),
    )
    squared_improvement = old_squared - new_squared
    strict_squared_progress = squared_improvement > roundoff_threshold
    progress = bool(
        point_count_nonincreasing and surface_count_nonincreasing
        and no_introduced_typed_pairs and same_query_coverage
        and maximum_nonincreasing and strict_squared_progress
    )
    return {
        "classification": ("CONSTRUCTION_DEPTH_PROGRESS" if progress
                           else "NO_CONSTRUCTION_DEPTH_PROGRESS"),
        "construction_progress": progress,
        "point_pair_count_nonincreasing": point_count_nonincreasing,
        "surface_pair_count_nonincreasing": surface_count_nonincreasing,
        "same_query_coverage": same_query_coverage,
        "no_introduced_typed_pairs": no_introduced_typed_pairs,
        "introduced_point_pairs": [list(k) for k in introduced_point_pairs],
        "introduced_surface_pair_count": len(introduced_surface_pairs),
        "maximum_depth_nonincreasing": maximum_nonincreasing,
        "strict_squared_depth_progress": strict_squared_progress,
        "before_point_pairs": len(old_points),
        "after_point_pairs": len(new_points),
        "before_surface_pairs": old_surface["total_crossing_pairs"],
        "after_surface_pairs": new_surface["total_crossing_pairs"],
        "before_maximum_depth_m": old_max,
        "after_maximum_depth_m": new_max,
        "before_squared_depth_m2": old_squared,
        "after_squared_depth_m2": new_squared,
        "squared_depth_improvement_m2": squared_improvement,
        "squared_depth_roundoff_threshold_m2": roundoff_threshold,
        "typed_bank_count": len(old_bank_counts),
        "topology_points": base["topology_points"],
        "topology_faces": base["topology_faces"],
        "before_candidate_sha256": base["before_candidate_sha256"],
        "after_candidate_sha256": base["after_candidate_sha256"],
        "geometry_admitted": False,
        "native_admission": False,
        "accepted_physical_steps": 0,
    }



if __name__ == "__main__":
    main()

