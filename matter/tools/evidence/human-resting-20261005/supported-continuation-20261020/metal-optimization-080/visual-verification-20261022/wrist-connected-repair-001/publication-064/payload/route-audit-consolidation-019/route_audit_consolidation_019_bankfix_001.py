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
          query_node_file: str | None = None) -> dict[str, Any]:
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
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = audit(args.candidate_npz, args.candidate_sha256, args.output_dir,
                   args.evidence_root, args.bank_export_dir,
                   args.candidate_report, args.candidate_report_sha256,
                   args.node_ids_npz, args.node_ids_sha256,
                   args.matched_query_run_dir, args.matched_query_node_file)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("route_witnesses", "construction_planes_world", "uncertain_candidates")},
                     indent=2))


if __name__ == "__main__":
    main()

