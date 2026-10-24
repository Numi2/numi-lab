"""Small, auditable helpers for a per-bone D37 fit restart."""
from __future__ import annotations
import csv
import math
import re
from typing import Iterable, Mapping
import numpy as np

_REQUIRED_CSV = {
    "local_node", "bank_body", "bank_stable_id", "classification",
    "nearest_bone", "nearest_distance_m",
    "query_local_x", "query_local_y", "query_local_z",
    "closest_x", "closest_y", "closest_z",
}


def expand_one_ring(seed_columns: Iterable[int], oriented_outer_faces: np.ndarray) -> np.ndarray:
    """Return seed columns plus all vertices in O faces incident to the seed set."""
    F = np.asarray(oriented_outer_faces)
    if F.ndim != 2 or F.shape[1] != 3 or F.dtype.kind not in "iu":
        raise ValueError("outer faces must be an integer [F,3] array")
    seeds = {int(x) for x in seed_columns}
    if any(x < 0 for x in seeds):
        raise ValueError("negative seed column")
    ring = set(seeds)
    if len(F):
        incident = np.isin(F, np.fromiter(seeds, dtype=np.int64)).any(axis=1) if seeds else np.zeros(len(F), bool)
        ring.update(int(x) for x in F[incident].ravel())
    if F.size and min(ring, default=0) < 0:
        raise ValueError("negative face column")
    return np.asarray(sorted(ring), dtype=np.int64)


def merge_quality_rows(active_rows: Iterable[int], newly_low_rows: Iterable[int]) -> np.ndarray:
    """Monotone active-set union in global affected-tet row IDs."""
    a = np.asarray(list(active_rows), dtype=np.int64).reshape(-1)
    b = np.asarray(list(newly_low_rows), dtype=np.int64).reshape(-1)
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("negative tetrahedron row")
    return np.unique(np.concatenate((a, b)))


def discover_quality_rows(affected_global_rows: np.ndarray,
                          q_float64: np.ndarray,
                          q_emitted_float32: np.ndarray,
                          threshold: float) -> np.ndarray:
    """Find every affected cell below the restart threshold in either arithmetic."""
    rows = np.asarray(affected_global_rows, dtype=np.int64).reshape(-1)
    q64 = np.asarray(q_float64, dtype=np.float64).reshape(-1)
    q32 = np.asarray(q_emitted_float32, dtype=np.float64).reshape(-1)
    if rows.shape != q64.shape or rows.shape != q32.shape or len(np.unique(rows)) != len(rows):
        raise ValueError("quality scan extents/row identities invalid")
    if not np.isfinite(threshold) or threshold <= 0 or not np.all(np.isfinite(q64)) or not np.all(np.isfinite(q32)):
        raise ValueError("nonfinite/nonpositive quality scan input")
    mask = (q64 < threshold) | (q32 < threshold)
    return rows[mask]


class ActiveQualityRows:
    """Monotone global-row active set driven by a full affected-cell scan."""
    def __init__(self, affected_global_rows, initial_active_rows, threshold):
        self.affected = np.asarray(affected_global_rows, dtype=np.int64).reshape(-1)
        if len(np.unique(self.affected)) != len(self.affected) or np.any(self.affected < 0):
            raise ValueError("affected quality row identities invalid")
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("quality threshold must be finite and positive")
        self.threshold = float(threshold)
        self.active = merge_quality_rows([], initial_active_rows)
        if not np.all(np.isin(self.active, self.affected)):
            raise ValueError("initial active rows fall outside affected domain")
        self.scans = 0
        self.last_new = np.empty(0, dtype=np.int64)
        self.last_low = np.empty(0, dtype=np.int64)

    def scan(self, q_float64, q_emitted_float32):
        """Scan every affected cell; merge all new low rows before restart."""
        low = discover_quality_rows(self.affected, q_float64, q_emitted_float32, self.threshold)
        new = np.setdiff1d(low, self.active, assume_unique=True)
        self.active = merge_quality_rows(self.active, low)
        self.scans += 1
        self.last_new = new
        self.last_low = low
        return new.copy()

    def checkpoint(self, path, *, x_mm, variable_local_rows, q_float64, q_emitted_float32):
        x = np.asarray(x_mm, dtype=np.float64)
        v = np.asarray(variable_local_rows, dtype=np.int64).reshape(-1)
        q64 = np.asarray(q_float64, dtype=np.float64).reshape(-1)
        q32 = np.asarray(q_emitted_float32, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or x.shape[1] != 3 or not np.all(np.isfinite(x)):
            raise ValueError("checkpoint iterate must be finite [N,3]")
        if q64.shape != self.affected.shape or q32.shape != self.affected.shape:
            raise ValueError("checkpoint quality vectors do not cover every affected row")
        if not np.all(np.isfinite(q64)) or not np.all(np.isfinite(q32)):
            raise ValueError("checkpoint quality contains nonfinite values")
        np.savez_compressed(path, x_mm=x, variable_local_rows=v,
                            affected_tet_rows=self.affected,
                            active_tet_rows=self.active,
                            last_low_tet_rows=self.last_low,
                            last_new_tet_rows=self.last_new,
                            q_float64=q64, q_emitted_float32=q32,
                            scan_count=np.asarray(self.scans, dtype=np.int64),
                            restart_threshold=np.asarray(self.threshold, dtype=np.float64))


def _uint(text: str, field: str) -> int:
    s = str(text).strip()
    if not re.fullmatch(r"\+?\d+", s):
        raise ValueError(f"{field} must be an exact nonnegative integer token")
    return int(s)


def parse_all_bank_csv(source, *, node_count: int | None = None,
                       distance_tolerance_m: float = 2e-9):
    """Parse the 035 all-bank CSV without collapsing distinct bone constraints.

    Returns (inside_rows, unresolved_rows, summary). Only classification=inside
    with positive reconstructed distance becomes a plane candidate. Ambiguous,
    invalid, and zero-distance rows are preserved as unresolved diagnostics.
    """
    if not math.isfinite(distance_tolerance_m) or distance_tolerance_m <= 0:
        raise ValueError("distance tolerance must be finite and positive")
    close = False
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        f = open(source, newline="", encoding="utf-8")
        close = True
    else:
        f = source
    try:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not _REQUIRED_CSV.issubset(reader.fieldnames):
            raise ValueError("035 CSV is missing required per-bone fields")
        rows = []
        unresolved = []
        seen = set()
        for line_no, raw in enumerate(reader, start=2):
            local = _uint(raw["local_node"], "local_node")
            body = _uint(raw["bank_body"], "bank_body")
            bone = _uint(raw["bank_stable_id"], "bank_stable_id")
            nearest_bone = _uint(raw["nearest_bone"], "nearest_bone")
            tri_text = raw.get("nearest_triangle", "")
            tri = _uint(tri_text, "nearest_triangle") if str(tri_text).strip() else None
            if node_count is not None and local >= node_count:
                raise ValueError(f"local_node out of range at CSV line {line_no}")
            key = (local, body, bone)
            if key in seen:
                raise ValueError(f"duplicate node/body/bone row at CSV line {line_no}: {key}")
            seen.add(key)
            classification = str(raw["classification"]).strip().lower()
            if classification not in {"inside", "outside", "ambiguous", "invalid"}:
                raise ValueError(f"unknown classification at CSV line {line_no}")
            q = np.array([float(raw[f"query_local_{a}"]) for a in "xyz"], dtype=np.float64)
            c = np.array([float(raw[f"closest_{a}"]) for a in "xyz"], dtype=np.float64)
            dist_declared = float(raw["nearest_distance_m"])
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(c)) or not math.isfinite(dist_declared) or dist_declared < 0:
                raise ValueError(f"nonfinite/negative distance geometry at CSV line {line_no}")
            delta = c - q
            distance = float(np.linalg.norm(delta))
            if abs(distance - dist_declared) > max(distance_tolerance_m, 1e-7 * max(distance, dist_declared)):
                raise ValueError(f"closest-point distance mismatch at CSV line {line_no}")
            if classification == "inside":
                if nearest_bone != bone:
                    raise ValueError(f"nearest bone does not match bank at CSV line {line_no}")
                if distance <= distance_tolerance_m:
                    unresolved.append({"line": line_no, "node": local, "body": body, "bone": bone,
                                       "classification": classification, "reason": "zero_or_unresolved_direction"})
                else:
                    rows.append({"local_node": local, "body_index": body, "bone_stable_id": bone,
                                 "nearest_triangle": tri, "query_local": q, "closest_local": c,
                                 "outward_local": delta / distance, "inside_depth_m": distance,
                                 "classification": classification})
            elif classification in {"ambiguous", "invalid"}:
                unresolved.append({"line": line_no, "node": local, "body": body, "bone": bone,
                                   "classification": classification, "reason": "containment_not_certified"})
        return rows, unresolved, {"csv_rows": len(seen), "inside_constraints": len(rows),
                                  "unresolved_rows": len(unresolved), "distinct_pair_keys": len(seen)}
    finally:
        if close:
            f.close()


def _quat_rotation_xyzw(quat) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.all(np.isfinite(q)) or not math.isfinite(n) or n <= 0:
        raise ValueError("invalid current-pose quaternion")
    x, y, z, w = q / n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def transform_inside_constraints(rows, node_points_world_m: np.ndarray,
                                 poses_by_body: Mapping[int, Mapping]) -> dict[str, np.ndarray]:
    """Transform all per-bone inside rows to world planes, preserving duplicates by bone."""
    X = np.asarray(node_points_world_m, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3 or not np.all(np.isfinite(X)):
        raise ValueError("node world positions must be finite [N,3]")
    node, body, bone, tri, depth, cpw, nww = [], [], [], [], [], [], []
    for row in rows:
        i = int(row["local_node"]); b = int(row["body_index"])
        if i < 0 or i >= len(X) or b not in poses_by_body:
            raise ValueError("inside row has invalid node or missing pose")
        current = poses_by_body[b]["current"]
        R = _quat_rotation_xyzw(current["orientation_world_xyzw"])
        delta_local = np.asarray(row["closest_local"], dtype=np.float64) - np.asarray(row["query_local"], dtype=np.float64)
        normal_local = np.asarray(row["outward_local"], dtype=np.float64)
        normal_world = R @ normal_local
        length = float(np.linalg.norm(normal_world))
        if not np.all(np.isfinite(normal_world)) or abs(length - 1.0) > 1e-8:
            raise ValueError("transformed normal is invalid")
        p = X[i]
        cp = p + R @ delta_local
        if abs(float(np.dot(p-cp, normal_world)) + float(row["inside_depth_m"])) > 3e-8:
            raise ValueError("world plane reconstruction failed")
        node.append(i); body.append(b); bone.append(int(row["bone_stable_id"]))
        tri.append(-1 if row["nearest_triangle"] is None else int(row["nearest_triangle"]))
        depth.append(float(row["inside_depth_m"])); cpw.append(cp); nww.append(normal_world)
    return {"constraint_local_node": np.asarray(node, dtype=np.int64),
            "constraint_body_index": np.asarray(body, dtype=np.int64),
            "constraint_bone_stable_id": np.asarray(bone, dtype=np.int64),
            "constraint_nearest_triangle": np.asarray(tri, dtype=np.int64),
            "constraint_inside_depth_m": np.asarray(depth, dtype=np.float64),
            "constraint_closest_world_m": np.asarray(cpw, dtype=np.float64).reshape(-1,3),
            "constraint_outward_normal_world": np.asarray(nww, dtype=np.float64).reshape(-1,3)}


def normalized_tet_quality_jacobian(vertices: np.ndarray):
    """Normalized signed determinant and vertex gradient for [N,4,3] tetrahedra."""
    V = np.asarray(vertices, dtype=np.float64)
    if V.ndim != 3 or V.shape[1:] != (4, 3) or not np.all(np.isfinite(V)):
        raise ValueError("tetrahedron vertices must be finite [N,4,3]")
    u = V[:, 1] - V[:, 0]
    v = V[:, 2] - V[:, 0]
    w = V[:, 3] - V[:, 0]
    det = np.einsum("ij,ij->i", u, np.cross(v, w))
    gd1 = np.cross(v, w)
    gd2 = np.cross(w, u)
    gd3 = np.cross(u, v)
    gd0 = -(gd1 + gd2 + gd3)
    gd = np.stack((gd0, gd1, gd2, gd3), axis=1)
    pairs = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]], dtype=np.int64)
    ev = np.stack([V[:, i] - V[:, j] for i,j in pairs], axis=1)
    e2 = np.einsum("nij,nij->ni", ev, ev)
    em = np.argmax(e2, axis=1)
    L = np.sqrt(np.maximum(e2[np.arange(len(V)), em], 1e-30))
    q = det / L**3
    G = gd / L[:,None,None]**3
    ep = V[np.arange(len(V)), pairs[em,0]] - V[np.arange(len(V)), pairs[em,1]]
    cc = (-3.0 * det / L**5)[:,None]
    np.add.at(G, (np.arange(len(V)), pairs[em,0]), cc * ep)
    np.add.at(G, (np.arange(len(V)), pairs[em,1]), -cc * ep)
    return q, G


def projected_layer_thickness_mm(points_outer_m: np.ndarray, points_inner_m: np.ndarray,
                                 reference_normals: np.ndarray) -> np.ndarray:
    """Project O-I/I-D meter vectors on reference normals and return millimetres."""
    a = np.asarray(points_outer_m, dtype=np.float64)
    b = np.asarray(points_inner_m, dtype=np.float64)
    n = np.asarray(reference_normals, dtype=np.float64)
    if a.shape != b.shape or a.shape != n.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("layer projection arrays must all have shape [N,3]")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(n))):
        raise ValueError("nonfinite layer projection input")
    nl = np.linalg.norm(n, axis=1)
    if np.any(nl <= 0) or np.max(np.abs(nl - 1.0), initial=0.0) > 1e-6:
        raise ValueError("reference normals must be unit length")
    return 1000.0 * np.einsum("ij,ij->i", a - b, n)


def layer_floor_mm(thickness_m: np.ndarray) -> np.ndarray:
    thickness = np.asarray(thickness_m, dtype=np.float64)
    if not np.all(np.isfinite(thickness)) or np.any(thickness < 0):
        raise ValueError("invalid reference layer thickness")
    return np.minimum(0.2 * thickness * 1000.0, 0.10)
