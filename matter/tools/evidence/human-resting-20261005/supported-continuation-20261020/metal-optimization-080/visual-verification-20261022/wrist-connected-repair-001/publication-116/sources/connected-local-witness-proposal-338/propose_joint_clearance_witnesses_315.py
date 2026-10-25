#!/usr/bin/env python3
"""Sparse joint local proposal for self-pair and clearance witnesses,
with optional fixed-prefix nodes and an independently pinned fixed-037 area reference.

Construction-only: reuses the frozen self-separation and area linearization
owners, keeps the candidate mesh/topology fixed, and emits a proposal only.
It does not run exact global self, bone, native, or physical admission gates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
import tempfile
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
DEFAULT_PAIR_OWNER = ROOT / "barycentric-connected-clearance-170/connected-local-continuation-189-run001/round-005/proposal/round-001/fit/self_separation_026.py"
DEFAULT_AREA_OWNER = ROOT / "barycentric-connected-clearance-170/connected-local-continuation-189-run001/round-005/proposal/round-001/fit/area_scp_support_023.py"
DEFAULT_BARY_OWNER = ROOT / "publication-091/canonical/matter/tools/human_skin_fat_barycentric_constraints.py"
DEFAULT_FIXED_BOUNDARY_OWNER = ROOT / "connected-fixed-boundary-313/fixed_bone_triangle_rows_313.py"
DEFAULT_FIXED_BOUNDARY_EMITTER = ROOT / "D/connected-fixed-boundary-314/emit_introduced_skin_bone_barriers_314.py"

TRUST_MM = 1.0
SOURCE_CAP_MM = 12.0
AREA_FLOOR = 0.1
PAIR_MARGIN_FRACTION = 0.25


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import owner {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def interleaved_coordinate_columns(node_rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(node_rows, dtype=np.int64)
    return np.column_stack((3 * rows, 3 * rows + 1, 3 * rows + 2)).reshape(-1)




def build_edge_gradient_constraints(points_mm: np.ndarray, faces: np.ndarray,
                                   support: np.ndarray, max_component_gradient: float
                                   ) -> tuple[sparse.csr_matrix, np.ndarray, dict]:
    """Bound componentwise displacement differences across active mesh edges.

    For each unique edge (i,j) touching support, with nonsupport nodes fixed,
    impose |d_i[k]-d_j[k]| <= max_component_gradient * edge_length for xyz.
    This is an optional construction regularizer; it is not a self-collision
    certificate and does not alter any exact acceptance gate.
    """
    p = np.asarray(points_mm, dtype=np.float64)
    f_raw = np.asarray(faces)
    sup_raw = np.asarray(support)
    cap = float(max_component_gradient)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("points_mm must be finite with shape (N,3)")
    if f_raw.ndim != 2 or f_raw.shape[1] != 3 or not np.issubdtype(f_raw.dtype, np.integer):
        raise ValueError("faces must be an integer (F,3) array")
    f = f_raw.astype(np.int64, copy=False)
    if np.any(f < 0) or np.any(f >= len(p)) or np.any((f[:, 0] == f[:, 1]) |
            (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])):
        raise ValueError("faces contain invalid or repeated vertex IDs")
    if sup_raw.ndim != 1 or not np.issubdtype(sup_raw.dtype, np.integer):
        raise ValueError("support must be a one-dimensional integer array")
    sup = sup_raw.astype(np.int64, copy=False)
    if len(np.unique(sup)) != len(sup) or np.any(sup < 0) or np.any(sup >= len(p)):
        raise ValueError("support contains duplicate or out-of-range rows")
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("max_component_gradient must be finite and positive")

    if len(f):
        edges = np.unique(np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]), axis=0), axis=1), axis=0)
    else:
        edges = np.empty((0, 2), dtype=np.int64)
    active = edges[np.isin(edges, sup).any(axis=1)]
    if len(active) == 0:
        return sparse.csr_matrix((0, 3 * len(sup))), np.empty(0, dtype=np.float64), {
            "enabled": True, "max_component_gradient": cap, "unique_mesh_edges": int(len(edges)),
            "active_edges": 0, "inequality_rows": 0,
            "semantics": "componentwise |d_i-d_j| <= cap * reference_edge_length; nonsupport displacement is zero",
        }
    lengths = np.linalg.norm(p[active[:, 0]] - p[active[:, 1]], axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        bad = np.flatnonzero((~np.isfinite(lengths)) | (lengths <= 0))
        raise ValueError(f"active mesh contains nonpositive or nonfinite edge lengths at rows {bad[:8].tolist()}")
    local = {int(row): col for col, row in enumerate(sup.tolist())}
    rows, cols, vals = [], [], []
    rhs = []
    row = 0
    for edge, length in zip(active, lengths):
        endpoints = [(int(v), local[int(v)]) for v in edge if int(v) in local]
        coeff_norm = np.sqrt(float(len(endpoints)))
        bound = cap * float(length) / coeff_norm
        for axis in range(3):
            for sign in (1.0, -1.0):
                for vertex, local_row in endpoints:
                    rows.append(row)
                    cols.append(3 * local_row + axis)
                    vals.append(sign * (1.0 if vertex == int(edge[0]) else -1.0) / coeff_norm)
                rhs.append(-bound)
                row += 1
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(row, 3 * len(sup))).tocsr()
    return A, np.asarray(rhs, dtype=np.float64), {
        "enabled": True, "max_component_gradient": cap, "unique_mesh_edges": int(len(edges)),
        "active_edges": int(len(active)), "inequality_rows": int(row),
        "semantics": "componentwise |d_i-d_j| <= cap * reference_edge_length; nonsupport displacement is zero",
        "maximum_reference_edge_length_mm": float(lengths.max(initial=0.0)),
        "minimum_reference_edge_length_mm": float(lengths.min(initial=np.inf)),
    }

def verify_area_linearization_layout(area_owner) -> dict:
    """Directional finite-difference check of the real area-row owner."""
    pts = np.array([[0.2, -0.1, 0.3], [1.4, 0.2, 0.1], [-0.1, 1.1, 0.4]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    cross = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    area = float(np.linalg.norm(cross))
    normal = cross / area
    G, _, _, _ = area_owner.area_linearization(
        pts, faces, np.array([0]), normal[None, :], np.array([area]), minimum_ratio=0.1
    )
    order = np.array([2, 0, 1], dtype=np.int64)
    cols = interleaved_coordinate_columns(order)
    delta = np.array([[0.03, -0.02, 0.01], [-0.04, 0.01, 0.02], [0.02, 0.05, -0.03]])
    predicted = float(np.asarray(G[:, cols] @ delta[order].reshape(-1)).reshape(-1)[0])
    eps = 1e-7
    shifted = pts + eps * delta
    actual = (
        float(np.dot(normal, np.cross(shifted[1] - shifted[0], shifted[2] - shifted[0]))) - area
    ) / eps
    if not np.isclose(predicted, actual, rtol=2e-7, atol=2e-8):
        raise RuntimeError(f"area layout FD mismatch: {predicted} vs {actual}")
    return {
        "layout": "node-major xyz",
        "support_order": order.tolist(),
        "predicted_directional_derivative": predicted,
        "finite_difference_directional_derivative": actual,
        "relative_error": float(abs(predicted - actual) / max(abs(actual), 1e-30)),
    }


def compose_source_reference(candidate, source_base_path: Path | None) -> tuple[np.ndarray, list[dict]]:
    """Read an already composed origin or compose the candidate's three CSR levels."""
    n = len(candidate["points_world_m"])
    key = "local_selfpair_cluster_composed_source_reference_m"
    if key in candidate.files:
        ref = np.asarray(candidate[key], dtype=np.float64)
        if ref.shape != (n, 3) or not np.isfinite(ref).all():
            raise ValueError(f"{key} must be a finite (N,3) array")
        return ref, [{"kind": "candidate_embedded_composed_reference", "key": key}]

    if source_base_path is None:
        raise ValueError(
            "candidate lacks a composed source reference; pass --source-base-npz "
            "to compose its three pinned CSR levels"
        )
    source_base = np.load(source_base_path, allow_pickle=False)
    prefixes = ("", "prior_input_006__", "prior_run028__")
    P = sparse.eye(n, format="csr", dtype=np.float64)
    level_receipts: list[dict] = []
    for level, prefix in enumerate(prefixes):
        ip = np.asarray(candidate[prefix + "source_prolongation_indptr"], dtype=np.int64)
        ix = np.asarray(candidate[prefix + "source_prolongation_node_rows"], dtype=np.int64)
        w = np.asarray(candidate[prefix + "source_prolongation_weights"], dtype=np.float64)
        nr = len(ip) - 1
        if level + 1 < len(prefixes):
            nc = len(candidate[prefixes[level + 1] + "source_prolongation_indptr"]) - 1
        else:
            nc = len(source_base["points_before_joint_self_pair_repair_m"])
        if nr != P.shape[1] or ip.ndim != 1 or ip[0] != 0 or ip[-1] != len(ix) or len(ix) != len(w):
            raise ValueError(f"invalid source CSR dimensions at level {level}")
        if np.any(np.diff(ip) <= 0) or np.any(ix < 0) or np.any(ix >= nc) or not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError(f"invalid source CSR entries at level {level}")
        M = sparse.csr_matrix((w, ix, ip), shape=(nr, nc))
        sum_error = float(np.max(np.abs(np.asarray(M.sum(axis=1)).ravel() - 1.0)))
        if sum_error > 2e-10:
            raise ValueError(f"source CSR level {level} is not affine: {sum_error}")
        P = (P @ M).tocsr()
        level_receipts.append({
            "kind": "csr_level", "level": level, "prefix": prefix, "rows": nr,
            "columns": nc, "nnz": int(M.nnz), "max_row_sum_error": sum_error,
        })
    raw = np.asarray(source_base["points_before_joint_self_pair_repair_m"], dtype=np.float64)
    if P.shape != (n, len(raw)) or raw.shape != (P.shape[1], 3):
        raise ValueError("composed source CSR does not map candidate to source-base rows")
    ref = np.asarray(P @ raw, dtype=np.float64)
    if not np.isfinite(ref).all():
        raise ValueError("composed source reference contains nonfinite values")
    level_receipts.append({
        "kind": "composed_source_base", "path": str(source_base_path),
        "sha256": sha(source_base_path), "shape": list(raw.shape),
        "composed_shape": list(P.shape), "composed_nnz": int(P.nnz),
    })
    return ref, level_receipts


def source_safe_trust_bounds(points_mm: np.ndarray, source_mm: np.ndarray,
                             support: np.ndarray, anchors: np.ndarray,
                             trust_mm: float, source_cap_mm: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Construct a conservative component box inside both trust and source balls.

    For each row, the allowed displacement box radius is no greater than
    (source_cap - current source distance)/sqrt(3); triangle inequality then
    guarantees every point in the box remains inside the source-cap sphere.
    """
    p = np.asarray(points_mm, dtype=np.float64)
    r = np.asarray(source_mm, dtype=np.float64)
    support = np.asarray(support, dtype=np.int64)
    anchors = np.asarray(anchors, dtype=np.int64)
    if p.shape != r.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("point/source arrays must share shape (N,3)")
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError("point/source arrays contain nonfinite values")
    if not np.isfinite([trust_mm, source_cap_mm]).all() or trust_mm <= 0 or source_cap_mm <= 0:
        raise ValueError("trust and source cap must be finite and positive")
    source_distance = np.linalg.norm(p - r, axis=1)
    if float(source_distance.max(initial=0.0)) > source_cap_mm + 1e-7:
        raise ValueError("base candidate already exceeds the declared source cap")
    remaining = source_cap_mm - source_distance[support]
    if np.any(remaining < -1e-7):
        raise ValueError("support node exceeds source cap")
    component_bound = np.minimum(trust_mm, np.maximum(0.0, remaining)) / np.sqrt(3.0)
    lower = np.repeat(-component_bound, 3)
    upper = np.repeat(component_bound, 3)
    local = {int(row): i for i, row in enumerate(support)}
    for row in anchors:
        li = local.get(int(row))
        if li is not None:
            lower[3 * li:3 * li + 3] = 0.0
            upper[3 * li:3 * li + 3] = 0.0
    return lower, upper, {
        "maximum_base_source_displacement_mm": float(source_distance.max(initial=0.0)),
        "minimum_remaining_source_cap_mm_on_support": float(remaining.min(initial=source_cap_mm)),
        "maximum_component_bound_mm": float(component_bound.max(initial=0.0)),
        "trust_norm_guarantee_mm": float(trust_mm),
        "source_cap_norm_guarantee_mm": float(source_cap_mm),
        "bound_semantics": "per-node component cube with radius <= min(trust, source_cap - base_source_distance)/sqrt(3)",
    }


def array_sha256(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def validate_fixed_boundary_payload(arrays: dict, report: dict, *, candidate_sha256: str,
                                    candidate_points: np.ndarray, candidate_faces: np.ndarray,
                                    npz_path: Path) -> tuple[list[dict], dict]:
    """Validate a 314 typed fixed-triangle export against the exact base mesh."""
    if report.get("status") != "COMPLETE_TYPED_INTRODUCED_SKIN_BONE_BARRIER_EXPORT":
        raise ValueError("fixed-boundary report is not a complete 314 export")
    if report.get("base_candidate_sha256") != candidate_sha256:
        raise ValueError("fixed-boundary report candidate SHA differs from the fitter base")
    points = np.asarray(candidate_points)
    faces = np.asarray(candidate_faces)
    if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("fixed-boundary base points must be emitted Float32 (N,3)")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise ValueError("fixed-boundary base faces must be integer (F,3)")
    if (int(report.get("point_count", -1)) != len(points)
            or int(report.get("face_count", -1)) != len(faces)
            or report.get("candidate_points_world_m_array_sha256") != array_sha256(points)
            or report.get("skin_faces_array_sha256") != array_sha256(faces)):
        raise ValueError("fixed-boundary report topology/point-array pins differ from fitter base")
    if report.get("fixed_triangles_units") != "mm" or report.get("fixed_triangles_frame") != "world":
        raise ValueError("fixed-boundary triangles must be explicitly world-frame millimetres")
    if (int(report.get("bank_count", -1)) != 185
            or int(report.get("expected_bank_count", -1)) != 185):
        raise ValueError("fixed-boundary report does not pin the complete 185-bank source set")
    if report.get("geometry_or_native_admission") is not False:
        raise ValueError("fixed-boundary export must remain construction-only")
    if report.get("output_npz_sha256") != sha(npz_path):
        raise ValueError("fixed-boundary report does not pin the supplied NPZ bytes")
    if report.get("output_npz") != npz_path.name:
        raise ValueError("fixed-boundary report output filename differs from supplied NPZ")
    if int(report.get("introduced_pair_count", -1)) < 0:
        raise ValueError("fixed-boundary report lacks introduced pair count")
    inputs = report.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("fixed-boundary report lacks 314 base/trial audit input pins")
    base_inputs = [row for row in inputs if isinstance(row, dict) and row.get("role") == "base"]
    if len(base_inputs) != 1 or base_inputs[0].get("candidate_sha256") != candidate_sha256:
        raise ValueError("fixed-boundary report does not bind exactly one matching base audit")

    required = ("skin_face_rows", "bank_body", "stable_bone_id",
                "bank_triangle_stable_id", "fixed_triangles_world_mm")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"fixed-boundary NPZ missing arrays: {missing}")
    integer_names = required[:4]
    values = {}
    for name in integer_names:
        raw = np.asarray(arrays[name])
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise ValueError(f"fixed-boundary {name} must be a one-dimensional integer array")
        values[name] = raw.astype(np.int64, copy=False)
    count = len(values["skin_face_rows"])
    if any(len(values[name]) != count for name in integer_names[1:]):
        raise ValueError("fixed-boundary typed ID arrays have different row counts")
    raw_triangles = np.asarray(arrays["fixed_triangles_world_mm"])
    if raw_triangles.dtype != np.dtype(np.float64):
        raise ValueError("fixed-boundary triangles must retain their promoted Float64 world coordinates")
    triangles = raw_triangles.astype(np.float64, copy=False)
    if triangles.shape != (count, 3, 3) or not np.isfinite(triangles).all():
        raise ValueError("fixed-boundary world triangles must be finite (K,3,3) millimetre arrays")
    for name in integer_names[1:]:
        if np.any(values[name] < 0):
            raise ValueError(f"fixed-boundary {name} contains a negative typed ID")
    if np.any(values["skin_face_rows"] < 0) or np.any(values["skin_face_rows"] >= len(faces)):
        raise ValueError("fixed-boundary skin face row is outside base topology")
    typed = list(zip(values["skin_face_rows"].tolist(), values["bank_body"].tolist(),
                     values["stable_bone_id"].tolist(), values["bank_triangle_stable_id"].tolist()))
    if len(set(typed)) != count:
        raise ValueError("fixed-boundary NPZ contains duplicate typed skin/bone pairs")
    typed_report = report.get("typed_pairs")
    if not isinstance(typed_report, list) or len(typed_report) != count:
        raise ValueError("fixed-boundary report typed-pair list does not match NPZ")
    report_typed = []
    for record in typed_report:
        if not isinstance(record, dict):
            raise ValueError("malformed fixed-boundary typed-pair report record")
        fields = ("skin_face_row", "bank_body", "stable_bone_id", "bank_triangle_stable_id")
        parsed = []
        for name in fields:
            value = record.get(name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"fixed-boundary typed report field {name} is not an integer")
            parsed.append(int(value))
        report_typed.append(tuple(parsed))
    expected_report_order = sorted(typed, key=lambda item: (item[1], item[2], item[0], item[3]))
    if report_typed != expected_report_order:
        raise ValueError("fixed-boundary report typed-pair order/IDs differ from NPZ arrays")
    if int(report.get("introduced_pair_count", -1)) != count:
        raise ValueError("fixed-boundary introduced_pair_count differs from typed-pair payload")

    records = []
    for i, (face, body, bone, triangle_id) in enumerate(typed):
        records.append({
            "skin_face_row": face,
            "bank_body": body,
            "stable_bone_id": bone,
            "bank_triangle_stable_id": triangle_id,
            "fixed_triangle_world_mm": triangles[i],
        })
    receipt = {
        "enabled": True,
        "report_status": report["status"],
        "npz_path": str(npz_path),
        "npz_sha256": sha(npz_path),
        "report_sha256": None,
        "base_candidate_sha256": candidate_sha256,
        "point_count": int(len(points)),
        "face_count": int(len(faces)),
        "base_faces_array_sha256": array_sha256(faces),
        "typed_pair_count": int(count),
        "units": "world-frame millimetres",
        "geometry_or_native_admission": False,
    }
    return records, receipt


def build_fixed_boundary_barriers(points_mm: np.ndarray, faces: np.ndarray,
                                  records: list[dict], pair_owner, fixed_owner,
                                  margin_fraction: float) -> tuple[list[dict], list[dict]]:
    """Create skin-only barrier rows and verify their typed ranges/support."""
    build = getattr(fixed_owner, "build_fixed_triangle_rows", None)
    if not callable(build):
        raise ValueError("fixed-boundary owner lacks build_fixed_triangle_rows")
    rows, typed_records = build(points_mm, faces, records, pair_owner,
                                margin_fraction=float(margin_fraction))
    if not isinstance(rows, (list, tuple)) or not isinstance(typed_records, (list, tuple)):
        raise ValueError("fixed-boundary owner returned malformed row/record collections")
    expected = sorted((int(r["skin_face_row"]), int(r["bank_body"]), int(r["stable_bone_id"]),
                       int(r["bank_triangle_stable_id"])) for r in records)
    got = [(int(r.get("skin_face_row", -1)), int(r.get("bank_body", -1)),
            int(r.get("stable_bone_id", -1)), int(r.get("bank_triangle_stable_id", -1)))
           for r in typed_records]
    if got != expected:
        raise ValueError("fixed-boundary owner typed records differ from validated NPZ order")
    next_row = 0
    for record in typed_records:
        start, stop = record.get("rows_start"), record.get("rows_stop")
        if (isinstance(start, (bool, np.bool_)) or isinstance(stop, (bool, np.bool_))
                or not isinstance(start, (int, np.integer)) or not isinstance(stop, (int, np.integer))
                or int(start) != next_row or int(stop) < int(start) or int(stop) > len(rows)):
            raise ValueError("fixed-boundary row ranges are malformed or noncontiguous")
        face_row = int(record["skin_face_row"])
        allowed_nodes = set(map(int, np.asarray(faces[face_row], dtype=np.int64)))
        for row in rows[int(start):int(stop)]:
            if not isinstance(row, dict):
                raise ValueError("fixed-boundary owner emitted a non-dictionary barrier row")
            nodes = np.asarray(row.get("nodes"))
            coeff = np.asarray(row.get("coefficients"), dtype=np.float64)
            lower = row.get("lower")
            if (nodes.ndim != 1 or nodes.dtype.kind not in "iu" or coeff.shape != (len(nodes), 3)
                    or not np.isfinite(coeff).all() or isinstance(lower, (bool, np.bool_))
                    or not np.isscalar(lower) or not np.isfinite(float(lower))):
                raise ValueError("fixed-boundary owner emitted malformed coefficients/lower bound")
            if float(lower) > 2e-8:
                raise ValueError("fixed-boundary skin-only row is infeasible at the pinned zero-displacement base")
            node_ids = list(map(int, nodes))
            if (not node_ids or len(set(node_ids)) != len(node_ids)
                    or any(node not in allowed_nodes for node in node_ids)):
                raise ValueError("fixed-boundary barrier references a node outside its typed skin face")
            if not np.any(np.linalg.norm(coeff, axis=1) > 1e-14):
                raise ValueError("fixed-boundary barrier row has no skin displacement coefficient")
        next_row = int(stop)
    if next_row != len(rows):
        raise ValueError("fixed-boundary typed row ranges do not cover returned barrier rows")
    return list(rows), list(typed_records)


def combine_barrier_rows(self_rows, fixed_boundary_rows=()) -> list[dict]:
    """Keep self rows first, then fixed-triangle rows, before area/gradient blocks."""
    return list(self_rows) + list(fixed_boundary_rows)


def load_area_reference(path: Path, expected_faces: np.ndarray,
                        points_key: str = "points_world_m",
                        faces_key: str = "source_O_faces") -> tuple[np.ndarray, dict]:
    """Load the fixed accepted-037 geometry used only for surface area gates."""
    with np.load(path, allow_pickle=False) as z:
        if points_key not in z.files or faces_key not in z.files:
            raise ValueError(f"area reference must contain {points_key!r} and {faces_key!r}")
        points = np.asarray(z[points_key], dtype=np.float64)
        faces = np.asarray(z[faces_key], dtype=np.int64)
    expected = np.asarray(expected_faces, dtype=np.int64)
    if points.shape[0] == 0 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("fixed area-reference points must have shape (N,3)")
    if not np.isfinite(points).all():
        raise ValueError("fixed area-reference points contain nonfinite values")
    if not np.array_equal(faces, expected):
        raise ValueError("fixed area-reference faces do not exactly match candidate face rows/order")
    if np.any(faces < 0) or np.any(faces >= len(points)):
        raise ValueError("fixed area-reference faces index outside its point domain")
    receipt = {
        "path": str(path), "sha256": sha(path), "points_key": points_key,
        "faces_key": faces_key, "point_shape": list(points.shape),
        "face_shape": list(faces.shape), "points_array_sha256": array_sha256(points),
        "faces_array_sha256": array_sha256(faces),
        "semantics": "accepted-037 fixed reference geometry for area constraints only",
    }
    return points, receipt


def fixed_reference_area_requirements(current_mm: np.ndarray, reference_mm: np.ndarray,
                                      faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return current normals, current/reference areas, and local area target."""
    f = np.asarray(faces, dtype=np.int64)
    x = np.asarray(current_mm, dtype=np.float64)
    r = np.asarray(reference_mm, dtype=np.float64)
    if x.shape != r.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("current and fixed-reference points must share shape (N,3)")
    cross_current = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
    cross_reference = np.cross(r[f[:, 1]] - r[f[:, 0]], r[f[:, 2]] - r[f[:, 0]])
    current_area = np.linalg.norm(cross_current, axis=1)
    reference_area = np.linalg.norm(cross_reference, axis=1)
    if (not np.isfinite(current_area).all() or not np.isfinite(reference_area).all()
            or np.any(current_area <= 0.0) or np.any(reference_area <= 0.0)):
        raise ValueError("current or fixed-037 reference contains degenerate/nonfinite faces")
    normals = cross_current / current_area[:, None]
    required = np.maximum(current_area, reference_area)
    return normals, current_area, reference_area, required


def full_f32_area_gate(trial_m: np.ndarray, current_mm: np.ndarray,
                       reference_mm: np.ndarray, faces: np.ndarray,
                       floor: float) -> dict:
    """Evaluate current orientation and fixed-037 magnitude gates on every face."""
    f = np.asarray(faces, dtype=np.int64)
    x = np.asarray(trial_m, dtype=np.float32).astype(np.float64) * 1000.0
    base = np.asarray(current_mm, dtype=np.float64)
    ref = np.asarray(reference_mm, dtype=np.float64)
    c0 = np.cross(base[f[:, 1]] - base[f[:, 0]], base[f[:, 2]] - base[f[:, 0]])
    c1 = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
    area0 = np.linalg.norm(c0, axis=1)
    arearef = np.linalg.norm(np.cross(ref[f[:, 1]] - ref[f[:, 0]],
                                     ref[f[:, 2]] - ref[f[:, 0]]), axis=1)
    if np.any(area0 <= 0) or np.any(arearef <= 0) or not np.isfinite(c1).all():
        raise ValueError("full-mesh F32 area gate encountered degenerate/nonfinite geometry")
    oriented = np.einsum("ij,ij->i", c1, c0) / (area0 * area0)
    reference_magnitude = np.linalg.norm(c1, axis=1) / arearef
    # The final F32 area acceptance boundary is the declared floor itself.
    # Linearized proxy rows may fail independently and remain diagnostics only.
    bad_oriented = np.flatnonzero(oriented < floor)
    bad_reference = np.flatnonzero(reference_magnitude < floor)
    return {
        "faces_checked": int(len(f)),
        "minimum_current_oriented_ratio": float(oriented.min(initial=np.inf)),
        "minimum_fixed_037_reference_magnitude_ratio": float(reference_magnitude.min(initial=np.inf)),
        "bad_current_oriented_face_count": int(len(bad_oriented)),
        "bad_current_oriented_face_rows_first_64": bad_oriented[:64].tolist(),
        "bad_current_oriented_face_rows_truncated": bool(len(bad_oriented) > 64),
        "bad_fixed_037_reference_face_count": int(len(bad_reference)),
        "bad_fixed_037_reference_face_rows_first_64": bad_reference[:64].tolist(),
        "bad_fixed_037_reference_face_rows_truncated": bool(len(bad_reference) > 64),
        "area_floor": float(floor),
        "pass": bool(len(bad_oriented) == 0 and len(bad_reference) == 0),
    }


def f32_emission_gate(full_area: dict, step_norms_mm: np.ndarray,
                      source_norms_mm: np.ndarray, trust_mm: float,
                      source_cap_mm: float, outside_unchanged: bool,
                      anchors_unchanged: bool) -> dict:
    """Gate only emitted geometry/cap invariants; exact self audit stays pending."""
    step = np.asarray(step_norms_mm, dtype=np.float64)
    source = np.asarray(source_norms_mm, dtype=np.float64)
    if not np.isfinite(step).all() or not np.isfinite(source).all():
        raise ValueError("emitted F32 step/source displacement contains nonfinite values")
    limits = {
        "full_mesh_strict_area_pass": bool(full_area.get("pass", False)),
        "trust_pass": bool(float(step.max(initial=0.0)) <= trust_mm + 1e-6),
        "source_cap_pass": bool(float(source.max(initial=0.0)) <= source_cap_mm + 1e-6),
        "outside_support_bitwise_unchanged": bool(outside_unchanged),
        "anchors_bitwise_unchanged": bool(anchors_unchanged),
        "maximum_node_step_mm": float(step.max(initial=0.0)),
        "maximum_source_displacement_mm": float(source.max(initial=0.0)),
        "trust_mm": float(trust_mm),
        "source_cap_mm": float(source_cap_mm),
    }
    limits["pass"] = bool(all(limits[k] for k in (
        "full_mesh_strict_area_pass", "trust_pass", "source_cap_pass",
        "outside_support_bitwise_unchanged", "anchors_bitwise_unchanged",
    )))
    limits["exact_global_self_audit"] = "NOT_RUN_REQUIRED"
    limits["native_admission"] = False
    limits["accepted_physical_steps"] = 0
    return limits


def emit_raw_lp_points(points: np.ndarray, points_mm: np.ndarray,
                       support: np.ndarray, delta_mm: np.ndarray,
                       anchors: np.ndarray) -> np.ndarray:
    """Emit the unbacktracked LP delta once to the F32 point domain."""
    base = np.asarray(points, dtype=np.float32)
    x_mm = np.asarray(points_mm, dtype=np.float64)
    rows = np.asarray(support, dtype=np.int64)
    d = np.asarray(delta_mm, dtype=np.float64)
    fixed = np.asarray(anchors, dtype=np.int64)
    if x_mm.shape != base.shape or d.shape != (len(rows), 3):
        raise ValueError("raw LP emission inputs have inconsistent shapes")
    if (not np.isfinite(x_mm).all() or not np.isfinite(d).all()
            or np.any(rows < 0) or np.any(rows >= len(base))
            or np.any(fixed < 0) or np.any(fixed >= len(base))):
        raise ValueError("raw LP emission has nonfinite values or invalid row IDs")
    emitted = base.copy()
    emitted[rows] = ((x_mm[rows] + d) / 1000.0).astype(np.float32)
    emitted[fixed] = base[fixed]
    if not np.isfinite(emitted).all():
        raise ValueError("raw LP F32 emission is nonfinite")
    return emitted


def build_raw_lp_archive(base_archive, raw_points: np.ndarray, support: np.ndarray,
                         delta_mm: np.ndarray, input_hashes: dict) -> dict:
    """Preserve the solver result and its emitted F32 points before backtracking."""
    keys = base_archive.files if hasattr(base_archive, "files") else base_archive.keys()
    out = {key: np.asarray(base_archive[key]) for key in keys}
    old_points = np.asarray(base_archive["points_world_m"], dtype=np.float32)
    raw = np.asarray(raw_points, dtype=np.float32)
    rows = np.asarray(support, dtype=np.int64)
    delta = np.asarray(delta_mm, dtype=np.float64)
    if raw.shape != old_points.shape or delta.shape != (len(rows), 3):
        raise ValueError("raw LP archive dimensions do not match the base topology")
    if not np.isfinite(raw).all() or not np.isfinite(delta).all():
        raise ValueError("raw LP archive contains nonfinite values")
    out["points_world_m"] = raw
    out["local_witness_cluster_222_lp_delta_mm_f64"] = delta
    out["local_witness_cluster_222_lp_support_rows"] = rows
    out["local_witness_cluster_222_lp_emitted_delta_m_f32"] = raw - old_points
    out["local_witness_cluster_222_lp_input_hashes_json"] = np.asarray([json.dumps(input_hashes, sort_keys=True)])
    out["local_witness_cluster_222_lp_qualified"] = np.asarray([False])
    out["local_witness_cluster_222_lp_semantics"] = np.asarray([
        "unbacktracked LP point proposal; diagnostic only; not an admitted candidate"
    ])
    return out


def absolute_area_rhs_from_incremental(Gq: sparse.csr_matrix, rhs_q: np.ndarray,
                                         support: np.ndarray, delta_current: np.ndarray) -> np.ndarray:
    """Convert an area inequality at q into an absolute base-relative delta row.

    Gq @ increment >= rhs_q with Xq = X0 + delta_current becomes
    Gq @ delta_absolute >= rhs_q + Gq @ delta_current.
    """
    G = sparse.csr_matrix(Gq, dtype=np.float64)
    rhs = np.asarray(rhs_q, dtype=np.float64).reshape(-1)
    dc = np.asarray(delta_current, dtype=np.float64).reshape(-1)
    cols = interleaved_coordinate_columns(np.asarray(support, dtype=np.int64))
    if G.shape[0] != rhs.size or G.shape[1] <= int(cols.max(initial=-1)) or dc.shape != cols.shape:
        raise ValueError("incremental area RHS dimensions do not match support coordinates")
    if not np.isfinite(G.data).all() or not np.isfinite(rhs).all() or not np.isfinite(dc).all():
        raise ValueError("incremental area rows contain nonfinite values")
    return rhs + np.asarray(G[:, cols] @ dc, dtype=np.float64).reshape(-1)


def solve_sparse_l1(A: sparse.csr_matrix, lower_rows: np.ndarray, target: np.ndarray,
                    lower: np.ndarray, upper: np.ndarray, time_limit_s: float) -> dict:
    """HiGHS LP: minimize coordinatewise L1 distance from target under A d >= b."""
    A = sparse.csr_matrix(A, dtype=np.float64)
    b = np.asarray(lower_rows, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    lo = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper, dtype=np.float64).reshape(-1)
    n = t.size
    if A.shape[1] != n or b.shape != (A.shape[0],) or lo.shape != t.shape or hi.shape != t.shape:
        raise ValueError("LP input shapes do not align")
    if not np.isfinite(A.data).all() or not np.isfinite(b).all() or not np.isfinite(t).all():
        raise ValueError("LP rows and target must be finite")
    if np.any(lo > hi):
        raise ValueError("invalid LP variable bounds")
    I = sparse.eye(n, format="csr")
    Z = sparse.csr_matrix((A.shape[0], n))
    A_self = sparse.hstack((-A, Z), format="csr")
    A_abs_pos = sparse.hstack((I, -I), format="csr")
    A_abs_neg = sparse.hstack((-I, -I), format="csr")
    A_ub = sparse.vstack((A_self, A_abs_pos, A_abs_neg), format="csr")
    b_ub = np.concatenate((-b, t, -t))
    c = np.concatenate((np.zeros(n), np.ones(n)))
    bounds = list(zip(lo.tolist(), hi.tolist())) + [(0.0, None)] * n
    started = time.monotonic()
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs",
        options={"time_limit": float(time_limit_s), "presolve": True},
    )
    elapsed = time.monotonic() - started
    if not result.success:
        return {
            "success": False, "status": int(result.status), "message": str(result.message),
            "nit": int(getattr(result, "nit", 0)), "objective": None,
            "wall_seconds": float(elapsed),
        }
    d = np.asarray(result.x[:n], dtype=np.float64)
    return {
        "success": True, "status": int(result.status), "message": str(result.message),
        "nit": int(getattr(result, "nit", 0)), "objective": float(result.fun),
        "wall_seconds": float(elapsed),
        "delta": d,
        "minimum_normalized_linear_slack": float(np.min(A @ d - b, initial=np.inf)),
        "raw_target_l1_mm": float(np.sum(np.abs(t))),
        "residual_to_raw_target_l1_mm": float(np.sum(np.abs(d - t))),
    }


def _owner_matrix(pair_rows, area_rows: sparse.csr_matrix, area_rhs: np.ndarray,
                  support: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray]:
    local = {int(row): i for i, row in enumerate(support)}
    rr: list[int] = []
    cc: list[int] = []
    vv: list[float] = []
    lb_pair = np.empty(len(pair_rows), dtype=np.float64)
    for ri, row in enumerate(pair_rows):
        lb_pair[ri] = float(row["lower"])
        for node, coeff in zip(row["nodes"], row["coefficients"]):
            li = local.get(int(node))
            if li is None:
                raise ValueError(f"pair row references node outside one-ring support: {node}")
            for k, value in enumerate(np.asarray(coeff, dtype=np.float64)):
                if value:
                    rr.append(ri)
                    cc.append(3 * li + k)
                    vv.append(float(value))
    pair_A = sparse.coo_matrix((vv, (rr, cc)), shape=(len(pair_rows), 3 * len(support))).tocsr()
    area_cols = interleaved_coordinate_columns(support)
    area_A = area_rows[:, area_cols].tocsr()
    A = sparse.vstack((pair_A, area_A), format="csr")
    b = np.concatenate((lb_pair, np.asarray(area_rhs, dtype=np.float64)))
    norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    if np.any(norms <= 1e-14) or not np.isfinite(norms).all():
        raise ValueError("a local pair/area constraint row is zero or nonfinite")
    A = sparse.diags(1.0 / norms) @ A
    b = b / norms
    return A.tocsr(), b



def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"cannot read target report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"target report must be a JSON object: {path}")
    return value


def _report_candidate_sha(report: dict, path: Path) -> str:
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"target report lacks input pins: {path}")
    value = inputs.get("candidate_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"target report lacks a candidate SHA-256 pin: {path}")
    return value


def _require_target_candidate(report: dict, path: Path, expected_sha: str) -> None:
    actual = _report_candidate_sha(report, path)
    if actual != expected_sha:
        raise ValueError(f"target report candidate SHA does not match base candidate: {path}")



def select_face_target_directions(arrays: dict, report: dict) -> np.ndarray:
    """Select only report-declared target directions with their source-normal proof."""
    mode = report.get("direction_mode", "nearest-boundary")
    if mode == "nearest-boundary":
        return np.asarray(arrays.get("nearest_boundary_direction_world_unit", np.empty((0, 3))), dtype=np.float64)
    if mode not in ("source-outward", "outward-nearest"):
        raise ValueError("unsupported face proposal direction mode")
    direction = np.asarray(arrays.get("proposal_direction_world_unit", np.empty((0, 3))), dtype=np.float64)
    normals = np.asarray(arrays.get("source_surface_normal_world_unit", np.empty((0, 3))), dtype=np.float64)
    valid = np.asarray(arrays.get("source_normal_valid", np.empty(0)), dtype=bool)
    if direction.ndim != 2 or direction.shape[1] != 3 or normals.shape != direction.shape or valid.shape != (len(direction),):
        raise ValueError("outward direction/normal/validity domains mismatch")
    if not np.all(valid) or not np.isfinite(direction).all() or not np.isfinite(normals).all():
        raise ValueError("uncertified source-oriented proposal")
    direction_norm = np.linalg.norm(direction, axis=1)
    normal_norm = np.linalg.norm(normals, axis=1)
    if np.max(np.abs(direction_norm-1), initial=0) > 2e-10:
        raise ValueError("source-oriented proposal is not unit length")
    if np.max(np.abs(normal_norm-1), initial=0) > 2e-10:
        raise ValueError("registered source normal is not unit length")
    if mode == "source-outward":
        if not np.allclose(direction, normals, rtol=0, atol=2e-14):
            raise ValueError("source-outward direction differs from certified source normal")
        return direction
    # outward-nearest deliberately differs from the normal: it is the nearest
    # feasible contact vector constrained to the normal's closed hemisphere.
    hemisphere_dot = np.einsum("ij,ij->i", direction, normals)
    if np.any(hemisphere_dot < -128*np.finfo(np.float64).eps):
        raise ValueError("outward-nearest proposal leaves the certified source hemisphere")
    return direction

def load_clearance_target_inputs(point_npz_path: Path, point_report_path: Path,
                                 face_npz_path: Path, face_report_path: Path,
                                 candidate_path: Path, points_f32: np.ndarray,
                                 faces: np.ndarray) -> tuple[dict, dict]:
    """Load construction targets only when both reports pin this exact candidate."""
    candidate_sha = sha(candidate_path)
    point_report = _load_json(point_report_path)
    face_report = _load_json(face_report_path)
    _require_target_candidate(point_report, point_report_path, candidate_sha)
    _require_target_candidate(face_report, face_report_path, candidate_sha)

    with np.load(point_npz_path, allow_pickle=False) as z:
        p = {k: np.asarray(z[k]).copy() for k in z.files}
    with np.load(face_npz_path, allow_pickle=False) as z:
        f = {k: np.asarray(z[k]).copy() for k in z.files}

    point_rows = np.asarray(p.get("query_rows", np.empty(0)), dtype=np.int64)
    point_targets = np.asarray(p.get("target_world_f32_m", np.empty((0, 3))), dtype=np.float32)
    point_dirs = np.asarray(p.get("proposal_direction_world_unit", np.empty((0, 3))), dtype=np.float64)
    point_pass = np.asarray(p.get("pass_mask", np.empty(0)), dtype=bool)
    if point_rows.ndim != 1 or point_targets.shape != (len(point_rows), 3) or point_dirs.shape != (len(point_rows), 3) or point_pass.shape != (len(point_rows),):
        raise ValueError("point target NPZ arrays have inconsistent shapes")
    if len(np.unique(point_rows)) != len(point_rows) or np.any(point_rows < 0) or np.any(point_rows >= len(points_f32)):
        raise ValueError("point target rows are duplicated or outside the candidate")
    if not np.isfinite(point_targets).all() or not np.isfinite(point_dirs).all():
        raise ValueError("point target arrays contain nonfinite values")
    if not np.all(point_pass):
        raise ValueError("point target NPZ contains a failed target")
    if len(point_rows) and np.max(np.abs(np.linalg.norm(point_dirs, axis=1) - 1.0)) > 2e-10:
        raise ValueError("point proposal directions are not unit world vectors")
    records = point_report.get("targets")
    if not isinstance(records, list) or len(records) != len(point_rows):
        raise ValueError("point target report rows do not match NPZ")
    by_row = {}
    for record in records:
        if not isinstance(record, dict) or "query_row" not in record:
            raise ValueError("malformed point target report record")
        row = int(record["query_row"])
        if row in by_row:
            raise ValueError(f"duplicate point target report row {row}")
        by_row[row] = record
    if set(by_row) != set(map(int, point_rows)):
        raise ValueError("point target report row identities differ from NPZ")
    for i, row in enumerate(point_rows):
        rec = by_row[int(row)]
        if rec.get("status") != "PASS_FIRST_UNION_EXIT" or not rec.get("end_outside_all185_by_event_parity", False):
            raise ValueError(f"point row {row} lacks a passing proposal-only endpoint record")
        start = np.asarray(rec.get("start_world_f32_m"), dtype=np.float32)
        endpoint = np.asarray(rec.get("target_world_f32_m"), dtype=np.float32)
        direction = np.asarray(rec.get("proposal_direction_world_unit"), dtype=np.float64)
        if start.shape != (3,) or endpoint.shape != (3,) or direction.shape != (3,):
            raise ValueError(f"point report row {row} has malformed coordinate records")
        if not np.array_equal(start.view(np.uint32), points_f32[row].view(np.uint32)):
            raise ValueError(f"point target start does not match base candidate row {row}")
        if not np.array_equal(endpoint.view(np.uint32), point_targets[i].view(np.uint32)):
            raise ValueError(f"point target endpoint differs between NPZ and report row {row}")
        if not np.allclose(direction, point_dirs[i], rtol=0.0, atol=1e-14):
            raise ValueError(f"point target direction differs between NPZ and report row {row}")

    face_rows = np.asarray(f.get("face_rows", np.empty(0)), dtype=np.int64)
    face_nodes = np.asarray(f.get("face_node_rows", np.empty((0, 3))), dtype=np.int64)
    bary = np.asarray(f.get("barycentric_weights", np.empty((0, 3))), dtype=np.float64)
    face_sample = np.asarray(f.get("inside_sample_world_m", np.empty((0, 3))), dtype=np.float64)
    face_target = np.asarray(f.get("target_world_f32_m", np.empty((0, 3))), dtype=np.float32)
    face_dirs = select_face_target_directions(f, face_report)
    face_outside = np.asarray(f.get("target_exactly_outside_all185", np.empty(0)), dtype=bool)
    k = len(face_rows)
    if face_rows.ndim != 1 or face_nodes.shape != (k, 3) or bary.shape != (k, 3) or face_sample.shape != (k, 3) or face_target.shape != (k, 3) or face_dirs.shape != (k, 3) or face_outside.shape != (k,):
        raise ValueError("face target NPZ arrays have inconsistent shapes")
    if np.any(face_rows < 0) or np.any(face_rows >= len(faces)) or np.any(face_nodes < 0) or np.any(face_nodes >= len(points_f32)):
        raise ValueError("face target rows/nodes are outside candidate domains")
    if not np.array_equal(faces[face_rows], face_nodes):
        raise ValueError("face target node identities/order do not match candidate faces")
    if not np.isfinite(bary).all() or np.any(bary < 0.0) or np.any(bary > 1.0) or np.max(np.abs(bary.sum(axis=1) - 1.0), initial=0.0) > 2e-12:
        raise ValueError("face target barycentric rows are invalid")
    if not np.isfinite(face_sample).all() or not np.isfinite(face_target).all() or not np.isfinite(face_dirs).all() or not np.all(face_outside):
        raise ValueError("face target contains nonfinite data or endpoint not certified outside all 185")
    if k and np.max(np.abs(np.linalg.norm(face_dirs, axis=1) - 1.0)) > 2e-10:
        raise ValueError("face target directions are not unit world vectors")
    if (int(face_report.get("passing_targets", -1)) != k
            or int(face_report.get("result_rows", -1)) != k
            or int(face_report.get("candidate_points", -1)) != len(points_f32)
            or int(face_report.get("candidate_faces", -1)) != len(faces)
            or face_report.get("failure_counts") not in ({}, None)
            or not face_report.get("matched_159_complete", False)):
        raise ValueError("face target report is incomplete or has a different candidate/target domain")
    if (int(point_report.get("passing_targets", -1)) != len(point_rows)
            or int(point_report.get("query_rows", -1)) != len(point_rows)
            or int(point_report.get("all_bank_ray_results", -1)) != len(point_rows)
            or int(point_report.get("inside_node_bank_pairs", -1)) != len(point_rows)
            or int(point_report.get("physical_steps", -1)) != 0):
        raise ValueError("point target report is incomplete, mismatched, or records physical steps")
    return {
        "point_rows": point_rows, "point_targets_f32_m": point_targets,
        "point_directions": point_dirs, "point_report": point_report,
        "point_npz": p,
    }, {
        "face_rows": face_rows, "face_nodes": face_nodes, "barycentric_weights": bary,
        "face_samples_m": face_sample, "face_targets_f32_m": face_target,
        "face_directions": face_dirs, "face_report": face_report,
        "face_npz": f,
    }


def build_clearance_rows(points_mm: np.ndarray, faces: np.ndarray, support: np.ndarray,
                         anchors: np.ndarray, point_targets: dict, face_targets: dict,
                         bary_owner) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, list[dict]]:
    """Compose point and barycentric face target rows in node-major xyz coordinates."""
    p = np.asarray(points_mm, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    support = np.asarray(support, dtype=np.int64)
    anchors = np.asarray(anchors, dtype=np.int64)
    local = {int(node): i for i, node in enumerate(support)}
    point_rows = np.asarray(point_targets["point_rows"], dtype=np.int64)
    point_xyz = np.asarray(point_targets["point_targets_f32_m"], dtype=np.float64) * 1000.0
    point_dirs = np.asarray(point_targets["point_directions"], dtype=np.float64)
    face_rows = np.asarray(face_targets["face_rows"], dtype=np.int64)
    weights = np.asarray(face_targets["barycentric_weights"], dtype=np.float64)
    face_xyz = np.asarray(face_targets["face_targets_f32_m"], dtype=np.float64) * 1000.0
    face_dirs = np.asarray(face_targets["face_directions"], dtype=np.float64)
    B, tri, g, face_h, _ = bary_owner.build_barycentric_exit_system(
        p, f, face_rows, weights, face_dirs, face_xyz, anchor_rows=()
    )
    sample_mismatch = np.asarray(B @ p).reshape(-1, 3) - np.asarray(face_targets["face_samples_m"], dtype=np.float64) * 1000.0
    if sample_mismatch.size and float(np.max(np.abs(sample_mismatch))) > 1e-6:
        raise ValueError("barycentric target inside sample does not match exact base face interpolation")
    # Anchors remain fixed variables; zero only their target coefficients so
    # unreachable targets become explicit normalized slack, not movable anchors.
    g = np.asarray(g, dtype=np.float64).copy()
    if len(anchors):
        g[np.isin(tri, anchors)] = 0.0
    if any(int(node) not in local for node in point_rows) or any(int(node) not in local for node in tri.ravel()):
        raise ValueError("clearance target references a node outside expanded support")

    rr: list[int] = []
    cc: list[int] = []
    vv: list[float] = []
    h_point = np.einsum("ij,ij->i", point_dirs, point_xyz - p[point_rows]) if len(point_rows) else np.empty(0)
    if np.any(h_point <= 0.0):
        raise ValueError("point clearance endpoint is not ahead of its exact start")
    anchor_set = set(map(int, anchors))
    for ri, (node, direction) in enumerate(zip(point_rows, point_dirs)):
        if int(node) in anchor_set:
            continue
        li = local[int(node)]
        for axis, value in enumerate(direction):
            if value:
                rr.append(ri); cc.append(3 * li + axis); vv.append(float(value))
    n_point = len(point_rows)
    for fi, (nodes, coeff) in enumerate(zip(tri, g)):
        row = n_point + fi
        for node, vec in zip(nodes, coeff):
            li = local[int(node)]
            for axis, value in enumerate(vec):
                if value:
                    rr.append(row); cc.append(3 * li + axis); vv.append(float(value))
    C = sparse.coo_matrix((vv, (rr, cc)), shape=(n_point + len(face_rows), 3 * len(support))).tocsr()
    h = np.concatenate((h_point, np.asarray(face_h, dtype=np.float64)))
    if np.any(h <= 0.0) or not np.isfinite(h).all():
        raise ValueError("clearance targets must have positive finite projected distance")
    row_norm = np.sqrt(np.asarray(C.multiply(C).sum(axis=1)).ravel())
    nonzero = row_norm > 1e-14
    C = (sparse.diags(np.where(nonzero, 1.0 / np.maximum(row_norm, 1e-300), 1.0)) @ C).tocsr()
    h = np.where(nonzero, h / np.maximum(row_norm, 1e-300), h)

    n_face_unique = len(np.unique(face_rows))
    point_weight = 0.5 / len(point_rows) if len(point_rows) and len(face_rows) else (1.0 / len(point_rows) if len(point_rows) else 0.0)
    face_weights = np.zeros(len(face_rows), dtype=np.float64)
    if len(face_rows):
        _, inverse, counts = np.unique(face_rows, return_inverse=True, return_counts=True)
        face_total = 0.5 if len(point_rows) else 1.0
        face_weights = face_total / (n_face_unique * counts[inverse])
    row_weights = np.concatenate((np.full(len(point_rows), point_weight), face_weights))
    metadata = []
    for i, row in enumerate(point_rows):
        metadata.append({"kind": "point", "query_row": int(row), "target_mm": float(h[i]),
                         "normalized_weight": float(row_weights[i])})
    for i, row in enumerate(face_rows):
        krow = n_point + i
        metadata.append({"kind": "face", "face_row": int(row), "target_mm": float(h[krow]),
                         "bank_body": int(face_targets["face_npz"]["bank_body"][i]),
                         "bank_stable_bone_id": int(face_targets["face_npz"]["bank_stable_bone_id"][i]),
                         "normalized_weight": float(row_weights[krow])})
    return C, h, row_weights, metadata


def clearance_progress_summary(C: sparse.csr_matrix, h: np.ndarray, weights: np.ndarray,
                               delta: np.ndarray, metadata: list[dict]) -> dict:
    C = sparse.csr_matrix(C, dtype=np.float64)
    hh = np.asarray(h, dtype=np.float64)
    ww = np.asarray(weights, dtype=np.float64)
    dd = np.asarray(delta, dtype=np.float64).reshape(-1)
    projection = np.asarray(C @ dd).reshape(-1)
    fraction = np.divide(projection, hh, out=np.zeros_like(projection), where=hh > 0.0)
    by_kind = {}
    for kind in ("point", "face"):
        idx = np.asarray([i for i, item in enumerate(metadata) if item["kind"] == kind], dtype=np.int64)
        by_kind[kind] = {
            "rows": int(len(idx)),
            "minimum_progress_fraction": float(fraction[idx].min(initial=np.inf)) if len(idx) else None,
            "mean_progress_fraction": float(np.average(fraction[idx], weights=ww[idx])) if len(idx) else None,
            "weighted_unmet_fraction": float(np.average(np.maximum(0.0, 1.0 - fraction[idx]), weights=ww[idx])) if len(idx) else None,
        }
    return {
        "row_count": int(len(hh)),
        "minimum_absolute_projected_progress_mm": float(projection.min(initial=np.inf)) if len(projection) else None,
        "minimum_progress_fraction": float(fraction.min(initial=np.inf)) if len(fraction) else None,
        "weighted_unmet_fraction": float(np.dot(ww, np.maximum(0.0, 1.0 - fraction))) if len(ww) else 0.0,
        "by_kind": by_kind,
    }


def solve_sparse_l1_clearance(A: sparse.csr_matrix, lower_rows: np.ndarray,
                              target: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                              C: sparse.csr_matrix, h: np.ndarray, weights: np.ndarray,
                              time_limit_s: float, slack_tolerance: float = 1e-9) -> dict:
    """Lexicographic HiGHS solve: minimize normalized target slack, then raw L1 distance."""
    A = sparse.csr_matrix(A, dtype=np.float64)
    C = sparse.csr_matrix(C, dtype=np.float64)
    b = np.asarray(lower_rows, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    lo = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper, dtype=np.float64).reshape(-1)
    hh = np.asarray(h, dtype=np.float64).reshape(-1)
    ww = np.asarray(weights, dtype=np.float64).reshape(-1)
    n, m = len(t), len(hh)
    if A.shape != (len(b), n) or C.shape != (m, n) or lo.shape != (n,) or hi.shape != (n,) or ww.shape != (m,):
        raise ValueError("clearance LP dimensions do not align")
    if not all(np.isfinite(x).all() for x in (A.data, C.data, b, t, lo, hi, hh, ww)):
        raise ValueError("clearance LP inputs contain nonfinite values")
    if np.any(lo > hi) or np.any(hh <= 0) or np.any(ww < 0) or (m and float(ww.sum()) <= 0):
        raise ValueError("clearance LP has invalid bounds, targets, or weights")
    if m == 0:
        result = solve_sparse_l1(A, b, t, lo, hi, time_limit_s)
        if result["success"]:
            result.update({"normalized_slack_objective": 0.0, "normalized_slack": np.empty(0),
                           "stage1": {"success": True, "objective": 0.0, "status": "EMPTY_TARGETS"},
                           "stage2": {"success": True, "objective": result["objective"]},
                           "lexicographic_slack_tolerance": float(slack_tolerance)})
        return result

    started = time.monotonic()
    # Hard barrier rows and no-regression rows.
    hard = sparse.hstack((-A, sparse.csr_matrix((A.shape[0], m))), format="csr")
    noregress = sparse.hstack((-C, sparse.csr_matrix((m, m))), format="csr")
    clearance = sparse.hstack((-C, -sparse.diags(hh)), format="csr")
    Aub1 = sparse.vstack((hard, noregress, clearance), format="csr")
    bub1 = np.concatenate((-b, np.zeros(m), -hh))
    c1 = np.concatenate((np.zeros(n), ww))
    bounds1 = list(zip(lo.tolist(), hi.tolist())) + [(0.0, None)] * m
    stage1 = linprog(c1, A_ub=Aub1, b_ub=bub1, bounds=bounds1, method="highs",
                     options={"time_limit": float(time_limit_s), "presolve": True})
    if not stage1.success:
        return {"success": False, "status": int(stage1.status), "message": str(stage1.message),
                "nit": int(getattr(stage1, "nit", 0)), "objective": None,
                "wall_seconds": float(time.monotonic() - started),
                "stage1": {"success": False, "status": int(stage1.status), "message": str(stage1.message)}}
    slack_opt = float(stage1.fun)
    tol = float(slack_tolerance)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("slack tolerance must be finite and nonnegative")

    # Stage 2 retains optimal clearance progress within the declared numerical
    # slack budget, then minimizes the original L1 distance to the raw target.
    Znm = sparse.csr_matrix((n, m))
    Zmn = sparse.csr_matrix((m, n))
    Zmm = sparse.csr_matrix((m, m))
    hard2 = sparse.hstack((-A, sparse.csr_matrix((A.shape[0], n)), sparse.csr_matrix((A.shape[0], m))), format="csr")
    noregress2 = sparse.hstack((-C, sparse.csr_matrix((m, n)), Zmm), format="csr")
    clearance2 = sparse.hstack((-C, sparse.csr_matrix((m, n)), -sparse.diags(hh)), format="csr")
    I = sparse.eye(n, format="csr")
    abs_pos = sparse.hstack((I, -I, sparse.csr_matrix((n, m))), format="csr")
    abs_neg = sparse.hstack((-I, -I, sparse.csr_matrix((n, m))), format="csr")
    slack_budget = sparse.csr_matrix((ww.reshape(1, -1)), dtype=np.float64)
    slack_budget = sparse.hstack((sparse.csr_matrix((1, n + n)), slack_budget), format="csr")
    Aub2 = sparse.vstack((hard2, noregress2, clearance2, abs_pos, abs_neg, slack_budget), format="csr")
    bub2 = np.concatenate((-b, np.zeros(m), -hh, t, -t, [slack_opt + tol]))
    c2 = np.concatenate((np.zeros(n), np.ones(n), np.zeros(m)))
    bounds2 = list(zip(lo.tolist(), hi.tolist())) + [(0.0, None)] * n + [(0.0, None)] * m
    stage2 = linprog(c2, A_ub=Aub2, b_ub=bub2, bounds=bounds2, method="highs",
                     options={"time_limit": float(time_limit_s), "presolve": True})
    if not stage2.success:
        return {"success": False, "status": int(stage2.status), "message": str(stage2.message),
                "nit": int(getattr(stage2, "nit", 0)), "objective": None,
                "wall_seconds": float(time.monotonic() - started),
                "stage1": {"success": True, "objective": slack_opt, "status": int(stage1.status)},
                "stage2": {"success": False, "status": int(stage2.status), "message": str(stage2.message)}}
    d = np.asarray(stage2.x[:n], dtype=np.float64)
    slack = np.asarray(stage2.x[2 * n:], dtype=np.float64)
    hard_residual = np.asarray(A @ d).reshape(-1) - b
    progress = np.asarray(C @ d).reshape(-1)
    slack_residual = progress + hh * slack - hh
    if (np.min(hard_residual, initial=0.0) < -2e-7
            or np.min(progress, initial=0.0) < -2e-7
            or np.min(slack_residual, initial=0.0) < -2e-7
            or float(np.dot(ww, slack)) > slack_opt + tol + 2e-7):
        raise RuntimeError("lexicographic clearance LP returned a solution outside its constraints")
    return {
        "success": True, "status": int(stage2.status), "message": str(stage2.message),
        "nit": int(getattr(stage1, "nit", 0) + getattr(stage2, "nit", 0)),
        "objective": float(stage2.fun),
        "wall_seconds": float(time.monotonic() - started),
        "delta": d, "normalized_slack": slack,
        "normalized_slack_objective": float(np.dot(ww, slack)),
        "stage1": {"success": True, "status": int(stage1.status), "objective": slack_opt,
                   "iterations": int(getattr(stage1, "nit", 0))},
        "stage2": {"success": True, "status": int(stage2.status),
                   "objective": float(stage2.fun), "iterations": int(getattr(stage2, "nit", 0)),
                   "slack_budget": float(slack_opt + tol)},
        "lexicographic_slack_tolerance": tol,
        "minimum_normalized_linear_slack": float(np.min(hard_residual, initial=np.inf)),
        "minimum_clearance_progress_mm": float(np.min(progress, initial=np.inf)),
        "minimum_target_inequality_slack_mm": float(np.min(slack_residual, initial=np.inf)),
        "raw_target_l1_mm": float(np.sum(np.abs(t))),
        "residual_to_raw_target_l1_mm": float(np.sum(np.abs(d - t))),
    }



def validate_fixed_prefix_nodes(value, node_count: int) -> int:
    """Validate an optional contiguous node-prefix fixed at zero displacement."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("fixed_prefix_nodes must be an integer")
    count = int(value)
    n = int(node_count)
    if n < 0 or count < 0 or count > n:
        raise ValueError(f"fixed_prefix_nodes must lie in [0,{n}]")
    return count


def validate_self_pair_input(pair_array, face_count: int, has_clearance_targets: bool) -> np.ndarray:
    """Validate/deduplicate self-pair face rows; empty is valid only with targets."""
    raw = np.asarray(pair_array)
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.dtype.kind not in "iu":
        raise ValueError("self-pair input must be an integer (M,2) face-row array")
    pairs = np.asarray(sorted({tuple(sorted(map(int, row))) for row in raw}), dtype=np.int64).reshape(-1, 2)
    if np.any(pairs < 0) or np.any(pairs >= int(face_count)) or np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("self-pair face row out of range or self-pair")
    if len(pairs) == 0 and not bool(has_clearance_targets):
        raise ValueError("self-pair input and point/face clearance targets are all empty")
    return pairs


def apply_fixed_prefix_zero_bounds(lower, upper, support, fixed_prefix_nodes: int,
                                   node_count: int):
    """Set the LP displacement bounds to exact zero for supported prefix rows."""
    count = validate_fixed_prefix_nodes(fixed_prefix_nodes, node_count)
    lo = np.asarray(lower, dtype=np.float64).copy()
    hi = np.asarray(upper, dtype=np.float64).copy()
    raw_rows = np.asarray(support)
    if raw_rows.ndim != 1 or raw_rows.dtype.kind not in "iu":
        raise ValueError("support rows must be a one-dimensional integer array")
    rows = raw_rows.astype(np.int64, copy=False)
    if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= int(node_count)):
        raise ValueError("support rows are duplicate or outside node domain")
    if lo.shape != (3 * len(rows),) or hi.shape != lo.shape or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("coordinate bounds do not match node support")
    if np.any(lo > hi):
        raise ValueError("coordinate lower bound exceeds upper bound")
    local = np.flatnonzero(rows < count)
    if len(local):
        cols = (3 * local[:, None] + np.arange(3, dtype=np.int64)[None, :]).reshape(-1)
        lo[cols] = 0.0
        hi[cols] = 0.0
    return lo, hi, rows[local].copy()


def assert_prefix_bitwise_unchanged(base_points, candidate_points, fixed_prefix_nodes: int) -> None:
    """Fail unless an emitted point array preserves the base prefix byte-for-byte."""
    base = np.asarray(base_points)
    candidate = np.asarray(candidate_points)
    count = validate_fixed_prefix_nodes(fixed_prefix_nodes, len(base))
    if candidate.shape != base.shape or candidate.dtype != base.dtype:
        raise ValueError("candidate point array shape/dtype differs from base")
    if np.ascontiguousarray(base[:count]).tobytes() != np.ascontiguousarray(candidate[:count]).tobytes():
        raise ValueError(f"fixed prefix [0:{count}] changed at the emitted point stage")


def run(args) -> dict:
    paths = {
        "base_candidate": Path(args.base_candidate),
        "raw_trial": Path(args.raw_trial),
        "self_pairs": Path(args.self_pairs),
        "area_reference": Path(args.area_reference_npz),
        "pair_owner": Path(args.pair_owner),
        "area_owner": Path(args.area_owner),
        "bary_owner": Path(args.bary_owner),
        "point_target_npz": Path(args.point_target_npz),
        "point_target_report": Path(args.point_target_report),
        "face_target_npz": Path(args.face_target_npz),
        "face_target_report": Path(args.face_target_report),
    }
    fixed_boundary_enabled = any((args.fixed_boundary_npz, args.fixed_boundary_report,
                                  args.fixed_boundary_owner))
    if fixed_boundary_enabled:
        if not all((args.fixed_boundary_npz, args.fixed_boundary_report, args.fixed_boundary_owner)):
            raise ValueError("fixed-boundary NPZ, report, and owner must be supplied together")
        paths.update({
            "fixed_boundary_npz": Path(args.fixed_boundary_npz),
            "fixed_boundary_report": Path(args.fixed_boundary_report),
            "fixed_boundary_owner": Path(args.fixed_boundary_owner),
            "fixed_boundary_emitter": DEFAULT_FIXED_BOUNDARY_EMITTER,
        })
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output path: {output}")
    input_hashes = {k: sha(v) for k, v in paths.items()}
    base = np.load(paths["base_candidate"], allow_pickle=False)
    raw = np.load(paths["raw_trial"], allow_pickle=False)
    points = np.asarray(base["points_world_m"], dtype=np.float32)
    fixed_prefix_nodes = validate_fixed_prefix_nodes(args.fixed_prefix_nodes, len(points))
    points_mm = points.astype(np.float64) * 1000.0
    target_points = np.asarray(raw["points_world_m"], dtype=np.float32)
    faces_raw = np.asarray(base["source_O_faces"])
    faces = faces_raw.astype(np.int64, copy=False)
    raw_faces = np.asarray(raw["source_O_faces"], dtype=np.int64)
    if (target_points.shape != points.shape or not np.array_equal(faces, raw_faces)
            or faces_raw.dtype.kind not in "iu"):
        raise ValueError("base and raw trial node/face domains differ")
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(points)):
        raise ValueError("invalid source_O_faces index domain")
    area_reference_m, area_reference_receipt = load_area_reference(
        paths["area_reference"], faces,
        points_key=args.area_reference_points_key,
        faces_key=args.area_reference_faces_key,
    )
    if area_reference_m.shape != points.shape:
        raise ValueError("fixed 037 area-reference point domain differs from candidate")
    area_reference_mm = area_reference_m * 1000.0
    raw_pairs = np.load(paths["self_pairs"], allow_pickle=False)
    point_targets, face_targets = load_clearance_target_inputs(
        paths["point_target_npz"], paths["point_target_report"],
        paths["face_target_npz"], paths["face_target_report"],
        paths["base_candidate"], points, faces,
    )
    fixed_boundary_records_input = []
    fixed_boundary_receipt = None
    fixed_boundary_rows = []
    fixed_boundary_typed_records = []
    fixed_boundary_npz_arrays = None
    if fixed_boundary_enabled:
        with np.load(paths["fixed_boundary_npz"], allow_pickle=False) as z:
            fixed_boundary_npz_arrays = {k: np.asarray(z[k]).copy() for k in z.files}
        fixed_boundary_report = _load_json(paths["fixed_boundary_report"])
        fixed_boundary_records_input, fixed_boundary_receipt = validate_fixed_boundary_payload(
            fixed_boundary_npz_arrays, fixed_boundary_report,
            candidate_sha256=input_hashes["base_candidate"],
            candidate_points=points,
            candidate_faces=faces_raw,
            npz_path=paths["fixed_boundary_npz"],
        )
        fixed_boundary_receipt["report_sha256"] = input_hashes["fixed_boundary_report"]
        fixed_boundary_receipt["owner_path"] = str(paths["fixed_boundary_owner"])
        fixed_boundary_receipt["owner_sha256"] = input_hashes["fixed_boundary_owner"]
        fixed_boundary_receipt["emitter_path"] = str(paths["fixed_boundary_emitter"])
        fixed_boundary_receipt["emitter_sha256"] = input_hashes["fixed_boundary_emitter"]
    has_clearance_targets = bool(len(point_targets["point_rows"]) or len(face_targets["face_rows"])
                                 or len(fixed_boundary_records_input))
    pairs = validate_self_pair_input(raw_pairs, len(faces), has_clearance_targets)

    pair_owner = load_module("self_separation_owner_315", paths["pair_owner"])
    if fixed_boundary_enabled:
        fixed_owner = load_module("fixed_bone_rows_owner_315", paths["fixed_boundary_owner"])
        fixed_boundary_rows, fixed_boundary_typed_records = build_fixed_boundary_barriers(
            points_mm, faces, fixed_boundary_records_input, pair_owner, fixed_owner,
            float(args.margin_fraction),
        )
    fixed_boundary_face_rows = (
        np.asarray([r["skin_face_row"] for r in fixed_boundary_records_input], dtype=np.int64)
        if fixed_boundary_records_input else np.empty(0, dtype=np.int64)
    )

    source_base_path = Path(args.source_base_npz) if args.source_base_npz else None
    source_ref_m, source_receipt = compose_source_reference(base, source_base_path)
    source_ref_mm = source_ref_m * 1000.0
    all_source_distance = np.linalg.norm(points.astype(np.float64) * 1000.0 - source_ref_mm, axis=1)
    if float(all_source_distance.max(initial=0.0)) > args.source_cap_mm + 1e-7:
        raise ValueError("base candidate exceeds source cap before proposal")

    core_nodes = np.unique(np.concatenate((
        faces[pairs].ravel(),
        point_targets["point_rows"],
        face_targets["face_nodes"].ravel(),
        faces[fixed_boundary_face_rows].ravel() if len(fixed_boundary_face_rows) else np.empty(0, dtype=np.int64),
    ))).astype(np.int64)
    core_faces = np.flatnonzero(np.isin(faces, core_nodes).any(axis=1)).astype(np.int64)
    support = np.unique(faces[core_faces].ravel()).astype(np.int64)
    if len(core_nodes) == 0 or len(core_faces) == 0 or len(support) == 0:
        raise ValueError("empty threatened-pair local support")
    anchors_key = "typed_coarse_anchor_output_rows_018"
    if anchors_key not in base.files:
        raise ValueError(f"candidate lacks required typed fixed-anchor rows: {anchors_key}")
    anchors = np.asarray(base[anchors_key], dtype=np.int64)
    if anchors.ndim != 1 or np.any(anchors < 0) or np.any(anchors >= len(points)) or len(np.unique(anchors)) != len(anchors):
        raise ValueError("fixed-anchor rows are malformed")
    local_anchors = np.intersect1d(support, anchors, assume_unique=False)
    fixed_prefix_support = support[support < fixed_prefix_nodes]
    fixed_support = np.union1d(local_anchors, fixed_prefix_support)
    free_support = np.setdiff1d(support, fixed_support, assume_unique=True)
    if len(free_support) == 0:
        raise ValueError("threatened one-ring has no free support nodes")
    # Every triangle touched by a movable support node participates in the
    # local linearized area system. Nodes outside support remain fixed at zero.
    area_faces = np.flatnonzero(np.isin(faces, free_support).any(axis=1)).astype(np.int64)
    if len(area_faces) == 0:
        raise ValueError("no area faces touch a free support node")
    if float(args.max_edge_gradient) > 0.0:
        edge_gradient_A, edge_gradient_b, edge_gradient_report = build_edge_gradient_constraints(
            points_mm, faces, support, float(args.max_edge_gradient)
        )
    else:
        edge_gradient_A = sparse.csr_matrix((0, 3 * len(support)), dtype=np.float64)
        edge_gradient_b = np.empty(0, dtype=np.float64)
        edge_gradient_report = {
            "enabled": False, "max_component_gradient": 0.0,
            "unique_mesh_edges": 0, "active_edges": 0, "inequality_rows": 0,
            "semantics": "disabled; frozen 222 behavior",
        }

    area_owner = load_module("area_linearization_owner_199", paths["area_owner"])
    bary_owner = load_module("barycentric_clearance_owner_091", paths["bary_owner"])
    area_layout = verify_area_linearization_layout(area_owner)
    clearance_C, clearance_h, clearance_weights, clearance_metadata = build_clearance_rows(
        points_mm, faces, support, local_anchors, point_targets, face_targets, bary_owner
    )
    clearance_target_receipts = {
        "point_target_npz_sha256": input_hashes["point_target_npz"],
        "point_target_report_sha256": input_hashes["point_target_report"],
        "face_target_npz_sha256": input_hashes["face_target_npz"],
        "face_target_report_sha256": input_hashes["face_target_report"],
        "barycentric_owner_sha256": input_hashes["bary_owner"],
        "point_report_candidate_sha256": _report_candidate_sha(point_targets["point_report"], paths["point_target_report"]),
        "face_report_candidate_sha256": _report_candidate_sha(face_targets["face_report"], paths["face_target_report"]),
        "proposal_only": True,
    }
    self_pair_rows, pair_records = pair_owner.build_pair_rows(
        points_mm, faces, pairs, margin_fraction=float(args.margin_fraction)
    )
    pair_rows = combine_barrier_rows(self_pair_rows, fixed_boundary_rows)
    if fixed_boundary_enabled:
        support_set = set(map(int, support))
        required_fixed_nodes = set(map(int, faces[fixed_boundary_face_rows].ravel()))
        if not required_fixed_nodes.issubset(support_set):
            raise ValueError("fixed-boundary skin vertices were omitted from local LP support")

    normals_all, area0, area037, required_all = fixed_reference_area_requirements(
        points_mm, area_reference_mm, faces
    )
    normals = normals_all[area_faces]
    required = required_all[area_faces]
    lower, upper, bound_report = source_safe_trust_bounds(
        points_mm, source_ref_mm, support, local_anchors,
        float(args.trust_mm), float(args.source_cap_mm),
    )
    lower, upper, bounded_prefix_support = apply_fixed_prefix_zero_bounds(
        lower, upper, support, fixed_prefix_nodes, len(points)
    )
    target_delta = (target_points.astype(np.float64) * 1000.0 - points_mm)[support].reshape(-1)
    area_scp_records = []
    d_current = np.zeros(3 * len(support), dtype=np.float64)
    solved = None
    solve_wall = 0.0
    initial_baseline_max_violation = None
    scp_failure = None
    for scp_index in range(int(args.area_scp_iterations)):
        linearization_points_mm = points_mm.copy()
        linearization_points_mm[support] += d_current.reshape(-1, 3)
        Gq, rhs_q, _, _ = area_owner.area_linearization(
            linearization_points_mm, faces, area_faces, normals, required,
            minimum_ratio=float(args.area_floor),
        )
        rhs_absolute = absolute_area_rhs_from_incremental(Gq, rhs_q, support, d_current)
        A, b = _owner_matrix(pair_rows, Gq, rhs_absolute, support)
        if edge_gradient_A.shape[0]:
            A = sparse.vstack((A, edge_gradient_A), format="csr")
            b = np.concatenate((b, edge_gradient_b))
        base_violation = float(np.max(b, initial=-np.inf))
        if scp_index == 0:
            initial_baseline_max_violation = base_violation
            if base_violation > 2e-8:
                raise ValueError(f"base candidate infeasible for initial local pair/area rows: {base_violation} mm")
        t0 = time.monotonic()
        one = solve_sparse_l1_clearance(
            A, b, target_delta, lower, upper, clearance_C, clearance_h,
            clearance_weights, float(args.time_limit_s), float(args.slack_tolerance),
        )
        one_wall = time.monotonic() - t0
        solve_wall += one_wall
        record = {
            "iteration": scp_index + 1,
            "linearization_delta_l2_mm": float(np.linalg.norm(d_current)),
            "linearization_delta_max_abs_mm": float(np.max(np.abs(d_current), initial=0.0)),
            "incremental_rhs_min_mm": float(np.min(rhs_q, initial=np.inf)),
            "absolute_rhs_min_mm": float(np.min(rhs_absolute, initial=np.inf)),
            "zero_total_delta_max_violation_mm": base_violation,
            "zero_total_delta_feasibility_asserted": bool(scp_index == 0),
            "lp_success": bool(one["success"]),
            "lp_status": int(one["status"]),
            "lp_message": str(one["message"]),
            "lp_iterations": int(one.get("nit", 0)),
            "lp_objective": one.get("objective"),
            "lp_wall_seconds": float(one_wall),
            "edge_gradient_inequality_rows": int(edge_gradient_A.shape[0]),
            "minimum_normalized_linear_slack": one.get("minimum_normalized_linear_slack"),
            "clearance_slack_objective": one.get("normalized_slack_objective"),
            "clearance_stage1": one.get("stage1"),
            "clearance_stage2": one.get("stage2"),
            "minimum_clearance_progress_mm": one.get("minimum_clearance_progress_mm"),
        }
        area_scp_records.append(record)
        if not one["success"]:
            scp_failure = {k: v for k, v in record.items()}
            if solved is None:
                solved = one
            break
        d_new = np.asarray(one.pop("delta"), dtype=np.float64)
        if d_new.shape != d_current.shape or not np.isfinite(d_new).all():
            raise ValueError("area SCP LP returned malformed absolute delta")
        record["absolute_delta_step_l2_mm"] = float(np.linalg.norm(d_new - d_current))
        record["absolute_delta_max_abs_mm"] = float(np.max(np.abs(d_new), initial=0.0))
        solved = {**one, "delta": d_new}
        d_current = d_new
    if solved is None:
        raise RuntimeError("area SCP did not produce or report a solver result")
    if not solved["success"]:
        report = {
            "status": "LP_NO_PROPOSAL",
            "classification": "construction-only bounded area SCP; no candidate admitted",
            "native_admission": False, "accepted_physical_steps": 0,
            "inputs": {**{k: str(v) for k, v in paths.items()}, **{k + "_sha256": v for k, v in input_hashes.items()}},
            "source_reference_R0_for_cap_only": source_receipt,
            "area_reference_037_for_area_constraints_only": area_reference_receipt,
            "support": {"threatened_pairs": int(len(pairs)), "core_nodes": int(len(core_nodes)),
                        "core_faces": int(len(core_faces)), "area_faces_touching_free_support": int(len(area_faces)),
                        "support_nodes": int(len(support)), "free_support_nodes": int(len(free_support)),
                        "fixed_boundary_skin_faces": int(len(fixed_boundary_face_rows)),
                        "fixed_boundary_barrier_rows": int(len(fixed_boundary_rows)),
                        "fixed_prefix_nodes": fixed_prefix_nodes,
                        "fixed_prefix_support_rows": int(len(bounded_prefix_support)),
                        "fixed_prefix_semantics": "LP coordinate bounds are exactly zero; not reinterpreted as native anchors"},
            "area_scp_iterations": area_scp_records, "area_scp_failure": scp_failure,
            "solver": {k: v for k, v in solved.items() if k != "delta"},
            "wall_seconds": solve_wall,
        }
        output.mkdir(parents=True, exist_ok=False)
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    d = np.asarray(solved["delta"], dtype=np.float64).reshape(-1, 3)
    raw_lp_points = emit_raw_lp_points(points, points_mm, support, d, local_anchors)
    assert_prefix_bitwise_unchanged(points, raw_lp_points, fixed_prefix_nodes)
    raw_lp_area_gate = full_f32_area_gate(
        raw_lp_points, points_mm, area_reference_mm, faces, float(args.area_floor)
    )
    raw_lp_step_mm = np.linalg.norm(
        raw_lp_points.astype(np.float64) - points.astype(np.float64), axis=1
    ) * 1000.0
    raw_lp_source_mm = np.linalg.norm(
        raw_lp_points.astype(np.float64) - source_ref_m, axis=1
    ) * 1000.0
    output.mkdir(parents=True, exist_ok=False)
    raw_lp_path = output / "raw-lp-candidate.npz"
    np.savez_compressed(
        raw_lp_path,
        **build_raw_lp_archive(base, raw_lp_points, support, d, input_hashes),
    )
    raw_lp_receipt = {
        "path": str(raw_lp_path), "sha256": sha(raw_lp_path),
        "semantics": "unbacktracked LP result saved before alpha backtracking; diagnostic only",
        "qualified": False,
        "maximum_node_step_mm": float(raw_lp_step_mm.max(initial=0.0)),
        "maximum_source_displacement_mm": float(raw_lp_source_mm.max(initial=0.0)),
        "clearance_progress": clearance_progress_summary(
            clearance_C, clearance_h, clearance_weights,
            ((raw_lp_points[support].astype(np.float64) - points[support].astype(np.float64)) * 1000.0).reshape(-1),
            clearance_metadata,
        ),
        "solver_delta_shape_mm": list(d.shape),
        "solver_delta_max_abs_mm": float(np.max(np.abs(d), initial=0.0)),
        "area_scp_iterations": area_scp_records,
        "area_scp_failure": scp_failure,
        "full_mesh_f32_area_gate": raw_lp_area_gate,
        "exact_global_self_audit": "NOT_RUN",
        "all_185_bank_point_query": "NOT_RUN",
        "full_surface_bone_audit": "NOT_RUN",
    }
    alpha_failure_witnesses = []
    alpha_proxy_diagnostics = []
    alpha_used = None
    candidate_points = None
    emitted_metrics = None
    for exponent in range(0, 18):
        alpha = 0.5 ** exponent
        trial_points = points.copy()
        trial_points[support] = ((points_mm[support] + alpha * d) * 0.001).astype(np.float32)
        trial_points[local_anchors] = points[local_anchors]
        assert_prefix_bitwise_unchanged(points, trial_points, fixed_prefix_nodes)
        actual_d = ((trial_points[support].astype(np.float64) - points[support].astype(np.float64)) * 1000.0).reshape(-1)
        slack = np.asarray(A @ actual_d - b).ravel()
        full_area = full_f32_area_gate(
            trial_points, points_mm, area_reference_mm, faces, float(args.area_floor)
        )
        clearance_projection = np.asarray(clearance_C @ actual_d).reshape(-1)
        min_clearance_progress_mm = float(clearance_projection.min(initial=np.inf))
        clearance_no_regression = bool(min_clearance_progress_mm >= -2e-5)
        step_norms = np.linalg.norm(trial_points.astype(np.float64) - points.astype(np.float64), axis=1) * 1000.0
        source_norms = np.linalg.norm(trial_points.astype(np.float64) - source_ref_m, axis=1) * 1000.0
        unchanged_outside = np.array_equal(
            trial_points[np.setdiff1d(np.arange(len(points)), support)].view(np.uint32),
            points[np.setdiff1d(np.arange(len(points)), support)].view(np.uint32),
        )
        anchors_unchanged = np.array_equal(
            trial_points[anchors].view(np.uint32), points[anchors].view(np.uint32)
        )
        bad_linear = np.flatnonzero(slack < -2e-5)
        area_linear_start = len(pair_rows)
        area_linear_stop = area_linear_start + len(area_faces)
        bad_pair_rows = bad_linear[bad_linear < area_linear_start]
        self_barrier_count = len(self_pair_rows)
        bad_fixed_boundary_rows = bad_pair_rows[bad_pair_rows >= self_barrier_count] - self_barrier_count
        bad_self_pair_rows = bad_pair_rows[bad_pair_rows < self_barrier_count]
        bad_area_linear_rows = bad_linear[(bad_linear >= area_linear_start) &
                                          (bad_linear < area_linear_stop)] - area_linear_start
        bad_edge_gradient_rows = bad_linear[bad_linear >= area_linear_stop] - area_linear_stop
        edge_gradient_slack = np.asarray(edge_gradient_A @ actual_d - edge_gradient_b).reshape(-1)
        edge_gradient_emission_pass = bool(np.min(edge_gradient_slack, initial=np.inf) >= -1e-7)
        bad_step_rows = np.flatnonzero(step_norms > float(args.trust_mm) + 1e-6)
        bad_source_rows = np.flatnonzero(source_norms > float(args.source_cap_mm) + 1e-6)
        proxy_record = {
            "alpha": float(alpha),
            "linearized_pair_row_count": int(len(bad_pair_rows)),
            "linearized_pair_row_indices_first_64": bad_pair_rows[:64].tolist(),
            "linearized_self_pair_row_count": int(len(bad_self_pair_rows)),
            "linearized_fixed_boundary_row_count": int(len(bad_fixed_boundary_rows)),
            "linearized_fixed_boundary_row_indices_first_64": bad_fixed_boundary_rows[:64].tolist(),
            "linearized_area_face_count": int(len(bad_area_linear_rows)),
            "linearized_area_face_rows_first_64": area_faces[bad_area_linear_rows[:64]].tolist(),
            "linearized_area_face_rows_truncated": bool(len(bad_area_linear_rows) > 64),
            "linearized_edge_gradient_row_count": int(len(bad_edge_gradient_rows)),
            "linearized_edge_gradient_rows_first_64": bad_edge_gradient_rows[:64].tolist(),
            "edge_gradient_emission_pass": edge_gradient_emission_pass,
            "linearized_proxy_pass": bool(len(bad_linear) == 0),
            "full_mesh_strict_f32_area_pass": bool(full_area["pass"]),
            "clearance_no_regression_f32": clearance_no_regression,
            "minimum_clearance_progress_mm": min_clearance_progress_mm,
        }
        alpha_proxy_diagnostics.append(proxy_record)
        f32_gate = f32_emission_gate(
            full_area, step_norms, source_norms, float(args.trust_mm),
            float(args.source_cap_mm), unchanged_outside, anchors_unchanged,
        )
        # Pair and SCP area halfspaces diagnose the LP model; the true emitted
        # full-mesh F32 area gate decides whether an alpha may be selected.
        ok = bool(f32_gate["pass"] and clearance_no_regression and edge_gradient_emission_pass)
        if not ok:
            alpha_failure_witnesses.append({
                "alpha": float(alpha),
                "linearized_pair_row_count": int(len(bad_pair_rows)),
                "linearized_pair_row_indices_first_64": bad_pair_rows[:64].tolist(),
                "linearized_self_pair_row_count": int(len(bad_self_pair_rows)),
                "linearized_fixed_boundary_row_count": int(len(bad_fixed_boundary_rows)),
                "linearized_fixed_boundary_row_indices_first_64": bad_fixed_boundary_rows[:64].tolist(),
                "linearized_area_face_count": int(len(bad_area_linear_rows)),
                "linearized_area_face_rows_first_64": area_faces[bad_area_linear_rows[:64]].tolist(),
                "linearized_area_face_rows_truncated": bool(len(bad_area_linear_rows) > 64),
                "linearized_edge_gradient_row_count": int(len(bad_edge_gradient_rows)),
                "linearized_edge_gradient_rows_first_64": bad_edge_gradient_rows[:64].tolist(),
                "edge_gradient_emission_pass": edge_gradient_emission_pass,
                "f32_current_orientation_face_count": int(full_area["bad_current_oriented_face_count"]),
                "f32_current_orientation_face_rows_first_64": full_area["bad_current_oriented_face_rows_first_64"],
                "f32_fixed_037_magnitude_face_count": int(full_area["bad_fixed_037_reference_face_count"]),
                "f32_fixed_037_magnitude_face_rows_first_64": full_area["bad_fixed_037_reference_face_rows_first_64"],
                "clearance_no_regression_f32": clearance_no_regression,
                "minimum_clearance_progress_mm": min_clearance_progress_mm,
                "trust_violation_node_rows_first_64": bad_step_rows[:64].tolist(),
                "source_cap_violation_node_rows_first_64": bad_source_rows[:64].tolist(),
                "outside_support_bitwise_unchanged": bool(unchanged_outside),
                "anchors_bitwise_unchanged": bool(anchors_unchanged),
                "strict_f32_emission_gate": f32_gate,
            })
        if ok:
            alpha_used = float(alpha)
            candidate_points = trial_points
            emitted_metrics = {
                "alpha": alpha_used,
                "linearized_proxy_diagnostics": proxy_record,
                "linearized_proxy_pass_not_required_for_alpha_selection": bool(proxy_record["linearized_proxy_pass"]),
                "edge_gradient_emission_pass": edge_gradient_emission_pass,
                "edge_gradient_minimum_emitted_slack_mm": float(np.min(edge_gradient_slack, initial=np.inf)),
                "strict_f32_emission_gate": f32_gate,
                "full_mesh_area_gate": full_area,
                "minimum_constrained_current_oriented_ratio": float(full_area["minimum_current_oriented_ratio"]),
                "minimum_fixed_037_reference_magnitude_ratio": float(full_area["minimum_fixed_037_reference_magnitude_ratio"]),
                "maximum_node_step_mm": float(step_norms.max(initial=0.0)),
                "maximum_source_displacement_mm": float(source_norms.max(initial=0.0)),
                "changed_nodes": int(np.count_nonzero(np.any(trial_points != points, axis=1))),
                "clearance_progress_after_f32_emission": clearance_progress_summary(
                    clearance_C, clearance_h, clearance_weights, actual_d, clearance_metadata
                ),
                "outside_support_bitwise_unchanged": bool(unchanged_outside),
                "anchors_bitwise_unchanged": bool(anchors_unchanged),
                "fixed_prefix_bitwise_unchanged": True,
            }
            break

    if candidate_points is None:
        final_status = "NO_POSITIVE_F32_FEASIBLE_STEP_BASE_PRESERVED"
        report = {
            "status": final_status,
            "classification": "construction-only; no candidate emitted because full-mesh F32 gates failed",
            "native_admission": False, "accepted_physical_steps": 0,
            "inputs": {**{k: str(v) for k, v in paths.items()}, **{k + "_sha256": v for k, v in input_hashes.items()}},
            "area_reference": area_reference_receipt,
            "source_reference": source_receipt,
            "fixed_boundary_receipt": fixed_boundary_receipt,
            "support": {"threatened_pairs": int(len(pairs)), "core_nodes": int(len(core_nodes)),
                        "core_faces": int(len(core_faces)), "area_faces_touching_free_support": int(len(area_faces)),
                        "support_nodes": int(len(support)), "free_support_nodes": int(len(free_support)),
                        "fixed_boundary_skin_faces": int(len(fixed_boundary_face_rows)),
                        "fixed_boundary_barrier_rows": int(len(fixed_boundary_rows)),
                        "fixed_prefix_nodes": fixed_prefix_nodes,
                        "fixed_prefix_support_rows": int(len(bounded_prefix_support)),
                        "fixed_prefix_semantics": "LP coordinate bounds are exactly zero; not reinterpreted as native anchors"},
            "constraints": {"area_floor": float(args.area_floor), "trust_mm": float(args.trust_mm),
                            "source_cap_mm": float(args.source_cap_mm), "pair_margin_fraction": float(args.margin_fraction),
                            "edge_gradient": edge_gradient_report},
            "solver": {**{k: v for k, v in solved.items() if k != "delta"}, "wall_seconds": solve_wall},
            "area_scp_iterations": area_scp_records,
            "area_scp_failure": scp_failure,
            "raw_unbacktracked_lp_candidate": raw_lp_receipt,
            "alpha_backtracking_failure_witnesses": alpha_failure_witnesses,
            "alpha_linearized_proxy_diagnostics": alpha_proxy_diagnostics,
            "candidate_path": None, "exact_global_self_audit": "NOT_RUN_REQUIRED", "global_exact_self_audit": "NOT_RUN",
            "all_185_bank_point_query": "NOT_RUN", "full_surface_bone_audit": "NOT_RUN",
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    final_status = "CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED"
    assert_prefix_bitwise_unchanged(points, candidate_points, fixed_prefix_nodes)
    out = {key: base[key] for key in base.files}
    out["points_world_m"] = candidate_points
    out["local_witness_cluster_222_delta_m"] = candidate_points.astype(np.float64) - points.astype(np.float64)
    out["local_witness_cluster_222_support_rows"] = support
    out["local_witness_cluster_222_core_nodes"] = core_nodes
    out["local_witness_cluster_222_core_face_rows"] = core_faces
    out["local_witness_cluster_222_area_face_rows"] = area_faces
    out["local_witness_cluster_222_pair_face_rows"] = pairs
    out["local_witness_cluster_222_clearance_support_rows"] = support
    out["local_witness_cluster_230_edge_gradient_json"] = np.asarray([json.dumps(edge_gradient_report, sort_keys=True)])
    final_support_delta_mm = ((candidate_points[support].astype(np.float64)
                               - points[support].astype(np.float64)) * 1000.0).reshape(-1)
    final_clearance_progress_mm = np.asarray(clearance_C @ final_support_delta_mm).reshape(-1)
    final_clearance_slack_fraction = np.maximum(
        0.0, 1.0 - np.divide(final_clearance_progress_mm, clearance_h,
                             out=np.zeros_like(final_clearance_progress_mm), where=clearance_h > 0.0)
    )
    out["local_witness_cluster_222_clearance_emitted_slack_fraction"] = final_clearance_slack_fraction
    out["local_witness_cluster_222_clearance_lp_slack_fraction"] = np.asarray(solved.get("normalized_slack", np.empty(0)), dtype=np.float64)
    out["local_witness_cluster_222_clearance_h_mm"] = clearance_h
    out["local_witness_cluster_222_clearance_weights"] = clearance_weights
    out["local_witness_cluster_222_clearance_metadata_json"] = np.asarray([json.dumps(clearance_metadata, sort_keys=True)])
    out["local_witness_cluster_222_clearance_target_receipts_json"] = np.asarray([json.dumps(clearance_target_receipts, sort_keys=True)])
    out["local_witness_cluster_222_source_reference_R0_m"] = source_ref_m
    out["local_witness_cluster_222_area_reference_037_m"] = area_reference_m
    out["local_witness_cluster_222_input_hashes_json"] = np.asarray([json.dumps(input_hashes, sort_keys=True)])
    out["local_witness_cluster_222_semantics"] = np.asarray([
        "source R0 is used only for source-cap accounting; fixed accepted-037 points are used only for area constraints; exact global audits not run"
    ])
    if fixed_boundary_enabled:
        out["local_witness_cluster_315_fixed_boundary_typed_records_json"] = np.asarray([
            json.dumps(fixed_boundary_typed_records, sort_keys=True)
        ])
        out["local_witness_cluster_315_fixed_boundary_receipt_json"] = np.asarray([
            json.dumps(fixed_boundary_receipt, sort_keys=True)
        ])
    candidate_path = output / "candidate.npz"
    np.savez_compressed(candidate_path, **out)

    final_source_distance = np.linalg.norm(candidate_points.astype(np.float64) - source_ref_m, axis=1) * 1000.0
    final_step = np.linalg.norm(candidate_points.astype(np.float64) - points.astype(np.float64), axis=1) * 1000.0
    final_area_gate = full_f32_area_gate(candidate_points, points_mm, area_reference_mm, faces, float(args.area_floor))
    report = {
        "status": final_status,
        "classification": "construction-only F32 area/cap/trust proposal; exact self and other global gates pending",
        "native_admission": False,
        "accepted_physical_steps": 0,
        "inputs": {**{k: str(v) for k, v in paths.items()}, **{k + "_sha256": v for k, v in input_hashes.items()}},
        "source_reference_R0_for_cap_only": source_receipt,
        "area_reference_037_for_area_constraints_only": area_reference_receipt,
        "fixed_boundary_receipt": fixed_boundary_receipt,
        "area_reference_shape": list(area_reference_m.shape),
        "source_reference_shape": list(source_ref_m.shape),
        "source_reference_max_displacement_before_mm": float(all_source_distance.max(initial=0.0)),
        "source_reference_max_displacement_after_mm": float(final_source_distance.max(initial=0.0)),
        "source_cap_mm": float(args.source_cap_mm),
        "support": {
            "threatened_pairs": int(len(pairs)),
            "core_nodes": int(len(core_nodes)),
            "core_faces": int(len(core_faces)),
            "area_faces_touching_free_support": int(len(area_faces)),
            "free_support_nodes": int(len(free_support)),
            "support_nodes": int(len(support)),
            "fixed_anchor_rows_total": int(len(anchors)),
            "fixed_anchor_rows_in_support": local_anchors.tolist(),
            "fixed_prefix_nodes": fixed_prefix_nodes,
            "fixed_prefix_support_rows": int(len(bounded_prefix_support)),
            "fixed_prefix_semantics": "LP coordinate bounds are exactly zero; not reinterpreted as native anchors",
        },
        "constraints": {
            "pair_rows": int(len(pair_rows)),
            "self_pair_rows": int(len(self_pair_rows)),
            "fixed_boundary_barrier_rows": int(len(fixed_boundary_rows)),
            "fixed_boundary_typed_pair_count": int(len(fixed_boundary_records_input)),
            "fixed_boundary_omitted_fixed_only_rows": int(sum(
                int(record.get("omitted_fixed_only_rows", 0)) for record in fixed_boundary_typed_records
            )),
            "area_rows": int(Gq.shape[0]),
            "edge_gradient_rows": int(edge_gradient_A.shape[0]),
            "edge_gradient": edge_gradient_report,
            "combined_rows": int(A.shape[0]),
            "point_clearance_rows": int(len(point_targets["point_rows"])),
            "face_clearance_rows": int(len(face_targets["face_rows"])),
            "clearance_linear_rows": int(clearance_C.shape[0]),
            "clearance_slack_objective": solved.get("normalized_slack_objective"),
            "clearance_slack_tolerance": float(args.slack_tolerance),
            "clearance_progress": clearance_progress_summary(
                clearance_C, clearance_h, clearance_weights, d, clearance_metadata
            ),
            "clearance_target_receipts": clearance_target_receipts,
            "area_floor": float(args.area_floor),
            "pair_margin_fraction": float(args.margin_fraction),
            "trust_mm": float(args.trust_mm),
            "component_bound_minmax_mm": [float(np.min(lower)), float(np.max(upper))],
            "source_safe_box": bound_report,
            "final_linearized_zero_total_delta_slack_mm": float(np.min(-b, initial=np.inf)),
            "actual_pair_records": pair_records,
            "fixed_boundary_barrier_records": fixed_boundary_typed_records,
            "fixed_boundary_receipt": fixed_boundary_receipt,
            "area_linearization_layout_check": area_layout,
            "area_reference_semantics": "required=max(current mesh area, accepted-037 fixed reference area); R0 source field excluded",
            "global_f32_area_gate": final_area_gate,
        },
        "solver": {**{k: v for k, v in solved.items() if k not in ("delta", "normalized_slack")},
                   "wall_seconds": solve_wall,
                   "method": "lexicographic sparse HiGHS: minimum weighted normalized point/face target slack, then minimum L1 distance to raw trial; fixed pair rows, relinearized hard area rows"},
        "area_scp_iterations": area_scp_records,
        "area_scp_failure": scp_failure,
        "initial_base_feasibility_assertion_max_violation_mm": initial_baseline_max_violation,
        "emitted_f32_checks": emitted_metrics,
        "raw_unbacktracked_lp_candidate": raw_lp_receipt,
        "alpha_backtracking_failure_witnesses": alpha_failure_witnesses,
        "alpha_linearized_proxy_diagnostics": alpha_proxy_diagnostics,
        "exact_global_self_audit": "NOT_RUN_REQUIRED",
        "exact_global_self_audit_required_before_any_admission": True,
        "candidate_admission": "BLOCKED_PENDING_EXACT_GLOBAL_SELF_AND_OTHER_GATES",
        "maximum_emitted_node_step_mm": float(final_step.max(initial=0.0)),
        "maximum_source_displacement_mm": float(final_source_distance.max(initial=0.0)),
        "candidate_path": str(candidate_path),
        "candidate_sha256": sha(candidate_path),
        "global_exact_self_audit": "NOT_RUN",
        "all_185_bank_point_query": "NOT_RUN",
        "full_surface_bone_audit": "NOT_RUN",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def self_test() -> None:
    # Optional edge-gradient rows use node-major xyz coefficients.
    ep = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]])
    ef = np.array([[0, 1, 2], [1, 0, 2]], dtype=np.int64)
    eA, eb, em = build_edge_gradient_constraints(ep, ef, np.array([0, 1]), 0.25)
    if em["active_edges"] != 3 or eA.shape != (18, 6) or len(eb) != 18:
        raise AssertionError(f"edge-gradient rows failed unique-edge/dimension checks: {em}, {eA.shape}")
    dpass = np.array([0., 0., 0., 0.2, 0., 0.])
    dfail = np.array([0., 0., 0., 0.3, 0., 0.])
    if np.min(eA @ dpass - eb) < -1e-12 or np.min(eA @ dfail - eb) >= -1e-12:
        raise AssertionError("edge-gradient component cap did not separate feasible and violating displacements")
    try:
        build_edge_gradient_constraints(np.array([[0.,0.,0.],[0.,0.,0.],[0.,1.,0.]]),
                                        np.array([[0,1,2]], dtype=np.int64), np.array([0]), 0.25)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-length active edge was accepted")
    try:
        build_edge_gradient_constraints(ep, ef, np.array([0, 0]), 0.25)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate edge-gradient support row was accepted")
    # Input report identity is fail-closed.
    _require_target_candidate({"inputs": {"candidate_sha256": "a" * 64}}, Path("point.json"), "a" * 64)
    try:
        _require_target_candidate({"inputs": {"candidate_sha256": "a" * 64}}, Path("face.json"), "b" * 64)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched target report candidate SHA was accepted")
    # Lexicographic clearance LP: feasible target reaches the requested halfspace,
    # then stage 2 stays closest to the raw proposal.
    Ctoy = sparse.csr_matrix([[1.0]])
    feasible = solve_sparse_l1_clearance(
        sparse.csr_matrix((0, 1)), np.empty(0), np.array([0.0]),
        np.array([-1.0]), np.array([1.0]), Ctoy, np.array([0.5]), np.array([1.0]), 5.0,
    )
    if not feasible["success"] or not np.allclose(feasible["delta"], [0.5], atol=1e-7) or feasible["normalized_slack_objective"] > 1e-7:
        raise AssertionError(f"feasible clearance target was not met lexicographically: {feasible}")
    # A box that prevents progress retains the base and reports unit normalized slack.
    blocked = solve_sparse_l1_clearance(
        sparse.csr_matrix((0, 1)), np.empty(0), np.array([0.0]),
        np.array([-0.1]), np.array([0.0]), Ctoy, np.array([0.5]), np.array([1.0]), 5.0,
    )
    if not blocked["success"] or abs(float(blocked["delta"][0])) > 1e-8 or abs(float(blocked["normalized_slack"][0]) - 1.0) > 1e-7:
        raise AssertionError(f"infeasible target progress was not retained as explicit slack: {blocked}")
    # Opposed targets demonstrate that stage 1 minimizes normalized unmet
    # fractions before the raw L1 target can trade them against one another.
    opposed = solve_sparse_l1_clearance(
        sparse.csr_matrix((0, 1)), np.empty(0), np.array([0.8]),
        np.array([-1.0]), np.array([1.0]), sparse.csr_matrix([[1.0], [-1.0]]),
        np.array([1.0, 1.0]), np.array([0.5, 0.5]), 5.0,
    )
    if not opposed["success"] or abs(float(opposed["delta"][0])) > 1e-7 or abs(opposed["normalized_slack_objective"] - 1.0) > 1e-7:
        raise AssertionError(f"normalized target conflict was not lexicographically balanced: {opposed}")
    # Equivalent rows expressed in different coefficient units have identical
    # normalized slack semantics.
    scaled = solve_sparse_l1_clearance(
        sparse.csr_matrix((0, 1)), np.empty(0), np.array([0.0]),
        np.array([-1.0]), np.array([1.0]), sparse.csr_matrix([[1.0], [100.0]]),
        np.array([1.0, 100.0]), np.array([0.5, 0.5]), 5.0,
    )
    if not scaled["success"] or abs(float(scaled["delta"][0]) - 1.0) > 1e-7 or scaled["normalized_slack_objective"] > 1e-7:
        raise AssertionError(f"row-unit scaling changed normalized clearance: {scaled}")
    # Empty target lists preserve the previous sparse L1 contract.
    empty = solve_sparse_l1_clearance(
        sparse.csr_matrix([[1.0]]), np.array([0.25]), np.array([0.0]),
        np.array([-1.0]), np.array([1.0]), sparse.csr_matrix((0, 1)),
        np.empty(0), np.empty(0), 5.0,
    )
    if not empty["success"] or not np.allclose(empty["delta"], [0.25], atol=1e-8):
        raise AssertionError(f"empty clearance target set changed L1 behavior: {empty}")
    try:
        solve_sparse_l1_clearance(
            sparse.csr_matrix((0, 2)), np.empty(0), np.zeros(2), -np.ones(2), np.ones(2),
            sparse.csr_matrix((1, 1)), np.ones(1), np.ones(1), 5.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched clearance row dimensions were accepted")
    # Sparse target assembly has the expected point/face coordinate dimensions.
    class _BaryToy:
        @staticmethod
        def build_barycentric_exit_system(points, faces, rows, weights, normals, targets, *, anchor_rows=()):
            tri = np.asarray(faces, dtype=np.int64)[rows]
            B = sparse.coo_matrix((np.asarray(weights).ravel(),
                (np.repeat(np.arange(len(rows)), 3), tri.ravel())), shape=(len(rows), len(points))).tocsr()
            delta = np.asarray(targets) - np.asarray(B @ points)
            lower = np.einsum("ij,ij->i", normals, delta)
            g = np.asarray(weights)[:, :, None] * np.asarray(normals)[:, None, :]
            return B, tri, g, lower, delta
    toy_points = np.array([[0.,0.,0.],[0.5,0.,0.],[0.,1.,0.]])
    toy_faces = np.array([[0,1,2]])
    pt = {"point_rows": np.array([1]), "point_targets_f32_m": np.array([[0.001,0.,0.]],np.float32),
          "point_directions": np.array([[1.,0.,0.]])}
    ft = {"face_rows": np.array([0]), "barycentric_weights": np.array([[1/3,1/3,1/3]]),
          "face_targets_f32_m": np.array([[1/3,1/3,0.001]],np.float32),
          "face_directions": np.array([[0.,0.,1.]]), "face_samples_m": np.array([[1/6/1000,1/3/1000,0.]]),
          "face_npz": {"bank_body": np.array([1]), "bank_stable_bone_id": np.array([2])}}
    Ct, ht, wt, mt = build_clearance_rows(toy_points, toy_faces, np.array([0,1,2]),
                                          np.array([2]), pt, ft, _BaryToy)
    if Ct.shape != (2, 9) or ht.shape != (2,) or wt.shape != (2,) or [x["kind"] for x in mt] != ["point", "face"]:
        raise AssertionError("point/face sparse target rows have incorrect dimensions or metadata")
    if np.any(Ct[:, 6:9].toarray() != 0):
        raise AssertionError("fixed anchor retained a clearance coefficient")
    # Sparse L1 projection should satisfy x >= 0.5 while remaining closest to 0.
    A = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    result = solve_sparse_l1(A, np.array([0.5, -0.25]), np.zeros(2),
                             np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 5.0)
    if not result["success"] or not np.allclose(result["delta"], [0.5, 0.0], atol=1e-8):
        raise AssertionError(f"HiGHS L1 toy projection failed: {result}")
    # Raw LP serialization preserves all base fields, F32 point domain and explicit unqualified delta.
    base_points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
    support = np.array([1], dtype=np.int64)
    anchors = np.array([2], dtype=np.int64)
    delta_mm = np.array([[0.125, -0.25, 0.0]], dtype=np.float64)
    emitted = emit_raw_lp_points(base_points, base_points.astype(np.float64) * 1000.0,
                                 support, delta_mm, anchors)
    if emitted.dtype != np.float32 or not np.array_equal(emitted[[0, 2]].view(np.uint32), base_points[[0, 2]].view(np.uint32)):
        raise AssertionError("raw LP emitter changed non-support or pinned F32 rows")
    archive = build_raw_lp_archive({"points_world_m": base_points, "marker": np.array([7])},
                                   emitted, support, delta_mm, {"test": "sha"})
    with tempfile.TemporaryDirectory() as td:
        raw_path = Path(td) / "raw.npz"
        np.savez_compressed(raw_path, **archive)
        with np.load(raw_path, allow_pickle=False) as check:
            if not np.array_equal(check["points_world_m"], emitted) or bool(check["local_witness_cluster_222_lp_qualified"][0]):
                raise AssertionError("raw LP archive serialization/qualification marker failed")
            if not np.array_equal(check["local_witness_cluster_222_lp_delta_mm_f64"], delta_mm):
                raise AssertionError("raw LP requested delta was not preserved")
    # A full-mesh F32 ratio 1e-6 below the strict floor fails (old -2e-6 slack would pass).
    tri = np.array([[0, 1, 2]], dtype=np.int64)
    cur_m = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
    ref_mm = cur_m.astype(np.float64) * 1000.0
    below = cur_m.copy(); below[2, 1] = np.float32(0.099999)
    above = cur_m.copy(); above[2, 1] = np.float32(0.100001)
    strict_below = full_f32_area_gate(below, ref_mm, ref_mm, tri, 0.1)
    strict_above = full_f32_area_gate(above, ref_mm, ref_mm, tri, 0.1)
    if strict_below["pass"] or not strict_above["pass"]:
        raise AssertionError("strict F32 area gate did not enforce the declared floor")
    # Linearized pair/area proxies are intentionally not arguments to this gate.
    gate = f32_emission_gate(strict_above, np.array([0.2]), np.array([3.0]), 1.0, 12.0, True, True)
    proxy_diagnostic = {"linearized_proxy_pass": False, "failed_proxy_face_rows": [14]}
    if not gate["pass"] or proxy_diagnostic["linearized_proxy_pass"]:
        raise AssertionError("proxy failure incorrectly controls strict F32 emission selection")
    if gate["exact_global_self_audit"] != "NOT_RUN_REQUIRED" or gate["native_admission"]:
        raise AssertionError("strict emission gate must leave exact self audit and admission pending")
    # The conservative component box gives a norm no larger than either cap.
    p = np.zeros((2, 3), dtype=np.float64)
    ref = np.array([[0.0, 0.0, 0.0], [9.9, 0.0, 0.0]])
    lo, hi, _ = source_safe_trust_bounds(p, ref, np.array([0, 1]), np.array([], dtype=np.int64), 1.0, 12.0)
    for sign in (-1.0, 1.0):
        d = np.where(sign > 0, hi, lo).reshape(-1, 3)
        if np.any(np.linalg.norm(d, axis=1) > 1.0 + 1e-12) or np.any(np.linalg.norm(p + d - ref, axis=1) > 12.0 + 1e-12):
            raise AssertionError("conservative trust/source box guarantee failed")
    # Geometry area targets must use fixed 037, while source R0 remains separate for cap bounds.
    f = np.array([[0, 1, 2]], dtype=np.int64)
    current = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    fixed037 = current.copy(); fixed037[1:] *= 2.0
    source_R0 = current.copy(); source_R0[1:] *= 4.0
    _, _, a037, required = fixed_reference_area_requirements(current, fixed037, f)
    _, _, aR0, _ = fixed_reference_area_requirements(current, source_R0, f)
    if not np.allclose(a037, [4.0]) or not np.allclose(aR0, [16.0]) or not np.allclose(required, [4.0]):
        raise AssertionError("area reference accidentally uses R0 instead of the independent fixed-037 geometry")
    # Absolute-vs-incremental area RHS conversion is algebraically exact.
    Gtoy = sparse.csr_matrix([[2.0, -1.0, 0.5, 0.0, 3.0, -2.0]])
    support_toy = np.array([0, 1], dtype=np.int64)
    dcur = np.array([0.1, 0.2, -0.1, 0.0, -0.05, 0.03])
    rhs_q = np.array([0.75])
    rhs_abs = absolute_area_rhs_from_incremental(Gtoy, rhs_q, support_toy, dcur)
    increment = np.array([-0.2, 0.1, 0.0, 0.04, 0.02, -0.01])
    if not np.allclose(Gtoy @ (dcur + increment) - rhs_abs, Gtoy @ increment - rhs_q):
        raise AssertionError("SCP incremental-to-absolute area RHS conversion failed")
    # Layout and directional derivative parity use the real area owner at a nonzero center.
    area_owner = load_module("area_layout_test_owner_212", DEFAULT_AREA_OWNER)
    check = verify_area_linearization_layout(area_owner)
    if check["relative_error"] > 2e-7:
        raise AssertionError(f"area owner layout check failed: {check}")
    pts = np.array([[0.2, -0.1, 0.3], [1.4, 0.2, 0.1], [-0.1, 1.1, 0.4]], dtype=np.float64) * 1000.0
    face = np.array([[0, 1, 2]], dtype=np.int64)
    base_cross = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    base_n = base_cross / np.linalg.norm(base_cross)
    base_area = np.linalg.norm(base_cross)
    req = np.array([base_area * 1.1])
    shift = np.array([[0.01, -0.02, 0.0], [0.0, 0.01, 0.03], [-0.01, 0.0, 0.02]]) * 1000.0
    q = pts + shift
    Gq, rhsq, _, _ = area_owner.area_linearization(q, face, np.array([0]), base_n[None, :], req, minimum_ratio=0.1)
    cols = interleaved_coordinate_columns(np.array([2, 0, 1]))
    delta_fd = np.array([[0.03, -0.02, 0.01], [-0.04, 0.01, 0.02], [0.02, 0.05, -0.03]])
    predicted = float(np.asarray(Gq[:, cols] @ delta_fd[[2, 0, 1]].reshape(-1))[0])
    eps = 1e-3
    q2p = q + eps * delta_fd
    q2m = q - eps * delta_fd
    actual = (float(np.dot(base_n, np.cross(q2p[1] - q2p[0], q2p[2] - q2p[0])))
              - float(np.dot(base_n, np.cross(q2m[1] - q2m[0], q2m[2] - q2m[0])))) / (2.0 * eps)
    if not np.isclose(predicted, actual, rtol=3e-7, atol=2e-6):
        raise AssertionError(f"area SCP relinearization derivative mismatch: {predicted} vs {actual}; rhs={rhsq}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-candidate", required=False)
    parser.add_argument("--raw-trial", required=False)
    parser.add_argument("--self-pairs", required=False)
    parser.add_argument("--output", required=False)
    parser.add_argument("--source-base-npz", default=None)
    parser.add_argument("--area-reference-npz", default=None)
    parser.add_argument("--area-reference-points-key", default="points_world_m")
    parser.add_argument("--area-reference-faces-key", default="source_O_faces")
    parser.add_argument("--pair-owner", default=str(DEFAULT_PAIR_OWNER))
    parser.add_argument("--area-owner", default=str(DEFAULT_AREA_OWNER))
    parser.add_argument("--bary-owner", default=str(DEFAULT_BARY_OWNER))
    parser.add_argument("--point-target-npz", required=False)
    parser.add_argument("--point-target-report", required=False)
    parser.add_argument("--face-target-npz", required=False)
    parser.add_argument("--face-target-report", required=False)
    parser.add_argument("--slack-tolerance", type=float, default=1e-9)
    parser.add_argument("--trust-mm", type=float, default=TRUST_MM)
    parser.add_argument("--source-cap-mm", type=float, default=SOURCE_CAP_MM)
    parser.add_argument("--area-floor", type=float, default=AREA_FLOOR)
    parser.add_argument("--margin-fraction", type=float, default=PAIR_MARGIN_FRACTION)
    parser.add_argument("--time-limit-s", type=float, default=30.0)
    parser.add_argument("--area-scp-iterations", type=int, default=4)
    parser.add_argument("--max-edge-gradient", type=float, default=0.0,
                        help="optional componentwise displacement-gradient cap in mm/mm; zero preserves 222 behavior")
    parser.add_argument("--fixed-prefix-nodes", type=int, default=0,
                        help="fix displacement to exact zero for node rows [0,count); does not change native-anchor semantics")
    parser.add_argument("--fixed-boundary-npz", default=None,
                        help="optional 314 typed fixed skin-bone triangle export NPZ")
    parser.add_argument("--fixed-boundary-report", default=None,
                        help="optional report.json paired with --fixed-boundary-npz")
    parser.add_argument("--fixed-boundary-owner", default=None,
                        help="optional pinned 313 fixed triangle row-builder source")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print(json.dumps({"status": "PASS", "self_test": "lexicographic normalized clearance slacks; report candidate pins; sparse row dimensions and anchor elimination; empty target parity; strict F32 area selector; bounded area-SCP RHS/FD; raw LP serialization; source/trust box; fixed-037 area reference"}, indent=2))
        return
    if not all((args.base_candidate, args.raw_trial, args.self_pairs, args.output, args.area_reference_npz,
                args.point_target_npz, args.point_target_report, args.face_target_npz, args.face_target_report)):
        parser.error("base-candidate, raw-trial, self-pairs, area-reference-npz, both target NPZ/report pairs, and output are required unless --self-test")
    fixed_boundary_args = (args.fixed_boundary_npz, args.fixed_boundary_report, args.fixed_boundary_owner)
    if any(fixed_boundary_args) and not all(fixed_boundary_args):
        parser.error("fixed-boundary-npz, fixed-boundary-report, and fixed-boundary-owner must be supplied together")
    if args.trust_mm <= 0 or args.source_cap_mm <= 0 or args.time_limit_s <= 0 or args.slack_tolerance < 0:
        parser.error("trust, source cap, time limit, and slack tolerance must be positive/nonnegative")
    if args.area_scp_iterations < 1 or args.area_scp_iterations > 4:
        parser.error("area-scp-iterations must be in [1,4]")
    if not np.isfinite(args.max_edge_gradient) or args.max_edge_gradient < 0:
        parser.error("max-edge-gradient must be finite and nonnegative")
    try:
        validate_fixed_prefix_nodes(args.fixed_prefix_nodes, int(np.load(args.base_candidate, mmap_mode="r", allow_pickle=False)["points_world_m"].shape[0]))
    except Exception as exc:
        parser.error(str(exc))
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
