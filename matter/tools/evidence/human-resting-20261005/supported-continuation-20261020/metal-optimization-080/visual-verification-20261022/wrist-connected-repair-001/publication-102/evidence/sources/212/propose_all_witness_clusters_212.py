#!/usr/bin/env python3
"""Sparse joint local proposal for all exact self-pair witness clusters,
with an independently pinned fixed-037 area reference.

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
    bad_oriented = np.flatnonzero(oriented < floor - 2e-6)
    bad_reference = np.flatnonzero(reference_magnitude < floor - 2e-6)
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
    out["local_witness_cluster_212_lp_delta_mm_f64"] = delta
    out["local_witness_cluster_212_lp_support_rows"] = rows
    out["local_witness_cluster_212_lp_emitted_delta_m_f32"] = raw - old_points
    out["local_witness_cluster_212_lp_input_hashes_json"] = np.asarray([json.dumps(input_hashes, sort_keys=True)])
    out["local_witness_cluster_212_lp_qualified"] = np.asarray([False])
    out["local_witness_cluster_212_lp_semantics"] = np.asarray([
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


def run(args) -> dict:
    paths = {
        "base_candidate": Path(args.base_candidate),
        "raw_trial": Path(args.raw_trial),
        "self_pairs": Path(args.self_pairs),
        "area_reference": Path(args.area_reference_npz),
        "pair_owner": Path(args.pair_owner),
        "area_owner": Path(args.area_owner),
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output path: {output}")
    input_hashes = {k: sha(v) for k, v in paths.items()}
    base = np.load(paths["base_candidate"], allow_pickle=False)
    raw = np.load(paths["raw_trial"], allow_pickle=False)
    points = np.asarray(base["points_world_m"], dtype=np.float32)
    points_mm = points.astype(np.float64) * 1000.0
    target_points = np.asarray(raw["points_world_m"], dtype=np.float32)
    faces = np.asarray(base["source_O_faces"], dtype=np.int64)
    raw_faces = np.asarray(raw["source_O_faces"], dtype=np.int64)
    if target_points.shape != points.shape or not np.array_equal(faces, raw_faces):
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
    pairs = np.asarray(np.load(paths["self_pairs"], allow_pickle=False), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
        raise ValueError("self-pair input must be a nonempty (M,2) face-row array")
    pairs = np.asarray(sorted({tuple(sorted(map(int, p))) for p in pairs}), dtype=np.int64)
    if np.any(pairs < 0) or np.any(pairs >= len(faces)) or np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("self-pair face row out of range or self-pair")

    source_base_path = Path(args.source_base_npz) if args.source_base_npz else None
    source_ref_m, source_receipt = compose_source_reference(base, source_base_path)
    source_ref_mm = source_ref_m * 1000.0
    all_source_distance = np.linalg.norm(points.astype(np.float64) * 1000.0 - source_ref_mm, axis=1)
    if float(all_source_distance.max(initial=0.0)) > args.source_cap_mm + 1e-7:
        raise ValueError("base candidate exceeds source cap before proposal")

    core_nodes = np.unique(faces[pairs].ravel()).astype(np.int64)
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
    free_support = np.setdiff1d(support, local_anchors, assume_unique=True)
    if len(free_support) == 0:
        raise ValueError("threatened one-ring has no free support nodes")
    # Every triangle touched by a movable support node participates in the
    # local linearized area system. Nodes outside support remain fixed at zero.
    area_faces = np.flatnonzero(np.isin(faces, free_support).any(axis=1)).astype(np.int64)
    if len(area_faces) == 0:
        raise ValueError("no area faces touch a free support node")

    pair_owner = load_module("self_separation_owner_199", paths["pair_owner"])
    area_owner = load_module("area_linearization_owner_199", paths["area_owner"])
    area_layout = verify_area_linearization_layout(area_owner)
    pair_rows, pair_records = pair_owner.build_pair_rows(
        points_mm, faces, pairs, margin_fraction=float(args.margin_fraction)
    )

    normals_all, area0, area037, required_all = fixed_reference_area_requirements(
        points_mm, area_reference_mm, faces
    )
    normals = normals_all[area_faces]
    required = required_all[area_faces]
    lower, upper, bound_report = source_safe_trust_bounds(
        points_mm, source_ref_mm, support, local_anchors,
        float(args.trust_mm), float(args.source_cap_mm),
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
        base_violation = float(np.max(b, initial=-np.inf))
        if scp_index == 0:
            initial_baseline_max_violation = base_violation
            if base_violation > 2e-8:
                raise ValueError(f"base candidate infeasible for initial local pair/area rows: {base_violation} mm")
        t0 = time.monotonic()
        one = solve_sparse_l1(A, b, target_delta, lower, upper, float(args.time_limit_s))
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
            "minimum_normalized_linear_slack": one.get("minimum_normalized_linear_slack"),
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
                        "support_nodes": int(len(support)), "free_support_nodes": int(len(free_support))},
            "area_scp_iterations": area_scp_records, "area_scp_failure": scp_failure,
            "solver": {k: v for k, v in solved.items() if k != "delta"},
            "wall_seconds": solve_wall,
        }
        output.mkdir(parents=True, exist_ok=False)
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    d = np.asarray(solved["delta"], dtype=np.float64).reshape(-1, 3)
    raw_lp_points = emit_raw_lp_points(points, points_mm, support, d, local_anchors)
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
    alpha_used = None
    candidate_points = None
    emitted_metrics = None
    for exponent in range(0, 18):
        alpha = 0.5 ** exponent
        trial_points = points.copy()
        trial_points[support] = ((points_mm[support] + alpha * d) * 0.001).astype(np.float32)
        trial_points[local_anchors] = points[local_anchors]
        actual_d = ((trial_points[support].astype(np.float64) - points[support].astype(np.float64)) * 1000.0).reshape(-1)
        slack = np.asarray(A @ actual_d - b).ravel()
        full_area = full_f32_area_gate(
            trial_points, points_mm, area_reference_mm, faces, float(args.area_floor)
        )
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
        bad_area_linear_rows = bad_linear[bad_linear >= len(pair_rows)] - len(pair_rows)
        bad_pair_rows = bad_linear[bad_linear < len(pair_rows)]
        bad_step_rows = np.flatnonzero(step_norms > float(args.trust_mm) + 1e-6)
        bad_source_rows = np.flatnonzero(source_norms > float(args.source_cap_mm) + 1e-6)
        ok = (
            len(bad_linear) == 0
            and full_area["pass"]
            and len(bad_step_rows) == 0
            and len(bad_source_rows) == 0
            and unchanged_outside and anchors_unchanged
        )
        if not ok:
            alpha_failure_witnesses.append({
                "alpha": float(alpha),
                "linearized_pair_row_count": int(len(bad_pair_rows)),
                "linearized_pair_row_indices_first_64": bad_pair_rows[:64].tolist(),
                "linearized_area_face_count": int(len(bad_area_linear_rows)),
                "linearized_area_face_rows_first_64": area_faces[bad_area_linear_rows[:64]].tolist(),
                "linearized_area_face_rows_truncated": bool(len(bad_area_linear_rows) > 64),
                "f32_current_orientation_face_count": int(full_area["bad_current_oriented_face_count"]),
                "f32_current_orientation_face_rows_first_64": full_area["bad_current_oriented_face_rows_first_64"],
                "f32_fixed_037_magnitude_face_count": int(full_area["bad_fixed_037_reference_face_count"]),
                "f32_fixed_037_magnitude_face_rows_first_64": full_area["bad_fixed_037_reference_face_rows_first_64"],
                "trust_violation_node_rows_first_64": bad_step_rows[:64].tolist(),
                "source_cap_violation_node_rows_first_64": bad_source_rows[:64].tolist(),
                "outside_support_bitwise_unchanged": bool(unchanged_outside),
                "anchors_bitwise_unchanged": bool(anchors_unchanged),
            })
        if ok:
            alpha_used = float(alpha)
            candidate_points = trial_points
            emitted_metrics = {
                "alpha": alpha_used,
                "minimum_emitted_normalized_linear_slack_mm": float(slack.min(initial=np.inf)),
                "full_mesh_area_gate": full_area,
                "minimum_constrained_current_oriented_ratio": float(full_area["minimum_current_oriented_ratio"]),
                "minimum_fixed_037_reference_magnitude_ratio": float(full_area["minimum_fixed_037_reference_magnitude_ratio"]),
                "maximum_node_step_mm": float(step_norms.max(initial=0.0)),
                "maximum_source_displacement_mm": float(source_norms.max(initial=0.0)),
                "changed_nodes": int(np.count_nonzero(np.any(trial_points != points, axis=1))),
                "outside_support_bitwise_unchanged": bool(unchanged_outside),
                "anchors_bitwise_unchanged": bool(anchors_unchanged),
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
            "support": {"threatened_pairs": int(len(pairs)), "core_nodes": int(len(core_nodes)),
                        "core_faces": int(len(core_faces)), "area_faces_touching_free_support": int(len(area_faces)),
                        "support_nodes": int(len(support)), "free_support_nodes": int(len(free_support))},
            "constraints": {"area_floor": float(args.area_floor), "trust_mm": float(args.trust_mm),
                            "source_cap_mm": float(args.source_cap_mm), "pair_margin_fraction": float(args.margin_fraction)},
            "solver": {**{k: v for k, v in solved.items() if k != "delta"}, "wall_seconds": solve_wall},
            "area_scp_iterations": area_scp_records,
            "area_scp_failure": scp_failure,
            "raw_unbacktracked_lp_candidate": raw_lp_receipt,
            "alpha_backtracking_failure_witnesses": alpha_failure_witnesses,
            "candidate_path": None, "global_exact_self_audit": "NOT_RUN",
            "all_185_bank_point_query": "NOT_RUN", "full_surface_bone_audit": "NOT_RUN",
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    final_status = "CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED"
    out = {key: base[key] for key in base.files}
    out["points_world_m"] = candidate_points
    out["local_witness_cluster_205_delta_m"] = candidate_points.astype(np.float64) - points.astype(np.float64)
    out["local_witness_cluster_205_support_rows"] = support
    out["local_witness_cluster_205_core_nodes"] = core_nodes
    out["local_witness_cluster_205_core_face_rows"] = core_faces
    out["local_witness_cluster_205_area_face_rows"] = area_faces
    out["local_witness_cluster_205_pair_face_rows"] = pairs
    out["local_witness_cluster_205_source_reference_R0_m"] = source_ref_m
    out["local_witness_cluster_205_area_reference_037_m"] = area_reference_m
    out["local_witness_cluster_205_input_hashes_json"] = np.asarray([json.dumps(input_hashes, sort_keys=True)])
    out["local_witness_cluster_205_semantics"] = np.asarray([
        "source R0 is used only for source-cap accounting; fixed accepted-037 points are used only for area constraints; exact global audits not run"
    ])
    candidate_path = output / "candidate.npz"
    np.savez_compressed(candidate_path, **out)

    final_source_distance = np.linalg.norm(candidate_points.astype(np.float64) - source_ref_m, axis=1) * 1000.0
    final_step = np.linalg.norm(candidate_points.astype(np.float64) - points.astype(np.float64), axis=1) * 1000.0
    final_area_gate = full_f32_area_gate(candidate_points, points_mm, area_reference_mm, faces, float(args.area_floor))
    report = {
        "status": final_status,
        "classification": "construction-only local joint projection; exact global gates pending",
        "native_admission": False,
        "accepted_physical_steps": 0,
        "inputs": {**{k: str(v) for k, v in paths.items()}, **{k + "_sha256": v for k, v in input_hashes.items()}},
        "source_reference_R0_for_cap_only": source_receipt,
        "area_reference_037_for_area_constraints_only": area_reference_receipt,
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
        },
        "constraints": {
            "pair_rows": int(len(pair_rows)),
            "area_rows": int(Gq.shape[0]),
            "combined_rows": int(A.shape[0]),
            "area_floor": float(args.area_floor),
            "pair_margin_fraction": float(args.margin_fraction),
            "trust_mm": float(args.trust_mm),
            "component_bound_minmax_mm": [float(np.min(lower)), float(np.max(upper))],
            "source_safe_box": bound_report,
            "final_linearized_zero_total_delta_slack_mm": float(np.min(-b, initial=np.inf)),
            "actual_pair_records": pair_records,
            "area_linearization_layout_check": area_layout,
            "area_reference_semantics": "required=max(current mesh area, accepted-037 fixed reference area); R0 source field excluded",
            "global_f32_area_gate": final_area_gate,
        },
        "solver": {**{k: v for k, v in solved.items() if k != "delta"}, "wall_seconds": solve_wall,
                   "method": "bounded area SCP (fixed pair rows, relinearized area rows), each step sparse HiGHS L1 closest to raw target"},
        "area_scp_iterations": area_scp_records,
        "area_scp_failure": scp_failure,
        "initial_base_feasibility_assertion_max_violation_mm": initial_baseline_max_violation,
        "emitted_f32_checks": emitted_metrics,
        "raw_unbacktracked_lp_candidate": raw_lp_receipt,
        "alpha_backtracking_failure_witnesses": alpha_failure_witnesses,
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
            if not np.array_equal(check["points_world_m"], emitted) or bool(check["local_witness_cluster_212_lp_qualified"][0]):
                raise AssertionError("raw LP archive serialization/qualification marker failed")
            if not np.array_equal(check["local_witness_cluster_212_lp_delta_mm_f64"], delta_mm):
                raise AssertionError("raw LP requested delta was not preserved")
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
    parser.add_argument("--trust-mm", type=float, default=TRUST_MM)
    parser.add_argument("--source-cap-mm", type=float, default=SOURCE_CAP_MM)
    parser.add_argument("--area-floor", type=float, default=AREA_FLOOR)
    parser.add_argument("--margin-fraction", type=float, default=PAIR_MARGIN_FRACTION)
    parser.add_argument("--time-limit-s", type=float, default=30.0)
    parser.add_argument("--area-scp-iterations", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print(json.dumps({"status": "PASS", "self_test": "sparse LP, bounded area-SCP RHS/FD, raw LP serialization/pins, source/trust box, distinct R0/037 area reference"}, indent=2))
        return
    if not all((args.base_candidate, args.raw_trial, args.self_pairs, args.output, args.area_reference_npz)):
        parser.error("base-candidate, raw-trial, self-pairs, area-reference-npz, and output are required unless --self-test")
    if args.trust_mm <= 0 or args.source_cap_mm <= 0 or args.time_limit_s <= 0:
        parser.error("trust, source cap, and time limit must be positive")
    if args.area_scp_iterations < 1 or args.area_scp_iterations > 4:
        parser.error("area-scp-iterations must be in [1,4]")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
