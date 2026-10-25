"""Construction-time thickness inequalities for TetGen-created O/I/D boundary nodes.

Rows are derived from marked source-parent facets and reference-space barycentrics.
This constrains projected layer spacing while leaving every generated node free in 3D.
It is not a material transfer or physical acceptance predicate.
"""
from __future__ import annotations

from collections import Counter
import numpy as np
from scipy import sparse


_SURFACE_LAYER = {"O": 0, "I": 1, "D": 2}


class GeneratedLayerConstraintError(ValueError):
    pass


def _barycentric(point: np.ndarray, triangle: np.ndarray) -> tuple[np.ndarray, float]:
    a, b, c = triangle
    e0 = b - a
    e1 = c - a
    rhs = point - a
    gram = np.array([[e0 @ e0, e0 @ e1], [e0 @ e1, e1 @ e1]], dtype=np.float64)
    det = float(np.linalg.det(gram))
    if not np.isfinite(det) or det <= 1e-30:
        raise GeneratedLayerConstraintError("degenerate source parent face")
    uv = np.linalg.solve(gram, np.array([e0 @ rhs, e1 @ rhs], dtype=np.float64))
    weights = np.array([1.0 - uv.sum(), uv[0], uv[1]], dtype=np.float64)
    projection = a + uv[0] * e0 + uv[1] * e1
    return weights, float(np.linalg.norm(point - projection))


def build_generated_boundary_layer_inequalities(
    reference_points_m: np.ndarray,
    current_points_m_f32: np.ndarray,
    source_input_row: np.ndarray,
    source_faces: dict[str, np.ndarray],
    child_faces: np.ndarray,
    child_markers: np.ndarray,
    marker_parent: dict[int, tuple[str, int]],
    *,
    rows_per_layer: int = 2411,
    thickness_bounds: tuple[float, float] = (0.5, 1.5),
    plane_tolerance_m: float = 2e-7,
    barycentric_tolerance: float = 2e-6,
    baseline_tolerance_m: float = 1e-10,
) -> tuple[sparse.csr_matrix, np.ndarray, dict]:
    """Return C, b for C @ flattened(current-or-trial xyz metres) >= b.

    For every generated point on a marked O/I/D child face, exact source-face
    barycentrics locate its opposing-layer point using matching source columns.
    The signed projection at the frozen Float32 baseline is bounded to
    [low, high] times its own baseline separation. No positional tie is added.
    All generated boundary nodes must have at least one O/I/D parent relation;
    generated interior nodes are excluded.
    """
    R = np.asarray(reference_points_m, dtype=np.float64)
    Xraw = np.asarray(current_points_m_f32)
    src = np.asarray(source_input_row)
    cf = np.asarray(child_faces)
    cm = np.asarray(child_markers)
    if R.ndim != 2 or R.shape[1] != 3 or not len(R) or not np.isfinite(R).all():
        raise GeneratedLayerConstraintError("reference points must be finite N x 3")
    if Xraw.dtype != np.float32 or Xraw.shape != R.shape or not np.isfinite(Xraw).all():
        raise GeneratedLayerConstraintError("baseline current positions must be emitted Float32 N x 3")
    X = Xraw.astype(np.float64)
    n = len(R)
    if src.dtype.kind not in "iu" or src.shape != (n,):
        raise GeneratedLayerConstraintError("source_input_row must be an integer row map")
    src = src.astype(np.int64, copy=False)
    if cf.dtype.kind not in "iu" or cf.ndim != 2 or cf.shape[1] != 3:
        raise GeneratedLayerConstraintError("child_faces must be an integer M x 3 table")
    cf = cf.astype(np.int64, copy=False)
    if cm.dtype.kind not in "iu" or cm.shape != (len(cf),):
        raise GeneratedLayerConstraintError("child_markers must be an integer vector matching faces")
    cm = cm.astype(np.int64, copy=False)
    if len(cf) and (np.any(cf < 0) or np.any(cf >= n) or np.any(np.diff(np.sort(cf, axis=1), axis=1) == 0)):
        raise GeneratedLayerConstraintError("invalid child face vertex IDs")
    if not isinstance(rows_per_layer, (int, np.integer)) or isinstance(rows_per_layer, (bool, np.bool_)) or rows_per_layer <= 0:
        raise GeneratedLayerConstraintError("rows_per_layer must be a positive integer")
    low, high = map(float, thickness_bounds)
    if not np.isfinite([low, high, plane_tolerance_m, barycentric_tolerance, baseline_tolerance_m]).all():
        raise GeneratedLayerConstraintError("bounds/tolerances must be finite")
    if low <= 0 or high <= low or min(plane_tolerance_m, barycentric_tolerance, baseline_tolerance_m) < 0:
        raise GeneratedLayerConstraintError("invalid thickness bounds or tolerances")

    needed = {"O", "I", "D"}
    if set(source_faces) != needed:
        raise GeneratedLayerConstraintError("source_faces must contain exactly O, I, and D")
    faces = {}
    for name in ("O", "I", "D"):
        raw = np.asarray(source_faces[name])
        if raw.dtype.kind not in "iu" or raw.ndim != 2 or raw.shape[1] != 3 or len(raw) == 0:
            raise GeneratedLayerConstraintError(f"source {name} faces must be a nonempty integer M x 3 table")
        faces[name] = raw.astype(np.int64, copy=False)
        if np.any(faces[name] < 0) or np.any(faces[name] >= 3 * rows_per_layer):
            raise GeneratedLayerConstraintError(f"source {name} face index is outside the three-layer source domain")
        if np.any(faces[name] // rows_per_layer != _SURFACE_LAYER[name]):
            raise GeneratedLayerConstraintError(f"source {name} faces do not use their declared layer rows")
    O = faces["O"]
    if faces["I"].shape != O.shape or faces["D"].shape != O.shape:
        raise GeneratedLayerConstraintError("O/I/D source face tables have different shapes")
    if not np.array_equal(faces["I"], O + rows_per_layer):
        raise GeneratedLayerConstraintError("I source faces do not preserve O source-column topology/order")
    if not np.array_equal(faces["D"], O + 2 * rows_per_layer):
        raise GeneratedLayerConstraintError("D source faces do not preserve O source-column topology/order")
    nsource = 3 * rows_per_layer
    original = np.flatnonzero(src >= 0)
    if np.any(src < -1) or len(original) != nsource or np.any(src[original] >= nsource):
        raise GeneratedLayerConstraintError("source row map must contain every source node exactly once")
    if len(np.unique(src[original])) != nsource:
        raise GeneratedLayerConstraintError("duplicate source row in output mapping")
    source_to_output = np.full(nsource, -1, dtype=np.int64)
    source_to_output[src[original]] = original
    if np.any(source_to_output < 0):
        raise GeneratedLayerConstraintError("source row mapping is incomplete")

    parent = {}
    for marker, item in marker_parent.items():
        marker = int(marker)
        if isinstance(item, dict):
            surface, parent_row = item.get("surface"), item.get("source_face_row")
        else:
            try:
                surface, parent_row = item
            except Exception as exc:
                raise GeneratedLayerConstraintError("marker parent entries must be (surface, source_face_row)") from exc
        if surface not in needed:
            continue
        parent_row = int(parent_row)
        if not 0 <= parent_row < len(O):
            raise GeneratedLayerConstraintError(f"marker {marker} has invalid parent face row")
        if marker in parent:
            raise GeneratedLayerConstraintError(f"duplicate child marker {marker}")
        parent[marker] = (surface, parent_row)

    generated = src < 0
    boundary_generated = np.zeros(n, dtype=bool)
    if len(cf):
        used_boundary = np.unique(cf)
        boundary_generated[used_boundary] = generated[used_boundary]
    covered = set()
    dedup = {}
    max_plane = 0.0
    min_bary = 1.0
    by_relation = Counter()
    by_surface_generated = Counter()
    for child, marker in zip(cf, cm):
        info = parent.get(int(marker))
        if info is None:
            continue
        surface, parent_row = info
        layer = _SURFACE_LAYER[surface]
        source_face = faces[surface][parent_row]
        # Since O/I/D are the same ordered column topology, barycentric weights
        # transfer by adding/subtracting rows_per_layer, never by spatial search.
        base_columns = source_face - layer * rows_per_layer
        parent_output_rows = source_to_output[source_face]
        parent_triangle = R[parent_output_rows]
        if np.any(base_columns < 0) or np.any(base_columns >= rows_per_layer):
            raise GeneratedLayerConstraintError("parent face does not map to source columns")
        child_new = child[generated[child]]
        for node in child_new:
            bary, residual = _barycentric(R[int(node)], parent_triangle)
            max_plane = max(max_plane, residual)
            min_bary = min(min_bary, float(bary.min()))
            if residual > plane_tolerance_m:
                raise GeneratedLayerConstraintError(
                    f"generated boundary node {int(node)} misses parent plane by {residual:.6g} m")
            if bary.min() < -barycentric_tolerance or bary.max() > 1.0 + barycentric_tolerance:
                raise GeneratedLayerConstraintError(f"generated boundary node {int(node)} lies outside marked parent face")
            # Clip only tolerated roundoff at a simplex boundary.
            bary = np.maximum(bary, 0.0)
            bary /= bary.sum()
            covered.add(int(node))
            by_surface_generated[surface] += 1
            opposing = ("I",) if surface == "O" else (("O", "D") if surface == "I" else ("I",))
            for other_surface in opposing:
                other_layer = _SURFACE_LAYER[other_surface]
                other_source_rows = base_columns + other_layer * rows_per_layer
                other_rows = source_to_output[other_source_rows]
                pair_point = bary @ X[other_rows]
                delta = X[int(node)] - pair_point
                length = float(np.linalg.norm(delta))
                if not np.isfinite(length) or length <= 1e-12:
                    raise GeneratedLayerConstraintError(
                        f"generated node {int(node)} has degenerate baseline {surface}->{other_surface} spacing")
                normal = delta / length
                key = (int(node), surface, other_surface,
                       tuple(map(int, other_rows)),
                       tuple(np.round(bary, 12).tolist()))
                if key in dedup:
                    continue
                dedup[key] = (normal, length, bary.copy(), other_rows.copy(), parent_row)
                by_relation[f"{surface}->{other_surface}"] += 1

    missing = np.flatnonzero(boundary_generated & ~np.isin(np.arange(n), np.fromiter(covered, dtype=np.int64)))
    if len(missing):
        raise GeneratedLayerConstraintError(
            f"{len(missing)} generated boundary nodes lack O/I/D marked-parent coverage; first={missing[:12].tolist()}")
    if not dedup:
        raise GeneratedLayerConstraintError("no generated O/I/D boundary relationships were found")

    rows, cols, data, targets = [], [], [], []
    pair_records = []
    baseline_min_slack = np.inf
    for (node, surface, other, other_rows, bary_key), (normal, length, bary, opprows, parent_row) in sorted(dedup.items()):
        lower_row = len(targets)
        upper_row = lower_row + 1
        for axis in range(3):
            rows.extend((lower_row, upper_row))
            cols.extend((3 * node + axis, 3 * node + axis))
            data.extend((float(normal[axis]), float(-normal[axis])))
            for other_row, weight in zip(opprows, bary):
                rows.extend((lower_row, upper_row))
                cols.extend((3 * int(other_row) + axis, 3 * int(other_row) + axis))
                value = float(weight * normal[axis])
                data.extend((-value, value))
        targets.extend((low * length, -high * length))
        baseline = float(normal @ (X[node] - bary @ X[opprows]))
        baseline_min_slack = min(baseline_min_slack, baseline - low * length, high * length - baseline)
        pair_records.append({
            "generated_output_row": node, "from_surface": surface, "to_surface": other,
            "parent_face_row": int(parent_row), "opposing_output_rows": list(map(int, opprows)),
            "barycentric": bary.tolist(), "baseline_spacing_m": length,
        })
    if baseline_min_slack < -baseline_tolerance_m:
        raise GeneratedLayerConstraintError(
            f"frozen baseline violates generated-node thickness bounds by {-baseline_min_slack:.6g} m")
    C = sparse.coo_matrix((data, (rows, cols)), shape=(len(targets), 3 * n), dtype=np.float64).tocsr()
    C.sum_duplicates()
    C.eliminate_zeros()
    C.sort_indices()
    report = {
        "status": "CONSTRUCTION_ROWS_BUILT_BASELINE_FEASIBLE",
        "node_count": n,
        "source_nodes": nsource,
        "generated_nodes": int(generated.sum()),
        "generated_boundary_nodes": int(boundary_generated.sum()),
        "generated_interior_nodes_excluded": int((generated & ~boundary_generated).sum()),
        "generated_boundary_nodes_covered": len(covered),
        "generated_boundary_nodes_uncovered": [],
        "source_face_rows_per_surface": len(O),
        "O_I_D_same_column_topology": True,
        "thickness_bounds_multiple": [low, high],
        "relation_count_after_deduplication": len(dedup),
        "constraint_row_count": C.shape[0],
        "constraint_nnz": C.nnz,
        "relations_by_direction": dict(sorted(by_relation.items())),
        "generated_boundary_corner_visits": dict(sorted(by_surface_generated.items())),
        "max_reference_parent_plane_residual_m": max_plane,
        "minimum_barycentric_weight_before_roundoff_clip": min_bary,
        "minimum_frozen_baseline_slack_m": float(baseline_min_slack),
        "frozen_baseline_max_inequality_violation_m": float(max(0.0, -baseline_min_slack)),
        "rows": pair_records,
        "limitations": [
            "Construction-only projected spacing bounds for generated O/I/D boundary nodes.",
            "Generated interior nodes are not layer-paired.",
            "These rows do not certify all surface clearance, exact solid containment, mass transfer, or native physics.",
        ],
    }
    return C, np.asarray(targets, dtype=np.float64), report
