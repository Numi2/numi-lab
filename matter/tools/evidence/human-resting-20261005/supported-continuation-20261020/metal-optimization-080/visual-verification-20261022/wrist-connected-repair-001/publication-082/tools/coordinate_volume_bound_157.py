"""Arithmetic certificate for scalar-split child-volume accounting.

This module is isolated evidence code. It does not change the canonical splitter
or any physical/native gate. Barycentric reference partition retains 4096*eps;
the separate coordinate-space comparison is accepted only with a derived
rounding bound for encoded intersection coordinates and determinant arithmetic.
"""
from __future__ import annotations

import itertools
import math
import numpy as np
from scipy import sparse

EPS = np.finfo(np.float64).eps
REFERENCE_TOLERANCE_MULTIPLIER = 4096.0


def gamma(n: int) -> float:
    n = int(n)
    if n < 0 or n * EPS >= 1.0:
        raise ValueError("invalid operation count for gamma bound")
    return (n * EPS) / (1.0 - n * EPS)


def _l1_upper(values: np.ndarray) -> float:
    """Conservative upper bound for an L1 norm of a short rounded vector."""
    vals = np.asarray(values).ravel()
    total = math.fsum(abs(float(v)) for v in vals)
    return math.nextafter(total / (1.0 - gamma(len(vals) + 1)), math.inf)


def _det_edge_perturbation_bound(na, nb, nc, ea, eb, ec) -> float:
    terms = (
        ea * nb * nc, eb * na * nc, ec * na * nb,
        ea * eb * nc, ea * ec * nb, eb * ec * na, ea * eb * ec,
    )
    # The products and sum below are themselves rounded. Inflate by a gamma
    # factor covering each product and the short accumulation.
    return math.nextafter(math.fsum(terms) / (1.0 - gamma(12)), math.inf)


def _det3_and_round_bound(points: np.ndarray) -> tuple[float, float]:
    """Det(a,b,c) and an absolute bound to the exact stored-point determinant.

    The bound covers both rounded coordinate differences and the cross/dot
    determinant evaluation. It is an arithmetic bound, not a geometry slack.
    """
    x = np.asarray(points, dtype=np.float64)
    if x.shape != (4, 3) or not np.isfinite(x).all():
        raise ValueError("tetrahedron points must be finite 4x3")
    a, b, c = x[1] - x[0], x[2] - x[0], x[3] - x[0]
    det = float(np.dot(a, np.cross(b, c)))
    permanent = math.fsum(
        abs(float(a[i])) * (abs(float(b[j] * c[k])) + abs(float(b[k] * c[j])))
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    )
    g = gamma(24)
    permanent_upper = math.nextafter(permanent / (1.0 - g), math.inf)
    evaluation_error = math.nextafter(g * permanent_upper, math.inf)

    # Each edge is a rounded subtraction of two stored points. Bound the
    # exact edge length by the rounded edge plus its subtraction error.
    edge_bounds = []
    edge_length_bounds = []
    for i, edge in zip((1, 2, 3), (a, b, c)):
        coordinate_sum = np.nextafter(np.abs(x[i]) + np.abs(x[0]), math.inf)
        component_error = np.nextafter(EPS * coordinate_sum, math.inf)
        edge_bounds.append(_l1_upper(component_error))
        edge_length_bounds.append(_l1_upper(edge) + _l1_upper(component_error))
    ea, eb, ec = edge_bounds
    na, nb, nc = edge_length_bounds
    edge_error = _det_edge_perturbation_bound(na, nb, nc, ea, eb, ec)
    return det, math.nextafter(evaluation_error + edge_error, math.inf)


def _det4_permutation_and_bound(matrix: np.ndarray) -> tuple[float, float]:
    """4x4 determinant by all 24 signed products and a gamma(30) bound."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError("barycentric child matrix must be finite 4x4")
    rows = np.arange(4)
    signed_terms = []
    abs_terms = []
    for perm in itertools.permutations(range(4)):
        inv = sum(perm[i] > perm[j] for i in range(4) for j in range(i + 1, 4))
        term = 1.0
        for i, j in enumerate(perm):
            term *= float(m[i, j])
        abs_terms.append(abs(term))
        signed_terms.append(-term if inv & 1 else term)
    det = math.fsum(signed_terms)
    permanent = math.fsum(abs_terms)
    g = gamma(30)
    permanent_upper = permanent / (1.0 - g)
    return det, g * permanent_upper


def _barycentric_child_matrix(
    interpolation: sparse.csr_matrix, child: np.ndarray, parent_tet: np.ndarray
) -> np.ndarray:
    local = {int(node): i for i, node in enumerate(parent_tet)}
    out = np.zeros((4, 4), dtype=np.float64)
    for r, point_row in enumerate(child):
        lo, hi = interpolation.indptr[int(point_row):int(point_row) + 2]
        for k in range(lo, hi):
            source = int(interpolation.indices[k])
            if source not in local:
                raise ValueError("child interpolation references a source vertex outside its parent tetrahedron")
            out[r, local[source]] += float(interpolation.data[k])
    return out


def _coordinate_encoding_bounds(
    source_points: np.ndarray,
    output_points: np.ndarray,
    interpolation: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    source = np.asarray(source_points, dtype=np.float64)
    output = np.asarray(output_points, dtype=np.float64)
    W = interpolation.tocsr(copy=True)
    W.sum_duplicates()
    W.sort_indices()
    if source.ndim != 2 or source.shape[1] != 3 or output.ndim != 2 or output.shape[1] != 3:
        raise ValueError("source/output coordinates must be Nx3")
    if W.shape != (len(output), len(source)) or not np.isfinite(source).all() or not np.isfinite(output).all():
        raise ValueError("interpolation shape or coordinate finiteness mismatch")
    if np.any(~np.isfinite(W.data)) or np.any(W.data < 0.0):
        raise ValueError("interpolation weights must be finite and nonnegative")
    row_nnz = np.diff(W.indptr)
    if np.any(row_nnz == 0) or np.any(row_nnz > 2):
        raise ValueError("this certificate expects original vertices or two-endpoint edge intersections only")

    row_sums = np.asarray(W.sum(axis=1)).ravel()
    if np.max(np.abs(row_sums - 1.0), initial=0.0) > REFERENCE_TOLERANCE_MULTIPLIER * EPS:
        raise ValueError("interpolation row sums exceed the existing 4096-epsilon gate")

    expected = np.asarray(W @ source, dtype=np.float64)
    residual = np.abs(output - expected)
    nonzero_residual = residual > 0.0
    residual[nonzero_residual] = np.nextafter(residual[nonzero_residual], np.inf)
    absW = W.copy()
    absW.data = np.abs(absW.data)
    weighted_abs = np.asarray(absW @ np.abs(source), dtype=np.float64)
    dot_bound = np.zeros_like(output)
    for r, k in enumerate(row_nnz):
        lo, hi = W.indptr[r:r + 2]
        if k == 1 and W.data[lo] == 1.0:
            # Copying an original point is exact; no dot-product arithmetic.
            dot_bound[r] = 0.0
        else:
            g = gamma(2 * int(k))
            dot_bound[r] = (g / (1.0 - g)) * weighted_abs[r]

    construction_bound = np.zeros_like(output)
    for r, k in enumerate(row_nnz):
        lo, hi = W.indptr[r:r + 2]
        cols = W.indices[lo:hi]
        weights = W.data[lo:hi]
        if k == 1:
            col = int(cols[0])
            if weights[0] != 1.0 or not np.array_equal(output[r], source[col]):
                raise ValueError("original point rows must be bitwise identity copies")
            continue
        a, b = source[int(cols[0])], source[int(cols[1])]
        w0, w1 = float(weights[0]), float(weights[1])
        if w0 > 1.0 or w1 > 1.0:
            raise ValueError("edge interpolation weights exceed one")
        if not (0.0 <= w1 <= 1.0 and 0.0 <= w0 <= 1.0):
            raise ValueError("edge interpolation weights must lie in [0,1]")
        # Source expression is fl(a + fl(t * fl(b-a))). The CSR reference
        # stores (w0,t), with w0=fl(1-t). Bound each operation a priori using
        # unit roundoff and source coordinate magnitudes; do not use the
        # measured output residual to construct this allowance.
        u = EPS
        abs_a, abs_b, t = np.abs(a), np.abs(b), abs(w1)
        subtract_error = u * (abs_a + abs_b)
        diff_abs_upper = abs_a + abs_b + subtract_error
        multiply_error = u * t * diff_abs_upper
        product_abs_upper = t * diff_abs_upper + multiply_error
        add_error = u * (abs_a + product_abs_upper)
        one_minus_t_error = u * (1.0 + t) * abs_a
        construction_raw = (
            one_minus_t_error + t * subtract_error + multiply_error + add_error
        )
        # Inflate arithmetic used to evaluate the bound itself; this cannot
        # depend on the observed output residual.
        construction_bound[r] = np.nextafter(
            construction_raw / (1.0 - gamma(16)), np.inf
        )

    allowed_residual = np.nextafter(construction_bound + dot_bound, np.inf)
    if np.any(residual > np.nextafter(allowed_residual, np.inf)):
        bad = np.argwhere(residual > np.nextafter(allowed_residual, np.inf))[0]
        raise ValueError(f"stored point exceeds the declared edge-interpolation rounding bound at row/component {bad.tolist()}")
    # A priori determinant-coordinate allowance. One sparse-dot term covers
    # the rounded W@source comparison point; the second covers its displacement
    # from the exact real CSR interpolation used by the barycentric reference.
    coordinate_error_bound = np.nextafter(
        (construction_bound + 2.0 * dot_bound) / (1.0 - gamma(4)), np.inf
    )
    report = {
        "output_point_count": int(len(output)),
        "source_point_count": int(len(source)),
        "max_interpolation_row_sum_error": float(np.max(np.abs(row_sums - 1.0), initial=0.0)),
        "max_measured_point_vs_W_source_residual_m": float(np.max(residual, initial=0.0)),
        "max_edge_construction_rounding_bound_m": float(np.max(construction_bound, initial=0.0)),
        "max_sparse_dot_rounding_bound_m": float(np.max(dot_bound, initial=0.0)),
        "max_point_encoding_residual_allowance_m": float(np.max(allowed_residual, initial=0.0)),
        "max_a_priori_coordinate_error_bound_m": float(np.max(coordinate_error_bound, initial=0.0)),
        "point_residual_used_to_set_determinant_bound": False,
    }
    return expected, coordinate_error_bound, row_sums, report


def _coordinate_determinant_perturbation_bound(
    base_points: np.ndarray, point_error_bounds: np.ndarray
) -> float:
    """Bound determinant change from per-vertex componentwise coordinate error.

    The edge-error vectors are bounded from endpoint coordinate uncertainties.
    Cross-product magnitudes are bounded by their computed value plus the
    first-order and bilinear perturbation terms.
    """
    x = np.asarray(base_points, dtype=np.float64)
    d = np.asarray(point_error_bounds, dtype=np.float64)
    if x.shape != (4, 3) or d.shape != (4, 3) or np.any(d < 0.0):
        raise ValueError("expected 4x3 points and nonnegative 4x3 point bounds")
    a, b, c = x[1] - x[0], x[2] - x[0], x[3] - x[0]
    # L1 bounds dominate Euclidean norms. Include rounded subtraction error
    # in each base edge, and outward-round the positive uncertainty additions.
    e_components = [np.nextafter(d[i] + d[0], np.inf) for i in (1, 2, 3)]
    ea, eb, ec = (_l1_upper(v) for v in e_components)
    edge_norms = []
    for i, edge in zip((1, 2, 3), (a, b, c)):
        coordinate_sum = np.nextafter(np.abs(x[i]) + np.abs(x[0]), np.inf)
        sub_error = _l1_upper(np.nextafter(EPS * coordinate_sum, np.inf))
        edge_norms.append(_l1_upper(edge) + sub_error)
    na, nb, nc = edge_norms
    return _det_edge_perturbation_bound(na, nb, nc, ea, eb, ec)


def certify_parent_volume_partition(
    source_points: np.ndarray,
    source_tetrahedra: np.ndarray,
    output_points: np.ndarray,
    output_tetrahedra: np.ndarray,
    child_parent_rows: np.ndarray,
    interpolation_weights: sparse.spmatrix,
    *,
    emitted_f32_required_mask: np.ndarray | None = None,
) -> dict:
    """Certify reference partition and bound stored-coordinate volume residuals.

    The unchanged 4096*eps reference-volume test is applied to determinant
    fractions of each child's 4x4 parent-barycentric matrix. No fraction is
    normalized or child omitted. Stored-coordinate partition is checked against
    that fixed tolerance plus a per-child arithmetic bound.
    """
    P0 = np.asarray(source_points, dtype=np.float64)
    T0 = np.asarray(source_tetrahedra, dtype=np.int64)
    P = np.asarray(output_points, dtype=np.float64)
    C = np.asarray(output_tetrahedra, dtype=np.int64)
    parents = np.asarray(child_parent_rows, dtype=np.int64)
    W = interpolation_weights.tocsr(copy=True)
    if T0.ndim != 2 or T0.shape[1] != 4 or C.ndim != 2 or C.shape[1] != 4:
        raise ValueError("source and output tetrahedra must be Nx4")
    if len(C) != len(parents) or not len(T0) or not len(C):
        raise ValueError("child parent map is empty or has the wrong length")
    if np.any((parents < 0) | (parents >= len(T0))):
        raise ValueError("child parent rows are out of range")
    if C.min() < 0 or C.max() >= len(P):
        raise ValueError("output tetrahedron vertex row out of range")
    expected, point_error, row_sums, point_report = _coordinate_encoding_bounds(P0, P, W)

    f32 = P.astype(np.float32).astype(np.float64)
    f32_dets = np.empty(len(C), dtype=np.float64)
    f64_child_dets = np.empty(len(C), dtype=np.float64)
    for i, tet in enumerate(C):
        f64_child_dets[i] = _det3_and_round_bound(P[tet])[0]
        f32_child_det = _det3_and_round_bound(f32[tet])[0]
        f32_dets[i] = f32_child_det
    if np.any(~np.isfinite(f64_child_dets) | (f64_child_dets <= 0.0)):
        raise ValueError("stored Float64 child geometry contains nonpositive cells")
    f32_positive = np.isfinite(f32_dets) & (f32_dets > 0.0)
    if emitted_f32_required_mask is None:
        f32_required = np.ones(len(C), dtype=bool)
    else:
        f32_required = np.asarray(emitted_f32_required_mask, dtype=bool)
        if f32_required.shape != (len(C),):
            raise ValueError("emitted Float32 required mask must have one value per child")
    if np.any(f32_required & ~f32_positive):
        bad = np.flatnonzero(f32_required & ~f32_positive)[:16].tolist()
        raise ValueError(f"required emitted Float32 child geometry contains nonpositive cells at rows {bad}")

    # Stable sort creates child index ranges once, avoiding a full child scan
    # for every parent when this certificate is applied to a large clip.
    child_order = np.argsort(parents, kind="stable")
    child_counts = np.bincount(parents, minlength=len(T0))
    child_offsets = np.concatenate(([0], np.cumsum(child_counts, dtype=np.int64)))
    parent_reports = []
    parent_volume_residual_max = 0.0
    derived_bound_max = 0.0
    for parent_row, parent_tet in enumerate(T0):
        children = child_order[child_offsets[parent_row]:child_offsets[parent_row + 1]]
        if not len(children):
            raise ValueError(f"source parent {parent_row} has no children")
        parent_det, parent_det_error = _det3_and_round_bound(P0[parent_tet])
        if not np.isfinite(parent_det) or parent_det <= 0.0:
            raise ValueError(f"source parent {parent_row} is not positively oriented")
        parent_volume = parent_det / 6.0
        parent_volume_error = parent_det_error / 6.0 + EPS * abs(parent_volume)
        fixed_tolerance = REFERENCE_TOLERANCE_MULTIPLIER * EPS * parent_volume

        actual_volumes = []
        reference_volumes = []
        child_arithmetic_bounds = []
        barycentric_ratios = []
        q32 = []
        for child_row in children:
            child = C[child_row]
            v_actual = f64_child_dets[child_row] / 6.0
            actual_volumes.append(float(v_actual))
            B = _barycentric_child_matrix(W, child, parent_tet)
            detB, detB_error = _det4_permutation_and_bound(B)
            if not np.isfinite(detB) or detB <= 0.0:
                raise ValueError(f"child {int(child_row)} has nonpositive parent-barycentric orientation")
            ratio = float(detB)
            barycentric_ratios.append(ratio)
            v_reference = parent_volume * ratio
            reference_volumes.append(float(v_reference))

            actual_eval_error = _det3_and_round_bound(P[child])[1] / 6.0 + EPS * abs(v_actual)
            reference_eval_error = (
                parent_volume_error * (abs(ratio) + detB_error)
                + parent_volume * detB_error
                + EPS * abs(v_reference)
            )
            coord_det_error = _coordinate_determinant_perturbation_bound(
                expected[child], point_error[child]
            ) / 6.0
            child_arithmetic_bounds.append(actual_eval_error + reference_eval_error + coord_det_error)

            pts32 = f32[child]
            longest = max(float(np.linalg.norm(pts32[i] - pts32[j])) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)))
            q32.append(float(f32_dets[child_row] / longest**3) if longest > 0.0 else float("nan"))

        actual_sum = math.fsum(actual_volumes)
        reference_sum = math.fsum(reference_volumes)
        reference_error = abs(reference_sum - parent_volume)
        if reference_error > fixed_tolerance:
            raise ValueError(f"parent {parent_row} barycentric reference partition exceeds unchanged 4096-epsilon tolerance")
        accumulation_bound = gamma(max(1, len(children))) * (
            math.fsum(abs(v) for v in actual_volumes) + math.fsum(abs(v) for v in reference_volumes)
        )
        derived_bound = math.fsum(child_arithmetic_bounds) + accumulation_bound
        actual_error = abs(actual_sum - parent_volume)
        allowed = fixed_tolerance + derived_bound
        if actual_error > allowed or abs(actual_sum - reference_sum) > derived_bound:
            raise ValueError(f"parent {parent_row} stored-coordinate volume mismatch exceeds fixed tolerance plus arithmetic bound")
        parent_volume_residual_max = max(parent_volume_residual_max, actual_error)
        derived_bound_max = max(derived_bound_max, derived_bound)
        parent_reports.append({
            "parent_row": int(parent_row),
            "child_count": int(len(children)),
            "parent_reference_volume_m3": float(parent_volume),
            "reference_partition_residual_m3": float(reference_error),
            "fixed_4096eps_tolerance_m3": float(fixed_tolerance),
            "stored_coordinate_partition_residual_m3": float(actual_error),
            "derived_arithmetic_bound_m3": float(derived_bound),
            "derived_arithmetic_bound_relative_to_parent": float(derived_bound / parent_volume),
            "stored_coordinate_partition_relative_residual": float(actual_error / parent_volume),
            "actual_vs_reference_residual_m3": float(abs(actual_sum - reference_sum)),
            "actual_vs_reference_relative_residual": float(abs(actual_sum - reference_sum) / parent_volume),
            "max_child_q_f32_diagnostic": float(np.nanmin(q32)),
            "barycentric_fraction_sum": float(math.fsum(barycentric_ratios)),
            "children_omitted": 0,
            "fractions_normalized": False,
        })

    return {
        "status": "ARITHMETICALLY_BOUNDED_REFERENCE_PARTITION",
        "physical_or_native_admission": False,
        "simulation_allowed": False,
        "existing_reference_tolerance_multiplier": REFERENCE_TOLERANCE_MULTIPLIER,
        "reference_partition_basis": "4x4 child-to-parent interpolation determinant",
        "mass_fraction_rule": "raw barycentric determinant fractions; no normalization and no child drop",
        "coordinate_partition_rule": "measured stored-coordinate error <= unchanged reference tolerance + a-priori construction/dot/determinant arithmetic bound",
        "coordinate_bound_rule": "point residual is first checked against an a-priori operation-count/source-magnitude bound; determinant allowance uses construction_bound + 2*sparse_dot_bound, never measured residual",
        "strict_f64_positive_child_count": int(len(C)),
        "strict_emitted_f32_positive_child_count": int(np.count_nonzero(f32_positive)),
        "strict_required_emitted_f32_positive_child_count": int(np.count_nonzero(f32_required & f32_positive)),
        "emitted_f32_nonpositive_excluded_side_child_count": int(np.count_nonzero(~f32_required & ~f32_positive)),
        "parent_count": int(len(T0)),
        "child_count": int(len(C)),
        "max_stored_coordinate_partition_residual_m3": float(parent_volume_residual_max),
        "max_derived_arithmetic_bound_m3": float(derived_bound_max),
        "max_stored_coordinate_partition_relative_residual": float(max(p["stored_coordinate_partition_relative_residual"] for p in parent_reports)),
        "max_derived_arithmetic_bound_relative_to_parent": float(max(p["derived_arithmetic_bound_relative_to_parent"] for p in parent_reports)),
        "max_actual_vs_reference_relative_residual": float(max(p["actual_vs_reference_relative_residual"] for p in parent_reports)),
        **point_report,
        "parents": parent_reports,
    }
