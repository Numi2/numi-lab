"""Minimal construction-only Phase I for F32-sensitive 026 rows.

Appends stricter copies of selected 026 hard rows. It preserves every
pre-existing row and uses the sparse basis solver with an empty clearance
target. Exact emitted-F32 and all-bank geometry audits remain mandatory.
"""
from __future__ import annotations
import math
from typing import Mapping, Sequence
import numpy as np
from scipy import sparse

MIN_CERT_TOLERANCE_MM = 2.0e-7


def build_precision_phase_one_system(
    base_matrix: sparse.spmatrix,
    base_lower_mm: np.ndarray,
    precision_report: Mapping,
    *,
    cert_tolerance_mm: float = MIN_CERT_TOLERANCE_MM,
    expected_candidate_sha256: str | None = None,
    base_labels: Sequence[str] | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray, dict]:
    """Append duplicate hard rows tightened by certified F32 precision margin."""
    A = sparse.csr_matrix(base_matrix, dtype=np.float64)
    b = np.asarray(base_lower_mm, dtype=np.float64)
    if b.ndim != 1 or A.shape[0] != len(b):
        raise ValueError("base hard-row matrix/lower dimensions do not align")
    if A.shape[1] == 0 or not np.isfinite(A.data).all() or not np.isfinite(b).all():
        raise ValueError("base hard-row system must be finite and nonempty")
    tol = float(cert_tolerance_mm)
    if not math.isfinite(tol) or tol < MIN_CERT_TOLERANCE_MM:
        raise ValueError(f"cert_tolerance_mm must be finite and >= {MIN_CERT_TOLERANCE_MM:g}")
    if precision_report.get("diagnostic_only") is not True:
        raise ValueError("precision report must be explicitly diagnostic-only")
    candidate_sha = precision_report.get("candidate_sha256")
    rows_sha = precision_report.get("rows_payload_sha256")
    if not isinstance(candidate_sha, str) or len(candidate_sha) != 64:
        raise ValueError("precision report lacks a candidate SHA-256 pin")
    if expected_candidate_sha256 is not None and candidate_sha != expected_candidate_sha256:
        raise ValueError("precision report candidate hash does not match requested candidate")
    if not isinstance(rows_sha, str) or len(rows_sha) != 64:
        raise ValueError("precision report lacks a merged-row payload SHA-256 pin")
    rows = precision_report.get("rows")
    summary = precision_report.get("summary")
    if not isinstance(rows, list) or not isinstance(summary, Mapping):
        raise ValueError("precision report lacks row findings or summary")
    if int(summary.get("row_count", -1)) != len(rows) or len(rows) != A.shape[0]:
        raise ValueError("precision report must cover every supplied 026 hard row exactly once")
    if base_labels is not None and len(base_labels) != A.shape[0]:
        raise ValueError("base label count does not match hard-row count")

    seen: set[int] = set()
    selected: list[tuple[int, float]] = []
    for report_row in rows:
        if not isinstance(report_row, Mapping):
            raise ValueError("precision row finding must be an object")
        try:
            index = int(report_row["row_index"])
            old_rhs = float(report_row["delta_rhs_mm"])
            margin = float(report_row["preserved_pair_margin_mm"])
            bound = float(report_row["rounding_bound_within_axiswise_interval_mm"])
            label = report_row["label"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed precision row finding") from exc
        if index < 0 or index >= A.shape[0] or index in seen:
            raise ValueError("precision row indices must be unique and in range")
        seen.add(index)
        if not isinstance(label, str):
            raise ValueError("precision row label must be text")
        if base_labels is not None and label != base_labels[index]:
            raise ValueError(f"precision row label mismatch at {index}")
        if not all(math.isfinite(x) for x in (old_rhs, margin, bound)) or margin <= 0.0 or bound < 0.0:
            raise ValueError(f"invalid precision margin/bound at row {index}")
        if old_rhs != float(b[index]):
            raise ValueError(f"precision report RHS does not match hard row {index}")
        # This increment is the per-row amount needed beyond the margin already
        # protected by the original 026 RHS.
        extra = max(0.0, bound + tol - margin)
        if extra > 0.0:
            if A.getrow(index).nnz == 0:
                raise ValueError(f"cannot strengthen zero hard row {index}")
            selected.append((index, extra))

    if len(seen) != A.shape[0]:
        raise ValueError("precision report omits one or more supplied 026 hard rows")
    selected.sort(key=lambda item: item[0])
    if selected:
        indices = np.asarray([x[0] for x in selected], dtype=np.int64)
        extra_matrix = A[indices].tocsr()
        extras = np.asarray([x[1] for x in selected], dtype=np.float64)
        augmented = sparse.vstack((A, extra_matrix), format="csr")
        lower = np.concatenate((b.copy(), b[indices] + extras))
    else:
        indices = np.empty(0, dtype=np.int64)
        extras = np.empty(0, dtype=np.float64)
        augmented = A.copy()
        lower = b.copy()

    receipt = {
        "classification": "CONSTRUCTION_PRECISION_PHASE_ONE",
        "native_admission": False,
        "diagnostic_only": True,
        "exact_emitted_f32_reaudit_required": True,
        "candidate_sha256": candidate_sha,
        "rows_payload_sha256": rows_sha,
        "cert_tolerance_mm": tol,
        "base_hard_row_count": int(A.shape[0]),
        "phase_one_duplicate_row_count": int(len(indices)),
        "phase_one_source_row_indices": indices.tolist(),
        "original_rows_preserved_as_exact_prefix": True,
        "minimum_added_rhs_mm": float(np.min(extras)) if len(extras) else None,
        "maximum_added_rhs_mm": float(np.max(extras)) if len(extras) else None,
    }
    return augmented, lower, receipt


def solve_precision_phase_one(
    solver,
    base_matrix: sparse.spmatrix,
    base_lower_mm: np.ndarray,
    precision_report: Mapping,
    output_lower_mm: np.ndarray,
    output_upper_mm: np.ndarray,
    output_map: sparse.spmatrix,
    *,
    time_limit_s: float,
    cert_tolerance_mm: float = MIN_CERT_TOLERANCE_MM,
    expected_candidate_sha256: str | None = None,
    base_labels: Sequence[str] | None = None,
    lp_method: str = "highs",
) -> dict:
    """Call the existing basis solver with empty soft-target rows and zero target."""
    A1, b1, receipt = build_precision_phase_one_system(
        base_matrix, base_lower_mm, precision_report,
        cert_tolerance_mm=cert_tolerance_mm,
        expected_candidate_sha256=expected_candidate_sha256,
        base_labels=base_labels,
    )
    M = sparse.csr_matrix(output_map, dtype=np.float64)
    n_out = M.shape[0]
    result = solver(
        A1, b1, np.zeros(n_out, dtype=np.float64),
        np.asarray(output_lower_mm, dtype=np.float64),
        np.asarray(output_upper_mm, dtype=np.float64),
        sparse.csr_matrix((0, n_out), dtype=np.float64),
        np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64),
        M, float(time_limit_s),
        slack_tolerance=float(cert_tolerance_mm), lp_method=lp_method,
    )
    result["precision_phase_one"] = receipt
    if result.get("success"):
        delta = np.asarray(result.get("delta"), dtype=np.float64)
        if delta.shape != (n_out,) or not np.isfinite(delta).all():
            raise RuntimeError("phase-I solver returned malformed displacement")
        residual = np.asarray(A1 @ delta).reshape(-1) - b1
        lo = np.asarray(output_lower_mm, dtype=np.float64)
        hi = np.asarray(output_upper_mm, dtype=np.float64)
        if np.min(residual, initial=0.0) < -cert_tolerance_mm:
            raise RuntimeError("phase-I result failed independent hard-row certificate")
        if (np.min(delta - lo, initial=0.0) < -cert_tolerance_mm
                or np.min(hi - delta, initial=0.0) < -cert_tolerance_mm):
            raise RuntimeError("phase-I result failed independent output-bound certificate")
        receipt["minimum_hard_residual_mm"] = float(np.min(residual, initial=np.inf))
    return result

