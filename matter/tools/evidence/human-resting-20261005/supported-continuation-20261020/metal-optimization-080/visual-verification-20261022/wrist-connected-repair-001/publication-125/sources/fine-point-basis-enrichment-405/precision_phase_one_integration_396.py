"""Strict candidate/payload binding and construction-only 026 precision Phase I rows."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _exact_array(actual, expected, name: str) -> None:
    a = np.asarray(actual)
    e = np.asarray(expected)
    if a.shape != e.shape or not np.array_equal(a, e):
        raise ValueError(f"precision payload {name} differs from freshly generated 026 rows")


def assemble_phase_one_self_rows(base_rows, *, enabled, omitted_soft_clearance_rows=None, load_rows=None):
    """Assemble the actual run-path hard-row prefix and optional Phase-I duplicates."""
    base = list(base_rows)
    if not enabled:
        return base, None, np.empty(0, dtype=np.int64)
    if omitted_soft_clearance_rows is None or not isinstance(omitted_soft_clearance_rows, (int, np.integer)):
        raise ValueError("Phase-I assembly requires the captured omitted soft-row count")
    if int(omitted_soft_clearance_rows) < 0:
        raise ValueError("omitted soft-row count must be nonnegative")
    if load_rows is None:
        raise ValueError("Phase-I assembly requires a row loader")
    phase_rows, receipt = load_rows()
    if not isinstance(receipt, dict):
        raise ValueError("Phase-I row loader did not return a mutable receipt")
    source_indices = np.asarray(receipt.get("phase_one_source_row_indices"), dtype=np.int64)
    if source_indices.ndim != 1 or len(source_indices) != len(phase_rows):
        raise ValueError("Phase-I duplicate source-row receipt does not match rows")
    base_count = len(base)
    receipt["soft_clearance_rows_omitted"] = int(omitted_soft_clearance_rows)
    receipt["soft_clearance_target"] = "empty; phase-I objective is zero displacement"
    receipt["base_self_row_count"] = int(base_count)
    receipt["duplicate_self_row_start"] = int(base_count)
    receipt["duplicate_self_row_stop"] = int(base_count + len(phase_rows))
    return base + list(phase_rows), receipt, source_indices


def validate_and_build_phase_rows(
    *,
    points_world_m_f32: np.ndarray,
    faces: np.ndarray,
    base_candidate_sha256: str,
    seed_pairs_sha256: str,
    pair_owner_sha256: str,
    margin_fraction: float,
    report: Mapping,
    report_sha256: str,
    payload_path: Path,
    base_rows: Sequence[Mapping],
    pair_records: Sequence[Mapping],
    safety_factor: float = 4.0,
    cert_tolerance_mm: float = 2.0e-7,
) -> tuple[list[dict], dict]:
    """Verify a fresh 026 row payload and append stricter duplicate rows."""
    points = np.asarray(points_world_m_f32)
    face_array = np.asarray(faces)
    if points.dtype != np.dtype(np.float32) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("phase-I candidate points must be exact float32 (N,3)")
    if not np.isfinite(points).all():
        raise ValueError("phase-I candidate points contain nonfinite values")
    if face_array.ndim != 2 or face_array.shape[1] != 3 or face_array.dtype.kind not in "iu":
        raise ValueError("phase-I candidate faces must be integer (F,3)")
    if np.any(face_array < 0) or np.any(face_array >= len(points)):
        raise ValueError("phase-I candidate face index outside point domain")
    factor = float(safety_factor)
    tol = float(cert_tolerance_mm)
    margin_fraction = float(margin_fraction)
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("precision safety factor must be finite and >= 1")
    if not math.isfinite(tol) or tol < 2.0e-7:
        raise ValueError("precision certificate tolerance must be finite and >= 2e-7 mm")
    if not math.isfinite(margin_fraction) or margin_fraction <= 0.0:
        raise ValueError("pair margin fraction must be finite and positive")

    expected = {
        "candidate_sha256": str(base_candidate_sha256),
        "seed_pairs_sha256": str(seed_pairs_sha256),
        "owner_sha256": str(pair_owner_sha256),
        "candidate_points_sha256": sha256_array(points),
        "candidate_faces_sha256": sha256_array(face_array.astype(np.int64, copy=False)),
        "points_shape": list(points.shape),
        "faces_shape": list(face_array.shape),
        "points_dtype": "float32",
        "selected_pair_count": len(pair_records),
        "margin_fraction": margin_fraction,
    }
    for key, value in expected.items():
        found = report.get(key)
        if key == "margin_fraction":
            if not isinstance(found, (int, float)) or float(found) != value:
                raise ValueError("precision report margin fraction does not match phase-I owner")
        elif found != value:
            raise ValueError(f"precision report {key} does not match current candidate/026 inputs")
    if report.get("diagnostic_only") is not True:
        raise ValueError("precision input report must be explicitly diagnostic-only")
    if sha256_file(payload_path) != report.get("rows_payload_sha256"):
        raise ValueError("precision row payload bytes do not match report SHA")
    if not isinstance(report.get("rows"), list) or not isinstance(report.get("summary"), Mapping):
        raise ValueError("precision report lacks row-level findings")
    if int(report["summary"].get("row_count", -1)) != len(base_rows):
        raise ValueError("precision report row count differs from generated 026 rows")
    if len(report["rows"]) != len(base_rows):
        raise ValueError("precision report must cover every generated 026 row")

    with np.load(payload_path, allow_pickle=False) as z:
        required = ("row_indptr", "node_rows", "coefficients_mm", "lower_mm",
                    "labels", "pair_faces", "pair_kind")
        missing = [key for key in required if key not in z.files]
        if missing:
            raise ValueError(f"precision row payload is missing arrays: {missing}")
        arrays = {key: np.asarray(z[key]).copy() for key in required}
    ptr = arrays["row_indptr"]
    nodes = arrays["node_rows"]
    coeff = arrays["coefficients_mm"]
    rhs = arrays["lower_mm"]
    labels = arrays["labels"]
    pair_faces = arrays["pair_faces"]
    pair_kind = arrays["pair_kind"]
    nrows = len(base_rows)
    if ptr.dtype.kind not in "iu" or ptr.shape != (nrows + 1,) or int(ptr[0]) != 0:
        raise ValueError("precision payload row_indptr is malformed")
    if np.any(np.diff(ptr.astype(np.int64)) <= 0) or int(ptr[-1]) != len(nodes):
        raise ValueError("precision payload CSR row bounds are malformed")
    if nodes.dtype.kind not in "iu" or coeff.shape != (len(nodes), 3) or not np.isfinite(coeff).all():
        raise ValueError("precision payload nodes/coefficients are malformed")
    if rhs.shape != (nrows,) or not np.isfinite(rhs).all() or labels.shape != (nrows,):
        raise ValueError("precision payload RHS/labels are malformed")
    if (pair_faces.shape != (len(pair_records), 2) or pair_faces.dtype.kind not in "iu"
            or pair_kind.shape != (len(pair_records),)):
        raise ValueError("precision payload pair identity arrays are malformed")
    pair_kind_by_identity = {}
    for pair_index, rec in enumerate(pair_records):
        expected_pair = np.asarray(rec["faces"], dtype=np.int64)
        _exact_array(pair_faces[pair_index], expected_pair, f"pair_faces[{pair_index}]")
        if str(pair_kind[pair_index]) != str(rec["kind"]):
            raise ValueError(f"precision payload pair kind differs at pair {pair_index}")
        identity = tuple(sorted(map(int, expected_pair)))
        if identity in pair_kind_by_identity:
            raise ValueError("freshly generated 026 pair records contain duplicate pair identity")
        pair_kind_by_identity[identity] = str(rec["kind"])

    findings = []
    for index, row in enumerate(base_rows):
        start, stop = int(ptr[index]), int(ptr[index + 1])
        _exact_array(nodes[start:stop], np.asarray(row["nodes"], dtype=np.int64), f"nodes[{index}]")
        _exact_array(coeff[start:stop], np.asarray(row["coefficients"], dtype=np.float64), f"coefficients[{index}]")
        if float(rhs[index]) != float(row["lower"]) or str(labels[index]) != str(row["label"]):
            raise ValueError(f"precision payload RHS/label differs at row {index}")
        finding = report["rows"][index]
        if (not isinstance(finding, Mapping) or int(finding.get("row_index", -1)) != index
                or finding.get("label") != row["label"]
                or float(finding.get("delta_rhs_mm", math.nan)) != float(row["lower"])):
            raise ValueError(f"precision report row identity/RHS mismatch at {index}")
        pair = tuple(sorted(map(int, finding.get("pair_face_rows", ()))))
        if pair not in pair_kind_by_identity:
            raise ValueError(f"precision report row names unknown pair at row {index}")
        if finding.get("pair_kind") != pair_kind_by_identity[pair]:
            raise ValueError(f"precision report pair kind mismatch at row {index}")
        findings.append(finding)

    new_rows = []
    source_indices = []
    extras = []
    for index, (row, finding) in enumerate(zip(base_rows, findings)):
        bound = float(finding["rounding_bound_within_axiswise_interval_mm"])
        preserved_margin = float(finding["preserved_pair_margin_mm"])
        old_rhs = float(row["lower"])
        if (not all(math.isfinite(v) for v in (bound, preserved_margin, old_rhs))
                or bound < 0.0 or preserved_margin <= 0.0):
            raise ValueError(f"invalid precision bound/margin at row {index}")
        extra = max(0.0, factor * (bound + tol) - preserved_margin)
        if extra <= 0.0:
            continue
        clone = dict(row)
        clone["lower"] = old_rhs + extra
        clone["label"] = str(row["label"]) + ":F32_PHASE1"
        if np.asarray(row["coefficients"]).shape != (len(row["nodes"]), 3):
            raise ValueError(f"cannot clone malformed 026 row {index}")
        new_rows.append(clone)
        source_indices.append(index)
        extras.append(extra)

    receipt = {
        "classification": "CONSTRUCTION_PRECISION_PHASE_ONE",
        "native_admission": False,
        "diagnostic_only": True,
        "exact_emitted_f32_reaudit_required": True,
        "candidate_sha256": str(base_candidate_sha256),
        "candidate_points_sha256": expected["candidate_points_sha256"],
        "candidate_faces_sha256": expected["candidate_faces_sha256"],
        "seed_pairs_sha256": str(seed_pairs_sha256),
        "pair_owner_sha256": str(pair_owner_sha256),
        "rows_payload_sha256": str(report["rows_payload_sha256"]),
        "precision_report_sha256": str(report_sha256),
        "safety_factor": factor,
        "cert_tolerance_mm": tol,
        "base_hard_self_row_count": int(len(base_rows)),
        "phase_one_duplicate_row_count": int(len(new_rows)),
        "phase_one_source_row_indices": source_indices,
        "minimum_added_rhs_mm": min(extras) if extras else None,
        "maximum_added_rhs_mm": max(extras) if extras else None,
        "original_self_rows_preserved_as_exact_prefix": True,
        "note": "F32 rounding sensitivity only; no self/bone/area/native acceptance gate is relaxed",
    }
    return new_rows, receipt


def evaluate_emitted_self_rows(base_points_world_m_f32: np.ndarray,
                               trial_points_world_m_f32: np.ndarray,
                               rows: Sequence[Mapping],
                               *, tolerance_mm: float = 2.0e-7) -> dict:
    base = np.asarray(base_points_world_m_f32)
    trial = np.asarray(trial_points_world_m_f32)
    tol = float(tolerance_mm)
    if (base.dtype != np.dtype(np.float32) or trial.dtype != np.dtype(np.float32)
            or base.shape != trial.shape or base.ndim != 2 or base.shape[1] != 3
            or not np.isfinite(base).all() or not np.isfinite(trial).all()):
        raise ValueError("emitted row check requires matching finite float32 point arrays")
    if not math.isfinite(tol) or tol < 0:
        raise ValueError("emitted row tolerance must be finite and nonnegative")
    delta_mm = (trial.astype(np.float64) - base.astype(np.float64)) * 1000.0
    residual = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        ids = np.asarray(row["nodes"], dtype=np.int64)
        coefs = np.asarray(row["coefficients"], dtype=np.float64)
        if (ids.ndim != 1 or len(ids) == 0 or np.any(ids < 0) or np.any(ids >= len(base))
                or coefs.shape != (len(ids), 3) or not np.isfinite(coefs).all()):
            raise ValueError(f"malformed emitted hard row {index}")
        residual[index] = float(np.sum(coefs * delta_mm[ids]) - float(row["lower"]))
    failed = np.flatnonzero(residual < -tol)
    return {
        "pass": bool(len(failed) == 0),
        "row_count": int(len(rows)),
        "minimum_residual_mm": float(np.min(residual, initial=np.inf)),
        "failed_row_count": int(len(failed)),
        "failed_row_indices_first_64": failed[:64].tolist(),
        "tolerance_mm": tol,
        "units": "world displacement millimetres",
    }
''
