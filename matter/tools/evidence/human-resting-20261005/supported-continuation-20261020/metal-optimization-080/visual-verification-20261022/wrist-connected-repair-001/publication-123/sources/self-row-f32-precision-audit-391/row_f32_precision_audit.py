"""Read-only audit of 026 merged self-separation row margins vs F32 rounding.

Rows use displacement millimetres and points are the exact emitted Float32
world coordinates in metres. This estimates a conservative componentwise
round-to-nearest error bound; it does not relax or replace any geometric gate.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np

_LABEL_PAIR = re.compile(r"^(\d+):(\d+):")


def _symmetric_half_ulp_mm(points_world_m_f32: np.ndarray) -> np.ndarray:
    """Return a conservative half-rounding interval per F32 coordinate, in mm."""
    p = np.asarray(points_world_m_f32)
    if p.dtype != np.dtype(np.float32):
        raise ValueError("points must be the exact emitted float32 array")
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("points must be finite float32 (N,3)")
    neg_inf = np.full(p.shape, np.float32(-np.inf), dtype=np.float32)
    pos_inf = np.full(p.shape, np.float32(np.inf), dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        prev = np.nextafter(p, neg_inf)
        nxt = np.nextafter(p, pos_inf)
    down = np.abs(p.astype(np.float64) - prev.astype(np.float64))
    up = np.abs(nxt.astype(np.float64) - p.astype(np.float64))
    # max is conservative at binade boundaries where the adjacent spacings differ.
    spacing = np.maximum(down, up)
    if not np.isfinite(spacing).all():
        raise ValueError("cannot bound a coordinate at the finite-float32 limit")
    return (0.5 * spacing) * 1000.0


def _pair_metadata(pair_records: Sequence[Mapping] | None) -> dict[tuple[int, int], dict]:
    by_pair: dict[tuple[int, int], dict] = {}
    for rec in pair_records or ():
        faces = rec.get("faces")
        if not isinstance(faces, (list, tuple, np.ndarray)) or len(faces) != 2:
            raise ValueError("each pair record must have two face IDs")
        pair = tuple(sorted((int(faces[0]), int(faces[1]))))
        if pair in by_pair:
            raise ValueError(f"duplicate pair record {pair}")
        kind = rec.get("kind")
        if kind not in ("common_vertex_wedge", "moving_sat_midplane"):
            raise ValueError(f"unknown 026 pair kind for {pair}: {kind!r}")
        by_pair[pair] = dict(rec)
    return by_pair


def audit_pair_rows_f32_precision(
    points_world_m_f32: np.ndarray,
    rows: Iterable[Mapping],
    pair_records: Sequence[Mapping] | None = None,
    *,
    interval_radius_mm: float = 0.0,
) -> dict:
    """Audit already-built, merged 026 rows without regenerating constraints.

    Each row must contain the pinned 026 keys `nodes`, `coefficients`, `lower`,
    and `label`. `lower` and coefficients are in mm. The returned pair/row
    findings are diagnostics only: no row is removed, altered, or accepted.
    """
    points = np.asarray(points_world_m_f32)
    if not math.isfinite(float(interval_radius_mm)) or float(interval_radius_mm) < 0.0:
        raise ValueError("interval_radius_mm must be finite and nonnegative")
    half_ulp_current_mm = _symmetric_half_ulp_mm(points)
    radius_m = float(interval_radius_mm) / 1000.0
    p64 = points.astype(np.float64)
    max_abs_interval = np.maximum(np.abs(p64 - radius_m), np.abs(p64 + radius_m))
    interval_probe = max_abs_interval.astype(np.float32)
    half_ulp_interval_mm = _symmetric_half_ulp_mm(interval_probe)
    if pair_records is None:
        raise ValueError("026 pair records with preserved margins are required")
    metadata = _pair_metadata(pair_records)
    for pair, rec in metadata.items():
        if "preserved_margin_mm" not in rec or not math.isfinite(float(rec["preserved_margin_mm"])) or float(rec["preserved_margin_mm"]) <= 0.0:
            raise ValueError(f"pair {pair} lacks a finite positive preserved_margin_mm")
    row_reports = []
    seen_pairs: set[tuple[int, int]] = set()
    margin_buckets = Counter()
    by_kind: dict[str, Counter] = {
        "common_vertex_wedge": Counter(),
        "moving_sat_midplane": Counter(),
        "unmatched_record": Counter(),
    }
    max_ratio = 0.0
    pair_subprecision: dict[tuple[int, int], bool] = {}
    pair_kinds: dict[tuple[int, int], str] = {}
    margins: list[float] = []
    bounds: list[float] = []

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} is not a mapping")
        try:
            nodes = np.asarray(row["nodes"])
            coeff = np.asarray(row["coefficients"], dtype=np.float64)
            margin = float(row["lower"])
            label = row["label"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed 026 row {index}") from exc
        if nodes.ndim != 1 or nodes.dtype.kind not in "iu":
            raise ValueError(f"row {index} nodes must be a 1D integer array")
        ids = nodes.astype(np.int64, copy=False)
        if len(ids) == 0 or np.any(ids < 0) or np.any(ids >= len(points)):
            raise ValueError(f"row {index} has empty or out-of-range node IDs")
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"row {index} coefficients are not merged by node")
        if coeff.shape != (len(ids), 3) or not np.isfinite(coeff).all():
            raise ValueError(f"row {index} coefficients must be finite (len(nodes),3)")
        if not math.isfinite(margin):
            raise ValueError(f"row {index} margin is nonfinite")
        if not isinstance(label, str):
            raise ValueError(f"row {index} label must be text")
        match = _LABEL_PAIR.match(label)
        if match is None:
            raise ValueError(f"row {index} lacks 026 face-pair label prefix")
        pair = tuple(sorted((int(match.group(1)), int(match.group(2)))))
        if pair[0] == pair[1]:
            raise ValueError(f"row {index} names a self-pair")
        seen_pairs.add(pair)
        rec = metadata.get(pair)
        if rec is None:
            raise ValueError(f"row {index} has no matching 026 pair record: {pair}")
        kind = rec["kind"]
        # Compare coordinate-rounding sensitivity to the preserved pair margin,
        # not row["lower"], which is the signed displacement RHS after 026
        # subtracts the baseline row value and is often negative by design.
        preserved_margin_mm = float(rec["preserved_margin_mm"])

        # F32 coordinates are m; 026 coefficients act on mm displacements.
        # Thus half-ULP(m)*1000 gives the conservative coordinate uncertainty
        # in the exact units of the row.
        abs_coeff = np.abs(coeff)
        current_error_bound_mm = math.fsum(float(v) for v in (abs_coeff * half_ulp_current_mm[ids]).ravel())
        interval_error_bound_mm = math.fsum(float(v) for v in (abs_coeff * half_ulp_interval_mm[ids]).ravel())
        current_ratio = current_error_bound_mm / preserved_margin_mm
        interval_ratio = interval_error_bound_mm / preserved_margin_mm
        current_subprecision = bool(preserved_margin_mm <= current_error_bound_mm)
        interval_subprecision = bool(preserved_margin_mm <= interval_error_bound_mm)
        pair_subprecision[pair] = pair_subprecision.get(pair, False) or interval_subprecision
        pair_kinds[pair] = kind
        margin_buckets["positive"] += 1
        c = by_kind[kind]
        c["rows"] += 1
        c["current_subprecision_rows"] += int(current_subprecision)
        c["interval_subprecision_rows"] += int(interval_subprecision)
        c["nonpositive_delta_rhs_rows"] += int(margin <= 0)
        c["max_current_error_bound_mm"] = max(c["max_current_error_bound_mm"], current_error_bound_mm)
        c["max_interval_error_bound_mm"] = max(c["max_interval_error_bound_mm"], interval_error_bound_mm)
        c["max_current_error_to_margin_ratio"] = max(c["max_current_error_to_margin_ratio"], current_ratio)
        c["max_interval_error_to_margin_ratio"] = max(c["max_interval_error_to_margin_ratio"], interval_ratio)
        max_ratio = max(max_ratio, interval_ratio)
        margins.append(preserved_margin_mm)
        bounds.append(interval_error_bound_mm)
        row_reports.append({
            "row_index": index,
            "label": label,
            "pair_face_rows": list(pair),
            "pair_kind": kind,
            "preserved_pair_margin_mm": preserved_margin_mm,
            "delta_rhs_mm": margin,
            "rounding_bound_at_current_f32_mm": current_error_bound_mm,
            "rounding_bound_within_axiswise_interval_mm": interval_error_bound_mm,
            "current_error_to_margin_ratio": current_ratio,
            "interval_error_to_margin_ratio": interval_ratio,
            "current_subprecision_margin": current_subprecision,
            "interval_subprecision_margin": interval_subprecision,
            "coefficient_node_count": int(len(ids)),
        })

    unmatched = sorted(seen_pairs - set(metadata))
    unused_records = sorted(set(metadata) - seen_pairs)
    for kind in by_kind:
        typed_pairs = [pair for pair in seen_pairs if pair_kinds[pair] == kind]
        by_kind[kind]["pairs"] = len(typed_pairs)
        by_kind[kind]["pairs_with_subprecision_rows"] = sum(pair_subprecision[pair] for pair in typed_pairs)
    summary = {
        "diagnostic_only": True,
        "row_count": len(row_reports),
        "pair_count_with_rows": len(seen_pairs),
        "pair_record_count": len(metadata),
        "unmatched_row_pairs": [list(p) for p in unmatched],
        "pair_records_without_rows": [list(p) for p in unused_records],
        "positive_margin_rows": margin_buckets["positive"],
        "nonpositive_preserved_margin_rows": margin_buckets["nonpositive"],
        "current_subprecision_rows": sum(r["current_subprecision_margin"] for r in row_reports),
        "interval_subprecision_rows": sum(r["interval_subprecision_margin"] for r in row_reports),
        "interval_radius_mm": float(interval_radius_mm),
        "coordinate_scope": ("current emitted F32 coordinates" if interval_radius_mm == 0 else "axiswise cube around emitted F32 coordinates"),
        "max_interval_error_to_positive_margin_ratio": max_ratio,
        "minimum_positive_margin_mm": min((m for m in margins if m > 0), default=None),
        "maximum_rounding_bound_mm": max(bounds, default=None),
        "by_pair_kind": {k: dict(v) for k, v in by_kind.items()},
        "units": {
            "input_points": "Float32 world metres, exact emitted array",
            "row_coefficients": "mm per mm displacement",
            "preserved_pair_margin_and_rounding_bound": "mm",
            "delta_rhs": "mm displacement threshold, often negative by design",
        },
        "method": "sum(abs(merged_coefficient)*half(max(adjacent_F32_spacings))*1000)",
        "interval_bound_method": "maximum absolute coordinate over axiswise +/-radius interval, rounded to F32, then symmetric neighbor-spacing bound",
        "scope_limit": "rounding sensitivity only; no constraint or geometry gate is changed",
    }
    return {"summary": summary, "rows": row_reports}
