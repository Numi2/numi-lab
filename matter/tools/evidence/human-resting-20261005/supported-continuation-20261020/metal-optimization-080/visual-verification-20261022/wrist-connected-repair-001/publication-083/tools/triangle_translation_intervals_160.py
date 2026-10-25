"""Proposal-only continuous SAT intervals for translating one triangle.
Intervals aid directional construction; they are not clearance certificates.
"""
from __future__ import annotations
from typing import Any, Iterable, Mapping
import math
import numpy as np


def _triangle(value: Any, name: str) -> np.ndarray:
    tri = np.asarray(value, dtype=np.float64)
    if tri.shape != (3, 3) or not np.all(np.isfinite(tri)):
        raise ValueError(f"{name} must be a finite (3, 3) array")
    edges = np.roll(tri, -1, axis=0) - tri
    if float(np.linalg.norm(np.cross(edges[0], -edges[2]))) == 0.0:
        raise ValueError(f"{name} must be nondegenerate")
    return tri


def _axes(a: np.ndarray, b: np.ndarray) -> list[tuple[str, np.ndarray]]:
    ea = np.roll(a, -1, axis=0) - a
    eb = np.roll(b, -1, axis=0) - b
    na, nb = np.cross(ea[0], -ea[2]), np.cross(eb[0], -eb[2])
    out: list[tuple[str, np.ndarray]] = []

    def add(name: str, axis: np.ndarray) -> None:
        norm = float(np.linalg.norm(axis))
        # Keep every representable nonzero axis; no near-parallel cutoff.
        if math.isfinite(norm) and norm > 0.0:
            out.append((name, axis / norm))

    add("moving_normal", na)
    add("fixed_normal", nb)
    for i in range(3):
        for j in range(3):
            add(f"edge_cross_{i}_{j}", np.cross(ea[i], eb[j]))
    # Necessary in-plane separators for coplanar triangles.
    for i in range(3):
        add(f"moving_normal_cross_edge_{i}", np.cross(na, ea[i]))
        add(f"fixed_normal_cross_edge_{i}", np.cross(nb, eb[i]))
    if not out:
        raise ValueError("no finite SAT axes")
    return out


def triangle_translation_interval(
    moving_triangle_m: Any,
    fixed_triangle_m: Any,
    direction_world: Any,
    max_distance_m: float,
) -> dict[str, Any] | None:
    """Candidate overlap interval for A+s*d against B, clipped to [0, cap].

    Direction is normalized and distances are meters. Contact is inclusive.
    This floating-point SAT proposal is not a clearance decision; exact
    all-bank and whole-surface audits remain mandatory.
    """
    a = _triangle(moving_triangle_m, "moving_triangle_m")
    b = _triangle(fixed_triangle_m, "fixed_triangle_m")
    d = np.array(direction_world, dtype=np.float64, copy=True)
    if d.shape != (3,) or not np.all(np.isfinite(d)):
        raise ValueError("direction_world must be a finite length-3 vector")
    dn = float(np.linalg.norm(d))
    if dn == 0.0 or not math.isfinite(dn):
        raise ValueError("direction_world must be nonzero")
    cap = float(max_distance_m)
    if cap < 0.0 or not math.isfinite(cap):
        raise ValueError("max_distance_m must be finite and nonnegative")
    d /= dn
    lo, hi = 0.0, cap
    lo_axes: list[dict[str, Any]] = []
    hi_axes: list[dict[str, Any]] = []
    axis_count = 0
    for name, axis in _axes(a, b):
        axis_count += 1
        pa, pb = a @ axis, b @ axis
        amin, amax = float(pa.min()), float(pa.max())
        bmin, bmax = float(pb.min()), float(pb.max())
        vel = float(d @ axis)
        if vel == 0.0:
            if amax < bmin or bmax < amin:
                return None
            continue
        s0, s1 = (bmin - amax) / vel, (bmax - amin) / vel
        axis_lo, axis_hi = min(s0, s1), max(s0, s1)
        if axis_lo > lo:
            lo = axis_lo
            lo_axes = [{"label": name, "axis": axis.tolist()}]
        elif axis_lo == lo:
            lo_axes.append({"label": name, "axis": axis.tolist()})
        if axis_hi < hi:
            hi = axis_hi
            hi_axes = [{"label": name, "axis": axis.tolist()}]
        elif axis_hi == hi:
            hi_axes.append({"label": name, "axis": axis.tolist()})
        if lo > hi:
            return None
    if hi < 0.0 or lo > cap:
        return None
    entry, exit_ = max(0.0, lo), min(cap, hi)
    if entry > exit_:
        return None
    return {
        "entry_distance_m": float(entry),
        "exit_distance_m": float(exit_),
        "entry_axes": lo_axes,
        "exit_axes": hi_axes,
        "sat_axis_count": axis_count,
        "candidate_only": True,
        "clearance_certificate": False,
    }


def union_triangle_translation_intervals(
    pair_intervals: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge overlapping candidate intervals without adding a tolerance.

    Input records contain pair_id and interval; interval may be None. Pair IDs
    are retained. Touching intervals merge only on exact float comparison.
    """
    rows = []
    for item in pair_intervals:
        if "pair_id" not in item or "interval" not in item:
            raise ValueError("each row needs pair_id and interval")
        val = item["interval"]
        if val is None:
            continue
        start, end = float(val["entry_distance_m"]), float(val["exit_distance_m"])
        if not (math.isfinite(start) and math.isfinite(end) and 0.0 <= start <= end):
            raise ValueError("invalid candidate interval")
        rows.append((start, end, item["pair_id"]))
    rows.sort(key=lambda r: (r[0], r[1], repr(r[2])))
    result: list[dict[str, Any]] = []
    for start, end, pair_id in rows:
        if not result or start > result[-1]["exit_distance_m"]:
            result.append({
                "entry_distance_m": start, "exit_distance_m": end,
                "pair_ids": [pair_id], "candidate_only": True,
                "clearance_certificate": False,
            })
        else:
            cur = result[-1]
            cur["exit_distance_m"] = max(cur["exit_distance_m"], end)
            if pair_id not in cur["pair_ids"]:
                cur["pair_ids"].append(pair_id)
    return result


def first_current_overlap_exit(
    merged_intervals: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Return only the union component containing s=0, never a later hit."""
    for item in merged_intervals:
        start, end = float(item["entry_distance_m"]), float(item["exit_distance_m"])
        if start <= 0.0 <= end:
            return {
                "candidate_first_exit_distance_m": end,
                "pair_ids": list(item.get("pair_ids", [])),
                "candidate_only": True, "clearance_certificate": False,
            }
    return None

