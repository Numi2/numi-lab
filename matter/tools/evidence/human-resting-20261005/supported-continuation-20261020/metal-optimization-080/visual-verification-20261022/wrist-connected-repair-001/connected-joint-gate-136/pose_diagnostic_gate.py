"""Fail-closed interpretation helpers for prescribed-LBS pose diagnostics.

A source-surface screen is kinematic geometry evidence only. It is not a
physical-motion result, and empty bone-pair lists do not clear skin self-crossings.
"""
from __future__ import annotations

import math
import struct


def classify_kinematic_pose(
    *,
    source_probe_accepted: bool | None,
    full_surface_self_pairs: int | None,
    incremental_roi_self_pairs: int | None = None,
    roi_vs_rest_skin_pairs: int | None = None,
    roi_vs_bone_pairs: int | None = None,
    bilateral_bone_surface_pairs: int | None = None,
    skin_to_bone_pairs: int | None = None,
) -> dict:
    """Classify static geometry without implying a physical solve.

    Unknown measurements remain unknown. In particular, a zero bilateral
    bone/bone pair count cannot override a nonzero full-surface skin count.
    """
    values = {
        "full_surface_self_pairs": full_surface_self_pairs,
        "incremental_roi_self_pairs": incremental_roi_self_pairs,
        "roi_vs_rest_skin_pairs": roi_vs_rest_skin_pairs,
        "roi_vs_bone_pairs": roi_vs_bone_pairs,
        "bilateral_bone_surface_pairs": bilateral_bone_surface_pairs,
        "skin_to_bone_pairs": skin_to_bone_pairs,
    }
    for name, value in values.items():
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer or None")

    if full_surface_self_pairs is not None and full_surface_self_pairs > 0:
        # This label is reserved for an actual measured self-intersection. A
        # failed source probe can reject for ancestry, transform, source-range,
        # or other admission reasons and is not itself a geometry witness.
        status = "rejected_source_self_intersection"
    elif source_probe_accepted is False and full_surface_self_pairs == 0:
        status = "rejected_source_probe_non_self_intersection"
    elif source_probe_accepted is False:
        status = "rejected_source_probe_unclassified"
    elif source_probe_accepted is not True or full_surface_self_pairs is None:
        status = "incomplete_source_surface_measurement"
    elif any(
        value is not None and value > 0
        for value in (
            incremental_roi_self_pairs,
            roi_vs_rest_skin_pairs,
            roi_vs_bone_pairs,
            bilateral_bone_surface_pairs,
            skin_to_bone_pairs,
        )
    ):
        status = "rejected_intersection_witness"
    elif any(
        value is None
        for value in (
            incremental_roi_self_pairs,
            roi_vs_rest_skin_pairs,
            roi_vs_bone_pairs,
            bilateral_bone_surface_pairs,
            skin_to_bone_pairs,
        )
    ):
        status = "incomplete_contact_geometry_measurements"
    else:
        status = "kinematic_geometry_clear_only"

    return {
        "status": status,
        "source_surface_clear": (
            source_probe_accepted is True and full_surface_self_pairs == 0
        ),
        "all_contact_geometry_measured": all(
            value is not None for value in values.values()
        ),
        "kinematic_geometry_clear": status == "kinematic_geometry_clear_only",
        "physical_motion_qualified": False,
        "physical_motion_reason": "prescribed LBS/FK pose diagnostic; no FEM motion was executed",
    }


def read_nheq_constraints(payload: bytes) -> tuple[int, int, list[tuple]]:
    """Decode the fixed NHEQ1 header and equality records needed for verification."""
    header_bytes = 80
    record_bytes_expected = 96
    if len(payload) < header_bytes:
        raise ValueError("truncated NHEQ payload")
    header = struct.unpack_from("<8s10I32s", payload, 0)
    magic = header[0]
    abi, nq, nv, count, record_bytes, source_count = header[1:7]
    reserved = header[7:11]
    if magic != b"NHEQ1\0\0\0" or abi != 1 or record_bytes != record_bytes_expected:
        raise ValueError("unsupported NHEQ header")
    if nq <= 0 or nv <= 0 or count <= 0 or count != source_count or any(reserved):
        raise ValueError("invalid NHEQ dimensions/reserved fields")
    if len(payload) != header_bytes + count * record_bytes:
        raise ValueError("NHEQ payload length mismatch")
    rows = []
    invalid_index = 0xFFFFFFFF
    for i in range(count):
        fields = struct.unpack_from("<4I20f", payload, header_bytes + i * record_bytes)
        dep_q, dep_v, master_q, master_v = fields[:4]
        floats = fields[4:]
        if not all(math.isfinite(value) for value in floats):
            raise ValueError(f"NHEQ row {i} has nonfinite coefficients or parameters")
        if dep_q >= nq or dep_v >= nv:
            raise ValueError(f"NHEQ row {i} dependent index out of range")
        fixed = master_q == invalid_index and master_v == invalid_index
        coupled = master_q < nq and master_v < nv
        if not fixed and not coupled:
            raise ValueError(f"NHEQ row {i} master index out of range or inconsistent")
        coeffs = floats[:7]  # y0, x0, a0..a4
        rows.append((dep_q, dep_v, master_q, master_v, *coeffs))
    return nq, nv, rows


def equality_residual_summary(q: list[float], payload: bytes) -> dict:
    nq, nv, rows = read_nheq_constraints(payload)
    if len(q) != nq or not all(math.isfinite(float(value)) for value in q):
        raise ValueError("pose q length or values invalid for NHEQ payload")
    out = []
    for dep_q, dep_v, master_q, master_v, y0, x0, a0, a1, a2, a3, a4 in rows:
        if master_q == 0xFFFFFFFF:
            target = y0 + a0
        else:
            dx = float(q[master_q]) - x0
            target = y0 + a0 + a1 * dx + a2 * dx**2 + a3 * dx**3 + a4 * dx**4
        actual = float(q[dep_q])
        if not math.isfinite(target) or not math.isfinite(actual):
            raise ValueError("NHEQ evaluation produced a nonfinite pose target")
        out.append({
            "dependent_q": dep_q,
            "master_q": None if master_q == 0xFFFFFFFF else master_q,
            "actual": actual,
            "expected": target,
            "absolute_residual": abs(actual - target),
        })
    return {
        "constraint_count": len(out),
        "max_absolute_residual": max(row["absolute_residual"] for row in out),
        "dependent_rows": out,
    }
