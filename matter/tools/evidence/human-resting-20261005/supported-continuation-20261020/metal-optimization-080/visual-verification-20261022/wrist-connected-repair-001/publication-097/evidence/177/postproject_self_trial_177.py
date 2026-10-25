"""Fail-closed construction proposal formed from the self-projected ADMM block.

This helper does not certify geometric self-intersection freedom and does not
relax other constraints. The caller must still run its exact emitted-F32 self
audit and every existing point, area, cap, anchor, and progress gate.
"""
from __future__ import annotations

import math
import numpy as np


def postproject_self_trial(
    self_projected_mm,
    anchors,
    self_rows,
    self_safety_margin_mm,
    *,
    anchor_values_mm=None,
):
    """Clamp fixed anchors on the self-projected iterate and check its rows.

    The caller should use the returned proposal directly (not average it with
    the node/area projections). The reported row check mirrors the existing
    projection-slack gate; it is not exact row satisfaction. If clamping
    exceeds that existing slack, the trial fails closed.
    """
    proposal = np.asarray(self_projected_mm, dtype=np.float64)
    if proposal.ndim != 2 or proposal.shape[1] != 3 or not np.isfinite(proposal).all():
        raise ValueError("self-projected proposal must be finite N x 3 millimetres")
    proposal = proposal.copy()

    fixed = np.asarray(anchors, dtype=np.int64)
    if fixed.ndim != 1 or np.any(fixed < 0) or np.any(fixed >= len(proposal)):
        raise ValueError("anchor rows must be a valid one-dimensional index array")
    if len(np.unique(fixed)) != len(fixed):
        raise ValueError("duplicate anchor row")
    if anchor_values_mm is None:
        values = np.zeros((len(fixed), 3), dtype=np.float64)
    else:
        values = np.asarray(anchor_values_mm, dtype=np.float64)
        if values.shape != (len(fixed), 3) or not np.isfinite(values).all():
            raise ValueError("anchor values must be finite K x 3 millimetres")
    safety = float(self_safety_margin_mm)
    if not math.isfinite(safety) or safety <= 0:
        raise ValueError("self safety margin must be finite and positive")

    proposal[fixed] = values
    max_violation = 0.0
    for row in self_rows:
        nodes = np.asarray(row.get("nodes"), dtype=np.int64)
        coeffs = np.asarray(row.get("coefficients"), dtype=np.float64)
        lower = float(row.get("lower", float("nan")))
        if nodes.ndim != 1 or len(nodes) == 0 or np.any(nodes < 0) or np.any(nodes >= len(proposal)):
            raise ValueError("self row has invalid node indices")
        if len(np.unique(nodes)) != len(nodes):
            raise ValueError("self row repeats a node index")
        if coeffs.shape != (len(nodes), 3) or not np.isfinite(coeffs).all() or not math.isfinite(lower):
            raise ValueError("self row has malformed coefficients or lower bound")
        value = math.fsum(
            float(coeffs[j, axis]) * float(proposal[node, axis])
            for j, node in enumerate(nodes)
            for axis in range(3)
        )
        max_violation = max(max_violation, max(0.0, lower - value))

    anchors_unchanged = bool(np.array_equal(proposal[fixed], values))
    report = {
        "status": "SELF_PROJECTION_SLACK_AND_ANCHORS_PASS" if anchors_unchanged and max_violation < safety else "REJECT_SELF_PROJECTION_SLACK_OR_ANCHOR_GATE",
        "self_projection_slack_gate_pass": bool(max_violation < safety),
        "self_barrier_rows_satisfied_exactly": bool(max_violation == 0.0),
        "max_self_linear_violation_mm": float(max_violation),
        "self_safety_margin_mm": safety,
        "anchors_unchanged": anchors_unchanged,
        "exact_emitted_f32_self_audit_required": True,
        "other_existing_gates_required": True,
        "geometry_admitted": False,
        "accepted_physical_steps": 0,
    }
    return proposal, report
