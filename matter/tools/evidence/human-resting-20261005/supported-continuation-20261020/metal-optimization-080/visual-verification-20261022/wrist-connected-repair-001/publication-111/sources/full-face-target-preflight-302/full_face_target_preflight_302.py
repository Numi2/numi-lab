"""Fail-closed preflight for matched full-surface face-target reports."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _count(report: Mapping[str, Any], key: str, expected: int) -> None:
    value = report.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer count")
    if value != expected:
        raise ValueError(f"{key} mismatch: expected {expected}, got {value}")


def validate_full_face_target_report(
    report: Mapping[str, Any],
    *,
    expected_candidate_sha256: str,
    expected_surface_report_sha256: str,
    expected_surface_pair_count: int,
    expected_banks: int = 185,
) -> dict[str, Any]:
    """Validate target completeness against the exact matched surface inventory.

    This certifies only that the construction-target report is complete and
    matched to its inputs. It does not certify clearance or physical admission.
    """
    if not isinstance(report, Mapping):
        raise ValueError("face-target report must be a JSON object")
    for label, value in (
        ("expected_candidate_sha256", expected_candidate_sha256),
        ("expected_surface_report_sha256", expected_surface_report_sha256),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{label} must be a 64-character SHA-256")
    for label, value in (
        ("expected_surface_pair_count", expected_surface_pair_count),
        ("expected_banks", expected_banks),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a nonnegative integer")

    if report.get("matched_159_complete") is not True:
        raise ValueError("face-target report lacks complete matched 159 audit")
    failures = report.get("failure_counts")
    if not isinstance(failures, Mapping) or failures:
        raise ValueError("face-target report has missing or nonempty failure_counts")
    _count(report, "banks", expected_banks)
    for key in (
        "passing_targets",
        "crossing_pairs",
        "matched_surface_bank_triangle_pairs",
        "result_rows",
    ):
        _count(report, key, expected_surface_pair_count)

    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("face-target report inputs are missing")
    if inputs.get("candidate_sha256") != expected_candidate_sha256:
        raise ValueError("face-target report candidate hash mismatch")
    if inputs.get("surface_report_sha256") != expected_surface_report_sha256:
        raise ValueError("face-target report surface hash mismatch")

    return {
        "status": "PASS_COMPLETE_MATCHED_FACE_TARGETS_ONLY",
        "classification": "construction target completeness; not clearance admission",
        "banks": expected_banks,
        "surface_pair_count": expected_surface_pair_count,
        "candidate_sha256": expected_candidate_sha256,
        "surface_report_sha256": expected_surface_report_sha256,
    }
