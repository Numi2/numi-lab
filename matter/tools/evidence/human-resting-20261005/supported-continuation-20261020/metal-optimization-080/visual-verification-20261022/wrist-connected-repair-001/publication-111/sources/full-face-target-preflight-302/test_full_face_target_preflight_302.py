from __future__ import annotations

import copy
import pytest

from full_face_target_preflight_302 import validate_full_face_target_report


CANDIDATE_SHA = "a" * 64
SURFACE_SHA = "b" * 64
PAIR_COUNT = 3886


def complete_report():
    return {
        "matched_159_complete": True,
        "failure_counts": {},
        "banks": 185,
        "passing_targets": PAIR_COUNT,
        "crossing_pairs": PAIR_COUNT,
        "matched_surface_bank_triangle_pairs": PAIR_COUNT,
        "result_rows": PAIR_COUNT,
        "inputs": {
            "candidate_sha256": CANDIDATE_SHA,
            "surface_report_sha256": SURFACE_SHA,
        },
    }


def validate(report):
    return validate_full_face_target_report(
        report,
        expected_candidate_sha256=CANDIDATE_SHA,
        expected_surface_report_sha256=SURFACE_SHA,
        expected_surface_pair_count=PAIR_COUNT,
        expected_banks=185,
    )


def test_complete_matched_report_passes_without_claiming_admission():
    result = validate(complete_report())
    assert result["status"] == "PASS_COMPLETE_MATCHED_FACE_TARGETS_ONLY"
    assert result["surface_pair_count"] == PAIR_COUNT
    assert "not clearance admission" in result["classification"]


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(matched_159_complete=False),
        lambda r: r.update(failure_counts={"no_exit": 1}),
        lambda r: r.update(banks=184),
    ],
    ids=["incomplete-159", "reported-failures", "not-185-banks"],
)
def test_incomplete_or_wrong_inventory_fails_closed(change):
    report = complete_report()
    change(report)
    with pytest.raises(ValueError):
        validate(report)


@pytest.mark.parametrize(
    "field",
    [
        "passing_targets",
        "crossing_pairs",
        "matched_surface_bank_triangle_pairs",
        "result_rows",
    ],
)
def test_any_target_or_surface_count_drop_fails(field):
    report = complete_report()
    report[field] = PAIR_COUNT - 1
    with pytest.raises(ValueError, match=field):
        validate(report)


@pytest.mark.parametrize(
    ("key", "foreign_sha"),
    [
        ("candidate_sha256", "c" * 64),
        ("surface_report_sha256", "d" * 64),
    ],
)
def test_foreign_input_hash_fails(key, foreign_sha):
    report = complete_report()
    report["inputs"][key] = foreign_sha
    with pytest.raises(ValueError, match="hash mismatch"):
        validate(report)


@pytest.mark.parametrize("field", ["passing_targets", "result_rows"])
def test_boolean_is_not_accepted_as_integer_count(field):
    report = complete_report()
    report[field] = True
    with pytest.raises(ValueError, match="integer count"):
        validate(report)


def test_missing_inputs_fails_closed():
    report = complete_report()
    report.pop("inputs")
    with pytest.raises(ValueError, match="inputs"):
        validate(report)
