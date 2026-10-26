import importlib.util
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).with_name("run_coherent_depth_targets546.py")
SPEC = importlib.util.spec_from_file_location("coherent_depth546_tested", SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_invocation_requires_ordinary_coherent_mesh_witness_mode():
    requested = M.validate_invocation([
        "--direction-mode", "coherent-depth",
        "--coherent-direction", "depth",
        "--minimum-depth-cosine", "0.05",
    ])
    assert requested["requested_direction_mode"] == "coherent-depth"
    assert requested["requested_coherent_direction"] == "depth"
    assert requested["joint_source_patch"] is False
    for argv in (
        [],
        ["--direction-mode", "direct-depth", "--coherent-direction", "depth"],
        ["--direction-mode", "coherent-depth"],
        ["--direction-mode", "coherent-depth", "--coherent-direction", "source-outward"],
        ["--direction-mode", "coherent-depth", "--coherent-direction", "depth", "--joint-source-patch"],
        ["--direction-mode", "coherent-depth", "--coherent-direction", "depth",
         "--minimum-depth-cosine", "0.1"],
    ):
        with pytest.raises(ValueError):
            M.validate_invocation(argv)


def _record(row, group_rows, *, status="PASS_FIRST_UNION_EXIT", exact=185,
            translation=(0.001, 0.0, 0.0), depth_cos=0.1):
    rec = {
        "query_row": row,
        "status": status,
        "target_exactly_outside_all185": status == "PASS_FIRST_UNION_EXIT",
        "exact_endpoint_bank_count": 185 if status == "PASS_FIRST_UNION_EXIT" else 0,
        "direction_compatibility": {"selected_depth_cosine": depth_cos},
        "selection": {},
        "coherent_group_rows": group_rows,
    }
    if status == "PASS_FIRST_UNION_EXIT":
        rec.update({
            "coherent_exact_endpoint_bank_count": exact,
            "coherent_translation_world_m": list(translation),
        })
        rec["selection"]["coherent_endpoint_exact_outside_all185"] = exact == 185
    return rec


def _report(targets, groups, *, status="PASS_FIRST_UNION_EXIT_TARGETS_ONLY"):
    return {
        "direction_mode": "coherent-depth",
        "coherent_direction_policy": "depth",
        "joint_source_support_patch": False,
        "minimum_depth_cosine": 0.05,
        "status": status,
        "query_rows": len(targets),
        "targets": targets,
        "coherent_group_rows": groups,
    }


def _arrays(path, rows, pass_mask):
    np.savez(
        path,
        query_rows=np.array(rows, np.int64),
        pass_mask=np.asarray(pass_mask, bool),
        target_exactly_outside_all185=np.asarray(pass_mask, bool),
        direction_compatibility_mask=np.ones(len(rows), bool),
    )


REQUESTED = {
    "requested_direction_mode": "coherent-depth",
    "requested_coherent_direction": "depth",
    "joint_source_patch": False,
    "requested_minimum_depth_cosine_metadata": 0.05,
}


def test_multiple_mesh_witness_groups_certify_independently(tmp_path):
    arrays = tmp_path / "targets.npz"
    groups = [[5, 6], [7]]
    _arrays(arrays, [5, 6, 7], [True, True, True])
    report = _report([
        _record(5, groups[0]),
        _record(6, groups[0]),
        _record(7, groups[1], translation=(0.002, 0.0, 0.0)),
    ], groups)
    summary = M.summarize_group_certificates(report, arrays, REQUESTED)
    assert summary["grouping_semantics"].startswith("545 mesh-witness")
    assert summary["joint_source_patch"] is False
    assert summary["all_groups_exact_all185"] is True
    assert summary["passing_group_count"] == 2
    assert summary["failed_group_count"] == 0
    assert summary["native_admission"] is False
    assert summary["groups"][0]["shared_translation_world_m"] == [0.001, 0.0, 0.0]


def test_partial_groups_are_reported_without_success_promotion(tmp_path):
    arrays = tmp_path / "targets.npz"
    groups = [[5, 6], [7]]
    _arrays(arrays, [5, 6, 7], [True, False, True])
    report = _report([
        _record(5, groups[0]),
        _record(6, groups[0], status="COHERENT_TRANSLATION_ENDPOINT_REJECTED"),
        _record(7, groups[1]),
    ], groups, status="INCOMPLETE_OR_REJECTED_TARGETS")
    summary = M.summarize_group_certificates(report, arrays, REQUESTED)
    assert summary["all_groups_exact_all185"] is False
    assert summary["passing_group_count"] == 1
    assert summary["failed_group_count"] == 1
    assert summary["groups"][0]["status"] == "FAILED_GROUP_CERTIFICATE"
    assert summary["groups"][1]["status"] == "PASS_SHARED_TRANSLATION_ALL185"
    assert summary["native_admission"] is False
    assert summary["accepted_physical_steps"] == 0


def test_wrong_shared_endpoint_certificate_marks_group_failed(tmp_path):
    arrays = tmp_path / "targets.npz"
    _arrays(arrays, [5, 6], [True, True])
    targets = [_record(5, [5, 6], exact=185), _record(6, [5, 6], exact=184)]
    summary = M.summarize_group_certificates(_report(targets, [[5, 6]]), arrays, REQUESTED)
    assert summary["all_groups_exact_all185"] is False
    assert summary["groups"][0]["status"] == "FAILED_GROUP_CERTIFICATE"


def test_mismatched_shared_translations_fail_group(tmp_path):
    arrays = tmp_path / "targets.npz"
    _arrays(arrays, [5, 6], [True, True])
    report = _report([
        _record(5, [5, 6]),
        _record(6, [5, 6], translation=(0.002, 0.0, 0.0)),
    ], [[5, 6]])
    summary = M.summarize_group_certificates(report, arrays, REQUESTED)
    assert summary["all_groups_exact_all185"] is False
    assert {"query_row": None, "status": "SHARED_TRANSLATION_MISMATCH"} in summary["groups"][0]["failed_rows"]


def test_inconsistent_npz_status_mask_is_rejected(tmp_path):
    arrays = tmp_path / "targets.npz"
    _arrays(arrays, [5, 6], [True, False])
    report = _report([_record(5, [5, 6]), _record(6, [5, 6])], [[5, 6]])
    with pytest.raises(ValueError, match="masks do not reflect"):
        M.summarize_group_certificates(report, arrays, REQUESTED)


def test_groups_must_partition_the_actual_target_rows(tmp_path):
    arrays = tmp_path / "targets.npz"
    _arrays(arrays, [5, 6], [True, True])
    report = _report([_record(5, [5, 6]), _record(6, [5, 6])], [[5], [6]])
    summary = M.summarize_group_certificates(report, arrays, REQUESTED)
    assert summary["all_groups_exact_all185"] is False
    assert summary["failed_group_count"] == 2
    assert all(group["status"] == "FAILED_GROUP_CERTIFICATE" for group in summary["groups"])


def test_exact_owner_infeasibility_is_retained_as_failure_receipt(tmp_path):
    error = "coherent cluster has no positive common depth direction: [12, 14, 19]"
    parsed = M.parse_group_infeasibility(error)
    assert parsed["reported_group_rows"] == [12, 14, 19]
    assert parsed["exact_owner_error"] == error
    out = tmp_path / "failure"
    path, receipt = M.write_group_infeasibility_receipt(out, REQUESTED, error)
    assert path.is_file()
    assert receipt["target_rows_emitted"] is False
    assert receipt["group_enumeration_complete"] is False
    assert receipt["exact_all185_endpoint_checks_run"] is False
    assert receipt["native_admission"] is False
    assert receipt["accepted_physical_steps"] == 0


@pytest.mark.parametrize("error", [
    "joint source patch requires one exact typed bone bank",
    "coherent cluster has no positive common depth direction: (__import__('os').system('x'),)",
    "coherent cluster has no positive common depth direction: [-1]",
])
def test_unrelated_or_malformed_builder_failures_are_not_reclassified(error):
    assert M.parse_group_infeasibility(error) is None

