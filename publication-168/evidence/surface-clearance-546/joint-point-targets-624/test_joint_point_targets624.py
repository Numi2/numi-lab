from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("joint_point_targets624_test_subject",
                                               HERE / "run_joint_point_targets624.py")
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def angular(source, current, depth, floor):
    # Same active-set implementation as the pinned 597 helper, imported by test
    # from the source file; this wrapper keeps the test independent of runner CLI.
    spec = importlib.util.spec_from_file_location(
        "angular597_test_subject", subject.ANGULAR597)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.nearest_joint_direction597(source, current, depth, floor)


def test_obtuse_source_current_normals_can_have_joint_direction():
    source = np.array([1.0, 0.0, 0.0])
    current = np.array([-0.1, np.sqrt(0.99), 0.0])
    depth = np.array([0.0, 1.0, 0.0])
    assert float(source @ current) < 0.0
    directions, diagnostics = subject.compute_joint_directions(
        [7], {7: source}, {7: current},
        {7: [{"typed_id": "body43/bone16", "source_row": 2,
              "normal_world_unit_covector": depth}]},
        angular)
    diag = diagnostics[7]
    assert directions[7].shape == (3,)
    assert diag["selected_source_hemisphere_cosine"] >= 0.05
    assert diag["selected_current_cosine"] >= 0.05
    assert diag["selected_depth_cosine"] >= 0.05
    assert diag["angular_selection"]["selected_depth_cosine"] == pytest.approx(
        float(depth @ directions[7]))


def test_opposite_depth_is_fail_closed_for_whole_inventory():
    source = current = np.array([1.0, 0.0, 0.0])
    depth = np.array([-1.0, 0.0, 0.0])
    with pytest.raises(subject.JointDirectionFailure) as exc:
        subject.compute_joint_directions(
            [9], {9: source}, {9: current},
            {9: [{"typed_id": "x", "source_row": 1,
                  "normal_world_unit_covector": depth}]},
            angular)
    assert exc.value.records[0]["query_row"] == 9
    assert exc.value.records[0]["status"] == "FAIL_CLOSED"


def test_infeasible_depth_does_not_return_partial_direction_map():
    rows = [4, 5]
    source = {r: np.array([1.0, 0.0, 0.0]) for r in rows}
    current = {r: np.array([1.0, 0.0, 0.0]) for r in rows}
    depths = {
        4: [{"typed_id": "ok", "source_row": 1,
             "normal_world_unit_covector": np.array([0.0, 1.0, 0.0])}],
        5: [{"typed_id": "bad", "source_row": 2,
             "normal_world_unit_covector": np.array([-1.0, 0.0, 0.0])}],
    }
    with pytest.raises(subject.JointDirectionFailure) as exc:
        subject.compute_joint_directions(rows, source, current, depths, angular)
    assert [r["query_row"] for r in exc.value.records] == rows
    assert [r["status"] for r in exc.value.records] == ["PASS", "FAIL_CLOSED"]


def test_duplicate_query_row_identity_is_rejected():
    with pytest.raises(ValueError, match="duplicate identities"):
        subject.compute_joint_directions(
            [3, 3], {3: np.array([1.0, 0.0, 0.0])},
            {3: np.array([1.0, 0.0, 0.0])},
            {3: [{"typed_id": "x", "source_row": 1,
                  "normal_world_unit_covector": np.array([1.0, 0.0, 0.0])}]},
            angular)


def test_duplicate_depth_rows_are_not_silently_selected():
    with pytest.raises(subject.JointDirectionFailure) as exc:
        subject.compute_joint_directions(
            [3], {3: np.array([1.0, 0.0, 0.0])},
            {3: np.array([1.0, 0.0, 0.0])},
            {3: [
                {"typed_id": "x", "source_row": 1,
                 "normal_world_unit_covector": np.array([1.0, 0.0, 0.0])},
                {"typed_id": "y", "source_row": 2,
                 "normal_world_unit_covector": np.array([1.0, 0.0, 0.0])},
            ]},
            angular)
    assert "exactly one" in exc.value.records[0]["reason"]


def test_parent_prefix_requires_bitwise_identity():
    parent = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    child = np.vstack([parent, np.array([[7, 8, 9]], dtype=np.float32)])
    digest = subject.sha256_array_bytes(parent)
    assert subject.validate_parent_prefix(parent, child, 2, digest) == digest
    child[1, 1] = np.nextafter(child[1, 1], np.float32(np.inf))
    with pytest.raises(ValueError, match="differs bitwise"):
        subject.validate_parent_prefix(parent, child, 2, digest)


def test_row_identity_and_all185_endpoint_counts_are_exact():
    rows = np.array([2, 8], dtype=np.int64)
    targets = [
        {"query_row": 2, "status": "PASS_FIRST_UNION_EXIT",
         "target_exactly_outside_all185": True, "exact_endpoint_bank_count": 185,
         "end_outside_all185_by_event_parity": True,
         "selection": {"target_exactly_outside_all_banks": True,
                       "classified_bank_count": 185}},
        {"query_row": 8, "status": "PASS_FIRST_UNION_EXIT",
         "target_exactly_outside_all185": True, "exact_endpoint_bank_count": 185,
         "end_outside_all185_by_event_parity": True,
         "selection": {"target_exactly_outside_all_banks": True,
                       "classified_bank_count": 185}},
    ]
    subject.validate_row_identity(rows, targets, rows.copy())
    subject.validate_all185_records(targets, 185)
    bad = [dict(targets[0]), dict(targets[1])]
    bad[1]["selection"] = {"target_exactly_outside_all_banks": True,
                           "classified_bank_count": 184}
    with pytest.raises(ValueError, match="all-185"):
        subject.validate_all185_records(bad, 185)
    with pytest.raises(ValueError, match="identities/order"):
        subject.validate_row_identity(rows, targets[::-1], rows.copy())


@pytest.mark.parametrize("target", [
    np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
    np.array([[-0.1, 0.0, 0.0]], dtype=np.float32),
])
def test_emitted_float32_zero_or_negative_outward_delta_fails(target):
    start = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    axis = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="emitted Float32 displacement"):
        subject.validate_emitted_f32_displacements(
            start, target, axis, axis, axis, axis)


def test_emitted_float32_positive_delta_passes_all_three_normals():
    start = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    target = np.array([[0.125, 0.0, 0.0]], dtype=np.float32)
    axis = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    result = subject.validate_emitted_f32_displacements(
        start, target, axis, axis, axis, axis)
    assert result["row_count"] == 1
    assert result["minimum_emitted_displacement_dots"] == {
        "source": 0.125, "current": 0.125, "depth": 0.125}
