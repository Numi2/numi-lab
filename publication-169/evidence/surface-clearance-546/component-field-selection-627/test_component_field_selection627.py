import numpy as np
import pytest

from component_field_selection627 import (
    ValidationError, connected_changed_components, select_safe_components,
    validate_159_report,
)


def bank(total, face_counts):
    return {"total": total, "face_counts": dict(face_counts)}


def test_disconnected_components_select_only_productive_component():
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    changed = np.array([0, 1, 3, 4])
    banks0 = {(43, 16): bank(4, {0: 2, 1: 2})}
    banks1 = {(43, 16): bank(4, {0: 1, 1: 3})}
    result = select_safe_components(faces, changed, banks0, banks1)
    assert result["selected_component_ids"] == [0]
    assert result["predicted_bank_counts"]["43/16"] == 3
    assert result["predicted_total_delta"] == -1
    assert result["components"][1]["no_bankwise_increase"] is False


def test_changed_nodes_sharing_face_are_unioned():
    faces = np.array([[0, 1, 2], [2, 3, 4], [8, 9, 10]], dtype=np.int64)
    components, component_faces, owner = connected_changed_components(faces, [0, 1, 3, 8])
    assert [x.tolist() for x in components] == [[0, 1], [3], [8]]
    assert component_faces[0].tolist() == [0]
    assert component_faces[1].tolist() == [1]
    assert owner == {0: 0, 1: 1, 2: 2}


def test_shared_triangle_merges_all_changed_vertices():
    components, component_faces, owner = connected_changed_components(
        np.array([[0, 2, 5], [2, 3, 7]], dtype=np.int64), [0, 2, 3, 7]
    )
    assert components[0].tolist() == [0, 2, 3, 7]
    assert component_faces[0].tolist() == [0, 1]
    assert owner == {0: 0, 1: 0}


def test_component_rejected_if_any_bank_increases_even_when_total_falls():
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    before = {(43, 16): bank(2, {0: 1}), (65, 99): bank(1, {0: 1})}
    after = {(43, 16): bank(0, {}), (65, 99): bank(2, {0: 2})}
    result = select_safe_components(faces, [0], before, after)
    assert result["selected_component_ids"] == []
    assert result["predicted_total"] == 3
    assert result["predicted_total_delta"] == 0


def test_report_hash_mismatch_and_incomplete_coverage_fail_closed():
    report = {
        "status": "COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT",
        "complete": True,
        "candidate_sha256": "a" * 64,
        "bank_count": 185,
        "expected_bank_count": 185,
        "banks": [],
    }
    with pytest.raises(ValidationError, match="candidate hash"):
        validate_159_report(report, "b" * 64)
    with pytest.raises(ValidationError, match="bank list length"):
        validate_159_report(report, "a" * 64)
