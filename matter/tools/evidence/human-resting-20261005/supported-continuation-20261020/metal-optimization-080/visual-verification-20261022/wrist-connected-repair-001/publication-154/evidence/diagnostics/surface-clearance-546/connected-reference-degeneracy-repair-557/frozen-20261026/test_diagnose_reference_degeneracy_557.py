from pathlib import Path
import importlib.util

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SOURCE = Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/round-029-retained028/candidate.npz")
SOURCE_027 = Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/topology/round-027-all-target-edges025/round-001/candidate.npz")
SHA_027 = "331b4debe2dfd8a2da90278c669673f4ddf591beea48e841cc39c5fb9036d2ef"

spec = importlib.util.spec_from_file_location("diagnostic557", HERE / "diagnose_reference_degeneracy_557.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_current_best029_is_lineage_repaired_but_flip_profile_rejected():
    proposed, repaired_base, report = m.build_proposal(SOURCE)
    assert report["status"] == "DIAGNOSTIC_FLIP_REJECTED_BY_REFERENCE_DEGENERACY_PROFILE"
    assert report["candidate_selected"] is False
    assert report["native_admission"] is False
    assert report["operation_candidate_emitted"] is False
    assert report["profile_rejections"] == ["minimum_current_angle_below_5deg"]
    assert min(report["profile_metrics"]["new_current_min_angles_deg"]) < 5.0
    assert report["legacy_541"]["pass"] is False
    assert report["legacy_541"]["reason"] == "angle_loss_bound"
    assert report["preservation"]["base_affine_lineage_audit"]["stale_singleton_face_barycentric_count"] == 266
    assert report["preservation"]["base_affine_lineage_audit"]["singleton_parent_face_count"] == 41098
    assert report["preservation"]["base_affine_lineage_audit"]["multi_parent_face_count"] == 635
    assert report["preservation"]["base_barycentric_repair"]["changed_row_count"] == 266
    assert report["preservation"]["base_non_bary_payload_unchanged"]
    assert report["preservation"]["base_barycentric_repair"]["remaining_singleton_mismatches"] == 0
    assert report["preservation"]["post_flip_barycentric_repair"]["remaining_singleton_mismatches"] == 0
    assert report["preservation"]["coordinates_bitwise_unchanged"]
    assert report["preservation"]["typed_anchor_rows_and_coordinates_bitwise_unchanged"]
    assert report["preservation"]["global_boundary_edges_unchanged"]
    assert report["preservation"]["two_face_patch_boundary_unchanged"]
    assert report["preservation"]["all_other_face_rows_unchanged"]
    original = m.load_candidate(SOURCE)
    assert repaired_base["points_world_m"].tobytes() == original["points_world_m"].tobytes()
    assert repaired_base["source_O_faces"].tobytes() == original["source_O_faces"].tobytes()
    assert proposed["points_world_m"].tobytes() == original["points_world_m"].tobytes()
    assert report["preservation"]["csr_unions"]["face_source_ancestry_face_rows"] == [88498]
    assert report["preservation"]["csr_unions"]["face_fine_subface_ancestry_rows"] == [88506]
    assert report["preservation"]["csr_unions"]["refinement_face_parent_ancestry_face_rows"] == [27587]


def test_topology027_profile_qualifies_as_unselected_diagnostic_flip():
    proposed, repaired_base, report = m.build_proposal(SOURCE_027, SHA_027)
    assert report["status"] == "DIAGNOSTIC_REFERENCE_DEGENERACY_REPAIR_PROPOSAL_NOT_FULLY_CHECKED"
    assert report["operation_candidate_emitted"] is True
    assert report["profile_rejections"] == []
    assert report["profile_metrics"]["minimum_reference_area_gain"] >= 1.0e6
    assert min(report["profile_metrics"]["new_current_min_angles_deg"]) >= 5.0
    assert min(report["profile_metrics"]["new_area_ratios"]) >= 0.1
    assert report["legacy_541"]["pass"] is False
    assert report["legacy_541"]["reason"] == "angle_loss_bound"
    assert report["preservation"]["base_affine_lineage_audit"]["stale_singleton_face_barycentric_count"] == 266
    assert report["preservation"]["base_affine_lineage_audit"]["singleton_parent_face_count"] == 41098
    assert report["preservation"]["base_affine_lineage_audit"]["multi_parent_face_count"] == 635
    assert report["preservation"]["base_barycentric_repair"]["changed_row_count"] == 266
    assert report["preservation"]["base_non_bary_payload_unchanged"]
    assert report["preservation"]["base_barycentric_repair"]["remaining_singleton_mismatches"] == 0
    assert report["preservation"]["post_flip_barycentric_repair"]["remaining_singleton_mismatches"] == 0
    assert repaired_base["points_world_m"].tobytes() == m.load_candidate(SOURCE_027)["points_world_m"].tobytes()
    assert proposed["points_world_m"].tobytes() == repaired_base["points_world_m"].tobytes()
    assert np.isfinite(proposed["refinement_face_parent_barycentric"][[30884, 30886]]).all()


def test_rejected_current_best_writer_emits_only_lineage_repaired_base(tmp_path):
    out = tmp_path / "diagnostic"
    report = m.write_proposal(SOURCE, out)
    assert report["operation_candidate_emitted"] is False
    assert (out / "lineage-repaired-base.npz").exists()
    assert not (out / "diagnostic-proposal.npz").exists()


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"old_reference_cross_mm2": 1e-5}, "old_reference_not_near_collinear"),
        ({"new_reference_cross_mm2": [1e-9, 2e-9]}, "reference_area_gain_below_1e6"),
        ({"normal_cos": [0.5, -1e-8]}, "current_reference_orientation_invalid"),
        ({"current_min_angle_deg": [4.9, 8.0]}, "minimum_current_angle_below_5deg"),
        ({"area_ratio": [0.099, 0.7]}, "037_area_ratio_below_0.1"),
    ],
)
def test_profile_rejects_each_failed_diagnostic_gate(kwargs, expected):
    values = {
        "old_reference_cross_mm2": 1e-13,
        "new_reference_cross_mm2": [0.1, 0.2],
        "normal_cos": [0.8, 0.9],
        "current_min_angle_deg": [8.0, 9.0],
        "area_ratio": [0.7, 2.0],
    }
    values.update(kwargs)
    reasons, _ = m.profile_rejections(
        values["old_reference_cross_mm2"],
        np.asarray(values["new_reference_cross_mm2"]),
        np.asarray(values["normal_cos"]),
        np.asarray(values["current_min_angle_deg"]),
        np.asarray(values["area_ratio"]),
    )
    assert expected in reasons


def test_non_witness_topology_is_rejected_without_generalizing_profile():
    c = m.load_candidate(SOURCE)
    c["source_O_faces"][30886, 0] = 18624
    assert m.witness_face_rows(c["source_O_faces"]) == []
