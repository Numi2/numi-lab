from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import build_coherent_point_targets_371 as owner


class FakeSourceOwner:
    def __init__(self, normals, fail_rows=()):
        self.normals = normals
        self.fail_rows = set(fail_rows)

    def registered_source_outward_face_normal(self, face_row, candidate, detail, base, orientation):
        if face_row in self.fail_rows:
            raise ValueError("missing source ancestry")
        return np.asarray(self.normals[face_row], dtype=np.float64), {"kind": "fixture", "row": face_row}


def candidate_for_normals(normals):
    points = np.asarray([
        [0, 0, 0], [0, 2, 0], [0, 0, 1], [3, 0, 0], [0, 0, 2],
    ], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
    return {
        "points_world_m": points,
        "source_O_faces": faces,
        "area_reference_037_points_world_m_f64": points.astype(np.float64),
    }


def test_source037_area_weighted_unique_incident_normals():
    candidate = candidate_for_normals(None)
    # Source face areas are 1 and 3 m^2; face normals are independently
    # supplied by the registered 303 ancestry resolver.
    source = FakeSourceOwner({0: [1, 0, 0], 1: [0, 1, 0]})
    normals, provenance = owner.vertex_source_outward_normals(
        candidate, source, {}, {}, {}, [0]
    )
    expected = np.asarray([1.0, 3.0, 0.0]) / np.sqrt(10.0)
    np.testing.assert_allclose(normals[0], expected, rtol=0, atol=2e-15)
    assert provenance[0]["incident_candidate_face_rows"] == [0, 1]
    assert provenance[0]["unique_incident_face_count"] == 2
    assert provenance[0]["source037_area_sum_m2"] == pytest.approx(4.0)


def test_normal_cancellation_fails_closed():
    candidate = candidate_for_normals(None)
    candidate["area_reference_037_points_world_m_f64"][4, 2] = 2.0 / 3.0
    source = FakeSourceOwner({0: [1, 0, 0], 1: [-1, 0, 0]})
    with pytest.raises(ValueError, match="cancel"):
        owner.vertex_source_outward_normals(candidate, source, {}, {}, {}, [0])


def test_missing_face_ancestry_fails_closed_without_fallback():
    candidate = candidate_for_normals(None)
    source = FakeSourceOwner({0: [1, 0, 0]}, fail_rows=(1,))
    with pytest.raises(ValueError, match="missing source ancestry"):
        owner.vertex_source_outward_normals(candidate, source, {}, {}, {}, [0])


def test_area_weights_reject_degenerate_or_misaligned_reference():
    candidate = candidate_for_normals(None)
    candidate["area_reference_037_points_world_m_f64"] = np.zeros((5, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="degenerate"):
        owner.vertex_source_outward_normals(candidate, FakeSourceOwner({}), {}, {}, {}, [0])


def test_300_selector_uses_first_union_empty_interval_and_complete_endpoint():
    selector_path = Path(owner.SELECTOR300)
    spec = importlib.util.spec_from_file_location("selector300_test_327", selector_path)
    selector = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(selector)
    keys = tuple((i, 0) for i in range(185))
    result = selector.first_representable_union_target_300(
        {keys[0]}, [{"distance_m": 0.002, "bank": keys[0]}],
        start_world_m=np.zeros(3), unit_direction_world=np.asarray([1.0, 0.0, 0.0]),
        clearance_m=5e-5,
        classify_endpoint=lambda p: ({key: "outside" for key in keys}, []),
        expected_bank_keys=keys,
    )
    assert result["status"] == "PASS_FIRST_CERTIFIED_PROPOSAL_INTERVAL"
    assert result["classified_bank_count"] == 185
    np.testing.assert_array_equal(result["target_world_f32_m"], np.asarray([0.00205, 0, 0], np.float32))


def test_cli_requires_all_four_input_paths():
    with pytest.raises(SystemExit):
        owner.main([])


def test_incomplete_or_crossing_surface_summary_is_rejected():
    with pytest.raises(ValueError, match="complete surface audit"):
        owner.validate_complete_surface_summary({"complete_surface_audit": False,
                                                 "exact_PLC_surface_pairs": 0})
    with pytest.raises(ValueError, match="zero exact PLC"):
        owner.validate_complete_surface_summary({"complete_surface_audit": True,
                                                 "exact_PLC_surface_pairs": 1})


@pytest.mark.parametrize(("bad_key", "label"), [
    ("points_sha256", "point"), ("nodes_sha256", "node"), ("pairs_sha256", "pair"),
])
def test_query_receipt_rejects_wrong_csv_hash(tmp_path, bad_key, label):
    paths = {key: tmp_path / f"{key}.csv" for key in ("points", "nodes", "pairs")}
    for key, path in paths.items():
        path.write_text(f"{key},fixture\n")
    receipt = {
        "query_rows": 1,
        "points_sha256": owner.sha(paths["points"]),
        "nodes_sha256": owner.sha(paths["nodes"]),
        "pairs_sha256": owner.sha(paths["pairs"]),
    }
    receipt[bad_key] = "0" * 64
    with pytest.raises(ValueError, match=f"{label} CSV SHA"):
        owner.validate_query_receipt(receipt, paths["points"], paths["nodes"],
                                      paths["pairs"], expected_rows=1)


def test_candidate_requires_native_f32_points_and_integer_faces():
    candidate = candidate_for_normals(None)
    points, faces = owner.validate_candidate_geometry_arrays(candidate)
    assert points.dtype == np.dtype(np.float32)
    assert np.issubdtype(faces.dtype, np.integer)
    candidate["points_world_m"] = candidate["points_world_m"].astype(np.float64)
    with pytest.raises(ValueError, match="stored as Float32"):
        owner.validate_candidate_geometry_arrays(candidate)


def test_segment_and_point_predicate_owners_have_separate_required_apis():
    segment_owner = {"classify_segment_triangle": lambda *_: {"classification": "disjoint"}}
    point_owner = SimpleNamespace(
        _records=lambda *args: (),
        prepare_point_location=lambda records: records,
        point_location_prepared_indexed=lambda point, prepared: {"location": "outside"},
    )
    owner.validate_exact_owner_apis(segment_owner, point_owner)
    del point_owner.prepare_point_location
    with pytest.raises(ValueError, match="point-location API"):
        owner.validate_exact_owner_apis(segment_owner, point_owner)


def test_pinned_point_owner_loads_as_its_package():
    point_owner = owner.load_point_location_owner()
    owner.validate_exact_owner_apis(
        {"classify_segment_triangle": lambda *_: {"classification": "disjoint"}},
        point_owner,
    )


def test_unbounded_union_interval_is_json_safe_and_explicit():
    selection = {
        "status": "PASS_FIRST_CERTIFIED_PROPOSAL_INTERVAL",
        "free_gap_m": float("inf"),
        "target_world_f32_m": np.asarray([1, 2, 3], dtype=np.float32),
        "skipped_intervals": [{"free_gap_m": float("inf"), "exit_distance_m": 0.1}],
    }
    record = owner.json_safe_selection_record(selection)
    assert record["free_gap_m"] is None
    assert record["free_gap_unbounded"] is True
    assert record["skipped_intervals"][0]["free_gap_m"] is None
    assert record["skipped_intervals"][0]["free_gap_unbounded"] is True
    assert "target_world_f32_m" not in record
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("dot", [0.05, 0.2, 0.999])
def test_depth_compatible_direction_keeps_source_when_already_feasible(dot):
    n = np.asarray([0.0, 0.0, 1.0])
    w = np.asarray([np.sqrt(1.0-dot*dot), 0.0, dot])
    d, diag = owner.depth_compatible_unit_direction(w, n, minimum_depth_cosine=0.05)
    np.testing.assert_array_equal(d, w)
    assert diag["adjusted"] is False
    assert np.dot(n, d) >= 0.05
    assert np.dot(w, d) > 0.0


def test_depth_compatible_direction_projects_to_nearest_cap_boundary():
    n = np.asarray([0.0, 0.0, 1.0])
    w = np.asarray([0.1, 0.0, -np.sqrt(0.99)])
    d, diag = owner.depth_compatible_unit_direction(w, n, minimum_depth_cosine=0.05)
    assert diag["adjusted"] is True
    assert np.dot(n, d) >= 0.05
    assert np.dot(w, d) > 0.0
    np.testing.assert_allclose(d, [np.sqrt(1.0-0.05**2), 0.0, 0.05], rtol=0, atol=2e-14)
    # The solution stays in the positive source-outward hemisphere and is the
    # closest point on the feasible cap boundary, not an arbitrary tangent.
    assert diag["selected_source_hemisphere_cosine"] == pytest.approx(0.1*np.sqrt(1.0-0.05**2)-np.sqrt(0.99)*0.05)


def test_depth_compatible_direction_rejects_opposite_and_degenerate_cases():
    n = np.asarray([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="infeasible or degenerate"):
        owner.depth_compatible_unit_direction(-n, n)
    with pytest.raises(ValueError, match="degenerate"):
        owner.depth_compatible_unit_direction(np.zeros(3), n)
    with pytest.raises(ValueError, match="finite"):
        owner.depth_compatible_unit_direction([np.nan, 0, 1], n)
    with pytest.raises(ValueError, match="unit length"):
        owner.depth_compatible_unit_direction([0, 0, 2], n)


def test_depth_compatible_direction_validates_threshold():
    with pytest.raises(ValueError, match=r"in \(0,1\)"):
        owner.depth_compatible_unit_direction([1, 0, 0], [1, 0, 0], minimum_depth_cosine=0)


def test_depth_rows_loader_rejects_multirow_nodes_and_pinned_mismatch(tmp_path):
    # The current single-row-per-node invariant is enforced by the construction
    # loop, rather than selecting one of duplicate typed contacts arbitrarily.
    rows = np.asarray([4, 4], dtype=np.int64)
    normals = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    loaded = {4: [
        {"normal_world_unit_covector": normals[0], "typed_id": [4, 1, 2, 3], "source_row": 0},
        {"normal_world_unit_covector": normals[1], "typed_id": [4, 1, 2, 4], "source_row": 1},
    ]}
    assert len(loaded[4]) == len(rows)
    with pytest.raises(ValueError, match="one inside depth row"):
        owner.single_depth_row_for_node(loaded[4], 4)
    assert owner.single_depth_row_for_node([loaded[4][0]], 4) is loaded[4][0]
