import importlib.util
from pathlib import Path
import numpy as np

MODULE_PATH = Path(__file__).with_name("edge_only_refinement585.py")
spec = importlib.util.spec_from_file_location("edge_only_refinement585_test", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def tetra():
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]], dtype=np.int32)
    return points, faces


def child_counts(result, nfaces):
    return np.bincount(result["parent_face_rows"], minlength=nfaces)


def test_one_split_edge_child_on_adjacent_face_and_shared_midpoint():
    points, faces = tetra()
    before = points.copy()
    result = mod.edge_only_refinement585(points, faces, np.asarray([0], dtype=np.int64))
    counts = child_counts(result, len(faces))
    assert counts[1] == counts[2] == counts[3] == 2
    assert np.array_equal(points.view(np.uint8), before.view(np.uint8))
    # The midpoint of the common (0,1) edge is one new row, referenced by both faces.
    edge_i = np.flatnonzero(np.all(result["selected_edges"] == np.asarray([0, 1]), axis=1))
    assert len(edge_i) == 1
    midpoint_row = int(result["edge_midpoint_rows"][edge_i[0]])
    assert sum(midpoint_row in set(map(int, face)) for face in result["source_O_faces"]) >= 3
    assert result["face_centroid_rows"].size == 0


def test_two_split_edges_yield_three_children():
    points, faces = tetra()
    result = mod.edge_only_refinement585(points, faces, np.asarray([0, 1], dtype=np.int64))
    counts = child_counts(result, len(faces))
    assert counts[2] == 3
    assert counts[0] == 4 and counts[1] == 4
    assert int(np.sum(counts)) == len(result["source_O_faces"])


def test_three_split_edges_yield_four_children_and_preserve_domain():
    points, faces = tetra()
    result = mod.edge_only_refinement585(points, faces, np.asarray([0], dtype=np.int64))
    counts = child_counts(result, len(faces))
    assert counts[0] == 4
    assert len(result["points_world_m"]) == len(points) + len(result["selected_edges"])
    assert len(result["source_O_faces"]) == len(faces) + sum(counts - 1)
    assert result["source_prolongation_indptr"].shape == (len(result["points_world_m"]) + 1,)
    assert result["refinement_node_parent_face_indptr"].shape == (len(result["points_world_m"]) + 1,)


def test_child_immediate_barycentrics_and_q_reconstruct_every_emitted_row():
    points, faces = tetra()
    result = mod.edge_only_refinement585(points, faces, np.asarray([0, 1], dtype=np.int64))
    mod._validate_result585(points, faces, result)
    for child, parent in enumerate(result["parent_face_rows"]):
        tri = faces[int(parent)]
        reconstructed = result["parent_face_barycentric"][child] @ points[tri].astype(np.float64)
        assert np.allclose(reconstructed, result["points_world_m"][result["source_O_faces"][child]], atol=1e-7, rtol=0)
    assert np.array_equal(result["points_world_m"][:len(points)].view(np.uint8), points.view(np.uint8))


def test_shared_edge_closure_and_boundary_are_conforming():
    points, faces = tetra()
    result = mod.edge_only_refinement585(points, faces, np.asarray([0, 1], dtype=np.int64))
    # Every selected edge has one midpoint row; two incident faces reuse that same row.
    edge_to_mid = {tuple(map(int, e)): int(r) for e, r in zip(result["selected_edges"], result["edge_midpoint_rows"])}
    edge = (1, 2)
    mid = edge_to_mid[edge]
    incident_parent_rows = np.flatnonzero(np.any(faces == 1, axis=1) & np.any(faces == 2, axis=1))
    assert len(incident_parent_rows) == 2
    child_rows = np.flatnonzero(np.isin(result["parent_face_rows"], incident_parent_rows))
    for parent in incident_parent_rows:
        rows = np.flatnonzero(result["parent_face_rows"] == parent)
        assert any(mid in result["source_O_faces"][row] for row in rows)
    assert result["boundary_edges_before"] == result["boundary_edges_after"] == 0
    assert result["boundary_components_before"] == result["boundary_components_after"] == 0


def test_bad_nonmanifold_or_inconsistently_oriented_input_rejected():
    points, faces = tetra()
    bad = np.vstack((faces, faces[0]))
    try:
        mod.edge_only_refinement585(points, bad, np.asarray([0]))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate/nonmanifold input must fail closed")
    flipped = faces.copy(); flipped[1] = flipped[1, ::-1]
    try:
        mod.edge_only_refinement585(points, flipped, np.asarray([0]))
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent winding must fail closed")


def test_full_546_build_keeps_anchors_qp_and_four_face_ancestry_families():
    points, faces = tetra()
    points = np.vstack((points, np.asarray([[2, 2, 2]], dtype=np.float32)))
    n, m = len(points), len(faces)
    identity_ptr = np.arange(n + 1, dtype=np.int64)
    identity_rows = np.arange(n, dtype=np.int64)
    identity_values = np.ones(n, dtype=np.float64)
    parent_face_rows = np.arange(m, dtype=np.int64)
    face_ptr = np.arange(m + 1, dtype=np.int64)
    face_rows = np.arange(m, dtype=np.int64)
    source = {
        "points_world_m": points,
        "source_O_faces": faces,
        "parent_source_face_marker": np.arange(10, 10 + m, dtype=np.int32),
        "node_kind": np.arange(n, dtype=np.int16),
        "node_source_row": np.arange(n, dtype=np.int32),
        "node_source_virtual_row": np.full(n, -1, dtype=np.int32),
        "node_parent_coarse_face_row": np.zeros(n, dtype=np.int32),
        "node_parent_coarse_edge_a": np.full(n, -1, dtype=np.int32),
        "node_parent_coarse_edge_b": np.full(n, -1, dtype=np.int32),
        "parent_coarse_face_row": np.arange(m, dtype=np.int32),
        "source_prolongation_indptr": identity_ptr,
        "source_prolongation_node_rows": identity_rows,
        "source_prolongation_weights": identity_values,
        "source_prolongation_semantics": np.asarray(["input identity Q"], dtype="U32"),
        "prior_input_006__refinement_node_kind": np.zeros(n, dtype=np.int8),
        "prior_input_parent__source_prolongation_indptr": identity_ptr.copy(),
        "prior_input_parent__source_prolongation_node_rows": identity_rows.copy(),
        "prior_input_parent__source_prolongation_weights": identity_values.copy(),
        "prior_input_parent__source_prolongation_semantics": np.asarray(["parent P"], dtype="U32"),
        "refinement_face_parent_face_row": parent_face_rows,
        "refinement_face_parent_barycentric": np.broadcast_to(np.eye(3), (m, 3, 3)).copy(),
        "refinement_face_parent_ancestry_indptr": face_ptr,
        "refinement_face_parent_ancestry_face_rows": face_rows,
        "refinement_face_parent_ancestry_semantics": np.asarray(["singleton immediate parent"], dtype="U64"),
        "face_source_ancestry_indptr": face_ptr.copy(),
        "face_source_ancestry_face_rows": face_rows.copy(),
        "face_fine_subface_ancestry_indptr": face_ptr.copy(),
        "face_fine_subface_ancestry_rows": face_rows.copy(),
        "face_fine_detail_ancestry_indptr": face_ptr.copy(),
        "face_fine_detail_ancestry_rows": face_rows.copy(),
        "coarse_boundary_anchor_rows": np.asarray([0], dtype=np.int32),
        "coarse_boundary_anchor_source_nodes": np.asarray([0], dtype=np.int32),
        "typed_coarse_anchor_output_rows_018": np.asarray([0], dtype=np.int32),
        "typed_coarse_anchor_source_ids_018": np.asarray([7], dtype=np.int32),
    }
    # A pre-existing refinement prefix is retained as history in the 546 adapter.
    source["prior_input_parent__refinement_face_parent_face_row"] = parent_face_rows.copy()
    source["prior_input_parent__refinement_face_parent_barycentric"] = source["refinement_face_parent_barycentric"].copy()
    source["area_reference_037_points_world_m_f64"] = points.astype(np.float64)
    source["area_reference_037_source_O_faces"] = faces.copy()
    candidate_sha = "a" * 64
    crossing_sha = "b" * 64
    reference_sha = "c" * 64
    crossing = {
        "complete": True,
        "geometry_or_native_admission": False,
        "candidate_sha256": candidate_sha,
        "banks": [{"result": {"count": 1, "triangle_pairs": [[0, 0]]}}],
    }
    refined, report = mod.build_refined_arrays585(
        source, crossing, points.astype(np.float64), faces,
        candidate_sha256=candidate_sha,
        crossing_report_sha256=crossing_sha,
        area_reference_sha256=reference_sha,
    )
    assert report["face_centroid_count"] == 0
    assert report["geometry_admitted"] is False and report["physical_steps"] == 0
    assert np.array_equal(refined["points_world_m"][:n].view(np.uint8), points.view(np.uint8))
    for key in ("coarse_boundary_anchor_rows", "coarse_boundary_anchor_source_nodes",
                "typed_coarse_anchor_output_rows_018", "typed_coarse_anchor_source_ids_018"):
        assert np.array_equal(refined[key], source[key])
    for family in ("face_source_ancestry", "face_fine_subface_ancestry", "face_fine_detail_ancestry"):
        ptr = refined[family + "_indptr"]
        values_key = family + ("_face_rows" if family == "face_source_ancestry" else "_rows")
        vals = refined[values_key]
        assert len(ptr) == len(refined["source_O_faces"]) + 1
        assert int(ptr[-1]) == len(vals)
        assert np.all(np.diff(ptr) >= 1)
    assert np.all(np.diff(refined["refinement_face_parent_ancestry_indptr"]) == 1)
    assert np.array_equal(refined["refinement_face_parent_ancestry_face_rows"], refined["refinement_face_parent_face_row"])
    for suffix in ("indptr", "node_rows", "weights"):
        assert np.array_equal(refined["source_prolongation_" + suffix],
                              refined["immediate_refinement_prolongation_" + suffix])
        assert np.array_equal(refined["prior_input_parent_546__source_prolongation_" + suffix],
                              source["prior_input_parent__source_prolongation_" + suffix])
    qptr = refined["immediate_refinement_prolongation_indptr"]
    qcols = refined["immediate_refinement_prolongation_node_rows"]
    qweights = refined["immediate_refinement_prolongation_weights"]
    area = refined["area_reference_037_points_world_m_f64"]
    for row in range(len(area)):
        lo, hi = int(qptr[row]), int(qptr[row + 1])
        replay = np.sum(points[qcols[lo:hi]].astype(np.float64) * qweights[lo:hi, None], axis=0)
        assert np.array_equal(replay, area[row])
    assert "edge midpoints only" in refined["source_prolongation_semantics"][0]
    assert np.array_equal(refined["area_reference_037_source_O_faces"], refined["source_O_faces"])
    assert np.array_equal(refined["area_reference_037_source_O_faces"], refined["source_O_faces"])


def test_bounded_edge_selection_uses_crossing_face_pair_weight_not_edge_claim():
    from prepare_edge_only_refinement585 import choose_high_incidence_edges585
    points, faces = tetra()
    report = {
        "complete": True,
        "geometry_or_native_admission": False,
        "banks": [
            {"result": {"count": 2, "triangle_pairs": [[0, 4], [0, 7]]}},
            {"result": {"count": 1, "triangle_pairs": [[1, 9]]}},
        ],
    }
    selected, receipt = choose_high_incidence_edges585(faces, report, max_edges=1)
    assert selected.shape == (1, 2)
    assert receipt["crossing_pair_count"] == 3
    assert receipt["selected_edge_count"] == 1
    assert "not a claim" in receipt["ranking_basis"]
    result = mod.edge_only_refinement585(points, faces, np.asarray([0, 1]),
                                        selected_edges_override=selected)
    assert len(result["selected_edges"]) == 1
    assert len(result["points_world_m"]) == len(points) + 1
    assert result["face_centroid_rows"].size == 0
