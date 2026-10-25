from __future__ import annotations
import numpy as np
import pytest
from human_skin_fat_boundary_refinement import (BoundaryRefinementError, subdivide_oriented_surface_edges,
    subdivide_registered_column_edges)


def provenance(point, *, faces=(17,), path=(2048, 2500, 2690), ambiguity=0, cosine=.8):
    return {
        "source_kind": "detailed_skin_surface_path",
        "source_face_rows": list(faces),
        "source_path_vertex_ids": list(path),
        "edge_endpoint_source_vertex_ids": [path[0], path[-1]],
        "sample_point_world_m": np.asarray(point, dtype=np.float64).tolist(),
        "ambiguity_count": ambiguity,
        "minimum_normal_cosine": cosine,
    }


def oriented_tetra():
    P = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float64)
    # Consistent outward orientation.
    F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    return P, F

def refine(P, F, edges, provenance_by_edge, *, original_point_count):
    # Synthetic positive markers stand in for the exact 084 source O marker map.
    markers = np.arange(100001, 100001 + len(F), dtype=np.int32)
    return subdivide_oriented_surface_edges(
        P, F, edges, provenance_by_edge, original_point_count=original_point_count,
        parent_face_markers=markers)


def test_one_shared_edge_splits_both_faces_once_and_removes_t_junction():
    P = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.], [.5, .5, 0.]])
    F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    original = P.copy()
    result = refine(P, F, {(0, 2): 4}, {(0, 2): provenance(P[4])}, original_point_count=4)
    assert result.faces.shape == (4, 3)
    assert result.child_parent_face_rows.tolist() == [0, 0, 1, 1]
    assert result.child_parent_face_markers.tolist() == [100001, 100001, 100002, 100002]
    assert np.array_equal(result.points[:4], original[:4])
    assert (0, 2) not in {tuple(sorted(e)) for f in result.faces for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}
    counts = {}
    for f in result.faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            e = tuple(sorted((int(a), int(b))))
            counts[e] = counts.get(e, 0) + 1
    assert counts[(0, 4)] == counts[(2, 4)] == 2
    assert result.midpoint_source_face_rows == ((17,),)
    assert result.midpoint_source_path_vertex_ids == ((2048, 2500, 2690),)
    assert result.midpoint_source_endpoint_vertex_ids == ((2048, 2690),)


def test_two_selected_edges_in_one_tetra_face_are_conforming():
    P, F = oriented_tetra()
    # Face 0 contains (0,2) and (2,1), both shared with adjacent O faces.
    P = np.vstack([P, (P[0] + P[2]) / 2, (P[2] + P[1]) / 2])
    selected = {(0, 2): 4, (1, 2): 5}
    prov = {e: provenance(P[row], faces=(17 + i,), path=(2048+i, 2500+i, 2690+i)) for i, (e, row) in enumerate(selected.items())}
    result = refine(P, F, selected, prov, original_point_count=4)
    assert result.faces.shape == (8, 3)  # base face: 3; three adjacent faces split once: 2 each
    assert len(result.child_parent_face_rows) == len(result.faces)
    assert result.child_parent_face_rows.tolist().count(0) == 3
    assert np.array_equal(result.points[:4], oriented_tetra()[0])


def test_three_selected_edges_make_four_children_and_keep_orientation():
    P, F = oriented_tetra()
    edge_rows = {(0, 2): 4, (1, 2): 5, (0, 1): 6}
    P = np.vstack([P, [(P[a] + P[b]) / 2 for a, b in edge_rows]])
    prov = {e: provenance(P[row], faces=(30+i,), path=(100+i, 200+i, 300+i)) for i, (e, row) in enumerate(edge_rows.items())}
    result = refine(P, F, edge_rows, prov, original_point_count=4)
    assert result.child_parent_face_rows.tolist().count(0) == 4
    assert result.faces.shape == (10, 3)
    for f, parent in zip(result.faces, result.child_parent_face_rows):
        n0 = np.cross(P[F[parent, 1]] - P[F[parent, 0]], P[F[parent, 2]] - P[F[parent, 0]])
        n1 = np.cross(P[f[1]] - P[f[0]], P[f[2]] - P[f[0]])
        assert n0 @ n1 > 0


def test_all_unselected_faces_are_preserved_with_parent_identity():
    P, F = oriented_tetra()
    P = np.vstack([P, (P[0] + P[2]) / 2])
    result = refine(P, F, {(0, 2): 4}, {(0, 2): provenance(P[4])}, original_point_count=4)
    unchanged = [tuple(map(int, f)) for f, p in zip(result.faces, result.child_parent_face_rows) if p in (1, 3)]
    assert unchanged == [tuple(F[1]), tuple(F[3])]
    assert result.child_parent_face_rows.tolist().count(2) == 2


def test_rejects_boundary_or_nonmanifold_edge():
    P = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.], [.5,0.,0.]])
    with pytest.raises(BoundaryRefinementError, match="two-face"):
        refine(P, np.array([[0,1,2]], dtype=np.int32), {(0,1):4}, {(0,1):provenance(P[4])}, original_point_count=4)


def test_rejects_nonopposite_incident_face_orientation():
    P = np.array([[0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.], [.5,.5,0.]])
    F = np.array([[0,1,2], [0,3,2]], dtype=np.int32)  # shared edge 0-2 has same direction
    with pytest.raises(BoundaryRefinementError, match="opposite orientation"):
        refine(P, F, {(0,2):4}, {(0,2):provenance(P[4])}, original_point_count=4)


@pytest.mark.parametrize("changes, message", [
    ({"ambiguity_count": 1}, "path tie count"),
    ({"minimum_normal_cosine": -0.1}, "opposed"),
    ({"source_face_rows": []}, "empty"),
    ({"source_path_vertex_ids": [7]}, "simple sequence"),
    ({"source_kind": "linear_edge_midpoint"}, "detailed skin"),
    ({"edge_endpoint_source_vertex_ids": [999, 1000]}, "endpoints do not match"),
])
def test_rejects_unsupported_ambiguous_or_opposed_sample(changes, message):
    P = np.array([[0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.], [.5,.5,0.]])
    F = np.array([[0,1,2], [0,2,3]], dtype=np.int32)
    record = provenance(P[4]); record.update(changes)
    with pytest.raises(BoundaryRefinementError, match=message):
        refine(P, F, {(0,2):4}, {(0,2):record}, original_point_count=4)


def test_rejects_midpoint_provenance_coordinate_mismatch():
    P = np.array([[0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.], [.5,.5,0.]])
    F = np.array([[0,1,2], [0,2,3]], dtype=np.int32)
    record = provenance([.5,.49,0.])
    with pytest.raises(BoundaryRefinementError, match="differs"):
        refine(P, F, {(0,2):4}, {(0,2):record}, original_point_count=4)


def test_rejects_midpoint_row_reused_by_two_edges():
    P, F = oriented_tetra()
    P = np.vstack([P, (P[0] + P[2]) / 2])
    edges = {(0,2):4, (1,2):4}
    prov = {e: provenance(P[4]) for e in edges}
    with pytest.raises(BoundaryRefinementError, match="unique and appended"):
        refine(P, F, edges, prov, original_point_count=4)


def test_rejects_out_of_domain_edge():
    P, F = oriented_tetra()
    P = np.vstack([P, (P[0] + P[2]) / 2])
    with pytest.raises(BoundaryRefinementError, match="outside"):
        refine(P, F, {(1,99):4}, {(1,99):provenance(P[4])}, original_point_count=4)


def test_rejects_marker_domain_mismatch_and_unused_appended_points():
    P, F = oriented_tetra()
    P = np.vstack([P, (P[0] + P[2]) / 2, [0.25, 0.25, 0.25]])
    with pytest.raises(BoundaryRefinementError, match="appended point"):
        refine(P, F, {(0,2):4}, {(0,2):provenance(P[4])}, original_point_count=4)
    P = P[:5]
    markers = np.array([100001, 100001, 100003, 100004], dtype=np.int32)
    with pytest.raises(BoundaryRefinementError, match="unique"):
        subdivide_oriented_surface_edges(P, F, {(0,2):4}, {(0,2):provenance(P[4])},
            original_point_count=4, parent_face_markers=markers)



def _layered_fixture(*, remove_d_edge=False, fraction=.25, dtype=np.float64):
    base, tetra_faces = oriented_tetra()
    P = np.vstack([base, base + [0., 0., .1], base + [0., 0., .2]]).astype(dtype)
    edges = {
        "O": (0, 2),
        "I": (4, 6),
        "D": (8, 10),
    }
    if remove_d_edge:
        edges["D"] = (8, 11)
    columns = {
        100: {"O": 0, "I": 4, "D": 8},
        200: {"O": 2, "I": 6, "D": 10},
    }
    outer = np.array([.3, .3, 0.], dtype=np.float64)
    source = provenance(outer, faces=(17, 18), path=(2048, 2500, 2690))
    item = {
        "column_ids": (100, 200),
        "column_endpoint_detail_vertex_ids": (2048, 2690),
        "layer_edges": edges,
        "sample_fraction": fraction,
        "outer_source_by_frame": {"reference": source},
    }
    faces = {layer: tetra_faces + 4 * li for li, layer in enumerate(("O", "I", "D"))}
    markers = {layer: np.arange(100001 + 10 * li, 100001 + 10 * li + 4, dtype=np.int32)
               for li, layer in enumerate(("O", "I", "D"))}
    return P, faces, markers, columns, item


def test_registered_column_refinement_builds_explicit_OID_midpoints_and_parent_markers():
    P, faces, markers, columns, item = _layered_fixture()
    result = subdivide_registered_column_edges(
        {"reference": P}, faces, markers, columns, [item],
        reference_field_sha256="a" * 64, source_column_mapping_sha256="b" * 64)
    rows = {layer: int(result.midpoint_rows_by_layer[layer][0]) for layer in ("O", "I", "D")}
    assert rows == {"O": 12, "I": 13, "D": 14}
    pts = result.points_by_frame["reference"]
    np.testing.assert_array_equal(pts[rows["O"]], [.3, .3, 0.])
    np.testing.assert_array_equal(pts[rows["I"]], [.3, .3, .1])
    np.testing.assert_array_equal(pts[rows["D"]], [.3, .3, .2])
    assert all(result.faces_by_layer[layer].shape == (6, 3) for layer in ("O", "I", "D"))
    assert result.child_parent_face_markers_by_layer["O"].tolist() == [100001, 100001, 100002, 100003, 100003, 100004]
    assert result.source_column_ids_by_edge == ((100, 200),)
    assert result.outer_path_provenance_by_edge[0]["minimum_normal_cosine"] == .8
    assert result.midpoint_source_path_vertex_ids_by_edge == ((2048, 2500, 2690),)
    assert result.sample_fractions_from_path_start.tolist() == [.25]
    np.testing.assert_array_equal(result.source_column_weights_by_edge, [[.75, .25]])


def test_registered_column_refinement_rejects_missing_or_mismatched_layer_edge():
    P, faces, markers, columns, item = _layered_fixture(remove_d_edge=True)
    with pytest.raises(BoundaryRefinementError, match="does not connect"):
        subdivide_registered_column_edges(
            {"reference": P}, faces, markers, columns, [item],
            reference_field_sha256="a" * 64, source_column_mapping_sha256="b" * 64)


def test_registered_column_refinement_rejects_missing_layer_map_and_bad_frame_provenance():
    P, faces, markers, columns, item = _layered_fixture()
    item["layer_edges"] = {k: v for k, v in item["layer_edges"].items() if k != "D"}
    with pytest.raises(BoundaryRefinementError, match="explicit O/I/D edge"):
        subdivide_registered_column_edges(
            {"reference": P}, faces, markers, columns, [item],
            reference_field_sha256="a" * 64, source_column_mapping_sha256="b" * 64)
    P, faces, markers, columns, item = _layered_fixture()
    item["outer_source_by_frame"] = {"current": item["outer_source_by_frame"]["reference"]}
    with pytest.raises(BoundaryRefinementError, match="every frame"):
        subdivide_registered_column_edges(
            {"reference": P}, faces, markers, columns, [item],
            reference_field_sha256="a" * 64, source_column_mapping_sha256="b" * 64)


def test_registered_offset_provenance_rejects_tampered_field_or_sample():
    P, faces, markers, columns, item = _layered_fixture()
    result = subdivide_registered_column_edges(
        {"reference": P}, faces, markers, columns, [item],
        reference_field_sha256="a" * 64, source_column_mapping_sha256="b" * 64)
    row = int(result.midpoint_rows_by_layer["I"][0])
    points = result.points_by_frame["reference"].copy()
    points[row, 2] += .001
    edge = (4, 6)
    prov = result.offset_provenance_by_layer["I"][0]
    with pytest.raises(BoundaryRefinementError, match="differs from emitted"):
        subdivide_oriented_surface_edges(
            points, faces["I"], {edge: row}, {edge: prov},
            original_point_count=len(P), parent_face_markers=markers["I"],
            layer="I", expected_midpoint_rows=[row])



def test_source_path_direction_is_aligned_to_explicit_column_endpoint_ids():
    P, faces, markers, columns, item = _layered_fixture()
    item["outer_source_by_frame"]["reference"] = provenance(
        [.3, .3, 0.], faces=(17, 18), path=(2690, 2500, 2048))
    result = subdivide_registered_column_edges(
        {"reference": P}, faces, markers, columns, [item],
        reference_field_sha256="a" * 64, source_column_mapping_sha256="b" * 64)
    np.testing.assert_array_equal(result.source_column_weights_by_edge, [[.25, .75]])
    # The offset is constant at both endpoints in this fixture, so reversing path
    # direction changes only the explicit interpolation provenance weights.
    assert result.offset_provenance_by_layer["I"][0]["source_column_weights"] == [.25, .75]



def test_registered_source_path_allows_recorded_deterministic_tie_and_reversed_winding():
    P, faces, markers, columns, item = _layered_fixture()
    source = item["outer_source_by_frame"]["reference"]
    source["ambiguity_count"] = 2
    source["path_selection_rule"] = "deterministic stable detail source row tie-break"
    source["minimum_normal_cosine"] = -.7
    source["normal_orientation_status"] = "reversed"
    result = subdivide_registered_column_edges(
        {"reference": P}, faces, markers, columns, [item],
        reference_field_sha256="a" * 64, source_column_mapping_sha256="b" * 64)
    assert result.outer_path_provenance_by_edge[0]["ambiguity_count"] == 2
    assert result.outer_path_provenance_by_edge[0]["normal_orientation_status"] == "reversed"


def test_default_parent_normal_gate_stays_strict_and_diagnostic_source_mode_keeps_chain():
    P = np.array([[0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.], [2.,2.,0.]])
    F = np.array([[0,1,2], [0,2,3]], dtype=np.int32)
    edge = {(0,2):4}
    prov = {(0,2): provenance(P[4])}
    with pytest.raises(BoundaryRefinementError, match="orientation-reversed"):
        refine(P, F, edge, prov, original_point_count=4)
    source_ref = np.tile([0.,0.,1.], (len(F),1))
    result = subdivide_oriented_surface_edges(
        P, F, edge, prov, original_point_count=4,
        parent_face_markers=np.array([100001,100002], dtype=np.int32),
        expected_parent_normals=source_ref, collect_orientation_failures=True)
    assert result.orientation_diagnostics["failed_child_count"] > 0
    assert result.orientation_diagnostics["oriented_boundary_chain_preserved"] is True
    assert result.faces.shape == (4,3)


def test_source_normal_layer_provenance_preserves_signed_endpoint_depths():
    P = np.array([[0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.], [.5,.5,-.1]])
    F = np.array([[0,1,2], [0,2,3]], dtype=np.int32)
    sample = P[4].astype(np.float64)
    record = {
        "source_kind": "detailed_skin_source_normal_scalar", "layer": "I",
        "sample_point_world_m": sample.tolist(), "outer_midpoint_world_m": [.5,.5,0.],
        "outward_normal": [0.,0.,1.], "skin_depth_m": .1, "fat_depth_m": .2,
        "endpoint_depths_m": [[-.01,.02],[-.03,.04]], "source_face_rows": [17],
        "source_path_vertex_ids": [10,11], "reference_field_sha256": "a"*64,
        "source_column_ids": [100,200],
    }
    result = subdivide_oriented_surface_edges(
        P, F, {(0,2):4}, {(0,2):record}, original_point_count=4,
        parent_face_markers=np.array([100001,100002], dtype=np.int32),
        layer="I", expected_midpoint_rows=[4])
    assert result.faces.shape == (4,3)
    assert result.orientation_diagnostics["oriented_boundary_chain_preserved"] is True


@pytest.mark.parametrize("normals", [
    np.zeros((2,3)), np.ones((1,3)), np.full((2,3), np.nan),
])
def test_expected_source_normal_reference_is_shape_finite_and_nonzero(normals):
    P = np.array([[0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.], [.5,.5,0.]])
    F = np.array([[0,1,2], [0,2,3]], dtype=np.int32)
    with pytest.raises(BoundaryRefinementError, match="expected_parent_normals"):
        subdivide_oriented_surface_edges(
            P, F, {(0,2):4}, {(0,2):provenance(P[4])}, original_point_count=4,
            parent_face_markers=np.array([100001,100002], dtype=np.int32),
            expected_parent_normals=normals)


def test_affine_crossed_face_refinement_is_conforming_and_retains_multiface_ancestry():
    from human_skin_fat_boundary_refinement import subdivide_crossed_faces_affine
    P, F = oriented_tetra()
    markers = np.array([1001, 1002, 1003, 1004], dtype=np.int32)
    result = subdivide_crossed_faces_affine(P.astype(np.float32), F, [0], parent_face_markers=markers)
    assert result.points.shape == (11, 3)
    assert result.faces.shape == (18, 3)
    assert result.selected_edges.tolist() == [[0, 1], [0, 2], [1, 2]]
    assert result.affected_parent_face_rows.tolist() == [0, 1, 2, 3]
    assert result.refinement_node_kind.tolist() == [0]*4 + [1]*3 + [2]*4
    np.testing.assert_array_equal(result.child_parent_face_markers,
                                  markers[result.child_parent_face_rows])
    np.testing.assert_array_equal(result.points[:len(P)].view(np.uint32), P.astype(np.float32).view(np.uint32))
    assert result.boundary_components_before == result.boundary_components_after == 0
    assert result.boundary_edges_before == result.boundary_edges_after == 0
    assert np.all(result.relative_parent_area_error < 1e-6)
    assert np.allclose(result.child_parent_face_barycentric.sum(axis=2), 1.0)
    # Every edge-midpoint node has the two incident source-face ancestors.
    for row in result.edge_midpoint_rows:
        a, b = result.refinement_node_parent_face_indptr[row:row+2]
        assert b-a == 2
    # Every child edge is either shared by two triangles or belongs to no boundary (closed tetra).
    incidence = {}
    for a, b, c in result.faces:
        for u, v in ((a,b),(b,c),(c,a)):
            key = tuple(sorted((int(u), int(v))))
            incidence.setdefault(key, []).append((int(u), int(v)))
    assert all(len(uses) == 2 and uses[0] == uses[1][::-1] for uses in incidence.values())
    # The returned affine CSR reproduces every emitted point.
    ptr, cols, weights = (result.source_prolongation_indptr,
                          result.source_prolongation_node_rows,
                          result.source_prolongation_weights)
    reconstructed = np.vstack([
        np.sum(P.astype(np.float64)[cols[ptr[i]:ptr[i+1]]] * weights[ptr[i]:ptr[i+1], None], axis=0)
        for i in range(len(result.points))
    ]).astype(np.float32)
    np.testing.assert_array_equal(reconstructed.view(np.uint32), result.points.view(np.uint32))


def test_affine_refinement_bisects_boundary_chain_without_changing_loop_topology():
    from human_skin_fat_boundary_refinement import subdivide_crossed_faces_affine
    P = np.array([[0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.]], dtype=np.float64)
    F = np.array([[0,1,2], [0,2,3]], dtype=np.int32)
    result = subdivide_crossed_faces_affine(P, F, [0])
    assert result.boundary_components_before == result.boundary_components_after == 1
    assert result.boundary_edges_before == 4
    assert result.boundary_edges_after == 6
    assert result.faces.shape == (10, 3)
    assert result.points.shape == (4 + 3 + 2, 3)


def test_affine_refinement_rejects_nonmanifold_or_inconsistently_oriented_input():
    from human_skin_fat_boundary_refinement import BoundaryRefinementError, subdivide_crossed_faces_affine
    P = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,-1.,0.], [0.,0.,1.]])
    nonmanifold = np.array([[0,1,2], [1,0,3], [0,1,4]], dtype=np.int32)
    with pytest.raises(BoundaryRefinementError, match="nonmanifold edge"):
        subdivide_crossed_faces_affine(P, nonmanifold, [0])
    inconsistent = np.array([[0,1,2], [0,1,3]], dtype=np.int32)
    with pytest.raises(BoundaryRefinementError, match="inconsistent directed winding"):
        subdivide_crossed_faces_affine(P, inconsistent, [0])


def test_affine_refinement_rejects_bad_rows_and_float32_collapsed_midpoint():
    from human_skin_fat_boundary_refinement import BoundaryRefinementError, subdivide_crossed_faces_affine
    P = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]], dtype=np.float32)
    F = np.array([[0,1,2]], dtype=np.int32)
    with pytest.raises(BoundaryRefinementError, match="crossed_face_rows"):
        subdivide_crossed_faces_affine(P, F, np.array([0.5]))
    with pytest.raises(BoundaryRefinementError, match="outside"):
        subdivide_crossed_faces_affine(P, F, [1])
    tiny = np.array([[1.0,0.,0.], [1.0+1e-9,0.,0.], [1.0,1e-6,0.]], dtype=np.float32)
    with pytest.raises(BoundaryRefinementError, match="collapses"):
        subdivide_crossed_faces_affine(tiny, F, [0])


def test_registered_depth_prolongation_is_convex_affine_prefix_exact_and_hash_pinned():
    from human_skin_fat_boundary_refinement import prolong_registered_depth_fields, subdivide_crossed_faces_affine
    P, F = oriented_tetra()
    result = subdivide_crossed_faces_affine(P, F, [0])
    H = np.column_stack((0.5 + .2*P[:,0] + .1*P[:,1], 0.8 + .1*P[:,1] + .05*P[:,2]))
    out = prolong_registered_depth_fields(
        H, result.source_prolongation_indptr, result.source_prolongation_node_rows,
        result.source_prolongation_weights, expected_source_field_sha256="a"*64, source_field_sha256="a"*64)
    expected = np.column_stack((0.5 + .2*result.points[:,0] + .1*result.points[:,1],
                                0.8 + .1*result.points[:,1] + .05*result.points[:,2]))
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(out[:len(H)], H)
    assert np.all(out > 0)
    with pytest.raises(BoundaryRefinementError, match="does not match"):
        prolong_registered_depth_fields(H, result.source_prolongation_indptr,
            result.source_prolongation_node_rows, result.source_prolongation_weights,
            expected_source_field_sha256="a"*64, source_field_sha256="b"*64)


def test_registered_depth_prolongation_rejects_nonconvex_duplicate_empty_and_nonpositive_rows():
    from human_skin_fat_boundary_refinement import BoundaryRefinementError, prolong_registered_depth_fields
    H = np.array([[.1,.2], [.3,.4]], dtype=np.float64)
    with pytest.raises(BoundaryRefinementError, match="strictly positive"):
        prolong_registered_depth_fields(np.array([[0.,.2], [.3,.4]]), [0,1,2], [0,1], [1.,1.])
    with pytest.raises(BoundaryRefinementError, match="repeats"):
        prolong_registered_depth_fields(H, [0,2,3], [0,0,1], [.5,.5,1.])
    with pytest.raises(BoundaryRefinementError, match="convex partition"):
        prolong_registered_depth_fields(H, [0,1,2], [0,1], [.8,.8])
    with pytest.raises(BoundaryRefinementError, match="empty"):
        prolong_registered_depth_fields(H, [0,1,1], [0], [1.])
    with pytest.raises(BoundaryRefinementError, match="provided together"):
        prolong_registered_depth_fields(H, [0,1,2], [0,1], [1.,1.], expected_source_field_sha256="c"*64)


def test_affine_face_fans_cover_one_two_and_three_selected_edges():
    from human_skin_fat_boundary_refinement import subdivide_crossed_faces_affine
    P, F = oriented_tetra()
    one = subdivide_crossed_faces_affine(P, F, [0])
    # With two crossed faces, the two remaining tetrahedron faces each inherit
    # exactly two selected edges from their neighbours.
    two = subdivide_crossed_faces_affine(P, F, [0, 1])
    two_face_parent_counts = np.bincount(two.child_parent_face_rows, minlength=len(F))
    assert two_face_parent_counts.tolist() == [6, 6, 5, 5]
    # Crossing three tetra faces selects every edge, so each parent polygon has
    # six boundary vertices and receives a six-triangle centroid fan.
    three = subdivide_crossed_faces_affine(P, F, [0, 1, 2])
    three_face_parent_counts = np.bincount(three.child_parent_face_rows, minlength=len(F))
    assert three_face_parent_counts.tolist() == [6, 6, 6, 6]
    assert len(one.selected_edges) == 3
    assert len(two.selected_edges) == 5
    assert len(three.selected_edges) == 6


def test_affine_child_barycentrics_reconstruct_each_emitted_child_triangle():
    from human_skin_fat_boundary_refinement import subdivide_crossed_faces_affine
    P, F = oriented_tetra()
    result = subdivide_crossed_faces_affine(P.astype(np.float32), F, [0])
    for child_row, (child, parent, bary) in enumerate(zip(
            result.faces, result.child_parent_face_rows, result.child_parent_face_barycentric)):
        reconstructed = bary @ P.astype(np.float64)[F[int(parent)]]
        actual = result.points[child].astype(np.float64)
        np.testing.assert_allclose(reconstructed, actual, rtol=0, atol=2e-7,
                                   err_msg=f"child row {child_row} ancestry mismatch")
    for row in result.face_centroid_rows:
        start, stop = result.refinement_node_parent_face_indptr[row:row+2]
        assert stop-start == 1
        np.testing.assert_allclose(result.refinement_node_parent_face_barycentric[start], [1/3]*3)


def test_affine_refinement_preserves_repeated_and_sentinel_parent_markers():
    from human_skin_fat_boundary_refinement import subdivide_crossed_faces_affine
    P, F = oriented_tetra()
    markers = np.array([-1, 4, -1, 4], dtype=np.int32)
    result = subdivide_crossed_faces_affine(P, F, [0], parent_face_markers=markers)
    np.testing.assert_array_equal(result.child_parent_face_markers,
                                  markers[result.child_parent_face_rows])
    # The edge-weight table uses -1 only as its non-edge sentinel.
    assert np.all(result.refinement_node_parent_edge_weights[:len(P)] == -1)
    assert np.all(result.refinement_node_parent_edge_weights[result.face_centroid_rows] == -1)
    assert np.all(result.refinement_node_parent_edge_weights[result.edge_midpoint_rows] == .5)
