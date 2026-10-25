import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name("refine_conforming_006.py")
spec = importlib.util.spec_from_file_location("refine_conforming_006_tested", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def square_fixture():
    points = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    markers = np.asarray([71, 83], dtype=np.int32)
    result = mod.p084_affine_refinement(points, faces, np.asarray([0], dtype=np.int64), markers)
    return points, faces, markers, result


def test_crossing_records_deduplicate_faces_but_preserve_pair_count():
    report = {
        "complete": True,
        "geometry_or_native_admission": False,
        "banks": [
            {"result": {"count": 3, "triangle_pairs": [[4, 9], [4, 10], [2, 7]]}},
            {"result": {"count": 2, "triangle_pairs": [[4, 12], [1, 5]]}},
        ],
    }
    rows, pair_count = mod.crossing_face_rows(report, 5)
    assert rows.tolist() == [1, 2, 4]
    assert pair_count == 5


def test_global_shared_edge_split_and_boundary_loops_are_conforming():
    old_points, old_faces, markers, result = square_fixture()
    assert result["points_world_m"].shape == (9, 3)
    assert result["source_O_faces"].shape == (10, 3)
    assert result["selected_edges"].tolist() == [[0, 1], [0, 2], [1, 2]]
    shared_midpoint = int(result["edge_midpoint_rows"][1])
    assert shared_midpoint == 5
    uses = np.count_nonzero(np.any(result["source_O_faces"] == shared_midpoint, axis=1))
    assert uses >= 4
    assert result["boundary_components_before"] == result["boundary_components_after"] == 1
    assert result["boundary_edges_before"] == 4
    assert result["boundary_edges_after"] == 6
    assert np.array_equal(result["points_world_m"][: len(old_points)].view(np.uint32), old_points.view(np.uint32))
    assert np.array_equal(result["p084_parent_face_markers"], markers[result["parent_face_rows"]])
    assert np.max(result["relative_parent_area_error"]) < 1.0e-6


def test_affine_barycentric_parent_and_prolongation_reproduce_points():
    old_points, old_faces, markers, result = square_fixture()
    points = result["points_world_m"]
    for node in range(len(points)):
        lo, hi = result["source_prolongation_indptr"][node : node + 2]
        cols = result["source_prolongation_node_rows"][lo:hi]
        weights = result["source_prolongation_weights"][lo:hi]
        assert abs(float(weights.sum()) - 1.0) < 1.0e-14
        reconstructed = (old_points[cols].astype(np.float64) * weights[:, None]).sum(axis=0).astype(np.float32)
        assert np.array_equal(reconstructed.view(np.uint32), points[node].view(np.uint32))
    parents = result["parent_face_rows"]
    bary = result["parent_face_barycentric"]
    for row, (parent, child_bary) in enumerate(zip(parents, bary)):
        reconstructed = (old_points[old_faces[parent]].astype(np.float64)[None, :, :] * child_bary[:, :, None]).sum(axis=1).astype(np.float32)
        assert np.array_equal(reconstructed.view(np.uint32), points[result["source_O_faces"][row]].view(np.uint32))


def test_prior_refinement_csr_levels_are_preserved_without_prefix_collision():
    points, faces, markers, result = square_fixture()
    source = {
        "points_world_m": points,
        "source_O_faces": faces,
        "parent_source_face_marker": markers,
        "node_kind": np.zeros(4, dtype=np.int8),
        "node_source_row": np.arange(4, dtype=np.int32),
        "node_source_virtual_row": np.full(4, -1, dtype=np.int32),
        "node_parent_coarse_face_row": np.full(4, -1, dtype=np.int32),
        "node_parent_coarse_edge_a": np.full(4, -1, dtype=np.int32),
        "node_parent_coarse_edge_b": np.full(4, -1, dtype=np.int32),
        "node_parent_coarse_edge_parameter": np.full(4, -1.0, dtype=np.float64),
        "refinement_node_kind": np.asarray([0, 0, 1, 2], dtype=np.int8),
        "refinement_node_parent_edge_rows": np.full((4, 2), -1, dtype=np.int32),
        "refinement_node_parent_edge_weights": np.full((4, 2), -1.0, dtype=np.float64),
        "refinement_node_parent_face_indptr": np.zeros(5, dtype=np.int64),
        "refinement_node_parent_face_rows": np.empty(0, dtype=np.int32),
        "refinement_node_parent_face_barycentric": np.empty((0, 3), dtype=np.float64),
        "source_prolongation_indptr": np.arange(5, dtype=np.int64),
        "source_prolongation_node_rows": np.arange(4, dtype=np.int32),
        "source_prolongation_weights": np.ones(4, dtype=np.float64),
        "refinement_face_parent_face_row": np.asarray([10, 11], dtype=np.int32),
        "refinement_face_parent_barycentric": np.tile(np.eye(3)[None, :, :], (2, 1, 1)),
        "prior_run028__source_prolongation_indptr": np.asarray([0, 1, 2], dtype=np.int64),
        "prior_run028__source_prolongation_node_rows": np.asarray([0, 1], dtype=np.int32),
    }
    out = mod.extend_candidate_arrays(source, result, "candidate-pin", "report-pin")
    assert np.array_equal(out["prior_run028__source_prolongation_indptr"], source["prior_run028__source_prolongation_indptr"])
    assert np.array_equal(out["prior_run028__source_prolongation_node_rows"], source["prior_run028__source_prolongation_node_rows"])
    assert np.array_equal(out["prior_input_006__source_prolongation_indptr"], source["source_prolongation_indptr"])
    assert np.array_equal(out["prior_input_006__refinement_node_kind"], source["refinement_node_kind"])
    assert len(out["source_prolongation_indptr"]) == len(out["points_world_m"]) + 1
    assert len(out["prior_input_006__source_prolongation_indptr"]) == len(points) + 1
