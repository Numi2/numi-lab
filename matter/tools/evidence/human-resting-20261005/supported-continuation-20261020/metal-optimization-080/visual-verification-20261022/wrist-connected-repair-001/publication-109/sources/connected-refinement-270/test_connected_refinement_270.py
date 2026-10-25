from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('connected_refinement_270_tested', HERE / 'connected_refinement_270.py')
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _csr_fields(matrix, prefix):
    matrix = matrix.tocsr()
    return {
        prefix + 'indptr': matrix.indptr.astype(np.int64),
        prefix + 'node_rows': matrix.indices.astype(np.int64),
        prefix + 'weights': matrix.data.astype(np.float64),
    }


def _fixture():
    X = np.asarray([[0., 0., 0.], [2., 0., 0.], [2., 1., 0.], [0., 1., 0.]], dtype=np.float32)
    F = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    # Current immediate parent P: four current nodes -> three source columns.
    P = csr_matrix(np.asarray([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., .25, .75]]))
    # Two prior maps must survive exactly and remain independent from P/Q.
    P006 = csr_matrix((np.asarray([1., 1., .4, .6]), np.asarray([0, 1, 0, 1]),
                       np.asarray([0, 1, 2, 4])), shape=(3, 2))
    P028 = csr_matrix((np.asarray([1., 1.]), np.asarray([0, 0]),
                       np.asarray([0, 1, 2])), shape=(2, 1))
    source = {
        'points_world_m': X,
        'source_O_faces': F,
        'parent_source_face_marker': np.asarray([101, 202], dtype=np.int32),
        'parent_coarse_face_row': np.asarray([700, 701], dtype=np.int32),
        'node_kind': np.asarray([0, 1, 2, 3], dtype=np.int8),
        'node_source_row': np.asarray([20, 21, 22, 23], dtype=np.int64),
        'node_source_virtual_row': np.asarray([-1, -1, -1, -1], dtype=np.int32),
        'node_parent_coarse_face_row': np.asarray([4, 4, 4, 5], dtype=np.int32),
        'node_parent_coarse_edge_a': np.asarray([-1, 1, -1, -1], dtype=np.int32),
        'node_parent_coarse_edge_b': np.asarray([-1, 2, -1, -1], dtype=np.int32),
        'node_parent_coarse_edge_parameter': np.asarray([np.nan, .5, np.nan, np.nan], dtype=np.float64),
        'face_kind': np.asarray([1, 2], dtype=np.int8),
        'face_source_ancestry_indptr': np.asarray([0, 1, 3], dtype=np.int64),
        'face_source_ancestry_face_rows': np.asarray([55, 66, 77], dtype=np.int32),
        'face_fine_subface_ancestry_indptr': np.asarray([0, 1, 2], dtype=np.int64),
        'face_fine_subface_ancestry_rows': np.asarray([501, 502], dtype=np.int32),
        'coarse_boundary_anchor_rows': np.asarray([0, 3], dtype=np.int32),
        'coarse_boundary_anchor_source_nodes': np.asarray([8, 9], dtype=np.int32),
        'typed_coarse_anchor_output_rows_018': np.asarray([0, 3], dtype=np.int32),
        'typed_coarse_anchor_source_ids_018': np.asarray([88, 99], dtype=np.int64),
        'refinement_node_kind': np.asarray([0, 0, 0, 1], dtype=np.int8),
        'refinement_node_parent_edge_rows': np.asarray([[-1, -1], [-1, -1], [-1, -1], [0, 2]], dtype=np.int64),
        'refinement_node_parent_edge_weights': np.asarray([[-1., -1.], [-1., -1.], [-1., -1.], [.5, .5]]),
        'refinement_node_parent_face_indptr': np.asarray([0, 0, 0, 0, 0], dtype=np.int64),
        'refinement_node_parent_face_rows': np.asarray([], dtype=np.int32),
        'refinement_node_parent_face_barycentric': np.empty((0, 3), dtype=np.float64),
        'refinement_face_parent_face_row': np.asarray([0, 1], dtype=np.int32),
        'refinement_face_parent_barycentric': np.tile(np.eye(3), (2, 1, 1)),
        'refinement_input_candidate_sha256': np.asarray(['parent-candidate']),
        'source_prolongation_semantics': np.asarray(['P: 4 current rows to 3 source columns']),
        'prior_input_006__refinement_node_kind': np.asarray([4, 5, 6], dtype=np.int8),
        'prior_input_006__source_prolongation_semantics': np.asarray(['historical 006 -> 10145 domain']),
        'prior_run028__refinement_node_kind': np.asarray([7, 8], dtype=np.int8),
        'prior_run028__source_prolongation_semantics': np.asarray(['historical run028 -> 8776 domain']),
    }
    source.update(_csr_fields(P, 'source_prolongation_'))
    source.update(_csr_fields(P006, 'prior_input_006__source_prolongation_'))
    source.update(_csr_fields(P028, 'prior_run028__source_prolongation_'))
    area = np.asarray([[10., 1., .2], [12., 2., .4], [14., 3.2, .9], [11., 5., .8]], dtype=np.float64)
    crossing = {
        'candidate_sha256': 'c' * 64,
        'complete': True,
        'geometry_or_native_admission': False,
        'banks': [{'result': {'count': 2, 'triangle_pairs': [[0, 90], [0, 91]]}}],
    }
    return source, crossing, area


def _assert_rows_sum_one(ptr, weights):
    sums = np.asarray([weights[ptr[i]:ptr[i + 1]].sum() for i in range(len(ptr) - 1)])
    assert np.allclose(sums, 1.0, atol=1e-12, rtol=0)


def test_refinement_composes_maps_and_keeps_all_prior_chain_bytes():
    source, crossing, area = _fixture()
    source_sha = crossing['candidate_sha256']
    history = {k: v.copy() for k, v in source.items() if k.startswith('prior_')}
    output, report = module.build_refined_arrays(
        source, crossing, area, source['source_O_faces'],
        candidate_sha256=source_sha, crossing_report_sha256='d' * 64,
        area_reference_sha256='e' * 64,
    )
    nold = len(source['points_world_m'])
    nnew = len(output['points_world_m'])
    assert nnew > nold
    assert np.array_equal(output['points_world_m'][:nold].view(np.uint8), source['points_world_m'].view(np.uint8))
    for key, value in history.items():
        assert key in output and module._same_bytes(output[key], value)
    assert output['prior_input_parent__source_prolongation_indptr'].shape == source['source_prolongation_indptr'].shape
    for suffix in ('indptr', 'node_rows', 'weights', 'semantics'):
        assert module._same_bytes(output['prior_input_parent__source_prolongation_' + suffix], source['source_prolongation_' + suffix])
    assert len(output['refinement_node_kind']) == nnew
    assert len(output['refinement_face_parent_face_row']) == len(output['source_O_faces'])
    assert np.array_equal(output['parent_source_face_marker'], np.asarray([101] * 6 + [202] * 4, dtype=np.int32))
    # The two input faces share their selected diagonal; all incident children
    # use the same split rows and the original boundary remains one simple loop.
    Fnew = output['source_O_faces'].astype(np.int64)
    edge_counts = {}
    for tri in Fnew:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(a), int(b))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    assert max(edge_counts.values()) == 2
    boundary = [edge for edge, count in edge_counts.items() if count == 1]
    assert len(boundary) == 6
    def directed_boundary(faces):
        uses = {}
        for tri in faces:
            for a, b in ((int(tri[0]), int(tri[1])), (int(tri[1]), int(tri[2])), (int(tri[2]), int(tri[0]))):
                uses.setdefault(tuple(sorted((a, b))), []).append((a, b))
        assert all(len(v) <= 2 for v in uses.values())
        return {v[0] for v in uses.values() if len(v) == 1}
    old_directed = directed_boundary(source['source_O_faces'])
    new_directed = directed_boundary(Fnew)
    midpoint_by_edge = {tuple(sorted(map(int, edge))): row
                        for row, kind, edge in zip(range(nold, nnew), output['refinement_node_kind'][nold:],
                                                   output['refinement_node_parent_edge_rows'][nold:]) if int(kind) == 1}
    expected_directed = set()
    for a, b in old_directed:
        mid = midpoint_by_edge.get(tuple(sorted((a, b))))
        expected_directed.update(((a, mid), (mid, b)) if mid is not None else ((a, b),))
    assert expected_directed == new_directed
    assert report['boundary_components_before'] == report['boundary_components_after'] == 1
    assert report['boundary_edges_before'] == 4 and report['boundary_edges_after'] == 6
    assert report['max_parent_area_relative_drift_037_reference'] < 1e-12
    # Verify source-column map equals Q @ P and the fixed 037 area points use Q.
    Q = module._csr_from_arrays(output, 'immediate_refinement_prolongation_', nnew, nold)
    P = module._csr_from_arrays(source, 'source_prolongation_', nold, 3)
    C = module._csr_from_arrays(output, 'source_prolongation_', nnew, 3)
    assert (Q @ P - C).nnz == 0
    _assert_rows_sum_one(output['source_prolongation_indptr'], output['source_prolongation_weights'])
    _assert_rows_sum_one(output['immediate_refinement_prolongation_indptr'], output['immediate_refinement_prolongation_weights'])
    assert np.allclose(output['area_reference_037_points_world_m_f64'], Q @ area, atol=1e-14, rtol=0)
    assert np.array_equal(output['area_reference_037_source_O_faces'], output['source_O_faces'])
    assert not np.allclose(output['area_reference_037_points_world_m_f64'][-1], output['points_world_m'][-1])
    # Typed face ancestry follows the actual parent map, including multi-row CSR.
    parent = output['refinement_face_parent_face_row']
    ptr = output['face_source_ancestry_indptr']; vals = output['face_source_ancestry_face_rows']
    for child, p in enumerate(parent):
        expected = [55] if int(p) == 0 else [66, 77]
        assert vals[ptr[child]:ptr[child + 1]].tolist() == expected


def test_face_alignment_mismatch_is_rejected():
    source, crossing, area = _fixture()
    bad_faces = source['source_O_faces'][::-1].copy()
    try:
        module.build_refined_arrays(source, crossing, area, bad_faces,
                                    candidate_sha256='c' * 64,
                                    crossing_report_sha256='d' * 64,
                                    area_reference_sha256='e' * 64)
    except ValueError as exc:
        assert 'faces must exactly match' in str(exc)
    else:
        raise AssertionError('mismatched area-reference faces were accepted')


def test_one_two_three_selected_edge_patterns_remain_conforming():
    owner = module._load_pinned(module.P084_PATH, module.P084_SHA256, '_p084_pattern_fixture')
    X = np.asarray([[0., 0., 0.], [1., 0., 0.], [1.5, 1., 0.], [0.5, 1.5, 0.], [-0.5, 1., 0.]], dtype=np.float32)
    # A coherently wound three-triangle fan. Crossing faces 0 and 2 selects all
    # three edges on each; the middle face receives exactly two shared splits.
    F = np.asarray([[0, 1, 2], [0, 2, 3], [0, 3, 4]], dtype=np.int64)
    result = owner.subdivide_crossed_faces_affine(X, F, np.asarray([0, 2], dtype=np.int64),
                                                   parent_face_markers=np.asarray([11, 22, 33], dtype=np.int32))
    counts = np.bincount(result.child_parent_face_rows, minlength=3)
    assert counts.tolist() == [6, 5, 6]
    assert result.boundary_components_before == result.boundary_components_after == 1
    assert result.boundary_edges_before == 5 and result.boundary_edges_after == 9
    assert np.array_equal(result.points[:len(X)].view(np.uint8), X.view(np.uint8))
    # Shared interior edges occur in two children with opposite directions.
    directed = {}
    for face in result.faces:
        for a, b in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
            directed.setdefault(tuple(sorted((a, b))), []).append((a, b))
    assert all(len(v) <= 2 for v in directed.values())
    assert all(v[0] == v[1][::-1] for v in directed.values() if len(v) == 2)
