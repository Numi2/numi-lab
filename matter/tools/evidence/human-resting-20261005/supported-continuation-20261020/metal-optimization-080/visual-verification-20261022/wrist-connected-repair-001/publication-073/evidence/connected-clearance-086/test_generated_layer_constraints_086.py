import numpy as np
import pytest

from generated_layer_constraints_086 import (
    GeneratedLayerConstraintError,
    build_generated_boundary_layer_inequalities,
)


def fixture(*, duplicate=False, uncovered=False):
    # One source triangle per layer, with exact matching O/I/D column IDs.
    O = np.array([[0, 1, 2]], dtype=np.int64)
    I = O + 3
    D = O + 6
    source = np.vstack([O, I, D])
    R = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [0, 0, -1], [1, 0, -1], [0, 1, -1],
        [0, 0, -2], [1, 0, -2], [0, 1, -2],
        [.2, .3, 0], [.2, .3, -1], [.2, .3, -2],
    ], dtype=np.float64)
    src = np.r_[np.arange(9), [-1, -1, -1]]
    faces = [[0, 1, 9], [3, 4, 10], [6, 7, 11]]
    markers = [10, 20, 30]
    marker_parent = {10: ("O", 0), 20: ("I", 0), 30: ("D", 0)}
    if duplicate:
        faces.append([0, 1, 9])
        markers.append(10)
    if uncovered:
        R = np.vstack([R, [.4, .2, -2.5]])
        src = np.r_[src, -1]
        faces.append([6, 7, 12])
        markers.append(40)  # cap marker without a layer parent
        marker_parent[40] = ("cap", 0)
    X = R.astype(np.float32)
    return R, X, src, source, np.asarray(faces), np.asarray(markers), marker_parent


def build(args):
    R, X, src, source, faces, markers, marker_parent = args
    return build_generated_boundary_layer_inequalities(
        R, X, src, {"O": source[0:1], "I": source[1:2], "D": source[2:3]},
        faces, markers, marker_parent, rows_per_layer=3)


def test_baseline_rows_are_feasible_and_leave_generated_nodes_free():
    C, b, report = build(fixture())
    X = fixture()[1]
    slack = C @ X.astype(np.float64).ravel() - b
    assert C.shape == (8, 36)  # O->I, I->O/D, D->I; two bounds each
    assert slack.min() >= -1e-10
    assert report["generated_boundary_nodes_covered"] == 3
    assert report["generated_interior_nodes_excluded"] == 0
    # Each generated point retains three independent coordinate columns.
    for node in (9, 10, 11):
        assert all(C[:, 3 * node + axis].nnz for axis in range(3))


def test_generated_outer_collapse_is_caught_when_original_layer_rows_are_unchanged():
    args = fixture()
    C, b, _ = build(args)
    R, X, *_ = args
    trial = X.copy()
    # Collapse only generated O row 9 onto the corresponding I barypoint.
    trial[9] = np.array([.2, .3, -1.0], dtype=np.float32)
    slack = C @ trial.astype(np.float64).ravel() - b
    assert slack.min() < -0.49
    assert np.array_equal(trial[:9], X[:9])


def test_duplicate_children_deduplicate_same_generated_relationship():
    C0, _, r0 = build(fixture())
    C1, _, r1 = build(fixture(duplicate=True))
    assert r0["relation_count_after_deduplication"] == 4
    assert r1["relation_count_after_deduplication"] == 4
    assert (C0 != C1).nnz == 0


def test_uncovered_generated_boundary_node_fails_closed():
    with pytest.raises(GeneratedLayerConstraintError, match="lack O/I/D marked-parent coverage"):
        build(fixture(uncovered=True))


def test_mismatched_inner_face_column_topology_fails_closed():
    R, X, src, source, faces, markers, marker_parent = fixture()
    source[1, 0] = 4  # breaks the exact O->I row/column correspondence
    with pytest.raises(GeneratedLayerConstraintError, match="I source faces do not preserve"):
        build((R, X, src, source, faces, markers, marker_parent))


def test_source_output_row_permutation_is_equivariant():
    args = fixture()
    C0, b0, _ = build(args)
    R, X, src, source, faces, markers, marker_parent = args
    n = len(R)
    perm = np.array([11, 5, 9, 1, 7, 2, 10, 4, 8, 0, 6, 3], dtype=np.int64)
    old_to_new = np.empty(n, dtype=np.int64)
    old_to_new[perm] = np.arange(n)
    permuted = (R[perm], X[perm], src[perm], source, old_to_new[faces], markers, marker_parent)
    C1, b1, _ = build(permuted)

    # Restore the permuted matrix columns to the original output-row order.
    reordered_columns = np.array([3 * old_to_new[row] + axis for row in range(n) for axis in range(3)])
    C1 = C1[:, reordered_columns].tocsr()
    assert C1.shape == C0.shape

    # Generated relation ordering follows output IDs, so compare canonical row multisets.
    def signatures(C, b):
        dense = C.toarray()
        return sorted(tuple(np.round(np.r_[row, target], 12)) for row, target in zip(dense, b))

    assert signatures(C0, b0) == signatures(C1, b1)
