import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

SCRIPT = Path(__file__).with_name("propose_joint_clearance_witnesses_279.py")
spec = importlib.util.spec_from_file_location("joint279", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_empty_self_pairs_allowed_with_clearance_and_owner_shapes():
    pairs = mod.validate_self_pair_input(np.empty((0, 2), dtype=np.int64), 1, True)
    assert pairs.shape == (0, 2)
    pair_owner = mod.load_module("pair_owner_279_test", mod.DEFAULT_PAIR_OWNER)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    rows, records = pair_owner.build_pair_rows(points, faces, pairs, margin_fraction=.25)
    assert rows == [] and records == []
    area = sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, 9))
    A, b = mod._owner_matrix(rows, area, np.array([2.]), np.array([0, 1, 2]))
    assert A.shape == (1, 9)
    np.testing.assert_array_equal(b, [2.])


def test_empty_self_pairs_rejected_without_any_clearance_targets():
    with pytest.raises(ValueError, match="all empty"):
        mod.validate_self_pair_input(np.empty((0, 2), dtype=np.int64), 3, False)


@pytest.mark.parametrize("bad", [
    np.empty((0, 3), dtype=np.int64),
    np.empty((0,), dtype=np.int64),
    np.empty((0, 2), dtype=np.float64),
    np.array([[0, 1.5]]),
    np.array([[0, 0]], dtype=np.int64),
    np.array([[0, 3]], dtype=np.int64),
])
def test_malformed_or_invalid_self_pairs_rejected(bad):
    with pytest.raises(ValueError):
        mod.validate_self_pair_input(bad, 3, True)


def test_self_pairs_are_sorted_deduplicated_and_stable():
    pairs = mod.validate_self_pair_input(np.array([[2, 1], [1, 2], [0, 2]], dtype=np.int32), 3, False)
    np.testing.assert_array_equal(pairs, [[0, 2], [1, 2]])


@pytest.mark.parametrize("prefix,nodes", [(0, 6), (6, 6), (3, 6)])
def test_fixed_prefix_count_boundaries(prefix, nodes):
    assert mod.validate_fixed_prefix_nodes(prefix, nodes) == prefix


@pytest.mark.parametrize("value", [-1, 7, 1.0, True, np.nan])
def test_fixed_prefix_count_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        mod.validate_fixed_prefix_nodes(value, 6)


def test_prefix_bounds_are_exact_zero_only_for_supported_prefix_rows():
    support = np.array([0, 3, 5], dtype=np.int64)
    lower = -np.ones(9)
    upper = np.ones(9)
    lo, hi, rows = mod.apply_fixed_prefix_zero_bounds(lower, upper, support, 4, 6)
    np.testing.assert_array_equal(rows, [0, 3])
    np.testing.assert_array_equal(lo[:6], np.zeros(6))
    np.testing.assert_array_equal(hi[:6], np.zeros(6))
    np.testing.assert_array_equal(lo[6:], [-1., -1., -1.])
    np.testing.assert_array_equal(hi[6:], [1., 1., 1.])
    np.testing.assert_array_equal(lower, -np.ones(9))
    np.testing.assert_array_equal(upper, np.ones(9))


@pytest.mark.parametrize("support", [
    np.array([0, 0], dtype=np.int64),
    np.array([-1], dtype=np.int64),
    np.array([6], dtype=np.int64),
    np.array([1.0]),
    np.array([[1]], dtype=np.int64),
])
def test_prefix_bounds_reject_malformed_support(support):
    with pytest.raises(ValueError):
        mod.apply_fixed_prefix_zero_bounds(-np.ones(3), np.ones(3), support, 2, 6)


def test_prefix_bounds_reject_inconsistent_or_inverted_coordinate_bounds():
    with pytest.raises(ValueError, match="do not match"):
        mod.apply_fixed_prefix_zero_bounds(-np.ones(6), np.ones(6), np.array([0]), 1, 2)
    with pytest.raises(ValueError, match="lower bound exceeds"):
        mod.apply_fixed_prefix_zero_bounds(np.ones(3), np.zeros(3), np.array([0]), 1, 1)


def test_prefix_bitwise_assertion_detects_value_and_signed_zero_changes():
    base = np.array([[0., -0., 1.], [2., 3., 4.]], dtype=np.float32)
    same = base.copy()
    mod.assert_prefix_bitwise_unchanged(base, same, 1)
    changed = base.copy()
    changed[0, 1] = 0.0
    with pytest.raises(ValueError, match="changed"):
        mod.assert_prefix_bitwise_unchanged(base, changed, 1)
    with pytest.raises(ValueError, match="shape/dtype"):
        mod.assert_prefix_bitwise_unchanged(base, base.astype(np.float64), 1)


def test_empty_self_pair_matrix_allows_empty_or_area_rows():
    empty = sparse.csr_matrix((0, 9), dtype=np.float64)
    A0, b0 = mod._owner_matrix([], empty, np.empty(0), np.array([0, 1, 2]))
    assert A0.shape == (0, 9) and b0.shape == (0,)
    A, b = mod._owner_matrix([], sparse.csr_matrix([[1., 0., 0., 0., 0., 0., 0., 0., 0.]]),
                              np.array([1.]), np.array([0, 1, 2]))
    assert A.shape == (1, 9) and b.shape == (1,)
