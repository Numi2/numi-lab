import numpy as np
import pytest

from connected_patch_sat_selector638 import (
    AxisOption, _row_big_m, build_component_milp,
    partition_pair_components, solve_component,
)


def _op(pair, normal, rhs):
    return AxisOption(
        pair_index=pair, normal=np.asarray(normal, dtype=np.float64),
        rhs_mm=np.asarray(rhs, dtype=np.float64), source="test_axis",
        axis_index=0, axis_sign=1, source_order=0, sign_order=0,
        extra_sign=1,
    )


def test_partition_uses_exact_shared_vertex_incidence_and_stable_order():
    nodes = np.array([[5, 6, 7], [7, 8, 9], [20, 21, 22], [23, 24, 25]])
    groups = partition_pair_components(nodes)
    assert [g.tolist() for g in groups] == [[0, 1], [2], [3]]


def test_shared_node_axis_conflict_selects_maximum_compatible_subset():
    nodes = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
    ids = np.array([[10, 1, 2, 3], [11, 1, 2, 4]], dtype=np.int64)
    lo = {i: np.array([-1., -1., -1.]) for i in range(5)}
    hi = {i: np.array([1., 1., 1.]) for i in range(5)}
    options = [
        [_op(0, [1., 0., 0.], [1., -1., -1.])],
        [_op(1, [-1., 0., 0.], [1., -1., -1.])],
    ]
    model = build_component_milp(np.array([0, 1]), nodes, ids, options, lo, hi)
    result = solve_component(model, time_limit_s=2.0, node_limit=1000)
    assert result["has_incumbent"]
    assert result["status"] == 0
    assert len(result["selected"]) == 1
    selected = result["selected"][0]["pair_index"]
    d0 = result["node_displacements_mm"][model["node_pos"][0], 0]
    assert (selected == 0 and d0 >= 1. - 1e-6) or (selected == 1 and d0 <= -1. + 1e-6)


def test_row_specific_big_m_is_finite_and_exactly_box_redundant():
    n = np.array([1., 0., 0.])
    lo = np.array([-0.5, -2., -3.])
    hi = np.array([0.25, 4., 5.])
    M, minimum = _row_big_m(n, rhs=1.75, lo=lo, hi=hi, relaxation_pad_mm=1e-8)
    assert minimum == pytest.approx(-0.5)
    assert M == pytest.approx(2.25000001)
    # y=0 at the minimizing box point satisfies -n·d <= M-rhs.
    assert -minimum <= M - 1.75 + 1e-12
    assert np.isfinite(M) and M > 0


def test_fixed_anchor_zero_bound_prevents_axis_selection():
    nodes = np.array([[0, 1, 2]], dtype=np.int64)
    ids = np.array([[10, 1, 2, 3]], dtype=np.int64)
    lo = {0: np.zeros(3), 1: np.array([-1., -1., -1.]), 2: np.array([-1., -1., -1.])}
    hi = {0: np.zeros(3), 1: np.array([1., 1., 1.]), 2: np.array([1., 1., 1.])}
    model = build_component_milp(np.array([0]), nodes, ids,
        [[_op(0, [1., 0., 0.], [0.5, -1., -1.])]], lo, hi, anchors=[0])
    result = solve_component(model, time_limit_s=2.0, node_limit=1000)
    assert result["status"] == 0
    assert result["selected"] == []
    assert np.array_equal(result["node_displacements_mm"][model["node_pos"][0]], np.zeros(3))


def test_pair_without_any_available_axis_is_an_explicit_unselected_result():
    nodes = np.array([[0, 1, 2]], dtype=np.int64)
    ids = np.array([[10, 1, 2, 3]], dtype=np.int64)
    lo = {i: np.zeros(3) for i in range(3)}
    hi = {i: np.zeros(3) for i in range(3)}
    model = build_component_milp(np.array([0]), nodes, ids, [[]], lo, hi)
    result = solve_component(model, time_limit_s=2.0, node_limit=1000)
    assert result["status"] == 0 and result["has_incumbent"]
    assert result["selected"] == []
    assert result["message"].startswith("No box-feasible")


def test_rejects_bad_axis_and_missing_or_unfixed_anchor_box():
    nodes = np.array([[0, 1, 2]], dtype=np.int64)
    ids = np.array([[10, 1, 2, 3]], dtype=np.int64)
    lo = {i: np.array([-1., -1., -1.]) for i in range(3)}
    hi = {i: np.array([1., 1., 1.]) for i in range(3)}
    with pytest.raises(ValueError, match="unit"):
        build_component_milp(np.array([0]), nodes, ids,
            [[_op(0, [2., 0., 0.], [0., 0., 0.])]], lo, hi)
    del lo[2]
    with pytest.raises(ValueError, match="missing"):
        build_component_milp(np.array([0]), nodes, ids, [[]], lo, hi)
    lo = {i: np.array([-1., -1., -1.]) for i in range(3)}
    hi = {i: np.array([1., 1., 1.]) for i in range(3)}
    with pytest.raises(ValueError, match="not fixed at zero"):
        build_component_milp(np.array([0]), nodes, ids, [[]], lo, hi, anchors=[1])
