import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse

SRC = Path(__file__).with_name("target_row_progress_feasibility_397.py")
spec = importlib.util.spec_from_file_location("targetrow397", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_global_no_regression_can_block_selected_row_progress():
    A = sparse.csr_matrix((0, 2))
    C = sparse.csr_matrix([[1.0, 0.0], [-1.0, 0.0]])
    got = mod.compare_target_progress_feasibility(
        A, np.empty(0), [-1.0, -1.0], [1.0, 1.0], None,
        C, [1.0, 1.0], 0,
    )
    no = got["results"]["without_global_no_regression"]
    yes = got["results"]["with_global_no_regression"]
    assert no["success"] and yes["success"]
    assert no["certificate_passed"] and yes["certificate_passed"]
    assert no["maximum_normalized_target_progress"] == pytest.approx(1.0)
    assert yes["maximum_normalized_target_progress"] == pytest.approx(0.0, abs=1e-9)
    assert got["normalized_target_progress_reduction"] == pytest.approx(1.0)
    assert len(no["inequality_dual_marginals"]) > 0
    assert len(yes["inequality_dual_marginals"]) > 0


def test_reduced_map_preserves_output_space_constraints():
    A = sparse.csr_matrix([[0.0, 1.0]])
    b = np.array([-0.5])
    M = sparse.csr_matrix([[1.0], [1.0]])
    C = sparse.csr_matrix([[1.0, 0.0], [-1.0, 0.0]])
    got = mod.compare_target_progress_feasibility(
        A, b, [-1.0, -1.0], [0.75, 0.75], M, C, [0.75, 1.0], 0,
    )
    no = got["results"]["without_global_no_regression"]
    yes = got["results"]["with_global_no_regression"]
    assert no["certificate_passed"] and yes["certificate_passed"]
    assert no["maximum_target_progress_mm"] == pytest.approx(0.75)
    assert yes["maximum_target_progress_mm"] == pytest.approx(0.0, abs=1e-9)
    assert no["minimum_hard_A_slack"] >= -1e-9
    assert yes["minimum_output_upper_slack"] >= -1e-9


def test_hard_constraint_changes_best_target_progress():
    A = sparse.csr_matrix([[1.0, 1.0]])
    C = sparse.csr_matrix([[1.0, 0.0]])
    got = mod.compare_target_progress_feasibility(
        A, [1.25], [-1.0, -1.0], [1.0, 1.0], None, C, [1.0], 0,
    )
    no = got["results"]["without_global_no_regression"]
    yes = got["results"]["with_global_no_regression"]
    assert no["success"] and yes["success"]
    assert no["maximum_normalized_target_progress"] == pytest.approx(1.0)
    assert yes["maximum_normalized_target_progress"] == pytest.approx(1.0)
    assert yes["primal_delta_output"][0] == pytest.approx(1.0)
    assert yes["primal_delta_output"][0] + yes["primal_delta_output"][1] >= 1.25 - 1e-8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_row": 2},
        {"target_row": 0.5},
        {"h": [0.0]},
        {"h": [-1.0]},
        {"h": [float("nan")]},
    ],
)
def test_invalid_target_or_height_rejects(kwargs):
    args = dict(A=sparse.csr_matrix((0, 1)), b=[], lower=[-1.0], upper=[1.0],
                output_map=None, C=sparse.csr_matrix([[1.0]]), h=[1.0], target_row=0)
    args.update(kwargs)
    with pytest.raises(ValueError):
        mod.compare_target_progress_feasibility(**args)


def test_shape_or_bound_errors_reject():
    with pytest.raises(ValueError):
        mod.compare_target_progress_feasibility(
            sparse.csr_matrix((0, 2)), [], [-1], [1], None,
            sparse.csr_matrix([[1]]), [1], 0,
        )
    with pytest.raises(ValueError):
        mod.compare_target_progress_feasibility(
            sparse.csr_matrix((0, 1)), [], [1], [-1], None,
            sparse.csr_matrix([[1]]), [1], 0,
        )


def test_infinite_output_bounds_are_omitted_from_lp_rows():
    got = mod.compare_target_progress_feasibility(
        sparse.csr_matrix((0, 1)), [], [-np.inf], [1.0], None,
        sparse.csr_matrix([[1.0]]), [1.0], 0,
    )
    no = got["results"]["without_global_no_regression"]
    yes = got["results"]["with_global_no_regression"]
    assert no["success"] and yes["success"]
    assert no["certificate_passed"] and yes["certificate_passed"]
    assert no["maximum_normalized_target_progress"] == pytest.approx(1.0)
    assert yes["maximum_normalized_target_progress"] == pytest.approx(1.0)


def test_target_row_can_be_diagnostic_without_geometry_mutation():
    points = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    before = points.tobytes()
    got = mod.compare_target_progress_feasibility(
        sparse.csr_matrix((0, 3)), [], [-0.5] * 3, [0.5] * 3, None,
        sparse.csr_matrix([[1.0, 0.0, 0.0]]), [0.25], 0,
    )
    assert got["results"]["with_global_no_regression"]["success"]
    assert points.tobytes() == before
