import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

SCRIPT = Path(__file__).with_name("propose_joint_clearance_witnesses_399.py")
spec = importlib.util.spec_from_file_location("joint399_tradeoffs_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _tradeoff_inputs():
    # The unchanged hard row requires d >= 0.25. One clearance target improves
    # for positive d while its paired target regresses, so C@d>=0 blocks the
    # hard-feasible move but normalized slack can still price the tradeoff.
    A = sparse.csr_matrix([[1.0]])
    b = np.array([0.25])
    target = np.array([0.0])
    lower = np.array([-0.5])
    upper = np.array([0.5])
    C = sparse.csr_matrix([[-1.0], [1.0]])
    h = np.array([1.0, 1.0])
    weights = np.array([1.0, 1.0])
    return A, b, target, lower, upper, C, h, weights


def _solve(*, allow, basis=False):
    A, b, target, lower, upper, C, h, weights = _tradeoff_inputs()
    common = (A, b, target, lower, upper, C, h, weights)
    if basis:
        return mod.solve_sparse_l1_clearance_in_basis(
            *common, sparse.eye(1, format="csr"), 5.0,
            allow_clearance_target_tradeoffs=allow,
        )
    return mod.solve_sparse_l1_clearance(
        *common, 5.0, allow_clearance_target_tradeoffs=allow,
    )


@pytest.mark.parametrize("basis", [False, True])
def test_default_policy_still_hard_blocks_clearance_regression(basis):
    result = _solve(allow=False, basis=basis)
    assert not result["success"]
    assert result["stage1"]["status"] == 2


@pytest.mark.parametrize("basis", [False, True])
def test_optin_keeps_hard_rows_and_normalized_slack_objective(basis):
    result = _solve(allow=True, basis=basis)
    assert result["success"]
    assert result["clearance_target_tradeoffs_opt_in"] is True
    delta = np.asarray(result["delta"])
    # The original A d >= b row remains active; the opt-in removes only C d >= 0.
    assert float(delta[0]) >= 0.25 - 2e-7
    assert float(delta[0]) == pytest.approx(0.25, abs=2e-7)
    progress = np.array([-delta[0], delta[0]])
    assert progress.min() < -0.24
    slack = np.asarray(result["normalized_slack"])
    assert np.all(slack >= 0.0)
    assert np.all(progress + np.array([1.0, 1.0]) * slack >= 1.0 - 2e-7)
    assert result["normalized_slack_objective"] == pytest.approx(2.0, abs=2e-7)
    assert result["minimum_target_inequality_slack_mm"] >= -2e-7


def _backtrack(*, allow, defer=False):
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
    points_mm = points.astype(np.float64) * 1000.0
    support = np.array([0, 1, 2], dtype=np.int64)
    anchors = np.empty(0, dtype=np.int64)
    lower = np.full(9, -1.0)
    upper = np.full(9, 1.0)
    C = sparse.csr_matrix(([-1.0], ([0], [0])), shape=(1, 9))
    edge_A = sparse.csr_matrix((0, 9))
    return mod.backtrack_area_feasible_scp_delta(
        points, points.copy(), points_mm, points_mm.copy(),
        np.array([[0, 1, 2]], dtype=np.int64), support, anchors, 0,
        np.array([0.1, 0., 0., 0., 0., 0., 0., 0., 0.]),
        lower, upper, C, points.astype(np.float64), 10.0, 10.0, 0.1,
        edge_A, np.empty(0), max_halvings=16,
        allow_clearance_target_tradeoffs=allow,
        defer_clearance_proxy=defer,
    )


def test_emitted_f32_proxy_tradeoff_requires_explicit_optin():
    strict_points, _, strict = _backtrack(allow=False)
    assert strict_points is not None
    assert strict["accepted_alpha"] < 1.0
    assert strict["accepted_minimum_clearance_progress_mm"] >= -2e-5

    trade_points, _, trade = _backtrack(allow=True, defer=True)
    assert trade_points is not None
    assert trade["accepted_alpha"] == 1.0
    assert trade["clearance_target_tradeoffs_opt_in"] is True
    assert trade["accepted_minimum_clearance_progress_mm"] < -0.09
    assert trade["accepted_clearance_proxy_gate"]["diagnostic_pass"] is False
    assert trade["accepted_clearance_proxy_gate"]["deferred"] is True


def test_cli_exposes_explicit_construction_optin():
    import subprocess
    out = subprocess.check_output(
        ["/Users/numi/numi-human-prep-venv-20261005/bin/python",
         str(SCRIPT), "--help"], text=True,
    )
    assert "--allow-clearance-target-tradeoffs" in out


def test_tradeoff_optin_rejects_precision_phase_one_combination():
    import subprocess
    proc = subprocess.run(
        ["/Users/numi/numi-human-prep-venv-20261005/bin/python", str(SCRIPT),
         "--self-test", "--allow-clearance-target-tradeoffs",
         "--precision-phase-one-report", "report.json"],
        text=True, capture_output=True,
    )
    assert proc.returncode == 2
    assert "cannot be combined with precision Phase-I" in proc.stderr


def test_tradeoff_optin_does_not_implicitly_defer_emitted_proxy():
    # The hard LP policy opt-in is separate from emitted-proxy deferral.
    trade_points, _, trade = _backtrack(allow=True, defer=False)
    assert trade_points is not None
    assert trade["accepted_alpha"] < 1.0
    assert trade["accepted_clearance_proxy_gate"]["deferred"] is False
