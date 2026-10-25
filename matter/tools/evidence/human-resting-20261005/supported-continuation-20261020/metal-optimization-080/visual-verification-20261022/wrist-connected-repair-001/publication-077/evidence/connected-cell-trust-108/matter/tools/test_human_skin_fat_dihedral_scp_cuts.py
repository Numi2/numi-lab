import numpy as np
import pytest
from scipy import sparse

from human_skin_fat_dihedral_scp_cuts import make_propagated_dihedral_cuts


class Surface:
    area_bounds = (.5, 2.)
    normal_cosine_min = .5

    def __init__(self, active, delta, jacobian, *, area=(1.,), cosine=(1.,), current_pass=True):
        self.active = np.asarray(active, dtype=np.int64)
        self.delta = np.asarray(delta, dtype=np.float64)
        self.jacobian = sparse.csr_matrix(jacobian, dtype=np.float64)
        self.area = np.asarray(area, dtype=np.float64)
        self.cosine = np.asarray(cosine, dtype=np.float64)
        self.current_pass = current_pass

    def dihedral_passes(self, points):
        return self.current_pass

    def metrics(self, points):
        n = len(self.area)
        return (np.zeros(n), np.zeros(n), self.area, self.cosine)

    def dihedral_change_linearization(self, points, free_nodes, edge_ids=None):
        wanted = self.active if edge_ids is None else np.intersect1d(self.active, edge_ids)
        rows = [int(np.flatnonzero(self.active == i)[0]) for i in wanted]
        return wanted, self.delta[rows], self.jacobian[rows]


def test_trial_tangent_is_recentered_and_keeps_current():
    X = np.zeros((1, 3), dtype=np.float32)
    Z = np.array([[.001, 0., 0.]], dtype=np.float32)
    surface = Surface([9], [-.2], [[-200., 0., 0.]])
    result = make_propagated_dihedral_cuts(
        surface, X, Z, np.array([0]), np.array([9]), limit_radians=.15)
    assert result["status"] == "cuts_ready"
    assert result["edge_ids"].tolist() == [9]
    assert result["jacobian_mm"].toarray()[0, 0] == pytest.approx(-.2)
    assert result["upper_zero_slack_rad"][0] == pytest.approx(.15)
    assert result["lower_zero_slack_rad"][0] == pytest.approx(.15)
    z_mm = result["trial_displacement_mm"]
    assert np.all(result["A_ub"] @ np.zeros(3) <= result["b_ub_rad"])
    assert np.any(result["A_ub"] @ z_mm > result["b_ub_rad"] + 1e-12)


def test_tangent_that_excludes_current_is_declined_not_forced():
    X = np.zeros((1, 3), dtype=np.float32)
    Z = np.array([[.001, 0., 0.]], dtype=np.float32)
    surface = Surface([3], [.5], [[0., 0., 0.]])
    result = make_propagated_dihedral_cuts(
        surface, X, Z, np.array([0]), np.array([3]), limit_radians=.1)
    assert result["status"] == "no_advisory_cuts"
    assert result["A_ub"].shape == (0, 3)
    assert result["skipped"][0]["reason"] == "tangent_cut_excludes_current"


def test_inactive_edge_and_zero_jacobian_are_reported_without_empty_rows():
    X = np.zeros((2, 3), dtype=np.float32)
    Z = X.copy()
    surface = Surface([7], [0.], [[0., 0., 0.]])
    result = make_propagated_dihedral_cuts(
        surface, X, Z, np.array([0]), np.array([7, 8]), limit_radians=.1)
    assert result["A_ub"].shape == (0, 3)
    reasons = {(r["edge_id"], r["reason"]) for r in result["skipped"]}
    assert (8, "no_free_support") in reasons
    assert (7, "zero_jacobian") in reasons


def test_degenerate_trial_surface_declines_all_cuts():
    X = np.zeros((1, 3), dtype=np.float32)
    Z = X.copy()
    surface = Surface([1], [0.], [[1., 0., 0.]], area=(0.,))
    result = make_propagated_dihedral_cuts(
        surface, X, Z, np.array([0]), np.array([1]), limit_radians=.2)
    assert result["status"] == "no_advisory_cuts"
    assert result["skipped"][0]["reason"] == "trial_surface_outside_area_guard"


@pytest.mark.parametrize("bad_edges", [
    np.array([1, 1], dtype=np.int64),
    np.array([-1], dtype=np.int64),
    np.array([1.], dtype=np.float64),
])
def test_invalid_edge_domains_fail_closed(bad_edges):
    X = np.zeros((1, 3), dtype=np.float32)
    surface = Surface([1], [0.], [[1., 0., 0.]])
    with pytest.raises(ValueError):
        make_propagated_dihedral_cuts(
            surface, X, X, np.array([0]), bad_edges, limit_radians=.2)


def test_invalid_current_state_fails_closed():
    X = np.zeros((1, 3), dtype=np.float32)
    surface = Surface([1], [0.], [[1., 0., 0.]], current_pass=False)
    with pytest.raises(ValueError, match="current_points"):
        make_propagated_dihedral_cuts(
            surface, X, X, np.array([0]), np.array([1]), limit_radians=.2)

