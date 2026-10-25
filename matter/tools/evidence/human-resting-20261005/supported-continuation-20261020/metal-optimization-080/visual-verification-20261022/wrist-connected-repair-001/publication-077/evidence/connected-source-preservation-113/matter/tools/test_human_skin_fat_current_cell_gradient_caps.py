import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import human_skin_fat_connected_repair as repair
from test_human_skin_fat_dihedral_constraint import _witness_fixture


def _thin_positive_tet(target_j):
    # A very flat but positive tetrahedron with q just above the physical floor.
    h = 2.96e-5
    X = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., h]])
    T = np.array([[0, 1, 2, 3]], dtype=np.int64)
    R = X / np.cbrt(target_j)
    J, _, q, _ = repair.cell_values_gradients(X, R, T)
    return X, R, T, float(q[0]), float(J[0])


@pytest.mark.parametrize('target_j', [0.351, 1.0, 2.49])
def test_sufficient_caps_preserve_near_floor_flat_tet_under_random_gradients(target_j):
    X, R, T, q0, J0 = _thin_positive_tet(target_j)
    q_min = 1e-5
    assert np.linalg.cond((R[T[0, 1:]] - R[T[0, 0]]).T) > 1e4
    assert q0 > q_min and J0 == pytest.approx(target_j, rel=2e-12)
    caps = repair.sufficient_cell_gradient_caps([q0], [J0], q_min, (0.35, 2.5), 0.2)
    assert caps.shape == (1,) and 0. <= caps[0] <= 0.2
    rng = np.random.default_rng(20261025 + int(target_j * 1000))
    for _ in range(100):
        G = rng.uniform(-0.95 * caps[0], 0.95 * caps[0], size=(3, 3))
        Y = X @ (np.eye(3) + G).T
        J, _, q, _ = repair.cell_values_gradients(Y, R, T)
        assert np.isfinite(q[0]) and q[0] >= q_min
        assert np.isfinite(J[0]) and 0.35 <= J[0] <= 2.5


def test_cap_rows_use_current_geometry_and_report_only_solved_caps(monkeypatch):
    kwargs = _witness_fixture()
    # A small non-reference deformation makes current and reference P1 gradients differ.
    kwargs['points'] = kwargs['points'].copy()
    kwargs['points'][4, 2] += 2e-4
    kwargs['cell_witness_gradient_limit'] = 0.15
    kwargs['free_nodes'] = np.array([2, 3, 4], dtype=np.int64)
    calls = []

    def fake(c, **kw):
        calls.append((kw['A_ub'].copy(), kw['b_ub'].copy()))
        x = np.zeros(len(c))
        x[5] = 1.0
        x[8] = -30.0
        return type('R', (), {'success': True, 'status': 0, 'message': 'injected', 'x': x})()

    monkeypatch.setattr(repair, 'linprog', fake)
    Y, report = repair.coupled_clearance_step(**kwargs)
    assert len(calls) == 2
    ids = np.asarray(report['lp_cell_gradient_cap_tetrahedron_rows'], dtype=np.int64)
    caps = np.asarray(report['lp_cell_gradient_cap_values'], dtype=np.float64)
    assert len(ids) == report['lp_cell_gradient_cap_count'] == 2
    assert report['lp_cell_gradient_cap_row_count'] == 9 * len(ids)
    assert report['lp_cell_gradient_cap_min'] == pytest.approx(caps.min())
    assert report['lp_cell_gradient_cap_max'] == pytest.approx(caps.max())
    assert np.all((caps >= 0.) & (caps <= 0.15))
    X = kwargs['points']
    J0, _, q0, _ = repair.cell_values_gradients(X, kwargs['reference'], kwargs['tetrahedra'])
    expected_caps = repair.sufficient_cell_gradient_caps(q0[ids], J0[ids], 1e-5, (0.35, 2.5), 0.15)
    np.testing.assert_allclose(caps, expected_caps, rtol=0., atol=0.)

    # The added LP rows are exactly ±DG(X), with nine component rows per cell.
    X = kwargs['points']
    T = kwargs['tetrahedra']
    F = kwargs['free_nodes']
    DG = repair.sparse_displacement_gradient(X, T[ids], F)
    DG_ref = repair.sparse_displacement_gradient(kwargs['reference'], T[ids], F)
    assert (DG - DG_ref).nnz and np.max(np.abs((DG - DG_ref).data)) > 1e-7
    block_rows = 9 * len(ids)
    added = calls[1][0][-2 * block_rows:]
    zero = sparse.csr_matrix((DG.shape[0], calls[1][0].shape[1] - DG.shape[1]))
    expected = sparse.vstack([sparse.hstack([DG, zero]), sparse.hstack([-DG, zero])], format='csr')
    np.testing.assert_allclose((added - expected).toarray(), 0., atol=1e-12)
    bound = np.r_[np.repeat(caps, 9), np.repeat(caps, 9)]
    np.testing.assert_allclose(calls[1][1][-2 * block_rows:], bound, atol=1e-15)

    # Mock LP output intentionally violates the full step; emitted Float32 gates decide.
    assert Y.dtype == np.float32
    assert report['components'][0]['attempts'][0]['first_failed_gate'] in ('cell_quality', 'cell_J')
    assert report['components'][0]['line_fraction'] < 1.
    J, _, q, _ = repair.cell_values_gradients(Y, kwargs['reference'], T)
    assert q.min() >= 1e-5 and J.min() >= 0.35 and J.max() <= 2.5
    assert report['cell_gradient_caps_require_exact_float32_q_J_postcheck'] is True
    assert report['lp_cell_witness_gradient_limit'] == 0.15


def test_failed_refinement_reports_only_caps_used_by_retained_direction(monkeypatch):
    kwargs = _witness_fixture()
    kwargs['cell_witness_gradient_limit'] = 0.15
    kwargs['free_nodes'] = np.array([2, 3, 4], dtype=np.int64)
    kwargs['dihedral_constraint_passes'] = 1
    calls = []

    def fake(c, **kw):
        calls.append(kw['A_ub'].shape[0])
        x = np.zeros(len(c))
        if len(calls) == 1:
            x[8] = -30.0  # creates cell witnesses and caps after this solve
            return type('R', (), {'success': True, 'status': 0, 'message': 'first', 'x': x})()
        return type('R', (), {'success': False, 'status': 1, 'message': 'timeout', 'x': x})()

    monkeypatch.setattr(repair, 'linprog', fake)
    _, report = repair.coupled_clearance_step(**kwargs)
    assert len(calls) == 2 and calls[1] > calls[0]
    assert report['lp_refinement_failure_retained'] is True
    # The retained successful solve predates those pending caps, so telemetry
    # must not imply the failed refinement actually enforced them.
    assert report['lp_cell_gradient_cap_count'] == 0
    assert report['lp_cell_gradient_cap_row_count'] == 0
    assert report['lp_cell_gradient_cap_values'] == []
    assert report['lp_constraint_generation'][0]['new_cell_gradient_caps'] == 2
    assert report['lp_constraint_generation'][1]['success'] is False
