import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

SCRIPT = Path(__file__).with_name("propose_joint_clearance_witnesses_399.py")
spec = importlib.util.spec_from_file_location("joint365", SCRIPT)
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


def _fixed_boundary_fixture(tmp_path, *, candidate_sha="a" * 64, face_rows=None,
                            typed_pairs=None, units="mm"):
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    face_rows = np.array([0] if face_rows is None else face_rows, dtype=np.int64)
    body = np.array([2], dtype=np.int64)
    bone = np.array([7], dtype=np.int64)
    triangle_id = np.array([100], dtype=np.int64)
    triangle = np.array([[[0., 0., 2.], [1., 0., 2.], [0., 1., 2.]]], dtype=np.float64)
    npz_path = tmp_path / "introduced-skin-bone-barriers.npz"
    np.savez_compressed(npz_path, skin_face_rows=face_rows, bank_body=body,
                        stable_bone_id=bone, bank_triangle_stable_id=triangle_id,
                        fixed_triangles_world_mm=triangle)
    typed_pairs = ([{"skin_face_row": 0, "bank_body": 2, "stable_bone_id": 7,
                     "bank_triangle_stable_id": 100}] if typed_pairs is None else typed_pairs)
    report = {
        "status": "COMPLETE_TYPED_INTRODUCED_SKIN_BONE_BARRIER_EXPORT",
        "base_candidate_sha256": candidate_sha,
        "candidate_points_world_m_array_sha256": mod.array_sha256(points),
        "skin_faces_array_sha256": mod.array_sha256(faces),
        "point_count": 3,
        "face_count": 1,
        "fixed_triangles_units": units,
        "fixed_triangles_frame": "world",
        "bank_count": 185,
        "expected_bank_count": 185,
        "geometry_or_native_admission": False,
        "introduced_pair_count": len(typed_pairs),
        "typed_pairs": typed_pairs,
        "inputs": [{"role": "base", "candidate_sha256": candidate_sha}],
        "output_npz": npz_path.name,
        "output_npz_sha256": mod.sha(npz_path),
    }
    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]).copy() for k in z.files}
    return arrays, report, npz_path, points, faces


def test_fixed_boundary_payload_pins_base_topology_units_and_typed_ids(tmp_path):
    arrays, report, npz_path, points, faces = _fixed_boundary_fixture(tmp_path)
    records, receipt = mod.validate_fixed_boundary_payload(
        arrays, report, candidate_sha256="a" * 64, candidate_points=points,
        candidate_faces=faces, npz_path=npz_path,
    )
    assert len(records) == 1 and records[0]["fixed_triangle_world_mm"].shape == (3, 3)
    assert receipt["units"] == "world-frame millimetres"
    changed = points.copy(); changed[0, 0] = np.float32(0.1)
    with pytest.raises(ValueError, match="point-array pins"):
        mod.validate_fixed_boundary_payload(arrays, report, candidate_sha256="a" * 64,
                                            candidate_points=changed, candidate_faces=faces,
                                            npz_path=npz_path)
    with pytest.raises(ValueError, match="candidate SHA"):
        mod.validate_fixed_boundary_payload(arrays, report, candidate_sha256="b" * 64,
                                            candidate_points=points, candidate_faces=faces,
                                            npz_path=npz_path)
    with pytest.raises(ValueError, match="world-frame millimetres"):
        bad = dict(report); bad["fixed_triangles_units"] = "m"
        mod.validate_fixed_boundary_payload(arrays, bad, candidate_sha256="a" * 64,
                                            candidate_points=points, candidate_faces=faces,
                                            npz_path=npz_path)


def test_fixed_boundary_payload_rejects_duplicate_or_out_of_range_typed_ids(tmp_path):
    arrays, report, npz_path, points, faces = _fixed_boundary_fixture(tmp_path)
    bad_report = dict(report)
    bad_report["typed_pairs"] = [dict(report["typed_pairs"][0]), dict(report["typed_pairs"][0])]
    bad_report["introduced_pair_count"] = 2
    with pytest.raises(ValueError, match="typed-pair list"):
        mod.validate_fixed_boundary_payload(arrays, bad_report, candidate_sha256="a" * 64,
                                            candidate_points=points, candidate_faces=faces,
                                            npz_path=npz_path)
    bad_arrays = dict(arrays); bad_arrays["skin_face_rows"] = np.array([1], dtype=np.int64)
    with pytest.raises(ValueError, match="outside base topology"):
        mod.validate_fixed_boundary_payload(bad_arrays, report, candidate_sha256="a" * 64,
                                            candidate_points=points, candidate_faces=faces,
                                            npz_path=npz_path)


def test_fixed_boundary_barriers_map_only_skin_nodes_and_keep_existing_row_schema():
    pair_owner = mod.load_module("pair_owner_315_test", mod.DEFAULT_PAIR_OWNER)
    fixed_owner = mod.load_module("fixed_owner_315_test", Path("/Users/n/Work/numi-human-connected-surface-069/connected-fixed-boundary-313/fixed_bone_triangle_rows_313.py"))
    points_mm = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    record = {"skin_face_row": 0, "bank_body": 2, "stable_bone_id": 7,
              "bank_triangle_stable_id": 100,
              "fixed_triangle_world_mm": np.array([[0., 0., 2.], [1., 0., 2.], [0., 1., 2.]])}
    rows, typed = mod.build_fixed_boundary_barriers(points_mm, faces, [record], pair_owner,
                                                   fixed_owner, 0.25)
    assert rows and len(typed) == 1
    assert typed[0]["kind"] == "fixed_bone_triangle"
    assert all(set(map(int, row["nodes"])).issubset({0, 1, 2}) for row in rows)
    A, b = mod._owner_matrix(rows, sparse.csr_matrix((0, 9)), np.empty(0), np.array([0, 1, 2]))
    assert A.shape[0] == len(rows) and b.shape == (len(rows),)


def test_fixed_boundary_barrier_metadata_uses_313_skin_face_canonical_order():
    pair_owner = mod.load_module("pair_owner_315_order_test", mod.DEFAULT_PAIR_OWNER)
    fixed_owner = mod.load_module("fixed_owner_315_order_test", Path("/Users/n/Work/numi-human-connected-surface-069/connected-fixed-boundary-313/fixed_bone_triangle_rows_313.py"))
    points_mm = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                          [3.,0.,0.],[4.,0.,0.],[3.,1.,0.]])
    faces = np.array([[0,1,2],[3,4,5]], dtype=np.int64)
    def rec(face, body, bone, tid, x):
        return {"skin_face_row": face, "bank_body": body, "stable_bone_id": bone,
                "bank_triangle_stable_id": tid,
                "fixed_triangle_world_mm": np.array([[x,0.,2.],[x+1.,0.,2.],[x,1.,2.]])}
    # 314 arrays are sorted body/bone/face; 313 returns skin-face/body/bone/triangle.
    input_records = [rec(1, 1, 9, 90, 3.), rec(0, 2, 7, 70, 0.)]
    rows, typed = mod.build_fixed_boundary_barriers(points_mm, faces, input_records,
                                                   pair_owner, fixed_owner, .25)
    assert rows
    assert [(r["skin_face_row"], r["bank_body"]) for r in typed] == [(0,2),(1,1)]


def test_fixed_boundary_rows_are_appended_after_self_rows_without_default_change():
    self_rows = [{"nodes": np.array([0]), "coefficients": np.array([[1., 0., 0.]]), "lower": 0.1}]
    fixed_rows = [{"nodes": np.array([1]), "coefficients": np.array([[0., 1., 0.]]), "lower": 0.2}]
    unchanged = mod.combine_barrier_rows(self_rows)
    assert len(unchanged) == 1 and unchanged[0] is self_rows[0]
    both = mod.combine_barrier_rows(self_rows, fixed_rows)
    assert len(both) == 2 and both[0] is self_rows[0] and both[1] is fixed_rows[0]


def _area_scp_backtrack_fixture():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                       [2., 2., 2.]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    points_mm = points.astype(np.float64) * 1000.0
    support = np.array([0, 1, 2], dtype=np.int64)
    anchors = np.array([0, 3], dtype=np.int64)
    lower = np.full(9, -10000., dtype=np.float64)
    upper = np.full(9, 10000., dtype=np.float64)
    C = sparse.csr_matrix((0, 9), dtype=np.float64)
    EG = sparse.csr_matrix((0, 9), dtype=np.float64)
    egb = np.empty(0, dtype=np.float64)
    return points, faces, points_mm, support, anchors, lower, upper, C, EG, egb


def _backtrack(points, faces, points_mm, support, anchors, lower, upper, C, EG, egb, lp_delta,
               *, max_halvings=16):
    return mod.backtrack_area_feasible_scp_delta(
        points, points.copy(), points_mm, points_mm.copy(), faces, support, anchors, 0,
        lp_delta, lower, upper, C, points.astype(np.float64), 20000., 20000., 0.1, EG, egb,
        max_halvings=max_halvings,
    )


def test_area_scp_backtracks_invalid_float32_endpoint_to_exact_feasible_delta():
    args = _area_scp_backtrack_fixture()
    points, faces, points_mm, support, anchors, lower, upper, C, EG, egb = args
    lp_delta = np.zeros(9, dtype=np.float64)
    lp_delta[0] = 250.0    # the native anchor request must remain exactly ignored
    lp_delta[3] = -2000.0  # endpoint reverses the only triangle; alpha=1/4 is valid
    raw = mod.emit_raw_lp_points(points, points_mm, support, lp_delta.reshape(-1, 3), anchors)
    assert not mod.full_f32_area_gate(raw, points_mm, points_mm, faces, .1)["pass"]
    accepted, actual_delta, receipt = _backtrack(
        points, faces, points_mm, support, anchors, lower, upper, C, EG, egb, lp_delta,
    )
    assert receipt["status"] == "ACCEPTED"
    assert receipt["accepted_alpha"] == .25
    gate = mod.full_f32_area_gate(accepted, points_mm, points_mm, faces, .1)
    assert gate["pass"] and gate["bad_current_oriented_face_count"] == 0
    assert gate["bad_fixed_037_reference_face_count"] == 0
    np.testing.assert_array_equal(accepted[anchors].view(np.uint32), points[anchors].view(np.uint32))
    assert np.linalg.norm(actual_delta) > 0


def test_area_scp_checks_actual_endpoint_when_path_has_interior_quadratic_fold():
    args = _area_scp_backtrack_fixture()
    points, faces, points_mm, support, anchors, lower, upper, C, EG, egb = args
    # With t in [0,1], the oriented double-area polynomial is
    # 1 - 6t + 6t^2: both endpoints pass, while t=1/2 folds.
    lp_delta = np.zeros(9, dtype=np.float64)
    lp_delta[4] = 1000.0      # e1(t) = (1,t)
    lp_delta[6] = -6000.0     # e2(t) = (-6t, 1-6t)
    lp_delta[7] = -6000.0
    midpoint = mod.emit_raw_lp_points(
        points, points_mm, support, (.5 * lp_delta).reshape(-1, 3), anchors
    )
    endpoint = mod.emit_raw_lp_points(
        points, points_mm, support, lp_delta.reshape(-1, 3), anchors
    )
    assert not mod.full_f32_area_gate(midpoint, points_mm, points_mm, faces, .1)["pass"]
    assert mod.full_f32_area_gate(endpoint, points_mm, points_mm, faces, .1)["pass"]
    accepted, _, receipt = _backtrack(
        points, faces, points_mm, support, anchors, lower, upper, C, EG, egb, lp_delta,
    )
    assert receipt["status"] == "ACCEPTED" and receipt["accepted_alpha"] == 1.
    np.testing.assert_array_equal(accepted.view(np.uint32), endpoint.view(np.uint32))


def test_area_scp_stops_when_no_float32_iterate_is_representable():
    args = _area_scp_backtrack_fixture()
    points, faces, points_mm, support, anchors, lower, upper, C, EG, egb = args
    tiny = np.zeros(9, dtype=np.float64)
    tiny[3] = 1e-12
    accepted, delta, receipt = _backtrack(
        points, faces, points_mm, support, anchors, lower, upper, C, EG, egb, tiny,
    )
    assert accepted is None and delta is None
    assert receipt["status"] == "NO_REPRESENTABLE_PROGRESS"
    assert receipt["trials"][0]["reason"] == "NO_REPRESENTABLE_PROGRESS"


def test_emitted_clearance_proxy_deferred_only_by_explicit_opt_in():
    strict = mod.emitted_clearance_proxy_gate(False, False)
    deferred = mod.emitted_clearance_proxy_gate(False, True)
    assert strict["diagnostic_pass"] is False
    assert strict["required_for_proposal_selection"] is True
    assert strict["proposal_selection_pass"] is False
    assert deferred["diagnostic_pass"] is False
    assert deferred["required_for_proposal_selection"] is False
    assert deferred["proposal_selection_pass"] is True
    assert deferred["native_admission"] is False


def _stage1_optimal(x, objective=0.0):
    from types import SimpleNamespace
    return SimpleNamespace(success=True, status=0, message="Optimization terminated successfully.",
                           x=np.asarray(x, dtype=np.float64), fun=float(objective), nit=4)


def _stage2_timeout():
    from types import SimpleNamespace
    return SimpleNamespace(success=False, status=1, message="Time limit reached",
                           x=None, fun=None, nit=200)


def _timeout_solve(monkeypatch, *, allow, stage1, stage2=None, A=None, b=None):
    calls = []
    results = [stage1, _stage2_timeout() if stage2 is None else stage2]
    def fake_linprog(*args, **kwargs):
        calls.append((args, kwargs))
        return results[len(calls) - 1]
    monkeypatch.setattr(mod, "linprog", fake_linprog)
    result = mod.solve_sparse_l1_clearance(
        sparse.csr_matrix((0, 1)) if A is None else sparse.csr_matrix(A),
        np.empty(0) if b is None else np.asarray(b, dtype=np.float64),
        np.array([0.0]), np.array([-1.0]), np.array([1.0]),
        sparse.csr_matrix([[1.0]]), np.array([0.5]), np.array([1.0]),
        1.0, 1e-9, allow_stage1_feasible_timeout_fallback=allow,
    )
    assert len(calls) == 2
    return result


def test_stage1_primal_is_used_only_when_opted_in_and_stage2_times_out(monkeypatch):
    result = _timeout_solve(monkeypatch, allow=True, stage1=_stage1_optimal([0.5, 0.0]))
    assert result["success"] is True
    assert result["fallback_used"] is True
    assert result["solution_source"] == "certified_stage1_primal_after_stage2_time_limit"
    assert result["stage1_primal_certificate"]["status"] == "CERTIFIED_STAGE1_PRIMAL"
    assert result["stage2"]["success"] is False and result["stage2"]["time_limit"] is True
    assert result["objective"] is None and result["secondary_l1_objective"] is None
    np.testing.assert_allclose(result["delta"], [0.5])
    assert result["minimum_normalized_linear_slack"] >= -2e-7
    assert result["minimum_clearance_progress_mm"] >= -2e-7


def test_stage1_timeout_fallback_is_off_by_default(monkeypatch):
    result = _timeout_solve(monkeypatch, allow=False, stage1=_stage1_optimal([0.5, 0.0]))
    assert result["success"] is False
    assert result["fallback_used"] is False
    assert result["stage2"]["time_limit"] is True


def test_stage1_timeout_fallback_rejects_uncertified_hard_row_primal(monkeypatch):
    result = _timeout_solve(monkeypatch, allow=True, stage1=_stage1_optimal([0.5, 0.0]),
                            A=[[1.0]], b=[0.75])
    assert result["success"] is False
    assert result["fallback_used"] is False
    assert result["stage1_primal_certificate"]["status"] == "STAGE1_PRIMAL_RESIDUAL_FAILED"
    assert result["stage1_primal_certificate"]["minimum_hard_row_residual"] < -2e-7


def test_stage1_timeout_fallback_does_not_treat_iteration_limit_as_time_limit(monkeypatch):
    from types import SimpleNamespace
    iteration_limit = SimpleNamespace(success=False, status=1, message="Iteration limit reached",
                                      x=None, fun=None, nit=200)
    result = _timeout_solve(monkeypatch, allow=True, stage1=_stage1_optimal([0.5, 0.0]),
                            stage2=iteration_limit)
    assert result["success"] is False
    assert result["fallback_used"] is False
    assert result["stage2"]["time_limit"] is False


def test_deferred_emitted_clearance_proxy_keeps_other_exact_gates(monkeypatch):
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                       [1., 2., 0.], [.01, 2., 0.]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    points_mm = points.astype(np.float64) * 1000.0
    support = np.arange(len(points), dtype=np.int64)
    anchors = np.array([0], dtype=np.int64)
    lower = np.full(3 * len(support), -1.0)
    upper = np.full(3 * len(support), 1.0)
    C = sparse.csr_matrix(([ -1.0, 1.0], ([0, 0], [9, 12])), shape=(1, 3 * len(support)))
    assert (C @ np.r_[np.zeros(9), .0003, np.zeros(2), .0003, np.zeros(2)]).item() == 0.0
    delta = np.zeros(3 * len(support), dtype=np.float64)
    delta[9] = .0003
    delta[12] = .0003
    edge = sparse.csr_matrix((0, 3 * len(support)))
    edge_b = np.empty(0)
    common = (points, points.copy(), points_mm, points_mm.copy(), faces, support, anchors, 0,
              delta, lower, upper, C, points.astype(np.float64), 1.0, 1.0, .1, edge, edge_b)
    strict_points, _, strict = mod.backtrack_area_feasible_scp_delta(*common, max_halvings=0)
    assert strict_points is None
    assert strict["trials"][0]["clearance_no_regression"] is False
    deferred_points, _, deferred = mod.backtrack_area_feasible_scp_delta(
        *common, max_halvings=0, defer_clearance_proxy=True
    )
    assert deferred_points is not None
    assert deferred["accepted"] is True
    assert deferred["accepted_clearance_no_regression"] is False
    assert deferred["accepted_clearance_proxy_gate"]["deferred"] is True
    assert deferred["accepted_emitted_gate"]["pass"] is True
    assert deferred["native_admission"] if "native_admission" in deferred else True


@pytest.mark.parametrize("solved", [
    {
        "success": True,
        "fallback_used": True,
        "solution_source": "certified_stage1_primal_after_stage2_time_limit",
        "delta": np.array([0.25, -0.5]),
        "normalized_slack": np.array([0.0, 0.125]),
        "normalized_slack_objective": 0.125,
        "stage2": {"success": False, "time_limit": True},
    },
    {
        "success": True,
        "delta": np.array([1.0]),
        "normalized_slack": np.array([0.0]),
        "normalized_slack_objective": 0.0,
        "stage2": {"success": True},
    },
])
def test_solver_failure_receipt_is_json_safe_and_hashes_slack(solved):
    import hashlib
    import json

    receipt = mod.solver_failure_receipt(solved)
    encoded = json.dumps({"solver": receipt}, sort_keys=True)
    assert "normalized_slack" not in receipt
    assert "delta" not in receipt
    summary = receipt["normalized_slack_receipt"]
    canonical = np.ascontiguousarray(solved["normalized_slack"], dtype=np.float64)
    assert summary["encoding"] == "float64-c-order"
    assert summary["shape"] == list(canonical.shape)
    assert summary["count"] == canonical.size
    assert summary["sha256"] == hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
    assert summary["all_finite"] is True
    assert json.loads(encoded)["solver"].get("fallback_used", False) is bool(solved.get("fallback_used", False))


def _tiny_lp_args():
    return (
        sparse.csr_matrix((0, 1)), np.empty(0), np.array([0.0]),
        np.array([-1.0]), np.array([1.0]), sparse.csr_matrix([[1.0]]),
        np.array([0.5]), np.array([1.0]), 2.0,
    )


def test_default_lp_method_matches_explicit_highs():
    default = mod.solve_sparse_l1_clearance(*_tiny_lp_args())
    explicit = mod.solve_sparse_l1_clearance(*_tiny_lp_args(), lp_method="highs")
    assert default["lp_method"] == explicit["lp_method"] == "highs"
    np.testing.assert_allclose(default["delta"], explicit["delta"], atol=1e-10)
    np.testing.assert_allclose(default["normalized_slack"], explicit["normalized_slack"], atol=1e-10)
    assert default["normalized_slack_objective"] == pytest.approx(explicit["normalized_slack_objective"])


@pytest.mark.parametrize("method", ["highs", "highs-ipm", "highs-ds"])
def test_lp_method_is_routed_to_both_lexicographic_stages(monkeypatch, method):
    real_linprog = mod.linprog
    seen = []
    def recording_linprog(*args, **kwargs):
        seen.append(kwargs.get("method"))
        return real_linprog(*args, **kwargs)
    monkeypatch.setattr(mod, "linprog", recording_linprog)
    result = mod.solve_sparse_l1_clearance(*_tiny_lp_args(), lp_method=method)
    assert result["success"]
    assert result["lp_method"] == method
    assert seen == [method, method]


def test_empty_target_lp_routes_selected_method(monkeypatch):
    real_linprog = mod.linprog
    seen = []
    def recording_linprog(*args, **kwargs):
        seen.append(kwargs.get("method"))
        return real_linprog(*args, **kwargs)
    monkeypatch.setattr(mod, "linprog", recording_linprog)
    result = mod.solve_sparse_l1_clearance(
        sparse.csr_matrix([[1.0]]), np.array([0.25]), np.array([0.0]),
        np.array([-1.0]), np.array([1.0]), sparse.csr_matrix((0, 1)),
        np.empty(0), np.empty(0), 2.0, lp_method="highs-ds",
    )
    assert result["success"] and result["lp_method"] == "highs-ds"
    assert seen == ["highs-ds"]


def test_invalid_lp_method_fails_before_solve():
    with pytest.raises(ValueError, match="unsupported LP method"):
        mod.solve_sparse_l1_clearance(*_tiny_lp_args(), lp_method="simplex")


def test_scp_progress_line_records_acceptance_and_exact_area(tmp_path):
    import json

    path = tmp_path / "out.solver-progress.jsonl"
    rejected = {
        "iteration": 1, "lp_method": "highs-ipm", "lp_wall_seconds": 4.5,
        "lp_success": True, "clearance_stage1": {"success": True},
        "clearance_stage2": {"success": True, "time_limit": False},
        "fallback_used": False, "absolute_delta_max_abs_mm": 1.25,
        "exact_f32_area_backtrack": {
            "accepted": False, "accepted_alpha": None,
            "trials": [{"area_gate": {
                "minimum_current_oriented_ratio": 0.2,
                "minimum_fixed_037_reference_magnitude_ratio": 0.4,
            }}],
        },
    }
    accepted = {
        "iteration": 2, "lp_method": "highs-ds", "lp_wall_seconds": 1.25,
        "lp_success": True, "clearance_stage1": {"success": True},
        "clearance_stage2": {"success": False, "time_limit": True},
        "fallback_used": True, "accepted_delta_max_abs_mm": 0.75,
        "exact_f32_area_backtrack": {"accepted_alpha": 0.5},
        "accepted_iterate_area_gate": {
            "pass": True, "minimum_current_oriented_ratio": 0.15,
            "minimum_fixed_037_reference_magnitude_ratio": 0.3,
        },
    }
    first = mod.append_scp_progress_line(path, rejected)
    second = mod.append_scp_progress_line(path, accepted)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0] == first and lines[1] == second
    assert first["accepted"] is False and first["accepted_alpha"] is None
    assert first["minimum_current_oriented_ratio"] == pytest.approx(0.2)
    assert second["accepted"] is True and second["accepted_alpha"] == 0.5
    assert second["fallback_used"] is True
    assert second["max_delta_mm"] == pytest.approx(0.75)


def _toy_immediate_q_archive():
    # Output rows 0..2 are parent identity rows; rows 3..4 are interpolated.
    return {
        "points_world_m": np.zeros((5, 3), dtype=np.float32),
        "immediate_refinement_prolongation_indptr": np.array([0, 1, 2, 3, 5, 7], dtype=np.int64),
        "immediate_refinement_prolongation_node_rows": np.array([0, 1, 2, 0, 1, 1, 2], dtype=np.int64),
        "immediate_refinement_prolongation_weights": np.array([1., 1., 1., .5, .5, .5, .5], dtype=np.float64),
    }


def test_immediate_parent_basis_expands_scope_and_excludes_anchor_controls():
    B, controls, receipt = mod.build_immediate_parent_basis(
        _toy_immediate_q_archive(), np.array([4], dtype=np.int64), np.array([2], dtype=np.int64)
    )
    np.testing.assert_array_equal(controls, [1])
    np.testing.assert_array_equal(receipt["output_scope_rows"], [1, 3, 4])
    assert B.shape == (3, 1)
    np.testing.assert_allclose(np.asarray(B.sum(axis=1)).ravel(), [1., .5, .5])
    assert receipt["added_output_rows"] == 2
    assert receipt["excluded_anchor_touching_controls"] == 1


def test_immediate_parent_basis_rejects_malformed_q_affine_rows():
    q = _toy_immediate_q_archive()
    q["immediate_refinement_prolongation_weights"] = q["immediate_refinement_prolongation_weights"].copy()
    q["immediate_refinement_prolongation_weights"][-1] = .25
    with pytest.raises(ValueError, match="affine partitions"):
        mod.build_immediate_parent_basis(q, np.array([3]), np.array([2]))


def test_kron_basis_preserves_random_output_linear_functionals():
    B, controls, receipt = mod.build_immediate_parent_basis(
        _toy_immediate_q_archive(), np.array([3], dtype=np.int64), np.array([2], dtype=np.int64)
    )
    M = sparse.kron(B, sparse.eye(3, format="csr"), format="csr")
    rng = np.random.default_rng(354)
    q = rng.normal(size=M.shape[1])
    ell = rng.normal(size=M.shape[0])
    output = np.asarray(M @ q).reshape(-1)
    np.testing.assert_allclose(ell @ output, np.asarray(M.T @ ell).reshape(-1) @ q,
                               rtol=2e-14, atol=2e-14)
    q_nodes = q.reshape(len(controls), 3)
    direct = np.asarray(B @ q_nodes)
    np.testing.assert_allclose(output.reshape(-1, 3), direct, rtol=0., atol=0.)


def test_basis_solver_uses_exact_output_l1_objective():
    M = sparse.csr_matrix([[1.], [1.]])
    A = sparse.csr_matrix([[1., 1.]])
    C = sparse.csr_matrix([[1., 0.]])
    result = mod.solve_sparse_l1_clearance_in_basis(
        A, np.array([0.]), np.array([.1, -.1]), np.array([-1., -1.]),
        np.array([1., 1.]), C, np.array([.2]), np.array([1.]), M,
        10., 1e-9, lp_method="highs",
    )
    assert result["success"]
    d = result["delta"]
    np.testing.assert_allclose(d, M @ result["basis_coordinates"], atol=1e-12)
    assert d[0] >= .2 - 1e-7
    assert d[0] == pytest.approx(d[1], abs=1e-10)
    expected = float(np.sum(np.abs(d - np.array([.1, -.1]))))
    assert result["objective"] == pytest.approx(expected, abs=2e-7)
    assert result["residual_to_raw_target_l1_mm"] == pytest.approx(expected, abs=2e-7)
    assert result["basis"]["output_l1_auxiliary_count"] == 2


def test_identity_basis_matches_full_output_solver_clearance_and_l1():
    A = sparse.csr_matrix([[1., 1.], [-1., 0.]])
    b = np.array([0., -.5])
    target = np.array([.1, -.1])
    lo, hi = np.array([-.5, -.5]), np.array([.5, .5])
    C = sparse.csr_matrix([[1., 0.], [0., 1.]])
    h, w = np.array([.2, .1]), np.array([.5, .5])
    full = mod.solve_sparse_l1_clearance(A, b, target, lo, hi, C, h, w, 10., 1e-9, lp_method="highs")
    reduced = mod.solve_sparse_l1_clearance_in_basis(
        A, b, target, lo, hi, C, h, w, sparse.eye(2, format="csr"),
        10., 1e-9, lp_method="highs",
    )
    assert full["success"] and reduced["success"]
    np.testing.assert_allclose(reduced["delta"], full["delta"], atol=3e-7, rtol=0.)
    assert reduced["normalized_slack_objective"] == pytest.approx(full["normalized_slack_objective"], abs=2e-7)
    assert reduced["objective"] == pytest.approx(full["objective"], abs=2e-7)


def test_basis_solver_fails_closed_on_transformed_zero_row_positive_rhs():
    M = sparse.csr_matrix((2, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="transformed zero rows with positive RHS"):
        mod.solve_sparse_l1_clearance_in_basis(
            sparse.csr_matrix([[1., 0.]]), np.array([.01]), np.zeros(2),
            -np.ones(2), np.ones(2), sparse.csr_matrix((0, 2)),
            np.empty(0), np.empty(0), M, 1.,
        )


def test_basis_solver_fails_closed_when_output_bounds_are_not_representable():
    M = sparse.csr_matrix((2, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="transformed zero rows with positive RHS"):
        mod.solve_sparse_l1_clearance_in_basis(
            sparse.csr_matrix((0, 2)), np.empty(0), np.zeros(2),
            np.array([.01, -1.]), np.ones(2), sparse.csr_matrix((0, 2)),
            np.empty(0), np.empty(0), M, 1.,
        )

def test_point_depth_rows_map_global_node_to_hard_owner_rows_in_mm():
    nodes = np.array([4, 2], dtype=np.int64)
    normals = np.array([[0., 1., 0.], [0., 0., -1.]], dtype=np.float64)
    lower_m = np.array([-0.001, 0.0], dtype=np.float64)
    typed = np.array([[4, 10, 11, 12], [2, 13, 14, 15]], dtype=np.int64)
    rows, receipt = mod.build_output_node_direction_rows(
        6, np.array([0, 2, 4, 5], dtype=np.int64), nodes, normals, lower_m, typed
    )
    assert receipt["row_count"] == 2 and receipt["units_output"] == "millimetres"
    assert [row["nodes"].tolist() for row in rows] == [[4], [2]]
    np.testing.assert_array_equal(rows[0]["coefficients"], [[0., 1., 0.]])
    assert rows[0]["lower"] == -1.0
    assert rows[1]["lower"] == 0.0
    np.testing.assert_array_equal(rows[1]["typed_ids"], typed[1])


def test_point_depth_row_mapper_rejects_target_outside_support():
    with pytest.raises(ValueError, match="outside output support"):
        mod.build_output_node_direction_rows(
            4, np.array([0, 1], dtype=np.int64), np.array([3], dtype=np.int64),
            np.array([[1., 0., 0.]], dtype=np.float64), np.array([0.], dtype=np.float64)
        )


@pytest.mark.parametrize("normal", [
    np.array([[2., 0., 0.]], dtype=np.float64),
    np.array([[np.nan, 0., 0.]], dtype=np.float64),
])
def test_point_depth_row_mapper_rejects_invalid_normals(normal):
    with pytest.raises(ValueError):
        mod.build_output_node_direction_rows(
            2, np.array([0], dtype=np.int64), np.array([0], dtype=np.int64),
            normal, np.array([0.], dtype=np.float64)
        )


def test_point_depth_row_mapper_rejects_malformed_or_duplicate_typed_ids():
    args = (3, np.array([0, 1], dtype=np.int64), np.array([0, 1], dtype=np.int64),
            np.array([[1., 0., 0.], [1., 0., 0.]], dtype=np.float64),
            np.array([0., 0.], dtype=np.float64))
    with pytest.raises(ValueError, match="typed IDs"):
        mod.build_output_node_direction_rows(*args, np.array([[1, 2, 3, 4]], dtype=np.int64))
    duplicated = np.array([[0, 2, 3, 4], [1, 2, 3, 4]], dtype=np.int64)
    # Different node IDs are distinct witnesses and are allowed.
    rows, _ = mod.build_output_node_direction_rows(*args, duplicated)
    assert len(rows) == 2
    duplicated = np.array([[0, 2, 3, 4], [0, 2, 3, 4]], dtype=np.int64)
    args_dup = (3, np.array([0, 1], dtype=np.int64), np.array([0, 0], dtype=np.int64),
                np.array([[1., 0., 0.], [1., 0., 0.]], dtype=np.float64),
                np.array([0., 0.], dtype=np.float64))
    with pytest.raises(ValueError, match="duplicate"):
        mod.build_output_node_direction_rows(*args_dup, duplicated)


def test_frozen_357_actual491_payload_pins_and_maps_to_hard_rows():
    root = Path("/Users/numi/Work/numi-human-connected-surface-069")
    rows_path = root / "D/point-depth-barriers-357/actual-491-v2/point-depth-rows.npz"
    report_path = root / "D/point-depth-barriers-357/actual-491-v2/report.json"
    report = mod._load_json(report_path)
    candidate_path = Path(report["candidate_npz"])
    owner_path = Path(report["builder_path"])
    with np.load(candidate_path, allow_pickle=False) as candidate:
        points = np.asarray(candidate["points_world_m"], dtype=np.float32).copy()
    with np.load(rows_path, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]).copy() for key in z.files}
    checked = mod.validate_point_depth_rows_payload(
        arrays, report, base_candidate_sha256=mod.sha(candidate_path), base_points=points,
        rows_npz_path=rows_path, owner_path=owner_path, candidate_path=candidate_path,
    )
    rows, receipt = mod.build_output_node_direction_rows(
        len(points), np.arange(len(points), dtype=np.int64), checked[0], checked[1],
        checked[2], checked[3],
    )
    assert len(rows) == 491 and receipt["support_membership"].startswith("all point-depth")
    assert report["inside_row_count"] == 489 and report["outside_row_count"] == 2
    assert all(row["kind"] == "point_depth_nonregression" for row in rows)
    assert all(row["lower"] <= 1e-7 for row in rows)
    bad = dict(report)
    bad["candidate_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="candidate"):
        mod.validate_point_depth_rows_payload(
            arrays, bad, base_candidate_sha256=mod.sha(candidate_path), base_points=points,
            rows_npz_path=rows_path, owner_path=owner_path, candidate_path=candidate_path,
        )


def test_successful_basis_solver_report_is_json_safe_and_hashes_every_lp_vector():
    import hashlib
    import json

    M = sparse.csr_matrix([[1.0], [1.0]])
    A = sparse.csr_matrix([[1.0, 1.0]])
    C = sparse.csr_matrix([[1.0, 0.0]])
    solved = mod.solve_sparse_l1_clearance_in_basis(
        A, np.array([0.0]), np.array([0.1, -0.1]), np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]), C, np.array([0.2]), np.array([1.0]), M,
        10.0, 1e-9, lp_method="highs",
    )
    assert solved["success"] is True
    assert isinstance(solved["basis_coordinates"], np.ndarray)
    receipt = mod.solver_report_receipt(solved)
    json.dumps({"solver": receipt}, sort_keys=True)
    for key in ("delta", "normalized_slack", "basis_coordinates"):
        assert key not in receipt
        raw = np.ascontiguousarray(solved[key], dtype=np.float64)
        summary = receipt[f"{key}_receipt"]
        assert summary["shape"] == list(raw.shape)
        assert summary["count"] == raw.size
        assert summary["sha256"] == hashlib.sha256(raw.tobytes(order="C")).hexdigest()
        assert summary["all_finite"] is True


def test_failure_receipt_drops_reduced_coordinate_vectors():
    receipt = mod.solver_failure_receipt({
        "success": False, "delta": np.array([1.0]),
        "normalized_slack": np.array([0.5]), "basis_coordinates": np.array([3.0]),
        "status": 2,
    })
    assert "delta" not in receipt
    assert "normalized_slack" not in receipt
    assert "basis_coordinates" not in receipt
    assert receipt["normalized_slack_receipt"]["count"] == 1


def test_scoped_point_depth_rows_do_not_expand_control_scope():
    support=np.array([1,2,3],dtype=np.int64)
    fixed=np.array([3],dtype=np.int64)
    nodes=np.array([0,1,3],dtype=np.int64)
    normals=np.array([[1.,0,0],[0,1.,0],[0,0,1.]],dtype=np.float64)
    lower=np.array([-.002,0.,0.],dtype=np.float64)
    typed=np.array([[0,10,20,30],[1,11,21,31],[3,12,22,32]],dtype=np.int64)
    rows,receipt=mod.build_scoped_output_node_direction_rows(
        5,support,fixed,nodes,normals,lower,typed)
    assert [r['nodes'].tolist() for r in rows]==[[1]]
    assert receipt['input_row_count']==3
    assert receipt['active_row_count']==1 and receipt['passive_zero_row_count']==2
    assert receipt['active_input_indices']==[1]
    assert receipt['passive_zero_rows'][0]['reason']=='outside_selected_output_scope'
    assert receipt['passive_zero_rows'][1]['reason']=='fixed_row'
    assert 'do not enlarge' in receipt['control_scope_policy']


def test_scoped_point_depth_rows_reject_positive_rhs_on_passive_row():
    with pytest.raises(ValueError,match='passive point-depth rows require positive motion'):
        mod.build_scoped_output_node_direction_rows(
            4,np.array([1],dtype=np.int64),np.array([],dtype=np.int64),
            np.array([2],dtype=np.int64),np.array([[1.,0.,0.]],dtype=np.float64),
            np.array([1e-6],dtype=np.float64),np.array([[2,10,20,30]],dtype=np.int64))
