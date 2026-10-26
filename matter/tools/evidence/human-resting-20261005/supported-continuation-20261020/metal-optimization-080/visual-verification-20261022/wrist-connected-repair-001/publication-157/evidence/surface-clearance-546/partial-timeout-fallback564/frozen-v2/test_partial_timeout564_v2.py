from pathlib import Path
import hashlib
import importlib.util
import sys

import numpy as np
import pytest
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solver = load("partial_component564_test", "component_lp_partial_timeout564.py")
wrapper = load("partial_wrapper564_test", "run_joint_full_coordinate_fit564.py")


def _run(fake, *, timeout_fallback=True, rhs=None):
    calls = {"count": 0}

    def original(*args, **kwargs):
        ci = calls["count"]
        calls["count"] += 1
        if ci == 1:
            return fake(ci, *args, **kwargs)
        return {
            "success": True, "status": 0, "message": "Optimal",
            "delta": np.asarray([0.1 if ci == 0 else 0.3]),
            "basis_coordinates": np.asarray([0.1 if ci == 0 else 0.3]),
            "normalized_slack": np.asarray([0.9 if ci == 0 else 0.7]),
            "stage1": {"objective": 0.09 if ci == 0 else 0.07},
            "normalized_slack_objective": 0.09 if ci == 0 else 0.07,
            "wall_seconds": 0.01,
        }

    A = sparse.eye(3, format="csr")
    b = np.full(3, -0.5) if rhs is None else np.asarray(rhs, dtype=float)
    C = sparse.eye(3, format="csr")
    h = np.ones(3)
    weights = np.full(3, 0.1)
    M = sparse.eye(3, format="csr")
    result = solver.solve_by_components(
        original, A, b, np.asarray([0.1, 0.2, 0.3]),
        np.full(3, -1.0), np.full(3, 1.0), C, h, weights, M,
        time_limit_s=2.0, retained_gain_fraction=0.95,
        max_total_wall_s=120.0,
        allow_partial_timeout_zero_fallback=timeout_fallback,
    )
    return result, calls


def test_single_timeout_keeps_solved_components_and_zeroes_only_failed_one():
    def timeout(ci, *args, **kwargs):
        return {"success": False, "status": 1, "message": "Time limit reached",
                "stage1": {"success": False, "status": 1}, "wall_seconds": 2.0}

    result, calls = _run(timeout)
    assert calls["count"] == 3
    assert result["success"] is False
    assert result["status"] == 1
    assert "component 1 LP failed: Time limit reached" == result["message"]
    partial = result["_partial_timeout_direction_564"]
    np.testing.assert_allclose(partial["delta"], [0.1, 0.0, 0.3])
    np.testing.assert_allclose(partial["basis_coordinates"], [0.1, 0.0, 0.3])
    np.testing.assert_allclose(partial["normalized_slack"], [0.9, 1.0, 0.7])
    cert = partial["feasibility_certificate"]
    assert cert["pass"] is True
    assert cert["minimum_hard_residual"] >= -2e-7
    assert cert["minimum_target_residual"] >= -2e-7
    decomp = result["component_decomposition_475"]
    assert decomp["partial_timeout_zero_fallback"]["global_lp_converged"] is False
    assert decomp["partial_timeout_zero_fallback"]["global_stage1_optimum_known"] is False
    assert decomp["global_stage2_budget_pass"] is None
    assert decomp["global_weighted_slack"] == pytest.approx(0.26)


def test_non_timeout_failure_is_not_zero_fallback():
    def infeasible(ci, *args, **kwargs):
        return {"success": False, "status": 1, "message": "Iteration limit reached"}

    result, _ = _run(infeasible)
    assert result["success"] is False
    assert "_partial_timeout_direction_564" not in result


def test_timeout_with_infeasible_zero_hard_rows_is_not_fallback():
    def timeout(ci, *args, **kwargs):
        return {"success": False, "status": 1, "message": "Time limit reached"}

    result, _ = _run(timeout, rhs=[-0.5, 0.1, -0.5])
    assert result["success"] is False
    assert "_partial_timeout_direction_564" not in result
    assert result["component_decomposition_475"]["component_results"][1]["partial_zero_fallback_eligible"] is False


def test_opt_out_preserves_original_timeout_failure():
    def timeout(ci, *args, **kwargs):
        return {"success": False, "status": 1, "message": "Time limit reached"}

    result, _ = _run(timeout, timeout_fallback=False)
    assert result["success"] is False
    assert "_partial_timeout_direction_564" not in result


def _basis_token(tmp_path):
    base = tmp_path / "base.npz"
    points = np.zeros((3, 3), dtype=np.float32)
    np.savez(base, points_world_m=points, source_O_faces=np.asarray([[0, 1, 2]], dtype=np.int64), typed_coarse_anchor_output_rows_018=np.asarray([0], dtype=np.int64))
    Q = sparse.csr_matrix(np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.float64))
    csr = Q
    return {
        "base_candidate": str(base),
        "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "capture_helper_sha256": "fa7df8771a37200c4ed30e61d92ebf1b6227eed9da8a5ff45ab54b4ff299975e",
        "base_points_world_m_f32": points,
        "output_scope_rows": np.asarray([0, 1, 2], dtype=np.int64),
        "anchor_rows": np.asarray([0], dtype=np.int64),
        "control_parent_rows": np.asarray([1, 2], dtype=np.int64),
        "control_scope_positions": np.asarray([1, 2], dtype=np.int64),
        "anchor_scope_positions": np.asarray([0], dtype=np.int64),
        "basis_csr": {"shape": list(csr.shape), "data": csr.data.copy(),
                      "indices": csr.indices.copy(), "indptr": csr.indptr.copy(),
                      "representation": "test"},
        "output_dimension": 9,
        "control_dimension": 6,
        "units": {"base_points": "meters_float32", "solver_delta": "millimeters"},
    }


def test_partial_capture_preserves_failed_solver_status_and_basis_mapping(tmp_path):
    token = _basis_token(tmp_path)
    partial = {
        "delta": np.asarray([0, 0, 0, 0.25, 0, 0, 0.5, 0, 0], dtype=np.float64),
        "basis_coordinates": np.asarray([0.25, 0, 0, 0.5, 0, 0], dtype=np.float64),
        "partial_receipt": {"timed_out_components": [7], "zeroed_components": [7]},
        "feasibility_certificate": {"pass": True, "actual_weighted_slack": 0.3},
    }
    original = {"success": False, "status": 1, "message": "component 7 timed out"}
    records = []
    wrapper.capture_partial_solver_direction(
        original, partial, token, None, records, lambda _: {"fit_entry_path": "test"}
    )
    assert original["success"] is False
    assert records[0]["status"] == "PARTIAL_FEASIBLE_TIMEOUT_DIRECTION_CAPTURED_UNADMITTED"
    assert records[0]["solver_receipt"]["success"] is False
    assert records[0]["solver_status"] == 1
    np.testing.assert_array_equal(records[0]["delta_mm"], partial["delta"])
    assert records[0]["basis_coordinate_reconstruction_residual_max_abs_mm"] == 0.0


def test_wrapper_removes_private_arrays_before_returning_failure_to_owner(tmp_path):
    token = _basis_token(tmp_path)
    partial = {
        "delta": np.zeros(9), "basis_coordinates": np.zeros(6),
        "partial_receipt": {"timed_out_components": [0], "zeroed_components": [0]},
        "feasibility_certificate": {"pass": True},
    }
    base_failure = {"success": False, "status": 1, "message": "Time limit reached"}
    base_failure["_partial_timeout_direction_564"] = partial
    seen = {}

    def original(*args, **kwargs):
        seen.update(kwargs)
        return base_failure

    records = []
    wrapped = wrapper.wrap_solver_for_raw_capture(
        original, None, [token], records,
        lambda _: {"receipt": "test"}, partial_timeout_opt_in=True,
    )
    result = wrapped()
    assert seen["allow_partial_timeout_zero_fallback"] is True
    assert result is base_failure
    assert result["success"] is False and result["status"] == 1
    assert "_partial_timeout_direction_564" not in result
    assert len(records) == 1


def test_partial_capture_finalizes_549_compatible_unadmitted_proposal(tmp_path):
    token = _basis_token(tmp_path)
    base = Path(token["base_candidate"])
    partial = {
        "delta": np.asarray([0, 0, 0, 0.25, 0, 0, 0.5, 0, 0], dtype=np.float64),
        "basis_coordinates": np.asarray([0.25, 0, 0, 0.5, 0, 0], dtype=np.float64),
        "partial_receipt": {"timed_out_components": [7], "zeroed_components": [7], "global_lp_converged": False},
        "feasibility_certificate": {"pass": True, "actual_weighted_slack": 0.3},
    }
    original = {"success": False, "status": 1, "message": "component 7 timed out"}
    records = []
    wrapper.capture_partial_solver_direction(
        original, partial, token, wrapper._capture_module, records, lambda _: {"pending": True}
    )
    fit_entry = Path(wrapper.__file__).resolve()
    fit_receipt = {
        "fit_entry_path": str(fit_entry),
        "fit_entry_sha256": hashlib.sha256(fit_entry.read_bytes()).hexdigest(),
        "objective_contract": "feasible partial construction direction; no global optimum claim",
        "goal_labels": ["face", "point"],
        "matched_input_pins": {str(base): hashlib.sha256(base.read_bytes()).hexdigest()},
        "row_receipt": {"all_original_rows_retained": True},
        "workload_receipt": {"component_count": 3},
        "policy_receipt": {"construction_only": True},
        "delta_semantics": "absolute_base_relative_millimeters",
    }
    records[0]["fit_receipt"] = fit_receipt
    out = tmp_path / "raw-joint-capture-563"
    summary_path, summary = wrapper._capture_module.finalize_capture(
        base, out, records, fit_receipt=fit_receipt
    )
    proposal = out / "raw-direction-001"
    report = __import__("json").loads((proposal / "report.json").read_text())
    assert summary["successful_delta_count"] == 1
    assert report["status"] == "RAW_JOINT_LP_DIRECTION_FOR_EXISTING_549_PROJECTION"
    assert report["solver_receipt"]["success"] is False
    assert report["solver_receipt"]["global_lp_converged"] is False
    assert report["source_objective_receipt"]["objective_contract"].startswith("feasible partial")
    assert report["base_sha256"] == token["base_sha256"]
    assert report["candidate_sha256"] == hashlib.sha256((proposal / "candidate.npz").read_bytes()).hexdigest()
    with np.load(proposal / "candidate.npz", allow_pickle=False) as a:
        assert a["points_world_m"][0].tobytes() == np.zeros(3, dtype=np.float32).tobytes()
        assert a["points_world_m"][1, 0] == np.float32(0.00025)
        assert a["points_world_m"][2, 0] == np.float32(0.0005)


def test_wrapper_pins_partial_helper_source():
    digest = hashlib.sha256((HERE / "component_lp_partial_timeout564.py").read_bytes()).hexdigest()
    assert digest == wrapper.EXPECTED_ACTIVE_SHA


def test_wrapper_requires_explicit_partial_fallback_opt_in(monkeypatch):
    monkeypatch.setattr(wrapper.sys, "argv", [str(wrapper.__file__)])
    with pytest.raises(ValueError, match="explicit opt-in flag required exactly once"):
        wrapper.main()


def test_multiple_timeouts_zero_only_those_components_and_continue_later():
    calls = {"count": 0}

    def original(A, b, target, lower, upper, C, h, w, M, limit, **kwargs):
        ci = calls["count"]
        calls["count"] += 1
        if ci in {1, 3}:
            return {"success": False, "status": 1, "message": "Time limit reached",
                    "stage1": {"success": False, "status": 1}, "wall_seconds": 2.0}
        value = 0.05 * (ci + 1)
        slack = 1.0 - value
        return {
            "success": True, "status": 0, "message": "Optimal",
            "delta": np.asarray([value]), "basis_coordinates": np.asarray([value]),
            "normalized_slack": np.asarray([slack]),
            "stage1": {"objective": 0.1 * slack},
            "normalized_slack_objective": 0.1 * slack, "wall_seconds": 0.01,
        }

    n = 5
    result = solver.solve_by_components(
        original, sparse.eye(n, format="csr"), np.full(n, -0.5),
        np.asarray([0.05 * (i + 1) for i in range(n)]),
        np.full(n, -1.0), np.full(n, 1.0), sparse.eye(n, format="csr"),
        np.ones(n), np.full(n, 0.1), sparse.eye(n, format="csr"),
        time_limit_s=2.0, retained_gain_fraction=0.95, max_total_wall_s=120.0,
        allow_partial_timeout_zero_fallback=True,
    )
    assert calls["count"] == n
    assert result["success"] is False and result["status"] == 1
    partial = result["_partial_timeout_direction_564"]
    np.testing.assert_allclose(partial["delta"], [0.05, 0, 0.15, 0, 0.25])
    np.testing.assert_allclose(partial["basis_coordinates"], partial["delta"])
    receipt = result["component_decomposition_475"]["partial_timeout_zero_fallback"]
    assert receipt["timed_out_components"] == [1, 3]
    assert receipt["zeroed_components"] == [1, 3]
    assert [x["component"] for x in receipt["timeout_failures"]] == [1, 3]
    assert receipt["first_original_failure"]["component"] == 1
    assert receipt["global_lp_converged"] is False
    assert receipt["retained_gain_budget_claimed"] is False


def test_total_wall_budget_zeros_only_feasible_unprocessed_components(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(solver.time, "monotonic", lambda: clock["now"])
    calls = {"count": 0}

    def original(A, b, target, lower, upper, C, h, w, M, limit, **kwargs):
        calls["count"] += 1
        value = 0.1
        slack = 0.9
        clock["now"] = 1.0
        return {
            "success": True, "status": 0, "message": "Optimal",
            "delta": np.asarray([value]), "basis_coordinates": np.asarray([value]),
            "normalized_slack": np.asarray([slack]),
            "stage1": {"objective": 0.09}, "normalized_slack_objective": 0.09,
            "wall_seconds": 0.01,
        }

    n = 3
    result = solver.solve_by_components(
        original, sparse.eye(n, format="csr"), np.full(n, -0.5),
        np.asarray([0.1, 0.2, 0.3]), np.full(n, -1.0), np.full(n, 1.0),
        sparse.eye(n, format="csr"), np.ones(n), np.full(n, 0.1),
        sparse.eye(n, format="csr"), time_limit_s=2.0, retained_gain_fraction=0.95,
        max_total_wall_s=0.5, allow_partial_timeout_zero_fallback=True,
    )
    assert calls["count"] == 1
    assert result["success"] is False and result["status"] == 1
    assert result["message"].startswith("total component LP time budget exhausted")
    partial = result["_partial_timeout_direction_564"]
    np.testing.assert_allclose(partial["delta"], [0.1, 0, 0])
    receipt = result["component_decomposition_475"]["partial_timeout_zero_fallback"]
    assert receipt["timed_out_components"] == []
    assert receipt["budget_exhausted_components"] == [1, 2]
    assert receipt["zeroed_components"] == [1, 2]
    assert receipt["global_output_bounds_pass"] is True
    assert receipt["global_soft_target_slack_rows_pass"] is True
    assert receipt["global_lp_converged"] is False
