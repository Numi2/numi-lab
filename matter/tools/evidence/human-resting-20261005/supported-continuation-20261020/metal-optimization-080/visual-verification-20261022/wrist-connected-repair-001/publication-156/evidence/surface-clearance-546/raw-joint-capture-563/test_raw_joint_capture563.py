from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("raw_joint_capture563_test", HERE / "raw_joint_capture563.py")
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _base(tmp_path):
    points = np.array([[0, 0, 0], [0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]], dtype=np.float32)
    path = tmp_path / "base.npz"
    np.savez_compressed(path, points_world_m=points,
                        source_O_faces=np.array([[0, 1, 2]], dtype=np.int32),
                        typed_coarse_anchor_output_rows_018=np.array([3], dtype=np.int32),
                        lineage=np.array([11, 12, 13, 14], dtype=np.int32))
    return path, points


def _receipt():
    here = Path(__file__).resolve()
    return {
        "fit_entry_path": str(here), "fit_entry_sha256": _sha(here),
        "objective_contract": "caller-declared joint point and face clearance LP objective",
        "goal_labels": ["point-depth", "fixed-bone-face-clearance"],
        "matched_input_pins": {"candidate": {"path": str(here), "sha256": _sha(here)}},
        "row_receipt": {"point_rows": 1245, "face_rows": 8198, "point_weight_mass": 0.5, "face_weight_mass": 0.5},
        "workload_receipt": {"budget_source": "component weighted allocation", "stages": 2},
        "policy_receipt": {"construction_trust_mm": 4.0, "source_cap_mm": 12.0, "area_floor": 0.1, "edge_gradient_floor_mm": 0.5},
        "delta_semantics": "absolute_base_relative_millimeters",
    }


def test_successful_joint_capture_writes_549_contract_and_exact_basis(tmp_path):
    base_path, base_points = _base(tmp_path)
    scope = np.array([0, 1, 2, 3], dtype=np.int64)
    anchors = np.array([3], dtype=np.int64)
    controls = np.array([0, 1, 2], dtype=np.int64)
    Q = sparse.csr_matrix(([1., 1., 1.], ([0, 1, 2], [0, 1, 2])), shape=(4, 3))
    token = mod.capture_basis(base_path, scope, anchors, controls, basis_matrix=Q)
    receipt = _receipt()
    delta = np.array([0.25, 0., 0., 0., 0.5, 0., -0.25, 0., 0., 0., 0., 0.])
    result = {"success": True, "status": 0, "message": "optimal", "objective": 1.25,
              "wall_seconds": 0.5, "delta": delta, "basis_coordinates": delta[:9].copy(),
              "component_decomposition_475": {"component_count": 1, "component_workload_weights": [9.0]}}
    records = []
    mod.capture_solver_return(result, token, fit_receipt=receipt, stage="joint_lp_raw_before_area_alpha", records=records)
    out = tmp_path / "capture"
    summary_path, summary = mod.finalize_capture(base_path, out, records, fit_receipt=receipt)
    assert summary["successful_delta_count"] == 1
    proposal = out / "raw-direction-001"
    report = json.loads((proposal / "report.json").read_text())
    required_549 = {"base_candidate", "base_sha256", "candidate_path", "candidate_sha256", "free_nodes"}
    assert required_549.issubset(report)
    assert report["source_objective_receipt"]["objective_contract"] == receipt["objective_contract"]
    assert report["source_objective_receipt"]["goal_labels"] == receipt["goal_labels"]
    assert report["solver_stage"] == "joint_lp_raw_before_area_alpha"
    assert report["solver_receipt"]["component_decomposition_475"]["component_workload_weights"] == [9.0]
    assert report["native_admission"] is False and report["accepted_physical_steps"] == 0
    with np.load(proposal / "candidate.npz", allow_pickle=False) as z:
        emitted = z["points_world_m"]
        assert z["lineage"].tobytes() == np.array([11, 12, 13, 14], dtype=np.int32).tobytes()
    assert emitted[3].tobytes() == base_points[3].tobytes()
    assert np.array_equal(emitted[:3], np.array([[0.00025, 0, 0], [0.001, 0.0005, 0], [-0.00025, 0.001, 0]], dtype=np.float32))
    with np.load(proposal / "basis-and-delta.npz", allow_pickle=False) as z:
        assert tuple(z["basis_shape"]) == (4, 3)
        assert np.array_equal(z["basis_indptr"], Q.indptr)
        assert np.array_equal(z["basis_indices"], Q.indices)
        assert np.array_equal(z["basis_data"], Q.data)
        assert np.array_equal(z["control_parent_rows"], controls)
        assert np.array_equal(z["delta_mm"], delta)
    assert summary_path.is_file()


def test_derived_identity_basis_maps_noncontiguous_controls(tmp_path):
    base_path, _ = _base(tmp_path)
    scope = np.array([0, 1, 2, 3], dtype=np.int64)
    anchors = np.array([3], dtype=np.int64)
    controls = np.array([2, 0, 1], dtype=np.int64)
    token = mod.capture_basis(base_path, scope, anchors, controls)
    csr = token["basis_csr"]
    Q = sparse.csr_matrix((csr["data"], csr["indices"], csr["indptr"]), shape=csr["shape"])
    expected = sparse.csr_matrix(([1., 1., 1.], ([2, 0, 1], [0, 1, 2])), shape=(4, 3))
    assert (Q != expected).nnz == 0


def test_unsuccessful_solver_is_reported_without_candidate(tmp_path):
    base_path, _ = _base(tmp_path)
    token = mod.capture_basis(base_path, [0, 1, 2, 3], [3], [0, 1, 2])
    records = []
    result = {"success": False, "status": 1, "message": "time limit"}
    mod.capture_solver_return(result, token, fit_receipt=_receipt(), records=records)
    _, summary = mod.finalize_capture(base_path, tmp_path / "failed", records, fit_receipt=_receipt())
    assert summary["successful_delta_count"] == 0
    assert summary["status"] == "NO_SUCCESSFUL_JOINT_LP_DELTA_CAPTURED"
    assert not list((tmp_path / "failed").glob("raw-direction-*/candidate.npz"))
    assert summary["solver_returns"][0]["status"] == "SOLVER_DID_NOT_RETURN_SUCCESS"


def test_invalid_provenance_or_delta_is_rejected(tmp_path):
    base_path, _ = _base(tmp_path)
    token = mod.capture_basis(base_path, [0, 1, 2, 3], [3], [0, 1, 2])
    bad = _receipt()
    bad["objective_contract"] = ""
    with np.testing.assert_raises(ValueError):
        mod.capture_solver_return({"success": True, "delta": np.zeros(12)}, token, fit_receipt=bad, records=[])
    records = []
    mod.capture_solver_return({"success": True, "status": 0, "delta": np.zeros(11)}, token,
                              fit_receipt=_receipt(), records=records)
    assert records[0]["delta_captured"] is False
    assert records[0]["status"] == "SUCCESSFUL_SOLVER_RETURN_WITH_MALFORMED_DELTA"


def test_base_and_anchor_mappings_are_bound(tmp_path):
    base_path, _ = _base(tmp_path)
    with np.testing.assert_raises(ValueError):
        mod.capture_basis(base_path, [0, 1, 2, 3], [2], [0, 1, 3])
    token = mod.capture_basis(base_path, [0, 1, 2, 3], [3], [0, 1, 2])
    records = []
    bad = np.zeros(12)
    bad[9] = 0.1
    result = {"success": True, "status": 0, "delta": bad}
    mod.capture_solver_return(result, token, fit_receipt=_receipt(), records=records)
    assert records[0]["delta_captured"] is False
    assert records[0]["status"] == "SUCCESSFUL_SOLVER_RETURN_MOVES_FIXED_ANCHOR_BASIS_ROWS"
    _, summary = mod.finalize_capture(base_path, tmp_path / "anchor", records, fit_receipt=_receipt())
    assert summary["successful_delta_count"] == 0
    assert not list((tmp_path / "anchor").glob("raw-direction-*/candidate.npz"))
