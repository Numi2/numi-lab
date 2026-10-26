from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
from scipy import sparse

HERE=Path(__file__).resolve().parent
D=Path("/Users/numi/Work/numi-human-connected-surface-069/D")
OWNER=D/"surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
SOFT=D/"D/soft-target-nonregression-470/soft_target_nonregression_470.py"

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

owner=load("owner408_for_485_test",OWNER)
mod=load("component485_for_test",HERE/"component_lp_decomposition_485.py")
soft=load("soft470_for_485_test",SOFT)

def test_default_matches_475_hard_soft_nonregression_behavior():
    A=sparse.csr_matrix([[-1.,0.]])
    b=np.array([-.05])
    C=sparse.csr_matrix([[1.,0.],[-1.,0.]])
    h=np.array([.2,.2]); w=np.array([.5,.5]); M=sparse.eye(2,format="csr")
    t=np.array([.3,0.]); lo=np.full(2,-1.); hi=np.full(2,1.)
    baseline=owner.solve_sparse_l1_clearance_in_basis(A,b,t,lo,hi,C,h,w,M,3.,retained_gain_fraction=1.)
    decomposed=mod.solve_by_components(owner.solve_sparse_l1_clearance_in_basis,A,b,t,lo,hi,C,h,w,M,3.,
                                      retained_gain_fraction=1.,max_total_wall_s=15.)
    assert baseline["success"] and decomposed["success"]
    assert np.allclose(decomposed["delta"],baseline["delta"],atol=2e-7)
    assert decomposed["minimum_soft_target_progress_mm"] >= -2e-7
    assert decomposed["soft_target_no_regression_rows_enforced"]
    assert decomposed["component_decomposition_475"]["global_original_space_certificate_pass"]

def test_470_marker_allows_soft_regression_but_preserves_hard_cap_and_slack():
    soft_owner=load("owner408_for_470_test",OWNER)
    soft.install_solver(soft_owner)
    solver=soft_owner.solve_sparse_l1_clearance_in_basis
    assert getattr(solver,"__soft_target_no_regression_470__",False)
    A=sparse.csr_matrix([[-1.,0.]])
    b=np.array([-.05])
    C=sparse.csr_matrix([[1.,0.],[-1.,0.]])
    h=np.array([.2,.2]); w=np.array([.5,.5]); M=sparse.eye(2,format="csr")
    t=np.array([.3,0.]); lo=np.full(2,-1.); hi=np.full(2,1.)
    result=mod.solve_by_components(solver,A,b,t,lo,hi,C,h,w,M,3.,
                    retained_gain_fraction=1.,allow_soft_target_regression=True,max_total_wall_s=15.)
    assert result["success"]
    assert result["soft_target_regression_allowed"]
    assert not result["soft_target_no_regression_rows_enforced"]
    assert result["negative_soft_target_progress_rows"] == 1
    assert result["minimum_soft_target_progress_mm"] < -0.049
    assert float(np.min(A@result["delta"]-b)) >= -2e-7
    assert result["minimum_target_inequality_slack_mm"] >= -2e-7
    assert result["component_decomposition_475"]["global_stage2_budget_pass"]

def test_soft_regression_opt_in_fails_without_exact_470_marker():
    A=sparse.csr_matrix((0,2)); b=np.empty(0)
    C=sparse.eye(2,format="csr"); h=np.ones(2); w=np.full(2,.5)
    M=sparse.eye(2,format="csr"); t=np.zeros(2); lo=np.full(2,-1.); hi=np.full(2,1.)
    clean_owner=load("clean_owner408_no470_marker",OWNER)
    try:
        mod.solve_by_components(clean_owner.solve_sparse_l1_clearance_in_basis,A,b,t,lo,hi,C,h,w,M,3.,
                                allow_soft_target_regression=True)
    except ValueError as exc:
        assert "470 solver marker" in str(exc)
    else:
        raise AssertionError("soft regression ran without the 470 marker")

def test_malformed_projected_dimensions_fail_closed():
    A=sparse.csr_matrix((0,2)); b=np.empty(0)
    C=sparse.csr_matrix([[1.]])
    h=np.ones(1); w=np.ones(1); M=sparse.eye(2,format="csr")
    t=np.zeros(2); lo=np.full(2,-1.); hi=np.full(2,1.)
    try:
        mod.solve_by_components(owner.solve_sparse_l1_clearance_in_basis,A,b,t,lo,hi,C,h,w,M,3.)
    except ValueError as exc:
        assert "dimensions" in str(exc)
    else:
        raise AssertionError("malformed component dimensions were accepted")


def test_real_matched_461_source_and_input_preflight():
    runner=load("runner485_preflight_test",HERE/"run_matched_component_fit_485.py")
    source_receipt=runner.validate_source_pins()
    candidate=D/"outward-plane-backtrack-461/run-001/backtrack-373/pass-011/candidate.npz"
    audit=candidate.parent/"audit-159/report.json"
    face=D/"connected-continuation-prep476/run-001/surface-targets-335/targets.npz"
    point=D/"connected-continuation-prep476/run-001/point-targets-390/nearest-union-exits.npz"
    matched=runner.validate_matched_inputs(candidate,audit,face,point)
    assert len(source_receipt)==5
    assert matched["candidate"]["sha256"]==runner.EXPECTED_CANDIDATE
    assert matched["audit159"]["sha256"]==runner.EXPECTED_AUDIT
    assert matched["face_target_npz"]["sha256"]==runner.EXPECTED_FACE
    assert matched["point_target_npz"]["sha256"]==runner.EXPECTED_POINT
