from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
from scipy import sparse

HERE=Path(__file__).resolve().parent
OWNER=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py")
spec=importlib.util.spec_from_file_location("owner408_component_test", OWNER)
owner=importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
spec2=importlib.util.spec_from_file_location("component475_test", HERE/"component_lp_decomposition_475.py")
mod=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(mod)

def fixture():
    # Two independent 2-control components, each with one hard coupling and
    # two target rows. Zero move satisfies all existing hard rows/bounds.
    A=sparse.csr_matrix([[1.,-1.,0.,0.],[0.,0.,1.,-1.]])
    b=np.array([-.2,-.2])
    C=sparse.eye(4,format="csr")
    h=np.array([.2,.1,.2,.1])
    w=np.full(4,.25)
    M=sparse.eye(4,format="csr")
    t=np.array([.25,-.1,.2,-.15])
    lo=np.full(4,-.5); hi=np.full(4,.5)
    return A,b,t,lo,hi,C,h,w,M

def run(A,b,t,lo,hi,C,h,w,M,**kw):
    return mod.solve_by_components(owner.solve_sparse_l1_clearance_in_basis,A,b,t,lo,hi,C,h,w,M,
                time_limit_s=3., slack_tolerance=1e-9, lp_method="highs", max_total_wall_s=20.,**kw)

def test_disconnected_component_stage1_and_stage2_parity():
    A,b,t,lo,hi,C,h,w,M=fixture()
    expected=owner.solve_sparse_l1_clearance_in_basis(A,b,t,lo,hi,C,h,w,M,3.,slack_tolerance=1e-9,lp_method="highs",retained_gain_fraction=1.0)
    actual=run(A,b,t,lo,hi,C,h,w,M,retained_gain_fraction=1.0)
    assert expected["success"] and actual["success"]
    assert actual["component_decomposition_475"]["component_count"]==2
    assert abs(actual["stage1"]["objective"]-expected["stage1"]["objective"])<2e-7
    assert abs(actual["objective"]-expected["objective"])<2e-6
    assert np.allclose(A@actual["delta"],A@expected["delta"],atol=2e-7)
    assert actual["component_decomposition_475"]["global_original_space_certificate_pass"]

def test_zero_cancellation_rows_become_fixed_slack_and_output_constants():
    A=sparse.csr_matrix((0,2)); b=np.empty(0); C=sparse.csr_matrix([[1.,1.]])
    h=np.array([1.]); w=np.array([1.]); M=sparse.csr_matrix([[1.],[-1.]])
    t=np.array([.2,.2]); lo=np.array([-.5,-.5]); hi=np.array([.5,.5])
    actual=run(A,b,t,lo,hi,C,h,w,M,retained_gain_fraction=.95)
    assert actual["success"]
    r=actual["component_decomposition_475"]
    assert r["component_count"]==1 and r["clearance_rows_fixed_zero_count"]==1
    assert np.isclose(actual["normalized_slack"][0],1.0)
    assert np.allclose(C@actual["delta"],[0.0])

def test_component_union_joins_hard_row_crossing_independent_controls():
    A=sparse.csr_matrix([[1.,-1.]])
    b=np.array([-.2]); C=sparse.csr_matrix([[1.,0.],[0.,1.]])
    h=np.array([.2,.1]); w=np.array([.5,.5]); M=sparse.eye(2,format="csr")
    t=np.array([.1,.2]); lo=np.full(2,-.4); hi=np.full(2,.4)
    r=run(A,b,t,lo,hi,C,h,w,M,retained_gain_fraction=1.0)
    assert r["success"] and r["component_decomposition_475"]["component_count"]==1
    assert float((A@r["delta"]-b).min())>=-2e-7

def test_zero_mapped_hard_row_positive_rhs_fails_closed():
    A=sparse.csr_matrix([[1.,1.]])
    b=np.array([.01]); C=sparse.csr_matrix([[1.,0.]])
    h=np.array([.2]); w=np.array([1.]); M=sparse.csr_matrix([[1.],[-1.]])
    t=np.zeros(2); lo=np.full(2,-.5); hi=np.full(2,.5)
    r=run(A,b,t,lo,hi,C,h,w,M)
    assert not r["success"] and "zero reduced coefficients" in r["message"]

def test_output_row_with_zero_map_is_counted_and_bounds_checked():
    A=sparse.csr_matrix([[1.,0.]])
    b=np.array([-.1]); C=sparse.csr_matrix([[1.,0.]])
    h=np.array([.2]); w=np.array([1.]); M=sparse.csr_matrix([[1.],[0.]])
    t=np.array([.1,.3]); lo=np.array([-.5,-.5]); hi=np.array([.5,.5])
    r=run(A,b,t,lo,hi,C,h,w,M,retained_gain_fraction=1.0)
    assert r["success"] and r["component_decomposition_475"]["output_rows_fixed_zero_count"]==1
    assert r["delta"][1]==0.0
    assert r["objective"] >= .3

def test_global_retained_gain_budget_equals_component_budget_sum():
    A,b,t,lo,hi,C,h,w,M=fixture()
    r=run(A,b,t,lo,hi,C,h,w,M,retained_gain_fraction=.95)
    assert r["success"]
    local=sum(x["stage2_slack_budget_policy"]["stage2_slack_budget"] for x in r["component_decomposition_475"]["component_results"])
    assert abs(local-r["component_decomposition_475"]["global_stage2_slack_budget"])<2e-7
    assert r["component_decomposition_475"]["global_stage2_budget_pass"]

def test_nonfinite_sparse_input_rejected():
    A,b,t,lo,hi,C,h,w,M=fixture(); C=C.copy(); C.data[0]=np.nan
    try: run(A,b,t,lo,hi,C,h,w,M)
    except ValueError as e: assert "nonfinite" in str(e)
    else: raise AssertionError("nonfinite sparse input was accepted")

def test_certified_stage1_fallback_does_not_claim_stage2_success():
    A=sparse.csr_matrix((0,2)); b=np.empty(0); C=sparse.eye(2,format="csr")
    h=np.array([.2,.2]); w=np.array([.5,.5]); M=sparse.eye(2,format="csr")
    t=np.array([.1,-.1]); lo=np.full(2,-.5); hi=np.full(2,.5)
    def fallback_solver(A,b,t,lo,hi,C,h,w,M,time_limit_s,**kwargs):
        return {"success": True, "status": 0, "message": "stage1 certificate retained",
                "nit": 1, "objective": None, "wall_seconds": .01,
                "delta": np.zeros(len(t)), "basis_coordinates": np.zeros(M.shape[1]),
                "normalized_slack": np.ones(len(h)), "stage1": {"objective": float(w.sum())},
                "normalized_slack_objective": float(w.sum()), "fallback_used": True,
                "stage2_slack_budget_policy": {"stage2_slack_budget": float(w.sum())}}
    r=mod.solve_by_components(fallback_solver,A,b,t,lo,hi,C,h,w,M,3.,
                retained_gain_fraction=.95,max_total_wall_s=10.)
    assert r["success"]
    assert r["fallback_used"]
    assert not r["stage2"]["success"]
    assert r["stage2"]["objective"] is None
    assert r["component_decomposition_475"]["global_original_space_certificate_pass"]
