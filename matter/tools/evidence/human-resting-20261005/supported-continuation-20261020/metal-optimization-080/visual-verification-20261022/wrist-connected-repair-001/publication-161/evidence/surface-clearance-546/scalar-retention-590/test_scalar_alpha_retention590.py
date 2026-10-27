from pathlib import Path
import importlib.util
import sys
import numpy as np
import pytest
from scipy import sparse

SPEC=importlib.util.spec_from_file_location("scalar_alpha_retention590",Path(__file__).with_name("scalar_alpha_retention590.py"))
M=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=M;SPEC.loader.exec_module(M)


def test_scalarized_edge_rows_match_xyz_rows_with_cancellation_and_zero_terms():
    E=sparse.csr_matrix(np.asarray([[2.,0,0,-2,0,0],[0,3,0,0,0,-3],[1,-1,2,0,1,-2]]))
    d=np.asarray([[.4,-.2,.1],[-.3,.5,-.25]])
    alpha=np.asarray([.37,.82])
    A=M.scalarize_edge_rows590(E,d)
    expected=E@(alpha[:,None]*d).reshape(-1)
    assert np.allclose(A@alpha,expected,atol=1e-15,rtol=0)
    assert A.shape==(3,2)
    cert=M.certify_alpha590(alpha,np.ones(2),A,np.zeros(3),sparse.csr_matrix((0,2)),np.empty(0))
    assert cert["pass"]
    bad=M.certify_alpha590(alpha,np.asarray([.2,.5]),A,np.zeros(3),sparse.csr_matrix((0,2)),np.empty(0))
    assert not bad["pass"] and bad["upper_bound_violation"]>0


def test_trust_and_source_sphere_segment_caps_are_analytic_and_directional():
    base=np.asarray([[11.,0,0],[11.,0,0],[12.,0,0]])
    source=np.zeros_like(base)
    delta=np.asarray([[2.,0,0],[-2.,0,0],[1.,0,0]])
    cap=M.analytic_alpha_caps590(base,source,delta,trust_mm=4.,source_cap_mm=12.)
    assert cap[0]==pytest.approx(.5,abs=1e-12)
    assert cap[1]==pytest.approx(1.)
    assert cap[2]==pytest.approx(0.)
    capped=M.analytic_alpha_caps590(base[:1],source[:1],delta[:1],trust_mm=.25,source_cap_mm=12.)
    assert capped[0]==pytest.approx(.125,abs=1e-12)
    with pytest.raises(ValueError,match="base violates source sphere"):
        M.analytic_alpha_caps590(np.asarray([[12.1,0,0]]),np.zeros((1,3)),np.ones((1,3)),trust_mm=4,source_cap_mm=12)


def _area_values(base,ref,face,free,d,alpha):
    o,m,_,_=M.face_area_values_gradients590(base,ref,np.asarray([face]),free,d,alpha)
    return np.asarray([o[0],m[0]])


def test_area_double_area_gradient_matches_finite_difference_for_both_gates():
    base=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    ref=np.asarray([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
    face=np.asarray([[0,1,2]],dtype=np.int64)
    free=np.asarray([0,1,2],dtype=np.int64)
    d=np.asarray([[.03,-.04,.01],[-.02,.01,.05],[.01,.02,-.03]])
    alpha=np.asarray([.2,.35,.6])
    orient,mag,go,gm=M.face_area_values_gradients590(base,ref,face,free,d,alpha)
    h=1e-6
    for j in range(3):
        plus=alpha.copy();minus=alpha.copy();plus[j]+=h;minus[j]-=h
        numeric=(_area_values(base,ref,face[0],free,d,plus)-_area_values(base,ref,face[0],free,d,minus))/(2*h)
        assert numeric[0]==pytest.approx(go[0,j],rel=2e-7,abs=2e-8)
        assert numeric[1]==pytest.approx(gm[0,j],rel=2e-7,abs=2e-8)
    assert np.isfinite(orient[0]) and orient[0] > 0
    assert np.isfinite(mag[0]) and mag[0] > 0


def test_near_floor_cut_is_explicitly_linearized_without_changing_floor():
    base=np.asarray([[0.,0.,0.],[.11,0.,0.],[0.,1.,0.]])
    ref=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    F=np.asarray([[0,1,2]],dtype=np.int64)
    free=np.asarray([1],dtype=np.int64)
    d=np.asarray([[1.,0.,0.]])
    cuts,blocked,diag=M.build_area_cuts590(base,ref,F,free,d,np.zeros(1))
    assert (0,"fixed_reference_magnitude") in cuts
    cut=cuts[(0,"fixed_reference_magnitude")]
    assert cut.rhs==pytest.approx(.10001-.11,abs=1e-12)
    assert cut.rhs<=0.0 # alpha=0 remains feasible from this passing baseline
    assert not blocked
    assert diag["near_face_count"]==1
    A,b,_=M.pack_area_cuts590(cuts,1)
    assert A.shape==(1,1) and b[0]==pytest.approx(cut.rhs)


def test_scalar_lp_empty_zero_and_bounded_retention_cases():
    F=np.asarray([[0,1,2]],dtype=np.int64)
    base=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    source=np.asarray([[0.,0.,-10.],[1.,0.,-10.],[0.,1.,-10.]])
    ref=base.copy()
    E=sparse.csr_matrix((0,3),dtype=np.float64);rhs=np.empty(0)
    free=np.asarray([1],dtype=np.int64);delta=np.asarray([[0.,0.,.01]])
    alpha,report=M.optimize_scalar_alpha590(E,rhs,base,source,ref,F,free,delta,
        trust_mm=4,source_cap_mm=12,edge_gradient=.5,max_lp_calls=2,lp_time_limit_s=1)
    assert alpha.shape==(1,) and alpha[0]==pytest.approx(1.)
    assert report["lp_calls"]==1 and report["optimality_claim"] is False
    alpha0,report0=M.optimize_scalar_alpha590(E,rhs,base,source,ref,F,free,np.zeros((1,3)),
        trust_mm=4,source_cap_mm=12,edge_gradient=.5,max_lp_calls=1,lp_time_limit_s=1)
    assert alpha0[0]==pytest.approx(1.)
    with pytest.raises(ValueError,match="nonempty integer free-node scope"):
        M.optimize_scalar_alpha590(sparse.csr_matrix((0,0)),rhs,base,source,ref,F,
            np.empty(0,dtype=np.int64),np.empty((0,3)),max_lp_calls=1,lp_time_limit_s=1)
    with pytest.raises(ValueError,match="at most eight"):
        M.optimize_scalar_alpha590(E,rhs,base,source,ref,F,free,delta,max_lp_calls=9)
