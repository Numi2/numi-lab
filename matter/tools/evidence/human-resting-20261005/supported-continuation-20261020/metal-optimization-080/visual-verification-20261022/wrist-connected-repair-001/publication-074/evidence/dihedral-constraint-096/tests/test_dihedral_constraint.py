import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

TOOLS=Path(__file__).resolve().parents[1]/"matter"/"tools"
sys.path.insert(0,str(TOOLS))
import human_skin_fat_connected_repair as repair
from human_skin_fat_embedded_surface import MappedSkinSurface, _dihedral_values_gradients, _wrap_radians


def pair(theta=0.0):
    return np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                     [0.,-np.cos(theta),np.sin(theta)]],dtype=np.float32)


def pair_surface(origin, *, bound=None, edge_ids=None, W=None, coarse=None):
    faces=np.array([[0,1,2],[1,0,3]],dtype=np.int64)
    C=origin.copy() if coarse is None else coarse.copy()
    if W is None:
        W=sparse.eye(4,format="csr")
    return MappedSkinSurface(C,origin,faces,W, max_dihedral_change_degrees=bound,
                             dihedral_edge_ids=edge_ids)


def test_signed_orientation_matches_094_and_gate_is_opt_in():
    base=pair(0.)
    moved=pair(.2)
    faces=np.array([[0,1,2],[1,0,3]],dtype=np.int64)
    verts=np.array([[0,1,2,3]],dtype=np.int64)
    angle,_=_dihedral_values_gradients(moved,verts)
    assert angle[0] == pytest.approx(-.2,abs=2e-7)
    disabled=pair_surface(base)
    enabled=pair_surface(base,bound=5.,edge_ids=np.array([0]))
    assert disabled.passes(moved)
    assert "mapped_skin_dihedral_constraint_enabled" not in disabled.report(base)
    assert enabled.passes(moved) is False
    report=enabled.report(moved)
    assert report["mapped_skin_dihedral_constraint_enabled"] is True
    assert report["mapped_skin_dihedral_abs_change_max_degrees"]==pytest.approx(np.degrees(.2),abs=2e-5)
    assert report["mapped_skin_dihedral_violations"]==1


def test_analytic_dihedral_and_mapped_W_directional_derivative():
    D=pair(.31)
    C=np.vstack([D, [0.4,.2,.1],[-.1,.3,-.2]]).astype(np.float32)
    rows=np.array([0,1,2,2,3,3],dtype=np.int64)
    cols=np.array([0,1,2,4,3,5],dtype=np.int64)
    vals=np.array([1,1,.6,.4,.25,.75],dtype=np.float64)
    W=sparse.coo_matrix((vals,(rows,cols)),shape=(4,6)).tocsr()
    surf=pair_surface(D,bound=45.,edge_ids=np.array([0]),W=W,coarse=C)
    free=np.arange(6,dtype=np.int64)
    ids,delta,J=surf.dihedral_change_linearization(C,free)
    assert ids.tolist()==[0]
    assert delta[0]==pytest.approx(0.,abs=1e-7)
    rng=np.random.default_rng(20261025)
    direction=rng.normal(size=(6,3))
    direction/=np.linalg.norm(direction)
    h=5e-4
    plus=surf.dihedral_change_metrics((C.astype(float)+h*direction).astype(np.float32))[1]
    minus=surf.dihedral_change_metrics((C.astype(float)-h*direction).astype(np.float32))[1]
    observed=(plus-minus)/(2*h)
    predicted=J@direction.ravel()
    assert observed[0]==pytest.approx(predicted[0],abs=1.5e-3,rel=2e-3)


def test_near_flat_dihedral_derivative_is_finite_and_accurate():
    base=pair(1e-5)
    verts=np.array([[0,1,2,3]],dtype=np.int64)
    angle,grad=_dihedral_values_gradients(base,verts)
    assert np.isfinite(angle).all() and np.isfinite(grad).all()
    direction=np.random.default_rng(7).normal(size=base.shape)
    direction/=np.linalg.norm(direction)
    h=2e-5
    ap=_dihedral_values_gradients((base.astype(float)+h*direction),verts)[0]
    am=_dihedral_values_gradients((base.astype(float)-h*direction),verts)[0]
    observed=float(_wrap_radians(ap-am)[0]/(2*h))
    predicted=float(np.einsum("ijk,ijk->",grad,direction[verts]))
    assert observed==pytest.approx(predicted,abs=2e-4,rel=2e-4)


def test_periodic_delta_uses_short_signed_change():
    base=pair(-np.deg2rad(179.))
    candidate=pair(np.deg2rad(179.))
    surf=pair_surface(base,bound=3.,edge_ids=np.array([0]))
    delta=surf.dihedral_change_metrics(candidate)[1]
    assert np.degrees(delta[0])==pytest.approx(2.,abs=2e-4)
    assert surf.passes(candidate)


def test_inactive_free_support_returns_empty_safe_linearization():
    D=pair(0.)
    C=np.vstack([D,[2.,2.,2.]]).astype(np.float32)
    W=sparse.csr_matrix((np.ones(4),(np.arange(4),np.arange(4))),shape=(4,5))
    surf=pair_surface(D,bound=5.,edge_ids=np.array([0]),W=W,coarse=C)
    ids,delta,J=surf.dihedral_change_linearization(C,np.array([4],dtype=np.int64))
    assert ids.size==delta.size==0
    assert J.shape==(0,3)
    assert surf.passes(C)


def test_connected_lp_includes_hard_dihedral_rows_when_enabled(monkeypatch):
    origin=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
    X=origin.copy();X[3,0]=.015
    T=np.array([[0,1,2,3]],dtype=np.int64)
    faces=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],dtype=np.int64)
    W=sparse.eye(4,format="csr")
    disabled=MappedSkinSurface(origin,origin,faces,W)
    enabled=MappedSkinSurface(origin,origin,faces,W,max_dihedral_change_degrees=20.,
                               dihedral_edge_ids=np.array([0],dtype=np.int64))
    current_delta=enabled.dihedral_change_metrics(X)[1][0]
    C=sparse.csr_matrix(([1.],([0],[0])),shape=(1,12))
    target=np.array([1e-4])
    captures=[]
    def fake_linprog(c,*,A_ub,b_ub,bounds,method,options):
        captures.append((A_ub.copy(),b_ub.copy()))
        return SimpleNamespace(success=True,status=0,message="captured",x=np.zeros(len(c)))
    monkeypatch.setattr(repair,"linprog",fake_linprog)
    kwargs=dict(points=X,reference=origin,tetrahedra=T,free_nodes=np.array([0],dtype=np.int64),
                contact_matrix=C,contact_targets=target,construction_origin=origin,
                max_incremental_gradient=None,step_limit_mm=1.,time_limit_s=1.)
    repair.coupled_clearance_step(**kwargs,embedded_surface=disabled)
    repair.coupled_clearance_step(**kwargs,embedded_surface=enabled)
    assert captures[1][0].shape[0]==captures[0][0].shape[0]+2
    added=captures[1][0][-2:].toarray()
    assert np.allclose(added[0],-added[1])
    limit=np.deg2rad(20.)
    assert captures[1][1][-2:]==pytest.approx([limit-current_delta,limit+current_delta])
