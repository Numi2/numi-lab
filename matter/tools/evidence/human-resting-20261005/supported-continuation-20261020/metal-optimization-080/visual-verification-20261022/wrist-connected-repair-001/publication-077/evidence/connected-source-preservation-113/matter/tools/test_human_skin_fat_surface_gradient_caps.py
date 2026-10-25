import numpy as np
import pytest
from scipy import sparse
from types import SimpleNamespace
import sys
from pathlib import Path

TOOLS=Path(__file__).resolve().parent
sys.path.insert(0,str(TOOLS))
import human_skin_fat_connected_repair as repair
from human_skin_fat_embedded_surface import MappedSkinSurface


def _triangle_metrics(points, origin):
    face=np.array([[0,1,2]],dtype=np.int64)
    _,_,area,cos=repair.boundary_face_values_gradients(points,origin,face)
    return float(area[0]),float(cos[0])


def test_sufficient_surface_caps_bound_area_and_normal_for_random_gradients():
    origin=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.]],dtype=np.float64)
    ratio=1.2; cosine=.85
    scale=.01*np.sqrt(ratio);angle=np.arccos(cosine)
    current=np.array([[0.,0.,0.],[scale,0.,0.],[0.,scale*np.cos(angle),scale*np.sin(angle)]])
    bounds=(.7,1.7);cos_min=.65
    r,c=_triangle_metrics(current,origin)
    assert r==pytest.approx(ratio) and c==pytest.approx(cosine)
    eps=repair.sufficient_surface_gradient_caps([r],[c],bounds,cos_min,.5)[0]
    assert 0.<eps<.5
    g=eps/3.
    rng=np.random.default_rng(10625)
    for _ in range(500):
        G=rng.uniform(-.999*g,.999*g,size=(3,3))
        changed=current@(np.eye(3)+G).T
        area,normal_cos=_triangle_metrics(changed,origin)
        assert bounds[0]<=area<=bounds[1]
        assert normal_cos>=cos_min


def test_triangle_gradient_jacobian_composes_through_sparse_map():
    coarse=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.],[.01,.01,.003]],dtype=np.float32)
    W=sparse.csr_matrix(np.array([[.5,.5,0.,0.],[0.,.5,.5,0.],[0.,0.,.5,.5]]))
    detail=(W@coarse.astype(np.float64)).astype(np.float32)
    faces=np.array([[0,1,2]],dtype=np.int64)
    surface=MappedSkinSurface(coarse,detail,faces,W,area_bounds=(.2,3.),normal_cosine_min=.2)
    free=np.arange(4,dtype=np.int64)
    ids,J=surface.triangle_gradient_linearization(coarse,free,[0])
    assert ids.tolist()==[0] and J.shape==(9,12)
    delta_mm=np.array([.2,-.1,.3,-.4,.5,.2,.1,-.3,.25,.4,.2,-.1])
    predicted=(J@delta_mm).reshape(1,3,3)[0]
    moved=coarse.astype(np.float64)+(delta_mm.reshape(-1,3)*.001)
    before=(surface.detail_origin.astype(np.float64)+
            surface.weights@(coarse.astype(np.float64)-surface.coarse_origin))
    after=(surface.detail_origin.astype(np.float64)+
           surface.weights@(moved-surface.coarse_origin))
    tri=faces[0];v=before[tri];u=after[tri]-v
    E=np.column_stack((v[1]-v[0],v[2]-v[0]))
    direct=np.column_stack((u[1]-u[0],u[2]-u[0]))@np.linalg.pinv(E)
    np.testing.assert_allclose(predicted,direct,rtol=2e-6,atol=2e-9)


def _surface_cap_fixture():
    X=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.],[0.,0.,.02]],dtype=np.float32)
    T=np.array([[0,1,2,3]],dtype=np.int64)
    faces=np.array([[0,1,2]],dtype=np.int64)
    W=sparse.csr_matrix(([1.,1.,1.],([0,1,2],[0,1,2])),shape=(3,4))
    mapped=MappedSkinSurface(X,X[:3].copy(),faces,W,area_bounds=(.5,1.5),normal_cosine_min=.7)
    C=sparse.csr_matrix(([1.],([0],[8])),shape=(1,12))
    return dict(points=X,reference=X.astype(np.float64),tetrahedra=T,
        free_nodes=np.array([2],dtype=np.int64),contact_matrix=C,
        contact_targets=np.array([.03]),construction_origin=X.copy(),
        max_incremental_gradient=None,step_limit_mm=30.,embedded_surface=mapped,
        boundary_faces=faces,boundary_area_bounds=(.5,1.5),
        boundary_normal_cosine_min=.7,dihedral_constraint_passes=1,
        surface_witness_gradient_limit=.5,time_limit_s=2.)


def test_coarse_and_mapped_face_caps_are_added_only_after_witness_and_solved(monkeypatch):
    kw=_surface_cap_fixture();calls=[]
    def fake(c,**opts):
        calls.append((opts['A_ub'].copy(),opts['b_ub'].copy()))
        x=np.zeros(len(c));x[2]=30. if len(calls)==1 else 1.
        return SimpleNamespace(success=True,status=0,message='injected',x=x)
    monkeypatch.setattr(repair,'linprog',fake)
    Y,report=repair.coupled_clearance_step(**kw)
    assert len(calls)==2
    assert report['lp_constraint_generation'][0]['proposed_surface_violation_count']==2
    assert report['lp_constraint_generation'][0]['new_surface_gradient_caps']==2
    assert calls[1][0].shape[0]-calls[0][0].shape[0]==36
    records=report['lp_surface_gradient_cap_solved']
    assert [(r['surface'],r['face_id']) for r in records]==[
        ('coarse_boundary',0),('mapped_skin',0)]
    assert report['lp_surface_gradient_cap_solved_count']==2
    assert report['lp_surface_gradient_cap_solved_component_count']==18
    assert report['lp_surface_gradient_cap_solved_inequality_row_count']==36
    assert report['lp_surface_gradient_cap_pending_count']==0
    assert report['lp_surface_gradient_cap_not_added_at_budget']==[]
    assert all(0.<r['epsilon_cap']<.5 and r['componentwise_g']==pytest.approx(r['epsilon_cap']/3.) for r in records)

    Gc=repair.sparse_triangle_displacement_gradient(kw['points'],kw['boundary_faces'],kw['free_nodes'],len(kw['points']))
    _,Gd=kw['embedded_surface'].triangle_gradient_linearization(kw['points'],kw['free_nodes'],[0])
    zero=sparse.csr_matrix((9,calls[1][0].shape[1]-Gc.shape[1]))
    expected=sparse.vstack([sparse.hstack([Gc,zero]),sparse.hstack([-Gc,zero]),
        sparse.hstack([Gd,zero]),sparse.hstack([-Gd,zero])],format='csr')
    np.testing.assert_allclose((calls[1][0][-36:]-expected).toarray(),0.,atol=1e-12)
    assert np.min(calls[1][1][-36:])>0.
    assert Y.dtype==np.float32
    assert kw['embedded_surface'].passes(Y)
    _,_,area,cos=repair.boundary_face_values_gradients(Y,kw['construction_origin'],kw['boundary_faces'])
    assert area.min()>=.5 and area.max()<=1.5 and cos.min()>=.7


def test_failed_refinement_reports_pending_surface_caps_not_as_solved(monkeypatch):
    kw=_surface_cap_fixture();calls=[]
    def fake(c,**opts):
        calls.append(1);x=np.zeros(len(c));x[2]=30.
        return SimpleNamespace(success=len(calls)==1,status=0 if len(calls)==1 else 1,
            message='injected timeout' if len(calls)>1 else 'injected',x=x)
    monkeypatch.setattr(repair,'linprog',fake)
    Y,report=repair.coupled_clearance_step(**kw)
    assert len(calls)==2 and report['lp_refinement_failure_retained']
    assert report['lp_surface_gradient_cap_solved_count']==0
    assert report['lp_surface_gradient_cap_solved']==[]
    assert report['lp_surface_gradient_cap_pending_count']==2
    assert [(r['surface'],r['face_id']) for r in report['lp_surface_gradient_cap_pending']]==[
        ('coarse_boundary',0),('mapped_skin',0)]
    assert all(r['componentwise_g']>0. for r in report['lp_surface_gradient_cap_pending'])
    assert report['lp_constraint_generation_stop_reason']=='refinement_lp_failed'
    assert Y.dtype==np.float32 and kw['embedded_surface'].passes(Y)


def test_surface_gradient_caps_require_generation_passes_and_surface():
    kw=_surface_cap_fixture()
    kw['dihedral_constraint_passes']=0
    with pytest.raises(ValueError,match='surface witness gradient limit'):
        repair.coupled_clearance_step(**kw)
    kw=_surface_cap_fixture();kw['boundary_faces']=None;kw['embedded_surface']=None
    with pytest.raises(ValueError,match='surface witness gradient limit'):
        repair.coupled_clearance_step(**kw)
