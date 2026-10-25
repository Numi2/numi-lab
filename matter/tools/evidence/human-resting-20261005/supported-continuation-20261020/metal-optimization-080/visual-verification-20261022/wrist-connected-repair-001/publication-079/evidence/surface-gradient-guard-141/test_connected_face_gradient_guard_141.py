import numpy as np
import pytest
from connected_face_gradient_guard_141 import build_face_gradient_guard

P=np.array([[.1,.2,.3],[.103,.201,.301],[.099,.204,.302],[.102,.203,.305]])
F=np.array([[0,1,2],[1,3,2]])

def test_affine_displacement_and_translations():
    G,limit,d=build_face_gradient_guard(P,F)
    A=np.array([[.1,.02,0],[-.03,.08,.01],[.01,0,-.04]])
    U=P@A.T+np.array([3.,-4.,5.])
    for f,face in enumerate(F):
        e=(P[face[1:]]-P[face[0]]).T
        t0=e[:,0]/np.linalg.norm(e[:,0])
        n=np.cross(e[:,0],e[:,1]);n/=np.linalg.norm(n)
        T=np.column_stack((t0,np.cross(n,t0)))
        np.testing.assert_allclose((G@U.ravel()).reshape(-1,3,2)[f],A@T,atol=3e-12,rtol=1e-10)
    np.testing.assert_allclose(G@np.tile([1.,2.,3.],len(P)),0,atol=1e-12)
    assert limit==.4/np.sqrt(6) and d["static_construction_only"]

def test_gradient_bound_preserves_oriented_area():
    G,limit,_=build_face_gradient_guard(P,F)
    rng=np.random.default_rng(141)
    for _ in range(100):
        U=rng.normal(size=P.shape)
        peak=np.max(np.abs(G@U.ravel()))
        U*=limit*.99/peak
        for f in F:
            e=P[f[1:]]-P[f[0]];y=(P+U)[f[1:]]-(P+U)[f[0]]
            assert np.dot(np.cross(*e),np.cross(*y))>0
    U=-2*P
    assert np.max(np.abs(G@U.ravel()))>limit

@pytest.mark.parametrize("value",[0,1,-1,np.inf,np.nan])
def test_invalid_bound(value):
    with pytest.raises(ValueError):build_face_gradient_guard(P,F,max_frobenius=value)

def test_degenerate_and_sliver_rejected():
    for Q in (np.array([[0,0,0],[1,0,0],[2,0,0]]),np.array([[0,0,0],[1,0,0],[1,1e-10,0]])):
        with pytest.raises(ValueError):build_face_gradient_guard(Q,np.array([[0,1,2]]))

def test_input_domains_and_empty():
    for ff in (F.astype(float),np.array([[0,1,8]]),np.array([[0,1,-1]])):
        with pytest.raises(ValueError):build_face_gradient_guard(P,ff)
    Q=P.copy();Q[0,0]=np.nan
    with pytest.raises(ValueError):build_face_gradient_guard(Q,F)
    G,limit,d=build_face_gradient_guard(P,np.zeros((0,3),int))
    assert G.shape==(0,12) and d["face_count"]==0
