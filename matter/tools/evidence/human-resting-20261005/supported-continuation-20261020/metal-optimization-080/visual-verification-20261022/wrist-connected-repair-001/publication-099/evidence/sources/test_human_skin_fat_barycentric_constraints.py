import numpy as np
import pytest
from human_skin_fat_barycentric_constraints import build_barycentric_exit_system as build, progressive_barycentric_lower as progress
P=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
F=np.array([[0,1,2],[1,3,2]])
def args():
 return [P,F,np.array([0,1]),np.array([[.2,.3,.5],[.4,.2,.4]]),np.array([[0.,0,1],[0,0,1]]),np.array([[.3,.5,1],[.6,.6,2]])]
def test_sparse_adjoint_and_target():
 B,tri,g,h,delta=build(*args())
 rng=np.random.default_rng(4);d=rng.normal(size=P.shape);force=rng.normal(size=(2,3))
 np.testing.assert_allclose(np.sum((B@d)*force),np.sum(d*(B.T@force)),rtol=1e-14)
 np.testing.assert_allclose(np.einsum('mij,mij->m',g,d[tri]),(B@d)[:,2])
 np.testing.assert_allclose(h,[1,2]);np.testing.assert_allclose(B@P+delta,args()[-1])
def test_anchor_support_and_trust():
 B,tri,g,h,delta=build(*args(),anchor_rows=np.array([0,1]))
 assert np.all(g[np.isin(tri,[0,1])]==0)
 d=np.zeros_like(P);v=progress(g,tri,d,h,.1)
 np.testing.assert_allclose(v,[.0475,.057])
def test_empty():
 a=args();a[2]=np.empty(0,dtype=int);a[3]=np.empty((0,3));a[4]=np.empty((0,3));a[5]=np.empty((0,3))
 B,tri,g,h,delta=build(*a)
 assert B.shape==(0,4) and g.shape==(0,3,3)
def test_fixed_support_rejected():
 with pytest.raises(ValueError):build(*args(),anchor_rows=np.arange(4))
@pytest.mark.parametrize("kind",["negative","sum","nan","normal","backward","face","index","target","anchor"])
def test_invalid(kind):
 a=args();kw={}
 if kind=="negative":a[3][0]=[-.1,.6,.5]
 if kind=="sum":a[3][0]*=.5
 if kind=="nan":a[3][0,0]=np.nan
 if kind=="normal":a[4]*=2
 if kind=="backward":a[5][:,2]=-1
 if kind=="face":a[2]=np.array([2,1])
 if kind=="index":a[1]=F.astype(float)
 if kind=="target":a[5]=np.zeros((3,3))
 if kind=="anchor":kw["anchor_rows"]=np.array([-1])
 with pytest.raises(ValueError):build(*a,**kw)
@pytest.mark.parametrize("seed",range(20))
def test_random_virtual_work(seed):
 rng=np.random.default_rng(seed);p=rng.normal(size=(50,3));f=np.array([rng.choice(50,3,replace=False) for _ in range(25)])
 w=rng.random((25,3));w/=w.sum(axis=1)[:,None];n=rng.normal(size=(25,3));n/=np.linalg.norm(n,axis=1)[:,None]
 target=np.einsum("mi,mij->mj",w,p[f])+n*rng.uniform(.01,2,size=(25,1))
 B,tri,g,h,delta=build(p,f,np.arange(25),w,n,target)
 d=rng.normal(size=p.shape);np.testing.assert_allclose(np.einsum("mij,mij->m",g,d[tri]),np.einsum("ij,ij->i",n,B@d),atol=1e-14)
 z=progress(g,tri,np.zeros_like(p),h,.2)
 assert np.all(z>0) and np.all(z<=.2+1e-14)
