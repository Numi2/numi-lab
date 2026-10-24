import numpy as np
import pytest
from human_skin_patch_geometry import triangulate_boundary_disk as cap

def test_convex_disk_boundary_no_center_vertex():
    x=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]])
    f,r=cap(x,np.arange(4))
    assert len(f)==2 and set(f.ravel())==set(range(4))
    assert r["oriented_boundary_exact"] and not r["physical_admission"]

def test_concave_polygon_not_centroid_fan():
    x=np.array([[0.,0,0],[3,0,0],[3,3,0],[2,3,0],[2,1,0],[1,1,0],[1,3,0],[0,3,0]])
    f,r=cap(x,np.arange(8))
    a=np.linalg.norm(np.cross(x[f[:,1]]-x[f[:,0]],x[f[:,2]]-x[f[:,0]]),axis=1).sum()/2
    assert len(f)==6 and abs(a-7)<1e-12

def test_nonplanar_simple_loop_keeps_all_coordinates():
    x=np.array([[0.,0,.01],[1,0,0],[1,1,.02],[0,1,0]])
    prior=x.copy();f,r=cap(x,np.arange(4))
    np.testing.assert_array_equal(x,prior)
    assert r["projection_nonplanarity_m"]>0 and len(f)==2

def test_reversed_boundary_preserved():
    x=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]])
    a,_=cap(x,np.arange(4));b,_=cap(x,np.arange(4)[::-1])
    az=np.cross(x[a[:,1]]-x[a[:,0]],x[a[:,2]]-x[a[:,0]])[:,2].sum()
    bz=np.cross(x[b[:,1]]-x[b[:,0]],x[b[:,2]]-x[b[:,0]])[:,2].sum()
    assert az*bz<0

def test_bow_tie_rejected():
    x=np.array([[0.,0,0],[1,1,0],[0,1,0],[1,0,0]])
    with pytest.raises(ValueError):cap(x,np.arange(4))

def test_repeated_identity_rejected():
    with pytest.raises(ValueError):cap(np.eye(3),np.array([0,1,0]))

def test_collinear_rejected():
    with pytest.raises(ValueError):cap(np.array([[0.,0,0],[1,0,0],[2,0,0]]),np.arange(3))

def test_scale_translation_same_topology():
    x=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]])
    a,_=cap(x,np.arange(4));b,_=cap(x*1e-4+1e2,np.arange(4))
    assert len(a)==len(b)==2
