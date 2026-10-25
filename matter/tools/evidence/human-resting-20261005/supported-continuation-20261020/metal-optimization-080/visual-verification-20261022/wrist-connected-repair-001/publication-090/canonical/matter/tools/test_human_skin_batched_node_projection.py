import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from human_skin_fat_surface_fit_constraints import *

REF=Path(__file__).parent/"fixtures/run032_node_projection_reference.py"
space=dict(np=np,project_ball_halfspace=project_ball_halfspace,
           project_two_balls_halfspace=project_two_balls_halfspace,InfeasibleProjection=InfeasibleProjection)
exec(compile(REF.read_text(),str(REF),"exec"),space)
old=space["project_node_sets"]

def case(seed,n=200,contained=False):
    r=np.random.default_rng(seed)
    source=np.zeros((n,3))
    trust=r.normal(size=(n,3));trust*=r.uniform(0,2.5,n)[:,None]/np.maximum(np.linalg.norm(trust,axis=1)[:,None],1e-30)
    trust[:3]=0
    anchors=np.arange(min(3,n),dtype=np.int64)
    rows=np.arange(3,n,4,dtype=np.int64)
    normals=r.normal(size=(len(rows),3));normals/=np.linalg.norm(normals,axis=1)[:,None]
    lower=np.einsum("ij,ij->i",normals,trust[rows]*.5)
    v=r.normal(size=(n,3))*4
    return v,source,12. if contained else 3.,rows,normals,lower,anchors,trust,2.

@pytest.mark.parametrize("seed",range(12))
@pytest.mark.parametrize("contained",[False,True])
def test_matches_serial_exact_active_set_projection(seed,contained):
    args=case(seed,contained=contained)
    a,ar=old(*args);b,br=project_node_sets_batched(*args)
    np.testing.assert_allclose(a,b,rtol=0,atol=3e-14)
    assert ar["overlap_nodes"]==br["overlap_nodes"]
    assert br["intersection_residual_mm"]<1e-8

def test_empty_domain():
    z=np.empty((0,3));o,m=project_node_sets_batched(z,z,3,[],z,[],[],z,2)
    assert o.shape==(0,3) and m["intersection_residual_mm"]==0

def test_anchored_exit_plane_infeasible():
    a=list(case(4));a[3]=np.array([0]);a[4]=np.array([[1.,0,0]]);a[5]=np.array([.1])
    with pytest.raises(InfeasibleProjection):project_node_sets_batched(*a)

def test_anchor_outside_trust_ball_rejected():
    a=list(case(4));a[7][0]=[3,0,0]
    with pytest.raises(ValueError,match="anchor"):project_node_sets_batched(*a)

def test_duplicate_plane_row_rejected():
    a=list(case(4));a[3][1]=a[3][0]
    with pytest.raises(ValueError,match="identities"):project_node_sets_batched(*a)
