import numpy as np
from column_fit_support import (patch_columns_from_seed,apply_shared_column_translations,
    normalized_tet_quality_and_gradient,reference_J_and_gradient,
    aggregate_vertex_gradient_to_columns)

def test_patch_boundary_and_interior():
    # Four triangles around a seeded center; the perimeter is anchored.
    F=np.asarray([[0,1,4],[1,2,4],[2,3,4],[3,0,4]],np.int64)
    rows,boundary,movable=patch_columns_from_seed([4],F)
    assert np.array_equal(rows,np.arange(4))
    assert np.array_equal(boundary,np.asarray([0,1,2,3]))
    assert np.array_equal(movable,np.asarray([4]))

def test_shared_column_translation_preserves_layer_vectors():
    C=5; X=np.arange(3*C*3,dtype=np.float64).reshape(3*C,3)/19
    cols=np.asarray([1,3]);d=np.asarray([[.001,-.002,.003],[-.004,.002,.001]])
    Y=apply_shared_column_translations(X,cols,d,column_count=C)
    for c,v in zip(cols,d):
        for layer in range(2):
            assert np.allclose(Y[(layer+1)*C+c]-Y[layer*C+c],X[(layer+1)*C+c]-X[layer*C+c],rtol=0,atol=1e-15)
    fixed=np.setdiff1d(np.arange(3*C),np.concatenate([l*C+cols for l in range(3)]))
    assert np.array_equal(Y[fixed],X[fixed])

def _tet_fixture():
    X=np.asarray([[0.02,0.03,0.01],[1.23,0.07,0.02],[0.11,0.82,0.09],[0.04,0.12,0.73]],float)
    T=np.asarray([[0,1,2,3]],np.int64)
    return X,T

def _fd(f,x,j,h=1e-6):
    xp=x.copy();xm=x.copy();xp[j]+=h;xm[j]-=h
    return (f(xp)-f(xm))/(2*h)

def test_quality_vertex_gradient_and_shared_column_chain():
    X,T=_tet_fixture()
    q,G=normalized_tet_quality_and_gradient(X[T])
    for k in range(4):
        for a in range(3):
            def f(flat):return normalized_tet_quality_and_gradient(flat.reshape(4,3)[T])[0][0]
            assert abs(_fd(f,X.ravel(),3*k+a)-G[0,k,a])<2e-6
    cols=np.arange(4)
    Q,J=__import__("column_fit_support").quality_and_column_jacobian(X,T,[0],cols,column_count=4)
    assert Q.shape==(1,) and J.shape==(1,12)
    for c in range(4):
        for a in range(3):
            step=np.zeros((4,3));step[c,a]=1e-6
            qp=__import__("column_fit_support").quality_and_column_jacobian(X+step,T,[0],cols,column_count=4)[0]
            qm=__import__("column_fit_support").quality_and_column_jacobian(X-step,T,[0],cols,column_count=4)[0]
            assert abs((qp[0]-qm[0])/2e-6-J[0,3*c+a])<3e-6

def test_reference_J_gradient_and_shared_column_chain():
    X,T=_tet_fixture();R=X.copy();X[1,0]=1.05;X[2,1]=.98;X[3,2]=1.03
    j,G=reference_J_and_gradient(X[T],R[T])
    for k in range(4):
        for a in range(3):
            def f(flat):return reference_J_and_gradient(flat.reshape(4,3)[T],R[T])[0][0]
            assert abs(_fd(f,X.ravel(),3*k+a)-G[0,k,a])<2e-7
    JJ,J=__import__("column_fit_support").J_and_column_jacobian(X,R,T,[0],np.arange(4),column_count=4)
    assert abs(JJ[0]-j[0])<1e-12
    for c in range(4):
        for a in range(3):
            step=np.zeros((4,3));step[c,a]=1e-6
            jp=__import__("column_fit_support").J_and_column_jacobian(X+step,R,T,[0],np.arange(4),column_count=4)[0]
            jm=__import__("column_fit_support").J_and_column_jacobian(X-step,R,T,[0],np.arange(4),column_count=4)[0]
            assert abs((jp[0]-jm[0])/2e-6-J[0,3*c+a])<2e-7
import numpy as np
import pytest
from column_fit_support import expand_seed_patch, plane_constraint_slack_and_jacobian, apply_shared_column_translations


def _grid_faces(n=4):
    faces=[]
    for y in range(n-1):
        for x in range(n-1):
            a=y*n+x; b=a+1; c=a+n; d=c+1
            faces.extend(((a,b,d),(a,d,c)))
    return np.asarray(faces,np.int64)


def test_expanded_patch_keeps_measured_seeds_interior():
    F=_grid_faces(4)
    p=expand_seed_patch(np.array([5],np.int64),F,max_rings=2)
    assert 5 in p["movable_columns"]
    assert p["trace"][0]["seed_columns_still_on_boundary"]==0
    assert len(p["boundary_columns"])>0


def test_entry_plane_jacobian_matches_shared_column_translation():
    X=np.arange(27,dtype=np.float64).reshape(9,3)/100
    cols=np.array([0],np.int64)
    hits=np.array([[0.,0.,0.]])
    normals=np.array([[1.,0.,0.]])
    endpoints=np.array([[0,3]],np.int64)
    r=plane_constraint_slack_and_jacobian(X,cols,hits,normals,endpoints,.001,column_count=3)
    delta=np.array([[1e-4,0,0]])
    Y=apply_shared_column_translations(X,cols,delta,column_count=3,layer_count=3)
    r2=plane_constraint_slack_and_jacobian(Y,cols,hits,normals,endpoints,.001,column_count=3)
    assert np.allclose((r2["slack"]-r["slack"])/1e-4,1.,rtol=1e-10,atol=1e-10)
    assert np.allclose(r["jacobian"][:,0],1.)


def test_entry_plane_rejects_nonunit_normals_and_bad_endpoint():
    X=np.zeros((9,3))
    with pytest.raises(ValueError):
        plane_constraint_slack_and_jacobian(X,np.array([0]),np.zeros((1,3)),np.array([[2.,0.,0.]]),np.array([[0]]),0.,column_count=3)
    with pytest.raises(ValueError):
        plane_constraint_slack_and_jacobian(X,np.array([0]),np.zeros((1,3)),np.array([[1.,0.,0.]]),np.array([[9]]),0.,column_count=3)
import numpy as np
from column_fit_support import shared_column_components, surface_edge_pairs


def test_components_follow_cells_and_preserve_independent_islands():
    T=np.array([[0,1,2,3],[2411+4,2411+5,2411+6,2411+7]],np.int64)
    groups,gid=shared_column_components(T,np.array([0,1,4,5,6,7],np.int64))
    assert sorted(map(len,groups))==[2,4]
    assert gid[0]==gid[1] and gid[4]==gid[7] and gid[0]!=gid[4]


def test_components_join_by_explicit_surface_edge_term():
    T=np.array([[0,1,2,3],[2411+4,2411+5,2411+6,2411+7]],np.int64)
    groups,gid=shared_column_components(T,np.array([0,1,4,5,6,7],np.int64),coupling_edges=[(1,4)])
    assert len(groups)==1


def test_surface_edges_are_unique_sorted():
    E=surface_edge_pairs(np.array([[2,1,0],[0,1,3]],np.int64))
    assert E.tolist()==[[0,1],[0,2],[0,3],[1,2],[1,3]]
import numpy as np
import pytest
from column_fit_support import route_entry_endpoint_constraints


def test_route_policy_maps_entry_targets_to_downstream_layer_nodes():
    r=route_entry_endpoint_constraints(
        np.array([3,8,9],np.int64),np.array([0,1,1],np.int64),
        np.array([3,2419,2420],np.int64),np.array([2414,4830,4831],np.int64),
        np.array(["entry","entry","exit"]),np.array(["outside"]*3),np.array(["inside"]*3),
        np.zeros((3,3)),np.tile([1.,0.,0.],(3,1)),np.tile([.001,0.,0.],(3,1)))
    assert r["source_hit_rows"].tolist()==[0,1]
    assert r["endpoint_node_rows"].tolist()==[[2414,4825],[4830,-1]]
    assert r["nonbinding_counts"]["not_entry"]==1


def test_route_policy_keeps_unproven_start_as_nonbinding():
    r=route_entry_endpoint_constraints(
        np.array([0],np.int64),np.array([0],np.int64),np.array([0],np.int64),np.array([2411],np.int64),
        np.array(["entry"]),np.array(["ambiguous"]),np.array(["inside"]),
        np.zeros((1,3)),np.array([[1.,0.,0.]]),np.array([[.001,0.,0.]]))
    assert not len(r["source_hit_rows"])
    assert r["nonbinding_counts"]["start_not_proven_outside"]==1


def test_route_policy_rejects_wrong_layer_endpoint_identity():
    with pytest.raises(ValueError):
        route_entry_endpoint_constraints(
            np.array([0],np.int64),np.array([0],np.int64),np.array([0],np.int64),np.array([2412],np.int64),
            np.array(["entry"]),np.array(["outside"]),np.array(["inside"]),
            np.zeros((1,3)),np.array([[1.,0.,0.]]),np.array([[.001,0.,0.]]))
import numpy as np
import pytest
from column_fit_support import expand_construction_plane_targets


def test_owner_plane_target_masks_expand_to_correct_endpoint_layers():
    r=expand_construction_plane_targets(np.array([3,8]),np.array([6,4]),np.zeros((2,3)),
                                        np.tile([1.,0.,0.],(2,1)),np.array([1e-4,2e-4]))
    assert r["endpoint_node_rows"].tolist()==[[2414,4825],[4830,-1]]
    assert r["clearance_m"].tolist()==[1e-4,2e-4]


def test_owner_plane_rejects_O_or_unknown_masks_and_bad_normals():
    with pytest.raises(ValueError):
        expand_construction_plane_targets(np.array([0]),np.array([1]),np.zeros((1,3)),np.array([[1.,0.,0.]]),np.array([1e-4]))
    with pytest.raises(ValueError):
        expand_construction_plane_targets(np.array([0]),np.array([2]),np.zeros((1,3)),np.array([[2.,0.,0.]]),np.array([1e-4]))


def test_entry_plane_zero_dof_evaluation_has_empty_width_jacobian():
    X=np.asarray([[.2,0,0],[.4,0,0],[.6,0,0]],np.float64)
    r=plane_constraint_slack_and_jacobian(X,np.asarray([],np.int64),np.asarray([[0.,0.,0.],[0.,0.,0.]]),np.asarray([[1.,0.,0.],[1.,0.,0.]]),np.asarray([[0],[2]],np.int64),np.asarray([.1,.2]),column_count=1)
    assert r["jacobian"].shape==(2,0)
    assert np.allclose(r["slack"],[.1,.4])
    e=plane_constraint_slack_and_jacobian(X,np.asarray([],np.int64),np.empty((0,3)),np.empty((0,3)),np.empty((0,1),np.int64),np.empty(0),column_count=1)
    assert e["jacobian"].shape==(0,0) and e["slack"].shape==(0,)
