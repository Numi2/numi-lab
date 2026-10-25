from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np

SOURCE=Path(__file__).with_name("build_nearest_face_union_targets.py")
SPEC=importlib.util.spec_from_file_location("nearest_face_union_190",SOURCE)
MOD=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

def test_nonorthogonal_raw_affine_roundtrip_and_world_metric_closest_point():
    M=np.array([[1.0,.23,0.0],[0.0,1.0,.17],[.06,0.0,1.0]],dtype=np.float64)
    pos=np.array([.3,-.2,.7]);off=np.array([.01,-.02,.03])
    local=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],
                    [[0.,0.,.4],[1.,0.,.4],[0.,1.,.4]]],dtype=np.float64)
    world=MOD.world_triangles_from_local(local,M,pos,off)
    recovered=(world.reshape(-1,3)-pos)@M.T-off
    assert np.allclose(recovered,local.reshape(-1,3),atol=2e-15,rtol=0)
    # A world-space query must select by Euclidean world distance after exact affine inversion.
    normal=np.cross(world[0,1]-world[0,0],world[0,2]-world[0,0])
    normal/=np.linalg.norm(normal)
    query=np.mean(world[0],axis=0)+.12*normal
    got=MOD.closest_points_on_triangle_mesh(query,world,np.array([20,10],np.int64))
    assert got["triangle_stable_id"]==20
    assert abs(got["distance_m"]-.12)<1e-12
    np.testing.assert_allclose(np.linalg.norm(got["point_world_m"]-query),.12,atol=1e-12,rtol=0)

def cube_triangles():
    v=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=np.float64)
    # Two triangles per cube face. Winding is irrelevant to closest point.
    f=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],
                [0,1,5],[0,5,4],[1,2,6],[1,6,5],
                [2,3,7],[2,7,6],[3,0,4],[3,4,7]],dtype=np.int64)
    return v[f]

def test_inside_bank_sample_finds_complete_shell_boundary():
    tris=cube_triangles()
    out=MOD.closest_points_on_triangle_mesh(np.zeros(3),tris,np.arange(100,112,dtype=np.int64))
    assert abs(out["distance_m"]-1.0)<1e-14
    assert np.max(np.abs(out["point_world_m"]))==1.0
    assert out["triangle_stable_id"]>=100

def test_exact_multiple_closest_ties_use_smallest_stable_triangle_id():
    tri=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],
                  [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    got=MOD.closest_points_on_triangle_mesh(np.array([.2,.2,1.]),tri,np.array([9,2],np.int64))
    assert got["triangle_stable_id"]==2
    assert got["triangle_row"]==1
    assert got["tie_count"]==2
    np.testing.assert_allclose(got["point_world_m"],[.2,.2,0.],atol=1e-15)

def test_nearest_direction_is_sample_to_closest_contact_not_surface_normal():
    tri=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    rec={(4,7):{"world_triangles":tri,"id":np.array([31],np.int64)}}
    direction,w=MOD.nearest_inside_union_boundary_direction([.2,.2,.3],{(4,7)},rec)
    np.testing.assert_allclose(direction,[0.,0.,-1.],atol=1e-15)
    np.testing.assert_allclose(w["contact_world_m"],[.2,.2,0.],atol=1e-15)
    assert w["distance_m"]==.3
    assert "world-metric" in w["selection_rule"]


def test_equal_nearest_banks_tie_by_typed_body_and_bone_ids():
    tri=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    rec={
        (8,1):{"world_triangles":tri,"id":np.array([2],np.int64)},
        (3,9):{"world_triangles":tri,"id":np.array([40],np.int64)},
    }
    direction,w=MOD.nearest_inside_union_boundary_direction([.2,.2,.2],{(8,1),(3,9)},rec)
    np.testing.assert_allclose(direction,[0.,0.,-1.],atol=1e-15)
    assert w["body"]==3 and w["stable_bone_id"]==9 and w["triangle_stable_id"]==40
    assert w["active_bank_count"]==2
