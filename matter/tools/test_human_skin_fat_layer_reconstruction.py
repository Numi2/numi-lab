import numpy as np
import pytest
from human_skin_fat_layer_reconstruction import reconstruct_connected_layers


def grid_mesh():
    points=np.array([[float(x),float(y),0.] for y in range(4) for x in range(4)],dtype=np.float32)
    faces=[]
    for y in range(3):
        for x in range(3):
            a=y*4+x;b=a+1;c=a+4;d=c+1
            faces.extend(((a,b,d),(a,d,c)))
    return points,np.asarray(faces,dtype=np.int32)


def invoke(P,F,*,active=None,boundary=None,sign=1,skin=.02,fat=.03):
    n=len(P)
    if active is None: active=np.arange(n,dtype=np.int32)
    if boundary is None: boundary=np.asarray([0,1,2,3,4,7,8,11,12,13,14,15],dtype=np.int32)
    di=np.tile(np.array([0.,0.,-.004]),(n,1))
    dd=np.tile(np.array([0.,0.,-.006]),(n,1))
    b=np.asarray(boundary,dtype=np.int32)
    bdata=np.clip(b,0,n-1)
    return reconstruct_connected_layers(
        P,F,di,dd,b,P[bdata],(P+di)[bdata],(P+di+dd)[bdata],
        skin_depth_m=skin,fat_depth_m=fat,orientation_sign=sign,
        active_vertex_indices=active)


def invoke_matching_normal_shell(P,F,sign=1,skin=.02,fat=.03):
    n=len(P)
    boundary=np.asarray([0,1,2,3,4,7,8,11,12,13,14,15],dtype=np.int32)
    P64=P.astype(np.float64)
    normal=np.array([0.,0.,float(sign)])
    I=P64-skin*normal
    D=I-fat*normal
    return reconstruct_connected_layers(
        P,F,I-P64,D-I,boundary,P64[boundary],I[boundary],D[boundary],
        skin_depth_m=skin,fat_depth_m=fat,orientation_sign=sign,
        active_vertex_indices=np.arange(n,dtype=np.int32))


def test_harmonic_patch_matches_exact_layer_boundary_and_keeps_outside_exact():
    P,F=grid_mesh()
    active=np.asarray([0,1,2,4,5,6,8,9,10],dtype=np.int32)
    boundary=np.asarray([0,1,2,4,6,8,9,10],dtype=np.int32)
    out=invoke(P,F,active=active,boundary=boundary)
    assert np.array_equal(out.outer_points[boundary],P[boundary].astype(float))
    base_i=P.astype(float)+np.tile([0.,0.,-.004],(len(P),1))
    base_d=base_i+np.tile([0.,0.,-.006],(len(P),1))
    assert np.array_equal(out.inner_points[boundary],base_i[boundary])
    assert np.array_equal(out.deep_points[boundary],base_d[boundary])
    outside=~out.active_vertex_mask
    assert np.array_equal(out.inner_points[outside],base_i[outside])
    assert np.array_equal(out.deep_points[outside],base_d[outside])
    assert out.diagnostics["outer_boundary_exact"] and out.diagnostics["inner_boundary_exact"]
    assert out.diagnostics["deep_boundary_exact"] and out.diagnostics["outside_patch_preserved_from_baseline_offsets"]
    assert out.diagnostics["physical_or_native_admission"] is False
    assert np.isclose(out.inner_points[5,2],-.004,atol=1e-12)
    assert np.isclose(out.deep_points[5,2],-.010,atol=1e-12)


def test_positive_scalar_depths_define_initial_normal_shells():
    P,F=grid_mesh()
    out=invoke_matching_normal_shell(P,F,skin=.02,fat=.03)
    assert np.allclose(out.outer_points[:,2]-out.inner_points[:,2],.02,atol=1e-12)
    assert np.allclose(out.inner_points[:,2]-out.deep_points[:,2],.03,atol=1e-12)
    assert np.allclose(out.outward_normals,[0.,0.,1.])


def test_orientation_sign_is_explicit_and_reverses_normals():
    P,F=grid_mesh()
    plus=invoke_matching_normal_shell(P,F,sign=1)
    minus=invoke_matching_normal_shell(P,F,sign=-1)
    assert np.allclose(plus.outward_normals,-minus.outward_normals)
    assert np.allclose(plus.inner_points[:,2],-.02)
    assert np.allclose(minus.inner_points[:,2],.02)


def test_missing_boundary_rejects_unanchored_active_mesh():
    P,F=grid_mesh()
    with pytest.raises(ValueError,match="no fixed boundary"):
        reconstruct_connected_layers(P,F,np.zeros_like(P),np.zeros_like(P),np.array([],dtype=np.int32),
            np.empty((0,3)),np.empty((0,3)),np.empty((0,3)),skin_depth_m=.01,fat_depth_m=.01,
            orientation_sign=1,active_vertex_indices=np.arange(len(P)))


def test_every_disconnected_active_component_requires_a_boundary_anchor():
    P,F=grid_mesh()
    active=np.array([0,1,4,10,11,15],dtype=np.int32)
    boundary=np.array([0,1],dtype=np.int32)
    with pytest.raises(ValueError,match="components without fixed boundary"):
        invoke(P,F,active=active,boundary=boundary)


def test_empty_active_patch_is_an_exact_baseline_noop():
    P,F=grid_mesh()
    out=invoke(P,F,active=np.array([],dtype=np.int32),boundary=np.array([],dtype=np.int32))
    base_i=P.astype(float)+np.tile([0.,0.,-.004],(len(P),1))
    base_d=base_i+np.tile([0.,0.,-.006],(len(P),1))
    assert np.array_equal(out.outer_points,P.astype(float))
    assert np.array_equal(out.inner_points,base_i)
    assert np.array_equal(out.deep_points,base_d)
    assert out.diagnostics["active_graph_component_count"]==0


@pytest.mark.parametrize("skin,fat",[(0.,.1),(-.1,.1),(np.nan,.1),(.1,np.inf)])
def test_nonpositive_or_nonfinite_layer_depth_is_rejected(skin,fat):
    P,F=grid_mesh()
    with pytest.raises(ValueError,match="finite and positive"):
        invoke(P,F,skin=skin,fat=fat)


@pytest.mark.parametrize("boundary,active,match",[
    (np.array([0,0],dtype=np.int32),np.arange(16,dtype=np.int32),"duplicates"),
    (np.array([16],dtype=np.int32),np.arange(16,dtype=np.int32),"out-of-range"),
    (np.array([0],dtype=np.int32),np.array([1,2,3],dtype=np.int32),"part of active"),
])
def test_invalid_or_inconsistent_boundary_rows_reject(boundary,active,match):
    P,F=grid_mesh()
    with pytest.raises(ValueError,match=match):
        invoke(P,F,active=active,boundary=boundary)


def test_active_vertex_without_surface_normal_rejects():
    P=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[2.,2.,2.]],dtype=np.float32)
    F=np.array([[0,1,2]],dtype=np.int32)
    with pytest.raises(ValueError,match="no oriented normal"):
        invoke(P,F,active=np.array([3],dtype=np.int32),boundary=np.array([3],dtype=np.int32))


def test_degenerate_face_rejects_before_layer_reconstruction():
    P=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]],dtype=np.float32)
    F=np.array([[0,1,2]],dtype=np.int32)
    with pytest.raises(ValueError,match="degenerate"):
        reconstruct_connected_layers(P,F,np.zeros_like(P),np.zeros_like(P),np.array([0]),
            P[[0]],P[[0]],P[[0]],skin_depth_m=.01,fat_depth_m=.02,orientation_sign=1,
            active_vertex_indices=np.array([0],dtype=np.int32))



def test_nonuniform_per_vertex_depths_are_preserved_on_planar_mesh():
    P,F=grid_mesh()
    n=len(P)
    skin=np.linspace(.012,.027,n)
    fat=np.linspace(.021,.039,n)
    normal=np.array([0.,0.,1.])
    desired_I=P.astype(np.float64)-skin[:,None]*normal
    desired_D=desired_I-fat[:,None]*normal
    boundary=np.asarray([0,1,2,3,4,7,8,11,12,13,14,15],dtype=np.int32)
    zeros=np.zeros((n,3),dtype=np.float64)
    out=reconstruct_connected_layers(
        P,F,zeros,zeros,boundary,P[boundary],desired_I[boundary],desired_D[boundary],
        skin_depth_m=skin,fat_depth_m=fat,orientation_sign=1,
        active_vertex_indices=np.arange(n,dtype=np.int32))
    assert np.allclose(out.outer_points-out.inner_points,skin[:,None]*normal,atol=1e-12)
    assert np.allclose(out.inner_points-out.deep_points,fat[:,None]*normal,atol=1e-12)
    assert out.diagnostics["requested_skin_depth_min_m"] == pytest.approx(skin.min())
    assert out.diagnostics["requested_fat_depth_max_m"] == pytest.approx(fat.max())


@pytest.mark.parametrize("field",[
    np.ones(15,dtype=np.float64),
    np.ones((16,1),dtype=np.float64),
    np.full(16,np.nan,dtype=np.float64),
    np.r_[np.ones(15),0.],
])
def test_per_vertex_depth_shape_and_values_are_validated(field):
    P,F=grid_mesh()
    with pytest.raises(ValueError,match="finite positive scalar|finite and positive"):
        invoke(P,F,skin=field)
