import numpy as np
import pytest
from human_skin_fat_layer_reconstruction import (
    reconstruct_connected_layers, split_tetrahedra_by_linear_scalar_field,
    extract_joint_zero_level_facets,
)


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



def unit_tetrahedron(offset=0.0):
    return np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=np.float64)+offset


def _cell_volumes(points,tets):
    out=[]
    for t in tets:
        q=points[t]
        out.append(abs(np.dot(q[1]-q[0],np.cross(q[2]-q[0],q[3]-q[0])))/6.)
    return np.asarray(out)


def test_scalar_split_conserves_parent_volume_mass_material_and_linear_interpolation():
    P=unit_tetrahedron()
    T=np.asarray([[0,1,2,3]],dtype=np.int32)
    phi=np.asarray([-1.,1.,1.,1.])
    out=split_tetrahedra_by_linear_scalar_field(
        P,T,phi,global_vertex_ids=np.asarray([40,10,30,20]),
        material_ids=np.asarray([7]),cell_mass=np.asarray([2.5]))
    assert out.diagnostics["intersection_vertex_count"]==3
    assert np.array_equal(np.unique(out.parent_cell_rows),np.asarray([0]))
    assert set(out.side_signs.tolist())=={-1,1}
    assert np.array_equal(out.points[:len(P)],P)
    assert np.allclose(out.interpolation_weights@P,out.points,rtol=0,atol=2e-16)
    assert np.allclose(out.interpolation_weights@phi,out.scalar_values,rtol=0,atol=1e-15)
    assert np.max(np.abs(out.scalar_values[len(P):]))<1e-15
    assert np.allclose(np.asarray(out.interpolation_weights.sum(axis=1)).ravel(),1.)
    assert np.max(np.diff(out.interpolation_weights.indptr))<=2
    vols=_cell_volumes(out.points,out.tetrahedra)
    assert vols[out.side_signs<0].sum()==pytest.approx(1./48.,rel=1e-12)
    assert vols[out.side_signs>0].sum()==pytest.approx(7./48.,rel=1e-12)
    assert vols.sum()==pytest.approx(1./6.,rel=1e-12)
    assert out.cell_masses.sum()==pytest.approx(2.5,rel=1e-12)
    assert np.all(out.material_ids==7)
    assert out.diagnostics["all_source_cells_have_children"]
    assert out.diagnostics["float32_nonpositive_cell_count"]==0
    assert out.diagnostics["float32_quality_warning"] is False
    assert out.diagnostics["parent_volume_max_relative_error"]<1e-12


def test_all_scalar_sign_permutations_make_positive_conservative_children():
    from itertools import permutations
    P=unit_tetrahedron()
    T=np.asarray([[0,1,2,3]],dtype=np.int32)
    values=(-1.,-2.,3.,4.)
    for order in permutations(values):
        out=split_tetrahedra_by_linear_scalar_field(P,T,np.asarray(order),
            global_vertex_ids=np.asarray([101,7,55,23]))
        assert set(out.side_signs.tolist())=={-1,1}
        assert np.all(_cell_volumes(out.points,out.tetrahedra)>0.)
        assert out.diagnostics["float32_nonpositive_cell_count"]==0
        assert out.diagnostics["parent_volume_max_relative_error"]<1e-12
        # The global edge cache and pulling order are deterministic.
        again=split_tetrahedra_by_linear_scalar_field(P,T,np.asarray(order),
            global_vertex_ids=np.asarray([101,7,55,23]))
        assert np.array_equal(out.points,again.points)
        assert np.array_equal(out.tetrahedra,again.tetrahedra)
        assert np.array_equal(out.side_signs,again.side_signs)


def test_shared_clipped_face_uses_identical_global_pulling_triangulation():
    P=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,-1.]])
    T=np.asarray([[0,1,2,3],[0,2,1,4]],dtype=np.int32)
    phi=np.asarray([-1.,1.,-1.,.5,.75])
    out=split_tetrahedra_by_linear_scalar_field(P,T,phi,
        global_vertex_ids=np.asarray([90,10,80,30,40]),
        material_ids=np.asarray([4,9]),cell_mass=np.asarray([1.25,2.75]))
    # The source tetrahedra have opposite sides of z=0 in the same shared
    # source triangle. Their clipped boundary triangulations must match as
    # exact vertex-index triples, once per parent and half-domain.
    for side in (-1,1):
        per_parent=[]
        for parent in (0,1):
            local=[]
            rows=np.flatnonzero((out.side_signs==side)&(out.parent_cell_rows==parent))
            for cellrow in rows:
                tet=out.tetrahedra[cellrow]
                for face in ((tet[1],tet[2],tet[3]),(tet[0],tet[3],tet[2]),
                             (tet[0],tet[1],tet[3]),(tet[0],tet[2],tet[1])):
                    xyz=out.points[np.asarray(face)]
                    if np.all(np.abs(xyz[:,2])<1e-14):
                        local.append(tuple(sorted(map(int,face))))
            assert local, f"parent {parent} has no clipped shared-face triangles for side {side}"
            assert len(local)==len(set(local)), "a parent emitted duplicate shared-face triangles"
            per_parent.append(set(local))
        assert per_parent[0]==per_parent[1]
    edge_keys=out.intersection_edge_global_ids.tolist()
    assert edge_keys.count([10,90])==1
    crossing_row=out.global_vertex_order_keys.index((1,10,90))
    for parent in (0,1):
        assert np.any(out.tetrahedra[out.parent_cell_rows==parent]==crossing_row)
    assert out.diagnostics["all_source_cells_have_children"]
    assert np.all(out.scalar_values[out.tetrahedra] * out.side_signs[:,None] >= -1e-14)
    assert np.array_equal(out.material_ids, np.asarray([4,9])[out.parent_cell_rows])
    parent_mass=np.bincount(out.parent_cell_rows,weights=out.cell_masses,minlength=2)
    assert np.allclose(parent_mass,[1.25,2.75],rtol=1e-12,atol=1e-14)


def test_zero_field_vertices_are_reused_without_zero_volume_children():
    P=unit_tetrahedron()
    T=np.asarray([[0,1,2,3]],dtype=np.int32)
    positive=split_tetrahedra_by_linear_scalar_field(P,T,np.asarray([0.,1.,2.,3.]))
    assert len(positive.points)==len(P)
    assert positive.side_signs.tolist()==[1]
    assert _cell_volumes(positive.points,positive.tetrahedra).sum()==pytest.approx(1./6.)
    # Owner quality uses the longest of all six edges, including the edge
    # opposite source vertex zero (length sqrt(2) in this fixture).
    assert positive.diagnostics["float32_min_owner_quality_det_over_longest_edge_cubed"] == pytest.approx(1.0/(2.0**1.5),rel=1e-6)
    mixed=split_tetrahedra_by_linear_scalar_field(P,T,np.asarray([0.,-1.,1.,-1.]))
    assert set(mixed.side_signs.tolist())=={-1,1}
    assert len(np.unique(mixed.tetrahedra))>=4
    assert mixed.diagnostics["parent_volume_max_relative_error"]<1e-12
    assert mixed.diagnostics["float32_nonpositive_cell_count"]==0


def test_tiny_cut_that_collapses_in_float32_fails_instead_of_dropping_cells():
    P=unit_tetrahedron(offset=1000.)
    T=np.asarray([[0,1,2,3]],dtype=np.int32)
    phi=np.asarray([-1e-8,1.,1.,1.])
    with pytest.raises(ValueError,match="Float32 emission contains nonpositive"):
        split_tetrahedra_by_linear_scalar_field(P,T,phi)


def test_subnormal_opposite_signs_fail_closed_without_missed_edge_cache():
    P=unit_tetrahedron()
    T=np.asarray([[0,1,2,3]],dtype=np.int32)
    phi=np.asarray([-1e-300,1e-300,1e-300,1e-300])
    # The edge crossing is real despite the product of endpoint values
    # underflowing in float64. The resulting midpoint remains representable.
    out=split_tetrahedra_by_linear_scalar_field(P,T,phi)
    assert out.diagnostics["intersection_vertex_count"]==3
    assert out.diagnostics["all_source_cells_have_children"]
    assert out.diagnostics["float32_nonpositive_cell_count"]==0


def test_negative_source_orientation_is_rejected_without_reindexing():
    P=unit_tetrahedron()
    T=np.asarray([[0,1,3,2]],dtype=np.int32)
    with pytest.raises(ValueError,match="negative orientation"):
        split_tetrahedra_by_linear_scalar_field(P,T,np.asarray([-1.,1.,1.,1.]))


def test_identically_zero_parent_field_is_rejected_as_ambiguous():
    with pytest.raises(ValueError,match="identically zero scalar field"):
        split_tetrahedra_by_linear_scalar_field(unit_tetrahedron(),np.asarray([[0,1,2,3]]),np.zeros(4))


def test_near_vertex_cut_keeps_positive_thin_children_and_reports_quality_warning():
    # The requested scalar plane passes close to one source vertex. The large
    # clipped side necessarily contains a thin positive tetrahedron; it must be
    # retained and conserved, while the diagnostic warning exposes its quality.
    P=unit_tetrahedron()
    T=np.asarray([[0,1,2,3]],dtype=np.int32)
    out=split_tetrahedra_by_linear_scalar_field(
        P,T,np.asarray([-1.0e-3,1.0,1.0,1.0]),
        global_vertex_ids=np.asarray([40,10,30,20]),
        material_ids=np.asarray([7]),cell_mass=np.asarray([2.5]))
    q=out.diagnostics["float32_min_owner_quality_det_over_longest_edge_cubed"]
    assert 0.0 < q < out.diagnostics["float32_quality_warning_threshold_det_over_longest_edge_cubed"]
    assert out.diagnostics["float32_quality_warning"] is True
    assert out.diagnostics["float32_quality_warning_cell_count"] > 0
    assert out.diagnostics["float32_quality_warning_parent_rows_first16"] == [0]
    assert out.diagnostics["float32_nonpositive_cell_count"] == 0
    assert np.all(_cell_volumes(out.points_float32,out.tetrahedra) > 0.0)
    assert out.diagnostics["all_source_cells_have_children"]
    assert np.all(out.material_ids == 7)
    assert out.cell_masses.sum() == pytest.approx(2.5,rel=1e-12)
    assert _cell_volumes(out.points,out.tetrahedra).sum() == pytest.approx(1.0/6.0,rel=1e-12)


def test_polygon_area_vector_is_translation_stable_for_tiny_clipped_faces():
    from human_skin_fat_layer_reconstruction import _polygon_area_vector

    local = np.asarray([[0., 0., 0.], [3e-9, 0., 0.],
                        [3e-9, 2e-9, 0.], [0., 2e-9, 0.]])
    face = np.arange(4, dtype=np.int64)
    local_area = _polygon_area_vector(local, face)
    world = local + np.asarray([0.25, -0.5, 0.75])
    world_area = _polygon_area_vector(world, face)
    expected = np.asarray([0., 0., 1.2e-17])
    assert np.linalg.norm(local_area - expected) <= 1e-30
    assert np.allclose(world_area, expected, rtol=1e-7, atol=1e-30)
    assert np.linalg.norm(world_area) > 0.0


def test_retained_side_filters_only_after_full_parent_volume_and_mass_partition():
    # A healthy positive-domain tet is retained while a separate tiny negative
    # exterior tet is F64-valid but collapses at this Float32 world offset.
    safe=unit_tetrahedron()
    tiny=np.asarray([[1000.,1000.,1000.], [1000.00001,1000.,1000.],
                     [1000.,1001.,1000.], [1000.,1000.,1001.]])
    P=np.vstack((safe,tiny))
    T=np.asarray([[0,1,2,3],[4,5,6,7]],dtype=np.int32)
    phi=np.asarray([1.,1.,1.,1.,-1.,-1.,-1.,-1.])
    out=split_tetrahedra_by_linear_scalar_field(
        P,T,phi,material_ids=np.asarray([3,7]),cell_mass=np.asarray([2.5,4.]),
        retained_sides=(1,))
    assert len(out.tetrahedra) > 0
    assert np.all(out.side_signs == 1)
    assert np.all(out.parent_cell_rows == 0)
    assert np.all(_cell_volumes(out.points_float32,out.tetrahedra) > 0.)
    assert out.diagnostics["float32_nonpositive_cell_count"] == 0
    assert out.diagnostics["selected_sides"] == [1]
    assert out.diagnostics["prefilter_child_count"] == 2
    assert out.diagnostics["excluded_child_count"] == 1
    assert np.allclose(out.retained_parent_volumes + out.excluded_parent_volumes,
                       _cell_volumes(P,T),rtol=1e-12,atol=1e-15)
    assert out.retained_parent_volumes[1] == 0.
    assert out.excluded_parent_volumes[1] > 0.
    assert np.allclose(out.retained_parent_masses + out.excluded_parent_masses,
                       [2.5,4.],rtol=1e-12,atol=1e-14)
    assert out.retained_parent_masses[0] == pytest.approx(2.5)
    assert out.excluded_parent_masses[1] == pytest.approx(4.)
    assert out.cell_masses.sum() == pytest.approx(2.5)
    assert np.all(out.material_ids == 3)
    assert out.diagnostics["parent_volume_max_relative_error"] < 1e-12
    assert out.diagnostics["parent_mass_max_abs_partition_error"] < 1e-12



def test_mixed_parent_retained_side_has_exact_excluded_volume_and_mass():
    P=unit_tetrahedron()
    T=np.asarray([[0,1,2,3]],dtype=np.int32)
    phi=np.asarray([-1.,1.,1.,1.])
    full=split_tetrahedra_by_linear_scalar_field(
        P,T,phi,material_ids=np.asarray([11]),cell_mass=np.asarray([3.]))
    positive=split_tetrahedra_by_linear_scalar_field(
        P,T,phi,material_ids=np.asarray([11]),cell_mass=np.asarray([3.]),
        retained_sides=(1,))
    assert np.all(positive.side_signs == 1)
    assert positive.diagnostics["prefilter_child_count"] == len(full.tetrahedra)
    assert positive.diagnostics["output_tetrahedron_count"] < positive.diagnostics["prefilter_child_count"]
    assert positive.retained_parent_volumes[0] + positive.excluded_parent_volumes[0] == pytest.approx(1./6.,rel=1e-12)
    assert positive.retained_parent_masses[0] + positive.excluded_parent_masses[0] == pytest.approx(3.,rel=1e-12)
    assert positive.retained_parent_volumes[0] == pytest.approx(_cell_volumes(positive.points,positive.tetrahedra).sum(),rel=1e-12)
    assert positive.retained_parent_masses[0] == pytest.approx(positive.cell_masses.sum(),rel=1e-12)
    assert positive.diagnostics["unused_intersection_vertex_count_after_filter"] == 0
    assert np.array_equal(positive.material_ids,np.asarray([11])[positive.parent_cell_rows])
    assert positive.diagnostics["float32_nonpositive_cell_count"] == 0


def test_default_and_requesting_collapsed_negative_side_still_fail():
    safe=unit_tetrahedron()
    tiny=np.asarray([[1000.,1000.,1000.], [1000.00001,1000.,1000.],
                     [1000.,1001.,1000.], [1000.,1000.,1001.]])
    P=np.vstack((safe,tiny))
    T=np.asarray([[0,1,2,3],[4,5,6,7]],dtype=np.int32)
    phi=np.asarray([1.,1.,1.,1.,-1.,-1.,-1.,-1.])
    with pytest.raises(ValueError,match="Float32 emission contains nonpositive"):
        split_tetrahedra_by_linear_scalar_field(P,T,phi)
    with pytest.raises(ValueError,match="Float32 emission contains nonpositive"):
        split_tetrahedra_by_linear_scalar_field(P,T,phi,retained_sides=(-1,))


@pytest.mark.parametrize("sides", [(), (0,), (1,1), (True,)])
def test_retained_sides_must_be_nonempty_unique_signed_halfdomains(sides):
    P=unit_tetrahedron()
    T=np.asarray([[0,1,2,3]],dtype=np.int32)
    with pytest.raises(ValueError,match="retained_sides"):
        split_tetrahedra_by_linear_scalar_field(P,T,np.asarray([-1.,1.,1.,1.]),retained_sides=sides)


def test_retained_side_with_no_selected_cells_fails_explicitly():
    P=unit_tetrahedron()
    T=np.asarray([[0,1,2,3]],dtype=np.int32)
    with pytest.raises(ValueError,match="selects no output cells"):
        split_tetrahedra_by_linear_scalar_field(P,T,np.ones(4),retained_sides=(-1,))
    out=split_tetrahedra_by_linear_scalar_field(P,T,np.ones(4),retained_sides=(1,))
    assert out.retained_parent_masses is None
    assert out.excluded_parent_masses is None
    assert out.retained_parent_volumes[0] == pytest.approx(1./6.)
    assert out.excluded_parent_volumes[0] == 0.

from human_skin_fat_layer_reconstruction import snap_shared_scalar_interface_vertices

def test_shared_interface_snap_keeps_coordinates_and_records_explicit_change():
    P=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    T=np.array([[0,1,2,3]],dtype=np.int32)
    phi=np.array([1.e-9,1.,-1.,-1.])
    original_phi=phi.copy();original_P=P.copy()
    out,stats,ledger=snap_shared_scalar_interface_vertices(P,T,phi,.001,fraction=.001)
    assert out[0]==0 and np.array_equal(out[1:],phi[1:])
    assert np.array_equal(P,original_P) and np.array_equal(phi,original_phi)
    assert stats["snapped_vertex_count"]==1
    assert np.array_equal(ledger["snapped_vertex_rows"],[0])
    assert np.array_equal(ledger["snap_delta_m"],out-original_phi)
    assert np.all(ledger["max_incident_crossing_displacement_m"][ledger["snapped_vertex_rows"]] <= ledger["allowed_displacement_m"][ledger["snapped_vertex_rows"]])
    split=split_tetrahedra_by_linear_scalar_field(P,T,out)
    X=split.points.astype(np.float32).astype(float)[split.tetrahedra]
    det=np.einsum("ij,ij->i",X[:,1]-X[:,0],np.cross(X[:,2]-X[:,0],X[:,3]-X[:,0]))
    assert np.all(det>0)
    assert np.isclose(det.sum(),1.,rtol=1.e-14,atol=0)
    assert set(split.side_signs)=={-1,1}

def test_shared_interface_snap_respects_every_incident_crossing_not_nearest_only():
    P=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    T=np.array([[0,1,2,3]],dtype=np.int32)
    phi=np.array([1.e-9,-1.,-1.e-9,1.])
    out,stats,ledger=snap_shared_scalar_interface_vertices(P,T,phi,.001,fraction=.001)
    assert out[0]==phi[0] # edge 0--2 crosses halfway along a one-metre edge
    assert ledger["max_incident_crossing_displacement_m"][0]==.5

def test_shared_interface_snap_is_independent_of_tetrahedron_visit_order():
    P=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,-1.]])
    T=np.array([[0,1,2,3],[0,2,1,4]],dtype=np.int32)
    phi=np.array([1.e-9,1.,-1.,-1.,-1.])
    a,sa,la=snap_shared_scalar_interface_vertices(P,T,phi,.001,fraction=.001)
    b,sb,lb=snap_shared_scalar_interface_vertices(P,T[::-1],phi,.001,fraction=.001)
    assert np.array_equal(a,b) and sa==sb
    for key in la: assert np.array_equal(la[key],lb[key])

@pytest.mark.parametrize("fraction",[-1.,.5,float("nan"),float("inf")])
def test_shared_interface_snap_rejects_invalid_explicit_budget(fraction):
    P=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    with pytest.raises(ValueError,match="fraction"):
        snap_shared_scalar_interface_vertices(P,np.array([[0,1,2,3]]),np.array([-1.,1.,1.,1.]),.001,fraction=fraction)

def test_shared_interface_snap_rejects_ambiguous_all_zero_parent():
    P=np.array([[0.,0.,0.],[1.e-9,0.,0.],[0.,1.e-9,0.],[0.,0.,1.e-9]])
    with pytest.raises(ValueError,match="ambiguous all-zero"):
        snap_shared_scalar_interface_vertices(P,np.array([[0,1,2,3]]),np.array([-1.e-9,1.e-9,1.e-9,1.e-9]),.001,fraction=.001)

def test_shared_interface_snap_rejects_invalid_geometry_and_fractional_indices():
    P=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    with pytest.raises(ValueError,match="positive"):
        snap_shared_scalar_interface_vertices(P,np.array([[0,2,1,3]]),np.array([-1.,1.,1.,1.]),.001,fraction=.001)
    with pytest.raises(ValueError,match="integer"):
        snap_shared_scalar_interface_vertices(P,np.array([[0.,1.,2.,3.]]),np.array([-1.,1.,1.,1.]),.001,fraction=.001)

def test_shared_interface_snap_zero_budget_is_exact_identity():
    P=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    phi=np.array([1.e-9,1.,-1.,-1.])
    out,stats,ledger=snap_shared_scalar_interface_vertices(P,np.array([[0,1,2,3]]),phi,.001,fraction=0.)
    assert np.array_equal(out,phi) and stats["snapped_vertex_count"]==0


def _joint_tet_points():
    return np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32)


def test_joint_zero_facets_1_to_3_and_2_to_2_use_stable_triangles():
    P = _joint_tet_points()
    T = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    one_three = extract_joint_zero_level_facets(P, T, [-1., 1., 1., 1.], [-.5, 2., 2., 2.])
    assert np.count_nonzero(one_three.interface_ids == 1) == 1
    assert np.count_nonzero(one_three.interface_ids == 2) == 1
    two_two = extract_joint_zero_level_facets(P, T, [-1., -1., 1., 1.], [-.5, -.5, 2., 2.])
    assert np.count_nonzero(two_two.interface_ids == 1) == 2
    assert np.count_nonzero(two_two.interface_ids == 2) == 2
    again = extract_joint_zero_level_facets(P, T, [-1., -1., 1., 1.], [-.5, -.5, 2., 2.])
    assert np.array_equal(two_two.facets, again.facets)
    assert np.array_equal(two_two.interface_ids, again.interface_ids)
    assert np.array_equal(two_two.points_float32, again.points_float32)


def test_joint_facet_intersections_are_shared_across_parent_tets():
    P = np.asarray([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.], [0.,0.,-1.]], dtype=np.float32)
    T = np.asarray([[0,1,2,3], [0,2,1,4]], dtype=np.int32)
    gids = np.asarray([40, 10, 30, 20, 50], dtype=np.int64)
    phiI = np.asarray([-1., 1., 1., 1., -1.])
    phiD = phiI + 0.25
    out = extract_joint_zero_level_facets(P, T, phiI, phiD, global_vertex_ids=gids,
                                          parent_tet_ids=np.asarray([700, 701]))
    rows = np.flatnonzero((out.point_interface_ids == 1) &
                          np.all(out.point_source_edge_global_ids == [10, 40], axis=1))
    assert len(rows) == 1
    Wrow = out.point_interpolation_weights.getrow(int(rows[0]))
    assert Wrow.nnz == 2
    assert np.isclose(Wrow.data.sum(), 1.)
    assert 700 in out.facet_parent_tet_rows and 701 in out.facet_parent_tet_rows


def test_joint_zero_vertices_are_reused_and_shared_facet_keeps_both_parents():
    P = np.asarray([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.], [0.,0.,-1.]], dtype=np.float32)
    T = np.asarray([[0,1,2,3], [0,2,1,4]], dtype=np.int32)
    phiI = np.asarray([0., 0., 0., 1., -1.])
    phiD = phiI + 2.0
    out = extract_joint_zero_level_facets(P, T, phiI, phiD)
    inner = out.facets[out.interface_ids == 1]
    assert len(inner) == 1
    assert set(map(int, inner[0])) == {0, 1, 2}
    row = int(np.flatnonzero(out.interface_ids == 1)[0])
    lo, hi = out.facet_parent_indptr[row:row + 2]
    assert set(map(int, out.facet_parent_local_rows[lo:hi])) == {0, 1}
    assert len(out.points_float64) == len(P)


def test_joint_interface_winding_is_outward_from_positive_field_and_deterministic():
    P = _joint_tet_points()
    T = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    phiI = np.asarray([-1., 1., 1., 1.])
    phiD = np.asarray([-.5, 2., 2., 2.])
    out = extract_joint_zero_level_facets(P, T, phiI, phiD)
    for fi, tri in enumerate(out.facets):
        parent = int(out.facet_parent_local_rows[out.facet_parent_indptr[fi]])
        phi = phiI if out.interface_ids[fi] == 1 else phiD
        tet = T[parent]
        x = P[tet].astype(np.float64)
        dv = phi[tet]
        grad = np.linalg.solve(x[1:] - x[0], dv[1:] - dv[0])
        q = out.points_float32[tri].astype(np.float64)
        area = np.cross(q[1] - q[0], q[2] - q[0])
        assert np.dot(area, -grad) > 0.
        assert np.allclose(out.point_scalar_values_m[tri, out.interface_ids[fi] - 1], 0., atol=1e-15)


def test_joint_fields_reject_crossing_or_nonpositive_fat_order():
    P = _joint_tet_points()
    T = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    with pytest.raises(ValueError, match="strictly positive"):
        extract_joint_zero_level_facets(P, T, [-1., 1., 1., 1.], [-2., 1., 1., 1.])


def test_joint_extraction_rejects_float32_collapsed_facet_instead_of_dropping():
    P = np.asarray([[1.,1.,1.], [1.01,1.,1.], [1.,1.01,1.], [1.,1.,1.01]], dtype=np.float32)
    T = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    with pytest.raises(ValueError, match="collapsed interface facets"):
        extract_joint_zero_level_facets(P, T, [-1e-9, 1., 1., 1.], [-0.5e-9, 2., 2., 2.])


import numpy as np
from human_skin_fat_layer_reconstruction import extract_joint_zero_level_facets


def test_strict_nodal_hfat_orders_affine_zero_surfaces_everywhere_in_tet():
    points=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=np.float32)
    tets=np.asarray([[0,1,2,3]],dtype=np.int32)
    phi_i=np.asarray([-1.,1.,1.,1.])
    phi_d=phi_i+0.2
    hfat=phi_d-phi_i
    assert np.all(hfat>0.)
    out=extract_joint_zero_level_facets(points,tets,phi_i,phi_d)
    inner_rows=set(map(int,out.facets[out.interface_ids==1].ravel()))
    deep_rows=set(map(int,out.facets[out.interface_ids==2].ravel()))
    assert inner_rows.isdisjoint(deep_rows)
    # Positive affine weights span the entire tetrahedron; the ordered level
    # fields therefore cannot be simultaneously zero anywhere in its interior.
    weights=np.asarray([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.],
                        [.25,.25,.25,.25],[.1,.2,.3,.4],[.4,.3,.2,.1]])
    vi=weights@phi_i
    vd=weights@phi_d
    assert np.all(vd-vi>0.)
    assert not np.any((np.abs(vi)<1e-14)&(np.abs(vd)<1e-14))


from human_skin_fat_layer_reconstruction import facet_incident_tetrahedron_quality_upper_bound

def test_fixed_facet_quality_bound_limits_random_incident_cells():
    rng=np.random.default_rng(481)
    p=rng.normal(size=(400,4,3)).astype(np.float32).astype(float)
    f=np.arange(1200).reshape(-1,3)
    bound=facet_incident_tetrahedron_quality_upper_bound(p[:,:3].reshape(-1,3),f)
    edges=np.stack([p[:,b]-p[:,a] for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]],axis=1)
    longest=np.linalg.norm(edges,axis=2).max(axis=1)
    det=np.abs(np.einsum("ij,ij->i",p[:,1]-p[:,0],np.cross(p[:,2]-p[:,0],p[:,3]-p[:,0])))
    assert np.all(det/longest**3 <= bound*(1+1e-14))

def test_fixed_facet_quality_obstruction_is_scale_independent():
    p=np.array([[0.,0.,0.],[1.,0.,0.],[.5,1e-6,0.]],dtype=np.float32)
    f=np.array([[0,1,2]],dtype=np.int32)
    a=facet_incident_tetrahedron_quality_upper_bound(p,f)
    b=facet_incident_tetrahedron_quality_upper_bound(p*1024,f)
    assert a[0]<1e-5 and np.array_equal(a,b)
    p[2]=[0.,1.,0.]
    assert facet_incident_tetrahedron_quality_upper_bound(p,f)[0]>1e-5

def test_fixed_facet_quality_empty_and_collapsed_are_explicit():
    p=np.zeros((3,3),dtype=np.float32)
    assert facet_incident_tetrahedron_quality_upper_bound(p,np.empty((0,3),dtype=int)).shape==(0,)
    assert np.array_equal(facet_incident_tetrahedron_quality_upper_bound(p,np.array([[0,1,2]])),[0.])

@pytest.mark.parametrize("facets", [np.array([[0.,1.,2.]]),np.array([[-1,1,2]]),np.array([[0,1,3]])])
def test_fixed_facet_quality_rejects_bad_indices(facets):
    with pytest.raises(ValueError):
        facet_incident_tetrahedron_quality_upper_bound(np.zeros((3,3)),facets)

def test_fixed_facet_quality_rejects_float32_coordinate_overflow():
    with pytest.raises(ValueError,match="Float32"):
        facet_incident_tetrahedron_quality_upper_bound(np.full((3,3),1e100),np.array([[0,1,2]]))
