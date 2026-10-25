from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np

SOURCE=Path(__file__).with_name("build_outward_nearest_union_targets_284.py")
SPEC=importlib.util.spec_from_file_location("nearest_face_union_284",SOURCE)
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


def _cube_local_bounds(xlo=0.0,xhi=1.0,ylo=-1.0,yhi=1.0,zlo=0.0,zhi=1.0):
    v=np.array([[xlo,ylo,zlo],[xhi,ylo,zlo],[xhi,yhi,zlo],[xlo,yhi,zlo],
                [xlo,ylo,zhi],[xhi,ylo,zhi],[xhi,yhi,zhi],[xlo,yhi,zhi]],dtype=np.float64)
    f=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],
                [0,1,5],[0,5,4],[1,2,6],[1,6,5],
                [2,3,7],[2,7,6],[3,0,4],[3,4,7]],dtype=np.int64)
    return v,f

def _exact_locator(vertices,faces):
    import importlib,sys
    sys.path.insert(0,str(MOD.EXACT.parent.parent))
    exact=importlib.import_module("numilab_human.cardiac_cavity_intersections")
    fr_vertices=tuple(tuple(MOD.Fraction.from_float(float(v)) for v in row) for row in vertices)
    fr_faces=tuple(tuple(int(i) for i in row) for row in faces)
    return exact,exact.prepare_point_location(exact._records(fr_vertices,fr_faces))

def test_virtual_f64_sample_survives_nanometer_slab_lost_by_f32_rounding():
    # A modest shear makes the local bank boundary cut across independently
    # rounded world coordinates. The virtual F64 point is inside by <1 nm,
    # while its diagnostic F32-rounded copy is outside.
    points=np.array([
        [np.float32(.59999996),np.float32(-.2),np.float32(.2)],
        [np.float32(.6),np.nextafter(np.float32(-.2),np.float32(np.inf)),np.float32(.2)],
        [np.float32(.59999996),np.float32(-.2),np.float32(.8)],
    ],dtype=np.float32)
    weights=np.array([.57,.429,.001],dtype=np.float64)
    virtual=MOD.virtual_sample_world_f64(points,np.array([0,1,2],dtype=np.int64),weights)
    rounded=virtual.astype(np.float32).astype(np.float64)
    M=np.array([[1.,3.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=np.float64)
    local_virtual=virtual@M.T
    local_rounded=rounded@M.T
    assert 0.0<local_virtual[0]<1e-9
    assert local_rounded[0]<0.0
    v,f=_cube_local_bounds(xlo=0.0,xhi=1.0,ylo=-1.0,yhi=1.0,zlo=0.0,zhi=1.0)
    exact,locator=_exact_locator(v,f)
    assert exact.point_location_prepared_indexed(MOD.fracpoint(local_virtual),locator)["location"]=="inside"
    assert exact.point_location_prepared_indexed(MOD.fracpoint(local_rounded),locator)["location"]=="outside"

def test_strict_interior_sample_reaches_tiny_inward_fraction():
    # The skin triangle enters a 0.1 mm bank slab at x=.5. Fractions through
    # 1/1024 overshoot the slab; 1/2048 lands strictly inside.
    v,f=_cube_local_bounds(xlo=.5,xhi=.5001,ylo=-1.,yhi=1.,zlo=-1.,zhi=1.)
    bank_tris=v[f]
    sign,top=MOD.bank_orientation(bank_tris)
    assert sign==1 and top["closed"] and top["consistently_oriented"]
    exact,locator=_exact_locator(v,f)
    skin=np.array([[.4,0.,0.],[.8,0.,0.],[.4,.4,0.]],dtype=np.float64)
    # This x=.5 face triangle covers the y=0..0.3 skin/bank intersection.
    bone=bank_tris[11]
    barymod=MOD.loadmod(MOD.CONSTRAINT,"test_virtual_constraint_198")
    lam,status,detail=MOD.strict_interior_sample(
        skin,bone,locator,sign,exact,barymod.exact_triangle_barycentric)
    assert status=="INSIDE", (status,detail)
    assert min(lam)>0
    assert MOD.STRICT_SAMPLE_INWARD_FRACTIONS[-1]==1/2048
    assert 1/1024 in MOD.STRICT_SAMPLE_INWARD_FRACTIONS


def _orientation_fixture():
    return {
        "detail_topology": {"component_count": 1, "adjacency_winding_conflict_count": 0, "nonmanifold_edges": 0},
        "coarse_O_topology": {"component_count": 1, "adjacency_winding_conflict_count": 0, "nonmanifold_edges": 0},
        "component_outward_sign_from_layer_order": {"0": {"chosen_outward_sign": 1, "anchor_count": 20, "agreement_fraction": .99}},
        "component_fine_to_coarse_winding_anchors": {"0": {"chosen_outward_parity": -1, "anchor_count": 18, "agreement_fraction": .98}},
    }

def test_registered_source_orientation_resolves_absolute_and_relative_signs():
    signs=MOD.validate_source_outward_orientation(_orientation_fixture())
    assert signs["fine_outward_sign"]==1
    assert signs["fine_to_coarse_winding_parity"]==-1
    assert signs["coarse_outward_sign"]==-1


def test_source_face_normals_use_typed_parent_ancestry_and_resolved_winding():
    candidate={
        "face_kind":np.array([2,1],np.int8),
        "parent_fine_detail_face_row":np.array([0,-1],np.int32),
        "parent_coarse_face_row":np.array([-1,0],np.int32),
    }
    detail={
        "source_surface_faces_detail_rows":np.array([[0,1,2]],np.int64),
        "detail_baseline_world_m":np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],np.float64),
    }
    base={"source_O_faces":np.array([[0,1,2]],np.int64),"points_world_m":detail["detail_baseline_world_m"]}
    signs=MOD.validate_source_outward_orientation(_orientation_fixture())
    nf,pf=MOD.registered_source_outward_face_normal(0,candidate,detail,base,signs)
    nc,pc=MOD.registered_source_outward_face_normal(1,candidate,detail,base,signs)
    np.testing.assert_allclose(nf,[0.,0.,1.],atol=0,rtol=0)
    np.testing.assert_allclose(nc,[0.,0.,-1.],atol=0,rtol=0)
    assert pf=={"kind":"original_NHSKIN_face","row":0}
    assert pc=={"kind":"original_coarse_O_face","row":0}


def test_source_orientation_fails_closed_on_conflict_ambiguity_or_missing_anchor():
    for mutate in (
        lambda j:j["detail_topology"].update(adjacency_winding_conflict_count=1),
        lambda j:j["component_outward_sign_from_layer_order"]["0"].update(agreement_fraction=.5),
        lambda j:j["component_fine_to_coarse_winding_anchors"]["0"].update(chosen_outward_parity=0),
        lambda j:j["component_outward_sign_from_layer_order"]["0"].update(anchor_count=0),
    ):
        import copy
        bad=copy.deepcopy(_orientation_fixture());mutate(bad)
        try:
            MOD.validate_source_outward_orientation(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous source orientation must fail closed")


def test_source_normal_resolution_rejects_missing_or_degenerate_parent_faces():
    candidate={"face_kind":np.array([2],np.int8),"parent_fine_detail_face_row":np.array([-1],np.int32),"parent_coarse_face_row":np.array([-1],np.int32)}
    detail={"source_surface_faces_detail_rows":np.array([[0,1,2]],np.int64),"detail_baseline_world_m":np.zeros((3,3),np.float64)}
    base={"source_O_faces":np.array([[0,1,2]],np.int64),"points_world_m":np.zeros((3,3),np.float64)}
    signs=MOD.validate_source_outward_orientation(_orientation_fixture())
    for c,d in ((candidate,detail),):
        try: MOD.registered_source_outward_face_normal(0,c,d,base,signs)
        except ValueError: pass
        else: raise AssertionError("missing parent ancestry must fail closed")
    candidate["parent_fine_detail_face_row"][0]=0
    try: MOD.registered_source_outward_face_normal(0,candidate,detail,base,signs)
    except ValueError: pass
    else: raise AssertionError("degenerate source face must fail closed")


def test_direction_mode_preserves_default_and_selects_only_explicit_source_normal():
    nearest=np.array([0.,-3.,4.])
    got=MOD.choose_proposal_direction("nearest-boundary",nearest)
    np.testing.assert_allclose(got,[0.,-.6,.8],atol=1e-15,rtol=0)
    outward=np.array([0.,0.,-2.])
    np.testing.assert_allclose(MOD.choose_proposal_direction("source-outward",nearest,outward),[0.,0.,-1.],atol=0,rtol=0)
    try: MOD.choose_proposal_direction("source-outward",nearest)
    except ValueError: pass
    else: raise AssertionError("source-outward mode cannot silently fall back")
    try: MOD.choose_proposal_direction("other",nearest,outward)
    except ValueError: pass
    else: raise AssertionError("unknown mode must fail closed")


def test_outward_hemisphere_uses_interior_clipped_chord_point():
    # The unconstrained triangle projection is on z<0. The nearest point in
    # z>=0 lies in the interior of the plane/triangle chord, not at a vertex.
    tri=np.array([[[-1.,0.,-1.],[1.,0.,-1.],[0.,2.,1.]]],dtype=np.float64)
    got=MOD.closest_points_on_triangle_mesh_in_hemisphere(
        np.zeros(3),tri,np.array([17],np.int64),np.array([0.,0.,5.]))
    np.testing.assert_allclose(got["point_world_m"],[0.,1.,0.],atol=1e-14,rtol=0)
    assert abs(got["distance_m"]-1.0)<1e-14
    assert got["hemisphere_dot_m"]>=-1e-14
    assert got["candidate_slot"]==10

def test_outward_hemisphere_retains_unconstrained_closest_when_feasible():
    tri=np.array([[[ -1.,-1.,1.],[1.,-1.,1.],[0.,1.,1.]]],dtype=np.float64)
    p=np.array([0.,0.,0.])
    plain=MOD.closest_points_on_triangle_mesh(p,tri,np.array([4],np.int64))
    clipped=MOD.closest_points_on_triangle_mesh_in_hemisphere(p,tri,np.array([4],np.int64),np.array([0.,0.,1.]))
    np.testing.assert_allclose(clipped["point_world_m"],plain["point_world_m"],atol=1e-15,rtol=0)
    assert abs(clipped["distance_m"]-plain["distance_m"])<1e-15

def test_outward_hemisphere_nearly_parallel_and_vertex_tangent_cases():
    near=np.array([[[-1.,1.,-1e-12],[1.,1.,1e-12],[0.,3.,1e-12]]],dtype=np.float64)
    got=MOD.closest_points_on_triangle_mesh_in_hemisphere(np.zeros(3),near,np.array([1],np.int64),np.array([0.,0.,1.]))
    assert np.isfinite(got["point_world_m"]).all()
    assert got["hemisphere_dot_m"]>=-1e-14
    tangent=np.array([[[0.,0.,0.],[-1.,1.,-1.],[1.,1.,-1.]]],dtype=np.float64)
    got2=MOD.closest_points_on_triangle_mesh_in_hemisphere(np.zeros(3),tangent,np.array([2],np.int64),np.array([0.,0.,1.]))
    np.testing.assert_allclose(got2["point_world_m"],[0.,0.,0.],atol=0,rtol=0)
    assert got2["distance_m"]==0.0

def test_outward_hemisphere_fails_when_triangles_are_entirely_opposite():
    tri=np.array([[[-1.,0.,-1.],[1.,0.,-1.],[0.,1.,-1.]]],dtype=np.float64)
    try:
        MOD.closest_points_on_triangle_mesh_in_hemisphere(np.zeros(3),tri,np.array([8],np.int64),np.array([0.,0.,1.]))
    except ValueError:
        pass
    else:
        raise AssertionError("no point in an all-opposite mesh may be returned")

def test_outward_hemisphere_union_ties_keep_typed_bank_order():
    tri=np.array([[[-1.,0.,-1.],[1.,0.,-1.],[0.,2.,1.]]],dtype=np.float64)
    records={(9,2):{"world_triangles":tri,"id":np.array([3],np.int64)},
             (4,8):{"world_triangles":tri,"id":np.array([30],np.int64)}}
    direction,witness=MOD.nearest_inside_union_outward_hemisphere_direction(np.zeros(3),{(9,2),(4,8)},records,[0,0,1])
    np.testing.assert_allclose(direction,[0,1,0],atol=1e-14,rtol=0)
    assert witness["body"]==4 and witness["stable_bone_id"]==8
    assert witness["source_hemisphere_dot"]>=-1e-14


def test_actual277_plane_contact_uses_stable_body_local_relative_vector():
    # Frozen first failure from 277: absolute contact-start cancellation makes
    # a clipped-chord tangent appear to point inward by -2.998e-13.
    p=np.array([-.2821348602960404,-.8557556150787458,.13089508736140762])
    n=np.array([.11185782691274716,-.5671854991495575,.8159585995090838])
    M=np.array([[-.039970270211853176,.09229745552701729,.9949290682004488],
                [-.9566383288785971,.2839847057841993,-.06477651354835245],
                [-.2885234457844632,-.9543762720700641,.07694422063519113]])
    pos=np.array([-.26655828952789307,-.8034539818763733,.1283719539642334])
    off=np.array([2.6727232319695426e-10,-6.810308358917716e-10,1.9591840233479374e-9])
    local_start=np.array([-.001694369429385778,-.00011515839136468066,.05460378216400775])
    tri_local=np.array([[[-.0026206078473478556,-.00087219406850636,.05395018681883812],
                         [-.0007413032581098378,-.001277237432077527,.05561244115233421],
                         [-.002342105144634843,.0008043653797358274,.05403834208846092]]])
    # The immutable 272 absolute-world result for this exact witness was:
    old_world_contact=np.array([-.2821496615520273,-.8558038331048435,.13086359933139546])
    old_absolute_delta=old_world_contact-p
    assert float(old_absolute_delta@n)/float(np.linalg.norm(old_absolute_delta)) < -128*np.finfo(float).eps

    records={(43,16):{"T":tri_local,"M":M,"id":np.array([144],np.int64)}}
    direction,w=MOD.nearest_inside_union_outward_hemisphere_direction(
        p,{(43,16)},records,n,local_sample_by_body={43:local_start})
    assert w["triangle_stable_id"]==144
    assert abs(w["distance_m"]-5.94605016259481e-5)<2e-15
    assert float(direction@n)>=-128*np.finfo(float).eps
    assert w["source_hemisphere_dot"]>=-128*np.finfo(float).eps
    np.testing.assert_allclose(
        direction,w["relative_offset_world_m"]/np.linalg.norm(w["relative_offset_world_m"]),
        atol=2e-15,rtol=0)
