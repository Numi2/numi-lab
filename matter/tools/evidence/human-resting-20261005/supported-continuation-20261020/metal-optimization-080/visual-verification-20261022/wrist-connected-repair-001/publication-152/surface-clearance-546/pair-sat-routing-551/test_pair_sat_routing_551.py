import importlib.util
from pathlib import Path
import numpy as np
import pytest

SRC=Path(__file__).with_name("pair_sat_routing_551.py")
spec=importlib.util.spec_from_file_location("pair_sat_551",SRC)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_both_signs_of_all_standard_axes_are_enumerated():
    skin=np.array([[0.,0.,0.],[1.,0.1,0.],[0.2,1.,0.3]])
    bone=np.array([[0.2,0.1,0.8],[1.1,0.2,1.1],[0.1,1.2,0.9]])
    base=m._sat.triangle_sat_axis_candidates(skin,bone)
    signed=m._signed_sat_axes(skin,bone)
    assert len(signed)==2*len(base)
    assert len({tuple(np.round(a["normal_world_unit"],12)) for a in signed})==len(signed)
    # Face normals plus nine edge-cross candidates are all nondegenerate here.
    assert len(base)==11
    assert len(signed)==22
    for i in range(0,len(signed),2):
        assert np.allclose(signed[i]["normal_world_unit"],-signed[i+1]["normal_world_unit"])

def test_target_rows_separate_the_whole_finite_triangle_pair():
    skin=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    bone=np.array([[0.35,0.3,-0.2],[0.35,0.3,0.4],[0.35,0.8,0.1]])
    r=m.choose_pair_sat_plane(skin,bone)
    n=r["normal_world_unit"]
    target=r["support_mm"]+r["float32_ulp_margin_mm"]
    moved=skin+r["corner_lower_mm"][:,None]*n
    assert np.all(r["corner_lower_mm"]>=0)
    assert np.min(moved@n)>=target-1e-12
    assert np.max(bone@n)<=r["support_mm"]+1e-12
    assert np.min(moved@n)>np.max(bone@n)
    assert r["whole_bank_clearance_certificate"] is False

def test_result_is_stable_to_vertex_permutation_and_winding():
    skin=np.array([[0.,0.,0.],[1.,0.2,0.],[0.1,0.9,0.3]])
    bone=np.array([[0.1,0.2,0.6],[0.9,0.1,0.9],[0.2,0.8,0.7]])
    a=m.choose_pair_sat_plane(skin,bone)
    b=m.choose_pair_sat_plane(skin[[2,0,1]],bone[[1,2,0]])
    assert np.allclose(a["normal_world_unit"],b["normal_world_unit"],atol=1e-12)
    assert a["support_mm"]==pytest.approx(b["support_mm"],abs=1e-12)
    assert a["target_offset_mm"]==pytest.approx(b["target_offset_mm"],abs=1e-12)
    # Corner order follows the input, so compare sorted deficits.
    assert np.allclose(np.sort(a["corner_lower_mm"]),np.sort(b["corner_lower_mm"]),atol=1e-12)

def test_coplanar_intersecting_triangles_receive_positive_margin_target():
    skin=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    bone=np.array([[0.2,0.2,0.],[1.1,0.2,0.],[0.2,1.1,0.]])
    r=m.choose_pair_sat_plane(skin,bone)
    assert r["float32_ulp_margin_mm"]>0
    assert r["max_positive_corner_deficit_mm"]>0
    assert np.isfinite(r["normal_world_unit"]).all()

def test_parallel_edge_cross_degeneracies_are_skipped():
    skin=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    bone=np.array([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
    signed=m._signed_sat_axes(skin,bone)
    unique={tuple(np.round(a["normal_world_unit"],12)) for a in signed}
    assert len(unique)<len(signed)
    assert all(abs(np.linalg.norm(a["normal_world_unit"])-1.)<1e-12 for a in signed)
    r=m.choose_pair_sat_plane(skin,bone)
    assert np.isfinite(r["corner_lower_mm"]).all()

def test_clear_corner_keeps_negative_458_lower_and_exact_support_point():
    skin=np.array([[5.,0.,0.],[5.,1.,0.],[5.,0.,1.]])
    bone=np.array([[0.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    r=m.choose_pair_sat_plane(skin,bone)
    n=r["normal_world_unit"]
    assert np.any(r["corner_lower_mm"]<0.)
    assert np.allclose(r["corner_lower_mm"],r["target_offset_mm"]-skin@n)
    assert np.isclose(r["plane_point_mm"]@n,r["support_mm"],rtol=0.,atol=1e-12)

def test_degenerate_or_nonfinite_triangle_fails_closed():
    bone=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    with pytest.raises(ValueError,match="degenerate"):
        m.choose_pair_sat_plane(np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]),bone)
    bad=bone.copy();bad[1,0]=np.nan
    with pytest.raises(ValueError,match="finite"):
        m.choose_pair_sat_plane(bone,bad)
