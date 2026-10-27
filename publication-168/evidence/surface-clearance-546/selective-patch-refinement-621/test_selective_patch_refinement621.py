import importlib.util
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).with_name("selective_patch_refinement621.py")
s=importlib.util.spec_from_file_location("patch621_test",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def bank(body,bone,pairs): return {"body":body,"stable_bone_id":bone,"result":{"count":len(pairs),"triangle_pairs":pairs}}
def rep(*banks): return {"complete":True,"geometry_or_native_admission":False,"banks":list(banks)}

def test_pair_weighted_connected_patch_aggregates_multiple_banks():
    f=np.array([[0,1,2],[1,3,2],[4,5,6]],np.int64)
    r=rep(bank(1,10,[[0,0],[0,1],[0,2],[1,3]]),bank(2,20,[[1,0],[1,1]]),bank(3,30,[[2,0],[2,1],[2,2],[2,3],[2,4]]))
    q=m.select_patch(f,r,max_edges=5,point_count=7)
    assert q["selected_face_rows"]==[0,1] and q["selected_pair_count"]==6 and q["selected_edge_count"]==5
    assert [(x["body"],x["stable_bone_id"]) for x in q["patch_bank_pair_counts"]]==[(1,10),(2,20)]

def test_null_bank_is_allowed_only_when_159_aabb_proves_no_candidates():
    f=np.array([[0,1,2]],np.int64)
    empty={"body":2,"stable_bone_id":3,"skin_aabb_active":0,"bank_aabb_active":0,"result":None}
    q=m.select_patch(f,rep(bank(1,2,[[0,0]]),empty),max_edges=3,point_count=3)
    assert q["selected_face_rows"]==[0]
    empty["skin_aabb_active"]=1
    with pytest.raises(ValueError,match="active AABB"):
        m.select_patch(f,rep(bank(1,2,[[0,0]]),empty),max_edges=3,point_count=3)

def test_equal_weight_tie_is_deterministic():
    f=np.array([[0,1,2],[3,4,5]],np.int64)
    q=m.select_patch(f,rep(bank(1,1,[[0,0],[0,1],[1,0],[1,1]])),max_edges=3,point_count=6)
    assert q["patch_count"]==2 and q["selected_face_rows"]==[0]

def test_vertex_touch_is_not_edge_connected():
    f=np.array([[0,1,2],[0,3,4]],np.int64)
    q=m.select_patch(f,rep(bank(1,1,[[0,0],[1,0],[1,1]])),max_edges=3,point_count=5)
    assert q["patch_count"]==2 and q["selected_face_rows"]==[1]

def test_invalid_face_row_rejected():
    with pytest.raises(ValueError,match="out of range"):
        m.select_patch(np.array([[0,1,2]],np.int64),rep(bank(1,2,[[1,0]])),max_edges=3,point_count=3)

def test_patch_capacity_is_hard_and_never_truncated():
    f=np.array([[0,1,2],[1,3,2]],np.int64);r=rep(bank(1,2,[[0,0],[1,0]]))
    q=m.select_patch(f,r,max_edges=5,point_count=4)
    assert q["selected_edges"]==[[0,1],[0,2],[1,2],[1,3],[2,3]]
    with pytest.raises(ValueError,match="cap is 4"):m.select_patch(f,r,max_edges=4,point_count=4)

def test_strict_baseline_rejects_containment_or_self():
    c=Path("/tmp/621-candidate.npz");h="a"*64
    st={"candidate":str(c),"candidate_sha256":h,"continuation_mode":"strict_surface","accepted_physical_steps":0,"native_admission":False}
    lin={"pass":True,"anchors":237,"source_domain":"R0 kind0 node_source_row","legacy_coarse_boundary_rows_used_as_current_anchors":False,"current_output_rows":list(range(237))}
    ck={"status":"COMPLETE_EXACT_CONSTRUCTION_CHECKS","candidate_path":str(c),"candidate_sha256":h,"arithmetic_pass":True,"complete_banks":185,"contained_pairs":0,"self_pairs":0,"surface_pairs":0,"current_anchor_lineage":lin}
    pt={"candidate_sha256":h,"complete_surface_audit":True,"inside_node_bank_pairs":0,"inside_nodes":0,"exact_PLC_surface_pairs":0}
    sf={"status":"COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT","candidate_sha256":h,"complete":True,"geometry_or_native_admission":False,"bank_count":185,"expected_bank_count":185,"banks":[bank(i,i,[]) for i in range(185)]}
    assert len(m.validate_baseline(st,ck,pt,sf,np.empty((0,2),np.int64),h,c))==237
    ck["self_pairs"]=1
    with pytest.raises(ValueError,match="strict construction gates"):m.validate_baseline(st,ck,pt,sf,np.empty((0,2),np.int64),h,c)
