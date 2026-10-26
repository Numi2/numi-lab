import importlib.util
from pathlib import Path
import numpy as np
import pytest

SRC=Path(__file__).with_name("run_source_outward_targets553.py")
spec=importlib.util.spec_from_file_location("source_outward553",SRC)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture():
    parent=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=np.float32)
    child=np.vstack((parent,np.array([[.25,.25,.01]],dtype=np.float32)))
    pa={"points_world_m":parent,"coarse_boundary_anchor_rows":np.array([0],np.int32)}
    ca={"points_world_m":child,"coarse_boundary_anchor_rows":np.array([0],np.int32),
        "refinement_node_kind":np.array([0,0,0,2],np.int8)}
    ps={"candidate_sha256":"parent","complete_surface_audit":True,
        "exact_PLC_surface_pairs":0,"inside_node_bank_pairs":0,"inside_nodes":0,"points":3}
    cs={"candidate_sha256":"child","complete_surface_audit":True,
        "exact_PLC_surface_pairs":0,"points":4,"inside_nodes":1,
        "inside_witnesses":[{"output_node":3,"bank_body":"43","bank_stable_id":"16"}]}
    return pa,ca,ps,cs

def test_new_only_prefix_accepts_appended_inside_refinement_node():
    pa,ca,ps,cs=fixture()
    rows,nold,nnew=m.validate_new_only_prefix(pa,ca,ps,cs,"parent","child")
    assert rows==[3] and (nold,nnew)==(3,4)

def test_old_inside_parent_is_rejected():
    pa,ca,ps,cs=fixture();ps["inside_node_bank_pairs"]=1
    with pytest.raises(ValueError,match="parent contains inside"):
        m.validate_new_only_prefix(pa,ca,ps,cs,"parent","child")

def test_old_node_inside_in_refined_report_is_rejected():
    pa,ca,ps,cs=fixture();cs["inside_witnesses"].append({"output_node":1,"bank_body":"43","bank_stable_id":"16"});cs["inside_nodes"]=2
    with pytest.raises(ValueError,match="not an appended"):
        m.validate_new_only_prefix(pa,ca,ps,cs,"parent","child")

def test_changed_parent_prefix_or_nonrefinement_node_is_rejected():
    pa,ca,ps,cs=fixture();ca["points_world_m"][0,0]=1.
    with pytest.raises(ValueError,match="bitwise parent point prefix"):
        m.validate_new_only_prefix(pa,ca,ps,cs,"parent","child")
    pa,ca,ps,cs=fixture();ca["refinement_node_kind"][3]=0
    with pytest.raises(ValueError,match="typed edge-midpoint"):
        m.validate_new_only_prefix(pa,ca,ps,cs,"parent","child")

def test_direction_mode_is_explicit_and_source_outward_only():
    m.validate_direction_mode("source-outward")
    with pytest.raises(ValueError,match="only supports"):
        m.validate_direction_mode("direct-depth")

def test_source_pin_mismatch_fails_closed(tmp_path):
    p=tmp_path/"pinned.txt";p.write_text("frozen")
    with pytest.raises(ValueError,match="pinned source SHA mismatch"):
        m.load_pinned(p,"0"*64,"bad_pin_test553")

def test_target_rows_must_equal_new_inside_rows_and_rays_match_normals(tmp_path):
    arrays=tmp_path/"targets.npz"
    np.savez(arrays,query_rows=np.array([3],np.int64),pass_mask=np.array([True]),
      target_exactly_outside_all185=np.array([True]),source_normal_available_mask=np.array([True]),
      proposal_direction_world_unit=np.array([[1.,0.,0.]]),source_surface_normal_world_unit=np.array([[1.,0.,0.]]))
    report={"direction_mode":"source-outward","status":"PASS_FIRST_UNION_EXIT_TARGETS_ONLY",
      "targets":[{"query_row":3,"status":"PASS_FIRST_UNION_EXIT","target_exactly_outside_all185":True,"exact_endpoint_bank_count":185}]}
    m.validate_target_artifacts(report,arrays,[3])
    with pytest.raises(ValueError,match="target report rows differ"):
      m.validate_target_artifacts(report,arrays,[2,3])
    bad=np.load(arrays)
    np.savez(arrays,query_rows=np.array([3],np.int64),pass_mask=np.array([True]),
      target_exactly_outside_all185=np.array([True]),source_normal_available_mask=np.array([True]),
      proposal_direction_world_unit=np.array([[0.,1.,0.]]),source_surface_normal_world_unit=np.array([[1.,0.,0.]]))
    with pytest.raises(ValueError,match="not the registered source-surface normals"):
      m.validate_target_artifacts(report,arrays,[3])
