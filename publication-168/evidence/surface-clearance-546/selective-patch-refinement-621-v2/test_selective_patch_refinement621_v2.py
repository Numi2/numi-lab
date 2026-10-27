import importlib.util
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).with_name("selective_patch_refinement621_v2.py")
s=importlib.util.spec_from_file_location("621v2_test",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_four_exact_exclusions():
 e=np.array([[1,4],[2,5],[3,6],[7,8],[9,10]],np.int64)
 assert m.select_remaining_edges(e,np.arange(20,25),e,20,[20,22,24])==([[2,5],[7,8]],[{"query_row":20,"parent_edge":[1,4]},{"query_row":22,"parent_edge":[3,6]},{"query_row":24,"parent_edge":[9,10]}])
def test_no_exclusions_is_identity():
 e=np.array([[1,2],[2,3]],np.int64);assert m.select_remaining_edges(e,[10,11],e,10,[])==([[1,2],[2,3]],[])
def test_old_or_gapped_rows_rejected():
 e=np.array([[1,2],[2,3]],np.int64)
 with pytest.raises(ValueError,match="appended midpoint"):m.select_remaining_edges(e,[10,11],e,10,[9])
 with pytest.raises(ValueError,match="contiguous appended suffix"):m.select_remaining_edges(e,[10,12],e,10,[12])
def test_duplicate_or_mismatched_mapping_rejected():
 with pytest.raises(ValueError,match="duplicate midpoint"):m.select_remaining_edges([[1,2],[2,3]],[10,10],[[1,2],[2,3]],10,[])
 with pytest.raises(ValueError,match="match midpoint ancestry"):m.select_remaining_edges([[1,2],[2,3]],[10,11],[[1,2],[5,6]],10,[])
def test_foreign_rows_and_empty_patch_rejected():
 e=np.array([[1,2],[2,3]],np.int64)
 with pytest.raises(ValueError,match="appended midpoint"):m.select_remaining_edges(e,[10,11],e,10,[12])
 with pytest.raises(ValueError,match="remove entire patch"):m.select_remaining_edges(e,[10,11],e,10,[10,11])
def test_duplicate_infeasible_ids_rejected():
 e=np.array([[1,2],[2,3]],np.int64)
 with pytest.raises(ValueError,match="duplicate infeasible"):m.select_remaining_edges(e,[10,11],e,10,[10,10])

def test_cone_pins_and_row_schema(tmp_path):
 import json
 normal=tmp_path/"normal.json"; normal.write_text("{}")
 ref=tmp_path/"ref.json"; ref.write_text("{}")
 cand=tmp_path/"candidate.npz"; cand.write_bytes(b"x")
 source=tmp_path/"source.py"; source.write_text("pinned")
 fs={"candidate_sha256":m.sha(cand),"refinement_report_sha256":m.sha(ref),"original_baseline_candidate_sha256":"base"}
 point={"inside_nodes":1,"inside_witnesses":[{"output_node":8}]}
 cone={"status":"JOINT_POINT_DIRECTION_FEASIBILITY_DIAGNOSTIC","candidate":str(cand),"candidate_sha256":m.sha(cand),
       "refinement_report":str(ref),"refinement_report_sha256":m.sha(ref),"minimum_cosine":.05,
       "all185_endpoints_certified":False,"native_admission":False,"normal_audit":str(normal),
       "normal_audit_sha256":m.sha(normal),"source_pins":{str(source):m.sha(source)},
       "rows":[{"query_row":8,"status":"FEASIBLE"}],"target_row_count":1,"feasible_count":1,"infeasible_count":0,"infeasible_rows":[]}
 assert m.validate_cone(cone,tmp_path/"cone.json",{"input_candidate_sha256":"base"},ref,fs,cand,m.sha(cand),point)==[]
 cone["source_pins"][str(source)]="0"*64
 with pytest.raises(ValueError,match="source pin changed"):m.validate_cone(cone,tmp_path/"cone.json",{"input_candidate_sha256":"base"},ref,fs,cand,m.sha(cand),point)

def test_failed_receipt_pins_reject_stale_summary_and_inputs(tmp_path):
 parent=tmp_path/"parent.npz"; parent.write_bytes(b"parent")
 candidate=tmp_path/"child.npz"; candidate.write_bytes(b"child")
 state=tmp_path/"state.json"; state.write_text("{}")
 check=tmp_path/"check.json"; check.write_text("{}")
 refpath=tmp_path/"refinement.json"; refpath.write_text("{}")
 pointpath=tmp_path/"point.json"; pointpath.write_text("{}")
 point={"candidate_sha256":m.sha(candidate)}
 failed={"candidate":str(candidate),"candidate_sha256":m.sha(candidate),
   "point_summary":str(pointpath),"point_summary_sha256":m.sha(pointpath),
   "refinement_report":str(refpath),"refinement_report_sha256":m.sha(refpath)}
 ref={"input_candidate_path":str(parent),"input_candidate_sha256":m.sha(parent),
   "output_candidate_path":str(candidate),"output_candidate_sha256":m.sha(candidate),
   "input_state_path":str(state),"input_state_sha256":m.sha(state),
   "input_construction_report_path":str(check),"input_construction_report_sha256":m.sha(check)}
 assert m.validate_failed_receipts(failed,ref,pointpath,point,state,check,refpath,candidate,parent,m.sha(parent))
 failed["point_summary_sha256"]="0"*64
 with pytest.raises(ValueError,match="point-summary pin"):m.validate_failed_receipts(failed,ref,pointpath,point,state,check,refpath,candidate,parent,m.sha(parent))
 failed["point_summary_sha256"]=m.sha(pointpath)
 ref["input_state_sha256"]="0"*64
 with pytest.raises(ValueError,match="input state pin"):m.validate_failed_receipts(failed,ref,pointpath,point,state,check,refpath,candidate,parent,m.sha(parent))
 ref["input_state_sha256"]=m.sha(state)
 ref["input_construction_report_sha256"]="0"*64
 with pytest.raises(ValueError,match="check-report pin"):m.validate_failed_receipts(failed,ref,pointpath,point,state,check,refpath,candidate,parent,m.sha(parent))
