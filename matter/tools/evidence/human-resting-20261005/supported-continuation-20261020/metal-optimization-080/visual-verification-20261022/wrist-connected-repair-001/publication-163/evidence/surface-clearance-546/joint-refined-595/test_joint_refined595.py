import importlib.util
from pathlib import Path
import numpy as np
import pytest

P=Path(__file__).with_name("joint_refined595.py")
spec=importlib.util.spec_from_file_location("joint_refined595_testmod",P)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def fixture():
    rows=np.array([10,11],np.int64)
    state={"candidate_sha256":"c","point_summary_sha256":"ps","continuation_mode":"point_continuation","temporary_unselected_construction":True,"candidate_selected":False}
    point={"candidate_sha256":"c","complete_surface_audit":True,"exact_PLC_surface_pairs":0,"inside_node_bank_pairs":2,"inside_witnesses":[{"output_node":10},{"output_node":11}]}
    surface={"candidate_sha256":"c","complete":True,"bank_count":185,"expected_bank_count":185}
    arrays={"points_world_m":np.zeros((12,3),np.float32),"typed_coarse_anchor_output_rows_018":np.array([0,1],np.int32)}
    report={"status":"PASS_FIRST_UNION_EXIT_TARGETS_ONLY","query_rows":2,"passing_targets":2,
            "inputs":{"candidate_sha256":"c","point_summary_sha256":"ps"},
            "outputs":{"parent_prefix_node_count":10}}
    targets={"query_rows":rows,"target_world_f32_m":np.ones((2,3),np.float32),
             "pass_mask":np.ones(2,bool),"target_exactly_outside_all185":np.ones(2,bool),
             "target_distance_m":np.ones(2),"proposal_direction_world_unit":np.ones((2,3)),
             "source_surface_normal_world_unit":np.ones((2,3)),"depth_covector_world_unit":np.ones((2,3))}
    depth_report={"candidate_sha256":"c","output_npz_sha256":"d","complete_bank_accounting_verified":True,
                  "bank_count":185,"inside_row_count":2,"outside_row_count":10,"row_count":12}
    depth={"node_rows":np.arange(12,dtype=np.int64)}
    prior={"candidate_sha256":"p"}
    return state,point,surface,arrays,report,targets,depth_report,depth,prior,rows

def validate():
    x=fixture()
    return m.validate_joint_refined_inputs(state=x[0],point_summary=x[1],surface_report=x[2],arrays=x[3],
        candidate_sha256="c",point_target_report=x[4],point_targets=x[5],point_target_sha256="t",
        point_depth_report=x[6],depth_npz=x[7],depth_sha256="d",prior_context=x[8],
        prior_candidate_sha256="p",ancestry_parent_sha256="p")

def test_joint_branch_accepts_exact_nonzero_containment_with_certified_targets():
    assert validate()["temporary_unselected_construction"] is True

def test_joint_branch_rejects_any_self_pair():
    x=fixture(); x[1]["exact_PLC_surface_pairs"]=1
    with pytest.raises(ValueError,match="self pairs"):
        m.validate_joint_refined_inputs(state=x[0],point_summary=x[1],surface_report=x[2],arrays=x[3],
            candidate_sha256="c",point_target_report=x[4],point_targets=x[5],point_target_sha256="t",
            point_depth_report=x[6],depth_npz=x[7],depth_sha256="d",prior_context=x[8],
            prior_candidate_sha256="p",ancestry_parent_sha256="p")

def test_joint_branch_requires_exact_inside_row_mapping():
    x=fixture(); x[5]["query_rows"]=np.array([10,12])
    with pytest.raises(ValueError,match="witness output-node"):
        m.validate_joint_refined_inputs(state=x[0],point_summary=x[1],surface_report=x[2],arrays=x[3],
            candidate_sha256="c",point_target_report=x[4],point_targets=x[5],point_target_sha256="t",
            point_depth_report=x[6],depth_npz=x[7],depth_sha256="d",prior_context=x[8],
            prior_candidate_sha256="p",ancestry_parent_sha256="p")

def test_joint_branch_rejects_wrong_prior_or_outside_status():
    x=fixture()
    with pytest.raises(ValueError,match="prior seed"):
        m.validate_joint_refined_inputs(state=x[0],point_summary=x[1],surface_report=x[2],arrays=x[3],
            candidate_sha256="c",point_target_report=x[4],point_targets=x[5],point_target_sha256="t",
            point_depth_report=x[6],depth_npz=x[7],depth_sha256="d",prior_context=x[8],
            prior_candidate_sha256="p",ancestry_parent_sha256="other")
    x=fixture(); x[5]["target_exactly_outside_all185"][1]=False
    with pytest.raises(ValueError,match="target_exactly_outside_all185"):
        m.validate_joint_refined_inputs(state=x[0],point_summary=x[1],surface_report=x[2],arrays=x[3],
            candidate_sha256="c",point_target_report=x[4],point_targets=x[5],point_target_sha256="t",
            point_depth_report=x[6],depth_npz=x[7],depth_sha256="d",prior_context=x[8],
            prior_candidate_sha256="p",ancestry_parent_sha256="p")

def test_fit_argv_preserves_source_area_edge_limits_and_changes_trust():
    argv=["py","entry","--base-candidate","a","--raw-trial","a","--self-pairs","s",
          "--area-reference-npz","a","--point-target-npz","p","--point-target-report","pr",
          "--face-target-npz","f","--face-target-report","fr","--point-depth-rows-npz","d",
          "--point-depth-rows-report","dr","--fixed-boundary-npz","g","--fixed-boundary-report","gr",
          "--matched-inputs-json-490","m","--audit159-490","ar","--output","o","--trust-mm","2",
          "--source-cap-mm","12","--area-floor","0.1","--max-edge-gradient","0.5",
          "--immediate-parent-basis","--allow-soft-target-regression-485"]
    out=m.fit_argv_from_template(argv,candidate=Path("c"),audit159=Path("ar2"),point_target_npz=Path("p2"),
        point_target_report=Path("pr2"),face_target_npz=Path("f2"),face_target_report=Path("fr2"),
        point_depth_npz=Path("d2"),point_depth_report=Path("dr2"),self_pairs=Path("s2"),
        fixed_guard_npz=Path("g2"),fixed_guard_report=Path("gr2"),matched_inputs=Path("m2"),
        output=Path("o2"),trust_mm=4)
    assert out[out.index("--trust-mm")+1]=="4.0"
    assert out[out.index("--source-cap-mm")+1]=="12"
    assert out[out.index("--area-floor")+1]=="0.1"
    assert out[out.index("--max-edge-gradient")+1]=="0.5"


def test_mixed_per_row_direction_allows_only_unused_covector_to_be_nan():
    x=fixture()
    x[4]["direction_mode"]="per-row-certified-source-or-depth-selection"
    x[4]["per_row_selected_parent_receipts"]=[
        {"query_row":10,"origin":"source-outward"},
        {"query_row":11,"origin":"depth-maximizing"},
    ]
    x[5]["depth_covector_world_unit"][0]=np.nan
    out=m.validate_joint_refined_inputs(state=x[0],point_summary=x[1],surface_report=x[2],arrays=x[3],
        candidate_sha256="c",point_target_report=x[4],point_targets=x[5],point_target_sha256="t",
        point_depth_report=x[6],depth_npz=x[7],depth_sha256="d",prior_context=x[8],
        prior_candidate_sha256="p",ancestry_parent_sha256="p")
    assert out["temporary_unselected_construction"] is True

def test_mixed_per_row_direction_requires_depth_covector_for_depth_target():
    x=fixture()
    x[4]["direction_mode"]="per-row-certified-source-or-depth-selection"
    x[4]["per_row_selected_parent_receipts"]=[
        {"query_row":10,"origin":"source-outward"},
        {"query_row":11,"origin":"depth-maximizing"},
    ]
    x[5]["depth_covector_world_unit"][1]=np.nan
    with pytest.raises(ValueError,match="finite depth covector"):
        m.validate_joint_refined_inputs(state=x[0],point_summary=x[1],surface_report=x[2],arrays=x[3],
            candidate_sha256="c",point_target_report=x[4],point_targets=x[5],point_target_sha256="t",
            point_depth_report=x[6],depth_npz=x[7],depth_sha256="d",prior_context=x[8],
            prior_candidate_sha256="p",ancestry_parent_sha256="p")
