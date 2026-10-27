import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

P=Path(__file__).with_name("make_mixed_targets595.py")
spec=importlib.util.spec_from_file_location("mixed595_testmod",P)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def fixture(tmp):
    rows=list(range(8,16))
    base=tmp/"base.npz"; summary=tmp/"summary.json"
    mixed=tmp/"mixed.npz"; mixed_report=tmp/"mixed.json"
    source_npz=tmp/"source.npz"; source_report=tmp/"source.json"
    depth_npz=tmp/"depth.npz"; depth_report=tmp/"depth.json"
    points=np.zeros((16,3),np.float32); faces=np.array([[0,1,2]],np.int64)
    np.savez(base,points_world_m=points,source_O_faces=faces,typed_coarse_anchor_output_rows_018=np.array([0],np.int64))
    summary.write_text("{}")
    origins=m.EXPECTED_ORIGINS
    source_target=np.asarray([[0.1*(i+1),0,0] for i in range(8)],np.float32)
    depth_target=np.asarray([[0.2*(i+1),0,0] for i in range(8)],np.float32)
    mixed_points=points.copy()
    selected=[]
    for i,(row,origin) in enumerate(zip(rows,origins)):
        target=source_target[i] if origin=="source-outward" else depth_target[i]
        mixed_points[row]=target
        selected.append(target)
    np.savez(mixed,points_world_m=mixed_points,source_O_faces=faces,typed_coarse_anchor_output_rows_018=np.array([0],np.int64))
    parent={}
    for label,target in (("source-outward",source_target),("depth-maximizing",depth_target)):
        cov=np.full((8,3),np.nan) if label=="source-outward" else np.tile([0.,1.,0.],(8,1))
        arrays={"query_rows":np.asarray(rows,np.int64),"target_world_f32_m":target,
            "proposal_direction_world_unit":np.tile([1.,0.,0.],(8,1)),
            "source_surface_normal_world_unit":np.tile([1.,0.,0.],(8,1)),
            "depth_covector_world_unit":cov,"direction_compatibility_mask":np.ones(8,bool),
            "source_normal_available_mask":np.ones(8,bool),"pass_mask":np.ones(8,bool),
            "target_distance_m":np.linalg.norm(target,axis=1).astype(np.float64),
            "target_exactly_outside_all185":np.ones(8,bool)}
        f=source_npz if label=="source-outward" else depth_npz
        np.savez(f,**arrays)
        rec=[]
        for i,row in enumerate(rows):
            rec.append({"query_row":row,"status":"PASS_FIRST_UNION_EXIT",
                "start_world_f32_m":[0.,0.,0.],"target_world_f32_m":target[i].tolist(),
                "proposal_direction_world_unit":[1.,0.,0.],
                "end_outside_all185_by_event_parity":True,"target_exactly_outside_all185":True})
        parent[label]=(f,rec)
    base_sha=m.sha(base); summary_sha=m.sha(summary)
    for label,path in (("source-outward",source_report),("depth-maximizing",depth_report)):
        npz,records=parent[label]
        report={"direction_mode":label,"status":"PASS_FIRST_UNION_EXIT_TARGETS_ONLY","physical_steps":0,
            "inputs":{"candidate_sha256":base_sha,"point_summary_sha256":summary_sha},"targets":records}
        path.write_text(json.dumps(report))
    mixed_sha=m.sha(mixed)
    mr={"base_candidate_sha256":base_sha,"candidate_sha256":mixed_sha,
        "586_v2_Q_and_037_refinement_validation":"PASS","individual_targets_all185_certified":True,
        "native_admission":False,"physical_steps":0,"origins":origins,"changed_point_rows":rows,
        "point_domain_count":16,"all_nonpoint_arrays_preserved":True,"anchors_unchanged":True,
        "rows":[{"row":row,"all185_endpoint_certified":True,
            "actual_displacement_dot_source_outward":1.0,"actual_displacement_dot_signed_current_outward":1.0}
            for row in rows]}
    mixed_report.write_text(json.dumps(mr))
    return base,summary,mixed,mixed_report,source_npz,source_report,depth_npz,depth_report

def build(tmp, **overrides):
    paths=fixture(tmp)
    names=("base_candidate","point_summary","mixed_candidate","mixed_report_path","source_npz_path",
           "source_report_path","depth_npz_path","depth_report_path")
    args=dict(zip(names,paths))
    args["out_dir"]=tmp/"out"
    args.update(overrides)
    return paths,m.build_mixed_targets(**args)

def test_mixed_adapter_preserves_591_rows_and_per_parent_provenance(tmp_path):
    paths,(npz,report,result)=build(tmp_path)
    with np.load(npz) as z:
        assert z["query_rows"].tolist()==list(range(8,16))
        assert z["pass_mask"].all() and z["target_exactly_outside_all185"].all()
        assert np.isnan(z["depth_covector_world_unit"][4]).all()
        assert np.isfinite(z["depth_covector_world_unit"][0]).all()
    r=json.loads(report.read_text())
    assert [x["origin"] for x in r["per_row_selected_parent_receipts"]]==m.EXPECTED_ORIGINS
    assert result["physical_steps"]==0

def test_mixed_adapter_rejects_endpoint_not_equal_to_emitted591_target(tmp_path):
    paths=fixture(tmp_path)
    b,s,mix,mr,sn,sr,dn,dr=paths
    with np.load(mix,allow_pickle=False) as z: data={k:np.asarray(z[k]).copy() for k in z.files}
    data["points_world_m"][8,0]+=1e-3
    np.savez(mix,**data)
    j=json.loads(mr.read_text());j["candidate_sha256"]=m.sha(mix);mr.write_text(json.dumps(j))
    with pytest.raises(ValueError,match="endpoint differs"):
        m.build_mixed_targets(base_candidate=b,point_summary=s,mixed_candidate=mix,mixed_report_path=mr,
            source_npz_path=sn,source_report_path=sr,depth_npz_path=dn,depth_report_path=dr,out_dir=tmp_path/"out")

def test_mixed_adapter_rejects_direction_witness_outward_violation(tmp_path):
    paths=fixture(tmp_path)
    b,s,mix,mr,sn,sr,dn,dr=paths
    j=json.loads(mr.read_text());j["rows"][4]["actual_displacement_dot_source_outward"]=-0.1;mr.write_text(json.dumps(j))
    with pytest.raises(ValueError,match="not source/current outward"):
        m.build_mixed_targets(base_candidate=b,point_summary=s,mixed_candidate=mix,mixed_report_path=mr,
            source_npz_path=sn,source_report_path=sr,depth_npz_path=dn,depth_report_path=dr,out_dir=tmp_path/"out")

