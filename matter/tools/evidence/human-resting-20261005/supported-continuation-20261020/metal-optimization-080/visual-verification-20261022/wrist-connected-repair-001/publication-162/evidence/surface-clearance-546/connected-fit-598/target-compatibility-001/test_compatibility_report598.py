import importlib.util, json
from pathlib import Path
import numpy as np
import pytest

P=Path(__file__).with_name("compatibility_report598.py")
s=importlib.util.spec_from_file_location("compat598",P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def fixture(tmp_path,n=1585,joint=1576,fallback=9):
    face_rows=np.r_[np.zeros(joint,dtype=int),np.resize(np.array([1,2,3]),fallback)]
    d=np.zeros((n,3));d[:,0]=1
    normals=np.zeros((n,3));normals[:,0]=1
    arrays={"face_rows":face_rows,"bank_body":np.zeros(n,int),
        "bank_stable_bone_id":np.zeros(n,int),"bank_triangle_stable_id":np.arange(n),
        "proposal_direction_world_unit":d,"source_surface_normal_world_unit":normals,
        "source_normal_valid":np.ones(n,bool),"target_exactly_outside_all185":np.ones(n,bool),
        "pair_index":np.arange(n)}
    records=[]
    for i in range(n):
        isfallback=i>=joint
        records.append({"pair_index":i,"face_row":int(face_rows[i]),
            "proposal_direction_world_unit":d[i].tolist(),"source_surface_normal_world_unit":normals[i].tolist(),
            "direction_source":"pinned 500 source-only direction retained" if isfallback else "545 current/source selector with cosine floors 0.05"})
    files={}
    for name,data in (("npz",b"n"),("recovery",b"v")):
        p=tmp_path/name;p.write_bytes(data);files[name]=p
    files["report"]=tmp_path/"report";files["report"].write_text(json.dumps({
        "direction_mode":m.MODE,"inputs":{"candidate_sha256":"a"*64},
        "direction_method":"joint source/current cone with explicit source-only fallback",
        "matched_159_complete":True,"banks":185,"crossing_pairs":n,
        "joint_outward_direction_count":joint,"source_only_fold_exception_count":fallback,
        "source_only_fold_exception_face_rows":sorted(set(map(int,face_rows[joint:])))}))
    files["records"]=tmp_path/"records"
    files["records"].write_text("".join(json.dumps(r)+"\n" for r in records))
    pins={str(files[k]):m.sha(files[k]) for k in ("report","npz","records")}
    rec={"status":"PASS_CERTIFIED_DIRECTION_TARGETS_ONLY","artifact_pins":pins,
        "candidate_sha256":"a"*64,"target_count":n,"joint_source_current_targets":joint,
        "source_only_exception_targets":fallback}
    return m.read(files["report"]),arrays,records,rec,files

def build(report,arrays,records,rec,files):
    return m.build_compatibility_report(report,arrays,records,rec,
        report_path=files["report"],npz_path=files["npz"],records_path=files["records"],
        recovery_path=files["recovery"],adapter_path=P)

def test_1576_joint_and_9_fallbacks_route_without_nearest_claim(tmp_path):
    report,arrays,records,rec,files=fixture(tmp_path)
    compat=build(report,arrays,records,rec,files)
    assert compat["direction_mode"]=="outward-nearest"
    info=compat["fit_direction_compatibility"]
    assert info["joint_direction_rows"]==1576 and info["source_only_fold_exception_rows"]==9
    assert info["nearest_optimality_claim"] is False
    assert info["operative_direction_method"]==report["direction_method"]

def test_rejects_bad_source_hemisphere(tmp_path):
    report,arrays,records,rec,files=fixture(tmp_path,2,2,0)
    arrays["proposal_direction_world_unit"][0]=[-1,0,0]
    records[0]["proposal_direction_world_unit"]=[-1,0,0]
    files["records"].write_text("".join(json.dumps(r)+"\n" for r in records))
    rec["artifact_pins"][str(files["records"])]=m.sha(files["records"])
    with pytest.raises(ValueError):build(report,arrays,records,rec,files)

def test_rejects_changed_receipt_hash_and_exception_faces(tmp_path):
    report,arrays,records,rec,files=fixture(tmp_path,2,1,1)
    rec["artifact_pins"][str(files["report"])]="0"*64
    with pytest.raises(ValueError):build(report,arrays,records,rec,files)
    report,arrays,records,rec,files=fixture(tmp_path,2,1,1)
    report["source_only_fold_exception_face_rows"]=[99]
    with pytest.raises(ValueError):build(report,arrays,records,rec,files)
