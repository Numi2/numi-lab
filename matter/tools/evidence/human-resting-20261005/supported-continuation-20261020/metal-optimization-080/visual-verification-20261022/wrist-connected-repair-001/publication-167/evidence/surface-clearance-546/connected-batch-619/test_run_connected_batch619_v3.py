from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

SRC=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/connected-batch-619/run_connected_batch619_v3.py")
spec=importlib.util.spec_from_file_location("batch619",SRC)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(tmp):
    candidate=tmp/"candidate.npz"
    np.savez_compressed(candidate,points_world_m=np.zeros((3,3),np.float32),
                        source_O_faces=np.array([[0,1,2]],np.int64))
    h=m.sha(candidate)
    q=tmp/"audit-130";(q/"query").mkdir(parents=True)
    point=q/"summary.json"
    point.write_text(json.dumps({"candidate_sha256":h,"complete_surface_audit":True,
        "exact_PLC_surface_pairs":0,"inside_node_bank_pairs":0}))
    np.save(q/"surface-self-pairs.npy",np.empty((0,2),np.int64))
    (q/"query/query-receipt.json").write_text("{}")
    surface=tmp/"audit-159.json"
    surface.write_text(json.dumps({"candidate_sha256":h,"complete":True,
        "expected_bank_count":185,"bank_count":185,"banks":[{} for _ in range(185)]}))
    state=tmp/"state.json"
    state.write_text(json.dumps({"candidate":str(candidate),"candidate_sha256":h,
       "point_summary":str(point),"point_summary_sha256":m.sha(point),
       "surface_report":str(surface),"surface_report_sha256":m.sha(surface),
       "native_admission":False,"accepted_physical_steps":0}))
    return state,point,surface,candidate,h

def test_zero_contained_complete_baseline_passes(tmp_path):
    state,*_=fixture(tmp_path)
    s,roles,pairs,query=m.strict_baseline(state)
    assert s["accepted_physical_steps"]==0 and pairs.is_file() and query.is_file()

def test_positive_inside_baseline_rejected(tmp_path):
    state,point,*_=fixture(tmp_path)
    x=json.loads(point.read_text());x["inside_node_bank_pairs"]=1
    point.write_text(json.dumps(x))
    st=json.loads(state.read_text());st["point_summary_sha256"]=m.sha(point)
    state.write_text(json.dumps(st))
    with pytest.raises(ValueError,match="positive-inside"):
        m.strict_baseline(state)

def test_missing_185_bank_rejected(tmp_path):
    state,_,surface,*_=fixture(tmp_path)
    x=json.loads(surface.read_text());x["banks"].pop();x["bank_count"]=184
    surface.write_text(json.dumps(x))
    st=json.loads(state.read_text());st["surface_report_sha256"]=m.sha(surface)
    state.write_text(json.dumps(st))
    with pytest.raises(ValueError,match="185-bank"):
        m.strict_baseline(state)

def test_cut_only_for_complete_candidate_bound_exact_self(tmp_path):
    audit=tmp_path/"summary.json";pairs=tmp_path/"pairs.npy"
    candidate=tmp_path/"trial.npz";candidate.write_bytes(b"trial");h=m.sha(candidate)
    audit.write_text(json.dumps({"candidate_sha256":h,"complete_surface_audit":True,
        "exact_PLC_surface_pairs":2}))
    np.save(pairs,np.array([[0,2],[3,4]],np.int64))
    report={"status":"FAILED","error":"exact self intersections"}
    assert m.is_exact_self_failure(report,audit,pairs,h)
    assert not m.is_exact_self_failure({"status":"FAILED","error":"area gate"},audit,pairs,h)
    bad=json.loads(audit.read_text());bad["complete_surface_audit"]=False;audit.write_text(json.dumps(bad))
    assert not m.is_exact_self_failure(report,audit,pairs,h)

def test_fit_argv_changes_only_scp_count_and_output():
    a=[str(m.PY),str(m.FIT603),"--area-scp-iterations","2","--time-limit-s","90",
       "--base-candidate","base.npz","--self-pairs","pairs.npy","--output","old",
       "--max-edge-gradient",".5"]
    out=m.set_scp1(a,Path("new-fit"))
    assert out[out.index("--area-scp-iterations")+1]=="1"
    assert out[out.index("--time-limit-s")+1]=="90"
    assert out[out.index("--output")+1]=="new-fit"
    assert out[out.index("--max-edge-gradient")+1]==".5"

def test_fit_argv_rejects_wrong_budget_or_duplicate_flag():
    a=[str(m.PY),str(m.FIT603),"--area-scp-iterations","2","--time-limit-s","120",
       "--base-candidate","base","--self-pairs","pairs","--output","old"]
    with pytest.raises(ValueError,match="90 seconds"):
        m.set_scp1(a,"new")
    a[2:2]=["--area-scp-iterations","2"]
    with pytest.raises(ValueError,match="exactly one"):
        m.set_scp1(a,"new")

def test_round_and_disk_bounds_are_explicit():
    assert m.validate_round_count(1)==1 and m.validate_round_count(3)==3
    with pytest.raises(ValueError):m.validate_round_count(0)
    with pytest.raises(ValueError):m.validate_round_count(4)
    m.disk_guard(m.MIN_FIT_FREE,m.MIN_FIT_FREE)
    with pytest.raises(OSError):m.disk_guard(m.MIN_FIT_FREE-1,m.MIN_FIT_FREE)
    assert m.MAX_EXTRA_CUT_FITS==2

def test_strict_progress_rejects_no_improvement_or_containment():
    good={"status":"COMPLETE_EXACT_CONSTRUCTION_CHECKS","self_pairs":0,
          "contained_pairs":0,"surface_pairs":9}
    assert m.strict_improvement(good,10)
    assert not m.strict_improvement(good,9)
    assert not m.strict_improvement({**good,"contained_pairs":1},10)
    assert not m.strict_improvement({**good,"self_pairs":1},10)


def test_check_stage_recovery_requires_failed_process_and_unchanged_pins(tmp_path):
    audit=tmp_path/"summary.json";pairs=tmp_path/"pairs.npy";trial=tmp_path/"candidate.npz"
    trial.write_bytes(b"t");h=m.sha(trial)
    audit.write_text(json.dumps({"candidate_sha256":h,"complete_surface_audit":True,"exact_PLC_surface_pairs":1}))
    np.save(pairs,np.array([[0,1]],np.int64))
    report={"status":"FAILED","error":"exact self intersections"}
    assert m.recoverable_self_check({"returncode":1,"source_pins_unchanged":True},report,audit,pairs,h)
    assert not m.recoverable_self_check({"returncode":0,"source_pins_unchanged":True},report,audit,pairs,h)
    assert not m.recoverable_self_check({"returncode":1,"source_pins_unchanged":False},report,audit,pairs,h)
    assert not m.recoverable_self_check({"returncode":1,"source_pins_unchanged":True},
        {"status":"FAILED","error":"area gate"},audit,pairs,h)

def test_additional_cut_budget_falls_back_only_above_retention_floor():
    assert m.additional_cut_action(m.MIN_FIT_FREE)=="cut"
    assert m.additional_cut_action(m.MIN_FIT_FREE-1)=="retain-last-fit"
    assert m.additional_cut_action(m.MIN_RETENTION_FREE)=="retain-last-fit"
    assert m.additional_cut_action(m.MIN_RETENTION_FREE-1)=="stop"

def test_initial_fit_still_requires_250_mib():
    assert m.MIN_FIT_FREE==250*1024*1024
    assert m.MIN_RETENTION_FREE==165*1024*1024
    with pytest.raises(OSError):
        m.disk_guard(m.MIN_FIT_FREE-1,m.MIN_FIT_FREE)
    m.disk_guard(m.MIN_FIT_FREE,m.MIN_FIT_FREE)
