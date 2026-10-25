from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pytest

RUNNER = Path("/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/run_connected_fixed_boundary_316.py")
SPEC = importlib.util.spec_from_file_location("fixed_boundary_316_tests", RUNNER)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

def report(pairs=None):
    pairs = pairs or {}
    banks = []
    for i in range(185):
        ps = pairs.get(i, [])
        banks.append({"body": i, "stable_bone_id": 10000+i, "skin_faces": 12,
                      "skin_aabb_active": int(bool(ps)), "bank_aabb_active": int(bool(ps)),
                      "result": ({"count": len(ps), "triangle_pairs": [list(x) for x in ps]} if ps else None)})
    return {"status":"COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT", "complete":True,
            "candidate_sha256":"a"*64, "surface_point_count":20, "surface_face_count":12,
            "script_sha256":"b"*64, "exact_predicate_sha256":"c"*64,
            "bank_count":185, "expected_bank_count":185, "banks":banks}

def jwrite(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))

def test_typed_identity_keeps_bank_face_and_triangle_domains():
    got = MOD.typed_bone_pairs(report({0:[(3,17)], 1:[(3,17)]}))
    assert got == {(0,10000,3,17), (1,10001,3,17)}

def test_learned_pairs_union_trials_minus_base(tmp_path, monkeypatch):
    base=tmp_path/"base.json"; a=tmp_path/"a.json"; b=tmp_path/"b.json"
    jwrite(base,report({0:[(1,10)]}))
    jwrite(a,report({0:[(1,10),(2,20)],2:[(4,40)]}))
    jwrite(b,report({0:[(3,30)]}))
    monkeypatch.setattr(MOD,"complete_bone_reports",lambda:[b,a])
    assert MOD.learned_fixed_bone_pairs({"surface_report":str(base)}) == {
        (0,10000,2,20),(2,10002,4,40),(0,10000,3,30)}

def test_complete_report_discovery_rejects_incomplete(tmp_path, monkeypatch):
    seed=tmp_path/"seed"; cur=tmp_path/"cur"
    bad=seed/"audit-159"/"report.json"; r=report(); r["complete"]=False; jwrite(bad,r)
    monkeypatch.setattr(MOD,"SEED_ROOT",seed); monkeypatch.setattr(MOD,"CURRENT_OUT",cur,raising=False)
    with pytest.raises(ValueError,match="incomplete retained159"):
        MOD.complete_bone_reports()

@pytest.mark.xfail(strict=True, reason="316 typed parser does not yet validate complete typed-bank inventory/pair schema")
def test_typed_parser_should_reject_missing_bank_identity():
    malformed=report({0:[(2,3)]}); malformed["banks"]=malformed["banks"][:1]
    with pytest.raises(ValueError):
        MOD.typed_bone_pairs(malformed)

def test_prepare_fixed_boundary_passes_trials_and_checks_count(tmp_path,monkeypatch):
    out=tmp_path/"attempt"; out.mkdir()
    cand=tmp_path/"base.npz"; cand.write_bytes(b"candidate")
    surf=tmp_path/"base.json"; jwrite(surf,report())
    trials=[tmp_path/"t0.json",tmp_path/"t1.json"]
    for p in trials:jwrite(p,report())
    monkeypatch.setattr(MOD,"complete_bone_reports",lambda:trials)
    monkeypatch.setattr(MOD,"learned_fixed_bone_pairs",lambda _s:{(1,2,3,4),(5,6,7,8)})
    emitter=tmp_path/"emit.py"; emitter.write_text("# mock\n")
    monkeypatch.setattr(MOD,"FIXED_EMITTER",emitter)
    seen={}
    def fake_run(command,log,*,slot=True,timeout=900):
        a=list(map(str,command)); seen.update(args=a,slot=slot,timeout=timeout)
        Path(log).write_text("")
        target=Path(a[a.index("--output-dir")+1]); target.mkdir()
        jwrite(target/"report.json",{"introduced_pair_count":2})
        (target/"introduced-skin-bone-barriers.npz").write_bytes(b"data")
    monkeypatch.setattr(MOD,"run",fake_run)
    npz,rpt=MOD.prepare_fixed_boundary({"candidate":str(cand),"surface_report":str(surf)},out)
    a=seen["args"]
    assert a[a.index("--base-candidate")+1]==str(cand)
    assert a[a.index("--base-surface-report")+1]==str(surf)
    assert [a[i+1] for i,x in enumerate(a) if x=="--trial-surface-report"]==list(map(str,trials))
    assert seen["slot"] and seen["timeout"]==900 and npz.is_file() and rpt.is_file()

def test_prepare_fixed_boundary_rejects_dropped_pair_count(tmp_path,monkeypatch):
    out=tmp_path/"attempt"; out.mkdir()
    cand=tmp_path/"base.npz"; cand.write_bytes(b"x")
    surf=tmp_path/"base.json"; jwrite(surf,report())
    trial=tmp_path/"trial.json"; jwrite(trial,report())
    monkeypatch.setattr(MOD,"complete_bone_reports",lambda:[trial])
    monkeypatch.setattr(MOD,"learned_fixed_bone_pairs",lambda _s:{(1,2,3,4)})
    emitter=tmp_path/"emit.py"; emitter.write_text("")
    monkeypatch.setattr(MOD,"FIXED_EMITTER",emitter)
    def fake_run(command,log,*,slot=True,timeout=900):
        Path(log).write_text("")
        a=list(map(str,command)); d=Path(a[a.index("--output-dir")+1]); d.mkdir()
        jwrite(d/"report.json",{"introduced_pair_count":0}); (d/"introduced-skin-bone-barriers.npz").write_bytes(b"x")
    monkeypatch.setattr(MOD,"run",fake_run)
    with pytest.raises(ValueError,match="lost typed witnesses"):
        MOD.prepare_fixed_boundary({"candidate":str(cand),"surface_report":str(surf)},out)

def test_fit_emission_preserves_payload_prefix_and_boundary_receipts(tmp_path,monkeypatch):
    out=tmp_path/"attempt"; out.mkdir()
    candidate=tmp_path/"candidate.npz"; n=MOD.FIXED_PREFIX+1
    points=np.zeros((n,3),np.float32); faces=np.array([[0,1,2]],np.int32)
    ancestry=np.arange(n,dtype=np.int32); anchors=np.array([0,1,2],np.int64)
    np.savez_compressed(candidate,points_world_m=points,source_O_faces=faces,
        typed_coarse_anchor_output_rows_018=anchors,source_ancestry_rows=ancestry)
    point=tmp_path/"point.npz"; point.write_bytes(b"p")
    point_r=tmp_path/"point.json"; jwrite(point_r,{})
    face=tmp_path/"face.npz"; face.write_bytes(b"f")
    face_r=tmp_path/"face.json"; jwrite(face_r,{})
    fixed=tmp_path/"fixed.npz"; fixed.write_bytes(b"fixed")
    fixed_r=tmp_path/"fixed.json"; jwrite(fixed_r,{"introduced_pair_count":2})
    owner=tmp_path/"owner.py"; owner.write_text("# owner\n")
    monkeypatch.setattr(MOD,"FIXED_OWNER",owner)
    monkeypatch.setattr(MOD,"prepare_fixed_boundary",lambda *_:(fixed,fixed_r))
    pairrows=np.array([[0,1]],np.int64); seen={}
    def fake_run(command,log,*,slot=True,timeout=900):
        a=list(map(str,command)); seen.update(args=a,slot=slot,timeout=timeout)
        proposal=Path(a[a.index("--output")+1]); proposal.mkdir()
        np.savez_compressed(proposal/"raw-lp-candidate.npz",
            joint_lp_delta_mm_f64=np.array([[.125,-.25,.5]],np.float64),
            joint_lp_support_rows=np.array([MOD.FIXED_PREFIX],np.int64))
        inputs={"base_candidate_sha256":MOD.sha(candidate),"self_pairs_sha256":MOD.sha(out/"self-pairs.npy"),
            "point_target_npz_sha256":MOD.sha(point),"point_target_report_sha256":MOD.sha(point_r),
            "face_target_npz_sha256":MOD.sha(face),"face_target_report_sha256":MOD.sha(face_r)}
        jwrite(proposal/"report.json",{"support":{"threatened_pairs":1},"constraints":{
            "actual_pair_records":[{"faces":[0,1]}],"fixed_boundary_typed_pair_count":2,
            "edge_gradient":{"enabled":True,"max_component_gradient":.5},"edge_gradient_rows":7},
            "inputs":inputs,"fixed_boundary_receipt":{"npz_sha256":MOD.sha(fixed),
                "report_sha256":MOD.sha(fixed_r),"owner_sha256":MOD.sha(owner)}})
        Path(log).write_text("")
    monkeypatch.setattr(MOD,"run",fake_run)
    trial=MOD.run_fit_attempt({"candidate":str(candidate)},candidate,pairrows,point,point_r,face,face_r,out)
    a=seen["args"]
    for flag,p in (("--fixed-boundary-npz",fixed),("--fixed-boundary-report",fixed_r),("--fixed-boundary-owner",owner)):
        assert a[a.index(flag)+1]==str(p)
    assert seen["slot"] and seen["timeout"]==900
    with np.load(trial,allow_pickle=False) as got, np.load(candidate,allow_pickle=False) as base:
        assert np.array_equal(got["source_O_faces"],base["source_O_faces"])
        assert np.array_equal(got["typed_coarse_anchor_output_rows_018"],anchors)
        assert np.array_equal(got["source_ancestry_rows"],ancestry)
        assert np.array_equal(got["points_world_m"][:MOD.FIXED_PREFIX].view(np.uint32),
                              points[:MOD.FIXED_PREFIX].view(np.uint32))
        assert np.array_equal(got["points_world_m"][MOD.FIXED_PREFIX],
                              np.array([.000125,-.00025,.0005],np.float32))
    assert json.loads((out/"verified-fit-inputs.json").read_text())["pair_count"]==1
    assert json.loads((out/"trial-provenance.json").read_text())["native_admission"] is False

def test_new_bone_pair_prevents_repeated_input_stop(tmp_path,monkeypatch):
    candidate=tmp_path/"candidate.bin"; candidate.write_bytes(b"candidate")
    csha=MOD.sha(candidate); point=tmp_path/"point.json"; surf=tmp_path/"surface.json"
    jwrite(point,{"candidate_sha256":csha,"complete_surface_audit":True,"inside_nodes":1})
    jwrite(surf,{"candidate_sha256":csha,"complete":True})
    state={"candidate":str(candidate),"candidate_sha256":csha,"point_summary":str(point),
           "surface_report":str(surf),"parent_fit":"fit"}
    state_path=tmp_path/"state.json"; jwrite(state_path,state)
    faces=np.array([[0,1,2]],np.int64)
    monkeypatch.setattr(MOD,"verify_file_pins",lambda:{})
    monkeypatch.setattr(MOD,"validate_state",lambda _:(state,candidate,faces,np.array([0])))
    monkeypatch.setattr(MOD,"seed_archive_index",lambda:({},{"verified":True}))
    monkeypatch.setattr(MOD,"collect_pairs",lambda *_a,**_k:({(0,1)},[{"seed":True}]))
    monkeypatch.setattr(MOD,"accumulate_pairs",lambda *_a,**_k:({(0,1)},[{"audit":True}]))
    monkeypatch.setattr(MOD,"surface_pair_count",lambda *_a,**_k:1)
    monkeypatch.setattr(MOD,"load_module",lambda *_a,**_k:object())
    face_npz=tmp_path/"face.npz"; face_npz.write_bytes(b"face")
    face_json=tmp_path/"face.json"; jwrite(face_json,{})
    pt_npz=tmp_path/"pt.npz"; pt_npz.write_bytes(b"point")
    pt_json=tmp_path/"pt.json"; jwrite(pt_json,{})
    monkeypatch.setattr(MOD,"run_face_targets",lambda *_a,**_k:(face_npz,face_json))
    monkeypatch.setattr(MOD,"run_point_targets",lambda *_a,**_k:(pt_npz,pt_json))
    trial=tmp_path/"trial.npz"; trial.write_bytes(b"trial")
    monkeypatch.setattr(MOD,"run_fit_attempt",lambda *_a,**_k:trial)
    monkeypatch.setattr(MOD,"run_backtracker",lambda *_a,**_k:None)
    calls={"learned":0}; newpair=(7,8,9,10)
    def learned(_):
        calls["learned"]+=1
        return set() if calls["learned"]==1 else {newpair}
    monkeypatch.setattr(MOD,"learned_fixed_bone_pairs",learned)
    monkeypatch.setattr(MOD,"require_headroom",lambda *_a,**_k:None)
    out=tmp_path/"out"
    monkeypatch.setattr(sys,"argv",[str(RUNNER),"--state",str(state_path),"--out",str(out),
                                   "--rounds","1","--attempts-per-round","1"])
    MOD.main()
    attempt=out/"round-001"/"attempt-001"
    assert calls["learned"]==3
    assert not (attempt/"stop-repeated-input.json").exists()
    assert json.loads((attempt/"result.json").read_text())["status"]=="NO_GLOBAL_OR_LOCAL_BACKTRACK_SELECTION"

