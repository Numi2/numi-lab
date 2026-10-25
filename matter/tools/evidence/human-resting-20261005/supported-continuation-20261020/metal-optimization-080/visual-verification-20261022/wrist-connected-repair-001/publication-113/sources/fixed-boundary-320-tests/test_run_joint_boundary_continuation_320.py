from __future__ import annotations
import importlib.util, json
from pathlib import Path
import numpy as np
import pytest

RUNNER=Path("/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/run_joint_boundary_continuation_320.py")
SPEC=importlib.util.spec_from_file_location("joint_boundary_320_tests",RUNNER)
MOD=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)

def jwrite(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value))

def bank_report(pairs=None, *, point_count=20889, face_count=41733):
    pairs=pairs or {}; banks=[]
    for i in range(185):
        ps=pairs.get(i,[])
        banks.append({"body":i,"stable_bone_id":10000+i,"skin_faces":face_count,
            "skin_aabb_active":int(bool(ps)),"bank_aabb_active":int(bool(ps)),
            "result":({"count":len(ps),"triangle_pairs":[list(p) for p in ps]} if ps else None)})
    return {"status":"COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT","complete":True,
        "candidate_sha256":"d"*64,"surface_point_count":point_count,"surface_face_count":face_count,
        "script_sha256":MOD.PINS["audit159_sha256"],"exact_predicate_sha256":MOD.SEED_SURFACE_PREDICATE_SHA256,
        "bank_count":185,"expected_bank_count":185,"banks":banks}

def state_for(surface):
    p=surface.parent/"state.json"; jwrite(p,{"surface_report":str(surface)}); return p

def test_strict_typed_parser_uses_pinned_base_bank_inventory(tmp_path,monkeypatch):
    base=tmp_path/"base-surface.json"; jwrite(base,bank_report())
    monkeypatch.setattr(MOD,"DEFAULT_STATE",state_for(base))
    got=MOD.typed_bone_pairs(bank_report({0:[(7,8)],1:[(7,8)]}))
    assert got=={(0,10000,7,8),(1,10001,7,8)}

@pytest.mark.parametrize("mutation",["missing","duplicate","out_of_range","duplicate_pair"])
def test_strict_typed_parser_rejects_incomplete_or_malformed_inventory(tmp_path,monkeypatch,mutation):
    base=tmp_path/"base-surface.json"; jwrite(base,bank_report())
    monkeypatch.setattr(MOD,"DEFAULT_STATE",state_for(base))
    bad=bank_report({0:[(7,8)]})
    if mutation=="missing":
        bad["banks"]=bad["banks"][:-1]
    elif mutation=="duplicate":
        bad["banks"][1]["body"]=bad["banks"][0]["body"]
        bad["banks"][1]["stable_bone_id"]=bad["banks"][0]["stable_bone_id"]
    elif mutation=="out_of_range":
        bad["banks"][0]["result"]["triangle_pairs"]=[[41733,8]]
    else:
        bad["banks"][0]["result"]["triangle_pairs"]=[[7,8],[7,8]]
        bad["banks"][0]["result"]["count"]=2
    with pytest.raises((ValueError,KeyError)):
        MOD.typed_bone_pairs(bad)

def test_accumulate_pairs_unions_all_three_roots_and_marks_current_optional(tmp_path,monkeypatch):
    faces=np.array([[0,1,2]],dtype=np.int64)
    roots=[tmp_path/"seed318",tmp_path/"extra309",tmp_path/"current"]
    monkeypatch.setattr(MOD,"SEED_ROOT",roots[0])
    monkeypatch.setattr(MOD,"EXTRA_SEED_ROOT",roots[1])
    monkeypatch.setattr(MOD,"CURRENT_OUT",roots[2],raising=False)
    values={str(roots[0]):({(1,2)},[{"root":"318"}]),
            str(roots[1]):({(2,3)},[{"root":"309"}]),
            str(roots[2]):({(3,4)},[{"root":"current"}])}
    calls=[]
    def collect(root,_faces,_archive,*,allow_missing_local_candidate=False):
        calls.append((root,allow_missing_local_candidate))
        return values[str(root)]
    monkeypatch.setattr(MOD,"collect_pairs",collect)
    pairs,receipts=MOD.accumulate_pairs(faces,{})
    assert pairs=={(1,2),(2,3),(3,4)}
    assert [r for r,_ in calls]==[roots[0],roots[2],roots[1]]
    assert [allow for _,allow in calls]==[False,True,False]
    assert [x["root"] for x in receipts]==["318","309","current"]

def test_prepared_point_targets_are_reused_only_for_exact_base_and_match_hashes(tmp_path,monkeypatch):
    candidate=tmp_path/"base.npz"; candidate.write_bytes(b"base candidate")
    candidate_sha=MOD.sha(candidate)
    monkeypatch.setattr(MOD,"PREPARED_BASE_SHA",candidate_sha)
    cache=tmp_path/"318"/"round-001"/"point-targets"
    rpt=cache/"nearest-union-exits.json"; npz=cache/"nearest-union-exits.npz"
    npz.parent.mkdir(parents=True); npz.write_bytes(b"target bytes")
    jwrite(rpt,{"inputs":{"candidate_sha256":candidate_sha},"passing_targets":2})
    monkeypatch.setattr(MOD,"PREPARED_TARGET_ROOT",cache.parent)
    monkeypatch.setattr(MOD,"PREPARED_POINT_REPORT_SHA",MOD.sha(rpt))
    monkeypatch.setattr(MOD,"PREPARED_POINT_NPZ_SHA",MOD.sha(npz))
    summary=tmp_path/"summary.json"; jwrite(summary,{"inside_nodes":2})
    out=tmp_path/"round"; out.mkdir()
    result_npz,result_report=MOD.run_point_targets(object(),{"point_summary":str(summary)},candidate,out)
    assert result_npz==npz and result_report==rpt
    assert json.loads((out/"reused-point-targets.json").read_text())["npz_sha256"]==MOD.sha(npz)

def test_prepared_point_cache_rejects_report_for_other_candidate(tmp_path,monkeypatch):
    candidate=tmp_path/"base.npz"; candidate.write_bytes(b"candidate")
    monkeypatch.setattr(MOD,"PREPARED_BASE_SHA",MOD.sha(candidate))
    cache=tmp_path/"318"/"round-001"/"point-targets"; cache.mkdir(parents=True)
    rpt=cache/"nearest-union-exits.json"; npz=cache/"nearest-union-exits.npz"
    jwrite(rpt,{"inputs":{"candidate_sha256":"e"*64},"passing_targets":1}); npz.write_bytes(b"npz")
    monkeypatch.setattr(MOD,"PREPARED_TARGET_ROOT",cache.parent)
    monkeypatch.setattr(MOD,"PREPARED_POINT_REPORT_SHA",MOD.sha(rpt))
    monkeypatch.setattr(MOD,"PREPARED_POINT_NPZ_SHA",MOD.sha(npz))
    summary=tmp_path/"summary.json"; jwrite(summary,{"inside_nodes":1})
    with pytest.raises(ValueError,match="candidate/count mismatch"):
        MOD.run_point_targets(object(),{"point_summary":str(summary)},candidate,tmp_path/"round")

def test_later_point_candidate_rebuilds_instead_of_using_318_cache(tmp_path,monkeypatch):
    candidate=tmp_path/"later.npz"; candidate.write_bytes(b"later candidate")
    monkeypatch.setattr(MOD,"PREPARED_BASE_SHA","a"*64)
    summary=tmp_path/"summary.json"; jwrite(summary,{"inside_nodes":0})
    emitter=tmp_path/"empty-emitter.py"; emitter.write_text("")
    monkeypatch.setattr(MOD,"POINT_EMITTER",emitter)
    seen={}
    def fake_run(command,log,*,slot=True,timeout=900):
        a=list(map(str,command)); seen["args"]=a; seen["slot"]=slot
        assert a[1]==str(emitter)
        rpt=Path(a[a.index("--report")+1]); npz=Path(a[a.index("--npz")+1])
        npz.write_bytes(b"new"); jwrite(rpt,{"status":"PASS_FIRST_UNION_EXIT_TARGETS_ONLY",
            "passing_targets":0,"inputs":{"candidate_sha256":MOD.sha(candidate)}})
        Path(log).write_text("")
    monkeypatch.setattr(MOD,"run",fake_run)
    out=tmp_path/"out"; out.mkdir()
    npz,rpt=MOD.run_point_targets(object(),{"point_summary":str(summary)},candidate,out)
    assert seen["slot"] and npz.is_file() and rpt.is_file()
    assert str(candidate) in seen["args"]

def test_prepared_face_targets_require_exact_base_hashes_and_complete_report(tmp_path,monkeypatch):
    candidate=tmp_path/"base.npz"; candidate.write_bytes(b"base")
    surface=tmp_path/"surface.json"; jwrite(surface,{"complete":True})
    monkeypatch.setattr(MOD,"PREPARED_BASE_SHA",MOD.sha(candidate))
    cache=tmp_path/"318"/"round-001"/"face-targets"; cache.mkdir(parents=True)
    npz=cache/"targets.npz"; npz.write_bytes(b"targets")
    builder=tmp_path/"builder.py"; builder.write_text("# builder\n")
    monkeypatch.setattr(MOD,"FACE_BUILDER",builder)
    report=cache/"report.json"; jwrite(report,{"runner_sha256":MOD.sha(builder),"direction_mode":"source-outward"})
    monkeypatch.setattr(MOD,"PREPARED_TARGET_ROOT",cache.parent)
    monkeypatch.setattr(MOD,"PREPARED_FACE_REPORT_SHA",MOD.sha(report))
    monkeypatch.setattr(MOD,"PREPARED_FACE_NPZ_SHA",MOD.sha(npz))
    captured={}
    class Checker:
        @staticmethod
        def validate_full_face_target_report(data,**kwargs):
            captured.update(kwargs); return {"status":"PASS_COMPLETE_MATCHED_FACE_TARGETS_ONLY"}
    monkeypatch.setattr(MOD,"load_module",lambda *_a,**_k:Checker)
    monkeypatch.setattr(MOD,"surface_pair_count",lambda *_a,**_k:23)
    out=tmp_path/"round"; out.mkdir()
    got_npz,got_report=MOD.run_face_targets(candidate,surface,out)
    assert got_npz==npz and got_report==report
    assert captured=={"expected_candidate_sha256":MOD.sha(candidate),
        "expected_surface_report_sha256":MOD.sha(surface),"expected_surface_pair_count":23,
        "expected_banks":185}
    reuse=json.loads((out/"reused-face-targets.json").read_text())
    assert reuse["complete"]["status"]=="PASS_COMPLETE_MATCHED_FACE_TARGETS_ONLY"

def test_later_face_candidate_rebuilds_and_runs_complete_preflight(tmp_path,monkeypatch):
    candidate=tmp_path/"later.npz"; candidate.write_bytes(b"later")
    surface=tmp_path/"surface.json"; jwrite(surface,{"complete":True})
    monkeypatch.setattr(MOD,"PREPARED_BASE_SHA","a"*64)
    builder=tmp_path/"builder.py"; builder.write_text("# builder\n")
    monkeypatch.setattr(MOD,"FACE_BUILDER",builder)
    monkeypatch.setattr(MOD,"surface_pair_count",lambda *_a,**_k:9)
    class Checker:
        @staticmethod
        def validate_full_face_target_report(data,**kwargs):
            assert kwargs["expected_candidate_sha256"]==MOD.sha(candidate)
            assert kwargs["expected_surface_report_sha256"]==MOD.sha(surface)
            assert kwargs["expected_surface_pair_count"]==9
            assert kwargs["expected_banks"]==185
            return {"status":"PASS"}
    monkeypatch.setattr(MOD,"load_module",lambda *_a,**_k:Checker)
    seen={}
    def fake_run(command,log,*,slot=True,timeout=900):
        a=list(map(str,command)); target=Path(a[a.index("--output-dir")+1])
        assert not target.exists()
        assert a[a.index("--direction-mode")+1]=="source-outward"
        target.mkdir(); (target/"targets.npz").write_bytes(b"targets")
        jwrite(target/"report.json",{"runner_sha256":MOD.sha(builder),"direction_mode":"source-outward",
            "inputs":{"candidate_sha256":MOD.sha(candidate)}})
        Path(log).write_text(""); seen["slot"]=slot
    monkeypatch.setattr(MOD,"run",fake_run)
    out=tmp_path/"round"; out.mkdir()
    npz,rpt=MOD.run_face_targets(candidate,surface,out)
    assert seen["slot"] and npz.is_file() and rpt.is_file()
    assert (out/"face-targets"/"completeness-preflight.json").is_file()

def test_first_face_cache_is_restricted_to_exact_309_base_candidate_and_surface(tmp_path,monkeypatch):
    candidate=tmp_path/"later.npz"; candidate.write_bytes(b"later")
    surface=tmp_path/"surface.json"; jwrite(surface,{})
    out=tmp_path/"round"; out.mkdir(); cache=tmp_path/"old-cache"
    with pytest.raises(ValueError,match="only applies to the exact initial 283"):
        MOD.run_face_targets(candidate,surface,out,cache)



def test_prepared_point_cache_rejects_changed_pinned_bytes(tmp_path,monkeypatch):
    candidate=tmp_path/"base.npz"; candidate.write_bytes(b"base")
    monkeypatch.setattr(MOD,"PREPARED_BASE_SHA",MOD.sha(candidate))
    cache=tmp_path/"318"/"round-001"/"point-targets"; cache.mkdir(parents=True)
    rpt=cache/"nearest-union-exits.json"; npz=cache/"nearest-union-exits.npz"
    jwrite(rpt,{"inputs":{"candidate_sha256":MOD.sha(candidate)},"passing_targets":0}); npz.write_bytes(b"npz")
    monkeypatch.setattr(MOD,"PREPARED_TARGET_ROOT",cache.parent)
    monkeypatch.setattr(MOD,"PREPARED_POINT_REPORT_SHA","0"*64)
    monkeypatch.setattr(MOD,"PREPARED_POINT_NPZ_SHA",MOD.sha(npz))
    summary=tmp_path/"summary.json"; jwrite(summary,{"inside_nodes":0})
    with pytest.raises(ValueError,match="prepared point target bytes changed"):
        MOD.run_point_targets(object(),{"point_summary":str(summary)},candidate,tmp_path/"round")

def test_prepared_face_cache_rejects_changed_bytes_before_preflight(tmp_path,monkeypatch):
    candidate=tmp_path/"base.npz"; candidate.write_bytes(b"base")
    surface=tmp_path/"surface.json"; jwrite(surface,{"complete":True})
    monkeypatch.setattr(MOD,"PREPARED_BASE_SHA",MOD.sha(candidate))
    cache=tmp_path/"318"/"round-001"/"face-targets"; cache.mkdir(parents=True)
    report=cache/"report.json"; npz=cache/"targets.npz"
    builder=tmp_path/"builder.py"; builder.write_text("# builder\n")
    jwrite(report,{"runner_sha256":MOD.sha(builder),"direction_mode":"source-outward"}); npz.write_bytes(b"npz")
    monkeypatch.setattr(MOD,"FACE_BUILDER",builder)
    monkeypatch.setattr(MOD,"PREPARED_TARGET_ROOT",cache.parent)
    monkeypatch.setattr(MOD,"PREPARED_FACE_REPORT_SHA","f"*64)
    monkeypatch.setattr(MOD,"PREPARED_FACE_NPZ_SHA",MOD.sha(npz))
    out=tmp_path/"round"; out.mkdir()
    with pytest.raises(ValueError,match="prepared face target bytes changed"):
        MOD.run_face_targets(candidate,surface,out)

def test_prepared_face_cache_rejects_incomplete_target_receipt(tmp_path,monkeypatch):
    candidate=tmp_path/"base.npz"; candidate.write_bytes(b"base")
    surface=tmp_path/"surface.json"; jwrite(surface,{"complete":True})
    monkeypatch.setattr(MOD,"PREPARED_BASE_SHA",MOD.sha(candidate))
    cache=tmp_path/"318"/"round-001"/"face-targets"; cache.mkdir(parents=True)
    report=cache/"report.json"; npz=cache/"targets.npz"
    builder=tmp_path/"builder.py"; builder.write_text("# builder\n")
    jwrite(report,{"runner_sha256":MOD.sha(builder),"direction_mode":"source-outward",
        "inputs":{"candidate_sha256":MOD.sha(candidate),"surface_report_sha256":MOD.sha(surface)}})
    npz.write_bytes(b"npz")
    monkeypatch.setattr(MOD,"FACE_BUILDER",builder)
    monkeypatch.setattr(MOD,"PREPARED_TARGET_ROOT",cache.parent)
    monkeypatch.setattr(MOD,"PREPARED_FACE_REPORT_SHA",MOD.sha(report))
    monkeypatch.setattr(MOD,"PREPARED_FACE_NPZ_SHA",MOD.sha(npz))
    monkeypatch.setattr(MOD,"surface_pair_count",lambda *_a,**_k:5)
    class RejectIncomplete:
        @staticmethod
        def validate_full_face_target_report(data,**kwargs):
            assert kwargs["expected_candidate_sha256"]==MOD.sha(candidate)
            assert kwargs["expected_surface_report_sha256"]==MOD.sha(surface)
            raise ValueError("incomplete target coverage")
    monkeypatch.setattr(MOD,"load_module",lambda *_a,**_k:RejectIncomplete)
    out=tmp_path/"round"; out.mkdir()
    with pytest.raises(ValueError,match="incomplete target coverage"):
        MOD.run_face_targets(candidate,surface,out)
