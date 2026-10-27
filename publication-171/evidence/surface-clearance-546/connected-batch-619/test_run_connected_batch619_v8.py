from __future__ import annotations
import copy
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

SRC=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/connected-batch-619/run_connected_batch619_v8.py')
spec=importlib.util.spec_from_file_location('batch619v8',SRC)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
ACTUAL_FIT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/connected-batch-619/run-002/round-001/cut-merge-001/fit')
ACTUAL_BASE=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/connected-batch-619/run-001/round-001/retention-582/candidate.npz')

def test_complete_path_is_unchanged_and_partial_is_opt_in(tmp_path):
    base=tmp_path/'base.npz';candidate=tmp_path/'complete.npz'
    np.savez(base,points_world_m=np.zeros((3,3),np.float32),x=np.arange(3,dtype=np.int64))
    np.savez(candidate,points_world_m=np.ones((3,3),np.float32),x=np.arange(3,dtype=np.int64))
    fit_report={'status':'CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED',
                'inputs':{'base_candidate_sha256':m.sha(base)},'candidate_path':str(candidate)}
    got,meta=m.select_fit_candidate(tmp_path,base,fit_report)
    assert got==candidate and meta=={'candidate_kind':'complete','fit_status':'CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED'}
    partial={'status':'LP_NO_PROPOSAL','inputs':{'base_candidate_sha256':m.sha(base)}}
    with pytest.raises(ValueError,match='--allow-certified-partial'):
        m.select_fit_candidate(tmp_path,base,partial)

def test_actual_saved_partial_is_read_only_receipt_validated():
    fit_report=m.readj(ACTUAL_FIT/'report.json')
    candidate,meta=m.select_fit_candidate(ACTUAL_FIT,ACTUAL_BASE,fit_report,allow_certified_partial=True)
    assert candidate==ACTUAL_FIT/'raw-joint-capture-563/raw-direction-001/candidate.npz'
    assert m.sha(candidate)=='9e118c3a4850904afbdc9dad243f362f6ee4bbe308d4435461ce516df67462d4'
    assert meta['candidate_kind']=='certified_partial'
    assert meta['partial_timeout']['completed_components']==13
    assert meta['partial_timeout']['failed_component']==7
    assert meta['partial_timeout']['zeroed_components']==[7]
    assert meta['partial_timeout']['global_lp_converged'] is False
    assert meta['partial_timeout']['global_l1_optimality_claim'] is False
    assert meta['partial_timeout']['retained_gain_budget_claimed'] is False

def test_missing_or_forged_partial_receipt_fails_closed():
    fit_report=m.readj(ACTUAL_FIT/'report.json')
    missing=copy.deepcopy(fit_report);missing.pop('full_coordinate_partial_timeout564')
    with pytest.raises(ValueError,match='missing 564 partial-timeout receipt'):
        m.select_fit_candidate(ACTUAL_FIT,ACTUAL_BASE,missing,allow_certified_partial=True)
    forged=copy.deepcopy(fit_report)
    forged['full_coordinate_partial_timeout564']['partial_timeout_zero_fallback_opt_in']=False
    with pytest.raises(ValueError,match='partial_timeout_zero_fallback_opt_in'):
        m.select_fit_candidate(ACTUAL_FIT,ACTUAL_BASE,forged,allow_certified_partial=True)

def test_forged_hard_row_certificate_fails_closed(monkeypatch):
    fit_report=m.readj(ACTUAL_FIT/'report.json')
    direction_path=ACTUAL_FIT/'raw-joint-capture-563/raw-direction-001/report.json'
    original=m.readj
    def altered_readj(path):
        value=original(path)
        if Path(path).resolve()==direction_path.resolve():
            value=copy.deepcopy(value)
            value['solver_receipt']['partial_fallback_receipt']['global_hard_rows_pass']=False
        return value
    monkeypatch.setattr(m,'readj',altered_readj)
    with pytest.raises(ValueError,match='global_hard_rows_pass'):
        m.select_fit_candidate(ACTUAL_FIT,ACTUAL_BASE,fit_report,allow_certified_partial=True)

def test_stale_or_wrong_base_is_rejected(tmp_path):
    fit_report=m.readj(ACTUAL_FIT/'report.json')
    with np.load(ACTUAL_BASE,allow_pickle=False) as z:
        arrays={k:np.array(z[k],copy=True) for k in z.files}
    arrays['points_world_m'][0,0]+=np.float32(1e-4)
    wrong=tmp_path/'wrong-base.npz';np.savez_compressed(wrong,**arrays)
    with pytest.raises(ValueError,match='base hash mismatch'):
        m.select_fit_candidate(ACTUAL_FIT,wrong,fit_report,allow_certified_partial=True)

def test_v4_source_is_unchanged_and_explicitly_pinned():
    assert m.sha(m.FROZEN_V4)==m.FROZEN_V4_SHA


def good_check(contained=2):
    return {"status":"COMPLETE_EXACT_CONSTRUCTION_CHECKS","arithmetic_pass":True,
            "self_pairs":0,"contained_pairs":contained,"complete_banks":185}

def test_restore_is_explicit_and_retained_check_must_be_complete():
    assert m.contained_restore_needed(good_check(0),False) is False
    with pytest.raises(ValueError,match="--restore-contained"):
        m.contained_restore_needed(good_check(2),False)
    assert m.contained_restore_needed(good_check(2),True) is True
    bad=good_check(2);bad["self_pairs"]=1
    with pytest.raises(ValueError,match="complete 545"):
        m.contained_restore_needed(bad,True)
    bad=good_check(2);bad["status"]="FAILED"
    with pytest.raises(ValueError,match="complete 545"):
        m.contained_restore_needed(bad,True)

def test_cli_exposes_restore_as_opt_in(monkeypatch,tmp_path):
    captured={}
    monkeypatch.setattr(m,"run",lambda a: captured.update(value=a.restore_contained) or {"status":"MAX_ROUNDS_COMPLETE","rounds":[]})
    args=["--state",str(tmp_path/"s"),"--prior-context",str(tmp_path/"p"),"--ancestry-context",str(tmp_path/"a"),
          "--failed-candidate",str(tmp_path/"c"),"--failed-audit130",str(tmp_path/"q"),
          "--failed-pairs",str(tmp_path/"pairs"),"--out",str(tmp_path/"o")]
    assert m.main(args+["--restore-contained"])==0 and captured["value"] is True
    assert m.main(args)==0 and captured["value"] is False

def test_restored_finalizer_inputs_are_hash_bound_and_strict(tmp_path):
    base=tmp_path/"base.npz";target=tmp_path/"target.npz";restored=tmp_path/"restored.npz"
    for p,b in ((base,b"base"),(target,b"target"),(restored,b"restored")):p.write_bytes(b)
    targetcheck=tmp_path/"target-check.json";targetcheck.write_text("{}")
    restore=tmp_path/"restore.json";restore.write_text("{}")
    fresh=tmp_path/"fresh-check.json";fresh.write_text("{}")
    point=tmp_path/"point.json";point.write_text("{}")
    surf=tmp_path/"surface.json";surf.write_text("{}")
    pin=tmp_path/"owner.py";pin.write_text("frozen")
    restore_data={"status":"CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED",
       "candidate_path":str(restored),"candidate_sha256":m.sha(restored),
       "base_candidate_sha256":m.sha(base),"target_candidate":str(target),"target_candidate_sha256":m.sha(target),
       "check_report":str(targetcheck),"check_report_sha256":m.sha(targetcheck),
       "restored_inside_output_rows":[3],"inside_witness_count":1,"unique_inside_node_count":1,
       "outside_node_field_bitwise_unchanged":True,"all_nonpoint_payload_bitwise_identical_to_target":True,
       "typed_anchors_bitwise_unchanged":True,"requires_582_and_fresh_full_545_before_strict_finalization":True,
       "source_pins":{str(pin):m.sha(pin),str(m.RESTORE620.resolve()):m.sha(m.RESTORE620),
          str(base.resolve()):m.sha(base),str(target.resolve()):m.sha(target),
          str(targetcheck.resolve()):m.sha(targetcheck)}}
    restore.write_text(__import__("json").dumps(restore_data))
    ck={"status":"COMPLETE_EXACT_CONSTRUCTION_CHECKS","candidate_sha256":m.sha(restored),"candidate_path":str(restored),
        "arithmetic_pass":True,"self_pairs":0,"contained_pairs":0,"complete_banks":185}
    fresh.write_text(__import__("json").dumps(ck))
    pt={"candidate_sha256":m.sha(restored),"complete_surface_audit":True,"inside_node_bank_pairs":0,"inside_nodes":0,"exact_PLC_surface_pairs":0}
    point.write_text(__import__("json").dumps(pt))
    sr={"candidate_sha256":m.sha(restored),"status":"COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT","complete":True,
        "bank_count":185,"expected_bank_count":185,"banks":[{} for _ in range(185)]}
    surf.write_text(__import__("json").dumps(sr))
    result=m.validate_restored_bundle(base,target,targetcheck,restore,restored,ck,pt,sr)
    assert result["candidate"]==str(restored) and result["candidate_sha256"]==m.sha(restored)
    assert result["inside_rows_restored"]==[3]
    argv=m.build_finalizer546_argv(result["candidate"],base,point,surf,tmp_path/"strict-final")
    assert Path(argv[argv.index("--candidate")+1])==restored
    assert Path(argv[argv.index("--point-summary")+1])==point
    assert Path(argv[argv.index("--surface-report")+1])==surf
    pins=restore_data["source_pins"].copy();pins.pop(str(target.resolve()))
    restore_data["source_pins"]=pins;restore.write_text(__import__("json").dumps(restore_data))
    with pytest.raises(ValueError,match="required source pin"):
        m.validate_restored_bundle(base,target,targetcheck,restore,restored,ck,pt,sr)
    restore_data["source_pins"][str(target.resolve())]=m.sha(target)
    restore.write_text(__import__("json").dumps(restore_data))
    ck["contained_pairs"]=1
    with pytest.raises(ValueError,match="strict 545 gates"):
        m.validate_restored_bundle(base,target,targetcheck,restore,restored,ck,pt,sr)

def test_v5_and_620_restore_sources_are_pinned():
    assert m.sha(m.FROZEN_V5)==m.FROZEN_V5_SHA
    assert m.sha(m.RESTORE620)==m.RESTORE620_SHA


def test_frozen_v7_source_is_pinned():
    assert m.sha(m.FROZEN_V7)==m.FROZEN_V7_SHA=="383e889de216a960db3de4d7b76fae613c4316f132ee1856765e9930106cae60"

def test_actual_901_restoration_bundle_and_round_receipt_are_json_serializable():
    import json
    root=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546")
    rd=root/"connected-batch-619/run-004-from1002/round-001"
    base=root/"connected-topology-580/run-005-from1043/lineage/candidate.npz"
    target=rd/"retention-582/candidate.npz"
    target_check=rd/"retained-check/report.json"
    restore_report=rd/"restore-620/report.json"
    restored=rd/"restore-620/candidate.npz"
    fresh=m.readj(rd/"restored-check/report.json")
    point=m.readj(rd/"restored-check/audit-130/summary.json")
    surface=m.readj(rd/"restored-check/audit-159/report.json")
    final=m.readj(rd/"final/report.json")
    assert final["surface_pairs"]==901 and final["contained_pairs"]==0 and final["self_pairs"]==0
    assert fresh["candidate_sha256"]==m.sha(restored)
    assert point["inside_nodes"]==0 and point["exact_PLC_surface_pairs"]==0
    bundle=m.validate_restored_bundle(base,target,target_check,restore_report,restored,fresh,point,surface)
    assert isinstance(bundle["candidate"],str) and bundle["candidate"]==str(restored.resolve())
    round_receipt={"round":1,"strict_finalizer_status":final["status"],
                   "restore_receipt":bundle,"restored_candidate_sha256":m.sha(restored)}
    encoded=json.dumps(round_receipt,sort_keys=True,allow_nan=False)
    decoded=json.loads(encoded)
    assert decoded["restore_receipt"]["candidate"]==str(restored.resolve())
    assert decoded["restore_receipt"]["inside_rows_restored"]==[16710]


def test_extra_cut_budget_default_parity_and_bounds():
    assert m.DEFAULT_EXTRA_CUT_FITS==2
    assert m.validate_extra_cut_fits(2)==2
    assert m.MAX_TOTAL_CUT_FITS==5
    for n in (0,1,2,3,4): assert m.validate_extra_cut_fits(n)==n
    for n in (-1,5,True,2.0):
        with pytest.raises(ValueError,match="integer in"): m.validate_extra_cut_fits(n)

def test_extra_cut_progress_requires_strict_certified_decline():
    attempts=[{"exact_self_failure":True,"self_pairs":501},
              {"exact_self_failure":True,"self_pairs":54},
              {"exact_self_failure":True,"self_pairs":15}]
    assert m.extra_cut_extension_decision(attempts,3,300*1024*1024,4)=={
        "allowed":True,"reason":"strict_self_progress_and_resource_guards_pass"}
    attempts.append({"exact_self_failure":True,"self_pairs":15})
    assert m.extra_cut_extension_decision(attempts,4,300*1024*1024,4)["reason"]=="no_strict_exact_self_pair_progress"

def test_extra_cut_extension_rejects_nonself_and_nonprogress():
    attempts=[{"exact_self_failure":True,"self_pairs":501},
              {"exact_self_failure":False,"self_pairs":54}]
    assert m.extra_cut_extension_decision(attempts,3,300*1024*1024,4)["reason"]=="latest_attempts_not_both_certified_exact_self_failures"
    attempts[1]={"exact_self_failure":True,"self_pairs":501}
    assert m.extra_cut_extension_decision(attempts,3,300*1024*1024,4)["reason"]=="no_strict_exact_self_pair_progress"
    assert m.extra_cut_extension_decision(attempts,2,300*1024*1024,4)=={"allowed":True,"reason":"historical_budget"}

def test_extra_cut_extension_obeys_capacity_and_request_cap():
    attempts=[{"exact_self_failure":True,"self_pairs":501},
              {"exact_self_failure":True,"self_pairs":54}]
    assert m.extra_cut_extension_decision(attempts,3,300*1024*1024,4)["allowed"] is True
    assert m.extra_cut_extension_decision(attempts,3,200*1024*1024,4)["reason"]=="resource_guard_retain-last-fit"
    assert m.extra_cut_extension_decision(attempts,3,100*1024*1024,4)["reason"]=="resource_guard_stop"
    assert m.extra_cut_extension_decision(attempts,3,300*1024*1024,2)["reason"]=="requested_cut_budget_exhausted"

def test_cli_extra_cut_budget_defaults_to_two_and_accepts_four(monkeypatch,tmp_path):
    captured={}
    monkeypatch.setattr(m,"run",lambda a: captured.update(value=a.extra_cut_fits) or {"status":"MAX_ROUNDS_COMPLETE","rounds":[]})
    args=["--state",str(tmp_path/"s"),"--prior-context",str(tmp_path/"p"),"--ancestry-context",str(tmp_path/"a"),
          "--failed-candidate",str(tmp_path/"c"),"--failed-audit130",str(tmp_path/"q"),
          "--failed-pairs",str(tmp_path/"pairs"),"--out",str(tmp_path/"o")]
    assert m.main(args)==0 and captured["value"]==2
    assert m.main(args+["--extra-cut-fits","4"])==0 and captured["value"]==4
    with pytest.raises(SystemExit): m.main(args+["--extra-cut-fits","5"])
