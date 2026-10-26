"""Independent construction-only verification for selected point-first backtrack 446.

Consumes the exact recorded 130/159 receipts; intentionally does not rerun the
expensive exact audits or any solver. The 159 report remains the authoritative
complete 185-bank triangle-crossing receipt.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
BASE=D/"D/grouped-backtrack-442/run-001/backtrack-373/pass-007/candidate.npz"
RUN=D/"D/point-first-backtrack-446/run-001/backtrack-373"
STATE=RUN/"selected-result.json"
FIT=D/"barycentric-connected-clearance-170/point-first-fit-445/run-001"
SOURCE=D/"full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz"
OUT=RUN/"root-independent-construction-verification.json"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def import_401():
    path=D/"D/retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py"
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location("verify446_401",path)
    if spec is None or spec.loader is None: raise RuntimeError("cannot import pinned 401 owner")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def typed_point_pairs(summary):
    rows=summary.get("inside_witnesses")
    if not isinstance(rows,list): raise AssertionError("point summary lacks inside witnesses")
    out=set()
    for row in rows:
        out.add((int(row["query_row"]),int(row["bank_body"]),int(row["bank_stable_id"])))
    if len(out)!=len(rows): raise AssertionError("duplicate typed point witnesses")
    return out


def typed_surface_pairs(report):
    if report.get("status")!="COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT" or report.get("complete") is not True:
        raise AssertionError("surface report is not complete")
    banks=report.get("banks")
    if not isinstance(banks,list) or len(banks)!=185 or report.get("bank_count")!=185 or report.get("expected_bank_count")!=185:
        raise AssertionError("surface report does not cover exactly 185 banks")
    out=set(); summed=0
    for bank in banks:
        body=int(bank["body"]); stable=int(bank["stable_bone_id"]); result=bank.get("result")
        if result is None:
            # The complete 159 exporter represents banks with no active broadphase candidates as null.
            continue
        if not isinstance(result,dict): raise AssertionError("malformed exact bank result")
        pairs=result.get("triangle_pairs")
        if not isinstance(pairs,list) or int(result.get("count",-1))!=len(pairs):
            raise AssertionError("exact bank pair count/list mismatch")
        summed+=len(pairs)
        for skin_face,bank_triangle in pairs:
            item=(int(skin_face),body,stable,int(bank_triangle))
            if item in out: raise AssertionError("duplicate typed exact surface pair")
            out.add(item)
    if len(out)!=summed: raise AssertionError("surface identity count mismatch")
    return out,summed


def main():
    state=load_json(STATE); candidate=Path(state["candidate"])
    if candidate.resolve()!= (RUN/"pass-015/candidate.npz").resolve():
        raise AssertionError("selected 446 candidate path is not pass-015")
    if sha(candidate)!=state["candidate_sha256"]: raise AssertionError("selected candidate SHA mismatch")
    if sha(BASE)!=load_json(D/"D/grouped-backtrack-442/run-001/backtrack-373/selected-result.json")["candidate_sha256"]:
        raise AssertionError("442 baseline bytes do not match selected state")
    fit_inputs=load_json(RUN/"source-inputs.json")
    fit_candidate=FIT/"candidate.npz"; raw_fit=FIT/"raw-lp-candidate.npz"
    if fit_inputs.get("start_sha256")!=sha(BASE) or fit_inputs.get("trial_sha256")!=sha(fit_candidate):
        raise AssertionError("446 fit input receipt does not bind 442 base and 445 trial")

    a=np.load(BASE,allow_pickle=False); b=np.load(candidate,allow_pickle=False)
    if set(a.files)!=set(b.files): raise AssertionError("candidate array key set changed")
    for key in a.files:
        if key=="points_world_m": continue
        if a[key].dtype!=b[key].dtype or a[key].shape!=b[key].shape or a[key].tobytes()!=b[key].tobytes():
            raise AssertionError(f"non-coordinate source array changed: {key}")
    anchors=np.asarray(a["typed_coarse_anchor_output_rows_018"],dtype=np.int64)
    if len(anchors)!=237 or not np.array_equal(a["points_world_m"][anchors],b["points_world_m"][anchors]):
        raise AssertionError("237 typed coarse anchors are not bitwise preserved")

    raw=np.load(raw_fit,allow_pickle=False)
    support=np.asarray(raw["local_witness_cluster_222_lp_support_rows"],dtype=np.int64)
    if support.ndim!=1 or len(np.unique(support))!=len(support) or np.any(support<0) or np.any(support>=len(a["points_world_m"])):
        raise AssertionError("445 raw point-first support rows are malformed")
    outside=np.setdiff1d(np.arange(len(a["points_world_m"]),dtype=np.int64),support,assume_unique=False)
    if not np.array_equal(a["points_world_m"][outside],b["points_world_m"][outside]):
        raise AssertionError("446 changed nodes outside the 445 raw LP support")

    owner=import_401()
    source_reference,source_receipt=owner.compose_source_reference(a,SOURCE)
    base_points=np.asarray(a["points_world_m"],dtype=np.float32)
    final_points=np.asarray(b["points_world_m"],dtype=np.float32)
    step=np.linalg.norm(final_points.astype(np.float64)-base_points.astype(np.float64),axis=1)*1000.0
    source_disp=np.linalg.norm(final_points.astype(np.float64)-source_reference,axis=1)*1000.0
    if float(step.max(initial=0.0))>1.0+1e-6: raise AssertionError("446 exceeds 1 mm construction step cap")
    if float(source_disp.max(initial=0.0))>12.0+1e-6: raise AssertionError("446 exceeds 12 mm source cap")
    area=owner.full_f32_area_gate(final_points,base_points.astype(np.float64)*1000.0,
        np.asarray(a["area_reference_037_points_world_m_f64"],dtype=np.float64)*1000.0,
        a["source_O_faces"],0.1)
    if not area.get("pass"): raise AssertionError("446 exact emitted Float32 area gate failed")

    point_path=Path(state["point_summary"]); surface_path=Path(state["surface_report"])
    q=load_json(point_path); s=load_json(surface_path)
    if sha(point_path)!=state["point_summary_sha256"]:
        raise AssertionError("selected point receipt byte hash mismatch")
    cand_sha=sha(candidate)
    if q.get("candidate_sha256")!=cand_sha or s.get("candidate_sha256")!=cand_sha:
        raise AssertionError("exact audit receipt candidate SHA mismatch")
    if q.get("exact_PLC_surface_pairs")!=0 or q.get("complete_surface_audit") is not True:
        raise AssertionError("130 summary does not record complete zero-self-pair audit")
    if q.get("points")!=20889 or q.get("faces")!=41733:
        raise AssertionError("130 summary topology mismatch")
    final_point_pairs=typed_point_pairs(q)
    if len(final_point_pairs)!=int(q["inside_node_bank_pairs"]):
        raise AssertionError("typed point witnesses differ from 130 summary count")
    surface_pairs,final_surface_count=typed_surface_pairs(s)

    base_run=D/"D/grouped-backtrack-442/run-001/backtrack-373/pass-007"
    qb=load_json(base_run/"audit-130/summary.json"); sb=load_json(base_run/"audit-159/report.json")
    if qb.get("candidate_sha256")!=sha(BASE) or sb.get("candidate_sha256")!=sha(BASE):
        raise AssertionError("442 baseline exact receipts do not bind baseline candidate")
    base_point_pairs=typed_point_pairs(qb); base_surface_pairs,base_surface_count=typed_surface_pairs(sb)
    introduced_surface=surface_pairs-base_surface_pairs; resolved_surface=base_surface_pairs-surface_pairs
    introduced_points=final_point_pairs-base_point_pairs; resolved_points=base_point_pairs-final_point_pairs
    if len(introduced_surface)!=60 or len(resolved_surface)!=67:
        raise AssertionError("exact 159 typed surface pair deltas disagree with selected result")
    if introduced_points: raise AssertionError("446 introduced typed contained node/bank pairs")

    result=load_json(RUN/"pass-015/result.json")
    pm=result["point_merit"]; jm=result["joint_merit"]; dm=result["depth_merit"]
    if (pm["before_inside_pairs"]!=len(base_point_pairs) or pm["after_inside_pairs"]!=len(final_point_pairs)
            or pm["introduced_typed_pairs"] or len(resolved_points)!=15
            or jm["before_surface_pairs"]!=base_surface_count or jm["after_surface_pairs"]!=final_surface_count
            or dm["introduced_surface_pair_count"]!=len(introduced_surface)):
        raise AssertionError("selected merit summary conflicts with exact 130/159 receipts")
    if not pm["maximum_depth_nonincreasing"] or not pm["squared_depth_nonincreasing"]:
        raise AssertionError("point depth merit regressed")
    if not jm["strict_surface_pair_reduction"] or not jm["strict_total_pair_reduction"]:
        raise AssertionError("joint count merit did not strictly improve")
    if jm.get("final_zero_clearance_unchanged") is not False:
        raise AssertionError("zero clearances should remain unachieved in this construction receipt")
    seed_scale=load_json(RUN.parent/"seed-scale-report.json")
    if seed_scale.get("status")!="GRADED_426_WITH_PINNED_SELF_SEED_ALPHA_CUTS_431":
        raise AssertionError("446 seed-scale report does not identify the actual 431 projector")
    seed_lp=seed_scale["seed_cut_lp"]
    if not seed_lp.get("lp_success") or seed_lp.get("seed_violated_rows")!=0 or seed_lp.get("seed_rows_checked")!=seed_lp.get("seed_rows_total"):
        raise AssertionError("446 431 seed row certificate failed or incomplete")

    cert={"status":"ROOT_INDEPENDENT_CONSTRUCTION_PASS",
        "candidate":str(candidate),"candidate_sha256":cand_sha,"selected_state_sha256":sha(STATE),
        "base_candidate":str(BASE),"base_candidate_sha256":sha(BASE),
        "fit_445_candidate_sha256":sha(fit_candidate),"fit_445_raw_candidate_sha256":sha(raw_fit),
        "fit_445_report_sha256":sha(FIT/"report.json"),"fit_445_raw_support_rows":int(len(support)),
        "fit_445_raw_support_sha256":hashlib.sha256(np.ascontiguousarray(support,dtype="<i8").tobytes()).hexdigest(),
        "outside_fit_support_unchanged":True,"changed_nodes":int(np.count_nonzero(step)),
        "source_arrays_bitwise_unchanged":int(len(a.files)-1),"anchors_237_bitwise_unchanged":True,
        "max_step_mm":float(step.max(initial=0.0)),"max_source_displacement_mm":float(source_disp.max(initial=0.0)),
        "area_gate":area,"source_reference_receipt":source_receipt,
        "point_pairs_before":len(base_point_pairs),"point_pairs_after":len(final_point_pairs),
        "point_pairs_resolved":len(resolved_points),"point_pairs_introduced":len(introduced_points),
        "max_point_depth_before_m":pm["before_maximum_depth_m"],"max_point_depth_after_m":pm["after_maximum_depth_m"],
        "squared_depth_before_m2":pm["before_squared_depth_m2"],"squared_depth_after_m2":pm["after_squared_depth_m2"],
        "surface_pairs_before":base_surface_count,"surface_pairs_after":final_surface_count,
        "surface_pairs_resolved":len(resolved_surface),"surface_pairs_introduced":len(introduced_surface),
        "total_pairs_before":jm["before_total_pairs"],"total_pairs_after":jm["after_total_pairs"],
        "typed_bank_count":185,"complete_185_bank_surface_receipt":True,"self_pairs":0,
        "point_summary_sha256":sha(point_path),"surface_report_sha256":sha(surface_path),
        "base_point_summary_sha256":sha(base_run/"audit-130/summary.json"),
        "base_surface_report_sha256":sha(base_run/"audit-159/report.json"),
        "seed_scale_status":seed_scale["status"],"seed_rows_checked":seed_lp["seed_rows_checked"],
        "seed_rows_violated":seed_lp["seed_violated_rows"],
        "fixed_bone_313_rows_in_446":False,
        "native_admission":False,"accepted_physical_steps":0,
        "geometry_admission":False,"zero_clearance_achieved":False}
    OUT.write_text(json.dumps(cert,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(cert,sort_keys=True,indent=2))

if __name__=="__main__": main()
