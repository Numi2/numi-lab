#!/usr/bin/env python3
"""Restore only exact contained witness rows to a pinned zero-contained baseline."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):
    x=json.loads(Path(p).read_text())
    if not isinstance(x,dict): raise ValueError("expected JSON object: "+str(p))
    return x
def writej(p,x):
    Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
def loadnpz(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def same(a,b):
    a=np.ascontiguousarray(a);b=np.ascontiguousarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def contained_rows(summary,npoints):
    if summary.get("complete_surface_audit") is not True or type(summary.get("inside_node_bank_pairs")) is not int or summary["inside_node_bank_pairs"]<=0:
        raise ValueError("130 report has no complete positive contained-witness set")
    witnesses=summary.get("inside_witnesses")
    if not isinstance(witnesses,list) or not witnesses: raise ValueError("130 report lacks inside_witnesses")
    if len(witnesses)!=summary["inside_node_bank_pairs"]: raise ValueError("inside witness count differs from 130 inside pairs")
    rows=[]
    for w in witnesses:
        if not isinstance(w,dict) or w.get("classification")!="inside": raise ValueError("invalid 130 inside witness")
        row=w.get("output_node")
        if type(row) is not int or not 0<=row<npoints: raise ValueError("inside witness output row invalid")
        rows.append(row)
    rows=np.unique(np.asarray(rows,dtype=np.int64))
    if len(rows)!=summary.get("inside_nodes"): raise ValueError("unique contained node rows disagree with 130 inside_nodes")
    return rows
def restore_rows(base,target,rows):
    if set(base)!=set(target): raise ValueError("candidate NPZ field set differs")
    if base["points_world_m"].dtype!=np.float32 or target["points_world_m"].dtype!=np.float32:
        raise ValueError("point fields must remain Float32")
    bp=base["points_world_m"];tp=target["points_world_m"]
    if bp.ndim!=2 or bp.shape[1]!=3 or tp.shape!=bp.shape or not np.isfinite(bp).all() or not np.isfinite(tp).all():
        raise ValueError("point domain/nonfinite mismatch")
    for k in base:
        if k!="points_world_m" and not same(base[k],target[k]): raise ValueError("nonpoint payload changed: "+k)
    anchors=np.asarray(base["typed_coarse_anchor_output_rows_018"])
    if anchors.ndim!=1 or anchors.dtype.kind not in "iu" or np.any(anchors<0) or np.any(anchors>=len(bp)) or len(np.unique(anchors))!=len(anchors):
        raise ValueError("typed anchor rows malformed")
    if not same(bp[anchors],tp[anchors]): raise ValueError("target moved typed anchors")
    if np.intersect1d(rows,anchors).size: raise ValueError("contained witness includes a typed anchor")
    if np.any(same_rows(bp,tp,rows)): raise ValueError("contained witness row is coordinate-identical to baseline")
    out={k:v.copy() for k,v in target.items()}
    out["points_world_m"][rows]=bp[rows]
    unchanged=np.ones(len(bp),dtype=bool);unchanged[rows]=False
    if not same(out["points_world_m"][unchanged],tp[unchanged]): raise AssertionError("outside rows changed")
    steps=np.linalg.norm((tp[rows].astype(np.float64)-bp[rows].astype(np.float64))*1000.0,axis=1)
    return out,steps,unchanged
def same_rows(a,b,rows):
    return np.all(a[rows]==b[rows],axis=1)
def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-state","candidate","check-report","out"): ap.add_argument("--"+name,type=Path,required=True)
    a=ap.parse_args(argv)
    if a.out.exists(): raise FileExistsError(a.out)
    st=readj(a.baseline_state);check=readj(a.check_report)
    if st.get("native_admission") is not False or st.get("accepted_physical_steps")!=0: raise ValueError("baseline state is not construction-only")
    roles={k:Path(st[k]) for k in ("candidate","point_summary","surface_report")}
    for k,p in roles.items():
        if sha(p)!=st.get(k+"_sha256"): raise ValueError("baseline state hash mismatch: "+k)
    h=st["candidate_sha256"];bp=roles["point_summary"];bs=readj(bp);b159=readj(roles["surface_report"])
    if bs.get("candidate_sha256")!=h or bs.get("complete_surface_audit") is not True or bs.get("exact_PLC_surface_pairs")!=0 or bs.get("inside_node_bank_pairs")!=0:
        raise ValueError("baseline must have complete zero-self, zero-contained 130")
    bpair=bp.parent/"surface-self-pairs.npy"
    bpa=np.load(bpair,allow_pickle=False)
    if bpa.shape!=(0,2) or bpa.dtype.kind not in "iu": raise ValueError("baseline self-pair file is nonempty/malformed")
    if b159.get("candidate_sha256")!=h or b159.get("complete") is not True or b159.get("bank_count")!=185 or b159.get("expected_bank_count")!=185 or len(b159.get("banks",[]))!=185:
        raise ValueError("baseline lacks complete 185-bank 159")
    ch=sha(a.candidate)
    if check.get("status")!="COMPLETE_EXACT_CONSTRUCTION_CHECKS" or check.get("candidate_sha256")!=ch or Path(check.get("candidate_path","")).resolve()!=a.candidate.resolve():
        raise ValueError("target is not the candidate in a complete 545 report")
    if check.get("arithmetic_pass") is not True or check.get("self_pairs")!=0 or check.get("complete_banks")!=185:
        raise ValueError("target 545 arithmetic/self/185 gates are not passed")
    audit130=a.check_report.parent/"audit-130/summary.json";audit159=a.check_report.parent/"audit-159/report.json"
    q=readj(audit130);s159=readj(audit159)
    if q.get("candidate_sha256")!=ch or q.get("exact_PLC_surface_pairs")!=0: raise ValueError("target 130 is not candidate-bound zero-self")
    if check.get("contained_pairs")!=q.get("inside_node_bank_pairs"): raise ValueError("545 contained count disagrees with candidate-bound 130")
    if s159.get("candidate_sha256")!=ch or s159.get("complete") is not True or s159.get("bank_count")!=185 or s159.get("expected_bank_count")!=185: raise ValueError("target 159 lacks complete 185-bank coverage")
    base=loadnpz(roles["candidate"]);target=loadnpz(a.candidate)
    rows=contained_rows(q,len(target["points_world_m"]))
    for w in q["inside_witnesses"]:
        r=int(w["output_node"])
        witness_point=np.asarray(w.get("point_m"),dtype=np.float32)
        if witness_point.shape!=(3,) or not np.isfinite(witness_point).all() or not same(witness_point,target["points_world_m"][r]):
            raise ValueError("130 witness coordinate contradicts candidate Float32 point row")
    payload,steps,outside=restore_rows(base,target,rows)
    pin_paths=(Path(__file__),a.baseline_state,roles["candidate"],bp,bpair,roles["surface_report"],a.candidate,a.check_report,audit130,audit159)
    pins={str(p.resolve()):sha(p) for p in pin_paths}
    if any(sha(p)!=pins[str(p.resolve())] for p in pin_paths): raise RuntimeError("pinned input changed during validation")
    a.out.mkdir(parents=True,exist_ok=False);op=a.out/"candidate.npz";np.savez_compressed(op,**payload)
    report={"status":"CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED","baseline_state":str(a.baseline_state),
      "base_candidate_sha256":h,"candidate_path":str(op),"candidate_sha256":sha(op),
      "target_candidate":str(a.candidate),"target_candidate_sha256":ch,"check_report":str(a.check_report),
      "check_report_sha256":sha(a.check_report),"restored_inside_output_rows":rows.tolist(),
      "inside_witness_count":int(q["inside_node_bank_pairs"]),"unique_inside_node_count":len(rows),
      "restored_row_steps_from_target_mm":steps.tolist(),"maximum_restored_row_step_mm":float(steps.max(initial=0)),
      "outside_node_field_bitwise_unchanged":bool(same(payload["points_world_m"][outside],target["points_world_m"][outside])),
      "all_nonpoint_payload_bitwise_identical_to_target":all(same(payload[k],target[k]) for k in target if k!="points_world_m"),
      "typed_anchors_bitwise_unchanged":same(payload["points_world_m"][base["typed_coarse_anchor_output_rows_018"]],base["points_world_m"][base["typed_coarse_anchor_output_rows_018"]]),
      "source_pins":pins,"requires_582_and_fresh_full_545_before_strict_finalization":True,
      "native_admission":False,"accepted_physical_steps":0}
    writej(a.out/"report.json",report)
    if any(sha(p)!=pins[str(p.resolve())] for p in pin_paths): raise RuntimeError("pinned input changed during write")
    print(json.dumps({"status":report["status"],"rows":len(rows),"max_step_mm":report["maximum_restored_row_step_mm"],"candidate_sha256":report["candidate_sha256"]},sort_keys=True))
if __name__=="__main__": main()
