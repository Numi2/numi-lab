"""Transfer a saved coordinate proposal and restore exact inside-point witnesses.
All face/lineage data comes from the passing baseline; no LP-feasibility claim.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def main():
 p=argparse.ArgumentParser();p.add_argument("--state",type=Path,required=True);p.add_argument("--target",type=Path,required=True);p.add_argument("--point-summary",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 s=json.loads(a.state.read_text());h=json.loads(a.point_summary.read_text())
 for k in("candidate","point_summary","surface_report"):
  if sha(s[k])!=s[k+"_sha256"]:raise ValueError("baseline pin mismatch "+k)
 hb=json.loads(Path(s["point_summary"]).read_text())
 if hb.get("inside_node_bank_pairs")!=0 or hb.get("exact_PLC_surface_pairs")!=0 or hb.get("complete_surface_audit")is not True:raise ValueError("passing baseline required")
 if h.get("candidate_sha256")!=sha(a.target)or h.get("complete_surface_audit")is not True:raise ValueError("target witness summary mismatch")
 with np.load(s["candidate"])as z:b={k:np.array(z[k],copy=True)for k in z.files}
 with np.load(a.target)as z:t={k:np.array(z[k],copy=True)for k in z.files}
 if b["points_world_m"].dtype!=np.float32 or b["points_world_m"].shape!=t["points_world_m"].shape:raise ValueError("matched Float32 point domain required")
 binding_keys=[k for k in b if k.startswith(("source_P","source_Q","source_prolongation","typed_coarse_anchor"))or k in("node_kind","refinement_node_kind")]
 for k in binding_keys:
  if k not in t or not equal(b[k],t[k]):raise ValueError("point identity/ancestry mismatch "+k)
 rows=sorted({int(w["output_node"])for w in h.get("inside_witnesses",[])})
 if len(rows)!=int(h.get("inside_nodes",-1))or not rows:raise ValueError("complete nonempty inside witness set required")
 if min(rows)<0 or max(rows)>=len(b["points_world_m"]):raise ValueError("witness row out of range")
 anchors=np.asarray(b["typed_coarse_anchor_output_rows_018"],dtype=np.int64)
 if len(anchors)!=237 or not equal(b["points_world_m"][anchors],t["points_world_m"][anchors]):raise ValueError("anchor mismatch")
 output=t["points_world_m"].copy();output[rows]=b["points_world_m"][rows]
 if not np.isfinite(output).all():raise ValueError("nonfinite target")
 b["points_world_m"]=output;a.out.mkdir(parents=True,exist_ok=False);np.savez_compressed(a.out/"candidate.npz",**b)
 report={"status":"WITNESS_RESTORED_COORDINATE_TARGET_PENDING_FULL_CHECKS","baseline_state":str(a.state),"baseline_state_sha256":sha(a.state),"target_input":str(a.target),"target_input_sha256":sha(a.target),"witness_summary":str(a.point_summary),"witness_summary_sha256":sha(a.point_summary),"restored_rows":rows,"binding_fields_checked":binding_keys,"output_candidate":str(a.out/"candidate.npz"),"output_candidate_sha256":sha(a.out/"candidate.npz"),"all_other_target_coordinates_preserved":True,"all_noncoordinate_fields_from_passing_baseline":True,"source_path":str(Path(__file__).resolve()),"source_sha256":sha(__file__),"linear_feasibility_claim":False,"native_admission":False,"accepted_physical_steps":0}
 (a.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps({"candidate":report["output_candidate"],"restored_rows":rows}),flush=True)
if __name__=="__main__":main()

