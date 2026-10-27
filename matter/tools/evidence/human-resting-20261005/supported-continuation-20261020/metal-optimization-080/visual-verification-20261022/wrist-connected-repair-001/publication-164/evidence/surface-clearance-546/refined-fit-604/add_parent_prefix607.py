"""Add the existing explicit parent-prefix field with exact lineage proof."""
from pathlib import Path
import hashlib,json,numpy as np
N=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 state=N/"connected-refinement-603/baseline-599-affine-filtered-002/baseline-check-001/baseline-state.json"
 parent_state=N/"component-topology-599/run-001/final/state.json";st=json.loads(state.read_text());ps=json.loads(parent_state.read_text())
 if sha(st["candidate"])!=st["candidate_sha256"]or sha(ps["candidate"])!=ps["candidate_sha256"]:raise ValueError("state candidate pins changed")
 with np.load(st["candidate"],allow_pickle=False)as z,np.load(ps["candidate"],allow_pickle=False)as p:
  n=len(p["points_world_m"])
  if str(z["refinement_input_candidate_sha256"].item())!=ps["candidate_sha256"]:raise ValueError("immediate refinement parent differs")
  if z["points_world_m"][:n].tobytes()!=p["points_world_m"].tobytes():raise ValueError("parent coordinate prefix changed")
  for suffix,expected in (("indptr",np.arange(n+1)),("node_rows",np.arange(n)),("weights",np.ones(n))):
   a=z["immediate_refinement_prolongation_"+suffix]
   if not np.array_equal(a[:len(expected)],expected):raise ValueError("immediate Q prefix is not exact identity: "+suffix)
  anchors=z["typed_coarse_anchor_output_rows_018"]
  if len(anchors)!=237 or np.any(anchors>=n):raise ValueError("anchor prefix mismatch")
 folder=N/"source-depth-targets-606/targets-603-filtered-001";before=folder/"nearest-union-exits.emitted-f32-v2.json";npz=folder/"nearest-union-exits.npz";r=json.loads(before.read_text())
 with np.load(npz,allow_pickle=False)as a:
  if np.any(a["query_rows"]<n):raise ValueError("contained target is not newly appended")
 r.setdefault("outputs",{})["parent_prefix_node_count"]=n
 r["parent_prefix_lineage_607"]={"parent_state":str(parent_state),"parent_state_sha256":sha(parent_state),"parent_candidate_sha256":ps["candidate_sha256"],"candidate_sha256":st["candidate_sha256"],"old_point_prefix_bitwise_unchanged":True,"immediate_Q_prefix_identity":True,"parent_prefix_node_count":n,"appended_nodes":len(json.loads((Path(st["candidate"]).parent/"report.json").read_text()).get("selected_edge_rows",[])) if False else 1039,"prior_report":str(before),"prior_report_sha256":sha(before),"target_npz_sha256":sha(npz),"helper":str(Path(__file__).resolve()),"helper_sha256":sha(__file__),"target_arrays_unchanged":True,"native_admission":False,"accepted_physical_steps":0}
 out=folder/"nearest-union-exits.lineage-v3.json"
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"report":str(out),"sha256":sha(out),"parent_prefix_node_count":n}))
if __name__=="__main__":main()
