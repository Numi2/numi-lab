"""Restore only exact new contained-point witnesses to the passing baseline.
This is a construction target, not an LP-feasible claim or geometry acceptance.
"""
from pathlib import Path
import json,hashlib
import numpy as np
N=Path(__file__).resolve().parent.parent
O=Path(__file__).resolve().parent/"target-001";O.mkdir(exist_ok=False)
S=N/"matched-connected-fit-575/run-001/retained-regions/final/state.json"
T=N/"matched-connected-fit-575/run-001/local-retention/candidate.npz"
H=N/"matched-connected-fit-575/run-001/local-retention/full-check/audit-130/summary.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads(S.read_text());h=json.loads(H.read_text())
if sha(s["candidate"])!=s["candidate_sha256"]or h["candidate_sha256"]!=sha(T):raise ValueError("mismatched evidence")
with np.load(s["candidate"])as z:b={k:np.array(z[k],copy=True)for k in z.files}
with np.load(T)as z:t={k:np.array(z[k],copy=True)for k in z.files}
if set(b)!=set(t):raise ValueError("payload mismatch")
for k in b:
 if k=="points_world_m":continue
 if b[k].shape!=t[k].shape or b[k].dtype!=t[k].dtype or b[k].tobytes()!=t[k].tobytes():raise ValueError("non-coordinate mismatch "+k)
rows=sorted({int(w["output_node"])for w in h["inside_witnesses"]})
if len(rows)!=int(h["inside_nodes"]):raise ValueError("incomplete inside witness set")
if set(rows)&set(map(int,b["typed_coarse_anchor_output_rows_018"])):raise ValueError("witness is anchored")
b["points_world_m"]=t["points_world_m"].copy()
with np.load(s["candidate"])as z:b["points_world_m"][rows]=z["points_world_m"][rows]
np.savez_compressed(O/"candidate.npz",**b)
(O/"report.json").write_text(json.dumps({"status":"WITNESS_RESTORED_CONSTRUCTION_TARGET_PENDING_CHECKS","baseline_state":str(S),"baseline_state_sha256":sha(S),"target_input":str(T),"target_input_sha256":sha(T),"witness_summary":str(H),"witness_summary_sha256":sha(H),"restored_rows":rows,"output_candidate":str(O/"candidate.npz"),"output_candidate_sha256":sha(O/"candidate.npz"),"all_other_target_coordinates_preserved":True,"all_noncoordinate_fields_from_passing_baseline":True,"linear_feasibility_claim":False,"native_admission":False,"accepted_physical_steps":0},indent=2,sort_keys=True)+"\n")
print(json.dumps({"target":str(O/"candidate.npz"),"restored_rows":rows}),flush=True)

