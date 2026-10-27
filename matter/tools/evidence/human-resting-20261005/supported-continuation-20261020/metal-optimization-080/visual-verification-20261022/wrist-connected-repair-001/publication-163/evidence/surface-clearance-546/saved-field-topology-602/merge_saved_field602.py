"""Replay an exactly matched saved coordinate proposal on changed connectivity.
No constraint solve is reused; fresh complete geometry checks remain mandatory.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def read_state(p):
 s=json.loads(p.read_text())
 for k in ("candidate","point_summary","surface_report"):
  if sha(s[k])!=s[k+"_sha256"]:raise ValueError("state pin mismatch: "+k)
 return s

def merge(base,old,proposal):
 if set(base)!=set(old) or set(base)!=set(proposal):raise ValueError("payload inventory mismatch")
 if not same(base["points_world_m"],old["points_world_m"]):raise ValueError("baseline coordinate field differs")
 if not same(old["source_O_faces"],proposal["source_O_faces"]):raise ValueError("saved proposal topology differs")
 if base["points_world_m"].dtype!=np.float32:raise ValueError("Float32 emission required")
 for k in old:
  if k!="points_world_m" and not same(old[k],proposal[k]):raise ValueError("saved proposal changed nonpoint field: "+k)
 # Only geometrical topology and its face-dependent ancestry may differ.
 allowed={"source_O_faces","refinement_face_parent_face_row","refinement_face_parent_barycentric",
          "area_reference_037_source_O_faces","refinement_face_parent_ancestry_indptr",
          "refinement_face_parent_ancestry_face_rows"}
 for k in base:
  if k not in allowed and not same(base[k],old[k]):raise ValueError("point domain/provenance differs: "+k)
 anchors=base["typed_coarse_anchor_output_rows_018"]
 if len(anchors)!=237 or not same(proposal["points_world_m"][anchors],old["points_world_m"][anchors]):
  raise ValueError("anchor mismatch")
 result={k:np.array(v,copy=True)for k,v in base.items()}
 result["points_world_m"]=proposal["points_world_m"].copy()
 for k in base:
  if k!="points_world_m" and not same(result[k],base[k]):raise AssertionError("unrelated field mutated")
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument("--baseline-state",type=Path,required=True);p.add_argument("--saved-baseline-state",type=Path,required=True);p.add_argument("--proposal-state",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 ss=[read_state(q)for q in (a.baseline_state,a.saved_baseline_state,a.proposal_state)]
 paths=[a.baseline_state,a.saved_baseline_state,a.proposal_state,Path(__file__)]
 paths += [Path(s[k])for s in ss for k in ("candidate","point_summary","surface_report")]
 pins={str(p):sha(p)for p in paths}
 payloads=[]
 for s in ss:
  with np.load(s["candidate"],allow_pickle=False)as z:payloads.append({k:z[k].copy()for k in z.files})
 result=merge(*payloads)
 a.out.mkdir(parents=True,exist_ok=False);target=a.out/"candidate.npz";np.savez_compressed(target,**result)
 for p,h in pins.items():
  if sha(p)!=h:raise ValueError("input changed during replay")
 changed=np.any(result["points_world_m"]!=payloads[0]["points_world_m"],axis=1)
 report={"status":"SAVED_COORDINATE_FIELD_REPLAY_PENDING_FRESH_GEOMETRY_CHECKS",
 "source_pins":pins,"candidate":str(target),"candidate_sha256":sha(target),
 "changed_nodes":int(changed.sum()),"maximum_step_mm":float(np.linalg.norm(result["points_world_m"].astype(float)-payloads[0]["points_world_m"],axis=1).max()*1000),
 "all_nonpoint_fields_bitwise_preserved":True,"source_anchor_count":237,
 "constraints_reused":False,"native_admission":False,"accepted_physical_steps":0}
 (a.out/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
 print(json.dumps(report))
if __name__=="__main__":main()
