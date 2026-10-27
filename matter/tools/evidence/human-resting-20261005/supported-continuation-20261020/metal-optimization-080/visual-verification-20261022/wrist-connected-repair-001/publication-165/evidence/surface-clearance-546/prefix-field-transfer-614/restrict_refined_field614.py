"""Restrict a checked refined field to its proven unchanged parent coordinate prefix.
This is a construction proposal; fresh exact geometry/physical admission is mandatory.
"""
import argparse,hashlib,json,importlib.util,sys
from pathlib import Path
import numpy as np
N=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy()for k in z.files}
def restrict(parent,refined,proposal,parent_sha):
 P=parent["points_world_m"];R=refined["points_world_m"];T=proposal["points_world_m"];n=len(P)
 if P.dtype!=np.float32 or R.dtype!=np.float32 or T.dtype!=np.float32:raise ValueError("Float32 required")
 if not np.isfinite(T).all() or R.shape!=T.shape or len(R)<=n:raise ValueError("invalid refined coordinate domain")
 if str(refined["refinement_input_candidate_sha256"].item())!=parent_sha:raise ValueError("refinement parent pin mismatch")
 if not same(R[:n],P):raise ValueError("refined baseline coordinate prefix differs")
 for name,expected in (("indptr",np.arange(n+1)),("node_rows",np.arange(n)),("weights",np.ones(n))):
  actual=refined["immediate_refinement_prolongation_"+name]
  if not np.array_equal(actual[:len(expected)],expected):raise ValueError("prefix Q is not identity: "+name)
 anchors=np.asarray(parent["typed_coarse_anchor_output_rows_018"],np.int64)
 if len(anchors)!=237 or not np.array_equal(anchors,refined["typed_coarse_anchor_output_rows_018"]) or not same(P[anchors],T[anchors]):raise ValueError("typed anchors changed")
 for k in ("area_reference_037_points_world_m_f64",):
  if not same(parent[k],refined[k][:n]):raise ValueError("fixed reference prefix differs")
 result={k:v.copy()for k,v in parent.items()};result["points_world_m"]=T[:n].copy()
 return result
def main():
 ap=argparse.ArgumentParser()
 for name in ("parent-state","refined-baseline-state","proposal-state","out"):ap.add_argument("--"+name,type=Path,required=True)
 a=ap.parse_args();stages=N/"continue_surface_round546.py"
 spec=importlib.util.spec_from_file_location("transfer614_stages",stages);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 states=[m.read_json(p)for p in (a.parent_state,a.refined_baseline_state,a.proposal_state)]
 pins={str(p):sha(p)for p in (Path(__file__),stages,a.parent_state,a.refined_baseline_state,a.proposal_state)}
 for s in states:
  for k in ("candidate","point_summary","surface_report"):
   if sha(s[k])!=s[k+"_sha256"]:raise ValueError("state pin "+k)
   pins[str(s[k])]=s[k+"_sha256"]
 hpath=N/"matched-connected-fit-583/run_matched_fit583_v2.py";spec=importlib.util.spec_from_file_location("target614_validate",hpath);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 pins[str(hpath)]=sha(hpath)
 h._validate_target_payload(Path(states[1]["candidate"]),Path(states[2]["candidate"]),allow_fit_receipt_updates=True)
 ps=sha(states[0]["candidate"]);p,r,t=[arrays(s["candidate"])for s in states]
 out=restrict(p,r,t,ps)
 a.out.mkdir(parents=True,exist_ok=False);candidate=a.out/"candidate.npz";np.savez_compressed(candidate,**out)
 for path,pin in pins.items():
  if sha(path)!=pin:raise ValueError("source changed "+path)
 changed=np.any(out["points_world_m"]!=p["points_world_m"],axis=1)
 report={"status":"REFINED_FIELD_PREFIX_RESTRICTION_PENDING_FRESH_CHECKS","candidate":str(candidate),"candidate_sha256":sha(candidate),"parent_prefix_node_count":len(p["points_world_m"]),"changed_nodes":int(changed.sum()),"maximum_step_mm":float(np.linalg.norm(out["points_world_m"].astype(float)-p["points_world_m"].astype(float),axis=1).max()*1000),"source_pins":pins,"all_parent_nonpoint_fields_bitwise_preserved":all(same(out[k],p[k])for k in p if k!="points_world_m"),"prefix_identity_proven":True,"native_admission":False,"accepted_physical_steps":0}
 m.write_json(a.out/"report.json",report);print(json.dumps(report))
if __name__=="__main__":main()
