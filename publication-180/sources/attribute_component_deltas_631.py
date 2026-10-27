"""Recompute 631-v2 component-wise exact surface-pair deltas and pair-set parity."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
HELPER=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-merit-retention-631/select_components631_v2.py")
sys.path.insert(0,str(HELPER.parent))
from select_components631_v2 import bank_pairs,component_deltas,labels_for_changes,loadnpz,readj,sha

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--baseline-state",required=True,type=Path)
 p.add_argument("--candidate-a-check",required=True,type=Path)
 p.add_argument("--candidate-a-surface",required=True,type=Path)
 p.add_argument("--candidate-b-check",required=True,type=Path)
 p.add_argument("--candidate-b-surface",required=True,type=Path)
 p.add_argument("--receipt",required=True,type=Path)
 a=p.parse_args();s=readj(a.baseline_state);base=loadnpz(s["candidate"])
 ca,cb=readj(a.candidate_a_check),readj(a.candidate_b_check)
 if any(r.get("status")!="COMPLETE_EXACT_CONSTRUCTION_CHECKS" or r.get("self_pairs")!=0 or r.get("contained_pairs")!=0 for r in (ca,cb)):
  raise ValueError("candidate checks are not exact passing 545 reports")
 if ca.get("candidate_sha256")!=sha(ca["candidate_path"]) or cb.get("candidate_sha256")!=sha(cb["candidate_path"]):
  raise ValueError("candidate report hash mismatch")
 xa=loadnpz(ca["candidate_path"]); xb=loadnpz(cb["candidate_path"])
 labels,owners=labels_for_changes(base["points_world_m"],xa["points_world_m"],base["source_O_faces"])
 bp=bank_pairs(readj(s["surface_report"]),s["candidate_sha256"],len(base["source_O_faces"]))
 ap=bank_pairs(readj(a.candidate_a_surface),ca["candidate_sha256"],len(base["source_O_faces"]),bp)
 bd,keep=component_deltas(owners,bp,ap)
 bpoints=bank_pairs(readj(a.candidate_b_surface),cb["candidate_sha256"],len(base["source_O_faces"]),bp)
 pair_diff={f"{k[0]}/{k[1]}":{"only_a":len(ap[k]-bpoints[k]),"only_b":len(bpoints[k]-ap[k])} for k in bp if ap[k]!=bpoints[k]}
 xlabels,_=labels_for_changes(base["points_world_m"],xb["points_world_m"],base["source_O_faces"])
 changed_counts={str(int(k)):int(np.count_nonzero(labels==k)) for k in np.unique(labels) if k>=0}
 delta_map={str(int(k)):{f"{b[0]}/{b[1]}":int(v) for b,v in ds.items() if v} for k,ds in bd.items() if any(ds.values())}
 if pair_diff: raise ValueError("exact pair identities differ")
 out={"status":"EXACT_COMPONENT_ATTRIBUTION_AND_PAIR_PARITY_VERIFIED","baseline_state":str(a.baseline_state),"baseline_state_sha256":sha(a.baseline_state),"baseline_candidate_sha256":s["candidate_sha256"],"candidate_a":ca["candidate_path"],"candidate_a_sha256":ca["candidate_sha256"],"candidate_a_full_check_sha256":sha(a.candidate_a_check),"candidate_a_surface_report_sha256":sha(a.candidate_a_surface),"candidate_b":cb["candidate_path"],"candidate_b_sha256":cb["candidate_sha256"],"candidate_b_full_check_sha256":sha(a.candidate_b_check),"candidate_b_surface_report_sha256":sha(a.candidate_b_surface),"selector631_v2":str(HELPER),"selector631_v2_sha256":sha(HELPER),"attribution_candidate":"A","changed_components":changed_counts,"nonzero_component_bank_deltas":delta_map,"expected_component_5883_nodes":changed_counts.get("5883"),"candidate_b_changed_node_count":int(np.count_nonzero(xlabels>=0)),"surface_pairs_a":int(ca["surface_pairs"]),"surface_pairs_b":int(cb["surface_pairs"]),"exact_pair_identity_equal":True,"per_bank_pair_identity_differences":pair_diff,"native_admission":False,"accepted_physical_steps":0,"limitation":"This attributes the audited endpoint deltas; it does not imply physical or native acceptance."}
 a.receipt.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({k:out[k] for k in ("status","nonzero_component_bank_deltas","expected_component_5883_nodes","candidate_b_changed_node_count","surface_pairs_a","surface_pairs_b","exact_pair_identity_equal")}))
if __name__=="__main__":main()
