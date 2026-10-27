"""Attribute the removed per-bank regression with the frozen 631-v2 component map."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HELPER=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-merit-retention-631/select_components631_v2.py")
sys.path.insert(0,str(HELPER.parent))
from select_components631_v2 import bank_pairs,component_deltas,labels_for_changes,loadnpz,readj,sha

def main():
 p=argparse.ArgumentParser()
 for k in ("baseline-state","regression-check","regression-surface","minimal-check","minimal-surface","receipt"):
  p.add_argument("--"+k,required=True,type=Path)
 a=p.parse_args();s=readj(a.baseline_state);base=loadnpz(s["candidate"])
 rg=readj(a.regression_check);mi=readj(a.minimal_check)
 if any(x.get("status")!="COMPLETE_EXACT_CONSTRUCTION_CHECKS" for x in (rg,mi)):raise ValueError("a full exact check is not passing")
 rnpz=loadnpz(rg["candidate_path"]);mnpz=loadnpz(mi["candidate_path"])
 labels,owners=labels_for_changes(base["points_world_m"],rnpz["points_world_m"],base["source_O_faces"])
 bp=bank_pairs(readj(s["surface_report"]),s["candidate_sha256"],len(base["source_O_faces"]))
 rp=bank_pairs(readj(a.regression_surface),rg["candidate_sha256"],len(base["source_O_faces"]),bp)
 mp=bank_pairs(readj(a.minimal_surface),mi["candidate_sha256"],len(base["source_O_faces"]),bp)
 deltas,_=component_deltas(owners,bp,rp)
 nonzero={str(int(k)):{f"{b[0]}/{b[1]}":int(v)for b,v in ds.items()if v}for k,ds in deltas.items()if any(ds.values())}
 pair_diffs={f"{k[0]}/{k[1]}":{"regression_only":len(rp[k]-mp[k]),"minimal_only":len(mp[k]-rp[k])}for k in bp if rp[k]!=mp[k]}
 sizes={str(int(k)):int(np.count_nonzero(labels==k))for k in np.unique(labels)if k>=0}
 if nonzero!={"5883":{"43/16":-12},"7273":{"114/117":1}}:raise ValueError(f"unexpected attribution {nonzero}")
 report={"status":"EXACT_REGRESSION_COMPONENT_ATTRIBUTION","baseline_state":str(a.baseline_state),"baseline_state_sha256":sha(a.baseline_state),"baseline_candidate_sha256":s["candidate_sha256"],"regression_candidate":rg["candidate_path"],"regression_candidate_sha256":rg["candidate_sha256"],"regression_full_check_sha256":sha(a.regression_check),"regression_surface_report_sha256":sha(a.regression_surface),"minimal_candidate":mi["candidate_path"],"minimal_candidate_sha256":mi["candidate_sha256"],"minimal_full_check_sha256":sha(a.minimal_check),"minimal_surface_report_sha256":sha(a.minimal_surface),"selector631_v2":str(HELPER),"selector631_v2_sha256":sha(HELPER),"changed_component_node_counts":sizes,"nonzero_component_bank_deltas":nonzero,"regression_surface_pairs":rg["surface_pairs"],"minimal_surface_pairs":mi["surface_pairs"],"per_bank_pair_identity_differences_regression_to_minimal":pair_diffs,"native_admission":False,"accepted_physical_steps":0,"limitation":"Exact component attribution and endpoint pair identity comparison only; both candidates remain construction-only."}
 a.receipt.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({k:report[k]for k in("status","nonzero_component_bank_deltas","regression_surface_pairs","minimal_surface_pairs","per_bank_pair_identity_differences_regression_to_minimal")}))
if __name__=="__main__":main()
