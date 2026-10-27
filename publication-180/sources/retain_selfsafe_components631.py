"""Reconstruct/verify 631-v2-style self-witness component retention from exact pinned inputs."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
HELPER = Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-merit-retention-631/select_components631_v2.py")
sys.path.insert(0, str(HELPER.parent))
from select_components631_v2 import labels_for_changes, same

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p, allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def main():
    a=argparse.ArgumentParser()
    for k in ("baseline","trial","self-summary","self-pairs","expected-candidate","receipt"):
        a.add_argument("--"+k, required=True, type=Path)
    x=a.parse_args()
    b,t=load(x.baseline),load(x.trial)
    if set(b)!=set(t): raise ValueError("candidate key inventory changed")
    s=json.loads(x.self_summary.read_text()); ph=sha(x.trial)
    if s.get("candidate_sha256")!=ph or s.get("complete_surface_audit") is not True or s.get("status")!="STOPPED_AT_SURFACE_GATE":
        raise ValueError("exact self summary is not complete and pinned to trial")
    pairs=np.load(x.self_pairs,allow_pickle=False)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or len(pairs)!=s.get("exact_PLC_surface_pairs"):
        raise ValueError("exact self-pair inventory does not match summary")
    f=b["source_O_faces"]
    if not same(f,t["source_O_faces"]): raise ValueError("source topology changed")
    anchors=b["typed_coarse_anchor_output_rows_018"]
    if len(anchors)!=237 or not same(b["points_world_m"][anchors],t["points_world_m"][anchors]):
        raise ValueError("typed anchor contract failed")
    labels,owners=labels_for_changes(b["points_world_m"],t["points_world_m"],f)
    implicated=set(); pair_components=[]
    for i,j in pairs.astype(np.int64):
        comps=sorted({int(owners[i]),int(owners[j])}-{ -1 })
        if not comps: raise ValueError("self witness touches no changed component")
        implicated.update(comps); pair_components.append({"faces":[int(i),int(j)],"components":comps})
    all_components=sorted(int(k) for k in np.unique(labels) if k>=0)
    kept=sorted(set(all_components)-implicated)
    rows=np.flatnonzero(np.isin(labels, kept))
    expected=load(x.expected_candidate)
    rebuilt={k:v.copy() for k,v in b.items()}; rebuilt["points_world_m"][rows]=t["points_world_m"][rows]
    if set(expected)!=set(rebuilt) or any(not same(expected[k],rebuilt[k]) for k in rebuilt):
        raise ValueError("retained candidate arrays differ from component reconstruction")
    report={"status":"RECONSTRUCTION_VERIFIED","baseline":str(x.baseline),"baseline_sha256":sha(x.baseline),"trial":str(x.trial),"trial_sha256":ph,"self_summary":str(x.self_summary),"self_summary_sha256":sha(x.self_summary),"self_pairs":str(x.self_pairs),"self_pairs_sha256":sha(x.self_pairs),"selector631_v2":str(HELPER),"selector631_v2_sha256":sha(HELPER),"expected_candidate":str(x.expected_candidate),"expected_candidate_sha256":sha(x.expected_candidate),"changed_component_count":len(all_components),"implicated_component_ids":sorted(implicated),"retained_component_ids":kept,"retained_rows":int(len(rows)),"excluded_rows":int(np.count_nonzero(np.isin(labels,list(implicated)))),"anchors_bitwise_baseline":True,"rebuilt_arrays_bitwise_match_expected":True,"pair_component_map":pair_components,"native_admission":False,"accepted_physical_steps":0}
    x.receipt.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ("status","changed_component_count","implicated_component_ids","retained_component_ids","retained_rows","expected_candidate_sha256")}))
if __name__=="__main__":main()
