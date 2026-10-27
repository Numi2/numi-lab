"""Compose named changed mesh components onto a pinned valid baseline."""
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
    p=argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--keep-component", type=int, action="append", required=True)
    a=p.parse_args(); b=load(a.baseline); t=load(a.source)
    if set(b)!=set(t) or any(not same(b[k],t[k]) for k in b if k!="points_world_m"):
        raise ValueError("source nonpoint payload differs from baseline")
    labels,_=labels_for_changes(b["points_world_m"],t["points_world_m"],b["source_O_faces"])
    present=sorted(int(k) for k in np.unique(labels) if k>=0); keep=sorted(set(a.keep_component))
    if not keep or not set(keep)<=set(present): raise ValueError("requested component missing")
    rows=np.flatnonzero(np.isin(labels,keep)); out={k:v.copy() for k,v in b.items()}; out["points_world_m"][rows]=t["points_world_m"][rows]
    anchors=b["typed_coarse_anchor_output_rows_018"]
    if len(anchors)!=237 or not same(out["points_world_m"][anchors],b["points_world_m"][anchors]): raise ValueError("typed anchors changed")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists(): raise FileExistsError(a.out)
    np.savez_compressed(a.out,**out)
    rec={"status":"COMPONENT_COMPOSITION_PROPOSAL","baseline":str(a.baseline),"baseline_sha256":sha(a.baseline),"source_candidate":str(a.source),"source_candidate_sha256":sha(a.source),"selector631_v2":str(HELPER),"selector631_v2_sha256":sha(HELPER),"present_component_ids":present,"kept_component_ids":keep,"kept_node_count":int(len(rows)),"kept_node_rows":rows.tolist(),"candidate":str(a.out),"candidate_sha256":sha(a.out),"all_nonpoint_arrays_bitwise_baseline":True,"typed_anchors_bitwise_baseline":True,"candidate_selected":False,"native_admission":False,"accepted_physical_steps":0,"limitation":"Fresh exact 545 is required; component endpoint deltas do not certify cross-component self or containment."}
    a.report.write_text(json.dumps(rec,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:rec[k] for k in ("status","kept_component_ids","kept_node_count","candidate_sha256")}))
if __name__=="__main__": main()
