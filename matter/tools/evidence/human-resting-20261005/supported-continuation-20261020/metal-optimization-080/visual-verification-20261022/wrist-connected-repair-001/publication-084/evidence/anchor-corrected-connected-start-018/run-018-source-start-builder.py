#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from resolve_typed_coarse_anchors import resolve_candidate

D=Path("/Users/n/Work/numi-human-connected-surface-069")
ROOT=D/"full-connected-clearance-150"
REF=ROOT/"run-017-conforming-crossing-refinement/run-003/refined-candidate.npz"
START=ROOT/"run-017-point-constrained-soft-continuation/candidate.npz"
SOURCE=ROOT/"run-009-certified-source-normal-union-rays/endpoint-candidate.npz"
HERE=ROOT/"anchor-identity-018"
HELP=HERE/"resolve_typed_coarse_anchors.py"
OUT=ROOT/"run-018-generated-node-exit-targets-002"
CAND=OUT/"anchor-corrected-refined-start.npz"
REPORT=OUT/"anchor-corrected-start-report.json"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    z=np.load(REF,allow_pickle=False); start=np.load(START,allow_pickle=False); source=np.load(SOURCE,allow_pickle=False)
    nold=len(start["points_world_m"]); nout=len(z["points_world_m"])
    if nout != 10145 or nold != 8776: raise RuntimeError(f"unexpected domains {nout}/{nold}")
    ptr=np.asarray(z["source_prolongation_indptr"],np.int64)
    cols=np.asarray(z["source_prolongation_node_rows"],np.int64)
    vals=np.asarray(z["source_prolongation_weights"],np.float64)
    if ptr.shape!=(nout+1,) or ptr[-1]!=len(cols) or len(cols)!=len(vals): raise RuntimeError("bad source prolongation CSR")
    if not np.all(np.isfinite(vals)) or np.any(cols<0) or np.any(cols>=nold): raise RuntimeError("invalid source prolongation entries")
    W=csr_matrix((vals,cols,ptr),shape=(nout,nold))
    anchor_map=resolve_candidate(z); anchor_ids=anchor_map["source_ids"]; anchor_rows=anchor_map["output_rows"]
    old_current=np.asarray(start["points_world_m"],np.float32).copy()
    source_points=np.asarray(source["points_world_m"],np.float32)
    if source_points.shape!=(nold,3): raise RuntimeError("source point shape mismatch")
    # Keep the accepted 017 nonanchor state; restore only uniquely typed boundary anchors.
    old_current[anchor_rows]=source_points[anchor_rows]
    mapped=np.asarray(W @ old_current.astype(np.float64),np.float32)
    if not np.array_equal(mapped[:nold].view(np.uint32),old_current.view(np.uint32)): raise RuntimeError("source prolongation changed old prefix")
    if not np.array_equal(mapped[anchor_rows].view(np.uint32),source_points[anchor_rows].view(np.uint32)): raise RuntimeError("typed anchors do not exactly match 702")
    data={k:z[k] for k in z.files}
    data["points_world_m"]=mapped
    data["typed_coarse_anchor_output_rows_018"]=anchor_rows.astype(np.int32)
    data["typed_coarse_anchor_source_ids_018"]=anchor_ids.astype(np.int64)
    with CAND.open("xb") as f: np.savez_compressed(f,**data)
    changed_prefix=np.any(old_current.view(np.uint32)!=np.asarray(start["points_world_m"],np.float32).view(np.uint32),axis=1)
    changed_generated=np.any(mapped[nold:].view(np.uint32)!=np.asarray(z["points_world_m"][nold:],np.float32).view(np.uint32),axis=1)
    report={
      "schema":"numi.human.anchor-corrected-refined-start-018.v1",
      "classification":"construction_initialization_requires_fresh_surface_and_bank_gates",
      "inputs":{"refined_candidate_sha256":sha(REF),"soft_candidate_017_sha256":sha(START),"source_702_sha256":sha(SOURCE),"typed_anchor_resolver_sha256":sha(HELP)},
      "candidate":{"path":str(CAND),"sha256":sha(CAND),"nodes":nout,"faces":int(len(z["source_O_faces"]))},
      "typed_anchors":{"source_ids_count":int(len(anchor_ids)),"resolved_output_rows_count":int(len(anchor_rows)),"wrong_numeric_id_intersection":int(len(set(map(int,anchor_ids))&set(map(int,anchor_rows)))),"all_restored_to_702_f32_bits":True},
      "start_update":{"restored_old_anchor_rows":int(changed_prefix.sum()),"changed_generated_rows_after_source_prolongation":int(changed_generated.sum()),"nonanchor_old_prefix_rows_bitwise_unchanged":int(nold-int(changed_prefix.sum())),"generated_rows":nout-nold,"prolongation_nnz":int(len(vals)),"max_abs_prolongation_row_sum_error":float(np.max(np.abs(np.asarray(W.sum(axis=1)).ravel()-1)))},
      "warning":"The run-003 exact point/self audits apply to the pre-correction candidate and do not transfer. Run fresh complete gates on this candidate before using any query classes."
    }
    REPORT.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
