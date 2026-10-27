#!/usr/bin/env python3
"""Carry the previously overwritten 349 operation history into a derived NPZ."""
from pathlib import Path
import hashlib,json,numpy as np
w=Path(__file__).resolve().parent;a=w/"combined-43-57-65smooth-109-124-001";b=w/"same-parent-bone-selector653-adapt-001";out=w/"same-parent-bone-selector653-provenance-001";out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
x=load(a/"candidate.npz");y=load(b/"candidate.npz");assert sha(a/"candidate.npz")=="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af" and sha(b/"candidate.npz")=="b6cd9ff372f76ee574fb5302f59a70b342ba67aecbc452cd3bc6af7a509f506d"
keys=("same_parent_flip_349_face_rows","same_parent_flip_349_pair_face_rows","same_parent_flip_349_parent_rows","same_parent_flip_349_semantics")
for k in keys:
 name="prior_selector653_"+k;assert name not in y;y[name]=x[k].copy()
y["prior_selector653_candidate_sha256"]=np.array([sha(a/"candidate.npz")]);y["selector653_ungrouped_proposal_sha256"]=np.array([sha(b/"candidate.npz")]);p=out/"candidate.npz"
with p.open("xb") as f:np.savez_compressed(f,**y)
with np.load(p,allow_pickle=False) as z:
 for k in keys:assert z["prior_selector653_"+k].tobytes()==x[k].tobytes()
 for k in ("points_world_m","source_O_faces","refinement_face_parent_face_row","refinement_face_parent_barycentric","face_source_ancestry_indptr","face_source_ancestry_face_rows"):assert z[k].tobytes()==y[k].tobytes()
receipt={"baseline_sha256":sha(a/"candidate.npz"),"proposal_sha256":sha(b/"candidate.npz"),"candidate_sha256":sha(p),"prior_operation_count":int(len(x["same_parent_flip_349_pair_face_rows"])),"new_operation_count":int(len(y["same_parent_flip_349_pair_face_rows"])),"preserved_arrays":list(keys),"geometry_byte_identical_to_proposal":True,"geometry_admitted":False,"accepted_physical_steps":0};(out/"provenance.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt,indent=2))
