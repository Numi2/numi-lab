#!/usr/bin/env python3
"""Rebuild six pre-screened same-parent flips as an offline candidate only."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
W=S/"per-bank-workers-20261027"
base=W/"b65-99/root-layer-guard-002/backtrack-0p25/candidate.npz"
common=S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz"
proposal=S/"partial-midpoint-continuation-root-001/same-parent-bone-selector653-provenance-001/candidate.npz"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(base)=="a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db"
assert sha(common)=="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af"
assert sha(proposal)=="308007c045ca2bda7d42a6267aa6558cac0f74affbd42e015e7f13e537db0083"
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
a,x,b=map(load,(base,common,proposal))
def eq(x,y):return x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()
for k in ("source_O_faces","refinement_face_parent_barycentric","area_reference_037_source_O_faces","same_parent_flip_349_face_rows","same_parent_flip_349_pair_face_rows","same_parent_flip_349_parent_rows"):
 assert eq(a[k],x[k]),k
assert eq(x["points_world_m"],b["points_world_m"])
assert eq(a["face_source_ancestry_indptr"],b["face_source_ancestry_indptr"])
assert eq(a["face_source_ancestry_face_rows"],b["face_source_ancestry_face_rows"])
allowed={(34722,34724),(19314,19316),(19203,19206),(34534,34536),(35437,35440),(12185,12187)}
pairs=b["same_parent_flip_349_pair_face_rows"]
mask=np.array([tuple(map(int,p)) in allowed for p in pairs]);chosen=pairs[mask];rows=chosen.ravel()
assert len(chosen)==6 and {tuple(map(int,p)) for p in chosen}==allowed
for k in ("source_O_faces","refinement_face_parent_barycentric","area_reference_037_source_O_faces"):a[k][rows]=b[k][rows]
for k in ("prior_selector653_candidate_sha256","prior_selector653_same_parent_flip_349_face_rows","prior_selector653_same_parent_flip_349_pair_face_rows","prior_selector653_same_parent_flip_349_parent_rows","prior_selector653_same_parent_flip_349_semantics","selector653_ungrouped_proposal_sha256"):
 assert k not in a;a[k]=b[k].copy()
a["same_parent_flip_349_face_rows"]=rows.copy();a["same_parent_flip_349_pair_face_rows"]=chosen.copy();a["same_parent_flip_349_parent_rows"]=b["same_parent_flip_349_parent_rows"][mask].copy();a["same_parent_flip_349_semantics"]=b["same_parent_flip_349_semantics"].copy()
p=Path(sys.argv[1]);assert not p.exists()
with p.open("xb") as f:np.savez_compressed(f,**a)
print(json.dumps({"output":str(p),"sha256":sha(p),"physical_admission":False,"physical_steps":0}))
