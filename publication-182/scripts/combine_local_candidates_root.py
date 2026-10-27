#!/usr/bin/env python3
"""Combine disjoint coordinate-only construction proposals; independent gates required."""
from pathlib import Path
import hashlib,json,sys,numpy as np
S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638");A=S/"edge-refinement-661-threebank-001/generation-002-all-banks-001/candidate.npz";out=Path(sys.argv[1]);inputs=[Path(v) for v in sys.argv[2:]];assert len(inputs)>=2;out.mkdir(exist_ok=False)
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
assert sha(A)=="d367af1383d920645c22552569e95614797de8aea8a757b658def68bad4412d8";a=load(A);x=a["points_world_m"];f=a["source_O_faces"];y=x.copy();all_nodes=set();all_faces=set();items=[];anchor=set(map(int,a["typed_coarse_anchor_output_rows_018"]))
for p in inputs:
 q=load(p);assert q.keys()==a.keys();assert all(a[k].shape==q[k].shape and a[k].dtype==q[k].dtype and a[k].tobytes()==q[k].tobytes() for k in a if k!="points_world_m")
 changed=np.flatnonzero(np.any(q["points_world_m"].view(np.uint32)!=x.view(np.uint32),axis=1));assert len(changed);ns=set(map(int,changed));assert not (ns&all_nodes) and not (ns&anchor);faces=set(map(int,np.flatnonzero(np.any(np.isin(f,changed),axis=1))));assert not (faces&all_faces)
 y[changed]=q["points_world_m"][changed];all_nodes.update(ns);all_faces.update(faces);items.append({"path":str(p),"sha256":sha(p),"changed_nodes":changed.tolist(),"incident_face_count":len(faces)})
a["points_world_m"]=y
with (out/"candidate.npz").open("xb") as fp:np.savez_compressed(fp,**a)
r={"status":"COMBINED_CONSTRUCTION_ONLY","candidate_sha256":sha(out/"candidate.npz"),"baseline_sha256":sha(A),"components":items,"shared_changed_nodes":0,"shared_incident_faces":0,"anchor_changes":0,"physical_steps":0,"admitted":False,"limitations":"Full combined exact surface and bone checks required. No physical transfer or native qualification."}
(out/"combination.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))
