#!/usr/bin/env python3
"""Compare completed 130/159 audits for the shared refined construction baseline."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638");A=S/"edge-refinement-661-threebank-001/generation-002-all-banks-001";B=Path(sys.argv[1]);out=B/"full-static-comparison-root.json"
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def pairs(p,expected_sha):
 j=json.loads(p.read_text());assert j["complete"] and j["bank_count"]==185 and j["candidate_sha256"]==expected_sha
 return {(int(b["body"]),int(b["stable_bone_id"])):{tuple(map(int,v)) for v in (b.get("result") or {}).get("triangle_pairs",[])} for b in j["banks"]}
def inside(p):
 with p.open() as f:return {(int(x["query_row"]),int(x["bank_body"]),int(x["bank_stable_id"])) for x in csv.DictReader(f) if x["classification"]=="inside"}
base_hash=sha(A/"candidate.npz");candidate_hash=sha(B/"candidate.npz");assert base_hash=="d367af1383d920645c22552569e95614797de8aea8a757b658def68bad4412d8"
with np.load(A/"candidate.npz",allow_pickle=False) as z:
 x=z["points_world_m"].copy();f=z["source_O_faces"].copy();par=z["refinement_face_parent_face_row"].copy();anc=z["typed_coarse_anchor_output_rows_018"].copy();keys=list(z.files);orig={k:z[k].copy() for k in keys if k!="points_world_m"}
with np.load(B/"candidate.npz",allow_pickle=False) as z:
 y=z["points_world_m"].copy();assert keys==z.files;assert all(orig[k].shape==z[k].shape and orig[k].dtype==z[k].dtype and orig[k].tobytes()==z[k].tobytes() for k in orig)
a=pairs(A/"audit-159/report.json",base_hash);b=pairs(B/"audit-159/report.json",candidate_hash);assert set(a)==set(b)
ap={k:{(int(par[i]),j) for i,j in v} for k,v in a.items()};bp={k:{(int(par[i]),j) for i,j in v} for k,v in b.items()};ia=inside(A/"audit-130/query/pairs.csv");ib=inside(B/"audit-130/query/pairs.csv");changed=np.flatnonzero(np.any(x.view(np.uint32)!=y.view(np.uint32),axis=1));assert not set(changed)&set(anc)
s130=json.loads((B/"audit-130/summary.json").read_text());assert s130["candidate_sha256"]==candidate_hash and s130["complete_surface_audit"] and s130["exact_PLC_surface_pairs"]==0
assert not (ib-ia);assert all(len(b[k])<=len(a[k]) and len(bp[k])<=len(ap[k]) for k in a);assert sum(map(len,bp.values()))<sum(map(len,ap.values()))
disp=np.linalg.norm((y.astype(float)-x.astype(float))*1000,axis=1);faceids=np.flatnonzero(np.any(np.isin(f,changed),axis=1));u=x.astype(float)[f[faceids]];v=y.astype(float)[f[faceids]];n0=np.cross(u[:,1]-u[:,0],u[:,2]-u[:,0]);n1=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]);m0=np.linalg.norm(n0,axis=1);m1=np.linalg.norm(n1,axis=1);assert np.all(m0>0) and np.all(m1>0);area=m1/m0;normal=np.sum(n0*n1,axis=1)/(m0*m1)
r={"status":"FULL_STATIC_AUDITS_ONLY","base_sha256":base_hash,"candidate_sha256":candidate_hash,"raw_pairs_before":sum(map(len,a.values())),"raw_pairs_after":sum(map(len,b.values())),"parent_pairs_before":sum(map(len,ap.values())),"parent_pairs_after":sum(map(len,bp.values())),"bank_changes":[{"bank":list(k),"raw_before":len(a[k]),"raw_after":len(b[k]),"parent_before":len(ap[k]),"parent_after":len(bp[k]),"removed_parent_pairs":sorted(map(list,ap[k]-bp[k])),"added_parent_pairs":sorted(map(list,bp[k]-ap[k]))} for k in a if a[k]!=b[k]],"changed_nodes":changed.tolist(),"max_displacement_mm":float(disp.max()),"changed_incident_faces":len(faceids),"min_area_ratio_to_baseline":float(area.min()),"max_area_ratio_to_baseline":float(area.max()),"minimum_normal_cosine_to_baseline":float(normal.min()),"anchors_unchanged":True,"metadata_arrays_unchanged":True,"inside_before":len(ia),"inside_after":len(ib),"inside_added":len(ib-ia),"inside_removed":len(ia-ib),"self_pairs":len(np.load(B/"audit-130/surface-self-pairs.npy")),"geometry_admitted":False,"accepted_physical_steps":0,"limits":"sampled containment is not solid containment; parent pairs are ancestor identities and raw child counts change under refinement; no physical skin/fat transfer or motion/native validation"};assert r["self_pairs"]==0
out.write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))
