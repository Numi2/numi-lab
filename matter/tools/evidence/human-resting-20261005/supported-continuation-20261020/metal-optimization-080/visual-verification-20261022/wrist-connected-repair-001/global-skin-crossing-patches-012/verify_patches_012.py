#!/usr/bin/env python3
"""Read-only verifier for exact physical-O crossing patch extraction 012."""
from pathlib import Path
import hashlib, json
import numpy as np
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=R/"global-skin-crossing-patches-012"
O=R/"global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz"
P=R/"global-inner-p1-full-domain-clipping-009/validation-001/zero-support-physical-O-face-pairs-010.npz"
C=R/"global-inner-p1-full-domain-clipping-009/exact-O-pair-011/run-001/canonical-pair-classification.npz"
Z=R/"global-inner-p1-full-domain-clipping-009/validation-001/zero-capacity-witness-index-009.npz"
INV=R/"registered-surrounding-tissue-native013-191/inventory.json"
SKIN=Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin")
POSE=R/"wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-004/registered-skin-poses.json"
HELPER=R/"registered-pose-frame-helper-194/tools/rebind_human_skin_reference.py"
NPZ=D/"connected-physical-O-patches-012.npz"; REPORT=D/"patch-report-012.json"
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def check_offsets(name, offsets, n, expected_count=None):
 a=np.asarray(offsets)
 if a.ndim!=1 or len(a)<2 or a[0]!=0 or a[-1]!=n or np.any(np.diff(a)<0):
  raise ValueError(f"{name}: invalid CSR offsets")
 if expected_count is not None and len(a)!=expected_count+1:
  raise ValueError(f"{name}: expected {expected_count} entries")
def verify_arrays(o,p,c,z,inv,out,report):
 x=o["points_world_m"]; faces=o["physical_outer_faces"].astype(np.int64)
 n_patch=len(out["patch_face_offsets"])-1
 check_offsets("face",out["patch_face_offsets"],len(out["patch_face_ids"]),n_patch)
 check_offsets("node",out["patch_node_offsets"],len(out["patch_node_ids"]),n_patch)
 check_offsets("seed",out["patch_seed_face_offsets"],len(out["patch_seed_face_ids"]),n_patch)
 check_offsets("boundary",out["patch_boundary_edge_offsets"],len(out["patch_boundary_edge_node_ids"]),n_patch)
 check_offsets("pair",out["patch_pair_offsets"],len(out["pair_row_ids"]),n_patch)
 check_offsets("node_source",out["patch_node_source_indptr"],len(out["patch_node_source_indices"]),len(out["patch_node_ids"]))
 n_pair=len(out["pair_row_ids"]); n_hit=len(out["pair_exact_hit_points_m"])
 check_offsets("hit",out["pair_exact_hit_offsets"],n_hit,n_pair)
 if n_patch!=9:raise ValueError(f"expected 9 components, got {n_patch}")
 if not np.array_equal(np.sort(out["patch_face_ids"]),np.unique(out["patch_face_ids"])):raise ValueError("face assigned more than once")
 if np.any(out["patch_face_ids"]<0) or np.any(out["patch_face_ids"]>=len(faces)):raise ValueError("O face out of range")
 if np.any(out["patch_node_ids"]<0) or np.any(out["patch_node_ids"]>=len(x)):raise ValueError("O node out of range")
 if not np.array_equal(out["patch_face_node_ids"],faces[out["patch_face_ids"]]):raise ValueError("O face node identity changed")
 if not np.array_equal(out["patch_node_points_world_m"],x[out["patch_node_ids"]]):raise ValueError("O coordinates changed")
 if len(out["patch_face_local_node_ids"])!=len(out["patch_face_ids"]):raise ValueError("local face map length mismatch")
 for k in range(n_patch):
  fa,fb=map(int,out["patch_face_offsets"][k:k+2]);na,nb=map(int,out["patch_node_offsets"][k:k+2])
  local=out["patch_face_local_node_ids"][fa:fb]
  if local.size and (local.min()<0 or local.max()>=nb-na):raise ValueError("local face node index outside patch")
  if not np.array_equal(out["patch_node_ids"][na:nb],np.unique(out["patch_face_node_ids"][fa:fb])):raise ValueError("patch node list differs from selected faces")
  if local.size and not np.array_equal(out["patch_node_ids"][na:nb][local],out["patch_face_node_ids"][fa:fb]):raise ValueError("local to global face map mismatch")
  for localrow,node in enumerate(out["patch_node_ids"][na:nb]):
   s0,s1=map(int,out["patch_node_source_indptr"][na+localrow:na+localrow+2]);a,b=map(int,o["node_source_indptr"][node:node+2])
   if not np.array_equal(out["patch_node_source_indices"][s0:s1],o["node_source_indices"][a:b]):raise ValueError("source CSR index lineage mismatch")
   if not np.array_equal(out["patch_node_source_weights"][s0:s1],o["node_source_weights"][a:b]):raise ValueError("source CSR weight lineage mismatch")
 # exact011 and original pair seed IDs have identical ordered canonical rows
 if not np.array_equal(c["bank_ids"],p["bank_ids"]) or not np.array_equal(c["bank_local_face_ids"],p["bank_local_face_ids"]) or not np.array_equal(c["physical_O_face_ids"],p["physical_O_face_ids"]):raise ValueError("011/pair-source row mismatch")
 rows=out["pair_row_ids"].astype(np.int64)
 if len(rows)!=len(c["bank_ids"]) or not np.array_equal(np.sort(rows),np.arange(len(rows))):raise ValueError("canonical rows not assigned exactly once")
 if not np.array_equal(out["pair_bank_ids"],c["bank_ids"][rows]) or not np.array_equal(out["pair_O_face_ids"],c["physical_O_face_ids"][rows]):raise ValueError("pair bank/O face identity mismatch")
 if not np.array_equal(out["pair_class_codes"],c["class_codes"][rows]):raise ValueError("pair class mismatch")
 if not np.array_equal(out["pair_witness_tet_ids"],p["witness_tet_ids"][rows]):raise ValueError("witness tet mismatch")
 # Flattened hit support must reproduce the canonical exact hit points on unchanged O triangles.
 h_off=p["exact_hit_offsets"]; hp=p["exact_hit_points_m"]
 expected=np.concatenate([hp[int(h_off[r]):int(h_off[r+1])] for r in rows],axis=0)
 if not np.array_equal(out["pair_exact_hit_points_m"],expected):raise ValueError("exact hit points differ from source rows")
 repeats=np.concatenate([np.full(int(h_off[r+1]-h_off[r]),int(c["physical_O_face_ids"][r]),np.uint32) for r in rows])
 if not np.array_equal(out["pair_hit_O_face_ids"],repeats):raise ValueError("hit O face identity mismatch")
 if not np.array_equal(out["pair_hit_O_projection_residual_m"]>=0,np.ones(n_hit,bool)):raise ValueError("negative hit projection residual")
 tri=x[faces[out["pair_hit_O_face_ids"].astype(np.int64)]].astype(np.float64)
 rebuilt=np.einsum("ij,ijk->ik",out["pair_hit_O_barycentric"],tri)
 err=np.linalg.norm(rebuilt-expected,axis=1)
 if np.max(err,initial=0)>2e-6:raise ValueError(f"exact hit barycentric reconstruction error {np.max(err)}")
 # Typed inventory identities are row-aligned and bank ids resolve.
 banks=inv["banks"]
 for i,bi in enumerate(out["pair_bank_ids"]):
  m=banks[int(bi)]
  if int(out["pair_bank_stable_ids"][i])!=int(m["stable_id"]) or int(out["pair_bank_layers"][i])!=int(m["layer"]):raise ValueError("typed bank identity mismatch")
  a,b=map(int,out["pair_bank_body_indptr"][i:i+2])
  if out["pair_bank_body_indices"][a:b].tolist()!=m["body_indices"]:raise ValueError("typed bank body ownership mismatch")
 # 009 is partial: only confirm labels against its key-indexed partial witnesses.
 lookup={}
 for i,(bi,li) in enumerate(zip(z["key_bank_ids"],z["key_local_face_ids"])):
  a,b=map(int,z["key_witness_row_offsets"][i:i+2]);lookup[(int(bi),int(li))]=bool(np.any(z["zero_capacity_mask"][a:b]))
 expect9=np.asarray([lookup.get((int(bi),int(li)),False) for bi,li in zip(c["bank_ids"][rows],c["bank_local_face_ids"][rows])])
 if not np.array_equal(out["pair_from_original009_zero_support"],expect9):raise ValueError("009 zero-capacity cross-reference mismatch")
 return {"patches":n_patch,"selected_faces":len(out["patch_face_ids"]),"selected_nodes":len(out["patch_node_ids"]),"canonical_pairs":n_pair,"exact_hit_points":n_hit,"hit_reconstruction_max_m":float(err.max(initial=0)),"source294_candidates":int(np.sum(out["source294_mapping_method_codes"]>0)),"source294_unmatched":int(np.sum(out["source294_mapping_method_codes"]==0)),"source294_global_map_certified":False}
def run():
 for f in (O,P,C,Z,INV,SKIN,POSE,HELPER,NPZ,REPORT):
  if not f.is_file():raise FileNotFoundError(f)
 rep=json.loads(REPORT.read_text());out=np.load(NPZ,allow_pickle=False);o=np.load(O,allow_pickle=False);p=np.load(P,allow_pickle=False);c=np.load(C,allow_pickle=False);z=np.load(Z,allow_pickle=False);inv=json.loads(INV.read_text())
 summary=verify_arrays(o,p,c,z,inv,out,rep)
 inputs=[O,P,C,Z,INV,SKIN,POSE,HELPER,NPZ,REPORT]
 hashes={str(f):sha(f) for f in inputs}
 expected=rep["inputs_sha256"]
 if any(expected.get(str(f))!=hashes[str(f)] for f in inputs[:-2]):raise ValueError("input hashes differ from extraction report")
 result={"schema":"numi.human.global-skin-crossing-patches.validation.v1","status":"PASS_EXTRACTION_AND_SOURCE_CSR_LINEAGE; SOURCE294_MAP_UNCERTIFIED","native_admission":False,"physical_admission":False,"inputs_sha256":hashes,"metrics":summary,"source294_mapping_scope":"bounded nearest direct/triangle candidates only; no topology ancestry claim; 85 unmatched retained as such"}
 (D/"validation-report-012.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 return result
if __name__=="__main__":print(json.dumps(run(),indent=2,sort_keys=True))
