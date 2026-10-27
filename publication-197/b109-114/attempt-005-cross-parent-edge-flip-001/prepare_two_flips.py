#!/usr/bin/env python3
"""Apply two freshly revalidated fixed-coordinate 402-owner local flips; no geometry audit."""
from pathlib import Path
import hashlib, importlib.util, json, sys, traceback
import numpy as np
S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
W=S/"per-bank-workers-20261027/b109-114"
D=W/"attempt-005-cross-parent-edge-flip-001"
BASE=ROOT/"D/publication-182-stage-001/combined-653/candidate.npz"
OUT=D/"candidate-two-cross-parent-flips.npz"
REPORT=D/"topology-and-lineage-report.json"
PARENT=ROOT/"D/surface-clearance-546/component-sat-selector-638/fit-from-663-001/seed-retry-663-001/fraction-small-001/candidate.npz"
AUDIT159=ROOT/"D/publication-182-stage-001/combined-653/audit-159/report.json"
GEN2GUARD=S/"edge-refinement-661-threebank-001/refresh_guards_multi_parent_generation002.py"
OWNER=ROOT/"D/local-two-parent-retriangulation-402/two_parent_patch_flips_402.py"
BASE_SHA="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af"
DETAIL=90091
PATCHES=W/"attempt-001-boundary-feasibility/patch-components"
PATCH_ROWS=np.concatenate([np.load(PATCHES/f"component-{i:02d}-face-rows.npy",allow_pickle=False).astype(np.int64) for i in (0,1)])
TARGETS=[{"seed":34705,"pair":[34666,34705],"edge":[19197,21045],"new_diagonal":[21149,21150]},
         {"seed":34706,"pair":[34668,34706],"edge":[17965,21045],"new_diagonal":[21046,21047]}]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def csr_rows(z,prefix,vals_key,n):
 p=np.asarray(z[prefix],np.int64);v=np.asarray(z[vals_key],np.int64)
 if p.shape!=(n+1,) or p[0]!=0 or p[-1]!=len(v) or np.any(p[1:]<p[:-1]):raise ValueError("invalid CSR "+prefix)
 return [list(map(int,v[p[i]:p[i+1]])) for i in range(n)]
def patch_boundary(faces, rows):
 inc={}
 for fi in map(int,rows):
  t=faces[fi]
  for a,b in zip(t,np.roll(t,-1)):
   e=(min(int(a),int(b)),max(int(a),int(b)))
   inc.setdefault(e,[]).append(fi)
 if any(len(v)>2 for v in inc.values()):raise ValueError("local patch nonmanifold edge")
 return {e for e,v in inc.items() if len(v)==1}
def main():
 for p in (OUT,REPORT):
  if p.exists():raise FileExistsError(p)
 if sha(BASE)!=BASE_SHA:raise ValueError("pinned candidate hash mismatch")
 owner=loadmod("two_parent_owner_402_preparation",OWNER)
 with np.load(BASE,allow_pickle=False) as src: z={k:src[k].copy() for k in src.files}
 owner.validate_candidate(z)
 base_faces=z["source_O_faces"].copy(); base_points=z["points_world_m"].copy()
 boundary=patch_boundary(base_faces,PATCH_ROWS)
 src0=csr_rows(z,"face_source_ancestry_indptr","face_source_ancestry_face_rows",len(base_faces))
 sub0=csr_rows(z,"face_fine_subface_ancestry_indptr","face_fine_subface_ancestry_rows",len(base_faces))
 det0=csr_rows(z,"face_fine_detail_ancestry_indptr","face_fine_detail_ancestry_rows",len(base_faces))
 par0=csr_rows(z,"refinement_face_parent_ancestry_indptr","refinement_face_parent_ancestry_face_rows",len(base_faces))
 hkeys=[k for k in z if k.startswith("quality_edge_flip_")]
 h0={k:z[k].copy() for k in hkeys}
 operations=[]; receipts=[]
 for spec in TARGETS:
  # Recompute the edge graph and owner constraints after each previous operation.
  delta,owner_report=owner.propose_patch_flips(z,[DETAIL],[spec["seed"]],max_flips=1,max_rounds=1,min_angle_gain_deg=1e-8,min_normal_dot=0.0)
  ops=owner_report.get("operations",[])
  if owner_report.get("accepted_flip_count")!=1 or len(ops)!=1:raise ValueError("seed did not yield exactly one owner-supported flip: "+json.dumps(owner_report,sort_keys=True))
  op=ops[0]
  if sorted(op["face_rows"])!=spec["pair"]:raise ValueError("revalidated seed chose unexpected face pair: "+repr(op["face_rows"]))
  if sorted(op["edge"])!=spec["edge"] or sorted(op["new_diagonal"])!=spec["new_diagonal"]:raise ValueError("seed changed its expected local operation")
  z=delta;owner.validate_candidate(z)
  operations.append(op);receipts.append({"seed":spec["seed"],"owner_status":owner_report["status"],"accepted_flip_count":owner_report["accepted_flip_count"],"changed_face_rows":owner_report["changed_face_rows"],"operation":op,"freshly_revalidated_after_previous_flip":bool(operations[:-1])})
  if not z["points_world_m"].tobytes()==base_points.tobytes():raise ValueError("owner moved coordinates")
  if not np.array_equal(z["area_reference_037_source_O_faces"],z["source_O_faces"]):raise ValueError("037 faces were not synchronized")
  if patch_boundary(z["source_O_faces"],PATCH_ROWS)!=boundary:raise ValueError("local component boundary changed")
  i,j=op["face_rows"]
  if z["refinement_face_parent_ancestry_indptr"].dtype!=np.int64:raise ValueError("parent CSR pointer dtype changed")
  now_src=csr_rows(z,"face_source_ancestry_indptr","face_source_ancestry_face_rows",len(base_faces))
  now_sub=csr_rows(z,"face_fine_subface_ancestry_indptr","face_fine_subface_ancestry_rows",len(base_faces))
  now_det=csr_rows(z,"face_fine_detail_ancestry_indptr","face_fine_detail_ancestry_rows",len(base_faces))
  now_par=csr_rows(z,"refinement_face_parent_ancestry_indptr","refinement_face_parent_ancestry_face_rows",len(base_faces))
  if any(now_src[r]!=src0[r] or now_sub[r]!=sub0[r] or now_det[r]!=det0[r] for r in (i,j)):raise ValueError("registered source/detail/subface ancestry changed unexpectedly")
  if now_par[i]!=[34132,34158] or now_par[j]!=[34132,34158]:raise ValueError(f"unexpected updated immediate-parent union: {now_par[i]}, {now_par[j]}")
  if int(z["refinement_face_parent_face_row"][i])!=-1 or int(z["refinement_face_parent_face_row"][j])!=-1:raise ValueError("multi-parent scalar parent must be -1")
  if not np.isnan(z["refinement_face_parent_barycentric"][[i,j]]).all():raise ValueError("multi-parent barycentrics must be explicit NaN")
  if any(not np.array_equal(z[k],v) for k,v in h0.items()):raise ValueError("historical quality-flip lineage changed")
  if int(z["parent_coarse_face_row"][i])!=-1 or int(z["parent_coarse_face_row"][j])!=-1:raise ValueError("coarse parent scalar unexpectedly changed")
  if int(z["parent_source_face_marker"][i])!=90091 or int(z["parent_source_face_marker"][j])!=90091:raise ValueError("source transfer marker mismatch")
 final_edges=owner.edge_map(z["source_O_faces"])
 if any(len(v)>2 for v in final_edges.values()):raise ValueError("nonmanifold edge after local flips")
 with np.load(BASE,allow_pickle=False) as src:
  for k in src.files:
   if k not in {"source_O_faces","area_reference_037_source_O_faces","face_source_ancestry_indptr","face_source_ancestry_face_rows","face_fine_subface_ancestry_indptr","face_fine_subface_ancestry_rows","refinement_face_parent_face_row","refinement_face_parent_barycentric","refinement_face_parent_ancestry_indptr","refinement_face_parent_ancestry_face_rows","refinement_face_parent_ancestry_semantics","refinement_face_parent_barycentric_semantics","parent_coarse_face_row","parent_source_face_marker"} and (z[k].dtype!=src[k].dtype or z[k].shape!=src[k].shape or z[k].tobytes()!=src[k].tobytes()):
    raise ValueError(f"unapproved mutation: {k}")
 with np.load(PARENT,allow_pickle=False) as pz: parent={k:pz[k].copy() for k in pz.files}
 surface=json.loads(AUDIT159.read_text())
 guard=loadmod("generation002_multi_parent_refresh_guard",GEN2GUARD)
 guard_receipt=guard.validate_ancestry(z,parent,surface)
 if sha(PARENT)!="beaff5ca4deeba288662cbb614b1750516948c9400476618c6a9d45f823c9a35":raise ValueError("immediate parent candidate hash mismatch")
 if surface.get("candidate_sha256")!=BASE_SHA:raise ValueError("saved audit159 does not match pinned 653 candidate")
 np.savez_compressed(OUT,**z)
 rep={"schema":"numi.human.b109_114.attempt005.owner_supported_cross_parent_flips.v1","status":"TOPOLOGY_ONLY_PREPARED_FOR_EXACT_LOCAL_CHECKS","candidate_path":str(OUT),"candidate_sha256":sha(OUT),"base_candidate_path":str(BASE),"base_candidate_sha256":sha(BASE),"owner_path":str(OWNER),"owner_sha256":sha(OWNER),"owner":"two_parent_patch_flips_402.propose_patch_flips","operations":receipts,"local_patch_component_face_count":int(len(PATCH_ROWS)),"local_outer_boundary_edge_count":int(len(boundary)),"local_boundary_preserved":True,"coordinates_bitwise_unchanged":True,"matched_037_connectivity_synchronized":True,"immediate_parent_unions_updated":True,"source_detail_subface_ancestry_preserved":True,"coarse_parent_scalar_remains_unset":True,"historical_quality_flip_lineage_preserved":True,"crossing_faces_targeted":[34705,34706],"final_immediate_parent_rows":{"34666":[34132,34158],"34705":[34132,34158],"34668":[34132,34158],"34706":[34132,34158]},"geometry_audit_run":False,"native_admission":False,"physical_steps":0,"generation002_multi_parent_refresh_guard":guard_receipt,"generation002_guard_path":str(GEN2GUARD),"generation002_guard_sha256":sha(GEN2GUARD),"immediate_parent_candidate_path":str(PARENT),"immediate_parent_candidate_sha256":sha(PARENT),"saved_audit159_path":str(AUDIT159),"saved_audit159_sha256":sha(AUDIT159),"exact_bone_self_inside_checks":"pending serialized local check under geometry slot","admission":False}
 REPORT.write_text(json.dumps(rep,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps(rep,indent=2,allow_nan=False))
if __name__=="__main__":main()
