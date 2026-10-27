#!/usr/bin/env python3
"""Exact local 109/114 bone, inside-node and nonadjacent-self checks for attempt 005."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import hashlib, importlib.util, json, sys, time, traceback
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
S=ROOT/"D/surface-clearance-546/component-sat-selector-638"
W=S/"per-bank-workers-20261027/b109-114"
D=W/"attempt-005-cross-parent-edge-flip-001"
BASE=ROOT/"D/publication-182-stage-001/combined-653/candidate.npz"
BASE_SHA="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af"
AUDIT159=BASE.parent/"audit-159/report.json"
OUTCAND=D/"candidate-two-cross-parent-flips.npz"
TOPO=D/"topology-and-lineage-report.json"
REPORT=D/"exact-local-report.json"
AUDIT_TOOL=ROOT/"audit_skin_bone_surfaces_159.py"
HELPER=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
BANKDIR=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
PATCHES=W/"attempt-001-boundary-feasibility/patch-components"
BODY=109; BONE=114

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def frac(x):return Fraction.from_float(float(x))
def exact_records_mesh(points,faces,face_ids,vertex_ids,exact):
 pts=[tuple(frac(x) for x in p) for p in np.asarray(points).reshape(-1,3)];out=[]
 for fid,face in zip(face_ids,faces):
  ids=tuple(map(int,face));tri=tuple(pts[i] for i in ids);n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]))
  if not any(n):raise ValueError("exact degenerate face "+str(fid))
  lo=tuple(min(p[k] for p in tri) for k in range(3));hi=tuple(max(p[k] for p in tri) for k in range(3))
  out.append((tri,lo,hi,int(fid),tuple(int(vertex_ids[i]) for i in ids)))
 return out
def pairset(result):return {tuple(sorted(map(int,x))) for x in result.get("triangle_pairs",[])}
def csr_rows(z,ptrkey,valskey,n):
 p=np.asarray(z[ptrkey],np.int64);v=np.asarray(z[valskey],np.int64)
 if p.shape!=(n+1,) or int(p[0])!=0 or int(p[-1])!=len(v) or np.any(p[1:]<p[:-1]):raise ValueError("invalid CSR "+ptrkey)
 return [list(map(int,v[p[i]:p[i+1]])) for i in range(n)]
def mapped_parent_pairs(raw,parents):return {tuple((int(p),int(b))) for f,b in raw for p in parents[int(f)]}
def mapped_registered_pairs(raw,parents,ancestry):return {tuple((int(a),int(b))) for f,b in raw for p in parents[int(f)] for a in ancestry[int(p)]}
def main():
 started=time.monotonic();rep={"schema":"numi.human.b109_114.attempt005.exact_local_check.v1","status":"STARTED","native_admission":False,"physical_steps":0,"full_body_audit_run":False,"admission":False}
 try:
  if sha(BASE)!=BASE_SHA:raise ValueError("pinned base candidate hash mismatch")
  topo=json.loads(TOPO.read_text());csha=sha(OUTCAND)
  if topo.get("candidate_sha256")!=csha or topo.get("base_candidate_sha256")!=BASE_SHA:raise ValueError("topology receipt/candidate pin mismatch")
  if not topo.get("coordinates_bitwise_unchanged") or not topo.get("local_boundary_preserved"):raise ValueError("topology receipt failed geometry/boundary invariants")
  with np.load(BASE,allow_pickle=False) as q:base={k:q[k].copy() for k in q.files}
  with np.load(OUTCAND,allow_pickle=False) as q:trial={k:q[k].copy() for k in q.files}
  P0=np.asarray(base["points_world_m"]);P1=np.asarray(trial["points_world_m"]);F0=np.asarray(base["source_O_faces"],np.int64);F1=np.asarray(trial["source_O_faces"],np.int64)
  if P0.dtype!=np.float32 or P1.dtype!=np.float32 or not np.array_equal(P0.view(np.uint32),P1.view(np.uint32)):raise ValueError("emitted Float32 coordinates changed")
  if F0.shape!=F1.shape:raise ValueError("candidate face domain changed")
  patch_rows=np.concatenate([np.load(PATCHES/f"component-{i:02d}-face-rows.npy",allow_pickle=False).astype(np.int64) for i in (0,1)])
  patch_set=set(map(int,patch_rows));nodes=np.unique(F0[patch_rows]).astype(np.int64)
  complement=np.asarray([i for i in range(len(F0)) if i not in patch_set],np.int64)
  # Resolve exact Float32 geometry and the existing ordered stable-bone identities.
  audit=loadmod("attempt005_audit159_owner",AUDIT_TOOL);helper=loadmod("attempt005_bank_loader",HELPER);exact=audit.exact_module()
  T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=helper.read_bank_inputs(BANKDIR,E)
  mask=(ids[:,0]==BODY)&(ids[:,1]==BONE);bone_ids=np.asarray(ids[mask,3],np.int64);bone_raw=np.asarray(T[mask],np.float64)
  if not len(bone_ids):raise ValueError("target bone export is empty")
  bone_records=audit.records(bone_raw,bone_ids,exact);prepared=exact.prepare_point_location(bone_records)
  def local_points(P):return (P.astype(np.float64)-pos[BODY])@M[BODY].T-off[BODY]
  L0=local_points(P0);L1=local_points(P1)
  def records_for(L,F,rows):return audit.records(L[F[rows]],rows,exact)
  bone0=exact._audit_pair(records_for(L0,F0,patch_rows),bone_records,same_surface=False)
  bone1=exact._audit_pair(records_for(L1,F1,patch_rows),bone_records,same_surface=False)
  raw0=pairset(bone0);raw1=pairset(bone1)
  saved=json.loads(AUDIT159.read_text())
  if saved.get("candidate_sha256")!=BASE_SHA or saved.get("complete") is not True or int(saved.get("bank_count",-1))!=185:raise ValueError("saved exact audit159 is not complete/pinned to baseline")
  bank=next(x for x in saved["banks"] if (int(x["body"]),int(x["stable_bone_id"]))==(BODY,BONE))
  expected={(int(f),int(b)) for f,b in (bank.get("result") or {}).get("triangle_pairs",[]) if int(f) in patch_set}
  if raw0!=expected:raise ValueError("zero-motion exact local control failed to reproduce saved 109/114 child pair identities")
  parent0=csr_rows(base,"refinement_face_parent_ancestry_indptr","refinement_face_parent_ancestry_face_rows",len(F0))
  parent1=csr_rows(trial,"refinement_face_parent_ancestry_indptr","refinement_face_parent_ancestry_face_rows",len(F1))
  src0=csr_rows(base,"face_source_ancestry_indptr","face_source_ancestry_face_rows",len(F0))
  src1=csr_rows(trial,"face_source_ancestry_indptr","face_source_ancestry_face_rows",len(F1))
  detail0=csr_rows(base,"face_fine_detail_ancestry_indptr","face_fine_detail_ancestry_rows",len(F0))
  detail1=csr_rows(trial,"face_fine_detail_ancestry_indptr","face_fine_detail_ancestry_rows",len(F1))
  sub0=csr_rows(base,"face_fine_subface_ancestry_indptr","face_fine_subface_ancestry_rows",len(F0))
  sub1=csr_rows(trial,"face_fine_subface_ancestry_indptr","face_fine_subface_ancestry_rows",len(F1))
  pp0=mapped_parent_pairs(raw0,parent0);pp1=mapped_parent_pairs(raw1,parent1)
  # Registered source/detail/subface mapping is also reported directly from each emitted child CSR.
  source_pairs0={(a,b) for f,b in raw0 for a in src0[int(f)]};source_pairs1={(a,b) for f,b in raw1 for a in src1[int(f)]}
  detail_pairs0={(a,b) for f,b in raw0 for a in detail0[int(f)]};detail_pairs1={(a,b) for f,b in raw1 for a in detail1[int(f)]}
  subface_pairs0={(a,b) for f,b in raw0 for a in sub0[int(f)]};subface_pairs1={(a,b) for f,b in raw1 for a in sub1[int(f)]}
  # Exact point locations for every patch node; coordinates are identical before/after.
  inside=[];loc_counts=Counter()
  for node in map(int,nodes):
   loc=exact.point_location_prepared(tuple(frac(x) for x in L1[node]),prepared)["location"]
   loc_counts[loc]+=1
   if loc in ("inside","boundary"):inside.append({"node_row":node,"location":loc,"world_m":P1[node].astype(float).tolist()})
  vertex_ids=np.arange(len(P1),dtype=np.int64)
  def self_screens(L,F):
   precords=exact_records_mesh(L,F[patch_rows],patch_rows,vertex_ids,exact)
   internal=exact._audit_pair(precords,precords,same_surface=True);iset=pairset(internal)
   patch_tris=L[F[patch_rows]];lo=patch_tris.min(axis=(0,1));hi=patch_tris.max(axis=(0,1))
   comp_tris=L[F[complement]];clo=comp_tris.min(axis=1);chi=comp_tris.max(axis=1)
   active=np.flatnonzero(np.all(chi>=lo,axis=1)&np.all(clo<=hi));comp_ids=complement[active]
   crecords=exact_records_mesh(L,F[comp_ids],comp_ids,vertex_ids,exact)
   cross=exact._audit_pair(precords,crecords,same_surface=True);cset=pairset(cross)
   return {"internal":internal,"internal_set":iset,"complement":cross,"complement_set":cset,"complement_aabb_active_faces":comp_ids.tolist()}
  self0=self_screens(L0,F0);self1=self_screens(L1,F1)
  no_new_raw=not (raw1-raw0);raw_removed=sorted(raw0-raw1)
  no_new_parent=not (pp1-pp0);parent_removed=sorted(pp0-pp1)
  no_new_self=not (self1["internal_set"]-self0["internal_set"]) and not (self1["complement_set"]-self0["complement_set"])
  no_new_inside=True # Exact point locations are unchanged because emitted F32 coordinates are byte-identical.
  strict_parent_reduction=bool(parent_removed)
  passed=bool(no_new_raw and no_new_parent and no_new_self and no_new_inside and strict_parent_reduction)
  rep.update({"status":"LOCAL_EXACT_CHECK_PASS" if passed else "LOCAL_EXACT_CHECK_REJECTED","base_candidate_sha256":BASE_SHA,"candidate_path":str(OUTCAND),"candidate_sha256":csha,"audit159_baseline_path":str(AUDIT159),"audit159_baseline_sha256":sha(AUDIT159),"target_bank":{"body":BODY,"stable_bone_id":BONE},"patch_face_count":int(len(patch_rows)),"patch_node_count":int(len(nodes)),"raw_child_bone_pair_count":{"baseline":len(raw0),"trial":len(raw1),"removed":[list(x) for x in sorted(raw0-raw1)],"new":[list(x) for x in sorted(raw1-raw0)],"trial_pairs":[list(x) for x in sorted(raw1)]},"immediate_parent_bone_pair_count":{"baseline":len(pp0),"trial":len(pp1),"removed":[list(x) for x in parent_removed],"new":[list(x) for x in sorted(pp1-pp0)],"trial_pairs":[list(x) for x in sorted(pp1)]},"registered_source_bone_pair_count":{"baseline":len(source_pairs0),"trial":len(source_pairs1),"removed":[list(x) for x in sorted(source_pairs0-source_pairs1)],"new":[list(x) for x in sorted(source_pairs1-source_pairs0)],"trial_pairs":[list(x) for x in sorted(source_pairs1)]},"registered_detail_bone_pair_count":{"baseline":len(detail_pairs0),"trial":len(detail_pairs1),"removed":[list(x) for x in sorted(detail_pairs0-detail_pairs1)],"new":[list(x) for x in sorted(detail_pairs1-detail_pairs0)],"trial_pairs":[list(x) for x in sorted(detail_pairs1)]},"registered_subface_bone_pair_count":{"baseline":len(subface_pairs0),"trial":len(subface_pairs1),"removed":[list(x) for x in sorted(subface_pairs0-subface_pairs1)],"new":[list(x) for x in sorted(subface_pairs1-subface_pairs0)],"trial_pairs":[list(x) for x in sorted(subface_pairs1)]},"inside_or_boundary_patch_node_count":len(inside),"inside_or_boundary_patch_node_witnesses":inside,"point_location_counts":dict(loc_counts),"inside_nodes_unchanged_by_coordinate_identity":True,"patch_internal_nonadjacent_self_pairs":{"baseline":int(self0["internal"]["count"]),"trial":int(self1["internal"]["count"]),"new":[list(x) for x in sorted(self1["internal_set"]-self0["internal_set"])],"removed":[list(x) for x in sorted(self0["internal_set"]-self1["internal_set"]) ]},"patch_vs_complement_nonadjacent_self_pairs":{"baseline":int(self0["complement"]["count"]),"trial":int(self1["complement"]["count"]),"new":[list(x) for x in sorted(self1["complement_set"]-self0["complement_set"])],"removed":[list(x) for x in sorted(self0["complement_set"]-self1["complement_set"])],"complement_aabb_active_face_count":len(self1["complement_aabb_active_faces"])},"gates":{"baseline_reproduced_187_saved_pairs":len(expected)==187 and raw0==expected,"no_new_raw_child_bone_pairs":no_new_raw,"no_new_parent_bone_pairs":no_new_parent,"strict_parent_bone_pair_reduction":strict_parent_reduction,"no_new_inside_or_boundary_nodes":no_new_inside,"no_new_nonadjacent_self_pairs":no_new_self,"passed":passed},"construction_only":True,"native_admission":False,"physical_steps":0})
 except Exception as e:
  rep.update({"status":"LOCAL_EXACT_CHECK_ABORTED","failure":repr(e),"traceback":traceback.format_exc()})
 rep["elapsed_seconds"]=time.monotonic()-started
 REPORT.write_text(json.dumps(rep,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({k:v for k,v in rep.items() if k!="traceback"},indent=2,allow_nan=False))
if __name__=="__main__":main()
