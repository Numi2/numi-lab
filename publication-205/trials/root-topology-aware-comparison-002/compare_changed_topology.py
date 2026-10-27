#!/usr/bin/env python3
"""Compare exact 130/159 diagnostics across a declared remesh ancestry map.

This checks map structure and identity accounting, not geometric ancestry truth,
solid clearance, material ownership, or physical admission.  Parent-face CSR
rows index baseline faces; a removed inside node is explicitly retained as an
unresolved removed witness and is never credited as cleared.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read_npz(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def check_scene(z):
 pts=np.asarray(z["points_world_m"]);faces=np.asarray(z["source_O_faces"])
 if pts.ndim!=2 or pts.shape[1]!=3 or not np.isfinite(pts).all():raise ValueError("invalid points")
 if faces.ndim!=2 or faces.shape[1]!=3 or not np.issubdtype(faces.dtype,np.integer) or faces.min()<0 or faces.max()>=len(pts):raise ValueError("invalid faces")
 if len(np.unique(faces))!=len(pts):raise ValueError("candidate contains unreferenced nodes")
 if np.any(np.diff(np.sort(faces,axis=1),axis=1)==0):raise ValueError("candidate contains repeated face vertex")
 q=np.asarray(pts,dtype=np.float32).astype(np.float64)[faces]
 if np.any(np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)==0):raise ValueError("candidate contains degenerate emitted-Float32 face")
 p=np.asarray(z["refinement_face_parent_face_row"])
 if p.shape!=(len(faces),) or not np.issubdtype(p.dtype,np.integer) or np.any(p<0):raise ValueError("invalid immediate parent face map")
 return pts,faces,p.astype(np.int64)
def read159(path,candidate_sha,face_count):
 j=json.loads(path.read_text())
 if not (j.get("status")=="COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT" and j.get("complete") is True and j.get("bank_count")==185 and j.get("expected_bank_count")==185 and j.get("candidate_sha256")==candidate_sha and j.get("geometry_or_native_admission") is False and j.get("script_sha256")=="de5f61b69d4b5621749df181f167b046a9d088daad081bf3dc453cef9f824ad1" and j.get("exact_predicate_sha256")=="f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d"):raise ValueError("159 audit incomplete or mismatched")
 banks={};domains={};allpairs=set()
 for b in j["banks"]:
  key=(int(b["body"]),int(b["stable_bone_id"]))
  if key in banks or int(b["skin_faces"])!=face_count:raise ValueError("bank identity or face-domain error")
  r=b.get("result")
  if int(b["bank_faces"])<=0:raise ValueError("empty bank triangle domain")
  domains[key]=int(b["bank_faces"])
  if r is None and (b.get("skin_aabb_active")!=0 or b.get("bank_aabb_active")!=0):raise ValueError("unjustified skipped bank")
  pairs=[] if r is None else r["triangle_pairs"]
  if r is not None and r.get("count")!=len(pairs):raise ValueError("bank pair count mismatch")
  s=set()
  for row,tri in pairs:
   row=int(row);tri=int(tri)
   if not 0<=row<face_count or not 0<=tri<int(b["bank_faces"]) or (row,tri) in s:raise ValueError("invalid or repeated exact pair")
   s.add((row,tri));allpairs.add((*key,row,tri))
  banks[key]=s
 if len(banks)!=185:raise ValueError("incomplete bank inventory")
 return banks,allpairs,domains

def read130(path,candidate_sha,node_count,face_count):
 j=json.loads(path.read_text())
 if not (j.get("status")=="OUTER_NODE_CLEARANCE_DIAGNOSTIC" and j.get("candidate_sha256")==candidate_sha and j.get("complete_surface_audit") is True and j.get("points")==node_count and j.get("faces")==face_count and j.get("geometry_admitted") is False and j.get("surface_predicate_sha256")=="f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d"):raise ValueError("130 audit incomplete or mismatched")
 rows=set()
 for x in j.get("inside_witnesses",[]):
  n=int(x["output_node"]);key=(n,int(x["bank_body"]),int(x["bank_stable_id"]))
  if x.get("classification")!="inside":raise ValueError("non-inside witness in 130 report")
  if not 0<=n<node_count or key in rows:raise ValueError("invalid or repeated inside witness")
  rows.add(key)
 if j.get("inside_node_bank_pairs")!=len(rows):raise ValueError("inside witness count mismatch")
 return j,rows

def make_face_parents(old_faces,new_faces,z,same,keyptr,keyrows):
 if same:
  if not np.array_equal(old_faces,new_faces):raise ValueError("declared same-face domain differs")
  return [np.array([i],dtype=np.int64) for i in range(len(new_faces))]
 ptr=np.asarray(z[keyptr]);rows=np.asarray(z[keyrows])
 if ptr.shape!=(len(new_faces)+1,) or not np.issubdtype(ptr.dtype,np.integer) or not np.issubdtype(rows.dtype,np.integer) or ptr[0]!=0 or ptr[-1]!=len(rows) or np.any(np.diff(ptr)<=0) or np.any(rows<0) or np.any(rows>=len(old_faces)):raise ValueError("invalid candidate face-to-baseline ancestry CSR")
 result=[]
 for a,b in zip(ptr[:-1],ptr[1:]):
  q=rows[int(a):int(b)].astype(np.int64)
  if len(np.unique(q))!=len(q):raise ValueError("duplicate baseline face in ancestry row")
  result.append(q)
 return result

def make_node_map(old,new,z,same,key):
 if same:
  if len(old)!=len(new):raise ValueError("declared same-node domain differs in size")
  return np.arange(len(old),dtype=np.int64)
 a=np.asarray(z[key])
 if a.shape!=(len(old),) or not np.issubdtype(a.dtype,np.signedinteger) or np.any(a< -1) or np.any(a>=len(new)):raise ValueError("invalid baseline-to-candidate node map")
 active=a[a>=0]
 if len(np.unique(active))!=len(active):raise ValueError("node map must use -1 for deleted/merged old nodes; no ambiguous many-to-one values")
 return a.astype(np.int64)

def verify_reverse_node_map(nmap,candidate,key,new_count):
 if key not in candidate:raise ValueError("missing candidate-to-baseline node map")
 rev=np.asarray(candidate[key])
 if rev.shape!=(new_count,) or not np.issubdtype(rev.dtype,np.signedinteger) or np.any(rev< -1) or np.any(rev>=len(nmap)):raise ValueError("invalid candidate-to-baseline node map")
 expected=np.full(new_count,-1,dtype=np.int64)
 for old,new in enumerate(nmap):
  if new>=0:expected[int(new)]=old
 if not np.array_equal(rev,expected):raise ValueError("node correspondence maps are not exact inverses")

def verify_parent_pin(candidate,key,expected):
 if key not in candidate:raise ValueError("changed domain lacks pinned baseline hash")
 v=np.asarray(candidate[key])
 if v.size!=1 or v.dtype.kind not in ("U","S"):raise ValueError("baseline hash must be one string")
 actual=v.reshape(-1)[0]
 if isinstance(actual,bytes):actual=actual.decode("ascii")
 if str(actual)!=expected:raise ValueError("candidate baseline ancestry pin differs")

def compare_inside(old_inside,new_inside,nmap):
 retained={(int(nmap[n]),b,bi) for n,b,bi in old_inside if nmap[n]>=0}
 deleted=sorted((n,b,bi) for n,b,bi in old_inside if nmap[n]<0)
 return retained,new_inside-retained,retained-new_inside,deleted

def compare_raw_identities(old_raw,new_raw,same_face_domain):
 if not same_face_domain:
  return {"before":len(old_raw),"after":len(new_raw),"identity_comparison_valid":False,"added":None,"removed":None,"added_count":None,"removed_count":None}
 added=sorted(new_raw-old_raw);removed=sorted(old_raw-new_raw)
 return {"before":len(old_raw),"after":len(new_raw),"identity_comparison_valid":True,"added":added,"removed":removed,"added_count":len(added),"removed_count":len(removed)}

def main():
 p=argparse.ArgumentParser();p.add_argument("baseline",type=Path);p.add_argument("baseline130",type=Path);p.add_argument("baseline159",type=Path);p.add_argument("candidate",type=Path);p.add_argument("candidate130",type=Path);p.add_argument("candidate159",type=Path);p.add_argument("output",type=Path)
 p.add_argument("--expected-baseline",required=True);p.add_argument("--expected-candidate",required=True)
 p.add_argument("--same-face-domain",action="store_true");p.add_argument("--face-parent-indptr-key");p.add_argument("--face-parent-rows-key")
 p.add_argument("--same-node-domain",action="store_true");p.add_argument("--old-to-new-node-key");p.add_argument("--new-to-old-node-key");p.add_argument("--baseline-pin-key",default="remesh806_baseline_sha256")
 a=p.parse_args();paths=[a.baseline,a.baseline130,a.baseline159,a.candidate,a.candidate130,a.candidate159]
 if not all(x.is_file() for x in paths) or a.output.exists():raise ValueError("inputs missing or output exists")
 if sha(a.baseline)!=a.expected_baseline or sha(a.candidate)!=a.expected_candidate:raise ValueError("candidate hash mismatch")
 if not a.same_face_domain and not (a.face_parent_indptr_key and a.face_parent_rows_key):raise ValueError("changed topology requires explicit face ancestry keys")
 if not a.same_node_domain and not (a.old_to_new_node_key and a.new_to_old_node_key):raise ValueError("changed node domain requires both explicit inverse node map keys")
 x,y=read_npz(a.baseline),read_npz(a.candidate);xp,xf,xparent=check_scene(x);yp,yf,yparent=check_scene(y)
 if not (a.same_face_domain and a.same_node_domain):verify_parent_pin(y,a.baseline_pin_key,a.expected_baseline)
 if a.same_face_domain:
  for k in ("refinement_face_parent_face_row","refinement_face_parent_ancestry_indptr","refinement_face_parent_ancestry_face_rows","face_source_ancestry_indptr","face_source_ancestry_face_rows","parent_source_face_marker","parent_fine_detail_face_row"):
   if (k in x)!=(k in y) or (k in x and not np.array_equal(x[k],y[k])):raise ValueError("declared same face ancestry differs: "+k)
 if a.same_node_domain:
  for k in ("node_kind","node_source_row","node_source_virtual_row","node_parent_coarse_face_row","node_parent_coarse_edge_a","node_parent_coarse_edge_b"):
   if k in x and k in y and not np.array_equal(x[k],y[k]):raise ValueError("declared same node identity differs: "+k)
 fparents=make_face_parents(xf,yf,y,a.same_face_domain,a.face_parent_indptr_key,a.face_parent_rows_key);nmap=make_node_map(xp,yp,y,a.same_node_domain,a.old_to_new_node_key)
 if not a.same_node_domain:verify_reverse_node_map(nmap,y,a.new_to_old_node_key,len(yp))
 xb,xraw,xd=read159(a.baseline159,a.expected_baseline,len(xf));yb,yraw,yd=read159(a.candidate159,a.expected_candidate,len(yf));
 if set(xb)!=set(yb) or xd!=yd:raise ValueError("typed bank inventory or triangle domain changed")
 old_parent={(body,bone,int(xparent[face]),tri) for body,bone,face,tri in xraw}
 new_parent={(body,bone,int(xparent[parent]),tri) for body,bone,face,tri in yraw for parent in fparents[face]}
 xj,xi=read130(a.baseline130,a.expected_baseline,len(xp),len(xf));yj,yi=read130(a.candidate130,a.expected_candidate,len(yp),len(yf))
 if any((b,bi) not in xb for _,b,bi in xi|yi):raise ValueError("inside witness has unknown typed bank")
 retained_old,new_inside,cleared_inside,deleted_inside=compare_inside(xi,yi,nmap)
 per=[]
 for bank in sorted(xb):
  ox={q for q in old_parent if q[:2]==bank};ny={q for q in new_parent if q[:2]==bank}
  if ox!=ny or len(xb[bank])!=len(yb[bank]):per.append({"bank":bank,"raw_before":len(xb[bank]),"raw_after":len(yb[bank]),"mapped_before":len(ox),"mapped_after":len(ny),"mapped_added":len(ny-ox),"mapped_removed":len(ox-ny)})
 report={"status":"COMPLETE_TOPOLOGY_AWARE_STATIC_COMPARISON","baseline_sha256":a.expected_baseline,"candidate_sha256":a.expected_candidate,"comparator_sha256":sha(Path(__file__)),"audit_sha256":{"baseline130":sha(a.baseline130),"baseline159":sha(a.baseline159),"candidate130":sha(a.candidate130),"candidate159":sha(a.candidate159)},"baseline_face_count":len(xf),"candidate_face_count":len(yf),"baseline_node_count":len(xp),"candidate_node_count":len(yp),"banks":len(xb),"face_map_policy":"same face identity" if a.same_face_domain else "candidate-supplied CSR into baseline faces","node_map_policy":"same typed node identity" if a.same_node_domain else "candidate-supplied old-to-new map; -1 is unresolved deletion","raw_crossings":compare_raw_identities(xraw,yraw,a.same_face_domain),"mapped_parent_pairs":{"before":len(old_parent),"after":len(new_parent),"added_count":len(new_parent-old_parent),"removed_count":len(old_parent-new_parent),"added":sorted(new_parent-old_parent),"removed":sorted(old_parent-new_parent)},"inside":{"before":len(xi),"after":len(yi),"retained_old_identity_added":sorted(new_inside),"retained_old_identity_removed":sorted(cleared_inside),"deleted_or_merged_old_inside_witnesses_unresolved":deleted_inside,"deleted_or_merged_old_inside_count":len(deleted_inside)},"self_pairs":{"before":xj.get("exact_PLC_surface_pairs"),"after":yj.get("exact_PLC_surface_pairs")},"per_changed_bank":per,"limitations":["Candidate-supplied geometric face ancestry and node correspondence require independent validation.","A deleted/merged inside node is unresolved, not a proven clearance.","Surface-only diagnostics do not certify solid containment, tissue mass, contact dynamics, or native physical admission."],"geometry_admitted":False,"physical_steps":0}
 a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");print(json.dumps({"output":str(a.output),"raw_before":len(xraw),"raw_after":len(yraw),"mapped_before":len(old_parent),"mapped_after":len(new_parent),"inside_before":len(xi),"inside_after":len(yi),"unresolved_deleted_inside":len(deleted_inside)}))
if __name__=="__main__":main()
