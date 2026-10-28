"""Reusable, read-only topology checker for points_world_m/source_O_faces NPZs."""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import argparse,hashlib,json
import numpy as np
DEFAULT_BASE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001/inputs/baseline-806.npz")
DEFAULT_CAND=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/envelope-006-local-lattice/candidate.npz")
DEFAULT_OUT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/boolean-prototype-001/topology-candidate006.json")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def find(parent,x):
 while parent[x]!=x:
  parent[x]=parent[parent[x]];x=parent[x]
 return int(parent[x])
def union(parent,rank,a,b):
 a=find(parent,a);b=find(parent,b)
 if a==b:return
 if rank[a]<rank[b]:a,b=b,a
 parent[b]=a
 if rank[a]==rank[b]:rank[a]+=1
def coord_bits(P):
 q=np.asarray(P,dtype=np.float32)
 if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all():raise ValueError("points_world_m must be finite Nx3 Float32-compatible data")
 return np.ascontiguousarray(q).view(np.uint32).reshape(-1,3)
def duplicate_inventory(P,F,limit=100):
 bits=coord_bits(P);used=np.unique(F.reshape(-1)) if len(F) else np.zeros(0,dtype=np.int64)
 if len(used):
  u,inv,c=np.unique(bits[used],axis=0,return_inverse=True,return_counts=True)
  groups=[]
  for gi in np.flatnonzero(c>1):
   rows=used[inv==gi]
   groups.append({"coordinate_bits_u32":[int(x) for x in u[gi]],"coordinate_f32":[float(x) for x in bits[rows[0]].view(np.float32)],"vertex_rows":rows.astype(int).tolist()})
 else:groups=[]
 return {"used_vertex_count":int(len(used)),"unused_point_count":int(len(P)-len(used)),"exact_duplicate_coordinate_group_count":int(len(groups)),"exact_duplicate_coordinate_extra_vertex_count":int(sum(len(g["vertex_rows"])-1 for g in groups)),"exact_duplicate_coordinate_groups_first_limit":groups[:limit]},Counter({tuple(int(x) for x in row):int(c) for row,c in zip(u,c)}) if len(used) else Counter()
def edge_counter_by_coords(P,edge_rows):
 bits=coord_bits(P);out=Counter()
 for a,b in edge_rows:
  x=tuple(int(v) for v in bits[int(a)]);y=tuple(int(v) for v in bits[int(b)])
  out[tuple(sorted((x,y)))]+=1
 return out
def boundary_loops(oriented_edges):
 nxt={}
 for a,b in oriented_edges:
  a=int(a);b=int(b)
  if a in nxt:return [],False
  nxt[a]=b
 if len(nxt)!=len(oriented_edges):return [],False
 unseen=set(nxt);loops=[]
 while unseen:
  start=min(unseen);loop=[start];v=nxt[start]
  while v!=start:
   if v not in nxt or v in loop:return loops,False
   loop.append(v);v=nxt[v]
  unseen.difference_update(loop);loops.append(loop)
 return loops,True
def volume(P,F):
 if not len(F):return 0.0
 ids=np.unique(F);origin=np.asarray(P,dtype=np.float64)[ids].mean(axis=0)
 t=np.asarray(P,dtype=np.float64)[F]-origin
 return float(np.einsum("ij,ij->i",t[:,0],np.cross(t[:,1],t[:,2])).sum()/6.0)
def analyse(P,F,label):
 P=np.asarray(P,dtype=np.float32);F=np.asarray(F,dtype=np.int64)
 if F.ndim!=2 or F.shape[1]!=3:raise ValueError("source_O_faces must be Fx3")
 if len(F) and (F.min()<0 or F.max()>=len(P)):raise ValueError("face index outside points array")
 n=len(F)
 # Block-major edge occurrence order matches the three concatenated triangle-edge blocks.
 E=np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]),axis=0) if n else np.zeros((0,2),dtype=np.int64)
 Eoriented=E.copy();Es=np.sort(E,axis=1)
 U,inv,cnt=np.unique(Es,axis=0,return_inverse=True,return_counts=True) if len(E) else (np.zeros((0,2),dtype=np.int64),np.zeros(0,dtype=np.int64),np.zeros(0,dtype=np.int64))
 order=np.argsort(inv,kind="stable") if len(inv) else np.zeros(0,dtype=np.int64)
 starts=np.r_[0,np.cumsum(cnt[:-1])] if len(cnt) else np.zeros(0,dtype=np.int64)
 occ_face=np.tile(np.arange(n,dtype=np.int64),3)
 parent=np.arange(n,dtype=np.int64);rank=np.zeros(n,dtype=np.int8)
 edge_faces=[];edge_comp=np.full(len(U),-1,dtype=np.int64);orient_conflicts=[];boundary_by_comp=defaultdict(list)
 for ei,(st,ct) in enumerate(zip(starts,cnt)):
  idx=order[st:st+ct];fs=occ_face[idx];edge_faces.append(fs)
  if len(fs):
   if ct>=2:
    for f in fs[1:]:union(parent,rank,int(fs[0]),int(f))
   if ct==2:
    e0=Eoriented[idx[0]];e1=Eoriented[idx[1]]
    if np.array_equal(e0,e1):orient_conflicts.append(ei)
 if n:
  roots=np.asarray([find(parent,i) for i in range(n)],dtype=np.int64)
  _,face_comp=np.unique(roots,return_inverse=True)
  for ei,fs in enumerate(edge_faces):
   if len(fs):edge_comp[ei]=face_comp[int(fs[0])]
  for ei,(st,ct) in enumerate(zip(starts,cnt)):
   if ct==1:
    occ=int(order[st]);fc=int(face_comp[occ_face[occ]])
    boundary_by_comp[fc].append(Eoriented[occ])
 else:face_comp=np.zeros(0,dtype=np.int64)
 comp_rows=[];component_loop_rows={};component_caps={};cap_points=[]
 for ci in range(int(face_comp.max()+1) if n else 0):
  fr=np.flatnonzero(face_comp==ci);v=np.unique(F[fr]);ei=np.flatnonzero(edge_comp==ci);bound=np.asarray(boundary_by_comp.get(ci,[]),dtype=np.int64).reshape(-1,2)
  loops,oriented_ok=boundary_loops(bound) if len(bound) else ([],True)
  # The undirected boundary graph must be a disjoint union of cycles too.
  deg=Counter(int(x) for e in bound for x in e)
  boundary_cycle_graph=all(d==2 for d in deg.values())
  caps=[];centers=[]
  for loop in loops:
   center=len(P)+len(cap_points);cap_points.append(np.asarray(P)[np.asarray(loop)].astype(np.float64).mean(axis=0))
   for a,b in zip(loop,loop[1:]+loop[:1]):caps.append([b,a,center])
  Fc=np.vstack((F[fr],np.asarray(caps,dtype=np.int64).reshape(-1,3))) if caps else F[fr]
  Pc=np.vstack((P,np.asarray(cap_points,dtype=np.float32))) if cap_points else P
  chi=int(len(v)-len(ei)+len(fr));b=len(loops);numer=2-b-chi;g=float(numer/2)
  vol=volume(Pc,Fc)
  orient_ct=sum(1 for e in orient_conflicts if edge_comp[e]==ci)
  # Geometric face checks stay independent from edge component logic.
  t=np.asarray(P,dtype=np.float64)[F[fr]]
  cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]) if len(t) else np.zeros((0,3))
  sorted_idx=np.sort(F[fr],axis=1);_,dupidx=np.unique(sorted_idx,axis=0,return_counts=True) if len(fr) else (None,np.zeros(0,dtype=int))
  comp_rows.append({"component":ci,"faces":int(len(fr)),"vertices_used":int(len(v)),"edges":int(len(ei)),"euler_characteristic":chi,"boundary_edge_count":int(len(bound)),"boundary_loop_count":int(b),"boundary_loop_lengths":sorted([len(q) for q in loops]),"boundary_orientation_is_cycle_consistent":bool(oriented_ok and boundary_cycle_graph),"genus_from_chi_and_boundary":g,"genus_is_nonnegative_integer":bool(numer>=0 and numer%2==0),"signed_volume_with_temporary_boundary_caps_m3":vol,"orientation_conflict_edge_count":int(orient_ct),"repeated_index_face_count":int(np.count_nonzero((F[fr,0]==F[fr,1])|(F[fr,1]==F[fr,2])|(F[fr,2]==F[fr,0]))),"exact_zero_cross_face_count":int(np.count_nonzero(np.all(cross==0,axis=1))),"duplicate_index_face_groups":int(np.count_nonzero(dupidx>1)),"bbox_min_mm":(np.asarray(P)[v].min(axis=0)*1000).round(6).tolist(),"bbox_max_mm":(np.asarray(P)[v].max(axis=0)*1000).round(6).tolist()})
  component_loop_rows[ci]=loops;component_caps[ci]=caps
 # Exact coordinate duplicate inventory and coordinate-keyed boundary edge multiset.
 dups,coord_counts=duplicate_inventory(P,F)
 b_edge_ids=np.flatnonzero(cnt==1);brows=np.asarray([Eoriented[order[starts[i]]] for i in b_edge_ids],dtype=np.int64).reshape(-1,2)
 boundary_coord_edges=edge_counter_by_coords(P,brows)
 # Duplicate geometric face rows, regardless of index identity.
 bits=coord_bits(P);geom_face_counts=Counter()
 for tri in F:
  key=tuple(sorted(tuple(int(x) for x in bits[int(v)]) for v in tri))
  geom_face_counts[key]+=1
 return {"label":label,"point_count":int(len(P)),"face_count":int(n),"component_count_edge_connected":len(comp_rows),"used_vertex_count":dups["used_vertex_count"],"unused_point_count":dups["unused_point_count"],"edge_incidence_gt2_count":int(np.count_nonzero(cnt>2)),"open_edge_count":int(np.count_nonzero(cnt==1)),"exact_zero_area_face_count":int(sum(c["exact_zero_cross_face_count"] for c in comp_rows)),"repeated_index_face_count":int(sum(c["repeated_index_face_count"] for c in comp_rows)),"duplicate_index_face_groups":int(sum(c["duplicate_index_face_groups"] for c in comp_rows)),"duplicate_geometric_face_groups":int(sum(v>1 for v in geom_face_counts.values())),"duplicate_geometric_face_extra_rows":int(sum(v-1 for v in geom_face_counts.values() if v>1)),"exact_duplicate_coordinate_group_count":dups["exact_duplicate_coordinate_group_count"],"exact_duplicate_coordinate_extra_vertex_count":dups["exact_duplicate_coordinate_extra_vertex_count"],"exact_duplicate_coordinate_groups_first_limit":dups["exact_duplicate_coordinate_groups_first_limit"],"boundary_loop_lengths_all_components":sorted([len(q) for group in component_loop_rows.values() for q in group]),"boundary_coordinate_edge_multiset":boundary_coord_edges,"components":sorted(comp_rows,key=lambda c:c["faces"],reverse=True),"component_loop_rows":component_loop_rows}
def load_npz(path):
 with np.load(path,allow_pickle=False) as z:
  if "points_world_m" not in z.files or "source_O_faces" not in z.files:raise ValueError(f"{path} lacks points_world_m/source_O_faces")
  return np.asarray(z["points_world_m"],dtype=np.float32),np.asarray(z["source_O_faces"],dtype=np.int64)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--candidate",type=Path,default=DEFAULT_CAND);ap.add_argument("--baseline",type=Path,default=DEFAULT_BASE);ap.add_argument("--out",type=Path,default=DEFAULT_OUT);ap.add_argument("--moving-pose",action="store_true");args=ap.parse_args()
 bp,bf=load_npz(args.baseline);cp,cf=load_npz(args.candidate)
 b=analyse(bp,bf,"baseline");c=analyse(cp,cf,"candidate")
 # Both boundary edge counters use exact Float32 coordinate bits; no geometric tolerance.
 bc=b.pop("boundary_coordinate_edge_multiset");cc=c.pop("boundary_coordinate_edge_multiset")
 if args.moving_pose:
  if len(bp)!=len(cp) or not np.array_equal(bf,cf):
   raise ValueError("moving-pose check requires exact source vertex count and oriented face ordering")
  # A moving material boundary keeps its IDs, rather than fixed world coordinates.
  def indexed_boundary(surface):
   out=Counter()
   for loops in surface["component_loop_rows"].values():
    for loop in loops:
     for a,b in zip(loop,loop[1:]+loop[:1]):out[tuple(sorted((a,b)))]+=1
   return out
  bc=indexed_boundary(b);cc=indexed_boundary(c)
 missing=bc-cc;extra=cc-bc
 bm=b["components"];cm=c["components"]
 bmain=max(bm,key=lambda r:r["faces"]) if bm else None;cmain=max(cm,key=lambda r:r["faces"]) if cm else None
 # Only newly introduced duplicate coordinate groups trigger the strict duplicate gate;
 # baseline coincidences remain fully inventoried for explicit seam/contact review.
 with np.load(args.baseline,allow_pickle=False) as bz,np.load(args.candidate,allow_pickle=False) as cz:
  bdup,_=duplicate_inventory(bz["points_world_m"],bz["source_O_faces"])
  cdup,_=duplicate_inventory(cz["points_world_m"],cz["source_O_faces"])
 bkeys=Counter({tuple(int(x) for x in row["coordinate_bits_u32"]):len(row["vertex_rows"]) for row in bdup["exact_duplicate_coordinate_groups_first_limit"]})
 # Recompute full coordinate-group counts so allowlist does not depend on the detail cap.
 def dup_counts(P,F):
  bits=coord_bits(P);used=np.unique(F.reshape(-1));_,inv,c=np.unique(bits[used],axis=0,return_inverse=True,return_counts=True)
  return Counter({tuple(int(x) for x in bits[used][np.flatnonzero(inv==i)[0]]):int(n) for i,n in enumerate(c) if n>1})
 bcounts=dup_counts(bp,bf);ccounts=dup_counts(cp,cf)
 new_dup_keys=[k for k,v in ccounts.items() if v>bcounts.get(k,0)]
 reasons=[]
 if len(cm)!=len(bm):reasons.append("edge_connected_component_count_differs_from_baseline")
 if not cmain or not bmain or cmain["genus_from_chi_and_boundary"]!=bmain["genus_from_chi_and_boundary"]:reasons.append("main_component_genus_differs_from_baseline")
 if not cmain or cmain["genus_from_chi_and_boundary"]!=0:reasons.append("main_component_genus_is_not_zero")
 if missing or extra:reasons.append("head_or_other_open_boundary_coordinate_edges_differ_from_baseline")
 if c["edge_incidence_gt2_count"]:reasons.append("nonmanifold_edges_present")
 if c["exact_zero_area_face_count"] or c["repeated_index_face_count"]:reasons.append("degenerate_faces_present")
 if c["duplicate_index_face_groups"] or c["duplicate_geometric_face_groups"]:reasons.append("duplicate_faces_present")
 if new_dup_keys:reasons.append("new_unadjudicated_exact_coordinate_duplicates_present")
 if c["unused_point_count"]:reasons.append("unused_points_present")
 if any(not x["genus_is_nonnegative_integer"] or x["orientation_conflict_edge_count"] for x in c["components"]):reasons.append("component_topology_or_winding_is_inconsistent")
 report={"status":"COMPLETE_READ_ONLY_TOPOLOGY_AUDIT","candidate_path":str(args.candidate),"candidate_sha256":sha(args.candidate),"baseline_path":str(args.baseline),"baseline_sha256":sha(args.baseline),"baseline":b,"candidate":c,"moving_pose":args.moving_pose,"boundary_identity_policy":"exact oriented topology and boundary row IDs" if args.moving_pose else "exact Float32 coordinate boundary edges","boundary_edge_multiset_exact_bitwise_match":not missing and not extra,"missing_baseline_boundary_edges":int(sum(missing.values())),"extra_candidate_boundary_edges":int(sum(extra.values())),"baseline_main_genus":bmain["genus_from_chi_and_boundary"] if bmain else None,"candidate_main_genus":cmain["genus_from_chi_and_boundary"] if cmain else None,"new_exact_duplicate_coordinate_groups_vs_baseline":len(new_dup_keys),"new_duplicate_coordinate_keys_first100":[list(k) for k in new_dup_keys[:100]],"strict_topology_acceptance":not reasons,"strict_topology_rejection_reasons":reasons,"duplicate_policy":"Exact coordinate duplicates are inventoried; groups already present in the pinned baseline are treated as inherited contacts/seams for comparison, while any new group or multiplicity increase blocks strict acceptance until annotated.","head_cap_policy":"Temporary caps are used only for volume; original head loops stay open. Reference checks compare exact coordinates; accepted moving-pose checks compare exact source row identities."}
 # Keep component report compact; no vertex/face arrays are emitted.
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
