#!/usr/bin/env python3
"""Bounded six-anchor retessellation diagnostic for the three run-003 O patches."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, socket, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
sys.path[:0]=[str(WORK),str(WORK/"reference-fit-layer-counterfactual-001")]
import reference_frame_mapping as ref
import connected_interior_reference_fit as fit

NPZ=HERE/"run-003/outer-target-preflight.npz"
OUT=HERE/"retessellation-001"
PRED_SRC=fit.PRED_SRC
PRED=fit.PRED
PATCHES={
  "donor23":(3019,3020,3021,3105),
  "donor20":(3908,3910,3945,3946),
  "donor99":(4682,4683,4697,4698),
}


def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()


def directed_boundary(faces):
 counts={}
 for tri in np.asarray(faces,dtype=np.int64):
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
   key=tuple(sorted((int(a),int(b))))
   counts.setdefault(key,[]).append((int(a),int(b)))
 if any(len(v)>2 for v in counts.values()): raise ValueError("nonmanifold patch edge")
 directed=[v[0] for v in counts.values() if len(v)==1]
 outgoing={};incoming={}
 for a,b in directed:
  if a in outgoing or b in incoming: raise ValueError("patch boundary branches")
  outgoing[a]=b; incoming[b]=a
 if not directed or set(outgoing)!=set(incoming): raise ValueError("open or inconsistent directed patch boundary")
 start=min(outgoing); cycle=[start]; cur=outgoing[start]
 while cur!=start:
  if cur in cycle or cur not in outgoing or len(cycle)>len(directed): raise ValueError("boundary is not one cycle")
  cycle.append(cur);cur=outgoing[cur]
 if len(cycle)!=len(directed): raise ValueError("multiple boundary cycles")
 return cycle,tuple(sorted(directed))


def triangulations(cycle):
 """All oriented Catalan triangulations of a simple ordered polygon."""
 n=len(cycle)
 from functools import lru_cache
 @lru_cache(None)
 def rec(i,j):
  if j<=i+1:return ((),)
  out=[]
  for k in range(i+1,j):
   for left in rec(i,k):
    for right in rec(k,j):
     out.append(left+right+((cycle[i],cycle[k],cycle[j]),))
  return tuple(out)
 raw=rec(0,n-1)
 unique={tuple(sorted(tuple(sorted(t)) for t in rows)):rows for rows in raw}
 return list(unique.values())


def directed_boundary_set(faces):
 return directed_boundary(faces)[1]


def point_triangle_distances(points, triangles):
 """Exact-in-branch, float64 point-to-triangle distances for a small local patch."""
 Q=np.asarray(points,dtype=np.float64); T=np.asarray(triangles,dtype=np.float64)
 out=np.full((len(Q),len(T)),np.inf)
 for qi,q in enumerate(Q):
  a,b,c=T[:,0],T[:,1],T[:,2]
  ab=b-a;ac=c-a;ap=q-a
  d00=np.einsum("ij,ij->i",ab,ab);d01=np.einsum("ij,ij->i",ab,ac);d11=np.einsum("ij,ij->i",ac,ac)
  d20=np.einsum("ij,ij->i",ap,ab);d21=np.einsum("ij,ij->i",ap,ac)
  den=d00*d11-d01*d01
  v=(d11*d20-d01*d21)/np.maximum(den,1e-300)
  w=(d00*d21-d01*d20)/np.maximum(den,1e-300)
  inside=(v>=0)&(w>=0)&(v+w<=1)&(den>0)
  n=np.cross(ab,ac); nn=np.einsum("ij,ij->i",n,n)
  plane=np.abs(np.einsum("ij,ij->i",ap,n))/np.sqrt(np.maximum(nn,1e-300))
  # Distance to each of the three closed segments handles points outside the face.
  ds=[]
  for x,y in ((a,b),(b,c),(c,a)):
   e=y-x; t=np.einsum("ij,ij->i",q-x,e)/np.maximum(np.einsum("ij,ij->i",e,e),1e-300)
   closest=x+np.clip(t,0,1)[:,None]*e
   ds.append(np.linalg.norm(q-closest,axis=1))
  edge=np.minimum.reduce(ds)
  out[qi]=np.where(inside,plane,edge)
 return out


def source_patch_for_ids(source_faces, anchor_ids, rings=2):
 active=set(map(int,anchor_ids)); chosen=set()
 for _ in range(rings):
  mask=np.any(np.isin(source_faces,np.fromiter(active,dtype=np.int64)),axis=1)
  ids=np.flatnonzero(mask); chosen.update(map(int,ids))
  active.update(map(int,source_faces[ids].ravel()))
 return np.asarray(sorted(chosen),dtype=np.int64)


def main():
 t0=time.time(); OUT.mkdir(parents=True,exist_ok=False)
 z=np.load(NPZ)
 P=np.asarray(z["outer_target_AP_m"],dtype=np.float64)
 F=np.asarray(z["outer_faces"],dtype=np.int64)
 direct=np.asarray(z["source_vertex_ids"],dtype=np.uint32)
 if P.shape!=(2411,3) or F.shape!=(4777,3):raise ValueError("unexpected D36 O cardinality")
 pred=fit.load_exact_predicate()
 # Detailed source surface in the exact AP coordinate frame used by the target.
 skin=ref.load_skin(ref.SKIN294)
 rows=fit.read_csv(ref.CSV294)
 X=np.asarray([[float(r["w"+c]) for c in "xyz"] for r in rows],dtype=np.float64)
 frame=json.loads(ref.FRAME.read_text());R=np.asarray(frame["placement_rotation"],dtype=np.float64);t=np.asarray(frame["placement_translation_m"],dtype=np.float64)
 XP=X@R.T+t
 SF=np.asarray(skin["faces"],dtype=np.int64)
 if np.max(np.linalg.norm(P[direct!=np.iinfo(np.uint32).max]-XP[direct[direct!=np.iinfo(np.uint32).max]],axis=1),initial=0)>1e-12:
  raise ValueError("direct source anchors do not match the pinned AP target frame")
 verts=tuple(tuple(__import__("fractions").Fraction.from_float(float(x)) for x in row) for row in P)
 baseline_records=pred._records(verts,F.tolist())
 original_edge_counts={}
 for tri in F:
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
   k=tuple(sorted((int(a),int(b))));original_edge_counts[k]=original_edge_counts.get(k,0)+1
 baseline_boundary=tuple(sorted(e for e,n in original_edge_counts.items() if n==1))
 trial_results={}; selected_faces={}
 source_tri=XP[SF]
 source_norm=np.cross(source_tri[:,1]-source_tri[:,0],source_tri[:,2]-source_tri[:,0])
 source_norm/=np.maximum(np.linalg.norm(source_norm,axis=1)[:,None],1e-300)
 source_cent=source_tri.mean(axis=1)

 for label, face_ids in PATCHES.items():
  old=F[np.asarray(face_ids,dtype=np.int64)]
  cycle,boundary=directed_boundary(old)
  options=triangulations(cycle)
  if len(cycle)!=6 or len(options)!=14:raise ValueError(f"expected simple hexagon/14 Catalan options for {label}; got {len(cycle)}/{len(options)}")
  anch_ids=direct[np.asarray(cycle,dtype=np.int64)].astype(np.int64)
  if np.any(anch_ids==np.iinfo(np.uint32).max):raise ValueError("non-direct boundary anchor: current evidence supports only direct ancestry")
  src_face_ids=source_patch_for_ids(SF,anch_ids,rings=2)
  if len(src_face_ids)==0:raise ValueError("empty detailed source neighborhood")
  local_src=source_tri[src_face_ids]; local_norm=source_norm[src_face_ids]; local_cent=source_cent[src_face_ids]
  # Detailed surface guidance is a ranking signal only; exact D36 predicates decide acceptance.
  rows=[]
  for oi, option in enumerate(options):
   cand=np.asarray(option,dtype=np.int64)
   if directed_boundary_set(cand)!=boundary:continue
   # Reject exact/near-zero Float64 triangles and rank source proximity + shape quality.
   tri=P[cand]; cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); area=.5*np.linalg.norm(cross,axis=1)
   if np.any(area<=0):continue
   norms=cross/np.linalg.norm(cross,axis=1)[:,None]
   cent=tri.mean(axis=1)
   dist=point_triangle_distances(cent,local_src)
   nearest=np.argmin(dist,axis=1); source_dist=dist[np.arange(len(cent)),nearest]
   normal_agree=np.abs(np.einsum("ij,ij->i",norms,local_norm[nearest]))
   edges=np.concatenate((np.linalg.norm(tri[:,1]-tri[:,0],axis=1),np.linalg.norm(tri[:,2]-tri[:,1],axis=1),np.linalg.norm(tri[:,0]-tri[:,2],axis=1)))
   min_alt=np.min((2*area[:,None])/np.maximum(np.asarray([[np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[1]),np.linalg.norm(q[0]-q[2])] for q in tri]),1e-300))
   # Replace four original records and audit every self-pair touching those rows against full surface.
   new_records=pred._records(verts,cand.tolist())
   by_id={int(r[3]):r for r in baseline_records}
   for j,rec in enumerate(new_records):
    by_id[int(face_ids[j])]=(rec[0],rec[1],rec[2],int(face_ids[j]),rec[4])
   records=tuple(by_id[i] for i in range(len(F)))
   report=pred._audit_self_pairs_touching_faces(records,set(map(int,face_ids)))
   pairs=[list(map(int,pair)) for pair in report.get("triangle_pairs",[])]
   rows.append({"option":oi,"faces":cand.tolist(),"pair_count":int(report["count"]),"pairs":pairs,
       "source_neighborhood_faces":int(len(src_face_ids)),"source_distance_m":source_dist.tolist(),
       "source_distance_mean_m":float(source_dist.mean()),"source_distance_max_m":float(source_dist.max()),
       "absolute_normal_agreement_mean":float(normal_agree.mean()),"minimum_triangle_area_m2":float(area.min()),
       "minimum_altitude_m":float(min_alt),"maximum_edge_m":float(edges.max())})
  if not rows:raise ValueError(f"no oriented valid triangulations for {label}")
  # Exact-pair count is the hard first key. Source surface fit then triangle quality are ranking only.
  rows.sort(key=lambda r:(r["pair_count"],r["source_distance_mean_m"],-r["absolute_normal_agreement_mean"],-r["minimum_altitude_m"]))
  trial_results[label]={"old_face_ids":list(face_ids),"boundary_cycle_outer_rows":cycle,
    "boundary_cycle_source_vertex_ids":anch_ids.tolist(),"directed_boundary_edge_count":len(boundary),
    "candidate_count":len(rows),"detailed_source_neighborhood_face_ids":src_face_ids.tolist(),
    "candidates":rows,"selected_option":rows[0]["option"],"selected":rows[0]}
  selected_faces.update({int(fi):np.asarray(rows[0]["faces"][j],dtype=np.int64) for j,fi in enumerate(face_ids)})

 # Apply all three individually best exact-valid triangulations and recheck the entire D36 outer surface.
 Fnew=F.copy()
 for fi,tri in selected_faces.items():Fnew[fi]=tri
 all_records=pred._records(verts,Fnew.tolist())
 final_self=pred._audit_self_pairs_touching_faces(all_records,set(selected_faces))
 edge_counts={}
 directed_counts={}
 for tri in Fnew:
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
   k=tuple(sorted((int(a),int(b))));edge_counts[k]=edge_counts.get(k,0)+1;directed_counts[(int(a),int(b))]=directed_counts.get((int(a),int(b)),0)+1
 final_boundary=tuple(sorted(e for e,n in edge_counts.items() if n==1))
 nonmanifold_edges=[list(k) for k,n in edge_counts.items() if n>2]
 # Candidate changes only face connectivity; all global vertices and positions remain bitwise identical.
 if final_boundary!=baseline_boundary:raise ValueError("global boundary changed during patch retessellation")
 report={"schema":"numi.human.d36-outer-target-local-retessellation.v1",
  "classification":"bounded_connectivity_only_counterfactual_exact_self_gate_pending_volume_and_physical_acceptance",
  "inputs":{"target_npz":{"path":str(NPZ),"sha256":fit.sha(NPZ)},"detail_skin":{"path":str(ref.SKIN294),"sha256":fit.sha(ref.SKIN294)},"detail_current_csv":{"path":str(ref.CSV294),"sha256":fit.sha(ref.CSV294)},"placement":{"path":str(ref.FRAME),"sha256":fit.sha(ref.FRAME)},"exact_predicate":{"path":str(PRED),"sha256":fit.sha(PRED)}},
  "candidate_topology":{"outer_vertices":int(len(P)),"outer_faces":int(len(F)),"changed_face_rows":sorted(selected_faces),"patches":trial_results,
    "combined_changed_patch_pair_count":int(final_self["count"]),"combined_changed_patch_pairs":[list(map(int,p)) for p in final_self.get("triangle_pairs",[])],
    "global_boundary_edges_unchanged":final_boundary==baseline_boundary,"global_boundary_edge_count":len(final_boundary),
    "nonmanifold_edges_after":nonmanifold_edges,"all_vertex_positions_bitwise_unchanged":True,"faces_with_ancestry":"Each replacement face belongs to its original four-face D36 patch; each corner retains its exact direct NHSKIN294 vertex ID. No one-to-one parent face identity is asserted."},
  "limits":["The detailed source surface is used only as a local ranking guide; no tissue/bone/tendon clearance, volumetric I/D rebuild, mass, constitutive, or native runtime gate was run.","A zero exact self-pair result on the coarse D36 O surface would only qualify this outer connectivity counterfactual; it does not qualify the connected skin-fat volume.","The run-003 stable_source_face column is retained in its original report but is not treated here as a verified detailed294 face ID mapping; direct source vertex IDs and incident source-face neighborhoods are the demonstrated ancestry."],
  "runtime":{"hostname":socket.gethostname(),"python":sys.version,"numpy":np.__version__,"seconds":time.time()-t0}}
 outnpz=OUT/"replacement-patches.npz"
 np.savez_compressed(outnpz,outer_vertices_m=P,baseline_faces=F,replacement_faces=Fnew,
    changed_face_ids=np.asarray(sorted(selected_faces),dtype=np.int32),
    original_source_vertex_ids=direct,source_vertex_ids_per_patch=np.concatenate([np.asarray(v["boundary_cycle_source_vertex_ids"],dtype=np.uint32) for v in trial_results.values()]),
    baseline_target_npz_sha256=np.asarray(fit.sha(NPZ)))
 report["outputs"]={"patch_npz":{"path":str(outnpz),"sha256":fit.sha(outnpz)}}
 (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"report":str(OUT/"report.json"),"npz_sha256":report["outputs"]["patch_npz"]["sha256"],"per_patch":{k:{"pairs":v["selected"]["pair_count"],"option":v["selected_option"],"mean_source_distance_m":v["selected"]["source_distance_mean_m"]} for k,v in trial_results.items()},"combined_pair_count":int(final_self["count"]),"nonmanifold_edges":len(nonmanifold_edges)},indent=2))

if __name__=="__main__":main()
