#!/usr/bin/env python3
"""Full-surface exact and topology gate for retessellation-001."""
from __future__ import annotations
import hashlib,json,socket,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import connected_interior_reference_fit as fit
PRED=fit.load_exact_predicate()
BASE=HERE/"run-003/outer-target-preflight.npz"
PATCH=HERE/"retessellation-001/replacement-patches.npz"
OUT=HERE/"retessellation-001/global-exact-gate.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def edge_stats(F):
 d={}
 for tri in np.asarray(F,dtype=np.int64):
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
   k=tuple(sorted((int(a),int(b))));d[k]=d.get(k,0)+1
 return {"edge_count":len(d),"boundary_edges":sorted([list(k) for k,v in d.items() if v==1]),"nonmanifold_edges":sorted([list(k) for k,v in d.items() if v>2])}
def exact_full(P,F):
 from fractions import Fraction
 v=tuple(tuple(Fraction.from_float(float(x)) for x in row) for row in np.asarray(P,dtype=np.float64))
 rec=PRED._records(v,np.asarray(F,dtype=np.int64).tolist())
 return PRED._audit_self_pairs_touching_faces(rec,set(range(len(F))))
def main():
 t=time.time();z=np.load(BASE);q=np.load(PATCH)
 P=np.asarray(z["outer_target_AP_m"],dtype=np.float64);F=np.asarray(z["outer_faces"],dtype=np.int64)
 Rp=np.asarray(q["outer_vertices_m"]);Fn=np.asarray(q["replacement_faces"],dtype=np.int64);changed=np.asarray(q["changed_face_ids"],dtype=np.int64)
 if not np.array_equal(P,Rp):raise ValueError("retessellation moved or replaced the frozen target vertex coordinates")
 if Fn.shape!=F.shape or len(changed)!=12:raise ValueError("replacement cardinality mismatch")
 if not np.array_equal(Fn[np.setdiff1d(np.arange(len(F)),changed)],F[np.setdiff1d(np.arange(len(F)),changed)]):raise ValueError("outside patch face rows changed")
 e0,e1=edge_stats(F),edge_stats(Fn)
 if e0["boundary_edges"]!=e1["boundary_edges"]:raise ValueError("global boundary changed")
 if e1["nonmanifold_edges"]:raise ValueError("nonmanifold edge incidence")
 area0=np.linalg.norm(np.cross(P[F[:,1]]-P[F[:,0]],P[F[:,2]]-P[F[:,0]]),axis=1)*.5
 area1=np.linalg.norm(np.cross(P[Fn[:,1]]-P[Fn[:,0]],P[Fn[:,2]]-P[Fn[:,0]]),axis=1)*.5
 if np.any(area1<=0):raise ValueError("zero-area output triangle")
 b=exact_full(P,F);a=exact_full(P,Fn)
 report={"schema":"numi.human.d36-retessellation-full-exact-gate.v1","classification":"connectivity_only_outer_surface_counterfactual_no_volume_or_physical_admission",
  "inputs":{"baseline_npz":{"path":str(BASE),"sha256":sha(BASE)},"patch_npz":{"path":str(PATCH),"sha256":sha(PATCH)},"exact_predicate":{"path":str(fit.PRED),"sha256":fit.sha(fit.PRED)}},
  "checks":{"baseline_full_exact_self_pairs":int(b["count"]),"candidate_full_exact_self_pairs":int(a["count"]),"candidate_full_pair_witnesses":[list(map(int,x)) for x in a.get("triangle_pairs",[])],"outer_vertex_positions_bitwise_unchanged":True,"outside_patch_faces_bitwise_unchanged":True,"changed_face_rows":changed.tolist(),"zero_area_candidate_faces":int(np.count_nonzero(area1<=0)),"minimum_candidate_area_m2":float(area1.min()),"baseline_boundary_edges":len(e0["boundary_edges"]),"candidate_boundary_edges":len(e1["boundary_edges"]),"global_directed_boundary_undirected_set_unchanged":e0["boundary_edges"]==e1["boundary_edges"],"candidate_nonmanifold_edges":e1["nonmanifold_edges"],"complete_surface_exact_audit":bool(a.get("complete_surface_audit",False))},
  "limitations":["This gate covers the 4,777-face D36 O surface only. It does not rebuild/validate connected I/D topology, material layers, bones, mass, FEM, native runtime, or physical acceptance."],
  "runtime":{"hostname":socket.gethostname(),"python":sys.version,"numpy":np.__version__,"seconds":time.time()-t}}
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"output":str(OUT),"sha256":sha(OUT),"checks":report["checks"]},indent=2))
if __name__=="__main__":main()
