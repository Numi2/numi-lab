#!/usr/bin/env python3
"""Replace the O/I/D prism cells under the three six-anchor surface patches."""
from __future__ import annotations
import hashlib,itertools,json,socket,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; WORK=HERE.parent
sys.path[:0]=[str(WORK),str(WORK/"reference-fit-layer-counterfactual-001"),str(HERE)]
import reference_frame_mapping as ref
import connected_interior_reference_fit as fit
from retriangulate_outer_target_patches import directed_boundary

SURFACE_NPZ=HERE/"retessellation-001/replacement-patches.npz"
NATIVE=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz")
AP=ref.AP
OUT=HERE/"retessellation-001/volume-cavity-001"
N=2411
PATCHES={"donor23":(3019,3020,3021,3105),"donor20":(3908,3910,3945,3946),"donor99":(4682,4683,4697,4698)}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def dets(X,T):
 q=np.asarray(X,dtype=np.float64)[np.asarray(T,dtype=np.int64)]
 return np.linalg.det(np.stack((q[:,1]-q[:,0],q[:,2]-q[:,0],q[:,3]-q[:,0]),axis=2))

def orient_tets_positive(T,X):
 out=np.asarray(T,dtype=np.int64).copy(); d=dets(X,out); bad=d<0
 if np.any(bad):out[bad,2],out[bad,3]=out[bad,3].copy(),out[bad,2].copy()
 if np.any(dets(X,out)<=0):raise ValueError("zero-volume tet in reference-oriented cavity")
 return out

def perm_sign_to_sorted(values):
 order=[sorted(values).index(x) for x in values]
 inv=sum(order[i]>order[j] for i in range(3) for j in range(i+1,3))
 return -1 if inv%2 else 1

def oriented_boundary(T,X):
 """Boundary chain of positive-oriented tets as sorted-face -> orientation sign."""
 tets=orient_tets_positive(T,X); accum={}
 for a,b,c,d in tets:
  oriented=((b,c,d),(a,d,c),(a,b,d),(a,c,b))
  for tri in oriented:
   key=tuple(sorted(map(int,tri))); sign=perm_sign_to_sorted(tri)
   accum[key]=accum.get(key,0)+sign
 return {k:v for k,v in accum.items() if v}

def cap_orientation(boundary,faces,layer):
 vals=[]
 for tri in np.asarray(faces,dtype=np.int64):
  ids=tuple(map(int,tri+layer*N)); key=tuple(sorted(ids))
  if key not in boundary or abs(boundary[key])!=1:raise ValueError("old cavity cap missing/duplicated")
  vals.append(boundary[key]*perm_sign_to_sorted(ids))
 if len(set(vals))!=1:raise ValueError("old cavity cap orientation is inconsistent")
 return vals[0]

def side_boundary(boundary,cycle,layer):
 out={}
 for i,u in enumerate(cycle):
  v=cycle[(i+1)%len(cycle)]
  quad={layer*N+u,layer*N+v,(layer+1)*N+u,(layer+1)*N+v}
  hits={k:s for k,s in boundary.items() if set(k).issubset(quad)}
  if len(hits)!=2:raise ValueError(f"expected exactly two old side triangles on boundary edge {(u,v)} layer {layer}, got {len(hits)}")
  out.update(hits)
 if len(out)!=2*len(cycle):raise ValueError("side triangles overlap across perimeter quads")
 return out

def prism_decomposition(tri,layer,perm):
 a,b,c=[int(tri[i]) for i in perm]
 lo=lambda x:layer*N+x
 hi=lambda x:(layer+1)*N+x
 return np.asarray([[lo(a),lo(b),lo(c),hi(c)],
                    [lo(a),lo(b),hi(b),hi(c)],
                    [lo(a),hi(a),hi(b),hi(c)]],dtype=np.int64)

def cavity_cells(old_tets,old_faces,layer):
 cell_rows=[];per_face={}
 for face_id in old_faces:
  tri=set(map(int,old_faces_array[face_id]))
  nodes={c+k*N for c in tri for k in (layer,layer+1)}
  rows=[i for i,t in enumerate(old_tets) if set(map(int,t)).issubset(nodes)]
  if len(rows)!=3:raise ValueError(f"face {face_id}, layer {layer}: expected 3 native old cells, found {len(rows)}")
  per_face[int(face_id)]=rows;cell_rows.extend(rows)
 if len(set(cell_rows))!=len(cell_rows):raise ValueError("old prism cells overlap between surface faces")
 if len(cell_rows)!=12:raise ValueError("expected 12 native old tets per patch/layer")
 return sorted(cell_rows),per_face

def patch_assignments(candidate_faces,old_tets,Xref,Xcurrent,cycle,layer,oldcap_rel,oldside):
 options=list(itertools.permutations(range(3)))
 cap_lower={tuple(sorted(map(int,tri+layer*N))) for tri in candidate_faces}
 cap_upper={tuple(sorted(map(int,tri+(layer+1)*N))) for tri in candidate_faces}
 expected_cap=cap_lower|cap_upper
 feasible=[];boundary_rejections=0;reference_rejections=0
 for choice in itertools.product(range(6),repeat=len(candidate_faces)):
  raw=np.concatenate([prism_decomposition(tri,layer,options[c]) for tri,c in zip(candidate_faces,choice)],axis=0)
  tets=orient_tets_positive(raw,Xref)
  dref=dets(Xref,tets);dcur=dets(Xcurrent,tets)
  if np.any(dref<=0):reference_rejections+=1;continue
  bnd=oriented_boundary(tets,Xref)
  newside=side_boundary(bnd,cycle,layer)
  if newside!=oldside:boundary_rejections+=1;continue
  cap_keys={k for k in bnd if len({x//N for x in k})==1}
  if cap_keys!=expected_cap or set(bnd)!=(set(oldside)|expected_cap):boundary_rejections+=1;continue
  lower_rel,upper_rel=oldcap_rel;oriented_ok=True
  for tri in candidate_faces:
   for layer_cap,rel in ((layer,lower_rel),(layer+1,upper_rel)):
    ids=tuple(map(int,tri+layer_cap*N));key=tuple(sorted(ids))
    if bnd.get(key,0)!=rel*perm_sign_to_sorted(ids) or abs(bnd.get(key,0))!=1:
     oriented_ok=False;break
   if not oriented_ok:break
  if not oriented_ok:boundary_rejections+=1;continue
  feasible.append({"choice":choice,"tets":tets,"reference_dets":dref,"current_dets":dcur,
    "reference_min_det":float(dref.min()),"current_nonpositive_count":int(np.count_nonzero(dcur<=0)),"current_min_det":float(dcur.min())})
 if not feasible:return None,{"reference_feasible_assignments":0,"boundary_rejections":boundary_rejections,"reference_rejections":reference_rejections}
 feasible.sort(key=lambda x:(x["current_nonpositive_count"],-x["current_min_det"],x["choice"]))
 return feasible[0],{"reference_feasible_assignments":len(feasible),"boundary_rejections":boundary_rejections,"reference_rejections":reference_rejections,"current_positive_assignments":sum(x["current_nonpositive_count"]==0 for x in feasible)}

old_faces_array=None

def main():
 global old_faces_array
 t0=time.time();OUT.mkdir(parents=True,exist_ok=False)
 s=np.load(SURFACE_NPZ);native=np.load(NATIVE);ap=np.load(AP)
 F0=np.asarray(native["O_faces"],dtype=np.int64);I0=np.asarray(native["I_faces"],dtype=np.int64);D0=np.asarray(native["D_faces"],dtype=np.int64)
 old=np.asarray(native["old_tetrahedra"],dtype=np.int64);mat=np.asarray(native["old_material_ids"],dtype=np.int64)
 if old.shape!=(28632,4) or mat.shape!=(len(old),):raise ValueError("native-026 old tet/material domain mismatch")
 if not (np.array_equal(I0-N,F0) and np.array_equal(D0-2*N,F0)):raise ValueError("O/I/D surface source faces do not share a common column topology")
 newF=np.asarray(s["replacement_faces"],dtype=np.int64);changed_faces=np.asarray(s["changed_face_ids"],dtype=np.int64)
 if newF.shape!=F0.shape:raise ValueError("surface replacement topology dimensions mismatch")
 outside=np.setdiff1d(np.arange(len(F0)),changed_faces)
 if not np.array_equal(newF[outside],F0[outside]):raise ValueError("outer patch candidate changes an undeclared face")
 old_faces_array=F0;Xref=np.asarray(ap["points"],dtype=np.float64);Xcur=Xref.copy();Xcur[:N]=np.asarray(s["outer_vertices_m"],dtype=np.float64)
 cand_tets=old.copy();all_summary={};changed_rows=[];cell_map=[]
 for label,patch_ids in PATCHES.items():
  oldpatch=F0[np.asarray(patch_ids,dtype=np.int64)];cycle,_=directed_boundary(oldpatch);candidate_faces=newF[np.asarray(patch_ids,dtype=np.int64)]
  if len(candidate_faces)!=4:raise ValueError("patch expected four candidate triangles")
  patch_report={"old_face_ids":list(patch_ids),"boundary_cycle_columns":cycle,"candidate_faces_columns":candidate_faces.tolist(),"layers":{}}
  for L in (0,1):
   rows,_=cavity_cells(old,patch_ids,L)
   if len(set(mat[rows].tolist()))!=1:raise ValueError(f"patch {label} layer {L} crosses native material labels")
   oldcav=orient_tets_positive(old[rows],Xref);oldbnd=oriented_boundary(oldcav,Xref);old_side=side_boundary(oldbnd,cycle,L)
   old_lower_rel=cap_orientation(oldbnd,oldpatch,L);old_upper_rel=cap_orientation(oldbnd,oldpatch,L+1)
   expected_old_cap={tuple(sorted(map(int,tri+K*N))) for tri in oldpatch for K in (L,L+1)}
   if set(oldbnd)!=(set(old_side)|expected_old_cap):raise ValueError("old cavity has unexpected boundary facets")
   selected,enum_stats=patch_assignments(candidate_faces,old,Xref,Xcur,cycle,L,(old_lower_rel,old_upper_rel),old_side)
   if selected is None:raise RuntimeError(f"no cavity triangulation satisfies exact old boundary at {label} layer {L}: {enum_stats}")
   row_order=sorted(rows);tet_order=sorted(selected["tets"].tolist(),key=lambda q:tuple(sorted(map(int,q))))
   if len(row_order)!=len(tet_order):raise ValueError("cavity replacement cell count changed")
   for row,tet in zip(row_order,tet_order):
    cand_tets[row]=tet;changed_rows.append(row)
    cell_map.append({"patch":label,"layer":L,"output_tet_row":int(row),"inherited_old_tet_row":int(row),"new_tet_node_ids":list(map(int,tet)),"material_id":int(mat[row])})
   patch_report["layers"][str(L)]={"old_tet_rows":rows,"old_tet_count":len(rows),"material_id":int(mat[rows[0]]),"old_boundary_side_triangle_count":len(old_side),
     "old_cap_orientation_relative_to_surface":[int(old_lower_rel),int(old_upper_rel)],"enumeration":enum_stats,"selected_permutation_indices":list(selected["choice"]),
     "selected_reference_min_det":selected["reference_min_det"],"selected_current_min_det":selected["current_min_det"],"selected_current_nonpositive_cells":selected["current_nonpositive_count"],"candidate_tets":selected["tets"].tolist()}
  all_summary[label]=patch_report
 if len(set(changed_rows))!=len(changed_rows):raise ValueError("overlapping patch tet rows")
 if not np.array_equal(cand_tets[outside_tet_rows:=np.setdiff1d(np.arange(len(old)),changed_rows)],old[outside_tet_rows]):raise ValueError("outside cavity tetrahedra changed")
 # Corresponding surface connectivity on O, I, and D layers.
 faces_layers=[newF.copy(),(newF+N).copy(),(newF+2*N).copy()]
 d0,_,q0=fit.tet_metrics(Xref,old);dold,_,qold=fit.tet_metrics(Xcur,old);dnew,_,qnew=fit.tet_metrics(Xcur,cand_tets)
 drefnew=dets(Xref,cand_tets)
 report={"schema":"numi.human.d36-connected-oid-cavity-retessellation.v1","classification":"reference-positive-volume-topology_counterfactual_current_state_may_fail_fixed_ID_no_admission",
  "inputs":{"surface_candidate_npz":{"path":str(SURFACE_NPZ),"sha256":sha(SURFACE_NPZ)},"native_026_candidate_npz":{"path":str(NATIVE),"sha256":sha(NATIVE)},"AP_reference_npz":{"path":str(AP),"sha256":sha(AP)}},
  "domain":{"columns_per_layer":N,"face_patch_count_per_region":4,"prism_tets_per_face_per_layer":3,"old_cells_per_patch_per_layer":12,"changed_cell_count":len(changed_rows),"uses_actual_native_026_old_tetrahedra":True,
    "reference_geometry":"AP 033 full O/I/D point array","current_geometry":"same AP points with only O replaced by exact current294 source-target positions; I/D deliberately held at original AP positions"},
  "patches":all_summary,"full_volume_gates":{"reference_old_nonpositive":int(np.count_nonzero(d0<=0)),"reference_candidate_nonpositive":int(np.count_nonzero(drefnew<=0)),
    "reference_candidate_min_quality":float(fit.tet_metrics(Xref,cand_tets)[2].min()),"current_old_nonpositive_with_target_O_fixed_ID":int(np.count_nonzero(dold<=0)),
    "current_candidate_nonpositive_with_target_O_fixed_ID":int(np.count_nonzero(dnew<=0)),"current_candidate_min_det":float(dnew.min()),"current_candidate_min_quality":float(qnew.min()),
    "current_candidate_below_1e-5":int(np.count_nonzero(qnew<1e-5)),"nonpositive_current_tet_rows":np.flatnonzero(dnew<=0).astype(int).tolist(),"outside_cavity_old_connectivity_bitwise_unchanged":True},
  "limits":["Reference positivity and exact cavity boundary preservation are topology-construction checks only.","The current-state test applies target O with baseline I/D, not an optimized interior state; failures are retained for the parent connected-volume solver.","No mass remapping, constitutive, collision, full anatomy, native runtime, or physical admission is asserted. Cell-to-old-row pairing is deterministic bookkeeping, not conservative mass ancestry."],
  "runtime":{"hostname":socket.gethostname(),"python":sys.version,"numpy":np.__version__,"seconds":time.time()-t0}}
 output=OUT/"oid-cavity-replacement.npz"
 np.savez_compressed(output,reference_points_AP_m=Xref.astype(np.float64),target_points_AP_m=Xcur.astype(np.float64),old_tetrahedra=old.astype(np.int32),candidate_tetrahedra=cand_tets.astype(np.int32),
   old_material_ids=mat.astype(np.int8),candidate_material_ids=mat.astype(np.int8),old_O_faces=F0.astype(np.int32),old_I_faces=I0.astype(np.int32),old_D_faces=D0.astype(np.int32),
   candidate_O_faces=faces_layers[0].astype(np.int32),candidate_I_faces=faces_layers[1].astype(np.int32),candidate_D_faces=faces_layers[2].astype(np.int32),
   changed_tet_rows=np.asarray(sorted(changed_rows),dtype=np.int32),replacement_cell_map_json=np.asarray(json.dumps(cell_map,sort_keys=True)))
 report["output"]={"npz_path":str(output),"npz_sha256":sha(output)}
 (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"report":str(OUT/"report.json"),"npz":str(output),"sha256":report["output"]["npz_sha256"],"reference_nonpositive":report["full_volume_gates"]["reference_candidate_nonpositive"],
  "current_fixed_ID_nonpositive":report["full_volume_gates"]["current_candidate_nonpositive_with_target_O_fixed_ID"],"patches":{k:{L:{"assignments":v["enumeration"]["reference_feasible_assignments"],"current_bad":v["selected_current_nonpositive_cells"]} for L,v in x["layers"].items()} for k,x in all_summary.items()}},indent=2))
if __name__=="__main__":main()
