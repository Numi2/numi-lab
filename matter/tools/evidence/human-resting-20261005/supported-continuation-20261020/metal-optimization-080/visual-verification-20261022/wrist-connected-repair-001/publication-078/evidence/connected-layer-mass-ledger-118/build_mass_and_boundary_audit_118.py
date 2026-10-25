#!/usr/bin/env python3
import csv,hashlib,json
from pathlib import Path
import numpy as np
R=Path("/Users/numi/Work/numi-human-connected-surface-069"); O=R/"connected-layer-mass-ledger-118"
P84=R/"whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz"
P115=R/"connected-capacity-reference-115/candidate.npz"; P116=R/"connected-layer-remesh-116/run-002/candidate.npz"
J115=R/"connected-capacity-reference-115/report.json"; J116=R/"connected-layer-remesh-116/run-002/cell-audit.json"
P114=R/"digit-source-diagnosis-110/batch-146-curved-source-midpoints-quality-114.json"
NATIVE=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-skin-parent-mass-transfer-067/compiled-cell-export-068/run-004/source-preflight/compiled-skin-fem-tets.csv")
DENSITY_BOUNDS={1:(900.,1100.),2:(850.,950.)}; MAT={15:1,16:2}; LABEL={1:"skin",2:"fat"}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def vol(points,tets):
 p=np.asarray(points,dtype=np.float64)[np.asarray(tets,dtype=np.int64)]
 return np.einsum("ij,ij->i",p[:,1]-p[:,0],np.cross(p[:,2]-p[:,0],p[:,3]-p[:,0]))/6
def surf(points,faces,name):
 f=np.asarray(faces,dtype=np.int64);p=np.asarray(points);n=len(p)
 rng=bool(f.size==0 or(f.min()>=0 and f.max()<n));finite=bool(np.isfinite(p).all())
 if not rng:return {"name":name,"face_count":len(f),"indices_in_range":False}
 sf=np.sort(f,axis=1);tri=p.astype(float)[f]
 areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
 de=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]));ce=np.sort(de,axis=1)
 ue,inv,cnt=np.unique(ce,axis=0,return_inverse=True,return_counts=True)
 coeff=np.bincount(inv,weights=np.where(de[:,0]<de[:,1],1,-1),minlength=len(ue))
 return {"name":name,"face_count":len(f),"point_count":n,"points_finite":finite,"indices_in_range":rng,
 "distinct_index_faces":int(np.count_nonzero(np.all(sf[:,1:]!=sf[:,:-1],axis=1))),
 "zero_area_faces":int(np.count_nonzero(areas==0)),"min_area_m2":float(areas.min()),
 "edges":len(ue),"edge_incidence_histogram":{str(int(k)):int(v) for k,v in zip(*np.unique(cnt,return_counts=True))},
 "boundary_edges":int(np.count_nonzero(cnt==1)),"nonmanifold_edges":int(np.count_nonzero(cnt>2)),
 "nonzero_raw_chain_coefficients":int(np.count_nonzero(coeff)),
 "closed_edge_incidence":bool(np.all(cnt==2)),"raw_oriented_chain_zero":bool(np.all(coeff==0)),
 "signed_surface_volume_m3":float(np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)}
def closed_shell(points,faces,name):
 f=np.asarray(faces,dtype=np.int64);p=np.asarray(points)
 de=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]));ce=np.sort(de,axis=1)
 ue,inv,cnt=np.unique(ce,axis=0,return_inverse=True,return_counts=True)
 signs=np.where(de[:,0]<de[:,1],1,-1).astype(np.int8)
 uses=[[] for _ in range(len(ue))]
 face_ids=np.tile(np.arange(len(f),dtype=np.int64),3)
 for face_id,(edge_id,sgn) in zip(face_ids,zip(inv,signs)): uses[int(edge_id)].append((int(face_id),int(sgn)))
 boundary=int(np.count_nonzero(cnt==1));nonman=int(np.count_nonzero(cnt>2))
 flip=np.zeros(len(f),dtype=np.int8);conflicts=0;components=0
 adjacency=[[] for _ in range(len(f))]
 if not boundary and not nonman:
  for edge_uses in uses:
   (fa,sa),(fb,sb)=edge_uses
   relation=-sa*sb
   adjacency[fa].append((fb,relation));adjacency[fb].append((fa,relation))
  for seed in range(len(f)):
   if flip[seed]:continue
   components+=1;flip[seed]=1;stack=[seed]
   while stack:
    a=stack.pop()
    for b,rel in adjacency[a]:
     want=flip[a]*rel
     if not flip[b]:flip[b]=want;stack.append(b)
     elif flip[b]!=want:conflicts+=1
 oriented=np.bincount(inv,weights=signs*np.tile(flip,3),minlength=len(ue)) if len(f) else np.zeros(len(ue))
 return {"name":name,"faces":int(len(f)),"vertices_referenced":int(len(np.unique(f))),
   "edge_count":int(len(ue)),"boundary_edges":boundary,"nonmanifold_edges":nonman,
   "closed_two_incidence":bool(np.all(cnt==2)),"orientability_conflicts":int(conflicts),
   "orientable_closed_boundary_chain_zero_after_consistent_orientation":bool(not boundary and not nonman and conflicts==0 and np.all(oriented==0)),
   "orientation_components":int(components)}
def cap_shell_sets(points,allfaces,markers,layers):
 caps=np.asarray(allfaces)[np.asarray(markers)//100000==4]
 cls=np.sort(caps//2411,axis=1)
 oi=caps[(cls[:,0]==0)&(cls[:,2]==1)]
 idcap=caps[(cls[:,0]==1)&(cls[:,2]==2)]
 unclassified=len(caps)-len(oi)-len(idcap)
 skin=np.concatenate((layers["O"],layers["I"][:,[0,2,1]],oi),axis=0)
 fat=np.concatenate((layers["I"],layers["D"][:,[0,2,1]],idcap),axis=0)
 return {"side_cap_faces_total":int(len(caps)),"O_I_side_cap_faces":int(len(oi)),
         "I_D_side_cap_faces":int(len(idcap)),"unclassified_side_cap_faces":int(unclassified),
         "skin_closed_boundary":closed_shell(points,skin,"skin O+reverse(I)+OI caps"),
         "fat_closed_boundary":closed_shell(points,fat,"fat I+reverse(D)+ID caps")}
def conservative_parent_transfer(pvol,pmat,prho,cvol,cmat,overlap,rtol=1e-10,atol=1e-14):
 """Transfer from parent-child overlap measures in m^3 (parent rows, child columns)."""
 from scipy import sparse
 pv=np.asarray(pvol,float);pm=np.asarray(pmat,int);rho=np.asarray(prho,float)
 cv=np.asarray(cvol,float);cm=np.asarray(cmat,int);A=sparse.csr_matrix(overlap,dtype=float)
 if pv.ndim!=1 or pm.shape!=pv.shape or rho.shape!=pv.shape or cv.ndim!=1 or cm.shape!=cv.shape or A.shape!=(len(pv),len(cv)):raise ValueError("shape")
 if not(np.isfinite(pv).all() and np.isfinite(cv).all() and np.isfinite(rho).all() and np.isfinite(A.data).all()):raise ValueError("finite")
 if np.any(pv<=0)|np.any(cv<=0)|np.any(rho<=0)|np.any(A.data<0):raise ValueError("positive measures required")
 i,j=A.nonzero()
 if np.any(pm[i]!=cm[j]):raise ValueError("cross-material overlap")
 if not np.allclose(np.asarray(A.sum(axis=1)).ravel(),pv,rtol=rtol,atol=atol):raise ValueError("parent measure not closed")
 child_mass=np.asarray(A.T@rho).ravel()
 by={}
 for m in sorted(set(pm)|set(cm)):
  old=float((pv[pm==m]*rho[pm==m]).sum());new=float(child_mass[cm==m].sum())
  if not np.isclose(old,new,rtol=rtol,atol=1e-12):raise ValueError("mass not closed")
  by[str(m)]={"parent_mass_kg":old,"child_mass_kg":new,"residual_kg":new-old}
 return {"child_mass_kg":child_mass,"child_density_kg_m3":child_mass/cv,"by_material":by}
def native_ledger():
 d={}
 with NATIVE.open(newline="") as f:
  for r in csv.DictReader(f):
   if int(r["identity_active_bit"])!=1:continue
   mid=int(r["material_index"])
   if mid not in MAT:continue
   m=MAT[mid];x=d.setdefault(m,{"rows":0,"volume":0.,"corner_mass":0.,"rhoV":0.,"rho":set(),"native_material_ids":set()})
   v=float(r["inverse_rest_w_m3"]);rho=float(r["compiled_bulk_x"])
   x["rows"]+=1;x["volume"]+=v;x["corner_mass"]+=4*float(r["bulk_density_corner_mass_kg"]);x["rhoV"]+=rho*v;x["rho"].add(rho);x["native_material_ids"].add(mid)
 for m,x in d.items():
  if len(x["rho"])!=1:raise RuntimeError("nonconstant compiled density")
  x["rho"]=next(iter(x["rho"]));x["native_material_ids"]=sorted(x["native_material_ids"]);x["mass_rounding_residual"]=x["corner_mass"]-x["rhoV"]
 return d
def row(m,oldv,newv,oldmass,rho):
 lo,hi=DENSITY_BOUNDS[m];req=oldmass/newv
 return {"material_id":m,"material":LABEL[m],"old_native_reference_volume_m3":oldv,"new_volume_m3":newv,
 "delta_volume_m3":newv-oldv,"delta_volume_percent":(newv/oldv-1)*100,"explicit_old_compiled_density_kg_m3":rho,
 "old_authoritative_compiled_mass_kg":oldmass,"mass_at_same_density_kg":newv*rho,"same_density_mass_delta_kg":newv*rho-oldmass,
 "required_aggregate_density_to_preserve_mass_kg_m3":req,"density_bounds_kg_m3":[lo,hi],
 "aggregate_preserving_mass_capacity_kg":[newv*lo,newv*hi],"aggregate_mass_feasible":lo<=req<=hi,
 "claim":"aggregate capacity only; no local conservative transfer implied"}
def main():
 a=np.load(P84,allow_pickle=False);b=np.load(P115,allow_pickle=False);c=np.load(P116,allow_pickle=False)
 v0=vol(a["reference_points_m"],a["tetrahedra_zero_based"]);v1=vol(b["proposed_reference_points_m"],b["tetrahedra"]);v2=vol(c["reference_points_m"],c["tetrahedra"])
 m0=a["material_ids"];m1=b["material_ids"];m2=c["material_ids"]
 if not(np.array_equal(a["tetrahedra_zero_based"],b["tetrahedra"]) and np.array_equal(m0,m1) and np.array_equal(a["cell_uids"],b["cell_uids"])):raise RuntimeError("115 does not preserve 084 cell rows")
 nd=native_ledger(); inp=[P84,P115,J115,P116,J116,P114,NATIVE]
 surfaces={k:surf(a["reference_points_m"],a[f"source_{k}_faces"],k) for k in ("O","I","D")}
 q114=json.loads(P114.read_text()); inc={}
 for fi,f in enumerate(a["source_O_faces"]):
  for x,y in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):inc.setdefault(tuple(sorted((int(x),int(y)))),[]).append(fi)
 bad=[]
 for ix,e in enumerate(q114["edges"]):
  ep=tuple(sorted(map(int,e["edge_output_rows"])));faces=inc.get(ep,[])
  p=np.asarray(e.get("registered_source_midpoint_m",[]),float)
  ok=len(faces)==2 and p.shape==(3,) and np.isfinite(p).all() and all(0<=q<len(a["reference_points_m"]) for q in ep)
  if not ok:bad.append({"proposal_row":ix,"edge":list(ep),"incident_O_faces":faces})
 mark={}
 for prefix,name in ((1,"O"),(2,"I"),(3,"D")):
  mask=(c["surface_face_markers"]//100000)==prefix
  mark[name]=surf(c["reference_points_m"],c["surface_faces_zero_based"][mask],f"116-{name}")
 shell084=cap_shell_sets(a["reference_points_m"],a["surface_faces_zero_based"],a["surface_face_markers"],
                         {k:a["source_"+k+"_faces"] for k in ("O","I","D")})
 shell116=cap_shell_sets(c["reference_points_m"],c["surface_faces_zero_based"],c["surface_face_markers"],
                         {k:c["surface_faces_zero_based"][c["surface_face_markers"]//100000==i] for i,k in ((1,"O"),(2,"I"),(3,"D"))})
 rep=json.loads(J116.read_text())["material_volume_m3"];agree={}
 led={}
 for mat in (1,2):
  new116=float(v2[m2==mat].sum());reportvol=float(rep[str(mat)])
  agree[str(mat)]={"computed_m3":new116,"frozen_report_m3":reportvol,"delta_m3":new116-reportvol}
  if abs(new116-reportvol)>max(1e-12,1e-9*abs(reportvol)):raise RuntimeError("116 volume report mismatch")
  led[str(mat)]={"115_same_topology":row(mat,nd[mat]["volume"],float(v1[m1==mat].sum()),nd[mat]["corner_mass"],nd[mat]["rho"]),
                 "116_remeshed":row(mat,nd[mat]["volume"],new116,nd[mat]["corner_mass"],nd[mat]["rho"])}
 report={"schema":"connected-layer-mass-ledger-118-v1",
  "input_sha256":{str(p):sha(p) for p in inp},
  "native_parent_mass_source":{"path":str(NATIVE),"active_tets_by_material":{str(m):nd[m]["rows"] for m in (1,2)},
    "material_id_mapping":{"native15":"skin candidate1","native16":"fat candidate2"},
    "density_kg_m3":{str(m):nd[m]["rho"] for m in (1,2)},
    "volume_m3":{str(m):nd[m]["volume"] for m in (1,2)},
    "mass_from_authoritative_corner_ledger_kg":{str(m):nd[m]["corner_mass"] for m in (1,2)},
    "density_times_compiled_inverse_rest_volume_kg":{str(m):nd[m]["rhoV"] for m in (1,2)},
    "rounding_residual_kg":{str(m):nd[m]["mass_rounding_residual"] for m in (1,2)}},
  "material_volume_mass":led,
  "115_mapping":{"same_084_connectivity_material_and_uid_rows":True,
    "nonpositive_proposed_cells_by_material":{str(m):int(np.count_nonzero(v1[m1==m]<=0)) for m in (1,2)},
    "interpretation":"Identity rows exist against the 084 remesh, not native compiled parent cells; the proposed reference has invalid cells, so it is not an admissible physical transfer map."},
  "116_mapping":{"cells":len(m2),"unique_cell_uids":bool(len(np.unique(c["cell_uids"]))==len(m2)),
    "native_parent_cell_csr_available":False,"geometric_parent_child_overlap_csr_available":False,
    "status":"BLOCKED_NO_NATIVE_PARENT_CELL_ANCESTRY_OR_GEOMETRIC_OVERLAP_CSR",
    "reason":"New TetGen cell rows and node source_input_row do not encode parent-cell mass ancestry."},
  "116_volume_crosscheck":agree,
  "boundary_topology":{"084_O_I_D":surfaces,"084_closed_material_shell_chains":shell084,
    "114_status":q114["status"],"114_is_geometry":False,"114_source_edge_proposals":len(q114["edges"]),
    "114_proposal_source_O_edge_and_finite_sample_checks_pass":not bad,
    "114_explicit_parent_O_face_rows_in_source_records":False,"114_invalid_proposals":bad,
    "116_marker_layer_surfaces":mark,"116_closed_material_shell_chains":shell116},
  "mass_transfer_helper":{"function":"conservative_parent_transfer in this source file",
    "input":"nonnegative parent-by-child geometric overlap measures in m^3",
    "requirements":["each parent row closes to parent reference volume","no cross-material overlap","mass closes by material"],
    "actual_116_map_supplied":False,"helper_applied_to_actual_candidate":False},
  "admission":{"density_rescaled":False,"conservative_transfer_claimed":False,"geometry_admitted":False,"accepted_physical_steps":0,
    "limitations":["115/116 volumes support only aggregate density-capacity arithmetic","no parent-cell overlap ancestry exists for 116","116 run-002 has three cells below q=1e-5","full material, detailed-skin, clearance, and native gates remain"]}}
 out=O/"mass-and-boundary-audit.json";out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"report":str(out),"report_sha256":sha(out),"material_volume_mass":led,
 "native_parent":report["native_parent_mass_source"],"084_layers":surfaces,"114_proposals":len(q114["edges"]),"114_invalid":len(bad),
 "116_layers":mark,"mapping_status":report["116_mapping"]["status"]},indent=2))
if __name__=="__main__":main()

