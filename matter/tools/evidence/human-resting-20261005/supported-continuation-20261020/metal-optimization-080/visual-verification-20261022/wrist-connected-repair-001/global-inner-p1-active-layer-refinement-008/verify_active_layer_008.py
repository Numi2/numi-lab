# -*- coding: utf-8 -*-
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=ROOT/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
D=E/"global-inner-p1-active-layer-refinement-008";OUT=D/"independent-verification-001.json"
P5=E/"global-inner-p1-edge-support-403/quality-coordinate-005/candidate.npz"
P3=E/"global-inner-p1-edge-support-403/quality-bistellar-003/candidate-geometry-only.npz"
P2=E/"global-inner-p1-edge-support-403/build-002/candidate.npz"
P6=E/"global-inner-p1-edge-support-403/quality-transfer-006/local-overlap-005-to-002.npz"
P398=E/"global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/local-quality-cavities-398/candidate-with-transfer.npz"
BANK=E/"registered-surrounding-tissue-native013-191/registered-tissue-banks.npz"
CLIP=E/"robust-p1-construction-clip-owner-302/tools/human_skin_p1_obstacles.py"
OWNER=E/"global-inner-p1-field-owner-integration-007/field_owner_integration_007.py"
C=D/"candidate-with-transfer-partial-field.npz";R=D/"round-004.npz"
EXPL=D/"exploration-report.json";REP=D/"solver-and-transfer-report.json"
FILES=[P5,P3,P2,P6,P398,BANK,CLIP,OWNER,C,R,EXPL,REP,Path("/tmp/refine_active_layer_008.py"),Path("/tmp/finalize_active_layer_008.py")]
PIN={P5:"5b54ffd34d74d717579fcb14677cbf2bfba01f4d4562652605b9207e65f8f698",P3:"176900d57eedb40255298ad77860317ab97e31a6792bc4dc76c4d48ec7d060e8",P2:"9c8c23c8229c46494d19f6fc12d6131667b901864dfab21cd1ef7de6a2fc6744",P6:"875bb77bfa65486c6856b04d3b3ad4c4ffd8d471a2fc4297d6d2809a70f5ae41",BANK:"39deb100a99601dc48550c3e5bcddfaac7aea266525e389abbf349d1d8c47792",C:"b76368d95c95e74c780999391eb5785a6311c43879ac3ceb6d5c134494a1850b",R:"20a3d1a2d7c6454548fe9b47dc545c121d02746b625d3e9087c0be0103c16255"}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(x,m):
 if not bool(x):raise RuntimeError(m)
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def boundary(T):
 F=np.concatenate((T[:,[1,2,3]],T[:,[0,3,2]],T[:,[0,1,3]],T[:,[0,2,1]]));K=np.sort(F,axis=1)
 _,first,inv,cnt=np.unique(K,axis=0,return_index=True,return_inverse=True,return_counts=True)
 if np.any(cnt>2):raise ValueError("nonmanifold tetra face")
 q=np.argmin(F,axis=1);i=np.arange(len(F));sgn=np.where(F[i,(q+1)%3]<F[i,(q+2)%3],1,-1)
 if np.any(np.bincount(inv,weights=sgn,minlength=len(cnt))[cnt==2]!=0):raise ValueError("interior face orientation")
 return F[first[cnt==1]]
def omap(F):
 d={}
 for f in np.asarray(F,np.int64):
  q=int(np.argmin(f));c=tuple(map(int,[f[(q+j)%3] for j in range(3)]));k=tuple(sorted(c))
  if k in d:raise ValueError("duplicate typed face")
  d[k]=c
 return d
def metrics(X,T):
 P=np.asarray(X,np.float64)[T]
 d=np.linalg.det(np.stack((P[:,1]-P[:,0],P[:,2]-P[:,0],P[:,3]-P[:,0]),axis=2));m=np.zeros(len(T))
 for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):m=np.maximum(m,np.linalg.norm(P[:,i]-P[:,j],axis=1))
 return d,d/np.maximum(m,1e-300)**3
def main():
 if OUT.exists():raise SystemExit("refuse overwrite "+str(OUT))
 for p in FILES:req(p.is_file(),"missing "+str(p))
 for p,h in PIN.items():req(sha(p)==h,"pin mismatch "+str(p))
 z=np.load(C,allow_pickle=False);r=np.load(R,allow_pickle=False);b=np.load(P5,allow_pickle=False)
 X=z["points_world_m"];T=z["tetrahedra"].astype(np.int64);par=z["parent_005_tet_rows"].astype(np.int64)
 X0=b["points_world_m"];T0=b["tetrahedra"].astype(np.int64);N0=len(X0);M0=len(T0)
 req(X.shape==(55657,3) and X.dtype==np.float32 and T.shape==(249227,4),"unexpected output domain")
 req(np.array_equal(X,r["points_world_m"]) and np.array_equal(T,r["tetrahedra"]) and np.array_equal(par,r["parent_005_tet_rows"]),"candidate differs from frozen round004")
 req(np.array_equal(X[:N0],X0),"original 005 coordinates changed")
 ptr=z["node_source_indptr"].astype(np.int64);ids=z["node_source_indices"].astype(np.int64);w=z["node_source_weights"].astype(np.float64)
 pptr=b["node_source_indptr"].astype(np.int64);pids=b["node_source_indices"].astype(np.int64);pw=b["node_source_weights"].astype(np.float64)
 req(np.array_equal(ptr[:N0+1],pptr) and np.array_equal(ids[:pptr[-1]],pids) and np.array_equal(w[:pptr[-1]],pw),"original node source-CSR prefix changed")
 sums=np.add.reduceat(w,ptr[:-1]);req(ptr.shape==(len(X)+1,) and ptr[0]==0 and ptr[-1]==len(ids)==len(w) and np.all(np.diff(ptr)>0),"malformed node ancestry CSR")
 req(np.isfinite(w).all() and np.all(w>=0) and np.max(np.abs(sums-1))<2e-12 and np.all(np.isin(ids,np.unique(pids))),"appended source ancestry is not normalized within source domain")
 for k in ("physical_outer_node_ids","physical_outer_faces","construction_closure_faces","physical_shell_faces"):req(np.array_equal(z[k],b[k]),"typed surface array changed: "+k)
 O=z["physical_outer_node_ids"].astype(np.int64);req(np.all(O<N0),"new support node entered physical O map")
 sh=omap(b["physical_shell_faces"]);oo=omap(b["physical_outer_faces"]);cc=omap(b["construction_closure_faces"])
 req(set(oo).isdisjoint(cc) and set(oo)|set(cc)==set(sh),"typed O/closure faces do not partition shell")
 req(all(sh[k]==v for k,v in oo.items()) and all(sh[k]==v for k,v in cc.items()),"typed face orientation changed")
 req(omap(boundary(T))==sh,"actual tet boundary not exact oriented shell")
 d,q=metrics(X,T);d0,_=metrics(X0,T0)
 req(np.isfinite(d).all() and np.all(d>0) and q.min()>=1e-5,"positive/quality gate failed")
 req(par.shape==(len(T),) and np.all((par>=0)&(par<M0)) and np.array_equal(z["region_attributes"],b["region_attributes"][par]),"child material/parent mapping mismatch")
 v=d/6.;v0=d0/6.;tot=np.bincount(par,weights=v,minlength=M0);req(np.all(tot>0),"uncovered 005 tet parent")
 fr=v/tot[par];closure=np.bincount(par,weights=fr,minlength=M0)
 req(np.array_equal(z["transfer_new_to_005_offsets"],np.arange(len(T)+1)) and np.array_equal(z["transfer_new_to_005_parent_rows"],par),"005 transfer CSR row map mismatch")
 req(np.allclose(z["transfer_new_to_005_parent_fractions"],fr,rtol=2e-14,atol=2e-15) and np.max(np.abs(closure-1))<2e-12,"005 child-volume fractions mismatch")
 tr=np.load(P6,allow_pickle=False);ao=tr["transfer_new_to_014_offsets"].astype(np.int64);ar=tr["transfer_new_to_014_parent_rows"].astype(np.int64);af=tr["transfer_new_to_014_parent_fractions"].astype(np.float64);n14=int(ar.max())+1
 A=csr_matrix((af,ar,ao),shape=(M0,n14));B=csr_matrix((fr,par,np.arange(len(T)+1)),shape=(len(T),M0));Ccalc=(B@A).tocsr();Ccalc.sort_indices()
 Cstored=csr_matrix((z["transfer_new_to_014_parent_fractions"],z["transfer_new_to_014_parent_rows"],z["transfer_new_to_014_offsets"]),shape=(len(T),n14));Cstored.sort_indices()
 req(np.array_equal(Ccalc.indptr,Cstored.indptr) and np.array_equal(Ccalc.indices,Cstored.indices) and np.allclose(Ccalc.data,Cstored.data,rtol=2e-14,atol=2e-15),"005-to-006 material CSR composition mismatch")
 cols=np.asarray(Ccalc.sum(axis=0)).ravel();req(np.all(cols>0) and np.max(np.abs(cols-1))<2e-12,"014 parent columns fail closure")
 # Compose output child-to-005 tet ancestry through 005->003->002->398.
 p3=np.load(P3,allow_pickle=False);p2=np.load(P2,allow_pickle=False);p398=np.load(P398,allow_pickle=False);owner=load(OWNER,"ownerverify008")
 o5=b["parent_input_candidate_offsets"].astype(np.int64);r5=b["parent_input_candidate_rows"].astype(np.int64);o3=p3["parent_403_candidate_offsets"].astype(np.int64);r3=p3["parent_403_candidate_rows"].astype(np.int64);p2base=p2["parent_398_tet_row"].astype(np.int64)
 o52,r52=owner.compose_parent_csrs(o5,r5,o3,r3,child_count=M0,intermediate_count=len(p3["tetrahedra"]),base_count=len(p2["tetrahedra"]))
 o5to398,r5to398=owner.compose_parent_csrs(o52,r52,np.arange(len(p2["tetrahedra"])+1),p2base,child_count=M0,intermediate_count=len(p2["tetrahedra"]),base_count=len(p398["tetrahedra"]))
 bo=np.zeros(len(T)+1,np.int64);br=[]
 for i,p in enumerate(par):br.extend(r5to398[o5to398[p]:o5to398[p+1]].tolist());bo[i+1]=len(br)
 req(np.all(np.diff(bo)>0),"a child lacks composed 398 ancestry")
 # Reclip exact bank face 2562 against final mesh and compare constraints/solution.
 bank=np.load(BANK,allow_pickle=False);tri=bank["bank_0004_vertices"].astype(np.float32)[bank["bank_0004_faces"].astype(np.int64)[2562]]
 clip=load(CLIP,"clipverify008");cons,polys,meta=owner.clip_typed_triangle_to_mesh(X,T,tri,clip.triangle_tet_barycentric_polygon)
 rt=np.asarray([ti for ti,bb in cons for _ in bb],np.int64);rb=np.concatenate([bb for ti,bb in cons]);ct=z["clip_constraint_tet_ids"].astype(np.int64);cb=z["clip_constraint_tet_barycentric"].astype(np.float64)
 req(np.array_equal(rt,ct) and np.array_equal(rb,cb),"stored clip constraints differ from reclip")
 phi=z["partial_single_face_phi"].astype(np.float64);vals=np.einsum("ij,ij->i",rb,phi[T[rt]])
 req(len(vals)==440 and vals.min()>=.341-1e-8 and np.array_equal(phi[O],np.zeros(len(O))),"field rows fail or O moved")
 used=np.unique(T);refs=np.sort(z["solver_referenced_node_ids"].astype(np.int64));req(np.array_equal(np.sort(used),refs),"solver referenced-node map mismatch")
 req(phi.shape==(len(X),) and np.isfinite(phi).all() and phi.min()>=0 and phi.max()<=1 and np.all(phi[np.setdiff1d(np.arange(len(X)),used)]==0),"partial field bounds/unused-node values invalid")
 rep=json.loads(REP.read_text());expl=json.loads(EXPL.read_text())
 req(expl["final_support_pass"] and expl["records"][-1]["bad_capacity"]==0 and rep["solver"]["pass"] and rep["native_admission"] is False and rep["physical_admission"] is False,"source reports disagree with frozen state")
 out={"schema":"numi.human.active-layer-refinement-008.independent-verification.v1","status":"pass_one_face_preparation_only","admission":{"native":False,"physical":False,"all_689_banks":False},"pins":{str(p):sha(p) for p in FILES},"domain":{"input_005_nodes":N0,"input_005_tets":M0,"output_nodes":len(X),"output_tets":len(T),"added_nodes":len(X)-N0,"added_tets":len(T)-M0},"source_ancestry":{"original_coordinate_prefix_exact":True,"original_node_source_csr_prefix_exact":True,"all_rows_normalized_nonnegative":True,"new_source_ids_within_original_domain":True,"composed_child_to_398_entries":len(br),"max_398_parents_per_child":int(np.diff(bo).max())},"geometry":{"all_positive":True,"normalized_determinant_min":float(q.min()),"below_1e_5_count":int(np.count_nonzero(q<1e-5)),"max_raw_parent_volume_ratio_error":float(np.max(np.abs(tot/v0-1))),"child005_fraction_column_error":float(np.max(np.abs(closure-1))),"global_volume_delta_m3":float(v.sum()-v0.sum())},"typed_boundary":{"physical_outer_exact":True,"construction_closure_exact":True,"shell_exact_oriented_tet_boundary":True,"physical_outer_count":len(oo),"construction_closure_count":len(cc),"shell_count":len(sh)},"material_transfer":{"005_to_014_matches_independent_005x006_CSR":True,"014_column_closure_max_error":float(np.max(np.abs(cols-1))),"CSR_entry_count":int(len(Ccalc.data))},"one_face_field":{"bank":"bank_0004","face_row":2562,"reclipped_rows":len(vals),"AABB_candidate_tets":int(meta["aabb_candidate_tet_count"]),"intersected_tets":int(meta["intersected_tet_count"]),"minimum_free_capacity":float(rep["minimum_free_capacity"]),"target":float(rep["target"]),"minimum_replayed_constraint":float(vals.min()),"physical_O_exact_zero":True,"nonzero_nodes":int(np.count_nonzero(phi))},"limitations":["One typed face only; no all-bank or full tissue qualification.","Child-volume fractions are conservative transfer weights, not a fresh exact geometric overlap certificate.","No native or physical admission."]}
 OUT.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":out["status"],"report":str(OUT),"domain":out["domain"],"source":out["source_ancestry"],"geometry":out["geometry"],"boundary":out["typed_boundary"],"transfer":out["material_transfer"],"field":out["one_face_field"]},sort_keys=True))
if __name__=="__main__":main()
