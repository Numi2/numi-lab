import csv,json,hashlib,pathlib,numpy as np
E=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
AP=E/"native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz"
APN=E/"native-candidate020-conforming-prism-030/source-preflight-006/initial-skin-fem-nodes.csv"
D=E/"connected-skin-parent-mass-transfer-067/compiled-cell-export-068"; C=D/"run-004/source-preflight"
a=np.load(AP)
with APN.open() as f: ar=sorted(csv.DictReader(f),key=lambda r:int(r["global_node"]))
with (C/"compiled-skin-fem-tets.csv").open() as f: cr=list(csv.DictReader(f))
with (C/"compiled-skin-fem-nodes.csv").open() as f: nr=sorted(csv.DictReader(f),key=lambda r:int(r["global_node"]))
P=np.array([[float(r[k]) for k in ("x_m","y_m","z_m")] for r in ar],np.float32)
N=np.array([float(r["mass_kg"]) for r in nr],np.float32)
Told=np.asarray(a["old_tetrahedra"],np.int64); Mold=np.asarray(a["old_material_ids"],np.int64)
Tnew=np.asarray(a["tetrahedra"],np.int64); Mnew=np.asarray(a["material_ids"],np.int64)
base=int(nr[0]["global_node"])
Tc=np.array([[int(r[f"global_node{i}"])-base for i in range(4)] for r in cr],np.int64)
Mc=np.array([int(r["material_index"]) for r in cr],np.int64)
W=np.array([float(r["inverse_rest_w_m3"]) for r in cr])
rho=np.array([float(r["compiled_bulk_x"]) for r in cr])
bulk={m:float(np.median(rho[Mc==m])) for m in np.unique(Mc)}
mp={1:15,2:16}
def vol(X,T):
 x=X[T];return np.linalg.det(np.stack([x[:,1]-x[:,0],x[:,2]-x[:,0],x[:,3]-x[:,0]],axis=2))/6
def assemble(T,M):
 V=vol(P,T); out=np.zeros(len(N),np.float64)
 for i,t in enumerate(T):out[t]+=abs(V[i])*bulk[mp[int(M[i])]]*.25
 return V,out.astype(np.float32)
Vo,mo=assemble(Told,Mold); Vn,mn=assemble(Tnew,Mnew)
do=(mo.astype(np.float64)-N.astype(np.float64))*1000; dn=(mn.astype(np.float64)-N.astype(np.float64))*1000
# Old T must be byte-order identical after row orientation to actual cooked row table.
ordered=bool(np.array_equal(Told,Tc))
# Compare W to old F32 geometry volume if aligned
out={"scope":"Counterfactual: rebuild node masses on old (pre-retriangulation) versus AP candidate retessellated T from identical actual AP fit-input F32 points and compiler material densities; compare to actual source-only cooked 039 node mass.","old_tetrahedra_ordered_equal_to_cooked":ordered,"old_cells":len(Told),"candidate_cells":len(Tnew),"new_candidate_unmatched_cell_rows":[11787,11788,11789,11884,11885,14310,14311,26103,26104,26105,26200,26201,28626,28627],"compiled_row0w_vs_old_fit_F32_volume_max_abs_m3":float(np.max(np.abs(W-np.abs(Vo)))) if ordered else None,"old_fit_F32_volume_total_m3":float(np.abs(Vo).sum()),"compiled_row0w_total_m3":float(W.sum()),"density_crosswalk":{"AP1_to_compiled15":bulk[15],"AP2_to_compiled16":bulk[16]},"oldT_fit_input_mass_vs_compiled_node_mass":{"max_abs_g":float(np.max(np.abs(do))),"nonzero_f32_nodes":int(np.count_nonzero(mo.view(np.uint32)!=N.view(np.uint32))),"sum_abs_g":float(np.abs(do).sum()),"worst_global_nodes":[{"global_node":int(i+base),"delta_g":float(do[i])} for i in np.argsort(np.abs(do))[::-1][:8]]},"newT_fit_input_mass_vs_compiled_node_mass":{"max_abs_g":float(np.max(np.abs(dn))),"nonzero_f32_nodes":int(np.count_nonzero(mn.view(np.uint32)!=N.view(np.uint32))),"sum_abs_g":float(np.abs(dn).sum()),"worst_global_nodes":[{"global_node":int(i+base),"delta_g":float(dn[i])} for i in np.argsort(np.abs(dn))[::-1][:8]]},"old_vs_new_node_mass_reallocation":{"max_abs_g":float(np.max(np.abs((mn.astype(np.float64)-mo.astype(np.float64))*1000))),"sum_signed_g":float(np.sum((mn.astype(np.float64)-mo.astype(np.float64))*1000)),"sum_abs_g":float(np.sum(np.abs(mn.astype(np.float64)-mo.astype(np.float64)))*1000)}}
p=D/"cooked-vs-ap-068-v2/counterfactual-old-vs-new-mass.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
