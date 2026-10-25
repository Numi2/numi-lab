import hashlib,json,pathlib,numpy as np
D=pathlib.Path("/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded")
P=D/"whole-skin-fat-reference-084-candidate.npz"
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
z=np.load(P); R=z["reference_points_m"]; X64=z["current035_points_m_f64"]; X32=z["current035_points_m_f32"]; src=z["source_input_row"]; R0=z["source_reference_points_m"]; X0=z["source_current035_points_m"]; T=z["tetrahedra_zero_based"].astype(np.int64); M=z["material_ids"]; T0=z["source_tetrahedra_zero_based"].astype(np.int64); M0=z["source_material_ids"]
assert R.shape==X64.shape==X32.shape==(7361,3) and len(src)==7361 and np.count_nonzero(src>=0)==7233
assert np.array_equal(R[src>=0],R0[src[src>=0]]) and np.array_equal(X64[src>=0],X0[src[src>=0]])
assert np.array_equal(X32,X64.astype(np.float32)) and set(M.tolist())=={1,2} and T.shape==(32801,4)
assert len(np.unique(np.sort(T,axis=1),axis=0))==len(T) and len(set(T.ravel().tolist()))==len(R)
by_node={int(r):[] for r in np.flatnonzero(src<0)}
ancrows=z["newnode_ancestry_output_row"].astype(int); ancmat=z["newnode_ancestry_material_id"]; anctet=z["newnode_ancestry_source_tet_row"].astype(int); ancw=z["newnode_ancestry_barycentric"]
max_ref=max_cur=0.0
for row,mat,tr,w in zip(ancrows,ancmat,anctet,ancw):
 assert row in by_node and np.all(w>=-1e-12) and abs(float(w.sum())-1)<1e-12
 tri=T0[tr]; max_ref=max(max_ref,float(np.max(np.abs(w@R0[tri]-R[row])))); max_cur=max(max_cur,float(np.max(np.abs(w@X0[tri]-X64[row])))); by_node[row].append(int(mat))
assert all(by_node.values()) and max_ref<1e-12 and max_cur<1e-12
# Recompute normalized tet quality from the serialized F32 geometry, independently of the audit report.
def det_q(P,T):
 Q=P[T].astype(np.float64); d=np.einsum("ij,ij->i",np.cross(Q[:,1]-Q[:,0],Q[:,2]-Q[:,0]),Q[:,3]-Q[:,0]); e2=sum(np.einsum("ij,ij->i",Q[:,i]-Q[:,j],Q[:,i]-Q[:,j]) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))); q=12*np.maximum(d/2,0)**(2/3)/e2; return d,q
Dr,Qr=det_q(R.astype(np.float32),T); Dx,Qx=det_q(X32,T); J=Dx/Dr
assert np.all(Dr>0) and np.all(Dx>0) and np.all(Qr>=1e-5) and np.all(Qx>=1e-5) and np.all((J>=.35)&(J<=2.5))
report={"status":"PASS_SERIALIZED_CANDIDATE_INTEGRITY; NOT_PHYSICAL_ADMISSION","candidate_sha256":sha(P),"nodes":len(R),"original_nodes_exact":int(np.count_nonzero(src>=0)),"generated_nodes":int(np.count_nonzero(src<0)),"generated_nodes_with_source_tet_ancestry":len(by_node),"ancestry_material_entries":len(ancrows),"max_reference_reconstruction_error_m":max_ref,"max_current035_reconstruction_error_m":max_cur,"tetrahedra":len(T),"material_counts":{str(m):int(np.count_nonzero(M==m)) for m in (1,2)},"F32_reference_positive":bool(np.all(Dr>0)),"F32_current035_positive":bool(np.all(Dx>0)),"F32_qfloor_reference_min":float(Qr.min()),"F32_qfloor_current035_min":float(Qx.min()),"J_min":float(J.min()),"J_max":float(J.max()),"all_material_attrs_1_2":set(M.tolist())=={1,2},"all_nodes_used":len(set(T.ravel().tolist()))==len(R),"source_facet_parent_chain_pass_from_audit":json.loads((D/"s128-ancestry-and-cell-audit-v2.json").read_text())["surface_domain"]["oriented_parent_chain_matches"],"limitation":"Verifies serialized geometry and construction ancestry only; does not establish conservative mass transfer, tissue/bone clearance, mechanics or native physical admission."}
(D/"candidate-file-integrity.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2,sort_keys=True))
