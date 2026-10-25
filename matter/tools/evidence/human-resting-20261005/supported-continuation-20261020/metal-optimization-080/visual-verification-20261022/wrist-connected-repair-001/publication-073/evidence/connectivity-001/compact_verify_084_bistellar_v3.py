import hashlib,importlib.util,json,pathlib,numpy as np
D=pathlib.Path("/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded")
B=D/"whole-skin-fat-reference-084-candidate.npz"
RUN=D/"connectivity-001/run-001"
RAW=RUN/"candidate-tetrahedra.npz"
SCRIPT=D/"connectivity-001/bistellar_connected_084.py"
CANON=pathlib.Path("/Users/n/Work/numi-human-connected-surface-069/canonical/matter/tools/human_skin_fat_connected_repair.py")
def sha(p):
 h=hashlib.sha256();
 with pathlib.Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
spec=importlib.util.spec_from_file_location("fit",SCRIPT);fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit)
rs=importlib.util.spec_from_file_location("repair",CANON);repair=importlib.util.module_from_spec(rs);rs.loader.exec_module(repair)
z0=np.load(B);z=np.load(RAW)
R0=z0["reference_points_m"];T0=z0["tetrahedra_zero_based"].astype(np.int64);M0=z0["material_ids"].astype(np.int64)
R=z["reference_points_m"];X=z["current037_points_m_f32"];T=z["tetrahedra_zero_based"].astype(np.int64);M=z["material_ids"].astype(np.int64)
assert np.array_equal(R,R0) and np.array_equal(z["surface_faces_zero_based"],z0["surface_faces_zero_based"]) and np.array_equal(z["surface_face_markers"],z0["surface_face_markers"])
used=np.unique(T.ravel());assert len(used)==len(R)-1
removed=set(range(len(R)))-set(map(int,used));assert removed=={7237},removed
keep=used.astype(np.int64);remap=np.full(len(R),-1,dtype=np.int64);remap[keep]=np.arange(len(keep))
Tc=remap[T];assert np.all(Tc>=0)
fc=remap[z["surface_faces_zero_based"]];assert np.all(fc>=0)
ancestry_keep=z["newnode_ancestry_output_row"]!=7237
ancrow=z["newnode_ancestry_output_row"][ancestry_keep].astype(np.int64);ancrow=remap[ancrow];assert np.all(ancrow>=0)
out=RUN/"candidate-tetrahedra-compacted.npz"
np.savez_compressed(out,reference_points_m=R[keep],current037_points_m_f32=X[keep],tetrahedra_zero_based=Tc.astype(np.int32),material_ids=M.astype(np.int8),cell_uids=z["cell_uids"],output_node_ids=z["output_node_ids"][keep],source_input_row=z["source_input_row"][keep],source_global_node_ids=z["source_global_node_ids"],surface_faces_zero_based=fc.astype(np.int32),surface_face_markers=z["surface_face_markers"],source_O_faces=z["source_O_faces"],source_I_faces=z["source_I_faces"],source_D_faces=z["source_D_faces"],newnode_ancestry_output_row=ancrow.astype(np.int32),newnode_ancestry_material_id=z["newnode_ancestry_material_id"][ancestry_keep],newnode_ancestry_source_tet_row=z["newnode_ancestry_source_tet_row"][ancestry_keep],newnode_ancestry_barycentric=z["newnode_ancestry_barycentric"][ancestry_keep])
# Independent whole-mesh cell and per-material oriented-boundary checks.
R32=R[keep].astype(np.float32).astype(np.float64);X32=X[keep].astype(np.float32).astype(np.float64);M=M.astype(np.int64)
Jr,_,qr,_=repair.cell_values_gradients(R32,R32,Tc);Jx,_,qx,_=repair.cell_values_gradients(X32,R32,Tc)
assert np.all(qr>=1e-5) and np.all(qx>=1e-5) and np.all(Jx>=.35) and np.all(Jx<=2.5)
def stable_boundary_chain(tets,points,stable_ids):
 oriented=fit.orient_ref(tets,points);chain={}
 for a,b,c,d in oriented:
  for face in ((b,c,d),(a,d,c),(a,b,d),(a,c,b)):
   stable=tuple(int(stable_ids[int(v)]) for v in face);key=tuple(sorted(stable));chain[key]=chain.get(key,0)+fit.sign3(stable)
 return {k:v for k,v in chain.items() if v}
chains={};vols={};oldvol={}
for mat in (1,2):
 before=stable_boundary_chain(T0[M0==mat],R0,z0["output_node_ids"])
 after=stable_boundary_chain(Tc[M==mat],R[keep],z["output_node_ids"][keep])
 chains[str(mat)]={"equal":before==after,"baseline_faces":len(before),"candidate_faces":len(after),"baseline_chain_sha256":hashlib.sha256(json.dumps(sorted((list(k),v) for k,v in before.items()),separators=(",",":")).encode()).hexdigest(),"candidate_chain_sha256":hashlib.sha256(json.dumps(sorted((list(k),v) for k,v in after.items()),separators=(",",":")).encode()).hexdigest()}
 vols[str(mat)]=float(fit.dets(R[keep],Tc[M==mat]).sum()/6)
 oldvol[str(mat)]=float(fit.dets(R0,T0[M0==mat]).sum()/6)
# Full incidence, duplicate-cell and surface record checks.
faceinc,edgeinc,owners=fit.make_topology(Tc,R[keep]); nonmanifold=int(sum(len(v)>2 for v in faceinc.values()))
facet_keys={tuple(sorted(map(int,f))) for f in fc}; all_material_boundary=set()
for f,inc in faceinc.items():
 if len(inc)==1 or (len(inc)==2 and M[inc[0][0]]!=M[inc[1][0]]):all_material_boundary.add(f)
assert facet_keys==all_material_boundary and nonmanifold==0
assert len({tuple(sorted(map(int,t))) for t in Tc})==len(Tc)
uid=z["cell_uids"].astype(np.int64)
base_ids=z0["output_node_ids"]
candidate_ids=z["output_node_ids"][keep]
base_by_uid={int(i):(tuple(sorted(int(base_ids[v]) for v in T0[i])),int(M0[i])) for i in range(len(T0))}
for row,cell_uid in enumerate(uid):
 if int(cell_uid) in base_by_uid:
  base_key,base_mat=base_by_uid[int(cell_uid)]
  assert tuple(sorted(int(candidate_ids[v]) for v in Tc[row]))==base_key and int(M[row])==base_mat
assert all(v["equal"] for v in chains.values())
source_rows=z["source_input_row"][keep];assert np.count_nonzero(source_rows>=0)==7233 and np.count_nonzero(source_rows<0)==127
assert 7238 not in set(z["output_node_ids"][keep].tolist()) # removed zero-based row 7237 had original output ID 7238
report={"status":"INDEPENDENT_COMPACTED_CANDIDATE_GATES_PASS; NOT_PHYSICAL_ADMISSION","raw_candidate_sha256":sha(RAW),"compacted_candidate_sha256":sha(out),"bistellar_source_sha256":sha(SCRIPT),"canonical_owner_sha256":sha(CANON),"removed_output_row_zero_based":7237,"removed_tetgen_output_node_id":7238,"nodes_before":len(R),"nodes_after":len(keep),"original_nodes_exact":int(np.count_nonzero(source_rows>=0)),"generated_nodes_after_compaction":int(np.count_nonzero(source_rows<0)),"cells":len(Tc),"material_counts":{str(m):int(np.count_nonzero(M==m)) for m in (1,2)},"F32":{"reference_nonpositive":0,"current_nonpositive":0,"reference_q_below_floor":int(np.sum(qr<1e-5)),"current_q_below_floor":int(np.sum(qx<1e-5)),"reference_q_min":float(qr.min()),"current_q_min":float(qx.min()),"J_min":float(Jx.min()),"J_max":float(Jx.max())},"per_material_oriented_boundary_chains_equal_to_084_baseline_by_stable_output_node_id":chains,"reference_volume_m3_by_material":vols,"baseline_reference_volume_m3_by_material":oldvol,"relative_volume_delta_by_material":{k:(vols[k]-oldvol[k])/oldvol[k] for k in vols},"whole_boundary_facets_equal_surface_records":facet_keys==all_material_boundary,"nonmanifold_faces_gt2":nonmanifold,"duplicate_tets":len(Tc)-len({tuple(sorted(map(int,t))) for t in Tc}),"surface_faces_and_markers_unchanged_from_084":True,"limitations":"Connectivity-only topology counterfactual. F32 q/J, cell orientation, material domains and marked boundaries pass. This does not prove mass conservation, local mechanical energy/stress, clearance or native physical admission."}
# The remap changes row indices, so verify all marked boundary triangles by converting both tables back to stable TetGen output IDs.
assert np.array_equal(np.sort(z["output_node_ids"][keep][fc],axis=1),np.sort(z["output_node_ids"][z["surface_faces_zero_based"]],axis=1))
assert np.array_equal(z["surface_face_markers"],z0["surface_face_markers"])
report["marked_surface_faces_and_markers_unchanged_by_stable_output_node_id"]=True
(RUN/"compacted-candidate-validation-v3.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2,sort_keys=True))
