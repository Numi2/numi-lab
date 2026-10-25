import csv, hashlib, importlib.util, json, pathlib, numpy as np
D=pathlib.Path("/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded")
B=D/"build_whole_reference_plc_083.py"
def sha(p):
 h=hashlib.sha256();
 with pathlib.Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
spec=importlib.util.spec_from_file_location("b",B); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
R,X,gids,T0,M0,O,I,Df,inc0,detR0,detX0,qR0,qX0,J0=b.load_inputs()
rep=json.loads((D/"s128-ancestry-and-cell-audit-v2.json").read_text())
run=json.loads((D/"tetgen-run.json").read_text())
# Parse TetGen output while retaining its original node identifiers.
lines=(D/"whole-skin-fat-reference.1.node").read_text().splitlines(); n,dim,nattr,nmark=map(int,lines[0].split()[:4]); P={}; marks={}
for line in lines[1:]:
 q=line.split()
 if not q or not q[0][0].isdigit():continue
 k=int(q[0]); P[k]=np.asarray(list(map(float,q[1:4])),dtype=np.float64); marks[k]=int(q[4])
ids=np.asarray(sorted(P),dtype=np.int64); idrow={int(k):i for i,k in enumerate(ids)}
coord_to_src={tuple(map(float,p)):i for i,p in enumerate(R)}
src=np.asarray([coord_to_src.get(tuple(map(float,P[int(k)])),-1) for k in ids],dtype=np.int64)
assert np.count_nonzero(src>=0)==len(R)==7233 and len(P)-len(R)==128
reference=np.asarray([P[int(k)] for k in ids],dtype=np.float64)
current=np.empty_like(reference,dtype=np.float64)
for j,r in enumerate(src):
 if r>=0: current[j]=X[int(r)]
else_nodes=rep["used_new_node_ancestry"]
ancestry_nodes=[]; ancestry_material=[]; ancestry_tet=[]; ancestry_weights=[]
marker_by_node={int(k):set() for k in ids if src[idrow[int(k)]]<0}
face_lines=(D/"whole-skin-fat-reference.1.face").read_text().splitlines(); nf,nfm=map(int,face_lines[0].split()[:2]); faces=[]; fmarkers=[]
for line in face_lines[1:]:
 q=line.split()
 if not q or not q[0][0].isdigit():continue
 tri=list(map(int,q[1:4])); marker=int(q[4]); faces.append([idrow[k] for k in tri]);fmarkers.append(marker)
 for k in tri:
  if k in marker_by_node: marker_by_node[k].add(marker//100000)
assert len(faces)==nf
for k in sorted(marker_by_node):
 row=idrow[k]; a=else_nodes[str(k)]
 assert a["interpolated_current_xyz_m"] is not None
 current[row]=np.asarray(a["interpolated_current_xyz_m"],dtype=np.float64)
 for mat,witness in a["source_tet_by_material"].items():
  if witness is None:continue
  ancestry_nodes.append(row); ancestry_material.append(int(mat)); ancestry_tet.append(int(witness["source_tet_row"])); ancestry_weights.append(witness["barycentric"])
# Parse tet connectivity and material attributes.
ele=(D/"whole-skin-fat-reference.1.ele").read_text().splitlines(); nt,nv,nattr=map(int,ele[0].split()[:3]); T=[]; M=[]
for line in ele[1:]:
 q=line.split()
 if not q or not q[0][0].isdigit():continue
 T.append([idrow[int(x)] for x in q[1:5]]); M.append(int(round(float(q[5]))))
T=np.asarray(T,dtype=np.int32); M=np.asarray(M,dtype=np.int8)
assert len(T)==nt and set(M.tolist())=={1,2}
current_f32=current.astype(np.float32)
ancnodes=np.asarray(ancestry_nodes,dtype=np.int32); ancmat=np.asarray(ancestry_material,dtype=np.int8); anctet=np.asarray(ancestry_tet,dtype=np.int32); ancw=np.asarray(ancestry_weights,dtype=np.float64)
npz=D/"whole-skin-fat-reference-084-candidate.npz"
np.savez_compressed(npz,output_node_ids=ids,reference_points_m=reference,current035_points_m_f64=current,current035_points_m_f32=current_f32,source_input_row=src,source_global_node_ids=gids,source_reference_points_m=R,source_current035_points_m=X, tetrahedra_zero_based=T,material_ids=M,surface_faces_zero_based=np.asarray(faces,dtype=np.int32),surface_face_markers=np.asarray(fmarkers,dtype=np.int32),newnode_ancestry_output_row=ancnodes,newnode_ancestry_material_id=ancmat,newnode_ancestry_source_tet_row=anctet,newnode_ancestry_barycentric=ancw,source_tetrahedra_zero_based=T0,source_material_ids=M0,source_O_faces=O,source_I_faces=I,source_D_faces=Df)
# Ensure generated ancestry directly reconstructs the saved reference/current position.
maxref=maxcur=0.0
for outrow,mat,trow,w in zip(ancnodes,ancmat,anctet,ancw):
 tri=T0[trow]; maxref=max(maxref,float(np.max(np.abs(w@R[tri]-reference[outrow])))); maxcur=max(maxcur,float(np.max(np.abs(w@X[tri]-current[outrow]))))
assert maxref<1e-12 and maxcur<1e-12
summary={"status":"BASELINE_REMESH_DIAGNOSTIC_ONLY_NOT_ADMITTED","candidate_npz":npz.name,"candidate_npz_sha256":sha(npz),"candidate_arrays":{k:list(v.shape) for k,v in np.load(npz).items()},"output_node_count":len(ids),"original_input_nodes":int(np.count_nonzero(src>=0),),"added_nodes":int(np.count_nonzero(src<0)),"added_boundary_nodes":int(sum(bool(marker_by_node[int(k)]) for k in ids if src[idrow[int(k)]]<0)),"added_interior_nodes":int(sum(not marker_by_node[int(k)] for k in ids if src[idrow[int(k)]]<0)),"added_node_marker_namespace_counts":{str(ns):int(sum(ns in marker_by_node[int(k)] for k in ids if src[idrow[int(k)]]<0)) for ns in (1,2,3,4)},"tet_count":len(T),"material_counts":{str(m):int(np.count_nonzero(M==m)) for m in (1,2)},"parent_ancestry_entries":len(ancnodes),"max_source_reference_reconstruction_error_m":maxref,"max_source_current035_reconstruction_error_m":maxcur,"original_input_coordinates_exact":rep["output"]["all_7233_original_point_coordinates_preserved_exactly"],"surface_parent_chain_pass":rep["surface_domain"]["oriented_parent_chain_matches"],"F32_cell_gate":rep["cell_gates"]["reference_float32_current_float32_by_material"],"limitations":"This is a same-domain baseline remesh diagnostic. TetGen-generated points carry source-tet barycentric construction ancestry; this is not conservative mass transfer or a physical/native admission. Later current deformation must be transported through the source ancestry and independently gated."}
# Source pins required to replay the ancestry mapping and geometry.
root=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019")
pins={"builder_sha256":sha(B),"tetgen_runner_sha256":sha(D/"tetgen084_runner.py"),"audit_v1_sha256":sha(D/"audit_tetgen084.py"),"audit_v2_sha256":sha(D/"audit_tetgen084_v2.py"),"poly_sha256":sha(D/"whole-skin-fat-reference.poly"),"facet_marker_map_sha256":sha(D/"facet-marker-map.csv"),"region_seeds_sha256":sha(D/"region-seeds.json"),"tetgen_binary_sha256":sha(run["command"][0]),"native_tet_csv_sha256":sha(b.NATIVE_TETS),"current035_csv_sha256":sha(b.CURRENT),"OID_npz_sha256":sha(b.OID),"native_array_expected_sha256":b.EXPECTED_NATIVE_TET_CSV_SHA}
summary["source_pins"]=pins
(D/"whole-skin-fat-reference-084-candidate-report.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
(D/"whole-skin-fat-reference-084-candidate-pins.json").write_text(json.dumps(pins,indent=2,sort_keys=True)+"\n")
print(json.dumps(summary,indent=2,sort_keys=True))
