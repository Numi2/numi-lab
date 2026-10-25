from pathlib import Path
import json,hashlib,importlib.util,numpy as np
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
run=D/"nonmatching-layer-comparison-142/run-005-full146-baseline084"
out=run/"alternate-cap-diagonal-edge-1516-1562"
out.mkdir(exist_ok=False)
asm=D/"coupled-layer-trial-128/assemble_layers_128.py"
sp=importlib.util.spec_from_file_location("assembler",asm);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
C=run/"candidate.npz"; z=np.load(C); X=z["points_world_m"].astype(np.float32); fat=z["fat_shell_faces"].copy()
Fc,_=m.face_csv(m.DEFAULT_FACE_CSV_GZ); Fc=Fc[:,[0,2,1]]; n=8776; nc=2411
boundary=np.asarray([(v[0][1],v[0][2]) for v in m.edges_of_faces(Fc).values() if len(v)==1],dtype=np.int64).reshape((-1,2))
edge_row=41; a,b=map(int,boundary[edge_row]); assert (a,b)==(1516,1562)
base=2*len(Fc)+2*edge_row
assert fat[base].tolist()==[n+b,n+a,n+nc+a]
assert fat[base+1].tolist()==[n+b,n+nc+a,n+nc+b]
# Same quad boundary, alternate diagonal I_a--D_b, with oriented triangles.
fat[base]=[n+b,n+a,n+nc+b]
fat[base+1]=[n+a,n+nc+a,n+nc+b]
top=m.shell_topology(fat)
result=m.exact_self_audit(X,fat)
pairs=result.pop("pairs_array")
np.save(out/"fat-shell-pairs.npy",pairs)
np.save(out/"alternate-fat-shell-faces.npy",fat)
report={"input_candidate_sha256":hashlib.sha256(C.read_bytes()).hexdigest(),"original_fat_pairs_sha256":hashlib.sha256((run/"fat_shell-pairs.npy").read_bytes()).hexdigest(),"assembler_sha256":hashlib.sha256(asm.read_bytes()).hexdigest(),"boundary_edge_row":edge_row,"coarse_edge":[a,b],"alternate_diagonal_global_ids":[n+a,n+nc+b],"face_rows_replaced":[base,base+1],"old_face_rows":z["fat_shell_faces"][base:base+2].tolist(),"new_face_rows":fat[base:base+2].tolist(),"topology":top,"exact_audit":result,"pairs":pairs.tolist(),"geometry_changed":False,"accepted_physical_steps":0,"geometry_admitted":False}
report["source_script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
np.savez_compressed(out/"candidate-topology-only.npz",points_world_f32_m=X,outer_faces=z["outer_faces"],inner_faces=z["inner_faces"],deep_faces=z["deep_faces"],skin_shell_faces=z["skin_shell_faces"],fat_shell_faces=fat)
report["output_sha256"]=hashlib.sha256((out/"candidate-topology-only.npz").read_bytes()).hexdigest()
(out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
