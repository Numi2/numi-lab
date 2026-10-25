import json,hashlib,importlib.util
from pathlib import Path
import numpy as np
base=Path("/Users/numi/Work/numi-human-connected-surface-069/connected-layer-remesh-116")
inp=base/"run-001/candidate.npz"; out=base/"quality-repair-117/run-002/candidate-tetrahedra.npz"; rep=base/"quality-repair-117/run-002/report.json"
z0=np.load(inp);z1=np.load(out)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
changed={}
for k in ("reference_points_m","points_world_f32","output_node_ids","source_input_row","surface_faces_zero_based","surface_face_markers"):
 changed[k]=bool(np.array_equal(z0[k],z1[k]))
T0=np.asarray(z0["tetrahedra"],int);T1=np.asarray(z1["tetrahedra"],int)
M0=np.asarray(z0["material_ids"],int);M1=np.asarray(z1["material_ids"],int)
U0=np.asarray(z0["cell_uids"],int);U1=np.asarray(z1["cell_uids"],int)
d0={int(u):(tuple(map(int,t)),int(m)) for u,t,m in zip(U0,T0,M0)}
d1={int(u):(tuple(map(int,t)),int(m)) for u,t,m in zip(U1,T1,M1)}
common=set(d0)&set(d1)
unchanged=[u for u in common if d0[u]==d1[u]]
changed_existing=[u for u in common if d0[u]!=d1[u]]
removed=set(d0)-set(d1);added=set(d1)-set(d0)
spec=importlib.util.spec_from_file_location("m",base/"quality-repair-117/bistellar_connected_116.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
X=np.asarray(z0["reference_points_m"],float)
boundary_equal=m.boundary_chain(T0,X)==m.boundary_chain(T1,X)
vol0=m.dets(X,T0)/6;vol1=m.dets(X,T1)/6
material_vol={str(mid):{"before":float(vol0[M0==mid].sum()),"after":float(vol1[M1==mid].sum()),"delta":float(vol1[M1==mid].sum()-vol0[M0==mid].sum())} for mid in np.unique(M0)}
report=json.loads(rep.read_text())
payload={"input_npz":str(inp),"input_sha256":sha(inp),"run002_candidate":str(out),"candidate_sha256":sha(out),"as_run_report":str(rep),"as_run_report_sha256":sha(rep),
"checks":{"coordinates_and_existing_node_ids_bitwise_equal":all(changed[k] for k in ("reference_points_m","points_world_f32","output_node_ids","source_input_row")),"surface_faces_and_markers_bitwise_equal":changed["surface_faces_zero_based"] and changed["surface_face_markers"],"global_oriented_tet_boundary_equal":bool(boundary_equal),"cell_count_unchanged":len(T0)==len(T1),"unchanged_cell_uid_connectivity_material_rows":len(unchanged),"changed_existing_uids":sorted(changed_existing),"removed_cell_uids":sorted(removed),"added_cell_uids":sorted(added),"material_counts_before":{str(k):int(v) for k,v in zip(*np.unique(M0,return_counts=True))},"material_counts_after":{str(k):int(v) for k,v in zip(*np.unique(M1,return_counts=True))},"material_volume_m3":material_vol},
"report_scope_correction":{"invalid_inherited_claim":"run-002 report field invariants.O_I_D_faces_bitwise_unchanged=true is not independently checkable: the 116 input NPZ contains no source_O_faces/source_I_faces/source_D_faces arrays. Do not cite that field.","verified_replacement":"The input and output node coordinates/IDs and explicit surface_faces_zero_based/surface_face_markers are byte-identical; the complete oriented tet boundary is independently equal. Source O/I/D-specific face arrays were unavailable and therefore were not checked."},
"quality":{"baseline_bad_rows":report["baseline"]["bad_cell_rows"],"final_bad_rows":report["final"]["bad_cell_rows"],"q_floor":1e-5,"status":report["status"]},"classification":"connectivity-only diagnostic, not mass-map or physical/native admission"}
outp=Path("/Users/numi/Work/numi-human-connected-surface-069/connected-layer-remesh-116/quality-repair-117/run-002/report-correction.json")
outp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
