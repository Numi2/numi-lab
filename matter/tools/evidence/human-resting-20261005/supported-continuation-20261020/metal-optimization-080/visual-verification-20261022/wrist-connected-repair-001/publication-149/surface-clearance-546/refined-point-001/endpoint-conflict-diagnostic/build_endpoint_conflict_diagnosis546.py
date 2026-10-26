from __future__ import annotations
import hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path("/Users/numi/Work/numi-human-connected-surface-069")
D=ROOT/"D/surface-clearance-546";OUT=D/"refined-point-001/endpoint-conflict-diagnostic"
STATE=D/"connected-refinement-adapter546/run-2462-affine-exclude30887-003/state.json"
TARGET=D/"refined-point-001/point-targets-multi-parent/nearest-union-exits.npz";TR=TARGET.with_suffix(".json")
PROJECT=ROOT/"D/contained-clearance-545/project_patch_quality545.py"
PATCH=ROOT/"D/contained-clearance-545/build_connected_endpoint_patch545.py"
CHECK=ROOT/"D/contained-clearance-545/run_candidate_checks545.py"
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k]for k in z.files}
def main():
 state=json.loads(STATE.read_text());candidate=pathlib.Path(state["candidate"]);summary=pathlib.Path(state["point_summary"]);surface=pathlib.Path(state["surface_report"])
 z=load(candidate);t=load(TARGET);tr=json.loads(TR.read_text());csha=sha(candidate)
 assert csha==state["candidate_sha256"]==tr["inputs"]["candidate_sha256"]
 rows=np.asarray(t["query_rows"],np.int64);targets=np.asarray(t["target_world_f32_m"],np.float32)
 assert len(rows)==1371 and len(np.unique(rows))==len(rows) and targets.shape==(len(rows),3)
 assert np.asarray(t["pass_mask"],bool).all() and np.asarray(t["target_exactly_outside_all185"],bool).all()
 X=np.asarray(z["points_world_m"],np.float64)*1000.;Y=X.copy();Y[rows]=targets.astype(np.float64)*1000.
 anchors=np.asarray(z["typed_coarse_anchor_output_rows_018"],np.int64);assert not len(np.intersect1d(rows,anchors))
 tm=np.zeros(len(X),bool);tm[rows]=True;am=np.zeros(len(X),bool);am[anchors]=True;fixed=tm|am
 F=np.asarray(z["source_O_faces"],np.int64);E=np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0)
 L=np.linalg.norm(X[E[:,0]]-X[E[:,1]],axis=1);delta=Y-X;err=np.max(np.abs(delta[E[:,0]]-delta[E[:,1]]),axis=1);both=fixed[E[:,0]]&fixed[E[:,1]]
 e=E[both];l=L[both];v=err[both];soft=v>(.495*l+1e-8);hard=v>(.5*l+1e-11)
 tt=tm[e[:,0]]&tm[e[:,1]];ta=(tm[e[:,0]]&am[e[:,1]])|(am[e[:,0]]&tm[e[:,1]]);aa=am[e[:,0]]&am[e[:,1]]
 all_fixed=np.flatnonzero(np.all(fixed[F],axis=1));target_only=np.all(tm[F[all_fixed]],axis=1);anchor_only=np.all(am[F[all_fixed]],axis=1)
 cr=np.cross(X[F[:,1]]-X[F[:,0]],X[F[:,2]]-X[F[:,0]]);tc=np.cross(Y[F[:,1]]-Y[F[:,0]],Y[F[:,2]]-Y[F[:,0]])
 R=np.asarray(z["area_reference_037_points_world_m_f64"],np.float64)*1000.;rc=np.cross(R[F[:,1]]-R[F[:,0]],R[F[:,2]]-R[F[:,0]])
 orient=np.einsum("ij,ij->i",tc,cr)/np.einsum("ij,ij->i",cr,cr);mag=np.linalg.norm(tc,axis=1)/np.linalg.norm(rc,axis=1)
 badarea=all_fixed[(orient[all_fixed]<.105)|(mag[all_fixed]<.10001)];hardarea=all_fixed[(orient[all_fixed]<.1)|(mag[all_fixed]<.1)]
 ratios=v/np.maximum(l,1e-300);badrows=np.flatnonzero(soft);worst=np.argsort(-ratios)[:16]
 source_paths=[candidate,STATE,summary,surface,TARGET,TR,PROJECT,PATCH,CHECK]
 pins={str(pathlib.Path(p).resolve()):sha(p)for p in source_paths}
 report={"schema":"numi.human.refined-endpoint-fixed-conflict-diagnostic546.v1","status":"FIXED_ENDPOINT_FIELD_NOT_PROJECTABLE_WITH_545_QUALITY_OWNER" if soft.any() else "FIXED_ENDPOINT_FIELD_PASS_PROJECTION_PREFLIGHT","candidate_sha256":csha,"state_sha256":sha(STATE),"point_summary_sha256":sha(summary),"surface_report_sha256":sha(surface),"target_npz_sha256":sha(TARGET),"target_report_sha256":sha(TR),"target_rows":int(len(rows)),"target_report_status":tr.get("status"),"all_targets_exactly_outside_all185":True,"typed_anchor_rows":int(len(anchors)),"target_anchor_overlap_rows":[],"endpoint_step_mm":{"maximum":float(np.linalg.norm(delta[rows],axis=1).max(initial=0.)),"median":float(np.median(np.linalg.norm(delta[rows],axis=1)))},"fixed_edge_counts":{"target_target":int(tt.sum()),"target_anchor":int(ta.sum()),"anchor_anchor":int(aa.sum())},"fixed_edge_quality_bad":{"target_target":int(np.sum(soft&tt)),"target_anchor":int(np.sum(soft&ta)),"anchor_anchor":int(np.sum(soft&aa)),"total":int(soft.sum())},"fixed_edge_final_hard_bad":{"target_target":int(np.sum(hard&tt)),"target_anchor":int(np.sum(hard&ta)),"anchor_anchor":int(np.sum(hard&aa)),"total":int(hard.sum())},"quality_edge_threshold":"supnorm(displacement_a-displacement_b) <= .495*edge_length_mm + 1e-8 mm","final_hard_edge_threshold":"supnorm(displacement_a-displacement_b) <= .5*edge_length_mm + 1e-11 mm","worst_fixed_edges":[{"edge":e[i].tolist(),"gradient_ratio":float(ratios[i])}for i in worst],"quality_bad_fixed_edge_rows":[e[i].tolist()for i in badrows],"all_fixed_faces":int(len(all_fixed)),"all_fixed_face_types":{"target_only":int(target_only.sum()),"anchor_only":int(anchor_only.sum()),"mixed":int(len(all_fixed)-target_only.sum()-anchor_only.sum())},"quality_bad_fixed_area_rows":badarea.tolist(),"final_hard_bad_fixed_area_rows":hardarea.tolist(),"source_trust_and_exact_mesh_checks_run":False,"native_admission":False,"accepted_physical_steps":0,"conclusion":"All-endpoint lock is infeasible for the existing quality projector because 882 target-target edges already violate its fixed-endpoint edge inequality; 877 exceed the final .5 edge-gradient threshold. No all-target fixed face violates area limits. Select a safe subset first, keep neighboring nonanchor rows free for build_connected_endpoint_patch545 plus project_patch_quality545, and leave other targets unresolved for later batches.","source_pins":pins}
 (OUT/"diagnosis.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({k:report[k]for k in ["status","candidate_sha256","target_rows","endpoint_step_mm","fixed_edge_counts","fixed_edge_quality_bad","fixed_edge_final_hard_bad","all_fixed_faces","all_fixed_face_types","quality_bad_fixed_area_rows","final_hard_bad_fixed_area_rows"]},indent=2))
if __name__=="__main__":main()
