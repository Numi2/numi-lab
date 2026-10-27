from pathlib import Path
import csv,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
S=ROOT/"D/surface-clearance-546/component-sat-selector-638"
W=S/"per-bank-workers-20261027"
D=W/"b109-114/attempt-032-self16-joint-correction-001"
BASE=W/"combined-d071-b109-b61-001/candidate.npz"
CAND=D/"candidate-segment-partial-alpha-0p0625.npz"
BLAYER=W/"combined-d071-b109-b61-001/skin-fat-layers-541.npz"
RECON=ROOT/"connected-layer-reconstruction-127/matter/tools/human_skin_fat_layer_reconstruction.py"
TRANSFER=W/"b65-99/attempt-024-topology-independent-layer-transfer-001/transfer_topology_independent_541.py"
OWNER=W/"physical-layer-diagnostic-001/transfer_current.py"
FITTER=D/"solve_self16_joint.py"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def summarize(new,old,affected):
 out={}
 for k in ("authored_bad","nonpositive","conditioned"):
  a=np.asarray(old[k]);b=np.asarray(new[k])
  add=(b&~a);remove=(a&~b)
  if b.ndim==1:
   out[k]={"baseline_count":int(a.sum()),"candidate_count":int(b.sum()),"added_face_rows":np.flatnonzero(add).astype(int).tolist(),"removed_face_rows":np.flatnonzero(remove).astype(int).tolist()}
  else:
   out[k]={"baseline_count":int(a.sum()),"candidate_count":int(b.sum()),"added_face_tet_ids":[[int(x),int(y)] for x,y in np.argwhere(add)],"removed_face_tet_ids":[[int(x),int(y)] for x,y in np.argwhere(remove)]}
  out[k]["added_count"]=int(add.sum());out[k]["removed_count"]=int(remove.sum())
 return out
def metrics(O,I,Dp,F,skin, fat, transfer, owner):
 lf=np.asarray(F[:,[0,2,1]],np.int64)
 skinflags=transfer.face_flags(O,I,lf,skin)
 fatflags=transfer.face_flags(I,Dp,lf,fat)
 sm=owner.layer_metrics(O,I,lf,skin);fm=owner.layer_metrics(I,Dp,lf,fat)
 return {"skin_flags":skinflags,"fat_flags":fatflags,"skin_metrics":sm,"fat_metrics":fm}
def main():
 with np.load(BASE,allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
 with np.load(CAND,allow_pickle=False) as z:c={k:z[k].copy() for k in z.files}
 with np.load(BLAYER,allow_pickle=False) as z:l={k:z[k].copy() for k in z.files}
 x0=np.asarray(b["points_world_m"],np.float32);x1=np.asarray(c["points_world_m"],np.float32);F=np.asarray(b["source_O_faces"],np.int64)
 if not np.array_equal(F,np.asarray(c["source_O_faces"],np.int64)):raise ValueError("position-only candidate topology mismatch")
 if not np.array_equal(np.asarray(l["outer_faces"]),F[:,[0,2,1]]):raise ValueError("541 layer winding differs from candidate outer faces")
 seg=json.loads((D/"exact-segment-boundary-preflight.json").read_text())
 selfdoc=json.loads((D/"self16-pairs.json").read_text())
 pair_ids=np.asarray([r["skin_face_rows"] for r in selfdoc["pair_ids"]],np.int64)
 inside=[]
 with (W/"combined-d071-b109-b61-001/audit-130/query/pairs.csv").open(newline="") as f:
  for r in csv.DictReader(f):
   if (r["bank_body"],r["bank_stable_id"],r["classification"])==("109","114","inside"):inside.append(int(r["query_row"]))
 fitter=mod("layer_patch_fitter",FITTER)
 patch,components,boundary,support,edges=fitter.make_patch(F,seg,np.asarray(inside,np.int64),pair_ids)
 x=x1.astype(np.float64)
 O=x
 baseO=np.asarray(l["outer_points_f64"],np.float64);baseI=np.asarray(l["inner_points_f64"],np.float64);baseD=np.asarray(l["deep_points_f64"],np.float64)
 skin=np.asarray(l["skin_thickness_transferred_m"],np.float64);fat=np.asarray(l["fat_thickness_transferred_m"],np.float64)
 Icarry=O+(baseI-baseO);Dcarry=Icarry+(baseD-baseI)
 transfer=mod("layer_transfer_metrics",TRANSFER);owner=mod("layer_541_owner",OWNER);recon=mod("connected_layer_reconstruction",RECON)
 oldmetrics=metrics(baseO,baseI,baseD,F,skin,fat,transfer,owner)
 carrymetrics=metrics(O,Icarry,Dcarry,F,skin,fat,transfer,owner)
 boundary=np.asarray(boundary,np.int64);support=np.asarray(support,np.int64)
 out=recon.reconstruct_connected_layers(O,F[:,[0,2,1]],baseI-baseO,baseD-baseI,boundary,O[boundary],baseI[boundary],baseD[boundary],skin_depth_m=skin,fat_depth_m=fat,orientation_sign=1,active_vertex_indices=support)
 Ir=np.asarray(out.inner_points,np.float64);Dr=np.asarray(out.deep_points,np.float64)
 recmetrics=metrics(O,Ir,Dr,F,skin,fat,transfer,owner)
 affected=np.flatnonzero(np.any(np.isin(F,support),axis=1))
 results={}
 for name,mm in (("offset_carry",carrymetrics),("local_reconstruction",recmetrics)):
  results[name]={}
  for layer,flags,oldflags,meas,oldmeas in (("skin",mm["skin_flags"],oldmetrics["skin_flags"],mm["skin_metrics"],oldmetrics["skin_metrics"]),("fat",mm["fat_flags"],oldmetrics["fat_flags"],mm["fat_metrics"],oldmetrics["fat_metrics"])):
   flagsummary=summarize(flags,oldflags,affected)
   olddet=np.asarray(oldflags["normalized_det"]);newdet=np.asarray(flags["normalized_det"])
   af=affected
   delta=newdet[af]-olddet[af]
   results[name][layer]={"flags":flagsummary,"metrics":meas,"baseline_metrics":oldmeas,"affected_face_count":len(affected),"affected_normalized_det_min_baseline":float(olddet[af].min()),"affected_normalized_det_min_candidate":float(newdet[af].min()),"affected_normalized_det_min_delta":float(newdet[af].min()-olddet[af].min()),"affected_tet_det_deterioration_count":int(np.count_nonzero(delta<-1e-12)),"new_invalid_flags":bool(flagsummary["authored_bad"]["added_count"] or flagsummary["nonpositive"]["added_count"] or flagsummary["conditioned"]["added_count"])}
  results[name]["no_new_invalid_flags"]=not (results[name]["skin"]["new_invalid_flags"] or results[name]["fat"]["new_invalid_flags"])
  results[name]["zero_nonpositive_global"]=bool(results[name]["skin"]["metrics"]["nonpositive_f32_tet_count"]==0 and results[name]["fat"]["metrics"]["nonpositive_f32_tet_count"]==0)
  results[name]["strict_conditioning_global"]=bool(results[name]["skin"]["metrics"]["f32_conditioned_fail_count_lt_1e-5"]==0 and results[name]["fat"]["metrics"]["f32_conditioned_fail_count_lt_1e-5"]==0)
 diag=dict(out.diagnostics)
 # Save reconstructed layers while retaining provenance fields bitwise from 541.
 rlayer={k:v.copy() for k,v in l.items()}
 rlayer["outer_points_f64"]=O;rlayer["inner_points_f64"]=Ir;rlayer["deep_points_f64"]=Dr
 rlayer["outer_points_f32"]=O.astype(np.float32);rlayer["inner_points_f32"]=Ir.astype(np.float32);rlayer["deep_points_f32"]=Dr.astype(np.float32)
 rlayer["outer_faces"]=F[:,[0,2,1]];rlayer["inner_faces"]=F[:,[0,2,1]];rlayer["deep_faces"]=F[:,[0,2,1]]
 N=np.asarray(out.outward_normals,np.float64)
 rlayer["skin_thickness_actual_norm_m"]=np.einsum("ij,ij->i",O-Ir,N)
 rlayer["fat_thickness_actual_norm_m"]=np.einsum("ij,ij->i",Ir-Dr,N)
 layerout=D/"skin-fat-layers-local-reconstruction.npz"
 with layerout.open("xb") as f:np.savez_compressed(f,**rlayer)
 rep={"schema":"numi.human.b109-114.self16-joint-local-layer-reconstruction.v1","candidate_sha256":sha(CAND),"baseline_candidate_sha256":sha(BASE),"baseline_layer_sha256":sha(BLAYER),"reconstruction_owner":str(RECON),"reconstruction_owner_sha256":sha(RECON),"prism_metric_owner":str(OWNER),"prism_metric_owner_sha256":sha(OWNER),"transfer_helper":str(TRANSFER),"transfer_helper_sha256":sha(TRANSFER),"support_node_count":len(support),"patch_face_count":len(patch),"component_face_counts":[len(c0) for c0 in components],"fixed_boundary_count":len(boundary),"affected_layer_face_count":len(affected),"reconstruction_diagnostics":diag,"layer_checks":results,"reconstructed_layer_path":str(layerout),"reconstructed_layer_sha256":sha(layerout),"typed_layer_provenance_equal_baseline":all(np.array_equal(rlayer[k],l[k],equal_nan=True) for k in l if k not in ("outer_points_f64","inner_points_f64","deep_points_f64","outer_points_f32","inner_points_f32","deep_points_f32","outer_faces","inner_faces","deep_faces","skin_thickness_actual_norm_m","fat_thickness_actual_norm_m")),"geometry_admitted":False,"physical_steps":0}
 p=D/"layer-reconstruction-and-cell-report.json";p.write_text(json.dumps(rep,indent=2,sort_keys=True,default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x)+"\n")
 print(json.dumps({"report":str(p),"sha256":sha(p),"layer_file":str(layerout),"layer_sha256":sha(layerout),"diagnostics":diag,"offset_carry":{"no_new_invalid":results["offset_carry"]["no_new_invalid_flags"],"skin_added":results["offset_carry"]["skin"]["flags"]["authored_bad"]["added_count"],"fat_added":results["offset_carry"]["fat"]["flags"]["authored_bad"]["added_count"],"skin_nonpositive_added":results["offset_carry"]["skin"]["flags"]["nonpositive"]["added_count"],"fat_nonpositive_added":results["offset_carry"]["fat"]["flags"]["nonpositive"]["added_count"],"skin_conditioned_added":results["offset_carry"]["skin"]["flags"]["conditioned"]["added_count"],"fat_conditioned_added":results["offset_carry"]["fat"]["flags"]["conditioned"]["added_count"]},"reconstruction":{"no_new_invalid":results["local_reconstruction"]["no_new_invalid_flags"],"skin_added":results["local_reconstruction"]["skin"]["flags"]["authored_bad"]["added_count"],"fat_added":results["local_reconstruction"]["fat"]["flags"]["authored_bad"]["added_count"],"skin_nonpositive_added":results["local_reconstruction"]["skin"]["flags"]["nonpositive"]["added_count"],"fat_nonpositive_added":results["local_reconstruction"]["fat"]["flags"]["nonpositive"]["added_count"],"skin_conditioned_added":results["local_reconstruction"]["skin"]["flags"]["conditioned"]["added_count"],"fat_conditioned_added":results["local_reconstruction"]["fat"]["flags"]["conditioned"]["added_count"],"normalized_det_min_delta":[results["local_reconstruction"]["skin"]["affected_normalized_det_min_delta"],results["local_reconstruction"]["fat"]["affected_normalized_det_min_delta"]]}},indent=2))
if __name__=="__main__":main()

