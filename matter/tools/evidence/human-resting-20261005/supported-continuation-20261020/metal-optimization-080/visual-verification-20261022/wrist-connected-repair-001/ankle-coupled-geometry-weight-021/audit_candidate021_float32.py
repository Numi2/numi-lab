#!/usr/bin/env python3
"""Independent exact audit of 021 after source-position Float32 packing."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=ROOT/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
OUT=E/"ankle-coupled-geometry-weight-021"
PAYLOAD=E/"finger-profile-069i-left-ring-web-retreat/replacement-bilateral-hands038.nhskin"
NPZ=OUT/"ankle-coupled-021-candidate.npz"
MAN=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json")
sys.path.insert(0,"/Users/n/numi-human-anatomy-finish-1378/src")
sys.path.insert(0,str(E/"joint-region-roi-014"));sys.path.insert(0,str(E))
from numilab_human.skin_source_payload_preflight import decode_payload
from numilab_human.compiled_quotient_embeddedness import coordinate_quotient
from numilab_human.cardiac_cavity_intersections import _aabb_candidate_pairs,triangle_intersection_points,_allowed_shared_point
from audit_bilateral_skin_retained_poses_002 import records_nonzero
import audit_exact_joint_region_intersections_015 as audit
import run_ankle_tibia_weight_counterfactual_018 as wa
audit.coordinate_quotient=coordinate_quotient;audit._aabb_candidate_pairs=_aabb_candidate_pairs
audit.triangle_intersection_points=triangle_intersection_points;audit._allowed_shared_point=_allowed_shared_point
audit.records_nonzero=records_nonzero
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def main():
 d=decode_payload(PAYLOAD.read_bytes());V0=np.asarray(d["vertices_f"][:,:3],dtype=np.float32)
 W0=np.asarray(d["full_weights"],dtype=np.float32);B=np.asarray(d["bindings_u"][:,0],dtype=np.int64)
 BF=np.asarray(d["bindings_f"],dtype=np.float64);F=np.asarray(d["indices"],dtype=np.int64).reshape(-1,3)
 z=np.load(NPZ);ids=np.asarray(z["source_vertex_ids"],int)
 V1=V0.copy();W1=W0.copy()
 # Candidate serialization fields are explicitly Float32; evaluate from these packed values.
 V1[ids]=np.asarray(z["source_positions_after_m"],dtype=np.float32)
 W1[ids]=np.asarray(z["full_weights_after"],dtype=np.float32)
 if not np.array_equal(V1[ids],z["source_positions_after_m"]):raise ValueError("candidate position pack mismatch")
 if not np.array_equal(W1[ids],z["full_weights_after"]):raise ValueError("candidate weight pack mismatch")
 changedFaces=np.asarray(z["changed_source_faces"],dtype=np.int64)
 man=json.load(open(MAN));names={int(a["core_body_index"]):str(a["myosim_body"]) for a in man["source"]["anchors"]}
 slotnames=[names[int(b)] for b in B];owners=[slotnames[int(np.argmax(W0[i]))] for i in range(len(W0))]
 regions=np.asarray([audit.skin_region(x) for x in owners])
 cases={
  "reference":E/"native-registered-skin-pose-clean-014/registered-reference-elevation-010-014",
  "dorsiflexion":E/"native-registered-joint-reference-pairs-016/ankle-dorsiflexion-020-ref010",
  "plantarflexion":E/"native-registered-joint-reference-pairs-016/ankle-plantarflexion-020-ref010"}
 original_positions={};results={};savedDiff={}
 for label,folder in cases.items():
  snap=json.load(open(folder/"registered-skin-poses.json"))
  native=wa.load_position(folder/"registered-skin.csv")
  old=wa.lbs(V0,W0,B,BF,snap);new=wa.lbs(V1,W1,B,BF,snap)
  candidate=(native.astype(np.float64)+(new.astype(np.float64)-old.astype(np.float64))).astype(np.float32)
  out=np.setdiff1d(np.arange(len(V0)),ids)
  if not np.array_equal(candidate[out],native[out]):raise ValueError("outside-source-patch rows differ")
  saved=np.asarray(z[label+"_positions_m"],dtype=np.float32)
  diff=np.linalg.norm(candidate.astype(float)-saved.astype(float),axis=1)*1000
  savedDiff[label]={"max_float32_pose_reconstruction_delta_mm":float(np.max(diff)),"changed_row_max_mm":float(np.max(diff[ids]))}
  checks=wa.exact_checks(candidate,F,changedFaces,snap,owners,regions)
  tri0=native[F[changedFaces]].astype(float);tri1=candidate[F[changedFaces]].astype(float)
  cr0=np.cross(tri0[:,1]-tri0[:,0],tri0[:,2]-tri0[:,0]);cr1=np.cross(tri1[:,1]-tri1[:,0],tri1[:,2]-tri1[:,0])
  ar0=.5*np.linalg.norm(cr0,axis=1);ar1=.5*np.linalg.norm(cr1,axis=1)
  ratio=np.divide(ar1,ar0,out=np.full_like(ar1,np.inf),where=ar0>1e-24)
  cos=np.divide(np.einsum("ij,ij->i",cr0,cr1),np.linalg.norm(cr0,axis=1)*np.linalg.norm(cr1,axis=1),
   out=np.full(len(ar0),np.nan),where=np.linalg.norm(cr0,axis=1)*np.linalg.norm(cr1,axis=1)>1e-30)
  results[label]={"requested_q":json.load(open(folder/"invocation.json")).get("requested_q"),
   "registered_csv_sha256":sha(folder/"registered-skin.csv"),"snapshot_sha256":sha(folder/"registered-skin-poses.json"),
   "bone_pairs":checks["patch_vs_left_ankle_bones"]["intersection_pairs"],
   "self_pairs":checks["patch_self"]["intersection_pairs"],
   "other_skin_pairs":checks["patch_vs_other_full_skin"]["intersection_pairs"],
   "exact_static_gates":checks,"changed_face_degenerate_count":int(np.sum(ar1<=1e-24)),
   "changed_face_orientation_reversal_count":int(np.sum(cos<=0)),
   "min_area_ratio":float(np.min(ratio)),"min_normal_cosine":float(np.nanmin(cos)),
   "max_candidate_motion_mm":float(np.max(np.linalg.norm(candidate[ids].astype(float)-native[ids].astype(float),axis=1)*1000))}
  original_positions[label]=candidate
  print(json.dumps({"pose":label,"bone":results[label]["bone_pairs"],"self":results[label]["self_pairs"],
   "other_skin":results[label]["other_skin_pairs"],"min_area_ratio":float(np.min(ratio)),
   "normal_min":float(np.nanmin(cos)),"max_pose_repack_delta_mm":savedDiff[label]["max_float32_pose_reconstruction_delta_mm"]}),flush=True)
 report={"schema":"numi.human.left-ankle-coupled-f32-emission-audit.v1","candidate_id":"021",
  "status":"candidate_only_not_admitted_pending_mechanical_FEM_mapping_and_independent_review",
  "boundary":"Exact static Float32 source-position and full-weight re-evaluation through matched reference/DF/PF registration snapshots; exact local patch self/cross and left ankle bone predicates. Not dynamics/contact or compiled admission.",
  "source_payload":str(PAYLOAD),"source_payload_sha256":sha(PAYLOAD),
  "candidate_npz":str(NPZ),"candidate_npz_sha256":sha(NPZ),
  "candidate_source_rows":ids.astype(int).tolist(),"candidate_source_face_count":int(len(changedFaces)),
  "source_identity_preserved":True,"outside_candidate_rows_bitwise_native_csv":True,
  "pose_results":results,"difference_vs_prior_double-source-candidate":savedDiff,
  "reproduction":{"script":__file__,"script_sha256":sha(__file__),"python":sys.executable},
  "limits":["No mechanical/FEM mapping regenerated or admitted.","No dynamics or clinical ROM claim."]}
 out=OUT/"ankle-coupled-021-f32-emission-audit.json"
 if out.exists():raise SystemExit("refusing overwrite "+str(out))
 out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("REPORT="+str(out));print("SHA256="+sha(out))
if __name__=="__main__":main()
