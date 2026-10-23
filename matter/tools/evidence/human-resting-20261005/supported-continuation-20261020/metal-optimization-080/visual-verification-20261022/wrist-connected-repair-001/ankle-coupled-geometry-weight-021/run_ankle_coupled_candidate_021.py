#!/usr/bin/env python3
"""Incremental local continuation from ankle candidate 020; exact gates remain mandatory."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=ROOT/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
OUT=E/"ankle-coupled-geometry-weight-021"
PAYLOAD=E/"finger-profile-069i-left-ring-web-retreat/replacement-bilateral-hands038.nhskin"
MAN=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json")
PREV=E/"ankle-coupled-geometry-weight-020/ankle-coupled-020-candidate.npz"
PREVREPORT=E/"ankle-coupled-geometry-weight-020/ankle-coupled-020-report.json"
WEIGHT_NPZ=E/"ankle-tibia-weight-counterfactual-018/s0p33-weights.npz"
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

def linear_maps(B,BF,W,snap):
 st={int(x["body_index"]):x for x in snap["bodies"]}
 if set(st)!=set(map(int,B)):raise ValueError("matched snapshot/body bindings mismatch")
 mats=[]
 for j,b in enumerate(B):
  cur=st[int(b)]["current"]
  mats.append(float(BF[j,8])*(wa.transform(cur["orientation_world_xyzw"])@wa.transform(BF[j,4:8])))
 return np.einsum("ij,jab->iab",W.astype(np.float64),np.asarray(mats))

def main():
 d=decode_payload(PAYLOAD.read_bytes());V0=np.asarray(d["vertices_f"][:,:3],dtype=np.float64)
 W0=np.asarray(d["full_weights"],dtype=np.float32);B=np.asarray(d["bindings_u"][:,0],dtype=np.int64)
 BF=np.asarray(d["bindings_f"],dtype=np.float64);F=np.asarray(d["indices"],dtype=np.int64).reshape(-1,3)
 Wdat=np.load(WEIGHT_NPZ);wids=np.asarray(Wdat["changed_vertex_ids"],int);W1=W0.copy()
 if not np.array_equal(W0[wids],Wdat["weights_before"]):raise ValueError("weight ancestry mismatch")
 W1[wids]=Wdat["weights_after"]
 prev=np.load(PREV);pr=json.load(open(PREVREPORT))
 if pr["input_payload_sha256"]!=sha(PAYLOAD):raise ValueError("020 source mismatch")
 Vprev=V0.copy();prevIDs=np.asarray(prev["source_vertex_ids"],int);Vprev[prevIDs]=prev["source_positions_after_m"]
 if not np.array_equal(Vprev[prevIDs].astype(np.float32),prev["source_positions_after_m"]):raise ValueError("020 position replay mismatch")
 cases={
  "reference":E/"native-registered-skin-pose-clean-014/registered-reference-elevation-010-014",
  "dorsiflexion":E/"native-registered-joint-reference-pairs-016/ankle-dorsiflexion-020-ref010",
  "plantarflexion":E/"native-registered-joint-reference-pairs-016/ankle-plantarflexion-020-ref010"}
 pose={}
 for name,folder in cases.items():
  pose[name]=(json.loads((folder/"registered-skin-poses.json").read_text()),
      wa.load_position(folder/"registered-skin.csv"),json.loads((folder/"invocation.json").read_text()),folder)
 df_snap,df_native,_,_=pose["dorsiflexion"]
 base_lbs=wa.lbs(V0,W0,B,BF,df_snap);prev_lbs=wa.lbs(Vprev,W1,B,BF,df_snap)
 df_prev=(df_native.astype(float)+(prev_lbs.astype(float)-base_lbs.astype(float))).astype(np.float32)
 _,_,br,bv,bi,anchors=audit.load_bones()
 boneV,boneF,boneOwner=audit.bone_scene(df_snap,"ankle","left",br,bv,bi,anchors)
 prevChanged=np.asarray(prev["source_vertex_ids"],int)
 prevFaces=np.flatnonzero(np.any(np.isin(F,prevChanged),axis=1)).astype(np.int64)
 pts,qf=audit.exact_scene(df_prev,F)
 u=np.unique(F[prevFaces].ravel());rm=np.full(len(V0),-1,dtype=np.int64);rm[u]=np.arange(len(u))
 sf=rm[F[prevFaces]];cv=np.vstack((df_prev[u],boneV));cf=np.vstack((sf,boneF+len(u)))
 cp,cq=audit.exact_scene(cv,cf)
 cross=audit.exact_pair_audit(cp,cq,np.arange(len(sf)),np.arange(len(sf),len(cf)),keep=5000)
 currentPairs=[]
 for ws in cross["witnesses_by_owner_pair"].values():
  for w in ws:
   currentPairs.append((int(prevFaces[int(w["face_a"])]),int(w["face_b"]-len(sf))))
 currentPairs=sorted(set(currentPairs))
 if len(currentPairs)!=cross["intersection_pairs"]:raise ValueError("truncated current exact pair list")
 priorPairs=[tuple(map(int,p)) for p in pr["exact_weight_only_residual_pairs"]]
 allPairs=sorted(set(priorPairs+currentPairs))
 if not currentPairs:raise ValueError("no residual pair for continuation")
 # Keep a local support around both preserved and currently intersecting faces.
 pairFaces=np.unique(np.asarray([p[0] for p in allPairs],int));pairRows=np.unique(F[pairFaces].ravel())
 oneRing=np.flatnonzero(np.any(np.isin(F,pairRows),axis=1))
 varRows=np.unique(F[oneRing].ravel());vm=np.full(len(V0),-1,dtype=np.int64);vm[varRows]=np.arange(len(varRows))
 A=linear_maps(B,BF,W1,df_snap)
 tibiaIDs=np.asarray([i for i,o in enumerate(boneOwner) if o=="tibia_l"],int)
 tibiaT=boneV[boneF[tibiaIDs]].astype(float)
 vol6=float(np.sum(np.einsum("ij,ij->i",tibiaT[:,0],np.cross(tibiaT[:,1],tibiaT[:,2]))))
 if abs(vol6)<1e-18:raise ValueError("indeterminate tibia winding volume")
 orient=1. if vol6>0 else -1.
 gap=0.00005
 rows=[];rhs=[];meta=[]
 for sfid,bfid in allPairs:
  if boneOwner[bfid]!="tibia_l":raise ValueError("unexpected non-tibial residual")
  tri=boneV[boneF[bfid]].astype(float);n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);norm=np.linalg.norm(n)
  if norm<1e-15:raise ValueError("degenerate tibia witness")
  n=orient*n/norm;q=tri[0]
  for vid0 in F[sfid]:
   vid=int(vid0);loc=int(vm[vid])
   if loc<0:raise ValueError("missing local variable")
   row=np.zeros(3*len(varRows));row[3*loc:3*loc+3]=n@A[vid]
   signed=float(n@(df_prev[vid].astype(float)-q))
   rows.append(row);rhs.append(gap-signed)
   meta.append({"skin_face":sfid,"bone_face":bfid,"source_vertex":vid,"signed_before_mm":signed*1000,"rhs_mm":(gap-signed)*1000})
 C=np.vstack(rows);b=np.asarray(rhs)
 # Graph Laplacian penalty over variable-variable edges; boundary edges anchor displacement to zero.
 edge=np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0)
 iv=vm[edge[:,0]];jv=vm[edge[:,1]]
 inside=(iv>=0)&(jv>=0);ei=iv[inside];ej=jv[inside]
 boundary=np.r_[iv[(iv>=0)&(jv<0)],jv[(iv<0)&(jv>=0)]]
 n=len(varRows);lam=1.0;H=np.eye(n)
 for i,j in zip(ei,ej):
  H[i,i]+=lam;H[j,j]+=lam;H[i,j]-=lam;H[j,i]-=lam
 for i in boundary:H[i,i]+=lam
 def fun(x):
  q=x.reshape(n,3);return 0.5*float(np.sum(q*(H@q)))
 def jac(x):return (H@x.reshape(n,3)).reshape(-1)
 res=minimize(fun,np.zeros(3*n),jac=jac,constraints=[{"type":"ineq","fun":lambda x:C@x-b,"jac":lambda x:C}],
  method="SLSQP",options={"maxiter":500,"ftol":1e-12,"disp":False})
 dx=res.x.reshape(n,3);margins=C@res.x-b
 if np.min(margins)<-1e-7:raise RuntimeError(f"continuation plane problem infeasible: {res.message}, min={np.min(margins)}")
 inc=np.zeros_like(V0);inc[varRows]=dx;Vnew=Vprev+inc
 changed=np.union1d(prevChanged,varRows);changedFaces=np.flatnonzero(np.any(np.isin(F,changed),axis=1)).astype(int)
 man=json.load(open(MAN));names={int(x["core_body_index"]):str(x["myosim_body"]) for x in man["source"]["anchors"]}
 slots=[names[int(b)] for b in B];owners=[slots[int(np.argmax(W0[i]))] for i in range(len(W0))]
 regions=np.asarray([audit.skin_region(x) for x in owners])
 results={};positions={}
 for label,(snap,native,inv,folder) in pose.items():
  pbase=wa.lbs(V0,W0,B,BF,snap);pcand=wa.lbs(Vnew,W1,B,BF,snap)
  cand=(native.astype(float)+(pcand.astype(float)-pbase.astype(float))).astype(np.float32)
  out=np.setdiff1d(np.arange(len(V0)),changed)
  if not np.array_equal(cand[out],native[out]):raise ValueError("out-of-patch source rows changed")
  a0=native[F[changedFaces]].astype(float);a1=cand[F[changedFaces]].astype(float)
  cr0=np.cross(a0[:,1]-a0[:,0],a0[:,2]-a0[:,0]);cr1=np.cross(a1[:,1]-a1[:,0],a1[:,2]-a1[:,0])
  ar0=.5*np.linalg.norm(cr0,axis=1);ar1=.5*np.linalg.norm(cr1,axis=1)
  ratio=np.divide(ar1,ar0,out=np.full_like(ar1,np.inf),where=ar0>1e-24)
  cos=np.divide(np.einsum("ij,ij->i",cr0,cr1),np.linalg.norm(cr0,axis=1)*np.linalg.norm(cr1,axis=1),out=np.full(len(ar0),np.nan),where=np.linalg.norm(cr0,axis=1)*np.linalg.norm(cr1,axis=1)>1e-30)
  checks=wa.exact_checks(cand,F,changedFaces,snap,owners,regions);positions[label]=cand
  results[label]={"requested_q":inv.get("requested_q"),"exact_static_gates":checks,
   "changed_row_motion_mm":wa.stats(np.linalg.norm(cand[changed].astype(float)-native[changed].astype(float),axis=1)*1000),
   "changed_face_area_ratio":wa.stats(ratio),"changed_face_normal_cosine":wa.stats(cos),
   "degenerate_count":int(np.sum(ar1<=1e-24)),"orientation_reversals":int(np.sum(cos<=0)),
   "registered_csv_sha256":sha(folder/"registered-skin.csv"),"snapshot_sha256":sha(folder/"registered-skin-poses.json")}
  print(json.dumps({"pose":label,"bone_pairs":checks["patch_vs_left_ankle_bones"]["intersection_pairs"],
   "self_pairs":checks["patch_self"]["intersection_pairs"],"other_skin_pairs":checks["patch_vs_other_full_skin"]["intersection_pairs"],
   "min_area_ratio":float(np.min(ratio)),"max_motion_mm":float(np.max(np.linalg.norm(cand[changed].astype(float)-native[changed].astype(float),axis=1)*1000))}),flush=True)
 outnpz=OUT/"ankle-coupled-021-candidate.npz"
 if outnpz.exists():raise SystemExit("refusing overwrite "+str(outnpz))
 np.savez_compressed(outnpz,source_vertex_ids=changed,source_positions_before_m=V0[changed].astype(np.float32),
  source_positions_after_m=Vnew[changed].astype(np.float32),source_rest_delta_from_020_m=inc[changed].astype(np.float32),
  source_rest_delta_total_m=(Vnew[changed]-V0[changed]).astype(np.float32),
  full_weights_before=W0[changed],full_weights_after=W1[changed],weight_delta=W1[changed]-W0[changed],
  preserved_020_crossing_pairs=np.asarray(priorPairs,int),new_020_crossing_pairs=np.asarray(currentPairs,int),
  reference_positions_m=positions["reference"],dorsiflexion_positions_m=positions["dorsiflexion"],
  plantarflexion_positions_m=positions["plantarflexion"],changed_source_faces=changedFaces,local_variable_source_rows=varRows)
 report={"schema":"numi.human.left-ankle-coupled-geometry-weight-candidate.v1","candidate_id":"021",
  "status":"candidate_only_not_admitted_pending_mechanical_FEM_mapping_and_independent_review",
  "boundary":"Incremental exact-pair-driven source-rest geometry continuation over candidate 020 plus the same 018 s=0.33 weight field. No shared source payload/runtime/header edits. Not a compiled admission or physical/dynamic/clinical qualification.",
  "source_payload":str(PAYLOAD),"source_payload_sha256":sha(PAYLOAD),"input_candidate":str(PREV),"input_candidate_sha256":sha(PREV),
  "weight_candidate":str(WEIGHT_NPZ),"weight_candidate_sha256":sha(WEIGHT_NPZ),
  "old_separated_pair_count":len(priorPairs),"current_df_pair_count":len(currentPairs),
  "current_df_pairs":currentPairs,"preserved_pair_constraints":len(priorPairs),"all_pair_constraints":len(allPairs),
  "source_faces_for_constraint_patch":pairFaces.astype(int).tolist(),"source_rows_for_constraint_patch":pairRows.astype(int).tolist(),
  "method":"At DF, exactly enumerate residual skin/tibia intersections on 020. Reapply oriented bone-face halfspace constraints to all prior separated and current crossing pairs with 0.05 mm construction margin. Solve minimum incremental source-rest displacement on affected source rows plus one mesh ring; boundary edges receive zero-displacement Laplacian anchoring. Propagate this same source delta through matched reference/DF/PF registrations and exact-test.",
  "tibia_oriented_signed_six_volume_m3":vol6,"tibia_outward_orientation_sign":orient,"separation_margin_m":gap,
  "qp":{"success":bool(res.success),"message":str(res.message),"iterations":int(res.nit),"objective":float(res.fun),
   "variable_rows":varRows.astype(int).tolist(),"variable_count":int(n),"constraint_count":int(len(b)),
   "minimum_linearized_margin_m":float(np.min(margins)),"max_increment_mm":float(np.max(np.linalg.norm(inc[varRows],axis=1))*1000),
   "max_total_rest_delta_mm":float(np.max(np.linalg.norm((Vnew-V0)[changed],axis=1))*1000),
   "constraint_witnesses":meta},
  "changed_source_rows":changed.astype(int).tolist(),"changed_source_faces":changedFaces.tolist(),
  "pose_results":results,"source_identity":{"payload_source_row_ids_preserved":True,"topology_order_preserved":True,
   "outside_changed_rows_equal_native_csv":True,"source_ancestry":[{"candidate_row":int(i),"source_vertex":int(i)} for i in changed]},
  "candidate_npz":str(outnpz),"candidate_npz_sha256":sha(outnpz),
  "limits":["All other candidate static checks are local to the ankle-patch plus whole-skin complement, and ankle-group bones.","No layer/FEM source mapping rebuilt or qualified.","No native dynamics/contact or whole-body admission claim."]}
 outjson=OUT/"ankle-coupled-021-report.json"
 if outjson.exists():raise SystemExit("refusing overwrite "+str(outjson))
 outjson.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("REPORT="+str(outjson));print("NPZ="+str(outnpz))

if __name__=="__main__":main()
