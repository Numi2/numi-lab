#!/usr/bin/env python3
"""Exact nearby-all-bone test for Float32 ankle candidate 021, filtered by patch AABB + 10 mm."""
import hashlib,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=ROOT/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
OUT=E/"ankle-coupled-geometry-weight-021"
PAYLOAD=E/"finger-profile-069i-left-ring-web-retreat/replacement-bilateral-hands038.nhskin"
NPZ=OUT/"ankle-coupled-021-candidate.npz"
sys.path.insert(0,"/Users/n/numi-human-anatomy-finish-1378/src");sys.path.insert(0,str(E/"joint-region-roi-014"));sys.path.insert(0,str(E))
from numilab_human.compiled_quotient_embeddedness import coordinate_quotient
from numilab_human.cardiac_cavity_intersections import _aabb_candidate_pairs,triangle_intersection_points,_allowed_shared_point
from audit_bilateral_skin_retained_poses_002 import records_nonzero
import audit_exact_joint_region_intersections_015 as audit
import run_ankle_tibia_weight_counterfactual_018 as wa
audit.coordinate_quotient=coordinate_quotient;audit._aabb_candidate_pairs=_aabb_candidate_pairs
audit.triangle_intersection_points=triangle_intersection_points;audit._allowed_shared_point=_allowed_shared_point
audit.records_nonzero=records_nonzero
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 Z=np.load(NPZ);G=np.load(E/"finger-profile-069i-left-ring-web-retreat/replacement-geometry.npz")
 F=np.asarray(G["faces"],int);changed=np.asarray(Z["changed_source_faces"],int)
 MAN,raw,records,bv,bi,anchors=audit.load_bones()
 cases={"reference":E/"native-registered-skin-pose-clean-014/registered-reference-elevation-010-014","dorsiflexion":E/"native-registered-joint-reference-pairs-016/ankle-dorsiflexion-020-ref010","plantarflexion":E/"native-registered-joint-reference-pairs-016/ankle-plantarflexion-020-ref010"}
 reports={}
 for label,folder in cases.items():
  pose=json.loads((folder/"registered-skin-poses.json").read_text())
  V=np.asarray(Z[label+"_positions_m"],np.float32)
  tri=V[F[changed]]
  lo=tri.min(axis=(0,1))-.010;hi=tri.max(axis=(0,1))+.010
  body={int(x["body_index"]):x for x in pose["bodies"]};allv=[];allf=[];owners=[];offset=0;selected_parts=[];counts=Counter()
  for rec,a in zip(records,anchors):
   bodyid,fv,nv,fi,ni,stable,tx,ty,tz,qx,qy,qz,qw,scale,_=rec
   name=str(a["myosim_body"]);cur=body[int(bodyid)]["current"]
   Rb=audit.mat((qx,qy,qz,qw));Rp=audit.mat(cur["orientation_world_xyzw"])
   local=(bv[fv:fv+nv].astype(np.float64)*scale)@Rb.T+np.array((tx,ty,tz),float)
   world=(local@Rp.T+np.asarray(cur["position_world_m"],float)).astype(np.float32)
   ff=bi[fi//3:(fi+ni)//3].astype(np.int64)-int(fv)
   tbox=world[ff]
   mask=np.all(tbox.max(axis=1)>=lo,axis=1)&np.all(tbox.min(axis=1)<=hi,axis=1)
   if not mask.any():continue
   picked=ff[mask]
   allv.append(world);allf.append(picked+offset);owners.extend([name]*len(picked))
   selected_parts.append(name);counts[name]+=int(len(picked));offset+=len(world)
  skin_ids=np.unique(F[changed].ravel());skin_remap=np.full(len(V),-1,int);skin_remap[skin_ids]=np.arange(len(skin_ids))
  skinF=skin_remap[F[changed]]
  if allv:
   boneV=np.vstack(allv).astype(np.float32);boneF=np.vstack(allf).astype(np.int64)
   combinedV=np.vstack([V[skin_ids],boneV]).astype(np.float32)
   combinedF=np.vstack([skinF,boneF+len(skin_ids)]).astype(np.int64)
   pts,qf=audit.exact_scene(combinedV,combinedF);nskin=len(skinF)
   result=audit.exact_pair_audit(pts,qf,np.arange(nskin,dtype=np.int64),np.arange(nskin,len(combinedF),dtype=np.int64),label_a=lambda x:"ankle-changed-skin",label_b=lambda x:owners[x-nskin],keep=100)
  else: result={"candidate_pairs":0,"intersection_pairs":0,"witnesses_by_owner_pair":{}}
  reports[label]={"pose_snapshot_sha256":sha(folder/"registered-skin-poses.json"),"registered_csv_sha256":sha(folder/"registered-skin.csv"),"changed_skin_faces":int(len(changed)),"patch_aabb_world_m":{"min":lo.tolist(),"max":hi.tolist(),"padding_m":.01},"bone_parts_tested":selected_parts,"bone_face_counts":dict(counts),"all_nearby_bones_exact_intersections":result}
  print(json.dumps({"pose":label,"bone_parts":selected_parts,"bone_faces":sum(counts.values()),"exact_pairs":result["intersection_pairs"],"candidate_pairs":result["candidate_pairs"]}),flush=True)
 out={"schema":"numi.human.ankle-021-nearby-all-bone-exact-audit.v1","status":"candidate_only_static_exact_triangle_intersection_diagnostic","candidate_npz_sha256":sha(NPZ),"payload_sha256":sha(PAYLOAD),"source_geometry_sha256":sha(E/"finger-profile-069i-left-ring-web-retreat/replacement-geometry.npz"),"bone_manifest_sha256":sha(audit.MAN),"bone_payload_sha256":sha(raw),"method":"Exact triangle intersection of all changed candidate skin faces against every bone-mesh triangle whose world AABB intersects the changed-patch world AABB expanded by 10 mm. Bone meshes are transformed by each matching native registered snapshot; no nearest-bone-group token filter.","poses":reports,"limits":["Static triangle crossing diagnostic only; no penetration-depth clearance, continuum contact, FEM transfer, dynamics, or admission.","Unselected distant bone triangles have disjoint AABBs from the padded changed-patch AABB and cannot intersect changed skin triangles."]}
 path=OUT/"ankle-coupled-021-nearby-all-bone-exact-audit-v2.json"
 if path.exists():raise SystemExit("refusing overwrite "+str(path))
 path.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");json.loads(path.read_text());print("REPORT="+str(path));print("SHA256="+sha(path))
if __name__=="__main__":main()
