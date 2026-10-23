#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import sys,json,hashlib,time,numpy as np
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
P=Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001")
D=E/"left-ankle-rest-preserving-weight-repair-128"; OUT=D/"joint-source128-coverage-002";OUT.mkdir(exist_ok=True)
out=OUT/"report.json"; partial=OUT/"report.partial.json"
if out.exists() or partial.exists(): raise SystemExit("refusing overwrite existing report")
sys.path.insert(0,str(R/"tools"));import rebind_human_skin_reference as rb
sys.path.insert(0,str(E/"joint-region-roi-014"));import run_ankle_tibia_weight_counterfactual_018 as wa
audit=wa.audit
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def snap_from_pose(doc):
 return {"bodies":[{"body_index":int(x["core_body"]),
  "current":{"position_world_m":(np.asarray(x["position"],float)+[0.,0.,.35]).tolist(),
             "orientation_world_xyzw":x["quaternion_xyzw"]}} for x in doc["source_poses"]]}
def pose_from_snap(doc):
 return {"source_poses":[{"core_body":int(x["body_index"]),
  "position":x["current"]["position_world_m"],
  "quaternion_xyzw":x["current"]["orientation_world_xyzw"]} for x in doc["bodies"]]}
def pairset(result):
 return {tuple(sorted((int(w["face_a"]),int(w["face_b"]))))
         for ws in result.get("witnesses_by_owner_pair",{}).values() for w in ws}
def exact_skin(V,F,roi):
 pts,qf=audit.exact_scene(V,F)
 other=np.setdiff1d(np.arange(len(F),dtype=np.int64),roi,assume_unique=True)
 se=audit.exact_pair_audit(pts,qf,roi,None,same_surface=True,keep=200000)
 cr=audit.exact_pair_audit(pts,qf,roi,other,keep=200000)
 return se,cr
def exact_bones(V,F,roi,snap,joint,records,bv,bi,anchors,face_side):
 vv=[];ff=[];owners=[];vbase=0
 for side in ("left","right"):
  x,y,z=audit.bone_scene(snap,joint,side,records,bv,bi,anchors)
  if len(y):vv.append(x);ff.append(y+vbase);owners.extend(z);vbase+=len(x)
 if not vv:return {"intersection_pairs":0,"note":"no bone triangles"}
 BV=np.vstack(vv).astype(np.float32);BF=np.vstack(ff).astype(np.int64)
 rv=F[roi];n=len(rv);CV=np.vstack((V,BV)).astype(np.float32);CF=np.vstack((rv,BF+len(V))).astype(np.int64)
 pts,qf=audit.exact_scene(CV,CF)
 return audit.exact_pair_audit(pts,qf,np.arange(n,dtype=np.int64),np.arange(n,len(CF),dtype=np.int64),
   label_a=lambda fi:f"{face_side[int(roi[int(fi)])]}_{joint}",
   label_b=lambda fi:owners[int(fi)-n],keep=200000)

src=E/"bilateral-knee-connected-weight-repair-100/candidate.nhskin";cand=D/"candidate.nhskin"
assert sha(src)=="27368e4bb9a0f59f41e67241a61b17de515a6ffb651b36dce6d47a81902869f3"
assert sha(cand)=="4fd80f449b2c4e6a94b0808c2e54a5735684711de1f7fc1c34e893c2494250a5"
p=rb.decode(src.read_bytes());q=rb.decode(cand.read_bytes())
F=np.asarray(p["inds"],dtype=np.int64);Fq=np.asarray(q["inds"],dtype=np.int64)
if not np.array_equal(F,Fq):raise ValueError("topology differs")
if p["binds"].tobytes()!=q["binds"].tobytes() or p["fingerprint"]!=q["fingerprint"] or p["source_sha256"]!=q["source_sha256"]:raise ValueError("binding/source identity differs")
w0=np.asarray(p["weights"]);w1=np.asarray(q["weights"]);x0=np.asarray(p["verts"]);x1=np.asarray(q["verts"])
cw=np.flatnonzero(np.any(w0!=w1,axis=1));cx=np.flatnonzero(np.any(x0!=x1,axis=1));changed=np.union1d(cw,cx)
initial=np.flatnonzero(np.any(np.isin(F,changed),axis=1));verts=np.unique(F[initial].ravel());roi=np.union1d(initial,np.flatnonzero(np.any(np.isin(F,verts),axis=1))).astype(np.int64)
outside=np.setdiff1d(np.arange(len(w0),dtype=np.int64),changed,assume_unique=True)
outside_exact=bool(np.array_equal(w0[outside],w1[outside]) and np.array_equal(x0[outside],x1[outside]))
if not outside_exact:raise ValueError("nonchanged source rows differ")
bone_manifest=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json")
names={int(a["core_body_index"]):str(a["myosim_body"]) for a in json.loads(bone_manifest.read_text())["source"]["anchors"]}
bindids=np.asarray(q["binds"]["body"],dtype=np.int64);own=[names[int(b)] for b in bindids[np.argmax(w1,axis=1)]]
faceowner=[Counter(own[int(v)] for v in tri).most_common(1)[0][0] for tri in F]
fside=[audit.side(x) for x in faceowner]
_,bonepath,brec,bv,bi,anchors=audit.load_bones()
base=json.loads((E/"bilateral-knee-connected-weight-repair-100/all-pose-regression.json").read_text())
old={x["pose"]:x for x in base["results"]}
pose_root=P/"joint-surface-audit-001/poses"; expanded=E/"joint-regional-quality-088-expanded-poses"; matched=E/"native-registered-joint-reference-pairs-016"
specs=[
 ("knee_bilateral_q106q120_0p25","knee",pose_root/"knee_bilateral_q106q120_0p25.json","pose"),
 ("knee_1p0","knee",expanded/"knee_1p0.json","pose"),
 ("knee_1p5","knee",expanded/"knee_1p5.json","pose"),
 ("knee_2p0","knee",expanded/"knee_2p0.json","pose"),
 ("ankle_dorsiflexion_020","ankle",matched/"ankle-dorsiflexion-020-ref010/registered-skin-poses.json","snapshot"),
 ("ankle_plantarflexion_020","ankle",matched/"ankle-plantarflexion-020-ref010/registered-skin-poses.json","snapshot"),
 ("mtp_flexion_020","mtp",matched/"mtp-flexion-020-ref010/registered-skin-poses.json","snapshot"),
 ("mtp_extension_020","mtp",matched/"mtp-extension-020-ref010/registered-skin-poses.json","snapshot"),
 ("mtp_m0p5","mtp",expanded/"mtp_m0p5.json","pose"),
 ("mtp_0p5","mtp",expanded/"mtp_0p5.json","pose")]
# Crosscheck q=.25 FK pose against its matched native capture, including the static +0.35m placement.
pd=json.loads(specs[0][2].read_text());cd=json.loads((matched/"knee-flexion-025-ref010/registered-skin-poses.json").read_text())
pa={int(x["core_body"]):x for x in pd["source_poses"]};cb={int(x["body_index"]):x for x in cd["bodies"]}
if not set(cb).issubset(pa) or len(cb)!=86:raise ValueError("q=.25 pose/capture body-set mismatch")
maxpos=max(float(np.max(np.abs(np.asarray(pa[i]["position"],float)+[0,0,.35]-np.asarray(cb[i]["current"]["position_world_m"],float)))) for i in cb)
maxquat=max(min(float(np.max(np.abs(np.asarray(pa[i]["quaternion_xyzw"],float)-np.asarray(cb[i]["current"]["orientation_world_xyzw"],float)))),
                float(np.max(np.abs(np.asarray(pa[i]["quaternion_xyzw"],float)+np.asarray(cb[i]["current"]["orientation_world_xyzw"],float))))) for i in cb)
if maxpos>3e-6 or maxquat>3e-6:raise ValueError(f"q=.25 FK mismatch {maxpos}m/{maxquat}")
report={"schema":"numi.human.source128.incremental-joint-roi-audit.v1",
 "scope":"Exact static LBS source-surface audit of changed rows, their incident faces and one face ring, against ROI self, full remainder skin, and bilateral nearby bones. Source geometry only; no FEM motion, physical admission, natural ROM, or clinical qualification.",
 "source100":{"path":str(src),"sha256":sha(src)},"candidate128":{"path":str(cand),"sha256":sha(cand)},
 "bone_mesh":{"path":str(bonepath),"sha256":sha(bonepath)},
 "topology_binding_ancestry_equal":True,"changed_weight_rows":len(cw),"changed_vertex_position_rows":len(cx),"changed_union_rows":len(changed),
 "changed_row_ids":changed.astype(int).tolist(),"incident_faces":len(initial),"one_ring_roi_faces":len(roi),
 "outside_vertex_records_exact":outside_exact,"roi_face_sides":dict(Counter(fside[int(i)] for i in roi)),
 "q025_fk_snapshot_crosscheck":{"source_pose_sha256":sha(specs[0][2]),"snapshot_sha256":sha(matched/"knee-flexion-025-ref010/registered-skin-poses.json"),"body_count":len(cb),"max_position_delta_m":maxpos,"max_quaternion_component_delta_antipodal_equivalent":maxquat},
 "baseline_source100_report":{"path":str(E/"bilateral-knee-connected-weight-repair-100/all-pose-regression.json"),"sha256":sha(E/"bilateral-knee-connected-weight-repair-100/all-pose-regression.json"),"candidate_sha256":base["candidate_sha256"]},
 "poses":{},"physical_admission":False}
t0=time.monotonic()
for name,joint,path,kind in specs:
 doc=json.loads(path.read_text())
 if kind=="snapshot":snap=doc;pose=pose_from_snap(doc);lift=[0,0,0]
 else:snap=snap_from_pose(doc);pose=doc;lift=[0,0,.35]
 A,B=rb.reference_transforms(q,pose,lift);V=rb.positions(q,A,B).astype(np.float32)
 se,cr=exact_skin(V,F,roi);bn=exact_bones(V,F,roi,snap,joint,brec,bv,bi,anchors,fside)
 current=pairset(se)|pairset(cr);base_row=old.get(name);base_pairs=set()
 hash_match=bool(base_row and base_row.get("pose_sha256")==sha(path))
 if base_row and hash_match:
  for ln in base_row.get("stderr","").splitlines():
   z=ln.strip().split(",")
   if len(z)==3 and z[0]=="crossing":base_pairs.add(tuple(sorted((int(z[1]),int(z[2])))))
 newpairs=sorted(current-base_pairs) if hash_match else None
 report["poses"][name]={"joint":joint,"pose_kind":kind,"pose_input":str(path),"pose_sha256":sha(path),
  "baseline_source100_pose_hash_matches":hash_match,"source100_full_surface_crossings":base_row.get("crossings") if hash_match else None,
  "current128_roi_self":se,"current128_roi_vs_rest_full_skin":cr,"current128_roi_vs_bilateral_nearby_bones":bn,
  "current128_roi_surface_pair_count":len(current),"new_roi_surface_pairs_vs_source100_global_set":newpairs,
  "current128_LBS_position_sha256":hashlib.sha256(V.tobytes()).hexdigest()}
 partial.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"pose":name,"roi":len(roi),"self":se["intersection_pairs"],"skin":cr["intersection_pairs"],"bones":bn["intersection_pairs"],"new":None if newpairs is None else len(newpairs)},sort_keys=True),flush=True)
report["elapsed_s"]=time.monotonic()-t0;report["physical_admission"]=False
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");partial.unlink(missing_ok=True)
print("REPORT="+str(out))

