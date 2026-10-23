from pathlib import Path
import sys,json,hashlib,time,numpy as np
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019");E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001";P=Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001")
sys.path.insert(0,str(R/"tools"));import rebind_human_skin_reference as rb
sys.path.insert(0,str(E/"joint-region-roi-014"));import run_ankle_tibia_weight_counterfactual_018 as wa
D=E/"left-ankle-rest-preserving-weight-repair-128";cp=D/"candidate.nhskin";p=rb.decode(cp.read_bytes());F=p["inds"]
a0,b0=rb.reference_transforms(p,json.loads((P/"neutral.json").read_text()),[0,0,.35]);X=rb.positions(p,a0,b0);C=X[F].mean(1);faces=np.flatnonzero((C[:,0]<0)&(C[:,2]<.9));u=np.unique(F[faces]);mapping=np.full(p["nv"],-1,dtype=int);mapping[u]=np.arange(len(u));skinF=mapping[F[faces]]
_,bp,brec,bv,bi,anchors=wa.audit.load_bones();report={"candidate_sha256":hashlib.sha256(cp.read_bytes()).hexdigest(),"scope":"Actual Float32 right lower leg/foot source skin vs right ankle bones, matched static poses only.","selected_faces":len(faces),"physical_admission":False,"poses":{}}
folders={"reference":wa.CAP/"registered-reference-elevation-010-014","DF":wa.MATCH/"ankle-dorsiflexion-020-ref010","PF":wa.MATCH/"ankle-plantarflexion-020-ref010"}
for name,folder in folders.items():
 snap=json.loads((folder/"registered-skin-poses.json").read_text());pose={"source_poses":[dict(core_body=x["body_index"],position=x["current"]["position_world_m"],quaternion_xyzw=x["current"]["orientation_world_xyzw"]) for x in snap["bodies"]]}
 a,b=rb.reference_transforms(p,pose,[0,0,0]);Y=rb.positions(p,a,b).astype(np.float32);V,BF,owners=wa.audit.bone_scene(snap,"ankle","right",brec,bv,bi,anchors)
 CV=np.vstack((Y[u],V));CF=np.vstack((skinF,BF+len(u)));pts,qf=wa.audit.exact_scene(CV,CF);nf=len(skinF)
 checks=wa.audit.exact_pair_audit(pts,qf,np.arange(nf),np.arange(nf,len(CF)),label_a=lambda fi:"right_skin",label_b=lambda fi:owners[int(fi)-nf])
 report["poses"][name]=checks;print(name,checks["intersection_pairs"],flush=True)
(D/"right-ankle-bone-exact.json").write_text(json.dumps(report,indent=2)+"\n")
