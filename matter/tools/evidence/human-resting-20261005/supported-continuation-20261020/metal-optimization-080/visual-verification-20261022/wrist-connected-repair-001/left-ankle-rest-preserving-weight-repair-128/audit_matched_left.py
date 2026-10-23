from pathlib import Path
import sys,json,hashlib,time,numpy as np
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019");E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001";P=Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001")
sys.path.insert(0,str(R/"tools"));import rebind_human_skin_reference as rb
sys.path.insert(0,str(E/"joint-region-roi-014"));import run_ankle_tibia_weight_counterfactual_018 as wa
D=E/"left-ankle-rest-preserving-weight-repair-128";sp=E/"bilateral-knee-connected-weight-repair-100/candidate.nhskin";cp=D/"candidate.nhskin";p=rb.decode(sp.read_bytes());q=rb.decode(cp.read_bytes());F=p["inds"]
a0,b0=rb.reference_transforms(p,json.loads((P/"neutral.json").read_text()),[0,0,.35]);X=rb.positions(p,a0,b0);C=X[F].mean(1)
faces=np.flatnonzero((C[:,0]>0)&(C[:,2]<.9))
changed=np.flatnonzero(np.any(p["weights"]!=q["weights"],axis=1));incident=np.flatnonzero(np.any(np.isin(F,changed),axis=1))
assert set(incident)<=set(faces),"ROI must contain every changed face"
outside=np.setdiff1d(np.arange(p["nv"]),changed);assert np.array_equal(p["verts"][outside],q["verts"][outside]);assert np.array_equal(p["weights"][outside],q["weights"][outside]);assert np.array_equal(p["inds"],q["inds"])
report={"source_sha256":hashlib.sha256(sp.read_bytes()).hexdigest(),"candidate_sha256":hashlib.sha256(cp.read_bytes()).hexdigest(),"scope":"Actual emitted Float32 left lower-leg/ankle/foot source surface against matched snapshot bone geometry and all other skin. Static registration gate, not dynamic/FEM/native admission.","selected_faces":len(faces),"changed_rows":len(changed),"incident_faces":len(incident),"outside_vertex_records_exact":True,"poses":{},"physical_admission":False}
owners=["current" for _ in range(p["nv"])];regions=np.array(["ankle" for _ in range(len(F))])
folders={"reference":wa.CAP/"registered-reference-elevation-010-014","DF":wa.MATCH/"ankle-dorsiflexion-020-ref010","PF":wa.MATCH/"ankle-plantarflexion-020-ref010"}
t=time.monotonic();print("faces",len(faces),"changed",len(changed),flush=True)
for name,folder in folders.items():
 snap=json.loads((folder/"registered-skin-poses.json").read_text());pose={"source_poses":[dict(core_body=x["body_index"],position=x["current"]["position_world_m"],quaternion_xyzw=x["current"]["orientation_world_xyzw"]) for x in snap["bodies"]]}
 a,b=rb.reference_transforms(p,pose,[0,0,0])
 report["poses"][name]={}
 for label,payload in [("baseline",p),("candidate",q)]:
  Y=rb.positions(payload,a,b).astype(np.float32);checks=wa.exact_checks(Y,F,faces,snap,owners,regions);report["poses"][name][label]=checks
  print(name,label,{k:v["intersection_pairs"] for k,v in checks.items() if isinstance(v,dict) and "intersection_pairs" in v},flush=True)
  (D/"full-left-ankle-exact.partial.json").write_text(json.dumps(report,indent=2)+"\n")
report["seconds"]=time.monotonic()-t
(D/"full-left-ankle-exact.json").write_text(json.dumps(report,indent=2)+"\n")
