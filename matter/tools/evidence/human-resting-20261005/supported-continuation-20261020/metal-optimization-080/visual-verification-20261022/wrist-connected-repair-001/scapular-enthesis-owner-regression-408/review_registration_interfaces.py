#!/usr/bin/env python3
import hashlib,json,math,sys
from pathlib import Path
R=Path("/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004")
FIT=R/"bilateral-all-upper-limb-fit-001"
COM=R/"bilateral-all-upper-limb-bone-compile-001"
OWNER=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/source-route-weighting-owner-385/validation/src")
sys.path.insert(0,str(OWNER))
from numilab_human import model
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
c=json.loads((FIT/"candidate.json").read_text())
u=c["upper_limb_source_mesh_registration"]
def bodyfit(name):
 return next(x for x in u["body_fits"] if x.get("myosim_body")==name)
def continuity(name):
 return next(x for x in u["continuity"] if x.get("name")==name)
source_sha="280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975"
base=COM/"baseline"; cand=COM/"candidate"
pairs=[(29,"clavicle_r",10,"clavicle_r"),(34,"scapula_r",11,"scapula_l"),(41,"humerus_r",12,"humerus_l"),(84,"scapula_l",16,"scapula_l")]
surfs={}
for key,folder in [("baseline",base),("candidate",cand)]:
 ss,meta,p=model._numi_human_bone_envelope_surfaces(folder,source_sha)
 surfs[key]=ss
for body,name,_,_ in pairs:
 a=next(x for x in surfs["baseline"][body] if x["member_id"]==({"clavicle_r":"FJ3362","scapula_r":"FJ3384","humerus_r":"FJ3368","scapula_l":"FJ3279"}[name]))
 b=next(x for x in surfs["candidate"][body] if x["member_id"]==a["member_id"])
 assert len(a["vertices"])==len(b["vertices"])
 ds=sorted(math.dist(x,y) for x,y in zip(a["vertices"],b["vertices"]))
 if name=="clavicle_r": clavicle={"member_id":a["member_id"],"vertices":len(ds),"mean_displacement_m":sum(ds)/len(ds),"p95_displacement_m":ds[int(.95*(len(ds)-1))],"max_displacement_m":ds[-1]}
 elif name=="scapula_r": scap_r={"member_id":a["member_id"],"vertices":len(ds),"mean_displacement_m":sum(ds)/len(ds),"p95_displacement_m":ds[int(.95*(len(ds)-1))],"max_displacement_m":ds[-1]}
 elif name=="humerus_r": hum_r={"member_id":a["member_id"],"vertices":len(ds),"mean_displacement_m":sum(ds)/len(ds),"p95_displacement_m":ds[int(.95*(len(ds)-1))],"max_displacement_m":ds[-1]}
 elif name=="scapula_l": scap_l={"member_id":a["member_id"],"vertices":len(ds),"mean_displacement_m":sum(ds)/len(ds),"p95_displacement_m":ds[int(.95*(len(ds)-1))],"max_displacement_m":ds[-1]}
h=bodyfit("humerus_r"); s=bodyfit("scapula_r")
report={
 "schema":"numi.human.scapular-registration-interface-review.v1",
 "admission":False,
 "source_revision":{"myosim_revision":"33c89c2bde282553dde3f526768eb3bdcfaa7649","myosim_archive_sha256":source_sha,"bodyparts3d_archive_sha256":"40665852c49f218326590e204db91064a1ecfc3c6f8cbd7bbbcaac62c7cd409e","baseline_registration_sha256":"b1b410ad6d4ac8c0c95fd0c3e10f655b24c890d5767a5c377cf78e28ef598f8e","candidate_registration_sha256":"b42e92c21f49b2fb7612447f0f0678295cae894ddffbe80cdf11f210ccee6a73","upper_limb_registration_source_sha256":"d169202d5a0e4356764dda0ab51f602524251449a57b5ab4ce890c022a074719","source_archive_pinned_NHRIGID_sha256":"2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4"},
 "input_hashes":{"candidate_json":sha(FIT/"candidate.json"),"run_result":sha(FIT/"run-result.json"),"compile_comparison":sha(COM/"compile-comparison-003.json"),"candidate_nhbones":sha(COM/"candidate/bodyparts3d-myosim-major-bones.nhbones"),"candidate_nhbones_manifest":sha(COM/"candidate/bodyparts3d-myosim-major-bones.manifest.json")},
 "candidate_status":{"fit_status":u["status"],"fit_scope":u["scope"],"candidate_endpoint_count":u["registration_candidate_endpoint_count"],"candidate_endpoints_passed":u["registration_candidate_endpoint_gate_pass_count"],"prior_admitted_endpoints":u["prior_admitted_endpoint_count"],"prior_admitted_endpoints_preserved":u["prior_admitted_endpoint_preserved_count"],"compile_status":json.loads((COM/"compile-comparison-003.json").read_text())["status"],"compile_admission":json.loads((COM/"compile-comparison-003.json").read_text())["admission"],"NHRIGID_unmodified":True},
 "candidate_registration_metrics":{"scapula_r":{"rotation_angle_rad":s["rotation_angle_rad"],"scale":s["uniform_scale"],"training":s["training_metrics"],"held_out":s["held_out_metrics"],"endpoint_translation_refinement":s["endpoint_landmark_refinement"]},"humerus_r":{"rotation_angle_rad":h["rotation_angle_rad"],"scale":h["uniform_scale"],"scale_bounds":h["uniform_scale_bounds"],"training":h["training_metrics"],"held_out":h["held_out_metrics"],"humeral_head_articular_gate":h["humeral_head_articular_gate"]}},
 "existing_interface_proxies":{"right_clavicle_to_scapula":continuity("right_clavicle_to_scapula"),"right_scapula_to_humerus":continuity("right_scapula_to_humerus"),"proxy_definition":"Bidirectional p90 of lowest 2% point-to-point vertex distances; generic whole-surface interface proxy, not glenoid- or acromion-localized and not cartilage/contact certification."},
 "corresponding_NHBONES_vertex_displacement_baseline_to_candidate":{"clavicle_r":clavicle,"scapula_r":scap_r,"humerus_r":hum_r,"scapula_l":scap_l},
 "interpretation":"The candidate passes the producer's source-mesh endpoint/default-pose gates and existing coarse interface proxies, and preserves NHRIGID. It changes the registered source geometry substantially. No explicit glenoid-fossa/acromion region correspondence or independent enthesis landmark gate was found; the three source route sites also participate in scapular endpoint refinement. Keep candidate unadmitted pending region-specific anatomy and coupled replay/force checks."
}
out=Path(__file__).resolve().parent/"registration-interface-review.json"
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(out)
print(json.dumps(report["corresponding_NHBONES_vertex_displacement_baseline_to_candidate"],indent=2))

