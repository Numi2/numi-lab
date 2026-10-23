import hashlib, json
from pathlib import Path
BASE = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020")
W = BASE / "metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
D36 = W / "wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001"
A = BASE / "joint-functional-owner-audit-20261023"
B100 = W / "bilateral-knee-connected-weight-repair-100"
ANK = W / "ankle-parent-tet-refinement-026"
POSE_GATE = Path("/tmp/numi-pose-gate-review-20261023")

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def evidence(p): return {"path": str(p), "sha256": sha(p)}

d36m = read(D36 / "D36-manifest.json")
d36r = read(D36 / "D36-remap-report.json")
native = read(D36 / "native-diagnostic-036-1ms-008x125us-001/postrun-audit.json")
struct = read(A / "joint-coverage-audit.json")
table = read(W / "connected-joint-integration-134/d36-reference-table-check.json")
toe = struct["toe_bone_mesh_bindings"]
allpose = read(B100 / "all-pose-regression.json")
knee = read(B100 / "knee-q2-source128-witness-001/deformation-bone-clearance.json")
ank084 = read(ANK / "generated-084-support95-source-curvature-patch/candidate-report.json")
ank027 = read(ANK / "audit-027-exact-reference-dfp/exact-pose-layer-bone-audit.json")

regions = [
 {"region":"shoulder (bilateral)","joints":[
    {"name":"shoulder_elv_r","q":36,"v":35,"limits_rad":[0.0,3.142]},
    {"name":"shoulder1_r2_r","q":37,"v":36,"limits_rad":[-2.242,1.658]},
    {"name":"shoulder_rot_r","q":38,"v":37,"limits_rad":[-1.571,2.094]},
    {"name":"shoulder_elv_l","q":74,"v":73,"limits_rad":[0.0,3.142]},
    {"name":"shoulder1_r2_l","q":75,"v":74,"limits_rad":[-2.242,1.658]},
    {"name":"shoulder_rot_l","q":76,"v":75,"limits_rad":[-1.571,2.094]}],
  "support_geometry":"D36 provides a whole-body 2,411-row / 4,777-face reduced surface and O/I/D layers; it does not establish a shoulder-specific transfer or support region.",
  "bone_contact":"The 28-member NHBones payload is available, but no current-D36 shoulder-region skin/fat-to-scapula/clavicle/humerus exact-contact receipt is present.",
  "existing_pose_evidence":"Historical candidate-100 source-only regression: shoulder elevation pose has 31 self-pair witnesses and shoulder-plane pose has 53; these are source-surface failures, not bone-contact results, and the candidate differs from the current D36 identity.",
  "unresolved_owner":"Shoulder surface owner must establish regional skin/fat mapping, exact shoulder-girdle bone clearance, and coupled motion/contact coverage."},
 {"region":"elbow (bilateral)","joints":[
    {"name":"elbow_flexion_r","q":39,"v":38,"limits_rad":[0.0,2.269,"humerus_r to ulna_r"]},
    {"name":"elbow_flexion_l","q":77,"v":76,"limits_rad":[0.0,2.269,"humerus_l to ulna_l"]}],
  "support_geometry":"D36 whole-body surface exists; no elbow-specific mapped skin/fat support or contact ROI is established.",
  "bone_contact":"No current-D36 elbow skin-to-humerus/ulna exact-contact receipt is present.",
  "existing_pose_evidence":"Historical candidate-100 source-only regression has 90 elbow-pose self-pair witnesses. Separate 10 ms / 80-step elbow-flexor receipt shows a small core q/v and torque response with bitwise replay; it does not exercise FEM skin or contact.",
  "unresolved_owner":"Elbow surface/contact owner must connect selected tendon motion to accepted deformable skin/fat and prove local bone clearance over flexion."},
 {"region":"knee (bilateral)","joints":[
    {"name":"knee_angle_r","q":106,"v":105,"limits_rad":[0.0,2.0944]},
    {"name":"knee_angle_l","q":120,"v":119,"limits_rad":[0.0,2.0944]}],
  "support_geometry":"D36 uses the source128 skin payload and the same 2,411/4,777 whole-body mapping; no current-D36 knee-specific collision/support audit is present.",
  "bone_contact":"Source128 q=2 rad witness is not clear: 155 self-crossing pairs and 3 normal reversals on 155 faces. It reports zero intersections only against ankle-bone surfaces, so it does not establish knee-bone clearance.",
  "existing_pose_evidence":"Historical candidate-100 source-only regression clears q=1.0 and 1.5 rad but retains 155 self-pairs at 2.0 rad. Source128 witness is explicitly bound to the 4fd80 payload, but is a kinematic deformation test, not D36 FEM contact.",
  "unresolved_owner":"Knee surface owner must evaluate knee bones and contact in the exact D36 mapped pose, including source-supported upper-range pose outcomes; retain the q=2 failure as a warning, not a pass."},
 {"region":"ankle/subtalar (bilateral; current detailed patch is left ankle)","joints":[
    {"name":"ankle_angle_r","q":109,"v":108,"limits_rad":[-0.698132,0.523599]},
    {"name":"subtalar_angle_r","q":110,"v":109,"limits_rad":[-0.349066,0.349066]},
    {"name":"ankle_angle_l","q":123,"v":122,"limits_rad":[-0.698132,0.523599]},
    {"name":"subtalar_angle_l","q":124,"v":123,"limits_rad":[-0.349066,0.349066]}],
  "support_geometry":"D36 local candidate 084 selects 95 current D36 parent surface faces, one connected patch with 39 exact boundary anchors; it has not produced validated O/I/D volumetric closure. Coarse-to-source closest support distance is median 2.267 mm, p95 7.409 mm, max 23.506 mm.",
  "bone_contact":"Candidate 084 has no exact bone audit. Older candidate 027 failed exact triangle audits in reference, dorsiflexion, and plantarflexion; it is a different geometry candidate and must not be substituted as a result for 084 or D36 native.",
  "existing_pose_evidence":"D36 8x125-us native diagnostic accepted all 8 steps (1 ms total, minJ 0.9653); this is a resting whole-body short mechanics diagnostic, not ankle ROM/bone clearance. Historical 027 bone counts were 492/477/480 fat-deep, 61/104/53 outer, 140/197/141 skin-inner for reference/DF/PF, respectively.",
  "unresolved_owner":"Ankle surface and volume owners must close the 084 local O/I/D patch, run exact bone/layer audits for both sides, and then test approved ankle/subtalar poses."},
 {"region":"toe / MTP (bilateral)","joints":[
    {"name":"mtp_angle_r","q":111,"v":110,"limits_rad":[-0.523599,0.523599],"parent":"calcn_r","child":"toes_r"},
    {"name":"mtp_angle_l","q":125,"v":124,"limits_rad":[-0.523599,0.523599],"parent":"calcn_l","child":"toes_l"}],
  "support_geometry":"Five separate toe-member source meshes per side exist, but all members on a side use one shared toes rigid-body transform. D36 has only aggregate foot/toe coverage; there is no independent toe or IP support mapping.",
  "bone_contact":"No D36 per-digit bone/contact audit exists. The available bone payload identifies the anatomical members, but the runtime model has zero independent toe/IP DOFs.",
  "existing_pose_evidence":"Historical candidate-100 source-only MTP +/-0.25 and -0.5/0.5 tests report zero pairs for their tested poses; this is aggregate source-surface geometry only, not per-toe motion or D36 FEM contact.",
  "unresolved_owner":"Toe owner must decide whether the product scope needs individual toe/IP articulation. If so, split body ownership and collision support; current MTP provides only aggregate toe motion."}
]

out = {
 "schema":"numi.human-functional-joint-support-inventory.v1",
 "status":"READ_ONLY_RECEIPT_INVENTORY; no new geometry or physical qualification",
 "scope":"Current D36 / source128 support evidence for shoulder, elbow, knee, ankle/subtalar, and toe/MTP; unresolved owners are explicit.",
 "input_identity":{
  "d36_status":d36m["status"],"d36_candidate_identity_sha256":d36m["candidate_identity_sha256"],
  "source128_payload_sha256":d36m["input_hashes"]["payload"],
  "d36_manifest":evidence(D36/"D36-manifest.json"),
  "source128_payload":evidence(W/"left-ankle-rest-preserving-weight-repair-128/candidate.nhskin"),
  "d36_tables":{"surface_rows":table["surface_rows"],"surface_faces":table["surface_faces"],"tissue_nodes":table["tissue_nodes"],"tets":table["tets"],"materials":table["materials"],"nodes_sha256":table["input_hashes"]["nodes.csv"],"faces_sha256":table["input_hashes"]["faces.csv"],"embedding_sha256":table["input_hashes"]["embedding.csv"],"tetrahedra_sha256":table["input_hashes"]["tetrahedra.csv"],"identity_check_scope":table["scope"]},
  "d36_remap":{"status":d36r["status"],"physical_qualification":d36r["physical_qualification"],"counts":d36r["counts"],"geometry_mass":d36r["geometry_mass"]},
  "structural_joint_audit":{"receipt":evidence(A/"joint-coverage-audit.json"),"nhrigid_sha256":struct["native_scene"]["sha256"],"manifest_expected_nhrigid_sha256":struct["native_scene"]["source_manifest_rigid_payload_sha256"],"nhrigid_matches_manifest":struct["native_scene"]["native_file_matches_manifest_rigid_payload_sha256"],"embedded_source_archive_matches":struct["native_scene"]["source_archive_identity_matches"],"warning":"Joint/body rows are source-manifest structural mappings; the inspected NHRIGID differs from its manifest-pinned rigid payload, so this is not a current-D36 runtime pose/contact certificate.","skin_fat_mapping":struct["skin_fat_and_detailed_to_fem"]},
  "bone_geometry":{"payload_sha256":struct["source_inputs"]["bone_payload_sha256"],"manifest_sha256":struct["source_inputs"]["bone_manifest_sha256"],"member_count":toe["bone_payload"]["member_count"],"interpretation":"Bone geometry is available; availability does not establish current-D36 regional collision coverage."},
  "native_d36_diagnostic":{"receipt":evidence(D36/"native-diagnostic-036-1ms-008x125us-001/postrun-audit.json"),"scope":native["scope"],"physical_qualification":native["physical_qualification"],"accepted_steps":native["accepted_steps"],"simulated_seconds":native["simulated_seconds"],"fem":native["candidate_native_metrics"]["fem"],"source_contact_surface":native["candidate_native_metrics"]["source"]["contact_surface"],"body_contact_samples":native["candidate_native_metrics"]["fascia"]["body_contact_samples"]},
  "source_only_history":{"all_pose_receipt":evidence(B100/"all-pose-regression.json"),"scope":allpose["scope"],"candidate_sha256":allpose["candidate_sha256"],"knee_q2_source128_receipt":evidence(B100/"knee-q2-source128-witness-001/deformation-bone-clearance.json"),"knee_q2_source128_sha256":knee["source128_sha256"],"ankle_027_receipt":evidence(ANK/"audit-027-exact-reference-dfp/exact-pose-layer-bone-audit.json"),"ankle_027_status":ank027["status"],"ankle_084_receipt":evidence(ANK/"generated-084-support95-source-curvature-patch/candidate-report.json"),"ankle_084_status":ank084["status"]}
 },
 "regions":regions,
 "pose_diagnostic_gate_private_fix":{
  "location":str(POSE_GATE),
  "original_gate":evidence(W/"bilateral-knee-connected-weight-repair-100/pose_diagnostic_gate.py"),
  "private_gate_sha256":sha(POSE_GATE/"pose_diagnostic_gate.py"),
  "private_tests_sha256":sha(POSE_GATE/"test_pose_diagnostic_gate.py"),
  "test_command":"/Users/n/numi-human-prep-venv-20261005/bin/python -m unittest -v",
  "test_result":"10 tests passed",
  "changes":["A rejected source probe is labelled as self-intersection only when measured full-surface self-pairs are positive; zero and missing counts have distinct non-clear statuses.","NHEQ1 parsing rejects nonfinite record floats, out-of-range master_v, and mixed fixed-master sentinels; q-dependent evaluation rejects nonfinite outputs."]
 }
}
output=POSE_GATE/"joint-region-inventory.json"
output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(output)
print(sha(output))
