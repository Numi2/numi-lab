#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
D=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/global-skin-crossing-patches-012")
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
files=[
 D/"extract_physical_O_patches_012.py",
 D/"test_extract_physical_O_patches_012.py",
 D/"verify_patches_012.py",
 D/"test_verify_patches_012.py",
 D/"tests-extraction-012.log",
 D/"tests-verification-012.log",
 D/"connected-physical-O-patches-012.npz",
 D/"patch-report-012.json",
 D/"validation-report-012.json",
 D/"patch_bounds_summary_012.py",
 D/"patch-bounds-summary-012.json",
 D/"source294-mapping-status-012.json",
 D/"freeze_manifest_012.py",
 R/"global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz",
 R/"global-inner-p1-full-domain-clipping-009/exact-O-pair-011/run-001/canonical-pair-classification.npz",
 R/"global-inner-p1-full-domain-clipping-009/validation-001/zero-support-physical-O-face-pairs-010.npz",
 R/"global-inner-p1-full-domain-clipping-009/validation-001/zero-capacity-witness-index-009.npz",
 R/"registered-surrounding-tissue-native013-191/inventory.json",
 R/"wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-004/registered-skin-poses.json",
 Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin"),
 R/"registered-pose-frame-helper-194/tools/rebind_human_skin_reference.py",
]
for p in files:
 if not p.is_file():raise FileNotFoundError(p)
manifest={
 "schema":"numi.human.global-skin-crossing-patches.source-output-manifest.v1",
 "status":"PATCH_INPUT_EXTRACTION_AND_LINEAGE_CHECK_ONLY",
 "physical_admission":False,"native_admission":False,
 "selection":{"edge_connected_patch_count":9,"expansion_rings":2,"seed_physical_O_faces":426,"selected_O_faces":1171,"selected_O_nodes":808,"canonical_exact011_pairs":11251,"exact_hit_points":22502},
 "source294_correspondence":{"status":"UNRESOLVED_GLOBAL_TOPOLOGY_MAP","candidate_nearest_face_rows_within_2um":507,"direct_vertex_rows_within_0p2um":216,"unmatched_rows":85,"status":"PROVISIONAL_MAP_FAILURE_UNRESOLVED","note":"Nearest map is explicitly a diagnostic candidate, not exact ancestry; 85 of 808 rows unmatched. Do not propagate repair from this map. Canonical typed O-to-detailed transfer remains required."},
 "tests":{"extraction":"4 passed","verification":"6 passed"},
 "sha256":{str(p):sha(p) for p in files},
}
(D/"source-output-manifest-012.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print(json.dumps({"manifest":str(D/"source-output-manifest-012.json"),"manifest_sha256":sha(D/"source-output-manifest-012.json"),"files":len(files),"hashes":manifest["sha256"]},indent=2,sort_keys=True))
