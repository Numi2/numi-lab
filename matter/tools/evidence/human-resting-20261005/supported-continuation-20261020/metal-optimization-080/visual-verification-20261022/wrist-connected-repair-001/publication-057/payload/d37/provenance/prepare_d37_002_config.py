#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D36=ROOT/"wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001"
D28=ROOT/"wholebody-attachment-source-rebind-028-final080-omit-stale-inferred-hand-sites-001"
OUT=ROOT/"wholebody-attachment-source-rebind-037-candidate020-identity004-002"
NATIVE=ROOT/"E/connected-skin-transfer-owner-015/candidate020-counterfactual/native-source-emission-identity004-001"
cfg=json.loads((D36/"provenance/build_candidate_rebind_inputs.json").read_text())
cfg.update({
 "candidate_id":OUT.name,"candidate_tag":"020","run_tag":"D37","output_root":str(OUT),
 "candidate_payload":str(ROOT/"bilateral-hand-functional-rebind-020/run-001/candidate.nhskin"),
 "registered_candidate_csv":str(ROOT/"wholebody-attachment-source-rebind-037-candidate020-identity004-001/provenance/registered-skin-020-candidate.csv"),
 "native_capture_csv":str(NATIVE/"registered-skin.csv"),
 "capture_manifest":str(NATIVE/"launch-manifest.snapshot.json"),
 "capture_receipt":str(NATIVE/"execution-receipt.json"),
 "capture_lineage":str(NATIVE/"native-source-emission-lineage.json"),
 "base_candidate_root":str(D28),
 "parent_payload":str(ROOT/"middle-pip-singular-strain-repair-080/iteration-0/candidate.nhskin"),
 "parent_registered_csv":str(ROOT/"elbow-refinement-current-main-024-final080-candidate-source-frames-001/provenance/registered-skin-080-candidate.csv"),
 "base_payload":"/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin",
 "placement_diagnostic":str(ROOT/"elbow-refinement-current-main-024-final080-candidate-source-frames-001/native-source-only-024/common-rigid-transfer-row-diagnostic-005.json")
})
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
actual={
 "payload":cfg["candidate_payload"],"registered_csv":cfg["registered_candidate_csv"],"native_candidate_capture":cfg["native_capture_csv"],
 "D28_payload":cfg["parent_payload"],"D28_registered_csv":cfg["parent_registered_csv"],
 "D28_nodes":str(D28/"generated/nodes.csv"),"D28_faces":str(D28/"generated/faces.csv"),
 "D28_tets":str(D28/"generated/tetrahedra.csv"),"D28_embedding":str(D28/"generated/embedding.csv"),
 "body_frames":cfg["body_frames"],"base_payload":cfg["base_payload"],
 "D28_transfer_header":str(D28/"source/apps/NumiHumanWholeBodySkinTransferAncestry.hpp"),
 "capture_manifest":cfg["capture_manifest"],"capture_receipt":cfg["capture_receipt"],
 "capture_lineage":cfg["capture_lineage"],"placement_diagnostic":cfg["placement_diagnostic"]
}
cfg["expected_input_hashes"]={k:sha(v) for k,v in actual.items()}
config=OUT/"provenance/build_candidate_rebind_inputs.json"
config.write_text(json.dumps(cfg,indent=2,sort_keys=True)+"\n")
pin={"schema":"numi.human.d37-candidate020-input-pins.v1","config":str(config),"config_sha256":sha(config),"input_paths":actual,"input_sha256":cfg["expected_input_hashes"],"candidate_id":cfg["candidate_id"],"physical_qualification":False}
(OUT/"provenance/d37-input-pins.json").write_text(json.dumps(pin,indent=2,sort_keys=True)+"\n")
print(json.dumps({"config":str(config),"config_sha256":sha(config),"input_hashes":cfg["expected_input_hashes"],"candidate_id":cfg["candidate_id"]},indent=2))
