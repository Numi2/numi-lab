#!/usr/bin/env python3
"""Check paired native skin CSV and current/rest body-pose snapshots."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

E=Path(__file__).resolve().parent.parent
POSE_DIR=Path(__file__).resolve().parent
AUDIT=E/"audit_bilateral_skin_reweighted_candidate_016.py"
ns={}
exec(AUDIT.read_text().split("def main():")[0],ns)
payload=E/"finger-profile-069i-left-ring-web-retreat/replacement-bilateral-hands038.nhskin"
EXPECTED_SKIN_SHA="a848a41368aeb0dda8af579fe0ea417bbf7ec7db641033b5ba917302189fe1bc"
EXPECTED_BINARY_SHA="b218756325ce95258c53124bf307838d0dd0ff5db45c1875ff030d928cc1d4f9"
EXPECTED_PATCH_SHA="4046e0522fcd751afafe62bb98b767a0edbb7152eae969e010cbc4b75cbc45cd"
POSES=["shoulder-elevation-050","shoulder-abduction-030","elbow-flexion-080","ankle-dorsiflexion-020","ankle-plantarflexion-020","mtp-flexion-020","mtp-extension-020"]
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(payload)==EXPECTED_SKIN_SHA
skin=ns["decode_payload"](payload.read_bytes())
report={"schema":"numi.human.paired-native-registration-pose-validation.v1","scope":"Same-callsite registration CSV versus existing renderer pose serializer; static/pre-admission data consistency only, not dynamic motion, accepted FEM, self-contact, or biological qualification.","skin_payload":str(payload),"skin_payload_sha256":sha(payload),"source_emitter_patch_sha256":EXPECTED_PATCH_SHA,"binary_sha256":EXPECTED_BINARY_SHA,"poses":{},"failed_knee_attempts":[{"name":"knee-flexion-025","result":"native process exit 1 before CSV or pose output","failure":"ACL source joint anchor residual exceeds 0.2 mm"},{"name":"knee-flexion-005","result":"native process exit 1 before CSV or pose output","failure":"ACL source joint anchor residual exceeds 0.2 mm"}]}
for name in POSES:
    out=POSE_DIR/name
    inv=json.loads((out/"invocation.json").read_text())
    result=json.loads((out/"result.json").read_text())
    csvp=out/"registered-skin.csv"; posep=out/"registered-skin-poses.json"
    assert inv["binary_sha256"]==EXPECTED_BINARY_SHA,(name,inv["binary_sha256"])
    assert inv["skin_payload_sha256"]==EXPECTED_SKIN_SHA
    assert Path(inv["paired_pose_snapshot"]).resolve()==posep.resolve()
    assert result["registered_rows"]==61950 and result["complete_61950_vertex_dump"]
    assert result["paired_pose_snapshot_valid"] and result["native_exit_code"]==1
    assert not result["native_process_success"]
    pose=json.loads(posep.read_text())
    assert pose["schema"]=="numi.human.native-skin-pose-snapshot.v1"
    assert pose["vertex_count"]==61950 and pose["binding_count"]==86 and len(pose["bodies"])==86
    with csvp.open(newline="") as f:rows=list(csv.DictReader(f))
    assert len(rows)==61950
    vertex=np.asarray([int(r["vertex"]) for r in rows],dtype=np.int64)
    assert np.array_equal(vertex,np.arange(61950,dtype=np.int64)),name
    csv_xyz=np.asarray([[float(r[k]) for k in ("wx","wy","wz")] for r in rows],dtype=np.float64)
    reconstructed=ns["posed"](skin,pose).astype(np.float64)
    err=np.abs(reconstructed-csv_xyz)
    i,j=np.unravel_index(np.argmax(err),err.shape)
    report["poses"][name]={"requested_q":inv["requested_q"],"native_exit_code":result["native_exit_code"],"native_success":result["native_process_success"],"csv_sha256":sha(csvp),"pose_sha256":sha(posep),"registration_rows":len(rows),"pose_bindings":pose["binding_count"],"csv_pose_reconstruction_max_abs_error_m":float(err.max()),"p50_abs_error_m":float(np.percentile(err,50)),"p95_abs_error_m":float(np.percentile(err,95)),"p99_abs_error_m":float(np.percentile(err,99)),"max_error_vertex":int(i),"max_error_axis":int(j),"max_error_csv_m":csv_xyz[i].tolist(),"max_error_reconstructed_m":reconstructed[i].tolist(),"paired_by_same_native_callsite":True,"boundary":"pose-source consistency witness only; native constructor later fails the recorded inner-fat self-intersection gate"}
out=POSE_DIR/"paired-registration-pose-validation.json"
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:{"rows":v["registration_rows"],"max_abs_error_m":v["csv_pose_reconstruction_max_abs_error_m"],"native_exit":v["native_exit_code"],"csv_sha256":v["csv_sha256"],"pose_sha256":v["pose_sha256"]} for k,v in report["poses"].items()},indent=2))
print("report",out)
