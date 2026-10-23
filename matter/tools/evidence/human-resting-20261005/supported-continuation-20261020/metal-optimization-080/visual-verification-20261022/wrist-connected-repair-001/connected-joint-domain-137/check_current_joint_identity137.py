from pathlib import Path
import json,hashlib,struct
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019");E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001";D=E/"connected-joint-domain-137";D.mkdir(exist_ok=True)
M=E/"wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001/native-diagnostic-036-1ms-008x125us-001/launch-manifest.snapshot.json";m=json.loads(M.read_text());actual=Path(m["argv"][1]);old=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid");a=actual.read_bytes();b=old.read_bytes();sha=lambda x:hashlib.sha256(x).hexdigest();assert sha(a)==m["input_hashes"][str(actual)]
h=struct.unpack_from("<7I",a,8);assert h==struct.unpack_from("<7I",b,8);assert a[:8]==b[:8]==b"NHRIGID2";_,_,sb,nb,nj,nq,nv=h
bo=224;jo=bo+nb*160;do=jo+nj*144
body=[i for i in range(nb) if a[bo+i*160:bo+i*160+16]!=b[bo+i*160:bo+i*160+16]]
joint=[i for i in range(nj) if a[jo+i*144:jo+i*144+32]!=b[jo+i*144:jo+i*144+32]]
dof=[i for i in range(nv) if a[do+i*64:do+i*64+32]!=b[do+i*64:do+i*64+32]]
limits=[{"v":i,"old":struct.unpack_from("<2f",b,do+i*64+32),"actual":struct.unpack_from("<2f",a,do+i*64+32)} for i in range(nv) if a[do+i*64+32:do+i*64+40]!=b[do+i*64+32:do+i*64+40]]
r={"actual_payload":str(actual),"actual_sha256":sha(a),"launch_manifest_sha256":sha(M.read_bytes()),"older_structural_audit_payload_sha256":sha(b),"header_counts":dict(zip(["abi","variant","source_bodies","bodies","joints","nq","nv"],h)),"embedded_source_archive_matches":a[48:80]==b[48:80],"changed_body_identity_parent_rows":body,"changed_joint_identity_parent_child_rows":joint,"changed_dof_identity_rows":dof,"changed_limits":limits,"scope":"Identity and parent-chain cross-check against actual D36 launch; not mapping/contact/ROM qualification"};(D/"actual-d36-joint-identity.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
