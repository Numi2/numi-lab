#!/usr/bin/env python3
import hashlib, importlib.util, json, pathlib, struct, sys
import numpy as np
R=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
npz_path=R/"E/connected-skin-mechanical-transfer-013/boundary-refinement-002/candidate.npz"
nh_path=R/"bilateral-hand-functional-rebind-020/run-001/candidate.nhskin"
snapshot_path=R/"wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-004/registered-skin-poses.json"
helper=R/"regional-refinement-compose-255/tools/rebind_human_skin_reference.py"
output=R/"E/connected-skin-transfer-owner-015/candidate020-counterfactual/actual-013-to-294-020-input.bin"
receipt=R/"E/connected-skin-transfer-owner-015/candidate020-counterfactual/actual-013-to-294-020-input.json"
expected={npz_path:"1849d24c14f8c0ddfedf3aa131877bdd38a852f9e48baebcf900a0efadf75db1",nh_path:"7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb",snapshot_path:"0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd"}
if output.exists() or receipt.exists(): raise SystemExit("refusing overwrite")
for p,h in expected.items():
 got=hashlib.sha256(p.read_bytes()).hexdigest()
 if got!=h: raise SystemExit(f"pinned input hash mismatch {p}: {got}")
spec=importlib.util.spec_from_file_location("rebind_exact294_015_cf020",helper); mod=sys.modules.setdefault(spec.name,importlib.util.module_from_spec(spec)); spec.loader.exec_module(mod)
payload=mod.decode(nh_path.read_bytes()); snap=json.loads(snapshot_path.read_text())
if (payload["nv"],payload["nb"],payload["fingerprint"])!=(61950,86,835155552): raise SystemExit("020 topology/ABI pin mismatch")
if snap.get("vertex_count")!=payload["nv"] or snap.get("binding_count")!=payload["nb"] or snap.get("registration_fingerprint32")!=payload["fingerprint"]: raise SystemExit("pose snapshot identity mismatch")
poses=snap.get("all_bodies",snap.get("bodies")); poseids={int(x["body_index"]) for x in poses}
bindbody=np.asarray(payload["binds"]["body"],dtype="<u4")
if set(map(int,bindbody))-poseids: raise SystemExit("snapshot lacks a 020 binding body")
source=np.asarray(mod.registered_positions_float32(payload,snap,frame="current"),dtype="<f4",order="C")
with np.load(npz_path,allow_pickle=False) as z:
 points=np.asarray(z["points_world_m"],dtype="<f4")
 node_ids=np.asarray(z["physical_outer_node_ids"],dtype=np.int64)
 faces0=np.asarray(z["physical_outer_faces"],dtype=np.int64)
 if points.ndim!=2 or points.shape[1]!=3 or node_ids.ndim!=1 or len(np.unique(node_ids))!=len(node_ids): raise SystemExit("candidate physical O node table malformed")
 if node_ids.min()<0 or node_ids.max()>=len(points): raise SystemExit("physical O node id out of range")
 rev=np.full(len(points),-1,dtype=np.int64); rev[node_ids]=np.arange(len(node_ids))
 if faces0.min()<0 or faces0.max()>=len(points): raise SystemExit("physical O face index out of range")
 faces=rev[faces0]
 if np.any(faces<0): raise SystemExit("physical O face uses a non-O volume node")
 outer=np.asarray(points[node_ids],dtype="<f4",order="C")
 faces=np.asarray(faces,dtype="<u4",order="C")
 sourcefaces=np.asarray(payload["inds"],dtype="<u4",order="C")
 weights=np.asarray(payload["weights"],dtype="<f4",order="C")
 ns,nf,no,nof=len(source),len(sourcefaces),len(outer),len(faces)
with output.open("xb") as f:
 f.write(struct.pack("<4Q",ns,nf,no,nof))
 for a in (source.reshape(-1),sourcefaces.reshape(-1),weights.reshape(-1),bindbody.reshape(-1),outer.reshape(-1),faces.reshape(-1)): f.write(a.tobytes(order="C"))
output_hash=hashlib.sha256(output.read_bytes()).hexdigest()
record={"schema":"numi.human.connected-skin-transfer-015.candidate020-counterfactual-input.v1","candidate_013":{"path":str(npz_path),"sha256":expected[npz_path],"point_count":len(points),"physical_outer_node_count":no,"physical_outer_face_count":nof},"source_020":{"path":str(nh_path),"sha256":expected[nh_path],"abi":5,"vertex_count":ns,"face_count":nf,"binding_count":len(bindbody),"payload_fingerprint32":payload["fingerprint"],"binding_body_indices":bindbody.astype(int).tolist(),"current_registered_positions_sha256":hashlib.sha256(source.tobytes()).hexdigest(),"pose_frame":"identity-004 current poses replayed by existing ABI5 helper; 020 full weights supplied"},"snapshot":{"path":str(snapshot_path),"sha256":expected[snapshot_path]},"input_binary":{"path":str(output),"sha256":output_hash,"bytes":output.stat().st_size},"mapping_status":"counterfactual input only; existing 015 accepted-transfer owner is run unchanged; output remains INFERRED_CONSTRUCTION_EMBEDDING, not measured ancestry","physical_or_native_admission":False}
receipt.write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))
