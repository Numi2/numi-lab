import sys,json,hashlib,math,gc
from pathlib import Path
import numpy as np
sys.path.insert(0,"/Users/n/numi-human-closed-outer-skin-20261017/src")
from numilab_human.resting_anatomy_verify import validate_capture_receipt,_pack_sections
p=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-suspension-reference-probe-20261018-001/attempt-020/native-run")
rows=[]
for rp in sorted((p/"accepted-geometry").glob("*.receipt.json"),key=lambda f:int(f.name.split("-")[1].split(".")[0])):
 r=json.loads(rp.read_bytes());pack=rp.with_name(rp.name.replace(".receipt.json",".mrvpack"))
 with pack.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
 endpoint=validate_capture_receipt(r,sha);sections=_pack_sections(pack);raw,n,stride=sections[2];assert stride==80
 assert hashlib.sha256(raw).hexdigest()==r["captured_vertex_buffer_sha256"]
 vertices=np.frombuffer(raw,"<f4").reshape(-1,20);assert np.isfinite(vertices[:,:3]).all() and np.isfinite(vertices[:,4:7]).all()
 b=next(b for b in r["accepted_body_poses"] if b["body_index"]==20);v=np.array([-.054597348,.141103089,-.00279765762]);q=np.array(b["quaternion_xyzw"]);site=np.array(b["position_m"])+v+2*np.cross(q[:3],np.cross(q[:3],v)+q[3]*v);anchor=np.array([-.022202339,.253997326,2.64830303])
 item={"pack_path":str(pack),"pack_sha256":sha,"capture_receipt_sha256":hashlib.sha256(rp.read_bytes()).hexdigest(),"step":int(rp.name.split("-")[1].split(".")[0]),"accepted_body_pose_count":r["accepted_body_pose_count"],"captured_vertices_finite":True,"vertex_buffer_sha256_verified":True,"endpoint_receipt_validated_by_existing_owner":True,"rope_site_from_accepted_pose_m":site.tolist(),"rope_anchor_site_distance_m":float(np.linalg.norm(site-anchor))}
 rows.append(item);del sections,raw,vertices;gc.collect()
report={"status":"PASS_CAPTURE_IDENTITY_FINITE_COORDINATES_AND_ACCEPTED_ENDPOINT","captures":rows,"rope_length_m":.800000012,"max_abs_rope_length_error_m":max(abs(x["rope_anchor_site_distance_m"]-.800000012) for x in rows),"scope":"capture identity and accepted-state rope geometry; not all-pair anatomical intersection qualification"}
(p/"capture-verification.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"captures":len(rows),"max_abs_rope_length_error_m":report["max_abs_rope_length_error_m"]}))
