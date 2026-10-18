import sys,json,hashlib,time
from pathlib import Path
import numpy as np
sys.path.insert(0,"/Users/n/numi-human-closed-outer-skin-20261017/src")
from numilab_human.resting_pleura_proxy import _parse_payload
from numilab_human.surface_precision_retriangulation import collapse_closed_component_edge_preserving_allocations
from numilab_human.resting_intercostal_mesh_arrangement import encode_payload_with_row_replacements
p=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-suspension-reference-probe-20261018-001")
out=p/"head-subulp-edge-retirement-001";out.mkdir(exist_ok=False)
src=p/"intercostal313-subulp-edge-retirement-001/resting-thorax.nhanatomy";raw=src.read_bytes();sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(raw)=="e617564e9f1ce6097848098c2ce55c89493c16468b793e30f50223a240f60f7c"
_,_,rows=_parse_payload(raw); inventory=json.loads((p/"nha-source-sub50nm-edge-inventory.json").read_bytes())
replacements={}; operations={}; t=time.monotonic()
for sid in (323,343,345,348):
 r=rows[sid];v=r["vertices6"];f=r["faces"].copy();orig=np.arange(len(f));ops=[]
 for a,b in inventory[str(sid)]["edges"]:
  keep,remove=sorted((int(a),int(b)),key=lambda i:tuple(float(x) for x in v[i,:3]))
  nf,proof=collapse_closed_component_edge_preserving_allocations(v[:,:3].copy(),f,identity=(51010,sid),keep_vertex=keep,remove_vertex=remove,max_endpoint_displacement_m=5e-8,max_abs_volume_delta_m3=2e-12)
  proof["source_payload_face_origins"]=orig[proof["kept_parent_face_origins"]].tolist();orig=orig[proof["kept_parent_face_origins"]]; f=nf;ops.append(proof)
 nr=dict(r);nr["faces"]=f;replacements[sid]=nr;operations[str(sid)]=ops
 print(json.dumps({"row":sid,"collapsed_edges":len(ops),"retired_faces":len(r["faces"])-len(f),"elapsed_s":time.monotonic()-t}),flush=True)
payload,encoding=encode_payload_with_row_replacements(raw,replacements)
proof={"source_path":str(src),"source_sha256":sha(raw),"output_sha256":sha(payload),"operations":operations,"encoding":encoding,"policy":"all seven source edges below50nm in passive head rows323,343,345,348; actual step127 failures343/345; exact shared-coordinate endpoint ordering consistent; source vertices and attributes unchanged","native_admission":False,"anatomical_admission":False}
(out/"resting-thorax.nhanatomy").write_bytes(payload);(out/"repair-report.json").write_text(json.dumps(proof,indent=2)+"\n")
print(json.dumps({"path":str(out),"sha256":sha(payload),"elapsed_s":time.monotonic()-t}),flush=True)
