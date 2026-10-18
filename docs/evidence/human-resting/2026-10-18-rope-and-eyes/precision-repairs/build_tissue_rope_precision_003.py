import sys,json,time,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,"/Users/n/numi-human-closed-outer-skin-20261017/src")
from numilab_human.passive_attachment_composition import _read_nhtiss4,_nhtiss_row_arrays,_row_slices
from numilab_human.surface_precision_retriangulation import collapse_closed_component_edge_preserving_allocations
from numilab_human.passive_reference_receipt_composition import compose_reference_receipt
p=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-suspension-reference-probe-20261018-001")
src=Path(json.loads((p/"tissue-native-precision-retirement-002/result.json").read_bytes())["payload_path"])
out=p/"tissue-native-precision-retirement-003";out.mkdir(exist_ok=False); sha=lambda b:hashlib.sha256(b).hexdigest()
data=_read_nhtiss4(src);assert data["sha256"]=="4a1c07c00d744ee4d228fb9a207bd09df399b131374df35f01151551dc2c381e"
inventory=json.loads((p/"native016-step0-tissue-sub200nm-edge-inventory.json").read_bytes()); native_positions=np.load(p/"native016-step0-tissue-precision-positions.npz",allow_pickle=False); replacements=[];t=time.monotonic()
for sid in sorted(map(int,inventory)):
 row=next(r for r in data["records"] if int(r[6])==sid);a=_nhtiss_row_arrays(data,row);f=a["faces"];orig=np.arange(len(f));ops=[]
 for x,y in inventory[str(sid)]["edges"]:
  if not np.isin([x,y],f).all(): continue
  keep,remove=sorted((int(x),int(y)),key=lambda i:tuple(float(z) for z in native_positions[str(sid)][i]))
  nf,proof=collapse_closed_component_edge_preserving_allocations(native_positions[str(sid)].copy(),f,identity=(51005,sid),keep_vertex=keep,remove_vertex=remove,max_endpoint_displacement_m=2e-7,max_abs_volume_delta_m3=2e-12)
  proof["source_payload_face_origins"]=orig[proof["kept_parent_face_origins"]].tolist();orig=orig[proof["kept_parent_face_origins"]];f=nf;ops.append(proof)
 patch=out/f"row{sid}.npz";report=out/f"row{sid}-report.json"
 np.savez(patch,vertices6=np.column_stack((a["positions"],a["normals"])),binding_indices=a["binding_indices"],weights=a["weights"],faces=f)
 report.write_text(json.dumps({"stable_id":sid,"source_payload_path":str(src),"source_sha256":data["sha256"],"operations":ops,"all_vertex_fields_and_body_bindings_preserved":True,"construction_positions":"actual accepted step0 GPU Float32 world positions; triangle index streams authenticated against current tissue owner","native_admission":False,"anatomical_admission":False},indent=2)+"\n")
 replacements.append((sid,patch,report));print(json.dumps({"row":sid,"retired":len(a["faces"])-len(f),"elapsed_s":time.monotonic()-t}),flush=True)
result=compose_reference_receipt(p/"tissue-native-precision-retirement-002/composed-receipt/resting-anatomy-receipt.json",out/"composed-receipt",replacements,extensions={})
after=_read_nhtiss4(Path(result["payload_path"]));same=0
for a,b in zip(data["records"],after["records"]):
 x,y=_row_slices(data,a),_row_slices(after,b)
 assert all(x[k]==y[k] for k in ("vertex_bytes","binding_bytes","vertex_count","binding_count","stable_id","layer"))
 if int(a[6]) not in set(map(int,inventory)):assert x["local_faces"]==y["local_faces"]
 same+=1
result.update(all_rows_vertex_and_binding_bytes_exact=same,source_sha256=data["sha256"],source_owner_revision="40c2d287",elapsed_s=time.monotonic()-t)
(out/"result.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2),flush=True)
