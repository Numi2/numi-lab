import sys,json,time,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,"/Users/n/numi-human-closed-outer-skin-20261017/src")
from numilab_human.passive_attachment_composition import _read_nhtiss4,_nhtiss_row_arrays,_row_slices
from numilab_human.surface_precision_retriangulation import collapse_closed_component_edge_preserving_allocations
from numilab_human.passive_reference_receipt_composition import compose_reference_receipt
p=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-suspension-reference-probe-20261018-001")
src=p.parent/"deltoid75-78-existing-shared-parent-source-824/conformed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
out=p/"tissue-subulp-edge-retirement-001";out.mkdir(exist_ok=False); sha=lambda b:hashlib.sha256(b).hexdigest()
data=_read_nhtiss4(src);assert data["sha256"]=="ff8db8396bf49b307ca50ad85f22f19c1167a0f43a625c6439a05b2548bd9126"
inventory=json.loads((p/"tissue-source-sub50nm-edge-inventory.json").read_bytes()); replacements=[];t=time.monotonic()
for sid in (156,157):
 row=next(r for r in data["records"] if int(r[6])==sid);a=_nhtiss_row_arrays(data,row);f=a["faces"];orig=np.arange(len(f));ops=[]
 for x,y in inventory[str(sid)]["edges"]:
  keep,remove=sorted((int(x),int(y)),key=lambda i:tuple(float(z) for z in a["positions"][i]))
  nf,proof=collapse_closed_component_edge_preserving_allocations(a["positions"].copy(),f,identity=(51005,sid),keep_vertex=keep,remove_vertex=remove,max_endpoint_displacement_m=5e-8,max_abs_volume_delta_m3=2e-12)
  proof["source_payload_face_origins"]=orig[proof["kept_parent_face_origins"]].tolist();orig=orig[proof["kept_parent_face_origins"]];f=nf;ops.append(proof)
 patch=out/f"row{sid}.npz";report=out/f"row{sid}-report.json"
 np.savez(patch,vertices6=np.column_stack((a["positions"],a["normals"])),binding_indices=a["binding_indices"],weights=a["weights"],faces=f)
 report.write_text(json.dumps({"stable_id":sid,"source_payload_path":str(src),"source_sha256":data["sha256"],"operations":ops,"all_vertex_fields_and_body_bindings_preserved":True,"native_admission":False,"anatomical_admission":False},indent=2)+"\n")
 replacements.append((sid,patch,report));print(json.dumps({"row":sid,"retired":len(a["faces"])-len(f),"elapsed_s":time.monotonic()-t}),flush=True)
result=compose_reference_receipt(p/"head-runtime-receipt-rebind-001/resting-anatomy-receipt.json",out/"composed-receipt",replacements,extensions={})
after=_read_nhtiss4(Path(result["payload_path"]));same=0
for a,b in zip(data["records"],after["records"]):
 x,y=_row_slices(data,a),_row_slices(after,b)
 assert all(x[k]==y[k] for k in ("vertex_bytes","binding_bytes","vertex_count","binding_count","stable_id","layer"))
 if int(a[6]) not in (156,157):assert x["local_faces"]==y["local_faces"]
 same+=1
result.update(all_rows_vertex_and_binding_bytes_exact=same,source_sha256=data["sha256"],source_owner_revision="40c2d287",elapsed_s=time.monotonic()-t)
(out/"result.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2),flush=True)
