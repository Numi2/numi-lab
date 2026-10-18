import sys,json,hashlib,time
from pathlib import Path
import numpy as np
sys.path.insert(0,"/Users/n/numi-human-closed-outer-skin-20261017/src")
from numilab_human.resting_pleura_proxy import _parse_payload
from numilab_human.surface_precision_retriangulation import collapse_closed_component_edge_preserving_allocations
from numilab_human.resting_intercostal_mesh_arrangement import encode_payload_with_row_replacements
from numilab_human import composed_bone_body_frame as cb
root=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-suspension-reference-probe-20261018-001")
out=root/"intercostal313-subulp-edge-retirement-001"; out.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
# Reuse retained read-only MRVPACK scanner definitions without its historical main.
scanner=root.parent/"D/rope-suspension-reference-probe-20261018-001/precision-preflight-001/scan_precision.py"
scope={"__file__":str(scanner)}
exec(compile(scanner.read_text().split("\nstarted = time.monotonic()")[0],str(scanner),"exec"),scope)
pack=root/"attempt-011/native-run/rejected-surface-audit/step-31.mrvpack"
assert sha(pack.read_bytes())=="70c640f9650c11dbfe82bd43467a0eafcf782370814c95b8c953f8c9972e9b87"
t=time.monotonic(); scan=scope["scan_pack"](pack,"rejected_physical_step31")
scan["surfaces"]={f"{k[0]}:{k[1]}":v for k,v in scan["surfaces"].items()}
(out/"rejected-step31-all-surface-scan.json").write_text(json.dumps(scan,indent=2)+"\n")
zero={k:v["zero_area_face_ordinals"] for k,v in scan["surfaces"].items() if v["zero_area_face_ordinals"]}
print(json.dumps({"scan_s":time.monotonic()-t,"surfaces":len(scan["surfaces"]),"zero_area_surfaces":zero}),flush=True)
src=root.parent/"pons323-348-current972-existing-owner-rebind-001/native-run/resting-thorax.nhanatomy"
raw=src.read_bytes(); assert sha(raw)=="d516b524ebe01f09c1e8a2586a5e52bd3143bc9c815bc6e007f8a754ca834be2"
_,_,rows=_parse_payload(raw); r=rows[313]; vertices=np.asarray(r["vertices6"]); faces=np.asarray(r["faces"])
assert faces[101289].tolist()==[103537,101843,101839]
bones=root/"tendon-owner-rebind-validation-001/stable49-owner-replay-002/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones"
br=bones.read_bytes(); h,recs,vo,io=cb._decode_bones(br); bone=next(x for x in recs if x[5]==49)
bv=np.frombuffer(br,dtype="<f4",count=h[3]*6,offset=vo).reshape(-1,6)[bone[1]:bone[1]+bone[2],:3]
dist=np.linalg.norm(vertices[[103537,101843],:3].astype(np.float64)[:,None,:]-bv[None,:,:].astype(np.float64),axis=2)
matched=[int(x) for x in dist.argmin(axis=1)]
edge={"nha_slots":[103537,101843],"nha_positions":vertices[[103537,101843],:3].tolist(),"closest_bone49_slots":matched,"bone_positions":bv[matched].tolist(),"distances_m":dist.min(axis=1).tolist()}
(out/"edge-correspondence.json").write_text(json.dumps(edge,indent=2)+"\n");print(json.dumps(edge),flush=True)
t=time.monotonic()
newfaces,proof=collapse_closed_component_edge_preserving_allocations(vertices[:,:3].copy(),faces,identity=(51010,313),keep_vertex=101843,remove_vertex=103537,max_endpoint_displacement_m=2e-8,max_abs_volume_delta_m3=2e-12)
replacement=dict(r);replacement["faces"]=newfaces
payload,encoding=encode_payload_with_row_replacements(raw,{313:replacement})
_,_,after=_parse_payload(payload);assert np.array_equal(after[313]["vertices6"],vertices)
proof.update(source_payload_path=str(src),source_payload_sha256=sha(raw),output_payload_sha256=sha(payload),encoding=encoding,edge_correspondence=edge,elapsed_s=time.monotonic()-t)
(out/"resting-thorax.nhanatomy").write_bytes(payload)
(out/"repair-report.json").write_text(json.dumps(proof,indent=2)+"\n")
print(json.dumps({"candidate":str(out/"resting-thorax.nhanatomy"),"sha256":sha(payload),"elapsed_s":proof["elapsed_s"],"retired":proof["collapse"]["deleted_face_origins"]}),flush=True)
