import sys,json,hashlib,time,subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0,"/Users/n/numi-human-closed-outer-skin-20261017/src")
from numilab_human import composed_bone_body_frame as cb
from numilab_human.model import numi_human_tendon_attachment_envelope_payload
from numilab_human.resting_pleura_proxy import _parse_payload
from numilab_human.surface_precision_retriangulation import collapse_closed_component_edge_preserving_allocations
from numilab_human.resting_intercostal_mesh_arrangement import encode_payload_with_row_replacements
from numilab_human.resting_common_field_passive_rebind import rebind_passive_rows
p=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-suspension-reference-probe-20261018-001")
out=p/"paired-rib49-intercostal-precision-009";out.mkdir(exist_ok=False);sha=lambda b:hashlib.sha256(b).hexdigest();t=time.monotonic()
parent=p/"tendon-owner-rebind-validation-001/stable49-owner-replay-002/bodyparts3d-myosim-major-bones";bp=parent/"bodyparts3d-myosim-major-bones.nhbones";mp=bp.with_suffix(".manifest.json");br=bp.read_bytes();pm=mp.read_bytes();assert sha(br)=="01c39cd8da439e2fed0770ff5d67e7028ea0d98d18a453721c26911dfd42d6bd"
params={"stable_id":49,"keep_vertex":10143,"remove_vertex":10144,"flip_edges":[],"max_endpoint_displacement_m":5e-8,"minimum_altitude_m":1e-12,"max_abs_volume_delta_m3":2e-12}
nb,proof=cb.repair_bone_source_precision_edges(br,**params);bm=json.loads(pm);bm["payload"].update(sha256=sha(nb),bytes=len(nb),index_count=cb._BONE_HEADER.unpack_from(nb)[4]);bm["source_precision_retriangulation"]={"input_artifact":str(parent),"input_manifest_sha256":sha(pm),"input_payload_sha256":sha(br),"parameters":params,"proof":proof}
bd=out/"bodyparts3d-myosim-major-bones";bd.mkdir();(bd/bp.name).write_bytes(nb);(bd/mp.name).write_text(json.dumps(bm,indent=2,sort_keys=True)+"\n");validation=cb.validate_composed_rib_reframe(bd,bm,nb);print("bone owner validation passed",flush=True)
h,rs,vo,io=cb._decode_bones(br);r=next(x for x in rs if x[5]==49);v=np.frombuffer(br,"<f4",h[3]*6,vo).reshape(-1,6)[r[1]:r[1]+r[2],:3]
receipt_src=p/"extensor-source-precision-separation-008/composed-receipt/resting-anatomy-receipt.json";receipt_raw=receipt_src.read_bytes();rec=json.loads(receipt_raw);nha=Path(rec["payload"]["path"]);nr=nha.read_bytes();_,_,rows=_parse_payload(nr);row=rows[313]
ids=[np.flatnonzero(np.all(row["vertices6"][:,:3]==v[i],axis=1)).astype(int).tolist() for i in (10143,10144)];assert all(len(x)==1 for x in ids)
keep,remove=ids[0][0],ids[1][0];nf,ip=collapse_closed_component_edge_preserving_allocations(row["vertices6"][:,:3].copy(),row["faces"],identity=(51010,313),keep_vertex=keep,remove_vertex=remove,max_endpoint_displacement_m=5e-8,max_abs_volume_delta_m3=2e-12)
newrow=dict(row);newrow["faces"]=nf;newnha,encoding=encode_payload_with_row_replacements(nr,{313:newrow});nha_out=out/"resting-thorax.nhanatomy";nha_out.write_bytes(newnha)
rec,rebind=rebind_passive_rows(rec,nr,newnha,changed_ids=(313,),output_path=str(nha_out))
maps={sid:rec["functional_bindings"][f"intercostal_313_rib{sid}_motion_attachment"]["surface"]["interface_local_vertex_ids"] for sid in (48,49,50)}
view={"payload":rec["payload"],"functional_bindings":rec["functional_bindings"],"provenance":{"bones_payload_sha256":sha(br)}}
cb.bind_intercostal_runtime_attachment_metadata(view,newrow,maps,nha_stable_id=313,nha_payload_sha256=sha(newnha),bones_payload_sha256=sha(nb))
cb.validate_intercostal_runtime_attachment_metadata(view,newrow,maps,nha_stable_id=313,nha_payload_sha256=sha(newnha),bones_payload_sha256=sha(nb))
rec["provenance"]["bones_payload"]=str(bd/bp.name);rec["provenance"]["bones_payload_sha256"]=sha(nb);sidecars=cb._preserve_receipt_common_field_sidecars(rec,receipt_src.parent,out)
evidence={"prior_receipt_path":str(receipt_src),"prior_receipt_sha256":sha(receipt_raw),"bones_source_sha256":sha(br),"bones_output_sha256":sha(nb),"bone_repair":proof,"nha_source_sha256":sha(nr),"nha_output_sha256":sha(newnha),"paired_nha_slots":[keep,remove],"intercostal_repair":ip,"nha_encoding":encoding,"existing_common_owner_rebind":rebind,"common_sidecars":sidecars,"physical_state_or_force_owner_changed":False,"native_admission":False}
rec["provenance"]["paired_rib49_intercostal_precision_009"]=evidence;receipt_out=out/"resting-anatomy-receipt.json";receipt_out.write_text(json.dumps(rec,separators=(",",":"),allow_nan=False)+"\n")
td=out/"compiled-tendon";td.mkdir();tm=numi_human_tendon_attachment_envelope_payload(p.parent/"current-manifest-recovery-004/recovered-artifact",bd,td,0.012,0.012,4.,True,0.025);tp=td/tm["payload"]["file"]
baseline=p/"tendon-owner-rebind-validation-001/stable49-owner-replay-002/compiled-tendon-002/numi-human-tendon-attachments.nhtendon";btm=json.loads(baseline.with_suffix(".manifest.json").read_bytes());old=baseline.read_bytes();new=tp.read_bytes();assert old[:112]==new[:112] and old[144:]==new[144:];assert tm["endpoints"]==btm["endpoints"];assert tm["coverage"]==btm["coverage"]
result={"bone_payload_path":str(bd/bp.name),"bone_sha256":sha(nb),"bone_manifest_sha256":sha((bd/mp.name).read_bytes()),"nha_payload_path":str(nha_out),"nha_sha256":sha(newnha),"receipt_path":str(receipt_out),"receipt_sha256":sha(receipt_out.read_bytes()),"tendon_path":str(tp),"tendon_sha256":sha(new),"tendon_bytes_changed_only_bone_hash112_143":True,"endpoints_exact":len(tm["endpoints"]),"coverage_exact":True,"elapsed_s":time.monotonic()-t}
(out/"repair-report.json").write_text(json.dumps(evidence,indent=2)+"\n");(out/"result.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2),flush=True)
