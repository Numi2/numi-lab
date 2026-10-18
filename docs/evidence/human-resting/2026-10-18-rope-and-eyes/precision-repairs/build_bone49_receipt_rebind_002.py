from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-suspension-reference-probe-20261018-001")
sys.path.insert(0,"/Users/n/numi-human-closed-outer-skin-20261017/src")
from numilab_human.resting_pleura_proxy import _parse_payload
from numilab_human.composed_bone_body_frame import (_preserve_receipt_common_field_sidecars,bind_intercostal_runtime_attachment_metadata,validate_intercostal_runtime_attachment_metadata,validate_bone_source_precision_retriangulation)
sha=lambda b:hashlib.sha256(b).hexdigest()
src=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/pons323-348-current972-existing-owner-rebind-001/native-run-002/resting-anatomy-receipt.json")
nha=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/pons323-348-current972-existing-owner-rebind-001/native-run/resting-thorax.nhanatomy")
old=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rib50-published-source-owner-reproducer-952/compiled/bodyparts3d-myosim-major-bones.nhbones")
bd=ROOT/"tendon-owner-rebind-validation-001/stable49-owner-replay-002/bodyparts3d-myosim-major-bones"
bp=bd/"bodyparts3d-myosim-major-bones.nhbones"; mp=bd/"bodyparts3d-myosim-major-bones.manifest.json"; rp=bd/"repair-report.json"
outdir=ROOT/"bone49-runtime-receipt-rebind-002"; out=outdir/"resting-anatomy-receipt.json"; reportpath=outdir/"rebind-report.json"
if outdir.exists(): raise SystemExit("refusing to reuse output directory")
for p in (src,nha,old,bp,mp,rp):
 if not p.is_file(): raise SystemExit(f"missing input: {p}")
raw=src.read_bytes(); rec=json.loads(raw); nha_raw=nha.read_bytes(); old_raw=old.read_bytes(); bone_raw=bp.read_bytes(); manifest_raw=mp.read_bytes(); repair_raw=rp.read_bytes()
nh=sha(nha_raw); oh=sha(old_raw); bh=sha(bone_raw); mh=sha(manifest_raw); rh=sha(repair_raw)
pins={"nha":"d516b524ebe01f09c1e8a2586a5e52bd3143bc9c815bc6e007f8a754ca834be2","old":"6a1ece3a3a282873175356df7c3b0430e9657d3322fef1e992eba74b1d1e80bb","bones":"01c39cd8da439e2fed0770ff5d67e7028ea0d98d18a453721c26911dfd42d6bd","manifest":"db42bf860c5386c7063a1c61ad07a659439521a5b318f1e88292ddadbf4278cb","repair":"202f31ef0def7b3fc039cc317431cb25b235672969a8644896b0a9c102bbd2c6"}
if (nh,oh,bh,mh,rh)!=(pins["nha"],pins["old"],pins["bones"],pins["manifest"],pins["repair"]): raise SystemExit(f"input pin mismatch: {(nh,oh,bh,mh,rh)}")
if rec["payload"]["sha256"]!=nh or Path(rec["payload"]["path"]).resolve()!=nha.resolve(): raise SystemExit("receipt/NHA identity mismatch")
if rec["provenance"]["bones_payload_sha256"]!=oh or rec["functional_bindings"]["bones_payload_sha256"]!=oh: raise SystemExit("receipt/parent bone identity mismatch")
manifest=json.loads(manifest_raw)
srcdir,prior,source_raw,proof=validate_bone_source_precision_retriangulation(bd,manifest,bone_raw)
if sha(source_raw)!=oh or srcdir.resolve()!=old.parent.resolve(): raise SystemExit("precision owner did not replay from registered parent")
params=manifest["source_precision_retriangulation"]["parameters"]
if (params.get("stable_id"),params.get("keep_vertex"),params.get("remove_vertex"),params.get("flip_edges"))!=(49,7527,11530,[]): raise SystemExit("unexpected stable49 repair parameters")
_,_,rows=_parse_payload(nha_raw); row=rows.get(313)
if not isinstance(row,dict) or (row.get("body_index"),row.get("layer"),row.get("flags"))!=(20,1,0): raise SystemExit("wrong typed row313")
bindings=rec["functional_bindings"]
keys=sorted(k for k in bindings if isinstance(k,str) and k.startswith("intercostal_313_rib") and k.endswith("_motion_attachment"))
expected=[f"intercostal_313_rib{x}_motion_attachment" for x in (48,49,50)]
if keys!=expected: raise SystemExit(f"unexpected row313 attachment inventory: {keys}")
maps={}
for sid in (48,49,50):
 a=bindings[f"intercostal_313_rib{sid}_motion_attachment"]; s=a.get("surface",{}); ids=s.get("interface_local_vertex_ids")
 if (s.get("semantic"),s.get("stable_id"),s.get("body_index"),s.get("layer"))!=(51010,313,20,1): raise SystemExit(f"rib{sid} attachment identity mismatch")
 if not isinstance(ids,list) or not ids or any(type(x) is not int for x in ids) or len(ids)!=len(set(ids)): raise SystemExit(f"rib{sid} map malformed")
 if (a.get("bones_payload_sha256"),a.get("anatomy_payload_sha256"))!=(oh,nh): raise SystemExit(f"rib{sid} old attachment pin mismatch")
 maps[sid]=ids
outdir.mkdir(parents=True,exist_ok=False)
sidecars=_preserve_receipt_common_field_sidecars(rec,src.parent,outdir)
view={"payload":rec["payload"],"functional_bindings":bindings,"provenance":{"bones_payload_sha256":oh}}
bind_intercostal_runtime_attachment_metadata(view,row,maps,nha_stable_id=313,nha_payload_sha256=nh,bones_payload_sha256=bh)
validate_intercostal_runtime_attachment_metadata(view,row,maps,nha_stable_id=313,nha_payload_sha256=nh,bones_payload_sha256=bh)
if view["provenance"]["bones_payload_sha256"]!=bh: raise SystemExit("existing owner failed to bind candidate bones")
oldpath=rec["provenance"].get("bones_payload")
if not isinstance(oldpath,str) or Path(oldpath).resolve()!=old.resolve(): raise SystemExit(f"unexpected active parent bone path: {oldpath}")
rec["provenance"]["bones_payload"]=str(bp); rec["provenance"]["bones_payload_sha256"]=bh
rec["provenance"]["registered_bone_source_rebind"]={
 "schema":"numi.human.registered-bone-source-rebind.v1","source_receipt_path":str(src),"source_receipt_sha256":sha(raw),
 "registered_nha_path":str(nha),"registered_nha_sha256":nh,"prior_bones_path":oldpath,"prior_bones_sha256":oh,
 "candidate_bones_path":str(bp),"candidate_bones_sha256":bh,"candidate_manifest_path":str(mp),"candidate_manifest_sha256":mh,
 "candidate_precision_report_path":str(rp),"candidate_precision_report_sha256":rh,
 "precision_owner_replayed_exactly":True,"runtime_attachment_owner":"bind_intercostal_runtime_attachment_metadata",
 "runtime_attachment_metadata_validated":True,"rebound_fields":["provenance.bones_payload_sha256","functional_bindings.bones_payload_sha256",
 "functional_bindings.intercostal_313_rib48_motion_attachment.bones_payload_sha256",
 "functional_bindings.intercostal_313_rib49_motion_attachment.bones_payload_sha256",
 "functional_bindings.intercostal_313_rib50_motion_attachment.bones_payload_sha256"],
 "common_field_sidecars_sha_verified_and_copied":sidecars,"physical_state_or_force_owner_changed":False,
 "anatomical_admission":False,"native_admission":False}
tmp=out.with_suffix(".json.tmp")
with tmp.open("x",encoding="utf-8") as f:
 json.dump(rec,f,ensure_ascii=False,allow_nan=False,separators=(",",":")); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
outsha=sha(out.read_bytes()); written=json.loads(out.read_bytes())
if written["payload"]["sha256"]!=nh or written["provenance"]["bones_payload_sha256"]!=bh or written["provenance"]["bones_payload"]!=str(bp) or written["functional_bindings"]["bones_payload_sha256"]!=bh: raise SystemExit("written receipt active identity reread failed")
for sid in (48,49,50):
 a=written["functional_bindings"][f"intercostal_313_rib{sid}_motion_attachment"]
 if a.get("bones_payload_sha256")!=bh or a.get("anatomy_payload_sha256")!=nh: raise SystemExit(f"written rib{sid} pin mismatch")
report={"schema":"numi.human.registered-bone-source-rebind-report.v1","status":"PASS_EXISTING_RUNTIME_OWNER_REBIND_SOURCE_ONLY_NATIVE_MOTION_NOT_ADMITTED",
 "source_receipt_path":str(src),"source_receipt_sha256":sha(raw),"output_receipt_path":str(out),"output_receipt_sha256":outsha,
 "nha_sha256":nh,"prior_bones_sha256":oh,"candidate_bones_sha256":bh,"candidate_manifest_sha256":mh,
 "candidate_precision_report_sha256":rh,"attachment_rows":[313],"rib_ids":[48,49,50],"active_bone_hash_fields_rebound":5,
 "common_field_sidecars":sidecars,"full_nha_geometry_changed":False,"physical_state_or_force_owner_changed":False,
 "anatomical_admission":False,"native_admission":False}
with reportpath.open("x",encoding="utf-8") as f:
 json.dump(report,f,indent=2); f.write("\n")
print(json.dumps(report,indent=2))
