from __future__ import annotations
import argparse, hashlib, json, os, sys
import numpy as np
from pathlib import Path
sys.path.insert(0,"/Users/n/numi-human-closed-outer-skin-20261017/src")
from numilab_human.resting_common_field_passive_rebind import rebind_passive_rows
from numilab_human.composed_bone_body_frame import _preserve_receipt_common_field_sidecars,bind_intercostal_runtime_attachment_metadata,validate_intercostal_runtime_attachment_metadata
from numilab_human.resting_pleura_proxy import _parse_payload
p=argparse.ArgumentParser(); p.add_argument("candidate"); p.add_argument("proof"); p.add_argument("output"); a=p.parse_args()
root=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-suspension-reference-probe-20261018-001")
src=root/"intercostal313-runtime-receipt-rebind-001/resting-anatomy-receipt.json"
raw=src.read_bytes(); rec=json.loads(raw); base=Path(rec["payload"]["path"]); br=base.read_bytes()
candidate=Path(a.candidate).resolve(); cr=candidate.read_bytes(); proof=Path(a.proof).resolve(); pr=proof.read_bytes()
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha(br)=="e617564e9f1ce6097848098c2ce55c89493c16468b793e30f50223a240f60f7c"
outdir=Path(a.output).resolve(); outdir.mkdir(parents=True,exist_ok=False); out=outdir/"resting-anatomy-receipt.json"
rec,audit=rebind_passive_rows(rec,br,cr,changed_ids=(323,343,345,348),output_path=str(candidate))
_,_,before=_parse_payload(br); _,_,after=_parse_payload(cr)
assert np.array_equal(before[313]["vertices6"],after[313]["vertices6"]),"source attributes changed"
maps={sid:rec["functional_bindings"][f"intercostal_313_rib{sid}_motion_attachment"]["surface"]["interface_local_vertex_ids"] for sid in (48,49,50)}
bh=rec["functional_bindings"]["bones_payload_sha256"]
view={"payload":rec["payload"],"functional_bindings":rec["functional_bindings"],"provenance":{"bones_payload_sha256":bh}}
bind_intercostal_runtime_attachment_metadata(view,after[313],maps,nha_stable_id=313,nha_payload_sha256=sha(cr),bones_payload_sha256=bh)
validate_intercostal_runtime_attachment_metadata(view,after[313],maps,nha_stable_id=313,nha_payload_sha256=sha(cr),bones_payload_sha256=bh)
sidecars=_preserve_receipt_common_field_sidecars(rec,src.parent,outdir)
evidence={"source_receipt_path":str(src),"source_receipt_sha256":sha(raw),"source_payload_path":str(base),"source_payload_sha256":sha(br),"output_payload_path":str(candidate),"output_payload_sha256":sha(cr),"proof_path":str(proof),"proof_sha256":sha(pr),"existing_common_owner_audit":audit,"runtime_attachment_metadata_validated":True,"all_vertex_slots_and_attributes_preserved":True,"common_field_sidecars":sidecars,"physical_state_or_force_owner_changed":False,"anatomical_admission":False,"native_admission":False}
rec["provenance"]["head_rope_precision_rebind"]=evidence
with out.open("x") as f:
 json.dump(rec,f,separators=(",",":"),allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
evidence["output_receipt_path"]=str(out); evidence["output_receipt_sha256"]=sha(out.read_bytes())
(outdir/"rebind-report.json").write_text(json.dumps(evidence,indent=2)+"\n")
print(json.dumps(evidence,indent=2))
