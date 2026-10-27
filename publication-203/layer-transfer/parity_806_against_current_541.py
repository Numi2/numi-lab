#!/usr/bin/env python3
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
OWNER=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/physical-layer-diagnostic-001/transfer_current.py")
C=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/combined-d071-b109-b61-001/candidate.npz")
REF=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/combined-d071-b109-b61-001/skin-fat-layers-541.npz")
OUT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b65-99/attempt-024-topology-independent-layer-transfer-001/parity-owner-806-001")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location("current_541_owner",OWNER);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
m.CURRENT=C
m.EXPECTED["current"]=sha(C)
m.run(OUT)
with np.load(OUT/"skin-fat-layers-541.npz",allow_pickle=False) as got,np.load(REF,allow_pickle=False) as ref:
 rows=[]
 for k in ref.files:
  same=(k in got.files and got[k].dtype==ref[k].dtype and got[k].shape==ref[k].shape and np.ascontiguousarray(got[k]).tobytes()==np.ascontiguousarray(ref[k]).tobytes())
  row={"key":k,"exact_byte_array_match":bool(same)}
  if k in got.files and got[k].shape==ref[k].shape and got[k].dtype.kind in "fiu":
   row["max_abs_delta"]=float(np.max(np.abs(got[k].astype(np.float64)-ref[k].astype(np.float64)),initial=0.0))
  rows.append(row)
 parity={"candidate_path":str(C),"candidate_sha256":sha(C),"current_541_reference_path":str(REF),"current_541_reference_sha256":sha(REF),"owner_run_output_path":str(OUT/"skin-fat-layers-541.npz"),"owner_run_output_sha256":sha(OUT/"skin-fat-layers-541.npz"),"keys_compared":len(rows),"all_existing_artifact_arrays_exact_byte_equal":all(x["exact_byte_array_match"] for x in rows),"array_comparisons":rows}
 p=OUT/"parity.json";p.write_text(json.dumps(parity,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"parity_path":str(p),"parity_sha256":sha(p),"all_exact":parity["all_existing_artifact_arrays_exact_byte_equal"],"keys_compared":len(rows)},indent=2))
 if not parity["all_existing_artifact_arrays_exact_byte_equal"]: raise SystemExit(3)
