"""Reproduce the retained constrained joint-weight repair through the existing NHSKIN owner."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,numpy as np
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument("--repo",type=Path,required=True)
ap.add_argument("--parameters",type=Path,required=True)
ap.add_argument("--output",type=Path,required=True)
args=ap.parse_args()
if args.output.exists():raise RuntimeError("refusing to overwrite evidence")
spec=importlib.util.spec_from_file_location("owner",args.repo/"tools/rebind_human_skin_reference.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=json.loads(args.parameters.read_text())
raw=Path(p["base_path"]).read_bytes();pose_raw=Path(p["pose_path"]).read_bytes()
assert hashlib.sha256(raw).hexdigest()==p["base_sha256"]
assert hashlib.sha256(pose_raw).hexdigest()==p["pose_sha256"]
payload=m.decode(raw);a,b=m.reference_transforms(payload,json.loads(pose_raw),p["root_lift_m"])
receipts=[]
for step in p["steps"]:
 weights=payload["weights"].copy()
 weights[np.array(step["rows"],dtype=int)]=np.array(step["weights"],dtype="<f4")
 raw,receipt=m.rebind(payload,weights,a,b)
 receipt["sha256"]=hashlib.sha256(raw).hexdigest()
 assert receipt["sha256"]==step["expected_sha256"],receipt
 payload=m.decode(raw);receipts.append(receipt)
args.output.write_bytes(raw)
print(json.dumps(dict(final_sha256=hashlib.sha256(raw).hexdigest(),steps=receipts),indent=2))
