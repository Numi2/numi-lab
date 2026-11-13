from pathlib import Path
import json,sys
p=Path("artifacts/crow-v15-coordinated-sweep-bank-20261113")/sys.argv[1];a=json.loads((p/"gate.json").read_text());b=json.loads((p/"sweep-gate.json").read_text());out={"existing_physical_and_signed_heading_gate":a["passed_identification"],"sweep_target_gate":b["passed"],"passed":a["passed_identification"] and b["passed"],"scope":"Used-seed coordinated joint-target turn screen only; not flight or biological qualification."};(p/"combined-gate.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
