#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, subprocess, time
D=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b109-114/attempt-005-cross-parent-edge-flip-001")
PYTHON="/Users/numi/numi-human-anatomy-finish-1378/.venv/bin/python"
CHECK=D/"exact_local_check_corrected.py";REPORT=D/"exact-local-report-corrected.json";RECEIPT=D/"execution-corrected.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
if RECEIPT.exists():raise FileExistsError(RECEIPT)
t=time.monotonic();r=subprocess.run([PYTHON,str(CHECK)],check=False);rep=json.loads(REPORT.read_text()) if REPORT.exists() else {}
status="COMPLETE" if r.returncode==0 and rep.get("status")=="LOCAL_EXACT_CHECK_PASS" else "REJECTED_OR_FAILED"
out={"schema":"numi.human.b109_114.attempt005.corrected_exact_execution.v1","status":status,"serialized_by":"/tmp/numi-human-geometry-slot.py","checker_path":str(CHECK),"checker_sha256":sha(CHECK),"checker_returncode":r.returncode,"checker_status":rep.get("status"),"exact_report_path":str(REPORT),"exact_report_sha256":sha(REPORT) if REPORT.exists() else None,"elapsed_seconds":time.monotonic()-t,"native_admission":False,"physical_steps":0}
RECEIPT.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps(out,indent=2))
if status!="COMPLETE":raise SystemExit(2)
