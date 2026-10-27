#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, subprocess, sys, time
D=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b109-114/attempt-005-cross-parent-edge-flip-001")
PYTHON="/Users/numi/numi-human-anatomy-finish-1378/.venv/bin/python"
PREP=D/"prepare_two_flips.py";CHECK=D/"exact_local_check.py";RECEIPT=D/"execution.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
if RECEIPT.exists():raise FileExistsError(RECEIPT)
start=time.monotonic();res=[]
for name,path in (("owner_flip_and_lineage",PREP),("exact_local_check",CHECK)):
 t=time.monotonic();r=subprocess.run([PYTHON,str(path)],check=False)
 res.append({"stage":name,"script":str(path),"script_sha256":sha(path),"returncode":r.returncode,"elapsed_seconds":time.monotonic()-t})
 if r.returncode:break
 if name=="exact_local_check":
  report=json.loads((D/"exact-local-report.json").read_text())
  if report.get("status")!="LOCAL_EXACT_CHECK_PASS":break
report={"schema":"numi.human.b109_114.attempt005.serialized_execution.v1","status":"COMPLETE" if len(res)==2 and res[-1]["returncode"]==0 and json.loads((D/"exact-local-report.json").read_text()).get("status")=="LOCAL_EXACT_CHECK_PASS" else "REJECTED_OR_FAILED","serialized_by":"/tmp/numi-human-geometry-slot.py","stages":res,"elapsed_seconds":time.monotonic()-start,"geometry_admitted":False,"physical_steps":0}
RECEIPT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2))
if report["status"]!="COMPLETE":raise SystemExit(2)
