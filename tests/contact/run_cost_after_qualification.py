#!/usr/bin/env python3
"""Run the registered cost study only after both live references finish."""
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from cost_provenance import verify_accuracy_provenance
from cost_selection import select_largest_certified

root = pathlib.Path(__file__).resolve().parents[2]
evidence = root / "docs/evidence/midpoint-first-failure-20261109"
build = pathlib.Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-be-precision-005")
manifest = evidence / "be-precision-005-inputs.sha256"
receipt = evidence / "cost-after-qualification-001-supervisor.json"
if receipt.exists():
    raise SystemExit("supervisor receipt already exists")
files = [pathlib.Path(__file__).resolve(), root / "tests/contact/run_accuracy_adjusted_cost.py",
         root / "tests/contact/cost_provenance.py", root / "tests/contact/cost_selection.py",
         root / "tests/contact/cost_summary.py", root / "tests/contact/native_trace_analysis.py", manifest]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
hashes = {str(path): digest(path) for path in files}
start = time.monotonic()
record = {"input_sha256": hashes, "wait_limit_seconds": 48 * 3600,
          "scope": "conditional launch only; accuracy and timing are established by separate retained evidence"}
def finish(reason, **values):
    record.update(reason=reason, wait_seconds=time.monotonic()-start, **values)
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return 1

def process_command(pid):
    r = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""

while True:
    cpu = process_command(23698)
    be = process_command(25005)
    if cpu and "numi_contact_fem_fp64_reference_n128" not in cpu:
        raise SystemExit(finish("reference process identity changed", observed=cpu))
    if be and "run_be_compressed_ladder.py" not in be:
        raise SystemExit(finish("control supervisor identity changed", observed=be))
    if not cpu and not be:
        break
    if time.monotonic() - start > record["wait_limit_seconds"]:
        raise SystemExit(finish("reference wait limit"))
    time.sleep(15)
if any(digest(path) != hashes[str(path)] for path in files):
    raise SystemExit(finish("study code or frozen manifest changed while waiting"))
try:
    verify = lambda e: verify_accuracy_provenance(e, manifest, build)
    midpoint, ml = select_largest_certified(evidence, "stage-ladder-n2-h*", "ipc_area_midpoint", verify)
    control, bl = select_largest_certified(evidence, "be-precision-006-n2-h*", "ipc_area", verify)
except (ValueError, OSError, KeyError) as error:
    raise SystemExit(finish("no eligible completed control", error=repr(error)))
command = [sys.executable, str(root / "tests/contact/run_accuracy_adjusted_cost.py"),
           midpoint, control, "cost-fair-n2-001"]
record.update(selected_midpoint=midpoint, selected_control=control, midpoint_ladder=ml,
              control_ladder=bl, command=command)
print(json.dumps(record), flush=True)
result = subprocess.run(command, cwd=root)
record.update(reason="cost runner exited", returncode=result.returncode,
              wait_seconds=time.monotonic()-start)
receipt.write_text(json.dumps(record, indent=2) + "\n")
raise SystemExit(result.returncode)
