#!/usr/bin/env python3
"""Wait for the cost study and apply the device-timestep correction."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
e = root / "docs/evidence/midpoint-first-failure-20261109"
script = root / "tests/contact/audit_cost_physical_time.py"
receipt = e / "cost-physical-time-001.json"
if len(sys.argv) != 2 or not sys.argv[1].isdigit() or receipt.exists():
    raise SystemExit("fresh receipt and coordinator pid required")
pid = int(sys.argv[1])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
files = [Path(__file__).resolve(), script, root / "tests/contact/cost_summary.py"]
hashes = {str(path): sha(path) for path in files}
result = {"status": "waiting", "coordinator_pid": pid, "inputs_sha256": hashes}
def save():
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(receipt)
save()
try:
    deadline = time.monotonic() + 7 * 24 * 3600
    while True:
        actual = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout.strip()
        if not actual:
            break
        if "run_recovery_sequence_002.py" not in actual:
            raise RuntimeError("coordinator identity changed")
        if time.monotonic() > deadline:
            raise RuntimeError("wait limit")
        time.sleep(15)
    if any(sha(Path(path)) != value for path, value in hashes.items()):
        raise RuntimeError("time-audit input changed")
    summary = e / "cost-fair-n2-002-summary.json"
    if not summary.exists() or not json.loads(summary.read_text()).get("complete"):
        raise RuntimeError("no completed cost comparison")
    check = subprocess.run([sys.executable, str(script), str(e), "cost-fair-n2-002"],
                           cwd=root, capture_output=True, text=True)
    result.update(status="passed" if check.returncode == 0 else "failed", returncode=check.returncode,
                  audit=json.loads(check.stdout), stderr=check.stderr, original_summary_sha256=sha(summary))
except Exception as error:
    result.update(status="stopped", error=repr(error))
save()
raise SystemExit(0 if result["status"] == "passed" else 1)
