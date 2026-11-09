#!/usr/bin/env python3
"""Audit completed control traces after the timing coordinator has stopped."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
e = root / "docs/evidence/midpoint-first-failure-20261109"
reference = root / "docs/evidence/midpoint-spatial-cost-dynamic-20261108/rk4-n2-h025.json"
script = root / "tests/contact/strict_native_trace_audit.py"
receipt = e / "strict-native-after-cost-002.json"
if len(sys.argv) != 2 or not sys.argv[1].isdigit() or receipt.exists():
    raise SystemExit("fresh receipt and cost coordinator pid required")
pid = int(sys.argv[1])
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
files = [Path(__file__).resolve(), script, root / "tests/contact/large_spatial_reference.py", reference]
hashes = {str(p): sha(p) for p in files}
result = {"status": "waiting", "coordinator_pid": pid, "inputs_sha256": hashes, "audits": []}
def save():
    tmp = receipt.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(receipt)
save()
try:
    deadline = time.monotonic() + 7 * 24 * 3600
    while True:
        actual = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                                capture_output=True, text=True).stdout.strip()
        if not actual:
            break
        if "run_recovery_sequence_003.py" not in actual:
            raise RuntimeError("cost coordinator identity changed")
        if time.monotonic() > deadline:
            raise RuntimeError("wait limit")
        time.sleep(15)
    if any(sha(Path(path)) != h for path, h in hashes.items()):
        raise RuntimeError("audit input changed while waiting")
    executions = sorted(e.glob("be-precision-008-n2-h*-execution.json"))
    for execution in executions:
        meta = json.loads(execution.read_text())
        if meta.get("all_gates_pass") is not True:
            continue
        prefix = str(execution)[:-len("-execution.json")]
        trace = Path(prefix + ".json.gz")
        check = subprocess.run([sys.executable, str(script), str(trace), str(execution), str(reference)],
                               capture_output=True, text=True)
        result["audits"].append({"execution": execution.name, "returncode": check.returncode,
                                 "stdout": check.stdout, "stderr": check.stderr})
        save()
    result["status"] = "passed" if result["audits"] and all(x["returncode"] == 0 for x in result["audits"]) else "no_passing_audits"
except Exception as error:
    result.update(status="stopped", error=repr(error))
save()
raise SystemExit(0 if result["status"] == "passed" else 1)
