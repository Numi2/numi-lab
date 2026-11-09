#!/usr/bin/env python3
"""Start the existing bounded spatial study only after rebound 003 exits."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
e = root / "docs/evidence/midpoint-first-failure-20261109"
script = root / "tests/contact/run_large_spatial_continuation_003.py"
receipt = e / "spatial-after-rebound-003.json"
if len(sys.argv) != 2 or not sys.argv[1].isdigit() or receipt.exists():
    raise SystemExit("fresh receipt and n64 supervisor pid required")
pid = int(sys.argv[1])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
files = [Path(__file__).resolve(), script, root / "tests/contact/large_spatial_reference.py"]
hashes = {str(path): sha(path) for path in files}
record = {"n64_pid": pid, "inputs_sha256": hashes, "status": "waiting"}
def save():
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(receipt)
save()
try:
    deadline = time.monotonic() + 7 * 24 * 3600
    while True:
        command = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                                 capture_output=True, text=True).stdout.strip()
        if not command:
            break
        if "run_spatial_n64_rebound_003.py" not in command:
            raise RuntimeError("n64 supervisor identity changed")
        if time.monotonic() > deadline:
            raise RuntimeError("wait limit")
        time.sleep(15)
    if any(sha(Path(path)) != value for path, value in hashes.items()):
        raise RuntimeError("study code changed while waiting")
    prior = e / "n64-rebound-003-supervisor.json"
    data = json.loads(prior.read_text())
    record["n64_receipt_sha256"] = sha(prior)
    if data.get("reason") != "both hash-bound references completed and analyzed":
        raise RuntimeError("n64 prerequisite did not complete")
    log = e / "spatial-after-rebound-003-run.log"
    if log.exists():
        raise RuntimeError("output exists")
    record["status"] = "running"
    save()
    with log.open("x") as output:
        result = subprocess.run([sys.executable, str(script)], cwd=root,
                                stdout=output, stderr=subprocess.STDOUT)
    record.update(status="finished_inspect_spatial_receipt", returncode=result.returncode,
                  log_sha256=sha(log))
except Exception as error:
    record.update(status="stopped", error=repr(error))
save()
raise SystemExit(0 if record["status"] == "finished_inspect_spatial_receipt" else 1)
