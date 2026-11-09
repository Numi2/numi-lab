#!/usr/bin/env python3
"""Audit the retained qualifying midpoint trace without assigning cost benefit."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from large_spatial_reference import read_reference
from strict_native_trace_audit import check_rows, finite_tree

root = Path(__file__).resolve().parents[2]
e = root / "docs/evidence/midpoint-first-failure-20261109"
reference = root / "docs/evidence/midpoint-spatial-cost-dynamic-20261108/rk4-n2-h025.json"
prefix = e / "stage-ladder-n2-h015625"
paths = [Path(str(prefix) + suffix) for suffix in (".json", "-execution.json", "-analysis.json")]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
record = json.loads(paths[1].read_text())
data = json.loads(paths[0].read_text())
analysis = json.loads(paths[2].read_text())
if (record.get("all_gates_pass") is not True or record.get("returncode") != 0 or
    record.get("timed_out") or data.get("arm") != "ipc_area_midpoint" or
    data.get("divisions") != 2 or data.get("h") != .000015625 or
    data.get("accepted_steps") != 640 or data.get("rejected_steps") != 0 or
    len(data.get("traces", [])) != 640 or not data.get("restore_replay") or
    not data.get("continuation_replay") or not data.get("forced_rollback_exact") or
    not data.get("identity_mismatch_rejected") or not finite_tree(data)):
    raise SystemExit("incomplete or nonfinite midpoint evidence")
check_rows(data["traces"], data["h"])
read_reference(reference, 2, .00000025)
for path in paths:
    relative = str(path.relative_to(root))
    published = subprocess.check_output(["git", "rev-parse", "e94999516:" + relative], cwd=root, text=True).strip()
    local = subprocess.check_output(["git", "hash-object", str(path)], cwd=root, text=True).strip()
    if published != local:
        raise SystemExit("retained midpoint evidence changed: " + relative)
check = subprocess.run([sys.executable, str(root / "tests/contact/native_trace_analysis.py"), str(paths[0]), str(reference)],
                       cwd=root, capture_output=True, text=True)
if check.returncode != 0 or json.loads(check.stdout) != analysis:
    raise SystemExit("retained midpoint analysis did not reproduce")
print(json.dumps({"valid": True, "steps": len(data["traces"]), "h": data["h"],
                  "published_revision": "e94999516", "input_sha256": {str(p.relative_to(root)): digest(p) for p in paths + [reference]},
                  "scope": "retained midpoint input integrity and reproduced analysis; no cost or spatial qualification"}, indent=2, sort_keys=True))
