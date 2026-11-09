#!/usr/bin/env python3
"""Resume independent studies after the recorded storage stop, without overlap."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from cost_provenance import verify_accuracy_provenance
from cost_selection import select_largest_certified

root = Path(__file__).resolve().parents[2]
e = root / "docs/evidence/midpoint-first-failure-20261109"
b = Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-be-precision-005")
receipt = e / "recovery-sequence-003.json"
files = [Path(__file__).resolve(), root / "tests/contact/run_be_compressed_ladder_008.py",
         root / "tests/contact/run_accuracy_adjusted_cost_002.py",
         root / "tests/contact/run_large_spatial_continuation_005.py",
         root / "tests/contact/large_spatial_reference.py", root / "tests/contact/cost_provenance.py",
         root / "tests/contact/cost_selection.py", root / "tests/contact/cost_summary.py",
         root / "tests/contact/native_trace_analysis.py", e / "be-precision-005-inputs.sha256",
         root / "tests/contact/strict_native_trace_audit.py", root / "tests/contact/audit_cost_physical_time.py",
         root / "docs/evidence/midpoint-spatial-cost-dynamic-20261108/rk4-n2-h025.json"]
if len(sys.argv) != 2 or not sys.argv[1].isdigit() or receipt.exists():
    raise SystemExit("usage: run_recovery_sequence_003.py N64_SUPERVISOR_PID (fresh receipt required)")
pid = int(sys.argv[1])
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
hashes = {str(p): digest(p) for p in files}
record = {"schema": "numi.matter.recovery-sequence.v1", "n64_supervisor_pid": pid,
          "input_sha256": hashes, "phases": [], "status": "waiting_for_n64"}
def save():
    tmp = receipt.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(receipt)
def unchanged():
    return all(digest(Path(p)) == value for p, value in hashes.items())
def invoke(name, args):
    if not unchanged():
        raise RuntimeError("frozen recovery study files changed")
    log = e / ("recovery-sequence-003-" + name + ".log")
    if log.exists():
        raise RuntimeError("refusing to overwrite " + str(log))
    record["status"] = "running_" + name
    save()
    with log.open("x") as output:
        result = subprocess.run([sys.executable, *map(str, args)], cwd=root,
                                stdout=output, stderr=subprocess.STDOUT)
    record["phases"].append({"name": name, "returncode": result.returncode,
                             "log": log.name, "log_sha256": digest(log)})
    save()
    return result.returncode
save()
try:
    deadline = time.monotonic() + 48 * 3600
    while True:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                                capture_output=True, text=True)
        observed = result.stdout.strip()
        if not observed:
            break
        if "run_spatial_n64_rebound_005.py" not in observed:
            raise RuntimeError("n64 supervisor process identity changed")
        if time.monotonic() > deadline:
            raise RuntimeError("n64 wait limit")
        time.sleep(15)
    if not unchanged():
        raise RuntimeError("frozen recovery study files changed while waiting")
    n64_receipt = e / "n64-rebound-005-supervisor.json"
    n64 = json.loads(n64_receipt.read_text())
    record["n64_receipt_sha256"] = digest(n64_receipt)
    record["n64_reason"] = n64.get("reason")
    invoke("control", [root / "tests/contact/run_be_compressed_ladder_008.py"])
    try:
        verify = lambda item: verify_accuracy_provenance(item, e / "be-precision-005-inputs.sha256", b)
        midpoint, ml = select_largest_certified(e, "stage-ladder-n2-h*", "ipc_area_midpoint", verify)
        control, bl = select_largest_certified(e, "be-precision-008-n2-h*", "ipc_area", verify)
        record["selected"] = {"midpoint": midpoint, "control": control,
                              "midpoint_ladder": ml, "control_ladder": bl}
        cost_status = invoke("cost", [root / "tests/contact/run_accuracy_adjusted_cost_002.py",
                                      midpoint, control, "cost-fair-n2-003"])
        # Complete these independent checks before starting the potentially
        # multi-day spatial continuation. No diagnostic snapshots are added to
        # the timed cost loop; the strict audits consume already saved traces.
        if cost_status == 0:
            audits = []
            for execution in sorted(e.glob("be-precision-008-n2-h*-execution.json")):
                meta = json.loads(execution.read_text())
                if meta.get("all_gates_pass") is not True:
                    continue
                prefix = str(execution)[:-len("-execution.json")]
                name = execution.name.removesuffix("-execution.json")
                status = invoke("strict-" + name, [root / "tests/contact/strict_native_trace_audit.py",
                    Path(prefix + ".json.gz"), execution,
                    root / "docs/evidence/midpoint-spatial-cost-dynamic-20261108/rk4-n2-h025.json"])
                audits.append({"execution": execution.name, "returncode": status})
            time_status = invoke("physical-time", [root / "tests/contact/audit_cost_physical_time.py",
                                                     e, "cost-fair-n2-003"])
            record["cost_independent_checks"] = {"strict_audits": audits, "physical_time_returncode": time_status,
                "all_pass": bool(audits) and all(a["returncode"] == 0 for a in audits) and time_status == 0}
            save()
    except (ValueError, OSError, KeyError) as error:
        record["cost_not_run"] = repr(error)
        save()
    if n64.get("reason") == "both hash-bound references completed and analyzed":
        invoke("spatial", [root / "tests/contact/run_large_spatial_continuation_005.py"])
    else:
        record["spatial_not_run"] = "n64 prerequisite did not complete"
    record["status"] = "finished_inspect_individual_results"
except Exception as error:
    record["status"] = "stopped"
    record["error"] = repr(error)
save()
raise SystemExit(0 if record["status"] == "finished_inspect_individual_results" else 1)
