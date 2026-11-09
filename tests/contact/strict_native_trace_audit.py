#!/usr/bin/env python3
"""Reject malformed numeric inputs before accepting a completed native comparison."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

from large_spatial_reference import read_reference


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def finite_tree(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return False


def check_rows(rows, h):
    for i, row in enumerate(rows):
        if (not finite_tree(row) or row.get("step") != i or row.get("accepted") is not True or
            row.get("status") != 0 or row.get("outer_status") != 0 or
            row.get("certificate_accepted") != 1 or row.get("gap", 0) <= 1e-7 or
            not math.isclose(row.get("mass", 0), .0088, rel_tol=1e-6) or
            row.get("force", -1) < 0 or row.get("accepted_normal_impulse", -1) < 0 or
            not math.isclose(row["accepted_normal_impulse"], row["force"] * h,
                             rel_tol=5e-11, abs_tol=1e-20) or
            not math.isclose(row.get("normal_impulse", -1), row["accepted_normal_impulse"],
                             rel_tol=5e-11, abs_tol=1e-20)):
            raise ValueError("invalid accepted native sample at step " + str(i))


def audit(trace, execution, reference):
    trace, execution, reference = map(Path, (trace, execution, reference))
    record = json.loads(execution.read_text())
    if (record.get("all_gates_pass") is not True or record.get("returncode") != 0 or
        record.get("stopped_reason") is not None or
        record.get("diagnostic_stdout", {}).get("compressed_sha256") != digest(trace)):
        raise ValueError("native execution is incomplete or trace identity differs")
    with gzip.open(trace, "rt") as source:
        data = json.load(source)
    if not finite_tree(data):
        raise ValueError("non-finite native numeric field")
    h = data["h"]
    steps = round(.01 / h)
    if (data.get("schema") != "numi.matter.plane-contact.v1" or data.get("arm") != "ipc_area" or
        data.get("divisions") != 2 or abs(h * steps - .01) > 1e-12 or
        data.get("accepted_steps") != steps or data.get("rejected_steps") != 0 or
        len(data["traces"]) != steps or not data.get("restore_replay") or
        not data.get("continuation_replay") or not data.get("forced_rollback_exact") or
        not data.get("identity_mismatch_rejected")):
        raise ValueError("native rollout is incomplete or not the declared control")
    check_rows(data["traces"], h)
    ref = read_reference(reference, 2, .00000025)
    return {"valid": True, "steps": steps, "h": h, "native_sha256": digest(trace),
            "execution_sha256": digest(execution), "reference_sha256": digest(reference),
            "reference_verified_area_m2": ref["verified_area_m2"],
            "scope": "input integrity only; physical and cost gates remain separate"}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: strict_native_trace_audit.py TRACE_GZ EXECUTION_JSON REFERENCE_JSON")
    try:
        print(json.dumps(audit(*sys.argv[1:]), indent=2, sort_keys=True))
    except (ValueError, KeyError, OSError, TypeError, ZeroDivisionError) as error:
        print(json.dumps({"valid": False, "error": repr(error)}))
        raise SystemExit(1)
