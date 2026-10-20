#!/usr/bin/env python3
"""Read-only decoder for NM_VASCULAR_TRACE_ROOT vascular_iterate records."""
import argparse
import json
import struct
from pathlib import Path

PREFIX = "vascular_iterate="

def read_records(path, root, tick):
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            at = line.find(PREFIX)
            if at < 0:
                continue
            try:
                record = json.loads(line[at + len(PREFIX):])
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: malformed vascular_iterate JSON: {error}") from error
            if record.get("root") == root and record.get("microtick") == tick:
                records.append(record)
    return records

def floats(data, count):
    if len(data) < 4 * count:
        raise ValueError(f"short float arena: need {4 * count} bytes, have {len(data)}")
    return struct.unpack_from("<" + "f" * count, data)

def summarize(record):
    arenas = record["arenas"]
    candidate = floats(bytes.fromhex(arenas["candidate"]), 4)
    residual = floats(bytes.fromhex(arenas["residual"]), 4)
    solution = floats(bytes.fromhex(arenas["solution"]), 4)
    alpha = floats(bytes.fromhex(arenas["alpha"]), 1)[0]
    changed_data = bytes.fromhex(arenas["working_set_changed"])
    changed = struct.unpack_from("<I", changed_data)[0] if len(changed_data) >= 4 else None
    fgmres_data = bytes.fromhex(arenas["fgmres"])
    fgmres = floats(fgmres_data, 8) if len(fgmres_data) >= 32 else ()
    status_data = bytes.fromhex(arenas["status"])
    status = struct.unpack_from("<8I4f", status_data) if len(status_data) >= 48 else ()
    return {
        "root": record["root"], "microtick": record["microtick"],
        "iteration": record["iteration"], "stage": record["stage"],
        "row0_candidate": candidate[0], "row0_residual_abs": abs(residual[0]),
        "row0_solution": solution[0], "shared_alpha": alpha,
        "working_set_changed": changed,
        "fgmres_diagnostics": list(fgmres[:4]) if fgmres else [],
        "fgmres_nonlinear": list(fgmres[4:8]) if fgmres else [],
        "status_code": status[0] if status else None,
        "status_failing_index": status[3] if status else None,
        "status_completed_microsteps": status[4] if status else None,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stderr", type=Path, help="native stderr log containing vascular_iterate records")
    parser.add_argument("--root", type=int, default=12)
    parser.add_argument("--microtick", type=int, default=1)
    parser.add_argument("--control", type=Path,
        default=Path(__file__).with_name("control-root12-trajectory.json"))
    args = parser.parse_args()
    records = read_records(args.stderr, args.root, args.microtick)
    if not records:
        raise SystemExit(f"no vascular_iterate records for root={args.root}, microtick={args.microtick}")
    summaries = [summarize(record) for record in records]
    try:
        control = json.loads(args.control.read_text(encoding="utf-8"))
        control_rows = {(row["root"], row["microtick"], row["iteration"], row["stage"]): row
                        for row in control.get("rows", [])}
    except (OSError, json.JSONDecodeError):
        control_rows = {}
    for row in summaries:
        key = (row["root"], row["microtick"], row["iteration"], row["stage"])
        ref = control_rows.get(key)
        if ref:
            row["control_candidate"] = ref.get("row0_candidate")
            row["control_residual_abs"] = abs(ref.get("row0_residual", 0.0))
            row["control_shared_alpha"] = ref.get("shared_alpha")
            row["candidate_delta_vs_control"] = (
                row["row0_candidate"] - ref["row0_candidate"]
                if isinstance(ref.get("row0_candidate"), (int, float)) else None)
            row["residual_ratio_vs_control"] = (
                row["row0_residual_abs"] / ref["row0_residual"]
                if isinstance(ref.get("row0_residual"), (int, float)) and ref["row0_residual"] != 0 else None)
    begin = next((r for r in summaries if r["stage"] == "predictor_begin"), None)
    end = next((r for r in reversed(summaries) if r["stage"] == "predictor_end"), None)
    if begin and end:
        print(json.dumps({
            "predictor_root": args.root, "microtick": args.microtick,
            "candidate_delta_row0": end["row0_candidate"] - begin["row0_candidate"],
            "residual_abs_begin": begin["row0_residual_abs"],
            "residual_abs_end": end["row0_residual_abs"],
            "residual_ratio_end_over_begin": (
                end["row0_residual_abs"] / begin["row0_residual_abs"]
                if begin["row0_residual_abs"] else None),
        }, separators=(",", ":")))
    for row in summaries:
        print(json.dumps(row, separators=(",", ":")))

if __name__ == "__main__":
    main()
