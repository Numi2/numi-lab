#!/usr/bin/env python3
"""Convert one verified Pascoletti et al. ramp file into explicit SI measurements."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

HEADERS = [
    "Total Time (s)", "Cycle Elapsed Time (s)",
    "Position(Linear:Position) (mm)", "Force(Linear:Load) (N)",
    "Displacement(Linear:Digital Position) (mm)",
]

def convert(source, expected, output):
    raw = source.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected.lower():
        raise ValueError(f"SHA-256 mismatch: {observed}")
    with io.StringIO(raw.decode("utf-16"), newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        if next(reader, None) != HEADERS:
            raise ValueError("unexpected instrument column names")
        rows = []
        for line, fields in enumerate(reader, start=2):
            if not fields:
                continue
            if len(fields) != len(HEADERS):
                raise ValueError(f"line {line}: incorrect column count")
            values = [float(x) for x in fields]
            if not all(math.isfinite(x) for x in values):
                raise ValueError(f"line {line}: non-finite measurement")
            if rows and values[0] <= rows[-1][0]:
                raise ValueError(f"line {line}: non-increasing total time")
            rows.append(values)
    if len(rows) < 2:
        raise ValueError("at least two samples required")
    # Preserve the instrument channels and signs; only convert millimetres to metres.
    with output.open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["total_time_s", "cycle_elapsed_time_s",
                         "position_m", "force_N", "displacement_m"])
        for t, cycle, pos, force, disp in rows:
            writer.writerow([t, cycle, pos / 1000.0, force, disp / 1000.0])
    return {"source": source.name, "source_sha256": observed,
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "rows": len(rows), "first_time_s": rows[0][0],
            "last_time_s": rows[-1][0],
            "dataset_doi": "10.17632/zpf2jrtgsx.1",
            "scope": "equine sutured organ, post-cyclic tensile ramp"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.sha256, args.output), indent=2))

if __name__ == "__main__":
    main()
