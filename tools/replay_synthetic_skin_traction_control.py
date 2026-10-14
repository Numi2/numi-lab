#!/usr/bin/env python3
"""Replay an integrity-checked deterministic zero-force baseline from a parent study."""

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed(path):
    envelope = json.loads(Path(path).read_text())
    if canonical_sha(envelope["payload"]) != envelope.get("sha256"):
        raise ValueError(f"sealed record checksum mismatch: {path}")
    return envelope


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: replay_synthetic_skin_traction_control.py OUTPUT_DIRECTORY PARENT_STUDY")
    output = Path(sys.argv[1]).resolve()
    parent_analysis_path = Path(sys.argv[2]).resolve()
    if parent_analysis_path.name != "analysis.json":
        raise ValueError("parent input must be the sealed analysis.json file")
    parent = parent_analysis_path.parent
    trial_id = "cosine24-mesh24x8x2-control"
    trial = parent / "trials" / trial_id
    attempt_path = parent / "attempts" / f"{trial_id}.json"
    receipt_path = trial / "receipt.json"
    attempt = sealed(attempt_path)
    receipt = sealed(receipt_path)
    analysis = sealed(parent_analysis_path)

    if attempt["payload"]["trial"].get("arm") != "control":
        raise ValueError("parent attempt is not a declared control")
    if receipt["payload"].get("returncode") != 0 or receipt["payload"].get("failure") is not None:
        raise ValueError("parent control execution did not succeed")
    records = analysis["payload"].get("trials", [])
    parent_control = next((row for row in records if row.get("id") == trial_id), None)
    if not parent_control or parent_control.get("arm") != "control" or parent_control.get("status") != "valid":
        raise ValueError("parent analysis does not mark the control valid")
    if parent_control.get("receipt_sha256") != receipt["sha256"]:
        raise ValueError("parent analysis receipt identity differs")

    source_output = trial / "output"
    expected_files = receipt["payload"].get("files", {})
    for relative, expected in expected_files.items():
        path = source_output / relative
        if not path.is_file() or file_sha(path) != expected:
            raise ValueError(f"parent control evidence changed: {relative}")
    summary_path = source_output / "stdout.json"
    summary = json.loads(summary_path.read_text())
    if (summary.get("valid") is not True or summary.get("force_per_site_n") != 0.0 or
            summary.get("sample_count") != 60 or summary.get("active_tetrahedra") != 2304):
        raise ValueError("parent control summary does not match the fixed zero-force baseline")

    trajectory = output / "trajectory"
    native = output / "native"
    trajectory.mkdir(parents=True, exist_ok=False)
    native.mkdir(parents=True, exist_ok=False)
    for name in ("traction-samples.csv", "lip-geometry.csv"):
        source = source_output / "trajectory" / name
        with source.open(newline="") as stream:
            if not list(csv.reader(stream)):
                raise ValueError(f"parent trajectory is empty: {name}")
        shutil.copyfile(source, trajectory / name)
    for name in ("stdout.log", "stderr.log"):
        shutil.copyfile(source_output / "native" / name, native / name)

    summary.update({
        "data_origin": "reused_parent_zero_force_baseline",
        "source_study": str(parent),
        "source_trial_id": trial_id,
        "source_attempt_sha256": attempt["sha256"],
        "source_receipt_sha256": receipt["sha256"],
        "source_analysis_sha256": analysis["sha256"],
        "source_stdout_json_sha256": file_sha(summary_path),
        "schedule_note": "The cosine60 ramp is registered only for the loaded treatment; zero-force data are schedule-invariant.",
    })
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
