#!/usr/bin/env python3
"""Calibrate the corrected collector against retained native output and a bad fixture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_trial_output", type=Path)
    parser.add_argument("material", type=Path)
    parser.add_argument("collector", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.source_trial_output.resolve()
    spec = importlib.util.spec_from_file_location("traction_collector", args.collector)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load collector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    material_sha = sha256(args.material)
    observed = module.measure(source / "native", 0.0, material_sha)
    if not observed.get("valid") or observed.get("sample_count") != 60:
        raise SystemExit("retained native control failed collector positive calibration")

    with tempfile.TemporaryDirectory(prefix="numi-skin-collector-negative-") as temporary:
        corrupted = Path(temporary) / "native"
        shutil.copytree(source / "native", corrupted)
        sample_path = corrupted / "traction-samples.csv"
        rows = list(csv.DictReader(sample_path.open(newline="", encoding="utf-8")))
        rows[17]["maximum_volume_residual"] = "0.00010001"
        with sample_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        shutil.copy2(source / "native-exit-code.txt", Path(temporary) / "native-exit-code.txt")
        shutil.copy2(source / "native-stdout.log", Path(temporary) / "native-stdout.log")
        try:
            module.measure(corrupted, 0.0, material_sha)
        except ValueError as exc:
            if "unchanged native volume gate" not in str(exc):
                raise SystemExit("corrupted residual did not fail for the expected gate: " + str(exc))
        else:
            raise SystemExit("collector accepted the corrupted native volume residual")

    payload = {
        "schema": "numi.science.calibration-check.v1",
        "passed": True,
        "checks": [
            "reparsed the retained 60-step native control and emitted its exact schema",
            "rejected a corrupted sample above the unchanged 1e-4 volume gate",
        ],
        "positive_control_only": "one previously recorded deterministic synthetic coupon; not a new replicate",
        "positive_metrics": {
            "sample_count": observed["sample_count"],
            "minimum_mean_gap_m": observed["minimum_mean_gap_m"],
            "maximum_volume_residual": observed["maximum_volume_residual"],
            "total_gpu_ms": observed["total_gpu_ms"],
        },
        "source_sha256": {
            str(source / "native/traction-samples.csv"): sha256(source / "native/traction-samples.csv"),
            str(source / "native/lip-geometry.csv"): sha256(source / "native/lip-geometry.csv"),
            str(source / "native-stdout.log"): sha256(source / "native-stdout.log"),
            str(source / "native-exit-code.txt"): sha256(source / "native-exit-code.txt"),
            str(source.parent / "process.json"): sha256(source.parent / "process.json"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
