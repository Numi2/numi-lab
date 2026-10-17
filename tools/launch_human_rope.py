#!/usr/bin/env python3
"""Replay a retained native Human invocation into a fresh evidence directory.

Run on SSH Numi. This schedules the existing native executable through the
existing exclusive-job wrapper; it implements no simulation or presentation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("invocation", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclusive-wrapper", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    receipt = json.loads(args.invocation.read_text())
    argv = receipt["argv"]
    if len(argv) < 5 or not all(isinstance(arg, str) for arg in argv):
        parser.error("expected the native Human argv with output directory at index 4")
    original_output = Path(argv[4])
    output = args.output.resolve()
    if not original_output.is_absolute() or output.exists():
        parser.error("output must be a fresh directory; existing evidence is never overwritten")
    wrapper = args.exclusive_wrapper.resolve(strict=True)
    pins = {Path(path).resolve(strict=True): sha
            for path, sha in receipt["asset_sha256"].items()}
    binary = Path(argv[0]).resolve(strict=True)
    if pins.get(binary) != receipt["binary_sha256"]:
        parser.error("active binary must be explicitly pinned in asset_sha256")
    runtime_source = Path(receipt["rope_runtime_source"]).resolve(strict=True)
    pins[runtime_source] = receipt["rope_runtime_source_sha256"]
    for path, expected in pins.items():
        if digest(path) != expected:
            parser.error(f"hash mismatch: {path}")

    def remap(value: str) -> str:
        lexical_child = value == str(original_output) or value.startswith(str(original_output) + "/")
        if Path(value).is_absolute():
            resolved = Path(value).resolve()
            canonical_root = original_output.resolve()
            if resolved.is_relative_to(canonical_root):
                return str(output / resolved.relative_to(canonical_root))
            if lexical_child:
                parser.error(f"output path escapes its retained run directory: {value}")
        return value

    active = {**receipt, "argv": [remap(value) for value in argv],
              "environment": {key: remap(value) for key, value in receipt["environment"].items()}}
    # This guard catches an inherited failure-receipt path outside the run.
    failure_path = active["environment"].get("NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT")
    if failure_path and not Path(failure_path).resolve().is_relative_to(output):
        parser.error("failure receipt must belong to this run's fresh output directory")
    if args.verify_only:
        print(f"Verified {len(pins)} files; native horizon {receipt['requested_physical_horizon_s']} s")
        return 0
    output.mkdir(parents=True, exist_ok=False)
    active["source_invocation"] = str(args.invocation.resolve())
    active["source_invocation_sha256"] = digest(args.invocation)
    active["exclusive_wrapper"] = str(wrapper)
    active["exclusive_wrapper_sha256"] = digest(wrapper)
    active["launcher_sha256"] = digest(Path(__file__))
    (output / "invocation.json").write_text(json.dumps(active, indent=2, sort_keys=True) + "\n")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("NUMI_HUMAN_", "DYLD_"))}
    env.update(active["environment"])
    print(f"Launching native Human into {output}", flush=True)
    with (output / "launcher.log").open("x") as log:
        result = subprocess.run([sys.executable, str(wrapper), "--", *active["argv"]],
                                env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(f"Native exit status {result.returncode}; evidence retained in {output}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
