#!/usr/bin/env python3
"""Run a source-frozen Human invocation with the existing evidence layout.

This schedules the native executable; all mechanics remain in its Metal owner.
It deliberately starts with a clean simulation-selector environment, verifies
the frozen source/artifact closure, and never reuses inherited receipt fields.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--step-seconds", type=float)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--unset-env", action="append", default=[], metavar="KEY")
    parser.add_argument("--metal-validation", choices=("0", "1"), default="1")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.step_seconds is not None and (
        not math.isfinite(args.step_seconds) or args.step_seconds <= 0
    ):
        parser.error("--step-seconds must be finite and positive")
    template_path = args.template.resolve(strict=True)
    provenance_path = args.provenance.resolve(strict=True)
    cwd = args.cwd.resolve(strict=True)
    template = json.loads(template_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    frozen = Path(provenance["frozen_artifact_directory"]).resolve(strict=True)
    # Validate every file, including the archived source patch and manifest.
    for rel, expected in provenance["artifacts"].items():
        artifact = (frozen / rel).resolve(strict=True)
        if not artifact.is_relative_to(frozen):
            raise ValueError("artifact manifest escapes frozen directory")
        if digest(artifact) != expected:
            raise ValueError("frozen artifact hash mismatch: " + rel)
    argv = list(template["argv"])
    if len(argv) < 5 or Path(argv[0]).name != "numi-human-native":
        raise ValueError("template must be the native Human positional invocation")
    argv[0] = str(frozen / "numi-human-native")
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output already exists; preserve prior evidence")
    argv[4] = str(output / "output")
    for option, value in (
        ("--muscle-step-count", args.steps),
        ("--muscle-step-seconds", args.step_seconds),
    ):
        if value is not None:
            if argv.count(option) != 1:
                raise ValueError("template must contain exactly one " + option)
            argv[argv.index(option) + 1] = str(value)
    selectors = dict(template["simulation_environment"])
    for key in args.unset_env:
        selectors.pop(key, None)
    for setting in args.env:
        key, separator, value = setting.partition("=")
        if not separator or not key or not key.replace("_", "").isalnum():
            parser.error("--env requires a valid KEY=VALUE")
        selectors[key] = value
    selectors.update(
        DYLD_LIBRARY_PATH=str(frozen),
        NUMI_MATTER_METALLIB_OVERRIDE=str(frozen / "NumiMatter.metallib"),
        MTL_DEBUG_LAYER=args.metal_validation,
        NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(output / "common-field-failure.json"),
    )
    # Ambient experiment switches must not silently alter a frozen comparison.
    prefixes = ("NM_", "NUMI_", "MTL_", "DYLD_")
    execution_env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(prefixes)
    }
    execution_env.update(selectors)
    receipt = dict(
        argv=argv, simulation_environment=selectors, cwd=str(cwd),
        binary=argv[0], binary_sha256=provenance["artifacts"]["numi-human-native"],
        source_head=provenance["base_revision"],
        source_manifest=str(frozen / "source-manifest.json"),
        input_sha256={str(Path(value).resolve()): digest(Path(value))
                      for value in argv[1:]
                      if value.startswith("/") and Path(value).is_file()},
        source_provenance=str(provenance_path),
        source_provenance_sha256=digest(provenance_path),
        source_archive_sha256=provenance["source_archive_sha256"],
        source_patch_sha256=provenance["source_patch_sha256"],
        artifact_sha256={str(frozen / key): value
                         for key, value in provenance["artifacts"].items()},
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        orchestration=str(Path(__file__).resolve()),
        orchestration_sha256=digest(Path(__file__)),
        template_invocation=str(template_path),
        template_invocation_sha256=digest(template_path),
    )
    output.mkdir(parents=True)
    (output / "invocation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    start = time.monotonic()
    with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
        result = subprocess.run(argv, cwd=cwd, env=execution_env,
                                stdout=stdout, stderr=stderr, check=False)
    process = dict(name=output.name, exit_code=result.returncode,
                   wall_seconds=time.monotonic() - start)
    (output / "process.json").write_text(json.dumps(process, indent=2) + "\n")
    print(json.dumps(process), flush=True)
    return result.returncode

if __name__ == "__main__":
    raise SystemExit(main())
