#!/usr/bin/env python3
"""Capture and bind the predeclared independent slider work histories."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
e = root / "docs/evidence/midpoint-first-failure-20261109"
source = root / "tests/contact/fem_slider_fp64_reference.cpp"
model = root / "tests/contact/fem_plane_fp64_reference.cpp"
plan = e / "SLIDER_WORK_HISTORY_REFERENCE_PLAN.md"
binary = Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-slider-work-history-near-002")
primary_source = e / "slider-work-history-primary-source-002.cpp"
primary_binary = Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-slider-work-history-primary-002")
receipt = e / "slider-work-history-reference-002-execution.json"
if binary.exists() or primary_binary.exists() or receipt.exists():
    raise SystemExit("existing work history build or receipt")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

record = {"source_sha256": digest(source), "included_model_sha256": digest(model),
          "plan_sha256": digest(plan), "historical_primary_source_sha256": "db502b2a3dd4347da624ba6eeb0966195b0f94fb73a75732baf7a6bc2545bfb3",
          "instrumented_primary_source_sha256": digest(primary_source), "runner_sha256": digest(Path(__file__).resolve()),
          "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
          "runs": []}
def finish(reason, code=1):
    record["result"] = reason
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    raise SystemExit(code)
if shutil.disk_usage(e).free < 3.12 * 1024**3:
    finish("free-space guard before build")
compile_command = ["xcrun", "clang++", "-O3", "-DNDEBUG", "-std=gnu++23", "-arch", "arm64",
                   "-DNUMI_SLIDER_WORK_HISTORY=1", str(source), "-o", str(binary)]
compiled = subprocess.run(compile_command, cwd=root, capture_output=True, text=True)
record.update(compile_command=compile_command, compiler_returncode=compiled.returncode,
              compiler_stderr=compiled.stderr,
              compiler_version=subprocess.check_output(["xcrun", "clang++", "--version"], text=True).splitlines()[0])
if compiled.returncode:
    finish("instrumented reference build failed")
record["binary_sha256"] = digest(binary)
primary_compile = ["xcrun", "clang++", "-O3", "-DNDEBUG", "-std=gnu++23", "-arch", "arm64",
                   "-I", str(source.parent), str(primary_source), "-o", str(primary_binary)]
compiled_primary = subprocess.run(primary_compile, cwd=root, capture_output=True, text=True)
record.update(primary_compile_command=primary_compile, primary_compiler_returncode=compiled_primary.returncode,
              primary_compiler_stderr=compiled_primary.stderr)
if compiled_primary.returncode:
    finish("historic primary instrument build failed")
record["primary_binary_sha256"] = digest(primary_binary)
for fixture, gap, legacy in (("primary", None, "slider-v2-rk4-h"), ("near", "0.000095", "slider-near-rk4-h")):
    for code, timestep in (("050", "0.0000005"), ("025", "0.00000025")):
        path = e / ("slider-work-history-002-" + fixture + "-h" + code + ".json")
        if path.exists() or shutil.disk_usage(e).free < 3.12 * 1024**3:
            finish("output exists or free-space guard")
        command = [str(primary_binary if fixture == "primary" else binary), timestep] + ([] if gap is None else [gap])
        started = time.monotonic()
        with path.open("x") as out:
            run = subprocess.run(command, cwd=root, stdout=out, stderr=subprocess.PIPE, text=True, timeout=120)
        entry = {"command": command, "returncode": run.returncode, "stderr": run.stderr,
                 "wall_seconds_observational_only": time.monotonic()-started,
                 "output_sha256": digest(path), "historical_sha256": digest(e / (legacy + code + ".json"))}
        record["runs"].append(entry)
        if run.returncode:
            finish("reference run failed")
        actual = json.loads(path.read_text())
        history = actual.pop("work_trace", None)
        original = json.loads((e / (legacy + code + ".json")).read_text())
        entry["historic_fields_exact_match"] = actual == original
        entry["work_trace_rows"] = len(history) if isinstance(history, list) else 0
        if actual != original or entry["work_trace_rows"] != 5001:
            finish("historic parity or work history count failed")
if digest(source) != record["source_sha256"] or digest(model) != record["included_model_sha256"] or digest(binary) != record["binary_sha256"] or digest(primary_binary) != record["primary_binary_sha256"] or digest(primary_source) != record["instrumented_primary_source_sha256"]:
    finish("input changed during reference")
finish("four references completed with exact historical-field parity", 0)
