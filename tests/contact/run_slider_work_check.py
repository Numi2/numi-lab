#!/usr/bin/env python3
"""Build and run the preregistered one-step FEM work instrument check."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from slider_work_validation import validate_fem_work_rows

root = Path(__file__).resolve().parents[2]
evidence = root / "docs/evidence/midpoint-first-failure-20261109"
build = Path("/Users/numi/numi-matter-slider-owner-build-20261109")
out = build / "evidence-work-check-003"
if out.exists():
    raise SystemExit("refusing to overwrite diagnostic attempt")
if shutil.disk_usage("/System/Volumes/Data").free < 3.12 * 1024**3:
    raise SystemExit("free-space limit before diagnostic build")
out.mkdir()
source = root / "apps/midpoint_slider_coupled_diagnostic.mm"
obj, exe = out / "probe.o", out / "probe"
lib = build / "lib/libmetalrobo.dylib"
shaders = [build / "matter/shaders/NumiMatter.metallib", build / "shaders/MetalRobo.metallib"]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

compile_command = ["/usr/bin/clang++", "-I" + str(root / "include"),
    "-I" + str(root / "matter/include"), "-x", "objective-c++", "-O3", "-DNDEBUG",
    "-std=c++23", "-arch", "arm64", "-fPIE", "-fobjc-arc", "-Wall", "-Wextra",
    "-Wpedantic", "-c", str(source), "-o", str(obj)]
link_command = ["/usr/bin/clang++", "-O3", "-DNDEBUG", "-arch", "arm64",
    "-Wl,-rpath," + str(build / "lib"), str(obj), "-o", str(exe), str(lib),
    "-framework", "Foundation", "-framework", "Metal"]
records = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
           "commands": [compile_command, link_command], "build": [], "runs": [],
           "environment": {"NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION": "1",
                           "NM_MATTER_MIDPOINT_STAGE_COMPENSATION": "1"},
           "scope": "one accepted step, diagnostic only; no performance or complete-rollout claim"}
for command in (compile_command, link_command):
    start = time.monotonic()
    try:
        r = subprocess.run(command, text=True, capture_output=True, timeout=120)
        records["build"].append({"exit_code": r.returncode, "stderr": r.stderr,
                                 "wall_seconds_observational_only": time.monotonic()-start})
        if r.returncode != 0:
            break
    except subprocess.TimeoutExpired:
        records["build"].append({"timeout": True})
        break
if len(records["build"]) == 2 and all(x.get("exit_code") == 0 for x in records["build"]):
    records["inputs_sha256"] = {str(p): digest(p) for p in [source, exe, lib, *shaders]}
    for name, gap, expect_contact in [("far", "0.0002", False), ("near", "0.000095", True)]:
        prefix = evidence / ("slider-fem-work-" + name + "-003")
        paths = [Path(str(prefix) + ext) for ext in (".json", ".log")]
        if any(p.exists() for p in paths):
            raise SystemExit("refusing to overwrite " + str(prefix))
        if shutil.disk_usage("/System/Volumes/Data").free < 3.12 * 1024**3:
            records["runs"].append({"name": name, "stopped_reason": "free-space limit"})
            break
        env = os.environ.copy()
        env["NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION"] = "1"
        env["NM_MATTER_MIDPOINT_STAGE_COMPENSATION"] = "1"
        cmd = [str(exe), *(str(p) for p in shaders), "0.000015625", gap, "1"]
        try:
            r = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=60)
            paths[0].write_text(r.stdout); paths[1].write_text(r.stderr)
            item = {"name": name, "command": cmd, "exit_code": r.returncode,
                    "stdout_sha256": digest(paths[0]), "stderr_sha256": digest(paths[1])}
            try:
                trace = json.loads(r.stdout)
                step = trace["steps"][0]
                item["independent_gate_ratio"] = validate_fem_work_rows(step, trace)
                item["gate_ratio"] = step["fem_work_reconstruction_gate_ratio"]
                item["fem_impulse_Ns"] = step["fem_impulse"]
                item["fem_work_J"] = step["fem_contact_work"]
                item["fem_law_work_J"] = step["fem_law_work"]
                item["passes"] = (r.returncode == 0 and step["accepted"] and
                    item["gate_ratio"] <= 1 and item["independent_gate_ratio"] <= 1 and
                    ((item["fem_impulse_Ns"] > 0) if expect_contact else
                     (item["fem_impulse_Ns"] == 0 and item["fem_work_J"] == 0)))
            except (ValueError, KeyError, IndexError) as error:
                item.update(passes=False, parse_error=repr(error))
            records["runs"].append(item)
        except subprocess.TimeoutExpired:
            records["runs"].append({"name": name, "command": cmd, "timeout": True, "passes": False})
records["passes"] = len(records["runs"]) == 2 and all(r.get("passes") for r in records["runs"])
path = evidence / "slider-fem-work-check-003-execution.json"
if path.exists():
    raise SystemExit("refusing to overwrite execution record")
path.write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records, indent=2))
raise SystemExit(0 if records["passes"] else 1)
