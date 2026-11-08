#!/usr/bin/env python3
"""Opt-in native probe for Window setup failures and runtime-check completion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
for name in ("build", "scene", "world-pack", "task-pack", "robot-actuator-pack",
             "sensor-program-pack", "reality-program-pack", "output"):
    parser.add_argument("--" + name, type=Path, required=True)
parser.add_argument("--camera-parent", required=True)
parser.add_argument("--timeout", type=float, default=120)
a = parser.parse_args()
repo = Path(__file__).resolve().parent.parent
output = a.output.resolve()
output.mkdir(parents=True, exist_ok=False)
workspace = output / "project with spaces"
(workspace / ".numi").mkdir(parents=True)
scene_source = a.scene.resolve()
scene = json.loads(scene_source.read_text())
# Resolve the input's authored geometry without editing or copying it.
for pack in scene["packs"]:
    path = Path(pack["path"])
    if not path.is_absolute():
        pack["path"] = str((scene_source.parent / path).resolve())
paths = {name: getattr(a, name.replace("-", "_")).resolve() for name in (
    "world-pack", "task-pack", "robot-actuator-pack", "sensor-program-pack",
    "reality-program-pack")}
inputs = [scene_source, *(Path(pack["path"]) for pack in scene["packs"]), *paths.values(),
          repo / "apps/task_rollout.swift", repo / ".numi/commands/window", repo / "tools/numi",
          Path(__file__).resolve()]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
plan = {"inputs_sha256": {str(p): digest(p) for p in inputs},
        "build": str(a.build.resolve()), "camera_parent": a.camera_parent,
        "timeout_seconds": a.timeout, "scope": "synthetic native failure reporting, not physical or UI qualification"}
(output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
env = os.environ.copy()
for name in ("NUMI_WINDOW_SCENE", "NUMI_WINDOW_SCENE_CATALOG", "NUMI_WINDOW_POLICY",
             "NUMI_WINDOW_POLICY_CATALOG", "NUMI_WINDOW_RUNTIME_CHECK"):
    env.pop(name, None)
env.update(NUMI_LAB_ROOT=str(repo), NUMI_WINDOW_BUILD_DIR=str(a.build.resolve()))
results = {}
for name, parent in (("invalid", "__numi_missing_camera_parent__"), ("valid", a.camera_parent)):
    scene["camera"]["parent_body"] = parent
    scene_path = output / (name + ".json")
    scene_path.write_text(json.dumps(scene, sort_keys=True) + "\n")
    argv = [str(repo / "tools/numi"), "window", "--check", "--window-scene",
            str(scene_path), "--envs", "1", "--steps", "8", "--chunk", "1"]
    for key, path in paths.items():
        argv += ["--" + key, str(path)]
    argv += ["--zero-actions"]
    p = subprocess.Popen(argv, cwd=workspace, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = p.communicate(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(p.pid, signal.SIGTERM)
        try:
            stdout, stderr = p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            stdout, stderr = p.communicate()
    (output / (name + ".stdout")).write_bytes(stdout)
    (output / (name + ".stderr")).write_bytes(stderr)
    runtime_files = [a.build / "bin/metalrobo_task_rollout", a.build / "lib/libmetalrobo.dylib",
                     a.build / "shaders/MetalRobo.metallib"]
    results[name] = {"argv": argv, "exit": p.returncode, "timed_out": timed_out,
                     "runtime_sha256": {str(f.resolve()): digest(f) for f in runtime_files if f.exists()}}
    if timed_out:
        break
    if name == "invalid" and (p.returncode == 0 or b"head-camera parent body" not in stderr):
        break
    if name == "valid" and (p.returncode != 0 or b"numi window runtime check passed" not in stderr or
                            b"numi window check restored configuration" not in stderr):
        break
else:
    results["inputs_unchanged"] = all(digest(p) == plan["inputs_sha256"][str(p)] for p in inputs)
    results["runtime_unchanged"] = results["invalid"]["runtime_sha256"] == results["valid"]["runtime_sha256"]
    results["passed"] = results["inputs_unchanged"] and results["runtime_unchanged"]
(output / "result.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
print(json.dumps(results, indent=2, sort_keys=True))
raise SystemExit(0 if results.get("passed") else 1)
