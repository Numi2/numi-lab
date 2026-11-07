#!/usr/bin/env python3
"""Opt-in GPU probe: real window training mailbox and cancellation lifecycle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

parser = argparse.ArgumentParser()
for option in ("workspace", "build", "scene-catalog", "python", "output"):
    parser.add_argument("--" + option, type=Path, required=True)
args = parser.parse_args()
root, build, catalog, python, output = (
    args.workspace.resolve(), args.build.resolve(), args.scene_catalog.resolve(),
    args.python.resolve(), args.output.resolve()
)
scene = catalog / "diagnostic.json"
source_files = [
    root / "apps/task_rollout.swift", root / "apps/NumiWindow.swift",
    root / "apps/NumiTraining.swift", root / "numi/commands/train",
    root / ".numi/commands/window", root / "tools/numi",
    root / "python/metalrobo/native.py", root / "python/metalrobo/mlx_policy_learning.py",
    root / "python/metalrobo/mlx_policy_worker.py",
    build / "bin/metalrobo_task_rollout", build / "bin/metalrobo_task_train",
    build / "lib/libmetalrobo.dylib", build / "shaders/MetalRobo.metallib",
    scene, catalog / "diagnostic.mrvpack", catalog / "synthetic.numi-window.json",
]

def fingerprints():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}

def processes():
    text = subprocess.check_output(["ps", "-axo", "pid=,ppid=,command="], text=True)
    result = {}
    for line in text.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            result[int(parts[0])] = (int(parts[1]), parts[2])
    return result

revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).strip()
if status:
    raise RuntimeError("probe requires a clean tracked source tree: " + status)
before = fingerprints()
output.mkdir(parents=True, exist_ok=False)
trigger = output / "close.trigger"
log = output / "window.log"
environment = os.environ.copy()
environment.update({
    "NUMI_LAB_ROOT": str(root),
    "NUMI_WORKSPACE_ROOT": str(root),
    "NUMI_WINDOW_BUILD_DIR": str(build),
    "NUMI_MLX_PYTHON": str(python),
    "NUMI_WINDOW_SCENE_CATALOG": str(catalog),
    "NUMI_WINDOW_POLICY_CATALOG": str(output / "empty-policies"),
    "NUMI_WINDOW_RUNTIME_TRAIN_CANCEL_CHECK": "1",
    "NUMI_WINDOW_RUNTIME_CANCEL_TRIGGER": str(trigger),
})
command = [
    "sh", str(root / ".numi/commands/window"), "--check",
    "--window-scene", str(scene), "--envs", "1", "--steps", "128",
    "--chunk", "1", "--robot-source", "unitree-g1",
    "--scene", "ground", "--task", "velocity", "--zero-actions",
]
record = {
    "revision": revision, "source_status_before": status,
    "sha256_before": before, "command": command,
    "boundary": "synthetic presentation; test-injected mailbox request; no saved-state or trained-control claim",
    "observed_descendants": {}, "triggered": False,
}
training_pid = None
run_directory = None
seen = {}
with log.open("w") as stream:
    child = subprocess.Popen(command, cwd=root, env=environment, stdout=stream,
                             stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 75
    try:
        while time.monotonic() < deadline and child.poll() is None:
            time.sleep(0.1)
            text = log.read_text(errors="replace")
            match = re.search(r"numi window train check pid=(\d+) run=(.+)", text)
            if not match:
                continue
            training_pid = int(match.group(1))
            run_directory = Path(match.group(2).strip())
            table = processes()
            descendants = {training_pid}
            changed = True
            while changed:
                changed = False
                for pid, (parent, command_line) in table.items():
                    if parent in descendants and pid not in descendants:
                        descendants.add(pid)
                        changed = True
            seen.update({pid: table[pid][1] for pid in descendants if pid in table})
            names = list(seen.values())
            trainer = any("metalrobo_task_train" in value for value in names)
            worker = any("metalrobo.mlx_policy_worker" in value for value in names)
            if trainer and worker and (run_directory / ".train.lock").is_dir():
                record["observed_descendants"] = {str(pid): value for pid, value in seen.items()}
                trigger.write_text("request AppKit window close\n")
                record["triggered"] = True
                break
        if child.poll() is None:
            remaining = max(1.0, deadline - time.monotonic())
            child.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        record["timeout"] = True
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    except Exception as error:
        record["monitor_error"] = repr(error)
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

record["process_exit"] = child.returncode
record["training_pid"] = training_pid
record["run_directory"] = str(run_directory) if run_directory else None
for _ in range(50):
    table = processes()
    alive = {pid: original for pid, original in seen.items()
             if pid in table and table[pid][1] == original}
    if not alive:
        break
    time.sleep(0.1)
record["surviving_observed_descendants"] = {str(pid): value for pid, value in alive.items()}
record["lock_exists_after"] = bool(run_directory and (run_directory / ".train.lock").exists())
record["sha256_after"] = fingerprints()
record["identities_unchanged"] = before == record["sha256_after"]
text = log.read_text(errors="replace")
record["success"] = (
    record["triggered"] and child.returncode == 0 and not alive and
    not record["lock_exists_after"] and record["identities_unchanged"] and
    "numi window train check closing" in text and
    "numi window train check exit=143 lock=false" in text
)
if run_directory and run_directory.is_dir():
    shutil.copytree(run_directory, output / "run")
(output / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: record[k] for k in (
    "success", "process_exit", "triggered", "surviving_observed_descendants",
    "lock_exists_after", "identities_unchanged", "run_directory")}, indent=2))
raise SystemExit(0 if record["success"] else 1)
