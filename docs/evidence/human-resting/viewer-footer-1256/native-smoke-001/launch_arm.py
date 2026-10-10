#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import shutil
import fcntl
import subprocess
import time

BASE = Path("/Users/n/numi-human-retained-delivery-20261009/viewer-footer-1256/native-smoke-001")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one prepared Q-window smoke arm; refuses every existing output.")
    parser.add_argument("arm", choices=("control", "window"))
    args = parser.parse_args()
    arm_dir = BASE / args.arm
    declaration_path = arm_dir / "run-declaration.json"
    guard_path = BASE / "launch-guard.json"
    if not declaration_path.is_file() or not guard_path.is_file():
        raise SystemExit("missing prepared declaration or launch guard")
    declaration = json.loads(declaration_path.read_text())
    guard = json.loads(guard_path.read_text())
    expected = guard["arms"][args.arm]["run_declaration_sha256"]
    if sha(declaration_path) != expected:
        raise SystemExit("declaration changed after preparation")
    launcher = Path(__file__).resolve()
    if sha(launcher) != guard["launcher_sha256"]:
        raise SystemExit("launcher changed after preparation")
    if declaration.get("launch_adapter", {}).get("sha256") != guard["launcher_sha256"]:
        raise SystemExit("declaration does not bind this launcher")
    native_output = Path(guard["arms"][args.arm]["native_output"])
    if native_output.exists():
        raise SystemExit("native output already exists; retain evidence and prepare a fresh arm")
    for path, digest in declaration["immutable_assets"].items():
        file_path = Path(path)
        if not file_path.is_file() or sha(file_path) != digest:
            raise SystemExit(f"immutable input changed: {path}")
    processes = subprocess.check_output(["ps", "-axo", "pid,comm"], text=True).splitlines()
    if any(row.split()[-1].endswith("/numi-human-native") for row in processes[1:]):
        raise SystemExit("another native Human owner is active")
    lock = open("/tmp/numi-human-native-root-owner.lock", "a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if shutil.disk_usage(BASE).free < 1000 * 1024 * 1024:
        raise SystemExit("insufficient storage for bounded native capture")
    start_path = arm_dir / "execution-start.json"
    execution_path = arm_dir / "execution.json"
    stdout_path = arm_dir / "launcher-stdout.log"
    if any(p.exists() for p in (start_path, execution_path, stdout_path)):
        raise SystemExit("launcher evidence path already exists; retain earlier attempt")
    start = {
        "schema": "numi.human.native-smoke.execution-start.v1",
        "declaration_sha256": sha(declaration_path),
        "launcher_sha256": sha(launcher),
        "utc_started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "input_pin_count": len(declaration["immutable_assets"]),
        "argv": declaration["argv"],
        "expected_native_argv": declaration["owner_cli_preview"]["native_argv"],
    }
    write_new(start_path, start)
    started = time.monotonic()
    with stdout_path.open("x") as output:
        child = subprocess.run(declaration["argv"], cwd=arm_dir,
                               stdout=output, stderr=subprocess.STDOUT,
                               check=False)
    changed = {}
    for path, digest in declaration["immutable_assets"].items():
        file_path = Path(path)
        observed = sha(file_path) if file_path.is_file() else None
        if observed != digest:
            changed[path] = {"expected": digest, "observed": observed}
    invocation_path = native_output / "invocation.json"
    observed_argv = None
    argv_match = False
    if invocation_path.is_file():
        invocation = json.loads(invocation_path.read_text())
        observed_argv = invocation.get("argv")
        argv_match = observed_argv == declaration["owner_cli_preview"]["native_argv"]
    record = {
        "schema": "numi.human.native-smoke.execution.v1",
        "declaration_sha256": sha(declaration_path),
        "launcher_sha256": sha(launcher),
        "returncode": child.returncode,
        "wall_seconds": time.monotonic() - started,
        "utc_finished": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "changed_inputs": changed,
        "native_invocation_present": invocation_path.is_file(),
        "native_argv_matches_prepared_cli_preview": argv_match,
        "observed_native_argv": observed_argv,
        "qualification": "short instrumentation smoke only; no long-horizon or overall acceptance claim",
    }
    write_new(execution_path, record)
    print(json.dumps({"arm": args.arm, **record}, indent=2), flush=True)
    return child.returncode if child.returncode else (1 if changed or not argv_match else 0)


if __name__ == "__main__":
    raise SystemExit(main())
