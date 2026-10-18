#!/usr/bin/env python3
"""Replay the exact bounded 009 native run into a fresh output directory on numi."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
INVOCATION = PACKAGE / "009" / "invocation.json"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def require_hash(path: Path, expected: str, what: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing {what}: {path}")
    actual = sha(path)
    if actual != expected:
        raise SystemExit(f"{what} SHA-256 mismatch: {path}: expected {expected}, got {actual}")

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="new, non-existing attempt directory")
    p.add_argument("--binary", help="optional path; must have the pinned 009 SHA-256")
    p.add_argument("--runtime-library", help="optional path; must have the pinned 226 runtime SHA-256")
    args = p.parse_args()
    original = json.loads(INVOCATION.read_text())
    run_root = Path(args.output).expanduser().resolve()
    if run_root.exists():
        raise SystemExit(f"refusing to reuse existing run directory: {run_root}")
    binary = Path(args.binary or original["argv"][0]).expanduser().resolve()
    runtime = Path(args.runtime_library or original["runtime_library"]).expanduser().resolve()
    require_hash(binary, original["binary_sha256"], "native binary")
    require_hash(runtime, original["runtime_library_sha256"], "runtime library")
    # Verify the exact authored inputs and source-loaded Metal modules from the accepted run.
    for asset, expected in original.get("asset_sha256", {}).items():
        require_hash(Path(asset), expected, "input asset")
    env = dict(original["environment"])
    source_pairs = [
        ("NUMI_HUMAN_STAND_SOURCE", "NUMI_HUMAN_STAND_SOURCE_SHA256"),
        ("NUMI_MATTER_TEST_SOURCE", "NUMI_MATTER_TEST_SOURCE_SHA256"),
        ("NUMANX_HUMAN_MATTER_TEST_SOURCE", "NUMANX_HUMAN_MATTER_TEST_SOURCE_SHA256"),
        ("NUMANX_COUPLED_HUMAN_TEST_SOURCE", "NUMANX_COUPLED_HUMAN_TEST_SOURCE_SHA256"),
        ("METALROBO_TEST_ARTICULATED_SOURCE", "METALROBO_TEST_ARTICULATED_SOURCE_SHA256"),
        ("NUMI_MATTER_SOURCE", "NUMI_MATTER_SOURCE_SHA256"),
    ]
    for path_key, hash_key in source_pairs:
        if path_key in env and hash_key in env:
            require_hash(Path(env[path_key]), env[hash_key], path_key)
    argv = list(original["argv"])
    if len(argv) < 6 or argv[5] != "--persistent-metal-stand":
        raise SystemExit("unexpected pinned command shape; refusing to rewrite arguments")
    old_output = Path(argv[4])
    run_root.mkdir(parents=True)
    output = run_root / "output"
    output.mkdir()
    argv[0] = str(binary)
    argv[4] = str(output)
    lib_dir = run_root / "lib"
    lib_dir.mkdir()
    (lib_dir / "libmetalrobo.dylib").symlink_to(runtime)
    env["DYLD_LIBRARY_PATH"] = str(lib_dir)
    env["DYLD_PRINT_LIBRARIES"] = "1"
    env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"] = str(run_root / "common-field-failure.json")
    # This replay is deliberately headless: reject movie, screenshot, and window-capture flags.
    forbidden = ("--resting-movie", "--screenshot", "--capture-view", "--window-capture")
    if any(flag in argv for flag in forbidden):
        raise SystemExit("pinned command unexpectedly contains a visual-recording option")
    launch = list(original["launcher_argv"][:3]) + argv
    record = dict(original)
    record.update({
        "scope": "two accepted intact body/FEM/fascia/CV steps, not whole-body continuum qualification",
        "argv": argv,
        "launcher_argv": launch,
        "environment": env,
        "binary_path": str(binary),
        "runtime_library": str(runtime),
        "runtime_library_sha256": sha(runtime),
        "fresh_output_root": str(run_root),
        "output_path": str(output),
        "visual_verification": False,
        "status": "running",
    })
    (run_root / "invocation.json").write_text(json.dumps(record, indent=2) + "\n")
    start = time.monotonic()
    with (run_root / "native.log").open("w") as log:
        completed = subprocess.run(launch, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    record["wall_seconds"] = time.monotonic() - start
    record["process_exit"] = completed.returncode
    record["status"] = "completed" if completed.returncode == 0 else "failed"
    (run_root / "invocation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "exit": completed.returncode,
                      "wall_seconds": record["wall_seconds"], "run": str(run_root)}, indent=2))
    return completed.returncode

if __name__ == "__main__":
    raise SystemExit(main())
