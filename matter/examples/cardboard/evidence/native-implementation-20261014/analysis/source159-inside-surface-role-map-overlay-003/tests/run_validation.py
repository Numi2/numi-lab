#!/usr/bin/env python3
"""Reproduce source159's bounded CPU fixtures and probe syntax check."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from pathlib import Path


BASE = Path("/Users/home/cardboard-implementation-20261014")
PACKAGE = BASE / "analysis/source159-inside-surface-role-map-overlay-003"
SOURCE157 = BASE / "snapshots/source-right-grip-telemetry-157"
COMPILE_DB = BASE / "analysis/source157-telemetry-native-build-001/build-metadata/compile_commands.json"
COMPILE_DB_SHA256 = "2789b25029e0e81b6ac938f0aed53763de29b733dd110f5c6013bfb2990ed53a"
TEST = PACKAGE / "tests/inside_clearance_check.cpp"
MEMORY_TEST = PACKAGE / "tests/shipping_clearance_memory_check.cpp"
PROBE_SOURCE = PACKAGE / "candidate/matter/tools/cardboard_probe.mm"


def run(args: list[str], cwd: Path | None = None) -> None:
    print("$ " + shlex.join(args))
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    if hashlib.sha256(COMPILE_DB.read_bytes()).hexdigest() != COMPILE_DB_SHA256:
        raise SystemExit("source157 compile database hash mismatch")

    base_flags = [
        "clang++", "-std=c++20", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
        "-I" + str(SOURCE157 / "matter/tools"),
        "-I" + str(SOURCE157 / "matter/include"),
        "-I" + str(SOURCE157 / "include"),
        "-I" + str(PACKAGE / "candidate/matter/tools"),
        str(TEST),
    ]
    for optimization in ("-O0", "-O3"):
        binary = Path("/tmp") / f"source159-inside-clearance-{optimization[2:]}"
        run(base_flags[:1] + [optimization] + base_flags[1:] + ["-o", str(binary)])
        run([str(binary)])
        memory_binary = Path("/tmp") / f"source159-clearance-memory-{optimization[2:]}"
        memory_flags = [
            "clang++", optimization, "-std=c++20", "-Wall", "-Wextra",
            "-Wpedantic", "-Werror",
            "-I" + str(SOURCE157 / "matter/tools"),
            "-I" + str(SOURCE157 / "matter/include"),
            "-I" + str(SOURCE157 / "include"),
            "-I" + str(PACKAGE / "candidate/matter/tools"),
            str(MEMORY_TEST), "-o", str(memory_binary),
        ]
        run(memory_flags)
        run([str(memory_binary)])

    snapshot_probe = SOURCE157 / "matter/tools/cardboard_probe.mm"
    entries = json.loads(COMPILE_DB.read_text(encoding="utf-8"))
    entry = next(item for item in entries if item["file"] == str(snapshot_probe))
    args = shlex.split(entry["command"])
    syntax_args: list[str] = []
    index = 0
    while index < len(args):
        if args[index] == "-o":
            index += 2
            continue
        if args[index] == "-c":
            index += 1
            continue
        syntax_args.append(str(PROBE_SOURCE) if args[index] == str(snapshot_probe) else args[index])
        index += 1
    syntax_args.extend(["-I" + str(SOURCE157 / "matter/tools"), "-fsyntax-only"])
    run(syntax_args, cwd=Path(entry["directory"]))
    print("source159 validation: PASS (inside O0/O3, memory O0/O3, probe syntax; no full build or GPU run)")


if __name__ == "__main__":
    main()
