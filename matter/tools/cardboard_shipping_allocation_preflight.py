#!/usr/bin/env python3
"""Compile and run the CPU-only shipping allocation preflight with input hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def included_sources(compiler: str, root: Path, source: Path) -> list[Path]:
    command = [
        compiler, "-std=c++20", "-O2", "-MM", "-MT", "shipping_preflight",
        "-I", str(root / "include"),
        "-I", str(root / "matter/include"),
        "-I", str(root / "matter/tools"),
        str(source),
    ]
    result = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    tokens = result.stdout.replace("\\\n", " ").split()[1:]
    paths = []
    for token in tokens:
        path = (root / token).resolve()
        if path.is_file() and path not in paths:
            paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(root).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "clang++"))
    parser.add_argument("--output", type=Path,
                        help="write source-bound JSON here; default writes to stdout")
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    source = Path(__file__).resolve().with_suffix(".cpp")
    compiler = arguments.compiler
    include_files = included_sources(compiler, root, source)
    if source not in include_files:
        include_files.append(source)
    script = Path(__file__).resolve()
    if script not in include_files:
        include_files.append(script)
    include_files.sort(key=lambda item: item.relative_to(root).as_posix())

    compile_command = [
        compiler, "-std=c++20", "-O2",
        "-I", str(root / "include"),
        "-I", str(root / "matter/include"),
        "-I", str(root / "matter/tools"),
        str(source), "-o",
    ]
    with tempfile.TemporaryDirectory(prefix="numi-cardboard-preflight-") as directory:
        executable = Path(directory) / "cardboard_shipping_allocation_preflight"
        subprocess.run(compile_command + [str(executable)], cwd=root, check=True)
        raw = subprocess.run([str(executable)], cwd=root, check=True,
                             capture_output=True, text=True).stdout
        result = json.loads(raw)
        compiler_version = subprocess.run(
            [compiler, "--version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0]
        result["provenance"] = {
            "git_head_at_run": git_revision(root),
            "compiled_executable_sha256": sha256(executable),
            "compiler": compiler_version,
            "compile_command": compile_command + ["<temporary executable path>"],
            "hashed_non_system_translation_dependencies": {
                str(path.relative_to(root)): sha256(path) for path in include_files
            },
            "artifact_binding_scope": (
                "C++ preflight translation unit and every non-system include dependency; "
                "does not bind runtime shader binaries because none are built or used"
            ),
        }

    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
