#!/usr/bin/env python3
"""Bounded clang preprocessing for the pinned Human resting-skin Metal source.

The output is preprocessed source for a later Metal runtime compiler. This
helper does not compile Metal or qualify any runtime behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

DEFAULT_WORKTREE = Path(__file__).resolve().parents[1]
CLANG = Path("/usr/bin/clang")
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_DEPENDENCIES = 4096
TIMEOUT_SECONDS = 60
STDERR_LIMIT_BYTES = 1024 * 1024
STDLIB_PREFIX = b"#include <metal_stdlib>\n"


class PrepError(RuntimeError):
    pass


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def parse_dependencies(path: Path) -> list[Path]:
    text = path.read_text(encoding="utf-8").replace("\\\r\n", " ").replace("\\\n", " ")
    escaped = False
    colon = None
    for i, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ":":
            colon = i
            break
    if colon is None:
        raise PrepError("clang dependency file has no target separator")

    tokens, token = [], []
    escaped = False
    for char in text[colon + 1 :]:
        if escaped:
            token.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char.isspace():
            if token:
                tokens.append("".join(token))
                token.clear()
        else:
            token.append(char)
    if escaped:
        token.append("\\")
    if token:
        tokens.append("".join(token))

    paths = set()
    for value in tokens:
        try:
            candidate = Path(value).resolve(strict=True)
        except OSError as exc:
            raise PrepError(f"dependency unavailable: {value}: {exc}") from exc
        if not candidate.is_file():
            raise PrepError(f"dependency is not a regular file: {candidate}")
        paths.add(candidate)
    if not paths:
        raise PrepError("clang reported no dependencies")
    if len(paths) > MAX_DEPENDENCIES:
        raise PrepError(f"dependency count exceeds {MAX_DEPENDENCIES}")
    return sorted(paths, key=str)


def hash_dependencies(paths: list[Path], stub: Path) -> list[dict[str, object]]:
    stub_real = stub.resolve()
    records = []
    for path in paths:
        digest, size = sha256_file(path)
        records.append(
            {
                "path": "<temporary>/metal_stdlib" if path == stub_real else str(path),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return records


def set_file_size_limit() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))


def preprocess(
    clang: Path,
    source: Path,
    roots: tuple[Path, ...],
    stub: Path,
    depfile: Path,
    output: Path,
    diagnostics: Path,
) -> tuple[list[Path], list[str], float]:
    argv = [str(clang), "-E", "-P", "-x", "c++", "-D__METAL_VERSION__=400", "--no-default-config"]
    for root in (stub.parent, *roots):
        argv.extend(["-I", str(root)])
    argv.extend(["-include", str(stub), "-MD", "-MF", str(depfile), "-MT", "flattened-skin-source", str(source)])

    env = {
        **{
            key: os.environ[key]
            for key in ("SDKROOT", "DEVELOPER_DIR", "MACOSX_DEPLOYMENT_TARGET")
            if os.environ.get(key)
        },
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C",
        "LC_ALL": "C",
    }
    start = time.monotonic()
    try:
        with output.open("wb") as out, diagnostics.open("wb") as err:
            result = subprocess.run(
                argv,
                stdout=out,
                stderr=err,
                timeout=TIMEOUT_SECONDS,
                check=False,
                preexec_fn=set_file_size_limit,
                env=env,
            )
    except subprocess.TimeoutExpired as exc:
        raise PrepError(f"clang exceeded {TIMEOUT_SECONDS}s") from exc
    elapsed = time.monotonic() - start

    if output.stat().st_size > MAX_OUTPUT_BYTES:
        raise PrepError("preprocessed output exceeded byte cap")
    if diagnostics.stat().st_size > STDERR_LIMIT_BYTES:
        raise PrepError("clang diagnostics exceeded byte cap")
    message = diagnostics.read_text(encoding="utf-8", errors="replace")
    if result.returncode:
        raise PrepError(f"clang exited {result.returncode}: {message[-8000:]}")
    return parse_dependencies(depfile), argv, elapsed


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PrepError(f"refusing to overwrite existing path: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise PrepError(f"refusing to overwrite existing path: {path}")
        os.link(temporary, path)  # Exclusive publication; never overwrite a raced destination.
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, default=DEFAULT_WORKTREE)
    parser.add_argument("--brain-source", required=True, type=Path, help="NumiBrain Sources directory used by this build")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    worktree = args.worktree.resolve(strict=True)
    source = (worktree / "matter/src/human_respiration.metal").resolve(strict=True)
    brain_source = args.brain_source.resolve(strict=True)
    roots = (
        worktree / "matter/src",
        worktree / "matter/include",
        worktree / "include",
        worktree / "src/metal",
        brain_source / "NumiBrainABI/include",
        brain_source / "NumiBrainMetal/Shaders",
    )
    roots = tuple(root.resolve(strict=True) for root in roots)
    if any(not root.is_dir() for root in roots):
        raise PrepError("one or more pinned include roots are not directories")

    clang = CLANG.resolve(strict=True)
    output = args.output.absolute()
    receipt_path = (
        args.receipt.absolute()
        if args.receipt
        else output.with_suffix(output.suffix + ".receipt.json")
    )
    if output.resolve(strict=False) == receipt_path.resolve(strict=False):
        raise PrepError("output and receipt paths must differ")
    if output.exists() or receipt_path.exists():
        raise PrepError("output or receipt exists; choose a new path")

    version = subprocess.run(
        [str(clang), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
        check=False,
        text=True,
    )
    if version.returncode:
        raise PrepError(f"unable to query clang: {version.stdout[-4000:]}")

    helper = Path(__file__).resolve(strict=True)
    helper_hash, helper_size = sha256_file(helper)
    clang_hash, clang_size = sha256_file(clang)
    source_hash_before, source_size_before = sha256_file(source)
    compile_environment = {
        key: os.environ.get(key)
        for key in ("SDKROOT", "DEVELOPER_DIR", "MACOSX_DEPLOYMENT_TARGET")
    }

    with tempfile.TemporaryDirectory(prefix="numi-resting-skin-preprocess-") as temp:
        temp = Path(temp)
        stub = temp / "metal_stdlib"
        stub.write_bytes(b"")
        results = []
        for label in ("a", "b"):
            depfile = temp / f"{label}.d"
            preprocessed = temp / f"{label}.out"
            diagnostics = temp / f"{label}.stderr"
            deps, argv, seconds = preprocess(
                clang, source, roots, stub, depfile, preprocessed, diagnostics
            )
            records = hash_dependencies(deps, stub)
            results.append((deps, records, preprocessed.read_bytes(), argv, seconds))

        deps_a, records_a, bytes_a, argv_a, seconds_a = results[0]
        deps_b, records_b, bytes_b, argv_b, seconds_b = results[1]
        if [str(p) for p in deps_a] != [str(p) for p in deps_b]:
            raise PrepError("dependency set changed between passes")
        if records_a != records_b:
            raise PrepError("an input changed between passes")
        if bytes_a != bytes_b:
            raise PrepError("preprocessed bytes changed between passes")
        normalized_a = [arg.replace(str(temp), "<temporary>") for arg in argv_a]
        normalized_b = [arg.replace(str(temp), "<temporary>") for arg in argv_b]
        normalized_a = [arg.replace("/a.d", "/<depfile>").replace("/b.d", "/<depfile>") for arg in normalized_a]
        normalized_b = [arg.replace("/a.d", "/<depfile>").replace("/b.d", "/<depfile>") for arg in normalized_b]
        if normalized_a != normalized_b:
            raise PrepError("preprocessor command changed between passes")
        source_hash_after, source_size_after = sha256_file(source)
        if (source_hash_before, source_size_before) != (source_hash_after, source_size_after):
            raise PrepError("shader source changed during preprocessing")

    output_bytes = STDLIB_PREFIX + bytes_b
    if len(output_bytes) > MAX_OUTPUT_BYTES:
        raise PrepError("prefixed output exceeded byte cap")
    output_hash = hashlib.sha256(output_bytes).hexdigest()
    protected = {source, helper, clang, *deps_b}
    if output.resolve(strict=False) in protected or receipt_path.resolve(strict=False) in protected:
        raise PrepError("output or receipt resolves to a protected input")

    receipt = {
        "schema": "numi.human.resting-skin-preprocessed-source.v1",
        "status": "preprocessed_only_not_metal_compiled",
        "scope": "full human_respiration.metal source for later runtime compilation; caller may replace only skinPipeline",
        "worktree": str(worktree),
        "source": {
            "path": str(source),
            "sha256": source_hash_after,
            "size_bytes": source_size_after,
        },
        "output": {
            "path": str(output),
            "sha256": output_hash,
            "size_bytes": len(output_bytes),
        },
        "compiler": {
            "path": str(clang),
            "sha256": clang_hash,
            "size_bytes": clang_size,
            "version": version.stdout.strip(),
        },
        "preprocessor": {
            "argv_template": normalized_b,
            "mode": "clang -E -P -x c++ -D__METAL_VERSION__=400",
            "include_roots": [str(root) for root in roots],
            "passes": 2,
            "pass_seconds": [seconds_a, seconds_b],
            "max_seconds_per_pass": TIMEOUT_SECONDS,
            "max_output_bytes": MAX_OUTPUT_BYTES,
            "dependency_count": len(records_b),
            "dependencies": records_b,
            "empty_metal_stdlib_stub_sha256": hashlib.sha256(b"").hexdigest(),
            "compile_environment": compile_environment,
        },
        "generator": {
            "path": str(helper),
            "sha256": helper_hash,
            "size_bytes": helper_size,
        },
        "checks": {
            "dependency_sets_and_hashes_identical": True,
            "preprocessed_bytes_identical": True,
            "source_hash_stable": True,
        },
        "limits": [
            "preprocessing is not Metal compilation",
            "this receipt does not show runtime use or anatomy qualification",
        ],
    }
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    atomic_write(output, output_bytes)
    try:
        atomic_write(receipt_path, receipt_bytes)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": receipt["output"],
                "receipt": {
                    "path": str(receipt_path),
                    "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                },
                "dependency_count": len(records_b),
                "pass_seconds": [seconds_a, seconds_b],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrepError as exc:
        print(f"prepare_human_resting_skin_source: {exc}", file=sys.stderr)
        raise SystemExit(2)
